#!/usr/bin/env python3
"""
train.py

Stage 4: train the MoE graph transformer and record everything the
interpretability analysis needs.

WHAT ONE RUN PRODUCES
    runs/<tag>/
        config.json      every hyperparameter, resolved (not the CLI string)
        metrics.json     per-epoch train/val curves + final test metrics
        best.pt          checkpoint at best val macro-AUC
        routing_init.npz routing BEFORE any training  <-- see below
        routing_test.npz routing after training, on the test split
        preds_test.npz   per-event probabilities, labels, and EVT_MVA1

    A run is one command with CLI arguments, so the 10-seed x 4-expert-count
    scan is a job array over the same script with no code changes.

WHY routing_init.npz EXISTS -- the methodological point
    Some role/expert correlation is present at INITIALISATION, before any
    learning. PV and EVT nodes have very different input magnitudes and their
    own type embeddings, so even a random gate separates them cleanly. Measured
    on an untrained model, PV and EVT reliably land on their own experts while
    tau / charm / bottom split near-identically across the same two experts.

    So the null hypothesis for the routing analysis is NOT uniform routing, it
    is routing at initialisation. Comparing trained routing against uniform
    would credit the model with structure it was handed for free. The claim has
    to be about tau separating from charm and bottom -- the roles that start out
    indistinguishable to the router -- and it has to be measured as a CHANGE
    from this snapshot. Hence the snapshot is taken automatically at step 0 of
    every run, rather than reconstructed afterwards.

LOSS
    L = weighted cross-entropy(logits, y) + balance_weight * L_aux

    Class weights are inverse-frequency from the TRAIN split counts recorded in
    stats.json, normalised to mean 1. With the built mixture the background
    class is ~47% of events, so unweighted training would lean on it.

    balance_weight is a first-class ablation axis, not a tuning constant: it
    trades off against specialisation. At 0 the router collapses onto one or two
    experts; too high and it is pushed to use all experts uniformly on every
    node, which actively destroys the role structure the study is looking for.

WHAT IS DELIBERATELY NOT TRAINED ON
    role / context / ctx_cat never enter the loss. They arrive from the dataset
    in separate keys and are only ever written to the routing files. The model
    is trained purely on the 3-class event label, so any role structure in the
    routing is something it found on its own.

    The exclusive tau channels (Bd2DTauNu, Lb2LcTauNu, ...) are not in the
    training mixture at all -- they are held out as probes, so context
    invariance is a generalisation claim about contexts never seen in training.

USAGE
    # single run
    python3 train.py --stats /eos/.../stats.json --out runs --tag n8_s0 \\
        --n_experts 8 --k 2 --seed 0 --epochs 30

    # fast smoke test on a small subset
    python3 train.py --stats ... --tag smoke --epochs 2 --limit_train 20000 \\
        --limit_eval 5000

    # ablations
    --balance_weight 0.0        router collapse control
    --use_moe 0                 plain FFN, no experts
    --no_shuffle_nodes          ordering-leak control (expect degenerate routing)
    --n_experts 6|8|10|12       the expert-count scan
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import GraphDataset, make_loader
from model import MoEGraphTransformer, N_CLASSES

CLASS_NAMES = {0: "Bc", 1: "Bu", 2: "bkg"}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def roc_auc(scores, labels):
    """One-vs-rest ROC AUC via the Mann-Whitney U statistic.

    Implemented directly rather than pulled from sklearn so the script has no
    dependency beyond numpy/torch, and so tie handling is explicit: tied scores
    receive their average rank, which is the standard correction and matters
    here because a saturated softmax produces many exact ties.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    # average ranks for ties
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return (r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def classification_metrics(probs, y):
    """probs (N, C), y (N,) -> dict of accuracy, per-class AUC/F1, macro."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pred = probs.argmax(1)
    out = {"accuracy": float((pred == y).mean())}
    aucs, f1s = [], []
    for c in range(probs.shape[1]):
        pos = (y == c)
        a = roc_auc(probs[:, c], pos)
        tp = int(((pred == c) & pos).sum())
        fp = int(((pred == c) & ~pos).sum())
        fn = int(((pred != c) & pos).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        nm = CLASS_NAMES.get(c, str(c))
        out[f"auc_{nm}"] = float(a)
        out[f"f1_{nm}"] = float(f1)
        out[f"precision_{nm}"] = float(prec)
        out[f"recall_{nm}"] = float(rec)
        aucs.append(a)
        f1s.append(f1)
    finite = [a for a in aucs if not np.isnan(a)]
    # nan only arises if a class is absent from the evaluation set, which would
    # mean the split or a subset is degenerate. Report it rather than letting a
    # nan propagate into early stopping, where every comparison silently fails.
    out["macro_auc"] = float(np.mean(finite)) if finite else float("nan")
    out["n_classes_present"] = len(set(y.tolist()))
    out["macro_f1"] = float(np.mean(f1s))
    return out


# --------------------------------------------------------------------------
# routing capture
# --------------------------------------------------------------------------
def _globals_to(batch, device):
    """Event-level scalars for the head, or None in the default evt_mode.

    collate() always produces an "evt_scalars" entry; it is width 0 when the
    EVT node is travelling through the graph as it always used to, and the
    model wants None rather than an empty tensor in that case.
    """
    gs = batch.get("evt_scalars")
    if gs is None or gs.shape[-1] == 0:
        return None
    return gs.to(device, non_blocking=True)


@torch.no_grad()
def capture(model, loader, device, max_events=None, want_routing=True):
    """Run the model over a loader, returning per-event predictions and, if
    asked, per-REAL-NODE routing joined to the truth labels.

    Only real nodes are stored: padded slots carry no role and would be ~60% of
    a flattened array. Everything is returned flat with an event_id column so
    routing_analysis.py can group by event, role, context or channel freely.
    """
    model.eval()
    if want_routing:
        model.record_routing(True)

    ev_probs, ev_y, ev_mva1, ev_chan = [], [], [], []
    nd_role, nd_ctx, nd_cat, nd_type, nd_event, nd_chan, nd_y = [], [], [], [], [], [], []
    nd_cand = []
    nd_idx, nd_gate = [], []
    seen = 0

    for batch in loader:
        nf = batch["node_feats"].to(device, non_blocking=True)
        nm = batch["node_mask"].to(device, non_blocking=True)
        nt = batch["node_type"].to(device, non_blocking=True)
        ef = batch["edge_feats"].to(device, non_blocking=True)
        gs = _globals_to(batch, device)
        logits, _ = model(nf, nm, nt, ef, evt_scalars=gs)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        B = probs.shape[0]
        ev_probs.append(probs)
        ev_y.append(batch["y"].numpy())
        ev_chan.append(batch["channel_id"].numpy())
        if "evt_mva1" in batch:
            ev_mva1.append(batch["evt_mva1"].numpy())

        if want_routing:
            mask_np = batch["node_mask"].numpy()
            flat = mask_np.reshape(-1)
            eid = (np.repeat(np.arange(B) + seen, mask_np.shape[1]))[flat]
            nd_event.append(eid)
            nd_role.append(batch["role"].numpy().reshape(-1)[flat])
            nd_ctx.append(batch["context"].numpy().reshape(-1)[flat])
            nd_cat.append(batch["ctx_cat"].numpy().reshape(-1)[flat])
            nd_type.append(batch["node_type"].numpy().reshape(-1)[flat])
            nd_cand.append(batch["is_cand"].numpy().reshape(-1)[flat])
            nd_chan.append(np.repeat(batch["channel_id"].numpy(),
                                     mask_np.shape[1])[flat])
            nd_y.append(np.repeat(batch["y"].numpy(), mask_np.shape[1])[flat])
            rt = model.get_routing()          # list over layers
            # (n_real_nodes, n_layers, k)
            idx = np.stack([r["top_idx"].reshape(-1, r["top_idx"].shape[-1])
                            .cpu().numpy()[flat] for r in rt], axis=1)
            gat = np.stack([r["top_gate"].reshape(-1, r["top_gate"].shape[-1])
                            .float().cpu().numpy()[flat] for r in rt], axis=1)
            nd_idx.append(idx)
            nd_gate.append(gat)

        seen += B
        if max_events is not None and seen >= max_events:
            break

    model.record_routing(False)

    preds = {
        "probs": np.concatenate(ev_probs),
        "y": np.concatenate(ev_y),
        "channel_id": np.concatenate(ev_chan),
    }
    if ev_mva1:
        preds["evt_mva1"] = np.concatenate(ev_mva1)

    routing = None
    if want_routing and nd_role:
        routing = {
            "event_id": np.concatenate(nd_event).astype(np.int32),
            "role": np.concatenate(nd_role).astype(np.int8),
            "context": np.concatenate(nd_ctx).astype(np.int32),
            "ctx_cat": np.concatenate(nd_cat).astype(np.int8),
            "node_type": np.concatenate(nd_type).astype(np.int8),
            # candidate flag from BEFORE any masking collapsed the node type
            "is_cand": np.concatenate(nd_cand).astype(np.int8),
            "channel_id": np.concatenate(nd_chan).astype(np.int8),
            "y": np.concatenate(nd_y).astype(np.int8),
            "expert_idx": np.concatenate(nd_idx).astype(np.int8),
            "expert_gate": np.concatenate(nd_gate).astype(np.float32),
        }
    return preds, routing


def routing_summary(routing, n_experts, role_names=None):
    """Compact P(top-1 expert | role) table, for the training log."""
    if routing is None:
        return ""
    idx = routing["expert_idx"][:, 0, 0]      # layer 0, top-1
    role = routing["role"]
    lines = []
    for r in sorted(set(role.tolist())):
        sel = role == r
        if sel.sum() < 10:
            continue
        counts = np.bincount(idx[sel], minlength=n_experts)
        frac = counts / counts.sum()
        nm = (role_names or {}).get(r, str(r))
        bar = " ".join(f"{f:.2f}" for f in frac)
        lines.append(f"      role {str(r):>3s} ({nm:<11s}) n={int(sel.sum()):>7d}  {bar}")
    return "\n".join(lines)


def maybe_subset(ds, limit, seed):
    """A random subset of `limit` events, or the dataset unchanged.

    Must be a random sample, not a prefix. GraphDataset builds its index by
    iterating channels in order, so val/test events arrive grouped by channel --
    and therefore by class. Taking the first N events yields a single class,
    which makes every one-vs-rest AUC undefined (n_pos or n_neg = 0 -> nan) and
    silently poisons early stopping, since every nan comparison is False and no
    checkpoint is ever saved. Sampling without replacement across the whole
    split keeps all three classes present in their natural proportions.
    """
    if limit is None or limit >= len(ds):
        return ds
    idx = np.random.default_rng(seed).choice(len(ds), size=limit, replace=False)
    return torch.utils.data.Subset(ds, idx.tolist())


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def evaluate(model, loader, device, class_w, balance_weight, max_events=None):
    model.eval()
    tot_loss = tot_n = 0
    P, Y = [], []
    with torch.no_grad():
        seen = 0
        for batch in loader:
            nf = batch["node_feats"].to(device, non_blocking=True)
            nm = batch["node_mask"].to(device, non_blocking=True)
            nt = batch["node_type"].to(device, non_blocking=True)
            ef = batch["edge_feats"].to(device, non_blocking=True)
            gs = _globals_to(batch, device)
            y = batch["y"].to(device, non_blocking=True)
            logits, aux = model(nf, nm, nt, ef, evt_scalars=gs)
            loss = F.cross_entropy(logits.float(), y, weight=class_w) \
                + balance_weight * aux
            b = y.shape[0]
            tot_loss += float(loss) * b
            tot_n += b
            P.append(torch.softmax(logits.float(), -1).cpu().numpy())
            Y.append(batch["y"].numpy())
            seen += b
            if max_events is not None and seen >= max_events:
                break
    m = classification_metrics(np.concatenate(P), np.concatenate(Y))
    m["loss"] = tot_loss / max(tot_n, 1)
    return m


def main():
    ap = argparse.ArgumentParser(description="Train the MoE graph transformer.")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default=None, help="run name; default is built "
                                                "from the config")
    # architecture
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--n_experts", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_moe", type=int, default=1)
    ap.add_argument("--evt_mode", default="node",
                    choices=["node", "pool", "mlp"],
                    help="where the 21 event-level scalars reach the model. "
                         "node (default, unchanged): the EVT node travels "
                         "through attention and is one term in the pooled "
                         "mean, so its share of the readout is "
                         "1/(n_vertices+1). pool: still in attention, but "
                         "taken out of the mean and given to the head "
                         "separately, so its readout weight no longer depends "
                         "on multiplicity. mlp: removed from the graph "
                         "entirely and seen only by the head, which tests "
                         "whether it belongs in attention at all.")
    # optimisation
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="1e-3 diverged on this task at epoch ~7; "
                         "3e-4 is the stable default")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--balance_weight", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6,
                    help="early stop after this many epochs without val "
                         "macro-AUC improvement")
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--divergence_factor", type=float, default=2.5,
                    help="if val loss exceeds this multiple of the best so far, "
                         "treat it as divergence: reload the best checkpoint and "
                         "cut the LR. 0 disables.")
    ap.add_argument("--max_recoveries", type=int, default=2,
                    help="give up after this many divergence recoveries")
    ap.add_argument("--recovery_lr_factor", type=float, default=0.3)
    # data
    ap.add_argument("--mixture", default="as_built")
    ap.add_argument("--channels", nargs="+", default=None,
                    help="restrict to these production channels (default: all "
                         "in stats.json). Class weights are recomputed from the "
                         "selected subset, so they stay correct.")
    ap.add_argument("--mask_features", nargs="*", default=None,
                    help="feature names and/or block names to zero out. "
                         "'cand3pi is_3pi_candidate' is THE critical ablation: "
                         "it removes the model's ability to see which vertices "
                         "the reconstruction flagged as 3pi candidates, and "
                         "additionally collapses the SV3pi node type into SV so "
                         "the type embedding cannot leak the same information. "
                         "99.2%% of tau vertices are candidates, so without this "
                         "mask a router can reach clean tau routing by detecting "
                         "the flag rather than tau kinematics.")
    ap.add_argument("--no_shuffle_nodes", action="store_true",
                    help="ORDERING-LEAK CONTROL: disables node permutation, so "
                         "isPV sits in slot 0 every event. Expect routing to "
                         "become slot-based; this is an ablation, not a mode "
                         "to train the real model in.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="dataset arrays are preloaded contiguously and shared "
                         "with workers by fork, so workers do not duplicate them")
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_eval", type=int, default=None)
    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0,
                    help="fp16 mixed precision. Off by default: the run is "
                         "data-bound so it buys no speed, and a T4 has no "
                         "bf16, leaving only the unstable fp16 path")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    use_amp = bool(args.amp) and device.type == "cuda"

    # evt_mode joins the auto tag so an --evt_mode pool/mlp run with no --tag
    # cannot silently overwrite the default-mode run of the same config. The
    # default mode contributes nothing, so tags of existing runs are unchanged.
    tag = args.tag or (f"n{args.n_experts}_k{args.k}_bw{args.balance_weight}"
                       f"_s{args.seed}" + ("" if args.use_moe else "_nomoe")
                       + ("_noshuf" if args.no_shuffle_nodes else "")
                       + ("_masked" if args.mask_features else "")
                       + ("" if args.evt_mode == "node"
                          else f"_evt{args.evt_mode}"))
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)

    print(f"=== run {tag} ===", flush=True)
    print(f"  device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
          + f"   amp: {use_amp}", flush=True)

    # ---- data ----------------------------------------------------------
    mf = args.mask_features or None
    if args.channels:
        # validate BEFORE constructing the dataset, which otherwise dies with
        # an unhelpful "no shards found for split 'train'"
        known = sorted(json.load(open(args.stats))["split"])
        missing = [c for c in args.channels if c not in known]
        if missing:
            raise SystemExit(f"--channels: unknown channel(s): "
                             f"{', '.join(missing)}\n  available: "
                             f"{', '.join(known)}")
    tr = GraphDataset(args.stats, split="train",
                      shuffle_nodes=not args.no_shuffle_nodes, seed=args.seed,
                      mask_features=mf, evt_mode=args.evt_mode,
                      channels=args.channels)
    va = GraphDataset(args.stats, split="val", shuffle_nodes=False,
                      mask_features=mf, evt_mode=args.evt_mode,
                      channels=args.channels)
    te = GraphDataset(args.stats, split="test", shuffle_nodes=False,
                      mask_features=mf, evt_mode=args.evt_mode,
                      channels=args.channels)
    if args.channels:
        print(f"  CHANNEL SUBSET: {', '.join(tr.channel_names)}", flush=True)
    if mf:
        print(f"  MASKED: {len(tr.masked_cols)} feature columns "
              f"({', '.join(mf)});  SV3pi collapsed: {tr.collapsed_sv3pi}",
              flush=True)
    nfeat = len(tr.feature_names)
    role_names = {int(k): v for k, v in tr.stats.get("role_names", {}).items()} \
        if isinstance(tr.stats.get("role_names"), dict) else {}

    # Eval limiting is done by RANDOM SUBSET, not by stopping early: see
    # maybe_subset(). Train limiting can stay a step cap because the train
    # loader is shuffled, so a prefix of batches is already a random draw.
    va_eval = maybe_subset(va, args.limit_eval, args.seed + 1)
    te_eval = maybe_subset(te, args.limit_eval, args.seed + 2)

    tr_loader = make_loader(tr, batch_size=args.batch_size, shuffle=True,
                            mixture=args.mixture, num_workers=args.num_workers,
                            drop_last=True)
    va_loader = make_loader(va_eval, batch_size=args.batch_size, shuffle=False,
                            mixture="as_built", num_workers=args.num_workers)
    te_loader = make_loader(te_eval, batch_size=args.batch_size, shuffle=False,
                            mixture="as_built", num_workers=args.num_workers)
    print(f"  events: train {len(tr)}  val {len(va)}  test {len(te)}"
          f"   features {nfeat}", flush=True)
    if args.limit_eval:
        print(f"  eval limited to a random subset: val {len(va_eval)}  "
              f"test {len(te_eval)}", flush=True)

    # ---- class weights ---------------------------------------------------
    # stats.json's train_summary counts every channel in the file. With
    # --channels that is the wrong denominator, so count the loaded subset
    # directly. Without --channels the two agree, and old commands reproduce.
    if args.channels:
        counts = np.array([(tr.y == c).sum() for c in range(N_CLASSES)],
                          dtype=np.float64)
        print(f"  class counts recomputed from the subset: "
              f"{counts.astype(int).tolist()}", flush=True)
    else:
        cc = tr.stats["train_summary"]["class_counts"]
        counts = np.array([cc.get(str(c), 0) for c in range(N_CLASSES)],
                          dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (N_CLASSES * counts)
    class_w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("  class weights: " + ", ".join(
        f"{CLASS_NAMES[c]}={class_w[c]:.3f}" for c in range(N_CLASSES)), flush=True)

    # ---- model ---------------------------------------------------------
    model = MoEGraphTransformer(
        nfeat, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, n_experts=args.n_experts,
        k=args.k, dropout=args.dropout, use_moe=bool(args.use_moe),
        n_global=tr.n_global).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {npar:,}", flush=True)

    cfg = vars(args) | {"channels_used": list(tr.channel_names),
                        "evt_mode": args.evt_mode,
                        "n_global": tr.n_global,
                        "masked_columns": tr.masked_cols,
                        "masked_feature_names": [tr.feature_names[c]
                                                 for c in tr.masked_cols],
                        "collapsed_sv3pi": tr.collapsed_sv3pi,
                        "resolved_tag": tag, "device": str(device),
                        "n_features": nfeat, "n_parameters": npar,
                        "class_weights": class_w.tolist(),
                        "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # ---- routing snapshot BEFORE training ------------------------------
    # This is the analysis baseline. See the module docstring: PV/EVT separation
    # is free at init, so the trained routing has to be read against this, not
    # against a uniform prior.
    print("\n  capturing routing at initialisation (analysis baseline)...",
          flush=True)
    # cap the init snapshot for speed; the loader is already a random subset
    # when --limit_eval is set, so this cannot bias it toward one class
    _, rt_init = capture(model, va_loader, device, max_events=40000)
    np.savez_compressed(os.path.join(outdir, "routing_init.npz"), **rt_init)
    print("  P(top-1 expert | role) at init, layer 0:")
    print(routing_summary(rt_init, args.n_experts, role_names), flush=True)

    # ---- optimiser -----------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(len(tr_loader), 1)
    if args.limit_train:
        steps_per_epoch = min(steps_per_epoch,
                              max(args.limit_train // args.batch_size, 1))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(int(args.warmup_frac * total_steps), 1)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ---- loop ----------------------------------------------------------
    hist = []
    best = {"macro_auc": -1.0, "epoch": -1}
    best_val_loss = float("inf")
    bad = 0
    gstep = 0
    n_recoveries = 0
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = run_ce = run_aux = 0.0
        seen = 0
        n_skipped = 0
        for bi, batch in enumerate(tr_loader):
            if args.limit_train and seen >= args.limit_train:
                break
            nf = batch["node_feats"].to(device, non_blocking=True)
            nm = batch["node_mask"].to(device, non_blocking=True)
            nt = batch["node_type"].to(device, non_blocking=True)
            ef = batch["edge_feats"].to(device, non_blocking=True)
            gs = _globals_to(batch, device)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, aux = model(nf, nm, nt, ef, evt_scalars=gs)
                ce = F.cross_entropy(logits.float(), y, weight=class_w)
                loss = ce + args.balance_weight * aux

            # A single non-finite batch, applied, poisons every weight it
            # touches and the run never recovers. Skip it and count it: a
            # nonzero count is itself diagnostic (a rising count means the run
            # is on the edge of instability even if it has not blown up yet).
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1

            b = y.shape[0]
            run_loss += float(loss) * b
            run_ce += float(ce) * b
            run_aux += float(aux) * b
            seen += b

        tr_loss = run_loss / max(seen, 1)
        vm = evaluate(model, va_loader, device, class_w, args.balance_weight)
        dt = time.time() - t0
        hist.append({"epoch": ep, "train_loss": tr_loss,
                     "skipped_batches": n_skipped,
                     "train_ce": run_ce / max(seen, 1),
                     "train_aux": run_aux / max(seen, 1),
                     "lr": opt.param_groups[0]["lr"],
                     "val": vm, "seconds": dt, "train_events": seen})
        print(f"  ep {ep:3d}  train {tr_loss:.4f} (ce {run_ce/max(seen,1):.4f} "
              f"aux {run_aux/max(seen,1):.4f})  val loss {vm['loss']:.4f}  "
              f"macroAUC {vm['macro_auc']:.4f}  acc {vm['accuracy']:.4f}  "
              f"[{dt:.0f}s]"
              + (f"  SKIPPED {n_skipped}" if n_skipped else ""), flush=True)

        # ---- divergence guard -------------------------------------------
        # A pre-norm transformer with an MoE can train cleanly for several
        # epochs and then spike: val loss jumps while train loss barely moves,
        # and within a few epochs the model collapses to predicting one class.
        # Observed on this task at lr=1e-3 with fp16, at epoch 7 of 40, and the
        # remaining 8 epochs were wasted compute. Rather than let that happen,
        # detect it, roll back to the best checkpoint, and cut the LR.
        diverged = (args.divergence_factor > 0
                    and np.isfinite(best_val_loss)
                    and vm["loss"] > args.divergence_factor * best_val_loss)
        if diverged and os.path.exists(os.path.join(outdir, "best.pt")):
            n_recoveries += 1
            if n_recoveries > args.max_recoveries:
                print(f"  DIVERGED again after {args.max_recoveries} recoveries "
                      f"(val loss {vm['loss']:.4f} vs best {best_val_loss:.4f}); "
                      f"stopping.", flush=True)
                break
            ckpt = torch.load(os.path.join(outdir, "best.pt"),
                              map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            for pg in opt.param_groups:
                pg["initial_lr"] = pg.get("initial_lr", args.lr) * args.recovery_lr_factor
            # LambdaLR multiplies base_lrs, so scale those directly
            sched.base_lrs = [b * args.recovery_lr_factor for b in sched.base_lrs]
            print(f"  DIVERGED: val loss {vm['loss']:.4f} > "
                  f"{args.divergence_factor}x best {best_val_loss:.4f}. "
                  f"Restored epoch {ckpt['epoch']}, LR x{args.recovery_lr_factor} "
                  f"(recovery {n_recoveries}/{args.max_recoveries})", flush=True)
            hist[-1]["diverged"] = True
            bad = 0
            continue

        if np.isfinite(vm["loss"]):
            best_val_loss = min(best_val_loss, vm["loss"])

        improved = (np.isfinite(vm["macro_auc"])
                    and vm["macro_auc"] > best["macro_auc"] + 1e-5)
        # Always write a checkpoint on the first epoch. Otherwise a run whose
        # metric never improves -- or is non-finite because a class is missing
        # from the eval set -- finishes with no best.pt and dies at the restore
        # step, losing the whole run.
        if improved or best["epoch"] < 0:
            best = {"macro_auc": (vm["macro_auc"] if np.isfinite(vm["macro_auc"])
                                  else -1.0),
                    "epoch": ep, "val": vm}
            torch.save({"model": model.state_dict(), "config": cfg,
                        "epoch": ep, "val": vm},
                       os.path.join(outdir, "best.pt"))
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop: no val improvement for {bad} epochs",
                      flush=True)
                break

        if not np.isfinite(vm["macro_auc"]):
            print(f"  WARNING: val macro-AUC is not finite "
                  f"({vm.get('n_classes_present')} of {N_CLASSES} classes present "
                  f"in the eval set). Early stopping is disabled until this "
                  f"resolves.", flush=True)

    # ---- restore best and evaluate on test -----------------------------
    ck = torch.load(os.path.join(outdir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"\n  restored best epoch {ck['epoch']} "
          f"(val macroAUC {ck['val']['macro_auc']:.4f})", flush=True)

    tm = evaluate(model, te_loader, device, class_w, args.balance_weight)
    print("  test: " + ", ".join(
        f"{k}={v:.4f}" for k, v in tm.items()
        if k in ("accuracy", "macro_auc", "macro_f1", "auc_Bc", "auc_Bu", "auc_bkg")),
        flush=True)

    preds, rt_test = capture(model, te_loader, device)
    np.savez_compressed(os.path.join(outdir, "routing_test.npz"), **rt_test)
    np.savez_compressed(os.path.join(outdir, "preds_test.npz"), **preds)

    print("\n  P(top-1 expert | role) after training, layer 0:")
    print(routing_summary(rt_test, args.n_experts, role_names), flush=True)

    # BDT1 reference point, if EVT_MVA1 came through
    mva_auc = None
    if "evt_mva1" in preds:
        sig = (preds["y"] != 2)
        mva_auc = roc_auc(preds["evt_mva1"], sig)
        print(f"\n  EVT_MVA1 (BDT1) signal-vs-background AUC on the same "
              f"events: {mva_auc:.4f}", flush=True)

    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump({"config": cfg, "history": hist, "best": best, "test": tm,
                   "evt_mva1_auc": mva_auc,
                   "total_seconds": time.time() - t_start}, fh, indent=2)

    print(f"\n  wrote {outdir}/  ({time.time() - t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
