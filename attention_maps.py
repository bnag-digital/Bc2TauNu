#!/usr/bin/env python3
"""Role-indexed attention maps -- the analogue of Fig. 5-7 of Genovese et al.

WHY THIS IS NOT A COPY OF THEIR FIGURE
--------------------------------------
Their graph has a FIXED node identity in every event: slot 0 is always jet1,
slot 4 is always the lepton, slot 5 is always the energy node. So an N x N
matrix indexed by slot is meaningful, and averaging it over events is
meaningful too.

Ours is not like that. The number of vertices varies event to event (4 to 13),
and which slot holds the tau vertex is an accident of reconstruction order. An
attention matrix indexed by slot would average the tau vertex of one event
against the charm vertex of the next, and the result would be mush.

So we index by TRUTH ROLE instead: entry (i, j) is the mean attention paid by
vertices of role i to vertices of role j, over every ordered pair of real nodes
in every selected event. That is the same question their figure asks ("which
kind of object attends to which"), asked in the only way our graph permits.
It is arguably a cleaner version, because the axes are physics labels rather
than reconstruction slots.

Note this makes the matrix NON-SYMMETRIC and that is the point: row = query
(the node doing the attending), column = key (the node attended to), exactly
as in their Section 4.1.

Their Fig. 5/6/7 are one plot over three event subsets; --subset reproduces
that split.

  python3 attention_maps.py --run RUNDIR                  # all test events
  python3 attention_maps.py --run RUNDIR --subset tn       # their Fig 7
  python3 attention_maps.py --run RUNDIR --subset tp       # their Fig 6
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import GraphDataset, collate, make_loader
from model import MoEGraphTransformer, N_CLASSES

ROLE_NAMES = {-2: "EVT", 0: "other", 1: "tau", 2: "charm", 3: "bottom",
              4: "strange", 5: "PV"}
# display order: PV first, then the decay products, EVT last -- reads like a
# decay chain rather than like an enum
ROLE_ORDER = [5, 1, 2, 3, 4, 0, -2]


def build_model(cfg, nfeat, n_global, device):
    m = MoEGraphTransformer(
        nfeat, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg.get("d_ff"),
        n_experts=cfg["n_experts"], k=cfg["k"], dropout=cfg["dropout"],
        use_moe=bool(cfg.get("use_moe", 1)), n_global=n_global)
    return m.to(device)


@torch.no_grad()
def accumulate(model, loader, device, n_layers, n_heads, subset, max_events):
    """Sum attention into (layer, head, role_i, role_j) bins."""
    R = len(ROLE_ORDER)
    idx_of = {r: i for i, r in enumerate(ROLE_ORDER)}
    tot = np.zeros((n_layers, n_heads, R, R), np.float64)
    cnt = np.zeros((n_layers, n_heads, R, R), np.float64)
    seen = 0

    model.eval().record_routing(True)
    for batch in loader:
        nf = batch["node_feats"].to(device)
        nm = batch["node_mask"].to(device)
        nt = batch["node_type"].to(device)
        ef = batch["edge_feats"].to(device)
        gs = batch.get("evt_scalars")
        gs = gs.to(device) if gs is not None and gs.shape[-1] else None

        logits, _ = model(nf, nm, nt, ef, evt_scalars=gs)
        y = batch["y"].numpy()
        pred = logits.argmax(-1).cpu().numpy()

        # their Fig 6 = true positives, Fig 7 = true negatives. Signal is
        # class 0 (Bc) or 1 (B+); background is class 2.
        if subset == "tp":
            keep = (pred == y) & (y != 2)
        elif subset == "tn":
            keep = (pred == y) & (y == 2)
        elif subset == "correct":
            keep = pred == y
        else:
            keep = np.ones_like(y, bool)
        if not keep.any():
            continue

        attns = model.get_attention()          # list of (B, H, N, N)
        role = batch["role"].numpy()
        mask = batch["node_mask"].numpy()

        kb = np.where(keep)[0]
        for L, A in enumerate(attns):
            a = A.float().cpu().numpy()[kb]     # (b, H, N, N)
            r = role[kb]
            mk = mask[kb]
            for b in range(a.shape[0]):
                real = np.where(mk[b])[0]
                if real.size < 2:
                    continue
                rr = r[b, real]
                # map roles to bin indices; -1 (unmatched) is dropped, since
                # "we could not tell what this vertex was" is not a physics
                # category and would contaminate every row it entered
                bins = np.array([idx_of.get(int(x), -1) for x in rr])
                ok = bins >= 0
                if ok.sum() < 2:
                    continue
                real, bins = real[ok], bins[ok]
                sub = a[b][:, real][:, :, real]        # (H, n, n)
                for h in range(sub.shape[0]):
                    np.add.at(tot[L, h], (bins[:, None].repeat(len(bins), 1),
                                          bins[None, :].repeat(len(bins), 0)),
                              sub[h])
                    np.add.at(cnt[L, h], (bins[:, None].repeat(len(bins), 1),
                                          bins[None, :].repeat(len(bins), 0)),
                              np.ones_like(sub[h]))
        seen += int(keep.sum())
        if max_events and seen >= max_events:
            break

    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return mean, cnt, seen


MIN_CELL = 200          # below this the ratio is noise, not a measurement


def _enrichment(counts, min_cell=MIN_CELL):
    """Turn raw routing counts into enrichment = P(expert|group) / P(expert).

    WHY RAW COUNTS CANNOT SHOW SPECIALISATION IN OUR DATA
    -----------------------------------------------------
    Their Fig. 8 works on raw counts because their node types are BALANCED:
    every event has exactly one jet1, one b1, one lepton, one energy node, so
    each bar is built from the same number of nodes and the heights are
    directly comparable.

    Ours are not. Charm occupies ~20% of all node-slots and tau only ~9%, while
    the load-balancing loss holds every expert to roughly the same total. With
    8 experts and 1.4M node-slots the mean expert capacity is ~175k, so charm
    (279k slots) physically CANNOT fit inside fewer than 1.6 experts no matter
    how well it is specialised. A raw-count plot therefore shows charm spread
    over many experts and tau concentrated, which is mostly a statement about
    how common charm is, not about the routing.

    Enrichment divides that out twice over:

        enrichment(role, expert) = P(expert | role) / P(expert)

    The numerator is the row normalised to 1; the denominator is the expert's
    overall share of all routing slots. So the value asks: "does this role use
    this expert MORE than the average node does?"

      1.0 = exactly as often as chance -- no specialisation
      4.0 = four times more than chance -- strong specialisation
      0.0 = never uses it

    This is the same quantity the NMI and JSD numbers summarise, just displayed
    per (role, expert) cell instead of collapsed to one scalar. It is scale
    free, so a rare role and a common role can be compared on one axis.
    """
    counts = counts.astype(np.float64)
    row = counts.sum(axis=-1, keepdims=True)          # slots per group
    col = counts.sum(axis=-2, keepdims=True)          # slots per expert
    tot = counts.sum(axis=(-2, -1), keepdims=True)
    p_e_given_g = np.divide(counts, np.maximum(row, 1))
    p_e = np.divide(col, np.maximum(tot, 1))
    enr = np.divide(p_e_given_g, np.maximum(p_e, 1e-12))
    # A ratio is only as good as its denominator. If an expert handles almost
    # none of this group, or the cell itself holds a handful of nodes, the
    # ratio explodes on Poisson noise: with 10 tau vertices on an expert, 4 of
    # them from B0 reads as "6.9x enriched" and means nothing. Those cells are
    # also the TALLEST bars, so a reader scanning for the biggest effect lands
    # exactly on the least trustworthy number. Blank them instead.
    small = (counts < min_cell) | (np.broadcast_to(col, counts.shape) < min_cell)
    return np.where(small, np.nan, enr)


def _bar_panels(counts, labels, ne, k, n_layers, title, ylabel, out, fname,
                enrich=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n_layers, figsize=(7.6 * n_layers, 4.8),
                             squeeze=False)
    cmap = plt.get_cmap("tab10" if ne <= 10 else "tab20")
    width = 0.8 / ne
    for L in range(n_layers):
        ax = axes[0][L]
        x = np.arange(len(labels))
        for e in range(ne):
            vals = np.nan_to_num(counts[L, :, e], nan=0.0)
            ax.bar(x + e * width - 0.4 + width / 2, vals,
                   width, label=f"Expert {e}", color=cmap(e % cmap.N))
        if enrich:
            # chance level: the line every bar would sit on with no
            # specialisation at all. Without it the reader has no anchor.
            ax.axhline(1.0, color="black", ls="--", lw=1, zorder=0)
            ax.text(len(labels) - 0.45, 1.05, "chance", fontsize=7,
                    ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Encoder layer {L}")
        ax.grid(axis="y", alpha=0.3)
        if L == n_layers - 1:
            ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, fname)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def _print_table(counts, labels, ne, header, fmt="{:>9,.0f}", total=True):
    for L in range(counts.shape[0]):
        print(f"\n=== layer {L}: {header} ===")
        print("  " + "group".ljust(11)
              + "".join(f"{'e%d' % e:>9}" for e in range(ne))
              + ("    total" if total else ""))
        for gi, lab in enumerate(labels):
            row = counts[L, gi]
            line = "  " + lab.ljust(11) + "".join(
                "        -" if np.isnan(v) else fmt.format(v) for v in row)
            if total:
                line += f"{row.sum():>9,.0f}"
            print(line)
        if total:
            print("  " + "TOTAL".ljust(11)
                  + "".join(f"{v:>9,.0f}" for v in counts[L].sum(0)))
        else:
            print(f"  ('-' = fewer than {MIN_CELL} nodes in the cell or on "
                  f"that expert; ratio would be noise)")
        # the quotable line: best expert and how enriched
        if not total:
            print("  strongest expert per group:")
            for gi, lab in enumerate(labels):
                if np.all(np.isnan(counts[L, gi])):
                    print(f"    {lab:<11} -> (all cells below the "
                          f"{MIN_CELL}-node floor)")
                    continue
                e = int(np.nanargmax(counts[L, gi]))
                print(f"    {lab:<11} -> e{e}  ({counts[L, gi, e]:.1f}x chance)")


def fig8_specialisation(run, out, n_experts=None, use_node_type=False,
                        split_context=False, ctx_role=1):
    """Their Fig. 8, plus an enrichment-normalised twin and a context split.

    Three plots are produced:
      counts     -- raw "nodes processed", direct parity with their Fig. 8
      enrichment -- P(expert|group)/P(expert), which is what actually shows
                    specialisation once role prevalence is divided out
      context    -- the same pair, but rows are the PARENT HADRON of a single
                    role (default tau), which is the B+ vs Bc+ vs B0 question

    On the context split: role and context are different labels. A tau vertex
    from B+ and one from Bc+ are BOTH role=tau, so the role plot merges them.
    Splitting by context is the only way to see them separately in count form.
    """
    import routing_analysis as ra

    cfg = json.load(open(os.path.join(run, "config.json")))
    ne = n_experts or cfg["n_experts"]
    test = ra.Routing(os.path.join(run, "routing_test.npz"), n_experts=ne)
    n_layers, k = test.n_layers, test.k
    written = []

    def build(sel_of, labels, tag, title_bit):
        counts = np.zeros((n_layers, len(labels), ne), np.int64)
        for L in range(n_layers):
            for gi, sel in enumerate(sel_of):
                ids = test.idx[sel, L, :].ravel()
                counts[L, gi] = np.bincount(ids, minlength=ne)[:ne]
        enr = _enrichment(counts)

        _print_table(counts, labels, ne, f"nodes processed ({title_bit})")
        _print_table(enr, labels, ne,
                     f"ENRICHMENT = P(expert|group)/P(expert)  ({title_bit})",
                     fmt="{:>9.2f}", total=False)

        p1 = _bar_panels(
            counts, labels, ne, k, n_layers,
            f"Expert specialisation, raw counts -- {title_bit}  "
            f"({ne} experts, top-{k} routing)",
            "nodes processed", out, f"fig8_{tag}_counts.png")
        p2 = _bar_panels(
            enr, labels, ne, k, n_layers,
            f"Expert specialisation, enrichment over chance -- {title_bit}  "
            f"({ne} experts, top-{k} routing)",
            "P(expert | group) / P(expert)", out,
            f"fig8_{tag}_enrichment.png", enrich=True)
        np.savez_compressed(os.path.join(out, f"fig8_{tag}.npz"),
                            counts=counts, enrichment=enr,
                            labels=np.array(labels), n_experts=ne, k=k)
        return [p1, p2]

    # ---- by role (their Fig. 8) -------------------------------------------
    if use_node_type:
        NT = {1: "PV", 2: "SV", 3: "SV3pi", 4: "EVT"}
        keys = [key for key in (1, 2, 3, 4) if (test.node_type == key).any()]
        labels = [NT[key] for key in keys]
        sel_of = [test.node_type == key for key in keys]
        written += build(sel_of, labels, "node_type", "by node type")
    else:
        order = [r for r in ROLE_ORDER if (test.role == r).any()]
        labels = [ROLE_NAMES[r] for r in order]
        sel_of = [test.role == r for r in order]
        written += build(sel_of, labels, "by_role", "by vertex role")

    # ---- by parent hadron, one role only ----------------------------------
    if split_context:
        rname = ROLE_NAMES.get(ctx_role, str(ctx_role))
        isrole = test.role == ctx_role
        ctxs, keep = [], []
        for c in np.unique(test.context[isrole]):
            n = int((isrole & (test.context == c)).sum())
            # a parent with a handful of vertices gives a meaningless
            # enrichment ratio; 500 keeps the bars interpretable
            if n >= 500 and int(c) != 0:
                ctxs.append(int(c))
                keep.append(n)
        if not ctxs:
            print(f"\n  no parent hadron has >=500 {rname} vertices; "
                  f"skipping the context split")
        else:
            odr = np.argsort(keep)[::-1]
            ctxs = [ctxs[i] for i in odr]
            labels = [f"{ra.CTX_NAMES.get(c, str(c))}" for c in ctxs]
            sel_of = [isrole & (test.context == c) for c in ctxs]
            print(f"\n  context split on role={rname}: "
                  + ", ".join(f"{l} (n={int(s.sum()):,})"
                              for l, s in zip(labels, sel_of)))
            written += build(sel_of, labels, f"{rname}_by_parent",
                             f"{rname} vertices, by parent hadron")
    return written


def figure(mean, cnt, out, layer_names, subset, seen, run, pct=85.0,
           drop_evt=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers, n_heads = mean.shape[0], mean.shape[1]
    labels = [ROLE_NAMES[r] for r in ROLE_ORDER]

    # Their figures "visualis[e] only attention scores above the 85th
    # percentile (computed across all layers and heads) to highlight the most
    # significant interactions". That threshold is why most of their cells are
    # white -- without it every panel is a wash of mid-blue and no structure is
    # visible. One global cutoff over all layers and heads, exactly as stated.
    evt_i = ROLE_ORDER.index(-2) if -2 in ROLE_ORDER else None
    if drop_evt and evt_i is not None:
        # The EVT node absorbs so much attention (0.48-0.63 in layer 1, vs
        # <=0.24 for every vertex-vertex pair) that a global 85th-percentile
        # cutoff keeps ONLY the EVT column and whites out the entire
        # vertex-to-vertex block. That is a real finding, but it is not what
        # their figure is for -- so this variant removes EVT from both axes and
        # recomputes the percentile over what is left, exposing the structure
        # among actual vertices.
        keep = [i for i in range(len(ROLE_ORDER)) if i != evt_i]
        mean = mean[:, :, keep][:, :, :, keep]
        labels = [labels[i] for i in keep]
    allv = mean[~np.isnan(mean)]
    thresh = np.percentile(allv, pct) if allv.size else 0.0
    shown = np.where(mean >= thresh, mean, np.nan)
    print(f"\n  {pct:.0f}th-percentile cutoff over all layers/heads: "
          f"{thresh:.4f}  ({np.isfinite(shown).sum()} of "
          f"{np.isfinite(mean).sum()} cells shown)"
          + ("   [EVT excluded from axes and from the cutoff]"
             if drop_evt and evt_i is not None else ""))
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3.1 * n_heads, 3.3 * n_layers),
                             squeeze=False)
    for L in range(n_layers):
        # one colour scale per layer, as they do -- layer 1 attention is much
        # weaker than layer 0 and a shared scale would render it blank
        vals = mean[L][~np.isnan(mean[L])]
        vmax = vals.max() if vals.size else 1.0
        for h in range(n_heads):
            ax = axes[L][h]
            M = np.ma.masked_invalid(shown[L, h])
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"Head {h + 1}, {layer_names[L]}", fontsize=9)
            if h == 0:
                ax.set_ylabel("query (attends)", fontsize=8)
            ax.set_xlabel("key (attended to)", fontsize=8)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if not np.isnan(shown[L, h, i, j]):
                        v = shown[L, h, i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=5.5,
                                color="white" if v > vmax * 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
    sub = {"all": "all test events", "tp": "correctly classified signal",
           "tn": "correctly classified background",
           "correct": "all correctly classified"}[subset]
    fig.suptitle(f"Mean attention by vertex role -- {sub}  (n={seen:,} events)"
                 f"\nrow = the vertex doing the attending, "
                 f"column = the vertex attended to;  only cells above the "
                 f"{pct:.0f}th percentile ({thresh:.3f}) are shown",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    tag = f"{subset}_noEVT" if drop_evt else subset
    p = os.path.join(out, f"attention_by_role_{tag}.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--subset", default="all",
                    choices=["all", "tp", "tn", "correct"])
    ap.add_argument("--max_events", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pct", type=float, default=85.0,
                    help="percentile cutoff for display, as in the paper's "
                         "Fig 5-7 (85th over all layers and heads)")
    ap.add_argument("--no_context_split", action="store_true",
                    help="skip the by-parent-hadron version of fig 8")
    ap.add_argument("--ctx_role", type=int, default=1,
                    help="role to split by parent hadron (1=tau, 2=charm, "
                         "3=bottom, 4=strange)")
    ap.add_argument("--fig8_node_type", action="store_true",
                    help="use the coarse PV/SV/SV3pi/EVT node_type on the x "
                         "axis of fig 8 instead of the truth role")
    ap.add_argument("--fig", default="both", choices=["7", "8", "both"],
                    help="7 = attention maps (needs best.pt and a forward "
                         "pass), 8 = expert specialisation counts (reads the "
                         "existing routing npz, no model needed)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    out = args.out or os.path.join(args.run, "report")
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))

    # Fig 8 first: it only needs routing_test.npz, which already exists, so it
    # costs seconds and works even if best.pt or a GPU is unavailable.
    if args.fig in ("8", "both"):
        for p8 in fig8_specialisation(
                args.run, out, use_node_type=args.fig8_node_type,
                split_context=not args.no_context_split,
                ctx_role=args.ctx_role):
            print(f"wrote {p8}")
        if args.fig == "8":
            return

    ds = GraphDataset(cfg["stats"], split="test", shuffle_nodes=False,
                      mask_features=cfg.get("mask_features") or None,
                      evt_mode=cfg.get("evt_mode", "node"),
                      channels=cfg.get("channels_used") or cfg.get("channels"), verbose=True)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2)
    model = build_model(cfg, len(ds.feature_names), ds.n_global, dev)
    sd = torch.load(os.path.join(args.run, "best.pt"), map_location=dev,
                    weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    print(f"  loaded best.pt  ({cfg['n_parameters']:,} params, "
          f"evt_mode={cfg.get('evt_mode', 'node')})", flush=True)

    mean, cnt, seen = accumulate(model, loader, dev, cfg["n_layers"],
                                 cfg["n_heads"], args.subset, args.max_events)
    layer_names = [f"Layer {i}" for i in range(cfg["n_layers"])]

    labels = [ROLE_NAMES[r] for r in ROLE_ORDER]
    for L in range(cfg["n_layers"]):
        print(f"\n=== {layer_names[L]}: mean attention, averaged over heads ===")
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                M = np.nanmean(mean[L], axis=0)
        print("      " + "".join(f"{l:>9}" for l in labels))
        for i, li in enumerate(labels):
            row = "".join("      n/a" if np.isnan(M[i, j])
                          else f"{M[i, j]:>9.3f}" for j in range(len(labels)))
            print(f"  {li:<5}{row}")
        # the single most-attended-to role per query, which is the sentence you
        # actually say out loud about the figure
        print("  strongest target per query role:")
        for i, li in enumerate(labels):
            if np.all(np.isnan(M[i])):
                continue
            j = int(np.nanargmax(M[i]))
            print(f"    {li:<8} -> {labels[j]:<8} ({M[i, j]:.3f})")

    np.savez_compressed(os.path.join(out, f"attention_by_role_{args.subset}.npz"),
                        mean=mean, count=cnt, roles=np.array(ROLE_ORDER),
                        role_names=np.array(labels), n_events=seen)
    p = figure(mean, cnt, out, layer_names, args.subset, seen, args.run,
               pct=args.pct)
    print(f"\nwrote {p}")
    # second variant with EVT dropped, since the first is dominated by it
    p2 = figure(mean, cnt, out, layer_names, args.subset, seen, args.run,
                pct=args.pct, drop_evt=True)
    print(f"wrote {p2}")
    print(f"wrote {os.path.join(out, f'attention_by_role_{args.subset}.npz')}")


if __name__ == "__main__":
    main()