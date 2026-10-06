#!/usr/bin/env python3
"""
probe_fig8.py

Run a held-out channel (default Bs2DsTauNu) through one or more trained models
and draw the Fig. 8 expert-specialisation plots with the PROBE SAMPLE AS ITS OWN
GROUP on the x axis, sitting next to the training-sample groups rather than
being merged into them.

Two plots per model:

  1. BY ROLE      rows are tau / charm / bottom / strange / other / PV, each one
                  appearing twice -- once from the run's own test set and once
                  from the probe -- so "does a probe tau go where a training tau
                  goes?" is a comparison of two adjacent bars.

  2. TAU BY PARENT  rows are the parent hadron of each TAU vertex: B+, Bc+, B0,
                  Bs0, ... from the test set, PLUS the probe's taus as separate
                  rows labelled with the sample they came from. This is the
                  requested plot: the probe channel is a distinct x-axis entry
                  instead of being absorbed into the B+ / Bc+ bars.

WHY THE SAMPLE TAG, AND NOT SOMETHING CLEVERER
    vertex_truth.parent_context resolves two generations, so a tau from
    B+ -> tau nu and a tau from B+ -> D0 tau nu BOTH come out with
    context = 521. On a single inclusive sample the PDG code alone cannot
    separate them and you would need an indirect proxy (e.g. "is there a charm
    vertex in the same event"), which is a reconstruction of information that
    was already lost.

    Here that is unnecessary. Bs2DsTauNu is its own directory, its own shards,
    its own dataset. Which decay a tau came from is known EXACTLY from which
    file it was read out of, so the group tag is just the sample name. Exact
    beats inferred. (--split_charm still offers the proxy, for the case where
    two topologies really are mixed inside one sample; it is off by default and
    on a whole-event basis, so the second b hadron of the Z -> bb event can
    contaminate it.)

WHAT YOU CAN AND CANNOT READ ACROSS TWO MODELS
    Within ONE model, comparing groups is exactly what these plots are for: if
    "Bs0 [Bs2DsTauNu]" lights up the same experts as "B+ [test]", that model is
    treating them alike.

    ACROSS two separately trained models, expert NUMBERS mean nothing in common
    -- expert 3 of one run and expert 3 of the other are unrelated objects. So
    do not read "model A sends it to e5, model B sends it to e2" as a
    difference. What IS comparable across models is the PATTERN: whether the
    probe bar looks like its neighbours, how concentrated each row is, and
    whether the probe row is more or less spread out than the training rows.
    Each model gets its own figure for that reason.

NOTE ON PARENT ASSIGNMENT FOR THIS CHANNEL
    In Bs -> Ds tau nu the tau's parent is the Bs0 (PDG 531), not a B+. So the
    probe's taus appear under Bs0. The sharpest comparison on the plot is
    "Bs0 [Bs2DsTauNu]" against "Bs0 [test]" if the training mix has Bs0 taus --
    same parent hadron, different decay -- with B+ and Bc+ as the wider context.

BEFORE RUNNING: BUILD THE PROBE SHARDS
    Bs2DsTauNu is a held-out probe, deliberately not built by
    `graph_build.py --channel all`. Build it once:

        cd /eos/user/b/bnag/Bc2TauNu/aug3_scripts
        python3 graph_build.py --channel probe:Bs2DsTauNu \\
            --production analysis \\
            --out /eos/user/b/bnag/Bc2TauNu/shards

    which reads
      .../prod_03/Batch_Analysis_stage1/p8_ee_Zbb_ecm91_EvtGen_Bs2DsTauNu/*.root
    and writes  .../shards/Bs2DsTauNu/*.pt

USAGE
    python3 probe_fig8.py \\
        --runs /eos/user/b/bnag/Bc2TauNu/aug3_runs/<RUN_NODE> \\
               /eos/user/b/bnag/Bc2TauNu/aug3_runs/<RUN_MLP> \\
        --probe_shards /eos/user/b/bnag/Bc2TauNu/shards \\
        --channel Bs2DsTauNu

    --count_mode topk|top1|gate   what one unit of bar height means (below)
    --sources both|probe|test     which samples to put on the x axis
    --min_n N                     drop a group with fewer than N vertices
    --split_charm                 extra split by "charm vertex elsewhere in the
                                  event"; off by default, see above
    --selftest                    check the counting on synthetic arrays, no data
"""

import argparse
import json
import os

import numpy as np

# reused unmodified from the existing analysis code
from attention_maps import MIN_CELL, _bar_panels, _enrichment
from routing_analysis import CTX_NAMES

ROLE_NAMES = {-2: "EVT", -1: "unmatched", 0: "other", 1: "tau", 2: "charm",
              3: "bottom", 4: "strange", 5: "PV"}
ROLE_ORDER = [5, 1, 2, 3, 4, 0]
ROLE_TAU = 1
ROLE_CHARM = 2
NODE_EVT = 4

COUNT_MODES = ("topk", "top1", "gate")
_COUNT_DOC = {
    "topk": ("node->expert assignments (k per node)",
             "each vertex contributes k counts, one per expert it was routed "
             "to, so it appears in k different bars and a row sums to "
             "k x (vertices in the group). This is what the existing fig8 "
             "does."),
    "top1": ("vertices (one count each, top-1 expert)",
             "each vertex contributes exactly 1, to its argmax expert, so a "
             "row sums to the number of vertices in the group."),
    "gate": ("gate-weighted vertices (sums to 1 per vertex)",
             "each vertex contributes its k gate weights, which sum to 1, so a "
             "row sums to the number of vertices but a barely-chosen second "
             "expert is not credited at full strength."),
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def flat_d2pv(ds):
    """Distance from the PV, in mm, for every REAL node, flattened in the same
    order that train.capture() flattens its routing arrays.

    ds.aux is the raw displacement vector from the PV, carried through
    graph_build precisely because it is not a model input -- dataset.py builds
    edge features from it and nothing standardises it. So its norm is d2PV in
    physical mm, not in units of some training-set sigma.
    """
    return np.linalg.norm(ds.aux, axis=-1)[ds.mask].astype(np.float32)


def attach_d2pv(d, ds):
    """Join per-node d2PV onto an already-loaded routing dict.

    capture() walks the loader with shuffle=False and stores real nodes in
    dataset order, so ds.mask flattened in the same order lines up row for row.
    That is ASSERTED against the role column rather than assumed: if the two
    disagree the join is silently wrong, and a wrong join would show up as a
    physically plausible but fabricated d2PV-vs-expert correlation, which is
    exactly the kind of error that survives a plausibility check.
    """
    dd = flat_d2pv(ds)
    rr = ds.role[ds.mask]
    n = min(len(dd), len(d["role"]))
    if not np.array_equal(rr[:n], d["role"][:n]):
        bad = int((rr[:n] != d["role"][:n]).sum())
        raise SystemExit(
            f"d2PV join failed: {bad:,} of {n:,} rows have a different role in "
            f"the dataset than in the routing arrays. The two are not in the "
            f"same order, so d2PV cannot be attached. (Was the routing captured "
            f"with shuffle_nodes=False over these exact shards?)")
    out = {k: (v[:n] if isinstance(v, np.ndarray) and len(v) == len(d["role"])
               else v) for k, v in d.items()}
    out["d2pv"] = dd[:n]
    return out


def drop_evt_rows(d):
    """Vertices only: the EVT node has no role and would be a meaningless bar."""
    keep = d["node_type"] != NODE_EVT
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep)
               else v) for k, v in d.items()}
    out["n_layers"] = out["expert_idx"].shape[1]
    out["k"] = out["expert_idx"].shape[2]
    return out


def split_by_channel(d, names):
    """One source dict per probe channel, so each gets its own x-axis group.

    build_probe_stats numbers channels by their position in the split dict,
    which is the order they were passed on the command line, so channel_id ci
    is names[ci]. Splitting here is what makes a multi-channel run a SCAN
    (Lc+ vs D0 vs Ds+ vs D+, side by side) rather than one merged bar that
    averages the very differences being looked for.
    """
    out = []
    present = sorted(int(c) for c in np.unique(d["channel_id"]))
    for ci in present:
        sel = d["channel_id"] == ci
        nm = names[ci] if 0 <= ci < len(names) else f"chan{ci}"
        sub = {k: (v[sel] if isinstance(v, np.ndarray)
                   and len(v) == len(d["role"]) else v)
               for k, v in d.items()}
        sub["label"] = nm
        out.append(sub)
    return out


def load_npz(path, label):
    """Read one routing npz into the plain dict this script works with.

    EVT rows are NOT dropped here: d2PV is joined on first, and that join needs
    the full real-node list to line up with the dataset.
    """
    z = np.load(path)
    d = {k: z[k] for k in ("role", "context", "node_type", "expert_idx",
                           "expert_gate")}
    d["event_id"] = (z["event_id"] if "event_id" in z.files
                     else np.arange(len(d["role"]), dtype=np.int32))
    d["is_cand"] = (z["is_cand"].astype(bool) if "is_cand" in z.files
                    else (d["node_type"] == 3))
    d["label"] = label
    d["n_layers"] = d["expert_idx"].shape[1]
    d["k"] = d["expert_idx"].shape[2]
    return d


def evaluate_probe(run_dir, stats_path, batch_size, max_events, device, label):
    """Forward the probe events through one trained model, return its routing.

    The probe features are standardised with the ORIGINAL run's statistics --
    that is what build_probe_stats guarantees -- never with statistics
    recomputed on the probe, which would re-centre every feature onto this
    channel's own mean and hand the model inputs on a scale it never trained on.
    """
    import torch
    from dataset import GraphDataset, make_loader
    from model import MoEGraphTransformer
    from train import capture

    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ds = GraphDataset(stats_path, split="test", shuffle_nodes=False,
                      mask_features=cfg.get("mask_features") or None,
                      evt_mode=cfg.get("evt_mode", "node"), verbose=False)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=2)
    model = MoEGraphTransformer(
        len(ds.feature_names), d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg.get("d_ff"),
        n_experts=cfg["n_experts"], k=cfg["k"], dropout=cfg["dropout"],
        use_moe=bool(cfg.get("use_moe", 1)), n_global=ds.n_global).to(device)
    sd = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    _, rt = capture(model, loader, device, max_events=max_events)
    if rt is None:
        raise SystemExit(f"{run_dir}: no routing captured (use_moe=0?)")

    d = {k: rt[k] for k in ("role", "context", "node_type", "expert_idx",
                            "expert_gate", "event_id", "is_cand",
                            "channel_id")}
    d["is_cand"] = d["is_cand"].astype(bool)
    d["label"] = label
    d["n_layers"] = d["expert_idx"].shape[1]
    d["k"] = d["expert_idx"].shape[2]
    return d, ds


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------
def count_rows(src, sel, layer, ne, mode):
    """Per-expert counts for one group of vertices, in one of three conventions."""
    if mode == "top1":
        return np.bincount(src["expert_idx"][sel, layer, 0].astype(np.int64),
                           minlength=ne)[:ne].astype(np.float64)
    if mode == "topk":
        return np.bincount(src["expert_idx"][sel, layer, :].ravel()
                           .astype(np.int64), minlength=ne)[:ne].astype(np.float64)
    if mode == "gate":
        out = np.zeros(ne, np.float64)
        if sel.sum():
            np.add.at(out, src["expert_idx"][sel, layer, :].ravel()
                      .astype(np.int64),
                      src["expert_gate"][sel, layer, :].ravel().astype(np.float64))
        return out
    raise ValueError(f"count_mode must be one of {COUNT_MODES}")


def build_counts(groups, n_layers, ne, mode):
    """groups is a list of (label, source_dict, boolean mask)."""
    counts = np.zeros((n_layers, len(groups), ne), np.float64)
    for L in range(n_layers):
        for gi, (_, src, sel) in enumerate(groups):
            counts[L, gi] = count_rows(src, sel, L, ne, mode)
    return counts


def print_table(counts, labels, ne, header, fmt="{:>9,.0f}", total=True,
                sizes=None):
    """Same layout as attention_maps._print_table, but with a label column wide
    enough for the sample tags -- "Bs0 [Bs2DsTauNu]" does not fit in 11."""
    w = max(12, max((len(l) for l in labels), default=12) + 1)
    for L in range(counts.shape[0]):
        print(f"\n=== layer {L}: {header} ===")
        print("  " + "group".ljust(w)
              + ("" if sizes is None else "vertices".rjust(10))
              + "".join(f"{'e%d' % e:>9}" for e in range(ne))
              + ("    total" if total else ""))
        for gi, lab in enumerate(labels):
            row = counts[L, gi]
            line = ("  " + lab.ljust(w)
                    + ("" if sizes is None else f"{sizes[gi]:>10,}")
                    + "".join("        -" if np.isnan(v) else fmt.format(v)
                              for v in row))
            if total:
                line += f"{np.nan_to_num(row).sum():>9,.0f}"
            print(line)
        if total:
            print("  " + "TOTAL".ljust(w)
                  + ("" if sizes is None else " " * 10)
                  + "".join(f"{v:>9,.0f}" for v in counts[L].sum(0)))


# --------------------------------------------------------------------------
# group construction
# --------------------------------------------------------------------------
def charm_elsewhere(src):
    """Per vertex: does its EVENT contain some other charm-role vertex?

    Only used with --split_charm. Whole-event, not per-hemisphere: the routing
    npz carries no hemisphere column, so the second b hadron of a Z -> bb event
    counts as "elsewhere" too. That dilution is why this is a fallback for
    single-sample cases and not the default -- when the topologies live in
    separate files, tag by file instead.
    """
    _, inv = np.unique(src["event_id"], return_inverse=True)
    isc = (src["role"] == ROLE_CHARM).astype(np.float64)
    per_event = np.bincount(inv, weights=isc, minlength=inv.max() + 1)
    return (per_event[inv] - isc) > 0.5


def groups_by_role(sources, min_n, split_charm=False):
    out = []
    for src in sources:
        extra = [(None, "")]
        if split_charm:
            ce = charm_elsewhere(src)
            extra = [(ce, " +charm"), (~ce, " -charm")]
        for r in ROLE_ORDER:
            base = src["role"] == r
            for m, sfx in extra:
                sel = base if m is None else (base & m)
                if int(sel.sum()) < min_n:
                    continue
                out.append((f"{ROLE_NAMES[r]}{sfx} [{src['label']}]", src, sel))
    return out


def groups_tau_by_parent(sources, min_n, split_charm=False, role=ROLE_TAU):
    """One group per (parent hadron, source sample) for a single role.

    The source tag is what keeps the probe channel as its own bar instead of
    being summed into the B+ / Bc+ bars of the training mix.
    """
    out = []
    for src in sources:
        isrole = src["role"] == role
        extra = [(None, "")]
        if split_charm:
            ce = charm_elsewhere(src)
            extra = [(ce, " +charm"), (~ce, " -charm")]
        # parents ordered by how many vertices they hold, biggest first
        ctxs = [int(c) for c in np.unique(src["context"][isrole]) if int(c) != 0]
        ctxs.sort(key=lambda c: -int((isrole & (src["context"] == c)).sum()))
        for c in ctxs:
            base = isrole & (src["context"] == c)
            for m, sfx in extra:
                sel = base if m is None else (base & m)
                if int(sel.sum()) < min_n:
                    continue
                nm = CTX_NAMES.get(c, str(c))
                out.append((f"{nm}{sfx} [{src['label']}]", src, sel))
    return out


D2PV_BINS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, np.inf)


def _bin_label(lo, hi):
    if hi == np.inf:
        return f">{lo:g}mm"
    return f"{lo:g}-{hi:g}mm"


def groups_by_d2pv(sources, min_n, bins=D2PV_BINS, exclude_pv=True, role=None):
    """One group per (distance-from-PV bin, source sample).

    WHY THIS PLOT
        Role and parent hadron are TRUTH labels -- useful for interpreting the
        routing, but not available to the model. d2PV is an OBSERVABLE the model
        actually sees (via log1p_d2PV and the edge features). So this plot asks
        a different and more mechanistic question: is the router simply cutting
        on displacement?

        If the expert pattern turns out to be a clean function of d2PV, then the
        "specialisation by role" seen elsewhere is largely a restatement of the
        fact that taus, charm and strange decay at different distances -- and a
        probe channel with an unusual lifetime (a long-lived Ks, say) should
        land in whichever expert owns its displacement range, not in a new one.
        That is precisely the prediction a lifetime probe is built to test.

    The PV itself is excluded by default: its d2PV is ~0 by construction, so it
    would pile into the first bin and swamp the genuinely small displacements.
    """
    out = []
    for src in sources:
        if "d2pv" not in src:
            continue
        base = np.ones(len(src["role"]), bool)
        if exclude_pv:
            base &= src["role"] != 5
        if role is not None:
            base &= src["role"] == role
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = base & (src["d2pv"] >= lo) & (src["d2pv"] < hi)
            if int(sel.sum()) < min_n:
                continue
            out.append((f"{_bin_label(lo, hi)} [{src['label']}]", src, sel))
    return out


def print_d2pv_summary(sources, role_names=(1, 2, 3, 4)):
    """The displacement distribution itself, before any routing is involved.

    Worth printing next to the routing plot: if a probe channel's vertices sit
    at systematically different d2PV from the training mix, any routing
    difference may be entirely explained by that, with nothing left over to
    attribute to the decay itself.
    """
    print("\n  === d2PV distribution (mm), by role and source ===")
    print("    " + "role".ljust(10) + "source".ljust(16)
          + "n".rjust(10) + "median".rjust(10) + "mean".rjust(10)
          + "p90".rjust(10))
    for src in sources:
        if "d2pv" not in src:
            continue
        for r in role_names:
            s = src["role"] == r
            if s.sum() < 50:
                continue
            d = src["d2pv"][s]
            print(f"    {ROLE_NAMES[r]:<10}{src['label'][:15]:<16}"
                  f"{int(s.sum()):>10,}{np.median(d):>10.3f}"
                  f"{d.mean():>10.3f}{np.percentile(d, 90):>10.3f}")


# --------------------------------------------------------------------------
# one model, all three plots
# --------------------------------------------------------------------------
def run_model(run_dir, probes, outdir, ne, count_mode, min_n, sources_wanted,
              split_charm, role=ROLE_TAU, d2pv_bins=D2PV_BINS, test_ds=None,
              d2pv_role=None):
    label = os.path.basename(run_dir.rstrip("/"))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    test = load_npz(os.path.join(run_dir, "routing_test.npz"), "test")
    if test_ds is not None:
        test = attach_d2pv(test, test_ds)
    test = drop_evt_rows(test)
    probes = [drop_evt_rows(p_) for p_ in probes]
    n_layers, k = test["n_layers"], test["k"]
    for p_ in probes:
        if p_["n_layers"] != n_layers:
            raise SystemExit("probe and test routing have different layer counts")

    sources = ([test] if sources_wanted in ("both", "test") else []) + \
              (probes if sources_wanted in ("both", "probe") else [])
    ylabel, expl = _COUNT_DOC[count_mode]
    os.makedirs(outdir, exist_ok=True)
    written = []

    print(f"\n{'#' * 78}")
    print(f"# {label}   (evt_mode={cfg.get('evt_mode', 'node')}, "
          f"{ne} experts, top-{k} routing)")
    print(f"{'#' * 78}")
    print(f"  bar height = {ylabel}")
    print(f"    {expl}")
    for s in sources:
        print(f"  source '{s['label']}': {len(s['role']):,} vertices, "
              f"{len(np.unique(s['event_id'])):,} events, "
              f"{int((s['role'] == role).sum()):,} of role "
              f"{ROLE_NAMES[role]}")

    if any("d2pv" in s_ for s_ in sources):
        print_d2pv_summary(sources)

    for tag, groups, title in (
            ("by_role", groups_by_role(sources, min_n, split_charm),
             "by vertex role"),
            (f"{ROLE_NAMES[role]}_by_parent",
             groups_tau_by_parent(sources, min_n, split_charm, role),
             f"{ROLE_NAMES[role]} vertices, by parent hadron"),
            ("by_d2pv" + (f"_{ROLE_NAMES[d2pv_role]}" if d2pv_role else ""),
             groups_by_d2pv(sources, min_n, d2pv_bins, role=d2pv_role),
             "by distance from the PV"
             + (f", {ROLE_NAMES[d2pv_role]} vertices only" if d2pv_role
                else " (non-PV vertices)"))):
        if len(groups) < 2:
            print(f"\n  !! fewer than 2 groups above the {min_n}-vertex floor "
                  f"for '{title}'; skipping")
            continue
        labels = [g[0] for g in groups]
        sizes = [int(g[2].sum()) for g in groups]
        counts = build_counts(groups, n_layers, ne, count_mode)
        enr = _enrichment(counts)

        print_table(counts, labels, ne, f"{ylabel} -- {title}", sizes=sizes)
        print_table(enr, labels, ne,
                    f"ENRICHMENT = P(expert|group)/P(expert) -- {title}",
                    fmt="{:>9.2f}", total=False, sizes=sizes)
        print(f"  ('-' = fewer than {MIN_CELL} in the cell or on that expert: "
              f"the ratio would be Poisson noise)")

        sfx = "" if count_mode == "topk" else f"_{count_mode}"
        written.append(_bar_panels(
            counts, labels, ne, k, n_layers,
            f"{label} -- {title}   ({ne} experts, top-{k}, "
            f"count_mode={count_mode})",
            ylabel, outdir, f"probe_fig8_{tag}_counts{sfx}.png"))
        written.append(_bar_panels(
            enr, labels, ne, k, n_layers,
            f"{label} -- {title}, enrichment over chance   ({ne} experts, "
            f"top-{k}, count_mode={count_mode})",
            "P(expert | group) / P(expert)", outdir,
            f"probe_fig8_{tag}_enrichment{sfx}.png", enrich=True))
        np.savez_compressed(
            os.path.join(outdir, f"probe_fig8_{tag}{sfx}.npz"),
            counts=counts, enrichment=enr, labels=np.array(labels),
            group_sizes=np.array(sizes), n_experts=ne, k=k,
            count_mode=count_mode)
    return written


# --------------------------------------------------------------------------
def _selftest():
    print("=== probe_fig8 self-test ===")
    rng = np.random.default_rng(0)
    ne, k, nl, n = 8, 2, 2, 6000
    ev = np.repeat(np.arange(n // 3), 3).astype(np.int32)
    role = rng.choice([1, 2, 3, 5], n).astype(np.int8)
    ctx = rng.choice([521, 541, 531], n).astype(np.int32)
    idx = rng.integers(0, ne, (n, nl, k)).astype(np.int8)
    gate = np.tile(np.array([0.7, 0.3], np.float32), (n, nl, 1))
    src = {"role": role, "context": ctx, "node_type": np.full(n, 2, np.int8),
           "expert_idx": idx, "expert_gate": gate, "event_id": ev,
           "is_cand": np.zeros(n, bool), "label": "fake",
           "n_layers": nl, "k": k}

    sel = role == 1
    ntau = int(sel.sum())
    c1 = count_rows(src, sel, 0, ne, "top1")
    ck = count_rows(src, sel, 0, ne, "topk")
    cg = count_rows(src, sel, 0, ne, "gate")
    print(f"  {ntau:,} tau vertices")
    print(f"    top1 row total {c1.sum():,.0f}  (must equal the vertex count)")
    print(f"    topk row total {ck.sum():,.0f}  (must equal k x that = "
          f"{k * ntau:,})")
    print(f"    gate row total {cg.sum():,.1f}  (must equal the vertex count)")
    assert c1.sum() == ntau
    assert ck.sum() == k * ntau
    assert abs(cg.sum() - ntau) < 1e-6

    # groups must be disjoint: no vertex may be drawn in two bars
    gs = groups_by_role([src], min_n=1)
    stack = np.stack([g[2] for g in gs])
    assert stack.sum(0).max() <= 1, "a vertex appears in two role groups"
    print(f"  by_role groups: {len(gs)}, no vertex in more than one "
          f"({stack.sum():,} of {n:,} vertices used; the rest are roles below "
          f"the floor)")

    gp = groups_tau_by_parent([src], min_n=1)
    stack = np.stack([g[2] for g in gp])
    assert stack.sum(0).max() <= 1, "a vertex appears in two parent groups"
    assert stack.sum() == ntau, "parent groups do not cover every tau"
    print(f"  tau_by_parent groups: {[g[0] for g in gp]}")
    print(f"    they partition the taus exactly ({stack.sum():,} = {ntau:,})")

    # two sources must stay separate rather than being summed together
    src2 = dict(src)
    src2["label"] = "Bs2DsTauNu"
    g2 = groups_tau_by_parent([src, src2], min_n=1)
    assert len(g2) == 2 * len(gp)
    assert any("[Bs2DsTauNu]" in g[0] for g in g2)
    assert any("[fake]" in g[0] for g in g2)
    print(f"  two sources -> {len(g2)} groups, probe tagged separately: "
          f"{[g[0] for g in g2 if 'Bs2DsTauNu' in g[0]]}")

    # d2PV grouping: bins must partition, and the PV must be excluded
    src_d = dict(src)
    src_d["d2pv"] = np.abs(rng.normal(0, 3, n)).astype(np.float32)
    src_d["role"] = np.where(np.arange(n) % 5 == 0, 5, src_d["role"]).astype(np.int8)
    gd = groups_by_d2pv([src_d], min_n=1)
    stack = np.stack([g[2] for g in gd])
    assert stack.sum(0).max() <= 1, "a vertex falls in two d2PV bins"
    npv = int((src_d["role"] != 5).sum())
    assert stack.sum() == npv, (
        f"d2PV bins cover {stack.sum()} vertices, expected {npv} non-PV")
    assert not any((src_d["role"][g[2]] == 5).any() for g in gd), \
        "a PV vertex leaked into the d2PV plot"
    print(f"  d2PV bins: {[g[0] for g in gd]}")
    print(f"    partition the {npv:,} non-PV vertices exactly, PV excluded")

    # the join must refuse a misaligned dataset rather than fabricate d2PV
    class _DS:
        def __init__(self, roles, aux):
            self.role = roles
            self.aux = aux
            self.mask = np.ones(roles.shape, bool)
    ev_, nn_ = 40, 5
    roles = rng.integers(0, 4, (ev_, nn_)).astype(np.int8)
    aux = rng.normal(0, 1, (ev_, nn_, 3)).astype(np.float32)
    dsrc = {"role": roles.reshape(-1), "node_type": np.full(ev_ * nn_, 2, np.int8),
            "expert_idx": np.zeros((ev_ * nn_, nl, k), np.int8),
            "expert_gate": np.zeros((ev_ * nn_, nl, k), np.float32),
            "event_id": np.repeat(np.arange(ev_), nn_).astype(np.int32),
            "is_cand": np.zeros(ev_ * nn_, bool), "context": np.zeros(ev_ * nn_, np.int32),
            "label": "x", "n_layers": nl, "k": k}
    ok = attach_d2pv(dsrc, _DS(roles, aux))
    assert "d2pv" in ok and len(ok["d2pv"]) == ev_ * nn_
    exp = np.linalg.norm(aux, axis=-1).reshape(-1)
    assert np.allclose(ok["d2pv"], exp), "d2PV values do not match ||aux||"
    print(f"  attach_d2pv: joined {len(ok['d2pv']):,} rows, values = ||aux||")
    shuffled = dsrc.copy()
    shuffled["role"] = (dsrc["role"] + 1).astype(np.int8)
    try:
        attach_d2pv(shuffled, _DS(roles, aux))
        raise AssertionError("should have refused a misaligned join")
    except SystemExit as ex:
        print(f"  misaligned join refused: {str(ex).splitlines()[0][:46]}...")

    counts = build_counts(g2, nl, ne, "topk")
    assert counts.shape == (nl, len(g2), ne)
    tot = counts[0].sum()
    assert abs(tot - 2 * k * ntau) < 1e-6, tot
    print(f"  build_counts shape {counts.shape}, layer-0 total {tot:,.0f} "
          f"(= 2 sources x k x taus)")

    # charm proxy: a vertex must never count itself as its own companion
    one_charm = dict(src)
    one_charm["role"] = np.where(np.arange(n) % 3 == 0, 2, 1).astype(np.int8)
    ce = charm_elsewhere(one_charm)
    # event_id groups vertices in threes, so exactly one charm per event: no
    # charm vertex may see a companion, every non-charm vertex must see one
    assert not ce[one_charm["role"] == 2].any(), "a charm vertex saw itself"
    assert ce[one_charm["role"] != 2].all(), "a companion charm was missed"
    print(f"  charm_elsewhere: correct on a one-charm-per-event sample "
          f"(no vertex is its own companion)")
    print("\nALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(
        description="Fig. 8 expert plots with a probe channel as its own group.")
    ap.add_argument("--runs", nargs="+",
                    help="one or more trained run directories")
    ap.add_argument("--probe_shards",
                    help="directory holding the probe shards")
    ap.add_argument("--channel", nargs="+", default=["Bs2DsTauNu"])
    ap.add_argument("--count_mode", default="topk", choices=COUNT_MODES,
                    help="what one unit of bar height means. topk (default) "
                         "matches the existing fig8: k counts per vertex, so "
                         "row totals are k x the vertex count. top1 gives one "
                         "count per vertex. gate is gate-weighted.")
    ap.add_argument("--sources", default="both",
                    choices=["both", "probe", "test"],
                    help="which samples go on the x axis")
    ap.add_argument("--role", type=int, default=ROLE_TAU,
                    help="which role gets the by-parent plot "
                         "(1=tau, 2=charm, 3=bottom)")
    ap.add_argument("--min_n", type=int, default=200,
                    help="drop a group holding fewer vertices than this")
    ap.add_argument("--split_charm", action="store_true",
                    help="additionally split each group by whether a charm "
                         "vertex sits elsewhere in the event. Off by default: "
                         "with separate samples the file tag is exact and this "
                         "proxy is not needed.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_events", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="output directory (default: <run>/probe_fig8_<channel>)")
    ap.add_argument("--d2pv_bins", default=None,
                    help="comma-separated bin edges in mm for the "
                         "distance-from-PV plot, e.g. "
                         "'0,0.5,1,2,5,10,30,inf'. Default: "
                         + ",".join(f"{b:g}" for b in D2PV_BINS))
    ap.add_argument("--d2pv_role", type=int, default=None,
                    help="restrict the distance-from-PV plot to one role "
                         "(1=tau, 2=charm, 3=bottom, 4=strange). Default: all "
                         "non-PV vertices. Use 2 for a charm-lifetime scan, "
                         "4 for the long-lived strange population.")
    ap.add_argument("--no_d2pv", action="store_true",
                    help="skip the distance-from-PV plot (it needs the test "
                         "split reloaded, which costs a minute or two)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    bins = D2PV_BINS
    if args.d2pv_bins:
        bins = tuple(float(x) for x in args.d2pv_bins.split(","))
        if len(bins) < 2 or any(b >= c for b, c in zip(bins[:-1], bins[1:])):
            raise SystemExit("--d2pv_bins must be increasing, with >=2 edges")

    if args.selftest:
        _selftest()
        return
    if not (args.runs and args.probe_shards):
        ap.error("need --runs and --probe_shards (or --selftest)")

    import torch
    from probe_eval import build_probe_stats

    dev = torch.device(args.device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    chan_tag = "_".join(args.channel)
    written = []
    for run_dir in args.runs:
        outdir = args.out or os.path.join(run_dir, f"probe_fig8_{chan_tag}")
        os.makedirs(outdir, exist_ok=True)
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        print(f"\n=== {run_dir}: building probe stats from "
              f"{args.probe_shards}", flush=True)
        stats = build_probe_stats(cfg["stats"], args.probe_shards,
                                  args.channel,
                                  os.path.join(outdir, "probe_stats.json"))
        probe, probe_ds = evaluate_probe(run_dir, stats, args.batch_size,
                                         args.max_events, dev, chan_tag)
        if not args.no_d2pv:
            probe = attach_d2pv(probe, probe_ds)
        del probe_ds
        # one source per channel: a multi-channel run is a scan, not a blend
        probes = split_by_channel(probe, args.channel)
        print("  probe sources: " + ", ".join(
            f"{p_['label']} ({len(p_['role']):,} nodes)" for p_ in probes))

        test_ds = None
        if not args.no_d2pv and args.sources in ("both", "test"):
            # no forward pass needed here -- only aux/mask/role are read, so
            # this is a data load, not an evaluation
            from dataset import GraphDataset
            print("  loading the test split to join d2PV onto the stored "
                  "routing (no model forward)", flush=True)
            test_ds = GraphDataset(cfg["stats"], split="test",
                                   shuffle_nodes=False,
                                   mask_features=cfg.get("mask_features") or None,
                                   evt_mode=cfg.get("evt_mode", "node"),
                                   verbose=False)

        written += run_model(run_dir, probes, outdir, int(cfg["n_experts"]),
                             args.count_mode, args.min_n, args.sources,
                             args.split_charm, role=args.role,
                             d2pv_bins=bins, test_ds=test_ds,
                             d2pv_role=args.d2pv_role)
        del test_ds

    print("\nwrote:")
    for p in written:
        print("  " + p)
    if len(args.runs) > 1:
        print("\nReminder: expert NUMBERS are not comparable between two "
              "separately trained\nruns. Compare the PATTERN across the two "
              "figures (is the probe bar shaped like\nits neighbours?), not "
              "which expert index it lands on.")


if __name__ == "__main__":
    main()