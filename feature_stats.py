#!/usr/bin/env python3
print("Starting feature_stats", flush=True)
"""
feature_stats.py

Stage 1 of the MoE-graph-transformer pipeline: decide the train/val/test file
split, then compute feature standardisation statistics ON THE TRAIN SPLIT ONLY,
and write both to a single stats.json that dataset.py and train.py consume.

WHY THE SPLIT LIVES HERE
    The split and the statistics have to agree, or the statistics leak test
    information into training. Deriving both in one place and writing them to
    one file makes that impossible to get wrong by accident. dataset.py reads
    the split from stats.json rather than recomputing it.

SPLIT SCHEME -- per channel, file level, symmetric hold-out
    Splitting by FILE, not by shuffled event: the Bc and Bu channels come from
    very few production jobs, so events within a file may share seed lineage,
    and an event-level split would leak that correlation across the boundary.

    Per channel, with n files and target hold-out fraction f (default 0.15):
        n_hold  = max(1, round(f * n))        # same count for val and test
        n_train = n - 2 * n_hold
    Symmetric val/test because both are used for genuine model decisions (early
    stopping vs final numbers) and want comparable statistical power.

    With the built channels this gives:
        8 files  -> 6 / 1 / 1   (75.0 / 12.5 / 12.5 %)
        10 files -> 6 / 2 / 2   (60.0 / 20.0 / 20.0 %)
        5 files  -> 3 / 1 / 1   (60.0 / 20.0 / 20.0 %)
    i.e. 60-75% train everywhere, which is close enough that the final class
    mixture is set by dataset.py's per-channel event caps rather than drifting
    out of the split.

    Assignment is by md5 of the file's basename, not by sorted order: chunk
    index may correlate with production batch, and hashing decorrelates from
    that while staying exactly reproducible (`--split_salt` changes it).

NORMALISATION GROUPS -- and why not one global scale per feature
    The feature table is block-sparse: cand_* is populated only on SV3pi nodes,
    evt_* only on the EVT node, and the PV legitimately differs from a
    secondary vertex by an order of magnitude in ntrk (~12 vs ~3) and mass
    (~40 vs ~2 GeV).

    Two distinct concepts are therefore tracked separately:

      APPLICABLE_TYPES[f] : the node types on which feature f carries
          information. Statistics for f are computed over ONLY those nodes.
          Without this, cand_* statistics would be dominated by the structural
          zeros on non-candidate vertices, and standardising would turn "0 =
          not applicable" into some arbitrary non-zero constant -- destroying
          the semantics of the sparse blocks.

      NORM_GROUP_OF[node_type] : which scale a node standardises against.
          PV / SV+SV3pi / EVT. SV and SV3pi share a scale deliberately: their
          vertex-level features are the same physical quantities, and giving
          them separate scales would leak "is this a 3pi candidate" into the
          normalisation itself.

    Consequence worth knowing: within-group standardisation means the model
    cannot read "the PV has more tracks than the SVs" from raw magnitudes. The
    is_PV flag makes that recoverable, and in exchange the SV ntrk range
    (2,3,4,5...) uses the full dynamic range instead of being compressed by the
    PV's bimodality. Global ("ALL") statistics are computed and written too, so
    the choice can be ablated from dataset.py without recomputing.

    Binary flags are never standardised (see NO_STANDARDIZE); statistics are
    still reported for them because "what fraction of SVs are 3pi candidates"
    is a useful sanity number.

TWO PASSES
    Pass 1: exact count / sum / sumsq / min / max per (group, feature), plus
            the applicability-violation check.
    Pass 2: with counts known, subsample each (group, feature) at a known
            fraction and compute quantiles from the sample.
    Exact quantiles over ~1e7 node rows would mean holding ~2 GB; a known-
    fraction subsample is exact-unbiased and bounded by --quantile_cap. Two
    passes are I/O bound and cost a few minutes.

USAGE
    python3 feature_stats.py --shards /eos/user/b/bnag/Bc2TauNu/shards \\
                             --out    /eos/user/b/bnag/Bc2TauNu/stats.json

    python3 feature_stats.py --shards ... --out ... --hold_frac 0.15 \\
                             --split_salt v1 --quantile_cap 300000
"""

import sys
#sys.path.insert(0, "/eos/user/b/bnag/python_packages")

import argparse
import datetime
import glob
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
torch.cuda.is_available = lambda: False

# --------------------------------------------------------------------------
# node grouping
# --------------------------------------------------------------------------
# Mirrors graph_build.NODE_* . Imported by value rather than by import so this
# script can run against a shard directory without graph_build's dependencies
# (uproot in particular); the values are cross-checked against each shard's
# meta["node_type_names"] at load time.
NODE_PAD, NODE_PV, NODE_SV, NODE_SV3PI, NODE_EVT = 0, 1, 2, 3, 4

NORM_GROUP_OF = {
    NODE_PV:    "PV",
    NODE_SV:    "SV",
    NODE_SV3PI: "SV",     # shares the SV scale on purpose -- see docstring
    NODE_EVT:   "EVT",
}
GROUPS = ["PV", "SV", "EVT"]
GROUP_TYPES = {g: [t for t, gg in NORM_GROUP_OF.items() if gg == g] for g in GROUPS}


def build_applicability(feature_blocks):
    """feature -> set of node types on which it carries information.

    Derived from the block structure in the shard metadata so it cannot drift
    out of sync with graph_build's FEAT_BLOCKS.
    """
    blocks = {name: list(names) for name, names in feature_blocks.items()}
    vertex_types = {NODE_PV, NODE_SV, NODE_SV3PI}
    sv_types = {NODE_SV, NODE_SV3PI}

    app = {}
    # geometry: the four displacement features are identically zero on the PV
    # (d2PV == 0 by construction), the thrust angle is meaningful there.
    for f in blocks["geometry"]:
        app[f] = sv_types if f != "cos_thrust_angle" else vertex_types
    for f in blocks["quality"]:
        app[f] = vertex_types
    for f in blocks["mass"]:
        app[f] = vertex_types
    for f in blocks["impact"]:
        app[f] = sv_types          # pseudotrack: one per NON-PV vertex
    for f in blocks["flags"]:
        app[f] = {NODE_PV, NODE_SV, NODE_SV3PI, NODE_EVT}
    for f in blocks["cand3pi"]:
        app[f] = {NODE_SV3PI}      # NOT all of SV -- structural zeros elsewhere
    for f in blocks["event"]:
        app[f] = {NODE_EVT}
    return app


# Binary indicators: already 0/1, and standardising them would destroy the
# semantics for no benefit.
NO_STANDARDIZE_BLOCKS = ("flags",)


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------
def split_files(files, hold_frac, salt):
    """Deterministic per-channel file split -> (train, val, test) lists."""
    n = len(files)
    if n < 3:
        raise ValueError(
            f"need >= 3 files to split, got {n}. Rebuild that channel with a "
            f"larger --max_files, or exclude it.")
    order = sorted(files, key=lambda p: hashlib.md5(
        f"{salt}:{os.path.basename(p)}".encode()).hexdigest())
    n_hold = max(1, int(hold_frac * n + 0.5))
    while n - 2 * n_hold < 1:      # never starve train
        n_hold -= 1
    n_train = n - 2 * n_hold
    return (order[:n_train],
            order[n_train:n_train + n_hold],
            order[n_train + n_hold:])


# --------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------
class Accum:
    """Exact count/sum/sumsq/min/max for one (group, feature) pair."""

    __slots__ = ("n", "s", "ss", "lo", "hi", "n_zero", "n_nonfinite")

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.lo = np.inf
        self.hi = -np.inf
        self.n_zero = 0
        self.n_nonfinite = 0

    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        bad = ~np.isfinite(x)
        if bad.any():
            self.n_nonfinite += int(bad.sum())
            x = x[~bad]
            if x.size == 0:
                return
        self.n += x.size
        self.s += float(x.sum())
        self.ss += float(np.dot(x, x))
        self.lo = min(self.lo, float(x.min()))
        self.hi = max(self.hi, float(x.max()))
        self.n_zero += int((x == 0.0).sum())

    def finish(self):
        if self.n == 0:
            return None
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        return {"count": self.n, "mean": mean, "std": float(np.sqrt(var)),
                "min": self.lo, "max": self.hi,
                "n_zero": self.n_zero, "n_nonfinite": self.n_nonfinite}


def load_shard(path):
    sh = torch.load(path, map_location="cpu", weights_only=False)
    return sh


def node_selector(node_type, node_mask, types):
    """Boolean (N, MAX_NODES) mask: real node whose type is in `types`."""
    sel = np.zeros(node_type.shape, dtype=bool)
    for t in types:
        sel |= (node_type == t)
    return sel & node_mask


def pass1(files, feature_names, applicability, diag):
    """Exact moments per (group, feature) + applicability violation check."""
    acc = defaultdict(Accum)
    acc_all = defaultdict(Accum)
    n_events = 0
    class_counts = Counter()
    node_type_counts = Counter()
    nvtx_hist = Counter()

    for fi, p in enumerate(files):
        sh = load_shard(p)
        feats = sh["node_feats"].numpy()
        ntype = sh["node_type"].numpy().astype(np.int64)
        nmask = sh["node_mask"].numpy().astype(bool)
        y = sh["y"].numpy().astype(np.int64)

        n_events += feats.shape[0]
        class_counts.update(y.tolist())
        for t, c in zip(*np.unique(ntype[nmask], return_counts=True)):
            node_type_counts[int(t)] += int(c)
        for v, c in zip(*np.unique(sh["n_vtx"].numpy(), return_counts=True)):
            nvtx_hist[int(v)] += int(c)

        # padding must be exactly zero -- re-verified here because
        # feature_stats and the pooled baseline both break silently otherwise
        if feats[~nmask].any():
            diag["padding_nonzero"] += int((feats[~nmask] != 0).sum())

        for j, fname in enumerate(feature_names):
            app = applicability[fname]
            col = feats[:, :, j]

            # violation check: a non-applicable real node must hold exactly 0
            notapp = node_selector(ntype, nmask, set(NORM_GROUP_OF) - app)
            if notapp.any():
                v = int((col[notapp] != 0.0).sum())
                if v:
                    diag[f"applicability_violation_{fname}"] += v

            for g in GROUPS:
                types = [t for t in GROUP_TYPES[g] if t in app]
                if not types:
                    continue
                sel = node_selector(ntype, nmask, types)
                if sel.any():
                    acc[(g, fname)].add(col[sel])
            sel_all = node_selector(ntype, nmask, app)
            if sel_all.any():
                acc_all[fname].add(col[sel_all])

        del sh, feats, ntype, nmask
        print(f"    pass1 [{fi + 1}/{len(files)}] {os.path.basename(p)}",
              flush=True)

    return (acc, acc_all, n_events, class_counts, node_type_counts, nvtx_hist)


def pass2(files, feature_names, applicability, counts, cap, seed):
    """Subsample at known per-key fractions and return quantiles."""
    rng = np.random.default_rng(seed)
    frac = {k: (1.0 if c <= cap else cap / c) for k, c in counts.items()}
    buf = defaultdict(list)

    for fi, p in enumerate(files):
        sh = load_shard(p)
        feats = sh["node_feats"].numpy()
        ntype = sh["node_type"].numpy().astype(np.int64)
        nmask = sh["node_mask"].numpy().astype(bool)

        for j, fname in enumerate(feature_names):
            app = applicability[fname]
            col = feats[:, :, j]
            for g in GROUPS:
                types = [t for t in GROUP_TYPES[g] if t in app]
                if not types:
                    continue
                key = (g, fname)
                f = frac.get(key)
                if f is None:
                    continue
                sel = node_selector(ntype, nmask, types)
                if not sel.any():
                    continue
                v = col[sel]
                v = v[np.isfinite(v)]
                if f < 1.0 and v.size:
                    v = v[rng.random(v.size) < f]
                if v.size:
                    buf[key].append(v.astype(np.float64))

        del sh, feats, ntype, nmask
        print(f"    pass2 [{fi + 1}/{len(files)}] {os.path.basename(p)}",
              flush=True)

    out = {}
    qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    names = ["q001", "q01", "q25", "median", "q75", "q99", "q999"]
    for key, parts in buf.items():
        v = np.concatenate(parts)
        vals = np.quantile(v, qs)
        d = {n: float(x) for n, x in zip(names, vals)}
        d["iqr"] = d["q75"] - d["q25"]
        # IQR/1.349 equals sigma for a Gaussian, so robust and standard scales
        # are directly comparable
        d["robust_std"] = d["iqr"] / 1.349 if d["iqr"] > 0 else 0.0
        d["n_sampled"] = int(v.size)
        out[key] = d
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Split shards per channel and compute train-only feature "
                    "standardisation statistics.")
    ap.add_argument("--shards", required=True, help="shard root directory")
    ap.add_argument("--out", required=True, help="output stats.json path")
    ap.add_argument("--hold_frac", type=float, default=0.15,
                    help="target fraction of files for EACH of val and test")
    ap.add_argument("--split_salt", default="v1",
                    help="changes the file->split assignment reproducibly")
    ap.add_argument("--quantile_cap", type=int, default=300000,
                    help="max sampled values per (group, feature) for quantiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", nargs="*", default=None,
                    help="restrict to these channel subdirectories")
    args = ap.parse_args()

    t0 = time.time()

    # ---- discover channels ----------------------------------------------
    subdirs = sorted(d for d in os.listdir(args.shards)
                     if os.path.isdir(os.path.join(args.shards, d)))
    if args.channels:
        subdirs = [d for d in subdirs if d in args.channels]
    if not subdirs:
        raise SystemExit(f"no channel subdirectories under {args.shards}")

    split = {}
    all_train = []
    for ch in subdirs:
        files = sorted(glob.glob(os.path.join(args.shards, ch, "*.pt")))
        if not files:
            print(f"  !! {ch}: no shards, skipping", flush=True)
            continue
        tr, va, te = split_files(files, args.hold_frac, args.split_salt)
        split[ch] = {"train": tr, "val": va, "test": te}
        all_train += tr

    if not all_train:
        raise SystemExit("no training shards found")

    # ---- schema from the first shard ------------------------------------
    probe = load_shard(all_train[0])
    meta = probe["meta"]
    feature_names = list(meta["feature_names"])
    feature_blocks = {k: list(v) for k, v in meta["feature_blocks"].items()}
    applicability = build_applicability(feature_blocks)
    no_std = set()
    for b in NO_STANDARDIZE_BLOCKS:
        no_std |= set(feature_blocks.get(b, []))

    # cross-check the node-type codes assumed above against the shard's own
    # names, so a renumbering in graph_build cannot silently misgroup nodes
    shard_type_names = {int(k): v for k, v in meta["node_type_names"].items()}
    expected = {NODE_PAD: "pad", NODE_PV: "PV", NODE_SV: "SV",
                NODE_SV3PI: "SV3pi", NODE_EVT: "EVT"}
    if shard_type_names != expected:
        raise SystemExit(f"node type codes changed in graph_build: shard says "
                         f"{shard_type_names}, this script assumes {expected}")
    max_nodes = int(meta["max_nodes"])
    del probe

    # ---- report the split ------------------------------------------------
    print(f"\n=== split (hold_frac={args.hold_frac}, salt='{args.split_salt}') ===",
          flush=True)
    hdr = f"  {'channel':16s} {'files':>5s}  {'tr/va/te':>10s}  {'train %':>8s}"
    print(hdr)
    for ch, s in split.items():
        n = sum(len(s[k]) for k in ("train", "val", "test"))
        print(f"  {ch:16s} {n:5d}  "
              f"{len(s['train']):3d}/{len(s['val']):2d}/{len(s['test']):2d}  "
              f"{100 * len(s['train']) / n:7.1f}%")

    # ---- pass 1 ----------------------------------------------------------
    print(f"\n=== pass 1: exact moments over {len(all_train)} train shards ===",
          flush=True)
    diag = Counter()
    acc, acc_all, n_events, class_counts, ntype_counts, nvtx_hist = pass1(
        all_train, feature_names, applicability, diag)

    stats = {g: {} for g in GROUPS}
    stats["ALL"] = {}
    counts = {}
    for (g, fname), a in acc.items():
        d = a.finish()
        if d is not None:
            stats[g][fname] = d
            counts[(g, fname)] = d["count"]
    for fname, a in acc_all.items():
        d = a.finish()
        if d is not None:
            stats["ALL"][fname] = d

    # ---- pass 2 ----------------------------------------------------------
    print(f"\n=== pass 2: quantiles (cap {args.quantile_cap} per key) ===",
          flush=True)
    q = pass2(all_train, feature_names, applicability, counts,
              args.quantile_cap, args.seed)
    for (g, fname), d in q.items():
        stats[g][fname].update(d)

    # ---- per-split event/class bookkeeping -------------------------------
    print("\n=== counting val/test events ===", flush=True)
    per_split = {}
    for ch, s in split.items():
        per_split[ch] = {}
        for k in ("train", "val", "test"):
            n = 0
            cc = Counter()
            for p in s[k]:
                sh = load_shard(p)
                n += int(sh["y"].shape[0])
                cc.update(sh["y"].numpy().astype(int).tolist())
                del sh
            per_split[ch][k] = {"n_events": n,
                                "class_counts": {str(a): b for a, b in sorted(cc.items())}}
        print(f"  {ch:16s} " + "  ".join(
            f"{k}={per_split[ch][k]['n_events']}" for k in ("train", "val", "test")),
            flush=True)

    # ---- write -----------------------------------------------------------
    out = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "shard_dir": os.path.abspath(args.shards),
        "split_config": {"hold_frac": args.hold_frac,
                         "salt": args.split_salt,
                         "scheme": "per-channel, by file, md5(basename) order, "
                                   "symmetric val/test"},
        "split": {ch: {k: [os.path.relpath(p, args.shards) for p in s[k]]
                       for k in ("train", "val", "test")}
                  for ch, s in split.items()},
        "per_split_counts": per_split,
        "feature_names": feature_names,
        "feature_blocks": feature_blocks,
        "max_nodes": max_nodes,
        "norm_group_of_node_type": {str(k): v for k, v in NORM_GROUP_OF.items()},
        "applicable_types": {f: sorted(int(t) for t in ts)
                             for f, ts in applicability.items()},
        "no_standardize": sorted(no_std),
        "train_summary": {
            "n_shards": len(all_train),
            "n_events": n_events,
            "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
            "node_type_counts": {str(k): v for k, v in sorted(ntype_counts.items())},
            "n_vtx_histogram": {str(k): v for k, v in sorted(nvtx_hist.items())},
        },
        "stats": {g: stats[g] for g in list(GROUPS) + ["ALL"]},
        "recommended": {
            "method": "robust",
            "center": "median",
            "scale": "robust_std",
            "clip": [-5.0, 5.0],
            "note": "standardise feature f on a node only if that node's type "
                    "is in applicable_types[f]; otherwise leave the value at "
                    "exactly 0. Skip features in no_standardize. Use the "
                    "statistics for norm_group_of_node_type[node_type], or the "
                    "'ALL' block to ablate global scaling.",
        },
        "diagnostics": dict(diag),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)

    # ---- summary ---------------------------------------------------------
    print(f"\n=== train summary ===", flush=True)
    print(f"  events {n_events}   shards {len(all_train)}")
    print("  class counts (0=Bc, 1=Bu, 2=bkg): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items())))
    tot = sum(class_counts.values())
    if tot:
        print("  suggested class weights (inverse frequency, mean 1): "
              + ", ".join(f"{k}:{(tot / (len(class_counts) * v)):.3f}"
                          for k, v in sorted(class_counts.items())))
    print("  node types: " + ", ".join(
        f"{expected[k]}:{v}" for k, v in sorted(ntype_counts.items())))

    print("\n=== per-feature statistics (train, by norm group) ===", flush=True)
    for g in GROUPS:
        if not stats[g]:
            continue
        print(f"\n  --- {g} ---")
        print(f"    {'feature':32s} {'count':>10s} {'median':>10s} "
              f"{'robust_std':>11s} {'mean':>10s} {'std':>10s} {'zeros':>7s}")
        for f in feature_names:
            d = stats[g].get(f)
            if d is None:
                continue
            frac_zero = d["n_zero"] / d["count"] if d["count"] else 0.0
            if f in no_std:
                flag = "  (flag, not standardised)"
            elif d.get("robust_std", 0) == 0 and d["std"] == 0:
                flag = "  <- CONSTANT"
            else:
                flag = ""
            print(f"    {f:32s} {d['count']:10d} {d.get('median', float('nan')):10.4f} "
                  f"{d.get('robust_std', float('nan')):11.4f} {d['mean']:10.4f} "
                  f"{d['std']:10.4f} {frac_zero:6.1%}{flag}")

    # ---- checks ----------------------------------------------------------
    print("\n=== checks ===", flush=True)
    problems = []
    for k, v in diag.items():
        problems.append(f"{k} = {v}")
    for g in GROUPS:
        for f, d in stats[g].items():
            if f in no_std:
                continue
            if d["std"] == 0.0:
                problems.append(f"{g}/{f}: constant on train (std=0) -- carries "
                                f"no information, consider dropping")
            if d["n_nonfinite"]:
                problems.append(f"{g}/{f}: {d['n_nonfinite']} non-finite values")
            if d.get("robust_std", 0.0) == 0.0 and d["std"] > 0.0:
                problems.append(f"{g}/{f}: IQR=0 but std>0 -- >50% of values "
                                f"identical; robust scaling will divide by zero, "
                                f"use mean/std for this feature")
    seen = {}
    for ch, s in split.items():
        for k in ("train", "val", "test"):
            for p in s[k]:
                if p in seen:
                    problems.append(f"file in two splits: {p}")
                seen[p] = k
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  no applicability violations, no non-finite values, no constant "
              "features, no file in two splits")

    print(f"\nwrote {args.out}   ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
