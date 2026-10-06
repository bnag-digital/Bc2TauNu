#!/usr/bin/env python3
"""
  python3 pv_distance_routing.py --run RUNDIR
"""
import argparse
import json
import os

import numpy as np

import routing_analysis as ra
from dataset import GraphDataset

# bin edges in mm. Fine near zero because that is where the interesting
# question is, coarse in the tail where there are few vertices.
EDGES = np.array([0.0, 1e-6, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 1e9])
LABELS = ["exactly 0\n(the PV)", "0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5",
          "0.5-1", "1-2", "2-5", "5-10", ">10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    out = args.out or os.path.join(args.run, "report")
    os.makedirs(out, exist_ok=True)
    ne = cfg["n_experts"]

    # routing (already on disk) and the dataset (for the raw displacement)
    rt = ra.Routing(os.path.join(args.run, "routing_test.npz"), n_experts=ne)
    ds = GraphDataset(cfg["stats"], split="test", shuffle_nodes=False,
                      mask_features=cfg.get("mask_features") or None,
                      evt_mode=cfg.get("evt_mode", "node"),
                      channels=cfg.get("channels_used") or cfg.get("channels"), verbose=False)

    # routing_*.npz stores ONLY real nodes, flattened in (event, slot) order
    # with shuffle_nodes off -- so the same flatten of ds.aux lines up. Verify
    # rather than assume: the counts and the roles must both agree, otherwise
    # every number below would be silently misaligned.
    d = np.linalg.norm(ds.aux, axis=-1)              # (N_events, MAX_NODES)
    flat_d = d[ds.mask]
    flat_role = ds.role[ds.mask]
    if flat_d.size != rt.role.size:
        raise RuntimeError(f"node count mismatch: dataset {flat_d.size} vs "
                           f"routing {rt.role.size}; cannot align")
    agree = (flat_role == rt.role).mean()
    print(f"  alignment check: {agree:.4f} of roles agree "
          f"({flat_d.size:,} real nodes)")
    if agree < 0.999:
        raise RuntimeError("roles do not line up; alignment is wrong")

    layers = [args.layer] if args.layer is not None else range(rt.n_layers)
    for L in layers:
        print(f"\n{'=' * 78}\nEXPERT ROUTING vs DISTANCE FROM THE PV -- layer {L}"
              f"\n{'=' * 78}")
        print(f"  {'displacement (mm)':<18}{'n':>9}  "
              + "".join(f"{'e%d' % e:>7}" for e in range(ne))
              + "   modal  PV-expert share")

        # which experts are the PV's home, measured on the PV itself
        pv = rt.role == ra.ROLE_PV if hasattr(ra, "ROLE_PV") else rt.role == 5
        pv_ids = rt.idx[pv, L, :].ravel()
        pv_counts = np.bincount(pv_ids, minlength=ne)[:ne]
        pv_home = set(np.argsort(pv_counts)[::-1][:rt.k].tolist())
        print(f"  (the PV's own top-{rt.k} experts are "
              f"{{{', '.join('e%d' % e for e in sorted(pv_home))}}})\n")

        rows = []
        for b in range(len(EDGES) - 1):
            lo, hi = EDGES[b], EDGES[b + 1]
            sel = (flat_d >= lo) & (flat_d < hi) if b else (flat_d == 0.0)
            n = int(sel.sum())
            if n < 50:
                continue
            ids = rt.idx[sel, L, :].ravel()
            c = np.bincount(ids, minlength=ne)[:ne].astype(float)
            p = c / c.sum()
            share = sum(p[e] for e in pv_home)
            rows.append((LABELS[b], n, p, int(p.argmax()), share))
            print(f"  {LABELS[b].replace(chr(10), ' '):<18}{n:>9,}  "
                  + "".join(f"{v:>7.2f}" for v in p)
                  + f"   e{int(p.argmax())}    {share:>6.3f}")

        # the sentence you actually say: does anything nonzero look like the PV?
        zero = [r for r in rows if r[1] and r[0].startswith("exactly")]
        nz = [r for r in rows if not r[0].startswith("exactly")]
        if zero and nz:
            z_share = zero[0][4]
            first = nz[0]
            print(f"\n  at exactly 0 mm  : {z_share:.3f} of routing weight on "
                  f"the PV experts")
            print(f"  at {first[0]:<12s}: {first[4]:.3f}  "
                  f"(n={first[1]:,})")
            drop = z_share - first[4]
            print(f"  -> the drop happens IMMEDIATELY, in the first bin "
                  f"({drop:+.3f})" if drop > 0.4 else
                  f"  -> the change is gradual, not a step ({drop:+.3f})")

        # figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8))
        labs = [r[0] for r in rows]
        P = np.array([r[2] for r in rows])
        x = np.arange(len(rows))
        cmap = plt.get_cmap("tab10" if ne <= 10 else "tab20")
        bot = np.zeros(len(rows))
        for e in range(ne):
            a1.bar(x, P[:, e], 0.8, bottom=bot, label=f"e{e}",
                   color=cmap(e % cmap.N))
            bot += P[:, e]
        a1.set_xticks(x); a1.set_xticklabels(labs, rotation=45, ha="right",
                                             fontsize=8)
        a1.set_ylabel("fraction of routing weight")
        a1.set_xlabel("distance from the primary vertex (mm)")
        a1.set_title(f"Which experts, by displacement -- layer {L}")
        a1.legend(fontsize=7, ncol=2, loc="upper right")

        a2.plot(x, [r[4] for r in rows], "o-", color="crimson", lw=2)
        a2.set_xticks(x); a2.set_xticklabels(labs, rotation=45, ha="right",
                                            fontsize=8)
        a2.set_ylabel(f"weight on the PV's own experts")
        a2.set_xlabel("distance from the primary vertex (mm)")
        a2.set_ylim(-0.02, 1.02)
        a2.grid(alpha=0.3)
        a2.set_title("Do displaced vertices use the PV's experts?")
        fig.tight_layout()
        p = os.path.join(out, f"pv_distance_routing_layer{L}.png")
        fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
