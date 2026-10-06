#!/usr/bin/env python3
"""
compare_runs.py

Put two or more finished runs next to each other in one table and one figure.
Built for the two ablations that answer questions asked directly in the
meeting:

    "does removing the mass change anything?"
        baseline : --mask_features cand3pi is_3pi_candidate
        ablation : --mask_features cand3pi is_3pi_candidate mass
        The only difference between them is log1p_vertex_mass, because the
        three candidate-block masses (cand_m3pi, cand_m_rho1, cand_m_rho2) are
        already gone in both. So any change is attributable to that one column.

    "...while maintaining same accuracy"  (the reference paper's claim)
        baseline : --mask_features cand3pi is_3pi_candidate
        ablation : the same, plus --use_moe 0

WHAT IT REPORTS
    classification : accuracy, macro-AUC, per-class AUC
    specialisation : NMI(expert, role) at init and trained, and the GAIN, over
                     displaced roles only -- the gain is the learned quantity
    the hard test  : JSD(tau, charm) among 3pi candidates
    invariance     : cross-parent JSD, its sampling-noise floor, cross-role JSD
    the PV         : purity and exclusivity of the primary-vertex expert

Runs with --use_moe 0 have no experts, so every routing column is reported as
"n/a" rather than silently zero.

USAGE
    python3 compare_runs.py runs/n8_k2_s0_masked runs/n8_k2_s0_masked_nomass
    python3 compare_runs.py runs/*_masked* --layer 0 --out compare/
"""

import argparse
import json
import os

import numpy as np

import routing_analysis as ra
import mgt_report as mr


def _quiet(fn, *a, **kw):
    """Call a routing_analysis reporter without its console output."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def collect(run, layer, n_perm=100):
    out = {"run": os.path.basename(run.rstrip("/")), "path": run}
    cfg = {}
    cp = os.path.join(run, "config.json")
    if os.path.exists(cp):
        cfg = json.load(open(cp))
    out["cfg"] = cfg
    out["use_moe"] = bool(cfg.get("use_moe", 1))
    out["masked"] = cfg.get("masked_feature_names") or []
    # "node" for every run predating the --evt_mode flag
    out["evt_mode"] = cfg.get("evt_mode", "node")
    out["n_global"] = cfg.get("n_global", 0)

    mp = os.path.join(run, "metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        t = m.get("test", {})
        out.update({k: t.get(k) for k in
                    ("accuracy", "macro_auc", "macro_f1",
                     "auc_Bc", "auc_Bu", "auc_bkg")})
        out["evt_mva1_auc"] = m.get("evt_mva1_auc")
        out["best_epoch"] = (m.get("best") or {}).get("epoch")

    if not out["use_moe"]:
        return out

    n_exp = cfg.get("n_experts")
    ip = os.path.join(run, "routing_init.npz")
    tp = os.path.join(run, "routing_test.npz")
    if not (os.path.exists(ip) and os.path.exists(tp)):
        return out
    init, test = ra.Routing(ip, n_exp), ra.Routing(tp, n_exp)
    init.event_id = np.load(ip)["event_id"]
    test.event_id = np.load(tp)["event_id"]

    # NMI over displaced roles: the number with a near-zero init baseline
    ki = np.isin(init.role, ra.DISPLACED_ROLES)
    kt = np.isin(test.role, ra.DISPLACED_ROLES)
    _, nmi_i = ra.mutual_information(init.role[ki], init.topset(layer)[ki])
    _, nmi_t = ra.mutual_information(test.role[kt], test.topset(layer)[kt])
    out.update({"nmi_init": nmi_i, "nmi_trained": nmi_t,
                "nmi_gain": nmi_t - nmi_i})

    neg = _quiet(ra.report_negative_control, test, layer)
    out["jsd_tau_charm_cand"] = (neg or {}).get("jsd_tau_charm")

    ctx = _quiet(ra.report_context_invariance, test, init, layer, role=1,
                 n_perm=n_perm)
    if ctx:
        out.update({"jsd_parent": ctx["jsd_mean"],
                    "jsd_parent_floor": ctx["jsd_null_mean"],
                    "jsd_parent_sigma": ctx["excess_sigma"]})
    xr = _quiet(ra.report_cross_role_jsd, test, layer)
    if xr:
        out["jsd_role"] = xr["jsd_mean"]
    if ctx and xr:
        out["role_over_parent"] = xr["jsd_mean"] / max(ctx["jsd_mean"], 1e-9)

    pv = _quiet(mr.pv_consistency, init, test, layer, {}, n_perm=max(n_perm, 20))
    if pv:
        out.update({"pv_purity": pv["purity_top1"],
                    "pv_purity_init": pv["purity_top1_init"],
                    # top-1 alone is misleading at k=2: a near-even split
                    # across two dedicated experts collapses the argmax purity
                    # while the SET stays clean. Carry the pair number too.
                    "pv_purity_pair": pv["purity_pair"],
                    "pv_exclusivity": pv["exclusivity"],
                    "pv_modal": pv["modal_expert"],
                    "pv_same_everywhere": pv["same_modal_expert_everywhere"]})

    # how many experts are clean specialists
    roles = [r for r in ra.PHYSICS_ROLES if (test.role == r).sum() >= 100]
    if not roles:
        # nothing clears the threshold (tiny run, or a mode that removed the
        # only carrier of a role). Report it as unavailable rather than
        # crashing in np.stack on an empty list.
        out["n_clean_specialists"] = None
        return out
    M = np.stack([test.dist_over_experts(test.role == r, layer, weighted=True)
                  for r in roles])
    n = np.array([(test.role == r).sum() for r in roles], float)
    joint = M * n[:, None]
    per_e = joint.sum(0)
    prior = n / n.sum()
    n_spec = 0
    for e in range(test.n_experts):
        if per_e[e] <= 0:
            continue
        p_role = joint[:, e] / per_e[e]
        j = int(np.argmax(p_role))
        if p_role[j] >= 0.45 and p_role[j] / max(prior[j], 1e-12) >= 2.0:
            n_spec += 1
    out["n_clean_specialists"] = n_spec
    return out


ROWS = [
    ("--- classification ---", None, None),
    ("accuracy", "accuracy", "{:.4f}"),
    ("macro-AUC", "macro_auc", "{:.4f}"),
    ("AUC Bc", "auc_Bc", "{:.4f}"),
    ("AUC B+", "auc_Bu", "{:.4f}"),
    ("AUC background", "auc_bkg", "{:.4f}"),
    ("BDT1 AUC (same events)", "evt_mva1_auc", "{:.4f}"),
    ("best epoch", "best_epoch", "{:.0f}"),
    ("--- specialisation (learned) ---", None, None),
    ("NMI(expert,role) init", "nmi_init", "{:.4f}"),
    ("NMI(expert,role) trained", "nmi_trained", "{:.4f}"),
    ("NMI gain", "nmi_gain", "{:+.4f}"),
    ("clean role specialists", "n_clean_specialists", "{:.0f}"),
    ("--- the hard test ---", None, None),
    ("JSD(tau,charm) on 3pi cands", "jsd_tau_charm_cand", "{:.4f}"),
    ("--- invariance ---", None, None),
    ("cross-parent JSD", "jsd_parent", "{:.4f}"),
    ("  its noise floor", "jsd_parent_floor", "{:.4f}"),
    ("  excess (null sd)", "jsd_parent_sigma", "{:+.1f}"),
    ("cross-role JSD", "jsd_role", "{:.4f}"),
    ("role / parent", "role_over_parent", "{:.1f}x"),
    ("--- primary vertex ---", None, None),
    ("PV purity (init)", "pv_purity_init", "{:.4f}"),
    ("PV purity (trained, top-1)", "pv_purity", "{:.4f}"),
    ("PV purity (trained, top-2 set)", "pv_purity_pair", "{:.4f}"),
    ("PV exclusivity", "pv_exclusivity", "{:.4f}"),
    ("PV expert same everywhere", "pv_same_everywhere", "{}"),
]


def print_table(res, layer):
    w = max(28, *(len(r["run"]) + 2 for r in res))
    print(f"\n{'=' * (30 + w * len(res))}")
    print(f"COMPARISON  (routing quantities at layer {layer})")
    print(f"{'=' * (30 + w * len(res))}")
    print("  " + "".ljust(30) + "".join(r["run"][:w - 2].ljust(w) for r in res))
    print("  " + "masked features".ljust(30)
          + "".join(str(len(r["masked"])).ljust(w) for r in res))
    print("  " + "MoE".ljust(30)
          + "".join(("yes" if r["use_moe"] else "NO (plain FFN)").ljust(w)
                    for r in res))
    print("  " + "EVT node".ljust(30)
          + "".join({"node": "in graph + pooled mean",
                     "pool": "in graph, head readout",
                     "mlp":  "head only, NOT in graph",
                     }.get(r.get("evt_mode", "node"), "?").ljust(w)
                    for r in res))
    for label, key, fmt in ROWS:
        if key is None:
            print("  " + label)
            continue
        cells = []
        for r in res:
            v = r.get(key)
            if v is None:
                cells.append(("n/a" if r["use_moe"] else "-").ljust(w))
            elif isinstance(v, bool):
                cells.append(("YES" if v else "no").ljust(w))
            else:
                cells.append(fmt.format(v).ljust(w))
        print("  " + label.ljust(30) + "".join(cells))


def figure(res, layer, outdir, plt):
    """One figure: what each ablation costs in accuracy, and what it costs in
    interpretability. Two panels so the trade-off is visible at a glance."""
    names = [r["run"] for r in res]
    x = np.arange(len(res))
    fig, axes = plt.subplots(1, 2, figsize=(6 + 1.5 * len(res), 4.4),
                             constrained_layout=True)

    acc = [r.get("macro_auc") or np.nan for r in res]
    a2 = [r.get("accuracy") or np.nan for r in res]
    axes[0].bar(x - 0.2, acc, 0.4, label="macro-AUC", color="#2c6fbb")
    axes[0].bar(x + 0.2, a2, 0.4, label="accuracy", color="#8ab4dd")
    for i, (u, v) in enumerate(zip(acc, a2)):
        if np.isfinite(u):
            axes[0].text(i - 0.2, u + .005, f"{u:.3f}", ha="center", fontsize=8)
        if np.isfinite(v):
            axes[0].text(i + 0.2, v + .005, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_ylim(0.5, 1.03)
    axes[0].set_title("does it still classify?")
    axes[0].legend(frameon=False, fontsize=8)

    jtc = [r.get("jsd_tau_charm_cand") or np.nan for r in res]
    gain = [r.get("nmi_gain") or np.nan for r in res]
    axes[1].bar(x - 0.2, jtc, 0.4, label="JSD(tau, charm) on same objects",
                color="#d1603d")
    axes[1].bar(x + 0.2, gain, 0.4, label="NMI gain over random gate",
                color="#e0a458")
    for i, (u, v) in enumerate(zip(jtc, gain)):
        if np.isfinite(u):
            axes[1].text(i - 0.2, u + .005, f"{u:.3f}", ha="center", fontsize=8)
        if np.isfinite(v):
            axes[1].text(i + 0.2, v + .005, f"{v:+.3f}", ha="center", fontsize=8)
    axes[1].set_title("does it still separate tau from charm?")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, fontsize=8, ha="right")
    fig.suptitle(f"what each ablation costs   (routing at layer {layer})",
                 fontsize=10)
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"compare_layer{layer}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Compare finished runs.")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default="compare")
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    res = []
    for r in args.runs:
        print(f"  reading {r} ...", flush=True)
        res.append(collect(r, args.layer, n_perm=args.n_perm))
    print_table(res, args.layer)

    os.makedirs(args.out, exist_ok=True)
    jp = os.path.join(args.out, f"compare_layer{args.layer}.json")
    with open(jp, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {jp}")
    if not args.no_figures:
        p = figure(res, args.layer, args.out, mr._plt())
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
