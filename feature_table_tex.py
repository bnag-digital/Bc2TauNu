#!/usr/bin/env python3
"""
feature_table_tex.py

Emit the input-variable table as LaTeX, straight from stats.json, so the table
in the writeup cannot drift out of sync with what the model was actually fed.

The applicability column is the direct analogue of the dash in Table 1 of
Genovese et al., where a jet has no lepton charge: a feature is only populated
on the node types where it means something, and is a structural zero elsewhere.

USAGE
    python3 feature_table_tex.py --stats $S > features.tex
    # mark the columns a given run masked
    python3 feature_table_tex.py --stats $S --run runs/n8_k2_s0_masked > features.tex
"""

import argparse
import json
import os

TYPE_NAME = {1: "PV", 2: "SV", 3: r"SV$_{3\pi}$", 4: "EVT"}

BLOCK_CAPTION = {
    "geometry": "Displacement of the vertex from the primary vertex. "
                "Identically zero on the PV by construction.",
    "quality":  "Vertex-fit quality and track multiplicity.",
    "mass":     "Invariant mass of the tracks at the vertex.",
    "impact":   "Impact parameters of the pseudo-track built from the "
                "vertex's summed momentum.",
    "flags":    "Binary indicators; not standardised.",
    "cand3pi":  r"$3\pi$-candidate kinematics. Populated only on vertices the "
                r"reconstruction flagged as candidates.",
    "event":    "Event-level scalars, carried on the single synthetic EVT node.",
}

# short human descriptions; anything absent falls back to the escaped name
DESC = {
    "log1p_d2PV": r"$\log(1+d_{\mathrm{PV}})$, displacement from the PV",
    "log1p_d2PV_significance": r"$\log(1+d_{\mathrm{PV}}/\sigma_d)$, displacement significance",
    "log1p_transverse_radius": r"$\log(1+r_{xy})$, transverse displacement",
    "cos_polar_displacement": r"$\cos\theta$ of the displacement vector",
    "cos_thrust_angle": r"$\cos$ of the angle to the thrust axis",
    "ntrk": "number of tracks fitted to the vertex",
    "log1p_chi2": r"$\log(1+\chi^2)$ of the vertex fit",
    "log1p_chi2_per_ndof": r"$\log(1+\chi^2/\mathrm{ndof})$",
    "log_position_resolution": "log of the combined position uncertainty",
    "log1p_vertex_mass": r"$\log(1+m_{\mathrm{vtx}})$, invariant mass at the vertex",
    "signed_log_DV_d0": r"$\mathrm{sgn}(d_0)\log(1+|d_0|)$ of the pseudo-track",
    "signed_log_DV_z0": r"$\mathrm{sgn}(z_0)\log(1+|z_0|)$ of the pseudo-track",
    "is_PV": "is this the primary vertex",
    "is_signal_hemisphere": "is the node in the minimum-energy hemisphere",
    "is_3pi_candidate": r"was this vertex flagged as a $3\pi$ candidate",
    "is_EVT": "is this the synthetic event-level node",
    "cand_m3pi": r"$m(3\pi)$",
    "cand_m_rho1": r"$m(\pi^+\pi^-)$, first combination",
    "cand_m_rho2": r"$m(\pi^+\pi^-)$, second combination",
    "cand_log1p_p": r"$\log(1+p)$ of the $3\pi$ system",
    "cand_cos_anglethrust": r"$\cos$ of the angle to the thrust axis",
    "cand_signed_log_d0": r"signed $\log$ impact parameter $d_0$",
    "cand_signed_log_z0": r"signed $\log$ impact parameter $z_0$",
    "cand_charge": "total charge of the three pions",
    "cand_log1p_B": r"$\log(1+E_B)$, nominal $B$ energy",
    "cand_log1p_pion_p1": r"$\log(1+p)$ of the leading pion",
    "cand_log1p_pion_p2": r"$\log(1+p)$ of the second pion",
    "cand_log1p_pion_p3": r"$\log(1+p)$ of the third pion",
    "cand_log1p_max_abs_pion_d0": r"$\log(1+\max|d_0|)$ over the three pions",
    "evt_Emin_E": "total energy, minimum-energy hemisphere",
    "evt_Emax_E": "total energy, maximum-energy hemisphere",
    "evt_Emin_Echarged": "charged energy, minimum-energy hemisphere",
    "evt_Emax_Echarged": "charged energy, maximum-energy hemisphere",
    "evt_Emin_Eneutral": "neutral energy, minimum-energy hemisphere",
    "evt_Emax_Eneutral": "neutral energy, maximum-energy hemisphere",
    "evt_Emin_Ncharged": "charged multiplicity, minimum-energy hemisphere",
    "evt_Emax_Ncharged": "charged multiplicity, maximum-energy hemisphere",
    "evt_Emin_Nneutral": "neutral multiplicity, minimum-energy hemisphere",
    "evt_Emax_Nneutral": "neutral multiplicity, maximum-energy hemisphere",
    "evt_NtracksPV": "number of tracks at the primary vertex",
    "evt_NVertex": "number of reconstructed vertices",
    "evt_NTau23Pi": r"number of $3\pi$ candidates in the event",
    "evt_Emin_NDV": "displaced vertices, minimum-energy hemisphere",
    "evt_Emax_NDV": "displaced vertices, maximum-energy hemisphere",
    "evt_thrust_mag": "thrust magnitude",
    "evt_missing_energy_proxy": r"$m_Z/2 - E_{\min}$, missing-energy proxy",
    "evt_hemisphere_imbalance": r"$E_{\max}-E_{\min}$",
    "evt_log1p_dPV2DVmin": "log minimum PV-to-vertex distance",
    "evt_log1p_dPV2DVmax": "log maximum PV-to-vertex distance",
    "evt_log1p_dPV2DVave": "log average PV-to-vertex distance",
}


def esc(s):
    return s.replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--run", default=None,
                    help="mark the features this run masked")
    ap.add_argument("--label", default="tab:features")
    args = ap.parse_args()

    st = json.load(open(args.stats))
    blocks = st["feature_blocks"]
    app = st.get("applicable_types", {})

    masked = set()
    if args.run:
        cp = os.path.join(args.run, "config.json")
        if os.path.exists(cp):
            masked = set(json.load(open(cp)).get("masked_feature_names") or [])

    n_feat = sum(len(v) for v in blocks.values())
    print(r"% generated by feature_table_tex.py -- do not edit by hand")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llc}")
    print(r"\hline")
    print(r"Variable & Description & Applies to \\")
    print(r"\hline")
    for bname, feats in blocks.items():
        cap = BLOCK_CAPTION.get(bname, "")
        print(r"\multicolumn{3}{l}{\textit{%s} --- %s} \\[2pt]"
              % (esc(bname), cap))
        for f in feats:
            types = app.get(f, [])
            tstr = ", ".join(TYPE_NAME.get(int(t), str(t)) for t in types) or "--"
            d = DESC.get(f, "")
            mark = r"$^{\dagger}$" if f in masked else ""
            print(r"\quad \texttt{%s}%s & %s & %s \\" % (esc(f), mark, d, tstr))
        print(r"\hline")
    print(r"\end{tabular}")
    cap = (r"Input variables, %d in total. A variable is populated only on the "
           r"node types listed in the last column and is an exact zero "
           r"elsewhere, so ``not applicable'' is never confused with a "
           r"measured value." % n_feat)
    if masked:
        cap += (r" Variables marked $\dagger$ (%d) are masked in the run "
                r"reported here." % len(masked))
    print(r"\caption{%s}" % cap)
    print(r"\label{%s}" % args.label)
    print(r"\end{table}")


if __name__ == "__main__":
    main()
