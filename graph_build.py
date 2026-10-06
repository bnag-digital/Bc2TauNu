#!/usr/bin/env python3
print("Starting graph_build", flush=True)
"""
graph_build.py

Stage 0 of the MoE-graph-transformer pipeline: turn stage-1 flat ntuples into
padded, fixed-shape per-file tensor shards.

WHAT THIS PRODUCES
    One .pt shard per input ROOT file (restartable, parallelisable, and -- the
    real reason -- it lets dataset.py split train/val/test BY FILE. The Bc and
    Bu signal channels come from only 2-10 production jobs, so a shuffled
    event-level split risks job-level correlation leaking across the split.)

    Each shard is a dict of torch tensors, all aligned on a leading event axis:

        node_feats  (N, 13, 50)  float32  the feature table (see FEATURE_NAMES)
        node_aux    (N, 13,  3)  float32  raw displacement vector from PV --
                                          NOT a model input; dataset.py uses it
                                          to build edge features on the fly
        node_mask   (N, 13)      bool     True = real node (incl. the EVT node)
        node_type   (N, 13)      int8     NODE_* below
        role        (N, 13)      int8     ROLE_* from vertex_truth; -2 = not a
                                          vertex (pad / EVT), -1 = unmatched
        context     (N, 13)      int32    raw grandmother PDG (int32: codes like
                                          100443 overflow int16)
        ctx_cat     (N, 13)      int8     CTX_CAT_* from vertex_truth
        match_chi2  (N, 13)      float32  truth-match quality, for cuts at
        match_iso   (N, 13)      float32    analysis time
        y           (N,)         int8     0 = Bc, 1 = Bu, 2 = background
        channel_id  (N,)         int8     index into meta["channels"]
        evt_mva1    (N,)         float32  BDT1 score. BENCHMARK ONLY -- must
                                          never be used as a model input
        n_vtx       (N,)         int16    reco vertices before truncation

    plus a `meta` dict (feature names, node-type names, channel list, source
    file, cut values, per-file diagnostic counters).

    role / context / ctx_cat are TRUTH. They are carried through so the routing
    analysis can condition on them, and must not reach the model. The
    train/val loop only ever consumes node_feats, node_aux, node_mask,
    node_type and y.

DESIGN NOTES / THINGS DELIBERATELY NOT DONE HERE
    * No node-order permutation. The audit found Vertex_isPV == slot 0 in
      5000/5000 events, which is a total ordering leak, but the fix belongs in
      dataset.py (per-epoch random permutation as augmentation). Keeping the
      build deterministic makes shards reproducible and diffable.
    * No feature standardisation. That needs statistics computed on the TRAIN
      SPLIT ONLY, which this script cannot know about; feature_stats.py does it.
    * No edge features. 13*13*6 float32 = 4 kB/event of fully derivable
      information, which would roughly triple shard size. dataset.py computes
      them in the collate step from node_aux.
    * No absolute vertex position, and no per-node azimuthal phi. e+e- is
      exactly rotationally symmetric about the beam, so per-node phi is noise;
      RELATIVE phi is physical and enters only via the edge features built
      downstream from node_aux.

USAGE
    python3 graph_build.py --channel Bc2TauNu   --out shards/
    python3 graph_build.py --channel all        --out shards/ --max_events_per_file 50000
    python3 graph_build.py --channel Zbb_incl   --out shards/ --production analysis

    Diagnostics only, no writing:
        python3 graph_build.py --channel Bc2TauNu --dry_run --max_events_per_file 2000
"""

import sys
#sys.path.insert(0, "/eos/user/b/bnag/python_packages")

import argparse
import glob
import json
import os
import time
from collections import Counter

import numpy as np
import torch
torch.cuda.is_available = lambda: False
import uproot

import vertex_truth as vt


# --------------------------------------------------------------------------
# sample layout
# --------------------------------------------------------------------------
PROD_BASE = ("/eos/experiment/fcc/ee/analyses/case-studies/flavour/Bc2TauNu/"
             "flatNtuples/spring2021/prod_03")
# /eos/experiment/fcc/ee/analyses/case-studies/flavour/Bc2TauNu/flatNtuples/spring2021/prod_03
PRODUCTIONS = {
    "training": f"{PROD_BASE}/Batch_Training_4stage1",   # loose  (EVT_MVA1 > -1)
    "analysis": f"{PROD_BASE}/Batch_Analysis_stage1",    # tight  (EVT_MVA1 > 0.6)
}

# channel -> (subdirectory, class label y)
#   y: 0 = Bc signal, 1 = Bu signal, 2 = background
#
# The four 3pi hadronic modes are background ON PURPOSE. The candidate feature
# block (features 16-28) is zero-filled on non-candidate vertices, so
# "candidate block populated" is trivially detectable whether or not the
# is3piCand flag is present -- deleting the flag would not remove the
# shortcut. These samples supply vertices with a fully populated candidate
# block whose role is NOT tau in ~99% of cases (measured: Bd2D3Pi 0.9%,
# Lb2Lc3Pi 1.3%), which decorrelates "has candidate features" from "is a tau"
# and forces the router onto the actual kinematics (mass ceiling at m_tau, the
# a1(1260) -> rho pi substructure) instead of mere presence.
CHANNELS = {
    "Bc2TauNu":   ("p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU", 0),
    "Bu2TauNu":   ("p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU", 1),
    "Zbb_incl":   ("p8_ee_Zbb_ecm91",                         2),
    "Zcc_incl":   ("p8_ee_Zcc_ecm91",                         2),
    "Bd2D3Pi":    ("p8_ee_Zbb_ecm91_EvtGen_Bd2D3Pi",          2),
    "Bu2D03Pi":   ("p8_ee_Zbb_ecm91_EvtGen_Bu2D03Pi",         2),
    "Bs2Ds3Pi":   ("p8_ee_Zbb_ecm91_EvtGen_Bs2Ds3Pi",         2),
    "Lb2Lc3Pi":   ("p8_ee_Zbb_ecm91_EvtGen_Lb2Lc3Pi",         2),
}
CHANNEL_ORDER = list(CHANNELS.keys())

# Zuds is deliberately absent: the entire Batch_Analysis_stage1 production holds
# 39,760 events (0.24% of Zbb). It cannot be usefully weighted -- either it is
# negligible, or upweighting it means memorising those few events. It is also
# rejected at the ~1e9 level by BDT1, so excluding it costs nothing physical.

# Exclusive tau modes are NOT built here. They are held out entirely as probes,
# so that context invariance is a generalisation claim about contexts never
# seen in training. Build them separately when the probe stage is reached:
#   Bd2DTauNu Bd2DstTauNu Bs2DsTauNu Bs2DsstTauNu
#   Bu2D0TauNu Bu2Dst0TauNu Lb2LcTauNu Lb2LcstTauNu
PROBE_CHANNELS = {
    "Bd2DTauNu":    "p8_ee_Zbb_ecm91_EvtGen_Bd2DTauNu",
    "Bd2DstTauNu":  "p8_ee_Zbb_ecm91_EvtGen_Bd2DstTauNu",
    "Bs2DsTauNu":   "p8_ee_Zbb_ecm91_EvtGen_Bs2DsTauNu",
    "Bs2DsstTauNu": "p8_ee_Zbb_ecm91_EvtGen_Bs2DsstTauNu",
    "Bu2D0TauNu":   "p8_ee_Zbb_ecm91_EvtGen_Bu2D0TauNu",
    "Bu2Dst0TauNu": "p8_ee_Zbb_ecm91_EvtGen_Bu2Dst0TauNu",
    "Lb2LcTauNu":   "p8_ee_Zbb_ecm91_EvtGen_Lb2LcTauNu",
    "Lb2LcstTauNu": "p8_ee_Zbb_ecm91_EvtGen_Lb2LcstTauNu",
    "Bd2Dst3Pi":    "p8_ee_Zbb_ecm91_EvtGen_Bd2Dst3Pi",
    "Bs2Dsst3Pi":   "p8_ee_Zbb_ecm91_EvtGen_Bs2Dsst3Pi",
    "Bu2Dst03Pi":   "p8_ee_Zbb_ecm91_EvtGen_Bu2Dst03Pi",
    "Lb2Lcst3Pi":   "p8_ee_Zbb_ecm91_EvtGen_Lb2Lcst3Pi",
    # held out of training by design (see the Zuds note below CHANNEL_ORDER):
    # too few events to weight, but that makes it a clean long-lifetime probe
    "Zuds_incl":    "p8_ee_Zuds_ecm91",

}


# --------------------------------------------------------------------------
# graph geometry
# --------------------------------------------------------------------------
# Measured Vertex_n max = 11 (both Bc signal and inclusive Zbb, q99 = 7-8), so
# 11 vertices + 1 EVT node = 12 needed; 13 leaves one slot of headroom.
MAX_NODES = 13
MAX_VERTICES = MAX_NODES - 1

NODE_PAD, NODE_PV, NODE_SV, NODE_SV3PI, NODE_EVT = 0, 1, 2, 3, 4
NODE_TYPE_NAMES = {
    NODE_PAD: "pad", NODE_PV: "PV", NODE_SV: "SV",
    NODE_SV3PI: "SV3pi", NODE_EVT: "EVT",
}

# Sentinel distinct from vertex_truth.ROLE_UNMATCHED (-1). A pad slot or the
# EVT node is not a vertex at all, and must not be counted in the "unmatched
# vertex" population during analysis.
ROLE_NOT_A_VERTEX = -2

M_Z = 91.1876  # GeV, for the missing-energy proxy


# --------------------------------------------------------------------------
# feature table
# --------------------------------------------------------------------------
# Built as named blocks so the layout is self-documenting and the total is
# asserted rather than hand-counted.
FEAT_BLOCKS = [
    ("geometry", [                       # SV / SV3pi (degenerate on PV, see below)
        "log1p_d2PV",
        "log1p_d2PV_significance",
        "log1p_transverse_radius",
        "cos_polar_displacement",
        "cos_thrust_angle",              # Vertex_thrust_angle: confirmed already
    ]),                                  # a cosine, range [-1,1] -- see note below
    ("quality", [                        # every real vertex
        "ntrk",
        "log1p_chi2",
        "log1p_chi2_per_ndof",
        "log_position_resolution",
    ]),
    ("mass", [
        "log1p_vertex_mass",
    ]),
    ("impact", [                         # SV / SV3pi
        "signed_log_DV_d0",              # pseudotrack impact parameters; see
        "signed_log_DV_z0",              # the DV_REMAP note below
    ]),
    ("flags", [                          # every node
        "is_PV",
        "is_signal_hemisphere",
        "is_3pi_candidate",
        "is_EVT",
    ]),
    ("cand3pi", [                        # SV3pi only, else exactly 0
        "cand_m3pi",
        "cand_m_rho1",
        "cand_m_rho2",
        "cand_log1p_p",
        "cand_cos_anglethrust",           # Tau23PiCandidates_anglethrust: confirmed
                                          # a RAW angle in radians [0, pi] -- cos()
                                          # applied here, see note below
        "cand_signed_log_d0",             # same heavy-tail treatment as the
        "cand_signed_log_z0",             # pseudotrack impact parameters
        "cand_charge",
        "cand_log1p_B",                    # confirmed an energy scale (14.6-86.3
                                          # GeV, bracketing E_beam=45.6 GeV), not
                                          # a mass or discriminant -- see note below
        "cand_log1p_pion_p1",             # pion momenta sorted DESCENDING, so the
        "cand_log1p_pion_p2",             # feature is permutation-invariant w.r.t.
        "cand_log1p_pion_p3",             # the arbitrary pion1/2/3 ordering
        "cand_log1p_max_abs_pion_d0",
    ]),
    ("event", [                          # EVT node only, else exactly 0
        "evt_Emin_E", "evt_Emax_E",
        "evt_Emin_Echarged", "evt_Emax_Echarged",
        "evt_Emin_Eneutral", "evt_Emax_Eneutral",
        "evt_Emin_Ncharged", "evt_Emax_Ncharged",
        "evt_Emin_Nneutral", "evt_Emax_Nneutral",
        "evt_NtracksPV", "evt_NVertex", "evt_NTau23Pi",
        "evt_Emin_NDV", "evt_Emax_NDV",
        "evt_thrust_mag",
        "evt_missing_energy_proxy",       # m_Z/2 - Emin_E
        "evt_hemisphere_imbalance",       # Emax_E - Emin_E
        "evt_log1p_dPV2DVmin",
        "evt_log1p_dPV2DVmax",
        "evt_log1p_dPV2DVave",
    ]),
]
FEATURE_NAMES = [n for _, names in FEAT_BLOCKS for n in names]
N_FEATS = len(FEATURE_NAMES)
FIDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
assert N_FEATS == 50, f"feature table drifted: {N_FEATS} != 50"

# THRUST_ANGLE / CAND_B: resolved by measurement (raw_ranges dump on a real
# 25000-event shard, Bc2TauNuTAUHADNU):
#   Vertex_thrust_angle          range [-1.0000, 1.0000]  -> already a cosine.
#       Stored as-is (cos_thrust_angle); applying cos() again would be wrong.
#   Tau23PiCandidates_anglethrust range [0.0009, 3.1404]  -> a raw angle in
#       radians (0..pi). cos() is applied at build time (cand_cos_anglethrust)
#       so it uses the same convention as the vertex-level feature and so that
#       "aligned with thrust axis" is a large value rather than wrapping through
#       pi. These two branches use DIFFERENT conventions despite similar names.
#   Tau23PiCandidates_B           range [14.6, 86.3] GeV -> too high for a B
#       mass (~5.3 GeV); consistent with a nominal reconstructed B-hadron
#       energy at the Z pole (E_beam = 45.6 GeV, widened by B->tau nu momentum
#       smearing). Treated as an energy scale (log1p), not raw.

# Guards
EPS = 1e-12
ERR_FLOOR = 1e-6      # mm, on quoted position errors before dividing/logging
DIST_FLOOR = 1e-9     # mm, before dividing by a displacement magnitude

VERTEX_BRANCHES = [
    "Vertex_x", "Vertex_y", "Vertex_z",
    "Vertex_xErr", "Vertex_yErr", "Vertex_zErr",
    "Vertex_isPV", "Vertex_ntrk", "Vertex_chi2", "Vertex_mass",
    "Vertex_thrust_angle", "Vertex_thrusthemis_emin",
    "Vertex_d2PV", "Vertex_d2PVx", "Vertex_d2PVy", "Vertex_d2PVz",
    "Vertex_d2PVErr",
    "DV_d0", "DV_z0",
]
# Pseudotrack collection: one entry per NON-PV vertex, not one per vertex.
# See the DV_REMAP note in build_event().
DV_BRANCHES = ("DV_d0", "DV_z0")
CAND_BRANCHES = [
    "Tau23PiCandidates_vertex", "Tau23PiCandidates_mass",
    "Tau23PiCandidates_rho1mass", "Tau23PiCandidates_rho2mass",
    "Tau23PiCandidates_p", "Tau23PiCandidates_anglethrust",
    "Tau23PiCandidates_d0", "Tau23PiCandidates_z0",
    "Tau23PiCandidates_q", "Tau23PiCandidates_B",
    "Tau23PiCandidates_pion1p", "Tau23PiCandidates_pion2p",
    "Tau23PiCandidates_pion3p",
    "Tau23PiCandidates_pion1d0", "Tau23PiCandidates_pion2d0",
    "Tau23PiCandidates_pion3d0",
]
EVENT_BRANCHES = [
    "EVT_ThrustEmin_E", "EVT_ThrustEmax_E",
    "EVT_ThrustEmin_Echarged", "EVT_ThrustEmax_Echarged",
    "EVT_ThrustEmin_Eneutral", "EVT_ThrustEmax_Eneutral",
    "EVT_ThrustEmin_Ncharged", "EVT_ThrustEmax_Ncharged",
    "EVT_ThrustEmin_Nneutral", "EVT_ThrustEmax_Nneutral",
    "EVT_NtracksPV", "EVT_NVertex", "EVT_NTau23Pi",
    "EVT_ThrustEmin_NDV", "EVT_ThrustEmax_NDV",
    "EVT_Thrust_Mag",
    "EVT_dPV2DVmin", "EVT_dPV2DVmax", "EVT_dPV2DVave",
    "EVT_MVA1",
]
# vt.BRANCHES supplies everything the truth matcher needs (MC_Vertex_*, and the
# reco position/error/ntrk columns it matches against).
ALL_BRANCHES = sorted(set(VERTEX_BRANCHES + CAND_BRANCHES + EVENT_BRANCHES
                          + list(vt.BRANCHES)))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _log1p_nonneg(x, counter=None, key=None):
    """log1p of a quantity that should be >= 0; clips negatives and counts them.

    Silent clipping is how sign-convention surprises turn into unexplainable
    training behaviour three weeks later, so it is counted instead.
    """
    x = np.asarray(x, dtype=np.float64)
    neg = x < 0
    if counter is not None and neg.any():
        counter[key] += int(neg.sum())
    return np.log1p(np.clip(x, 0.0, None))


def _signed_log1p(x):
    """sign(x) * log1p(|x|), for signed quantities with heavy tails.

    Needed for the impact parameters: DV_d0 was measured with median |d0| =
    0.045 mm but max |d0| = 92 mm, i.e. 3.5 decades of dynamic range. Fed raw,
    the tail would dominate the standardisation in feature_stats.py. The
    transform is monotone and sign-preserving, so no information is lost.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def _safe_div(num, den, floor):
    den = np.asarray(den, dtype=np.float64)
    den = np.where(np.abs(den) < floor, np.sign(den) * floor + (den == 0) * floor, den)
    return np.asarray(num, dtype=np.float64) / den


def _row(arrays, key, i):
    return np.asarray(arrays[key][i])


def _scalar(arrays, key, i):
    v = arrays[key][i]
    return float(np.asarray(v).reshape(-1)[0]) if np.ndim(v) else float(v)


# --------------------------------------------------------------------------
# per-event graph construction
# --------------------------------------------------------------------------
def build_event(arrays, i, labels, diag):
    """Build the node tensors for one event.

    Returns None if the event is unusable (no vertices, or no PV), else a dict
    of arrays with leading axis MAX_NODES.

    `labels` is the vertex_truth.label_events() output for this chunk; `diag`
    is a Counter for diagnostics.
    """
    n_vtx_raw = len(_row(arrays, "Vertex_x", i))
    if n_vtx_raw == 0:
        diag["skip_no_vertices"] += 1
        return None

    # ---- length consistency across per-vertex branches --------------------
    # DV_d0 / DV_z0 are excluded here and handled separately below: the dry run
    # showed len(DV_d0) != Vertex_n on 100% of events.
    vtx = {}
    for b in VERTEX_BRANCHES:
        if b in DV_BRANCHES:
            continue
        v = _row(arrays, b, i)
        if len(v) != n_vtx_raw:
            diag[f"len_mismatch_{b}"] += 1
            pad = np.zeros(n_vtx_raw, dtype=np.float64)
            pad[:min(len(v), n_vtx_raw)] = np.asarray(
                v, dtype=np.float64)[:min(len(v), n_vtx_raw)]
            v = pad
        vtx[b] = np.asarray(v, dtype=np.float64)

    # ---- DV_REMAP: pseudotrack impact parameters -------------------------
    # myUtils::get_pseudotrack builds one pseudo-track per vertex from the
    # summed momentum of its constituents. A pseudotrack for the PV is
    # meaningless (the PV *is* the reference), so the collection holds one
    # entry per NON-PV vertex, in original vertex order. Measured on both
    # Bc2TauNu and inclusive Zbb: len(DV_d0) == n_nonPV on 100% of events, and
    # |DV_d0| correlates with the d2PV of the non-PV vertices taken in order at
    # r = 0.12 / 0.19 (12 / 19 sigma for n ~ 1e4, versus 0 +- 0.01 expected
    # under a scrambled ordering).
    #
    # The correlation is weak by physics, not by misalignment: a pseudotrack d0
    # measures the NON-COLLINEARITY between the vertex displacement and the
    # reconstructed momentum sum (roughly d2PV * sin(theta) between them). For a
    # fully reconstructed decay the momentum sum points back at the PV and
    # d0 -> 0 no matter how displaced the vertex is. That is what makes this
    # feature valuable rather than redundant with d2PV: large d0 means momentum
    # is MISSING from the vertex, which is precisely the difference between
    # tau -> 3pi nu (escaping neutrino, non-collinear) and D -> 3pi (fully
    # reconstructed, collinear).
    #
    # Anything other than the two known length conventions is zeroed and
    # counted. Silently misaligned impact parameters would be worse than absent
    # ones, because they would still look plausible.
    nonpv_idx = np.flatnonzero(vtx["Vertex_isPV"].astype(np.int64) != 1)
    for b in DV_BRANCHES:
        raw = np.asarray(_row(arrays, b, i), dtype=np.float64)
        full = np.zeros(n_vtx_raw, dtype=np.float64)
        if len(raw) == n_vtx_raw:
            full = raw
            diag[f"{b}_per_vertex"] += 1
        elif len(raw) == len(nonpv_idx):
            full[nonpv_idx] = raw
            diag[f"{b}_per_nonPV_vertex"] += 1
        else:
            diag[f"{b}_unresolved_length"] += 1
        vtx[b] = full

    # ---- locate the PV ---------------------------------------------------
    # Never trust slot 0: the audit found isPV == slot 0 in 5000/5000 events,
    # which is exactly the kind of regularity that should be read from the flag
    # rather than assumed from position.
    pv_where = np.flatnonzero(vtx["Vertex_isPV"].astype(np.int64) == 1)
    if len(pv_where) == 0:
        diag["skip_no_PV"] += 1
        return None
    if len(pv_where) > 1:
        diag["multiple_PV_took_first"] += 1
    pv_idx = int(pv_where[0])

    # ---- 3pi candidate -> reco vertex map --------------------------------
    # Several candidates can point at the same reco vertex. The chosen one is
    # the HIGHEST-MOMENTUM candidate: deterministic, truth-free, and (unlike
    # e.g. "closest to m_tau") it encodes no prior about tau-ness, which would
    # otherwise smuggle the answer into the inputs of a study about whether the
    # model discovers tau vertices on its own.
    cand_for_vtx = {}
    if "Tau23PiCandidates_vertex" in arrays:
        c_vtx = _row(arrays, "Tau23PiCandidates_vertex", i).astype(np.int64)
        c_p = _row(arrays, "Tau23PiCandidates_p", i).astype(np.float64)
        n_cand = len(c_vtx)
        if len(c_p) != n_cand:
            diag["len_mismatch_cand_p"] += 1
            n_cand = min(n_cand, len(c_p))
        for c in range(n_cand):
            rv = int(c_vtx[c])
            if not (0 <= rv < n_vtx_raw):
                diag["cand_vertex_out_of_range"] += 1
                continue
            if rv not in cand_for_vtx or c_p[c] > c_p[cand_for_vtx[rv]]:
                if rv in cand_for_vtx:
                    diag["multiple_cand_same_vertex"] += 1
                cand_for_vtx[rv] = c

    # ---- choose which vertices survive truncation ------------------------
    # PV always kept, then remaining vertices in file order. Truncation is not
    # expected to fire (measured max 11 vertices <= MAX_VERTICES = 12) but is
    # counted so a silent change in the input would be visible.
    order = [pv_idx] + [v for v in range(n_vtx_raw) if v != pv_idx]
    if len(order) > MAX_VERTICES:
        diag["truncated_events"] += 1
        diag["truncated_vertices"] += len(order) - MAX_VERTICES
        order = order[:MAX_VERTICES]

    # ---- allocate --------------------------------------------------------
    feats = np.zeros((MAX_NODES, N_FEATS), dtype=np.float64)
    aux = np.zeros((MAX_NODES, 3), dtype=np.float64)
    mask = np.zeros(MAX_NODES, dtype=bool)
    ntype = np.full(MAX_NODES, NODE_PAD, dtype=np.int64)
    role = np.full(MAX_NODES, ROLE_NOT_A_VERTEX, dtype=np.int64)
    ctx = np.zeros(MAX_NODES, dtype=np.int64)
    ctx_cat = np.full(MAX_NODES, vt.CTX_CAT_NONE, dtype=np.int64)
    m_chi2 = np.zeros(MAX_NODES, dtype=np.float64)
    m_iso = np.zeros(MAX_NODES, dtype=np.float64)

    ev_role = labels["role"][i]
    ev_ctx = labels["context"][i]
    ev_cat = labels["ctx_cat"][i]
    ev_chi2 = labels["chi2"][i]
    ev_iso = labels["iso"][i]

    # ---- vertex nodes ----------------------------------------------------
    for slot, v in enumerate(order):
        is_pv = (v == pv_idx)
        is_cand = v in cand_for_vtx

        mask[slot] = True
        ntype[slot] = NODE_PV if is_pv else (NODE_SV3PI if is_cand else NODE_SV)

        # truth passthrough (labels arrays are indexed by ORIGINAL vertex index)
        if v < len(ev_role):
            role[slot] = int(ev_role[v])
            ctx[slot] = int(ev_ctx[v])
            ctx_cat[slot] = int(ev_cat[v])
            m_chi2[slot] = float(ev_chi2[v]) if np.isfinite(ev_chi2[v]) else -1.0
            m_iso[slot] = float(ev_iso[v])
        else:
            diag["truth_index_out_of_range"] += 1

        d2pv = vtx["Vertex_d2PV"][v]
        dx, dy, dz = (vtx["Vertex_d2PVx"][v], vtx["Vertex_d2PVy"][v],
                      vtx["Vertex_d2PVz"][v])
        aux[slot] = (dx, dy, dz)

        # -- geometry. Degenerate on the PV: d2PV == 0 identically, so
        #    magnitude / significance / transverse radius / polar angle carry no
        #    information and are left at zero (the is_PV flag is what tells the
        #    model this row is different). Direct analogue of the MGT paper's
        #    lepton row having no mass column.
        if not is_pv:
            feats[slot, FIDX["log1p_d2PV"]] = _log1p_nonneg(d2pv, diag, "neg_d2PV")
            feats[slot, FIDX["log1p_d2PV_significance"]] = _log1p_nonneg(
                _safe_div(d2pv, vtx["Vertex_d2PVErr"][v], ERR_FLOOR))
            feats[slot, FIDX["log1p_transverse_radius"]] = _log1p_nonneg(
                np.hypot(dx, dy))
            feats[slot, FIDX["cos_polar_displacement"]] = _safe_div(
                dz, max(abs(d2pv), DIST_FLOOR), DIST_FLOOR)

        # thrust angle is meaningful for the PV too; confirmed already a cosine
        feats[slot, FIDX["cos_thrust_angle"]] = vtx["Vertex_thrust_angle"][v]

        # -- quality
        ntrk = vtx["Vertex_ntrk"][v]
        feats[slot, FIDX["ntrk"]] = ntrk
        chi2 = vtx["Vertex_chi2"][v]
        feats[slot, FIDX["log1p_chi2"]] = _log1p_nonneg(chi2, diag, "neg_chi2")
        ndof = max(2.0 * ntrk - 3.0, 1.0)
        feats[slot, FIDX["log1p_chi2_per_ndof"]] = _log1p_nonneg(chi2 / ndof)
        res = np.sqrt(vtx["Vertex_xErr"][v] ** 2 + vtx["Vertex_yErr"][v] ** 2
                      + vtx["Vertex_zErr"][v] ** 2)
        feats[slot, FIDX["log_position_resolution"]] = np.log(max(res, ERR_FLOOR))

        # -- mass
        feats[slot, FIDX["log1p_vertex_mass"]] = _log1p_nonneg(
            vtx["Vertex_mass"][v], diag, "neg_vertex_mass")

        # -- impact parameters (pseudotrack; not meaningful for the PV)
        if not is_pv:
            feats[slot, FIDX["signed_log_DV_d0"]] = _signed_log1p(vtx["DV_d0"][v])
            feats[slot, FIDX["signed_log_DV_z0"]] = _signed_log1p(vtx["DV_z0"][v])

        # -- flags
        feats[slot, FIDX["is_PV"]] = 1.0 if is_pv else 0.0
        feats[slot, FIDX["is_signal_hemisphere"]] = float(
            vtx["Vertex_thrusthemis_emin"][v])
        feats[slot, FIDX["is_3pi_candidate"]] = 1.0 if is_cand else 0.0
        # is_EVT stays 0

        # -- 3pi candidate block
        if is_cand:
            c = cand_for_vtx[v]
            g = lambda b: float(_row(arrays, b, i)[c])  # noqa: E731
            feats[slot, FIDX["cand_m3pi"]] = g("Tau23PiCandidates_mass")
            feats[slot, FIDX["cand_m_rho1"]] = g("Tau23PiCandidates_rho1mass")
            feats[slot, FIDX["cand_m_rho2"]] = g("Tau23PiCandidates_rho2mass")
            feats[slot, FIDX["cand_log1p_p"]] = _log1p_nonneg(
                g("Tau23PiCandidates_p"))
            feats[slot, FIDX["cand_cos_anglethrust"]] = np.cos(
                g("Tau23PiCandidates_anglethrust"))
            feats[slot, FIDX["cand_signed_log_d0"]] = _signed_log1p(
                g("Tau23PiCandidates_d0"))
            feats[slot, FIDX["cand_signed_log_z0"]] = _signed_log1p(
                g("Tau23PiCandidates_z0"))
            feats[slot, FIDX["cand_charge"]] = g("Tau23PiCandidates_q")
            feats[slot, FIDX["cand_log1p_B"]] = _log1p_nonneg(
                g("Tau23PiCandidates_B"), diag, "neg_cand_B")

            # pion momenta sorted descending: pion1/2/3 ordering in the ntuple
            # is an implementation detail, and feeding it raw would make the
            # feature vector depend on it
            pions = np.sort(np.array([g("Tau23PiCandidates_pion1p"),
                                      g("Tau23PiCandidates_pion2p"),
                                      g("Tau23PiCandidates_pion3p")]))[::-1]
            for k, nm in enumerate(["cand_log1p_pion_p1", "cand_log1p_pion_p2",
                                    "cand_log1p_pion_p3"]):
                feats[slot, FIDX[nm]] = _log1p_nonneg(pions[k])
            feats[slot, FIDX["cand_log1p_max_abs_pion_d0"]] = _log1p_nonneg(max(
                abs(g("Tau23PiCandidates_pion1d0")),
                abs(g("Tau23PiCandidates_pion2d0")),
                abs(g("Tau23PiCandidates_pion3d0"))))

    # ---- EVT global node -------------------------------------------------
    # The analogue of the MGT paper's `energy` node: not a reconstructed object,
    # but an inferred event-level quantity. Loaded with BDT1's feature set so
    # the model competes with BDT1+BDT2 combined rather than BDT2 alone.
    ev_slot = len(order)
    mask[ev_slot] = True
    ntype[ev_slot] = NODE_EVT
    feats[ev_slot, FIDX["is_EVT"]] = 1.0

    emin_e = _scalar(arrays, "EVT_ThrustEmin_E", i)
    emax_e = _scalar(arrays, "EVT_ThrustEmax_E", i)
    direct = [
        ("evt_Emin_E", "EVT_ThrustEmin_E"), ("evt_Emax_E", "EVT_ThrustEmax_E"),
        ("evt_Emin_Echarged", "EVT_ThrustEmin_Echarged"),
        ("evt_Emax_Echarged", "EVT_ThrustEmax_Echarged"),
        ("evt_Emin_Eneutral", "EVT_ThrustEmin_Eneutral"),
        ("evt_Emax_Eneutral", "EVT_ThrustEmax_Eneutral"),
        ("evt_Emin_Ncharged", "EVT_ThrustEmin_Ncharged"),
        ("evt_Emax_Ncharged", "EVT_ThrustEmax_Ncharged"),
        ("evt_Emin_Nneutral", "EVT_ThrustEmin_Nneutral"),
        ("evt_Emax_Nneutral", "EVT_ThrustEmax_Nneutral"),
        ("evt_NtracksPV", "EVT_NtracksPV"), ("evt_NVertex", "EVT_NVertex"),
        ("evt_NTau23Pi", "EVT_NTau23Pi"),
        ("evt_Emin_NDV", "EVT_ThrustEmin_NDV"),
        ("evt_Emax_NDV", "EVT_ThrustEmax_NDV"),
        ("evt_thrust_mag", "EVT_Thrust_Mag"),
    ]
    for fname, bname in direct:
        feats[ev_slot, FIDX[fname]] = _scalar(arrays, bname, i)
    feats[ev_slot, FIDX["evt_missing_energy_proxy"]] = 0.5 * M_Z - emin_e
    feats[ev_slot, FIDX["evt_hemisphere_imbalance"]] = emax_e - emin_e
    for fname, bname in [("evt_log1p_dPV2DVmin", "EVT_dPV2DVmin"),
                         ("evt_log1p_dPV2DVmax", "EVT_dPV2DVmax"),
                         ("evt_log1p_dPV2DVave", "EVT_dPV2DVave")]:
        feats[ev_slot, FIDX[fname]] = _log1p_nonneg(_scalar(arrays, bname, i))

    # ---- final sanity: no NaN/inf may reach the shard --------------------
    bad = ~np.isfinite(feats)
    if bad.any():
        for fi in np.flatnonzero(bad.any(axis=0)):
            diag[f"nonfinite_{FEATURE_NAMES[fi]}"] += int(bad[:, fi].sum())
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    bad_aux = ~np.isfinite(aux)
    if bad_aux.any():
        diag["nonfinite_aux"] += int(bad_aux.sum())
        aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)

    # Padding must be exactly zero: attention masks it out, but a nonzero pad
    # row would still corrupt feature_stats.py and any pooled baseline.
    assert not feats[~mask].any(), "padding slots are not zero"

    return dict(feats=feats, aux=aux, mask=mask, ntype=ntype, role=role,
                ctx=ctx, ctx_cat=ctx_cat, m_chi2=m_chi2, m_iso=m_iso,
                n_vtx=n_vtx_raw)


# --------------------------------------------------------------------------
# file processing
# --------------------------------------------------------------------------
def process_file(path, y, channel_id, max_events=None, step=20000, diag=None):
    """Stream one ROOT file in chunks and build its shard tensors."""
    diag = Counter() if diag is None else diag
    tree = uproot.open(path)["events"]

    missing = [b for b in ALL_BRANCHES if b not in tree.keys()]
    if missing:
        raise KeyError(f"{path}: missing branches {missing}")

    out = {k: [] for k in ("feats", "aux", "mask", "ntype", "role", "ctx",
                           "ctx_cat", "m_chi2", "m_iso", "n_vtx", "mva1")}

    n_done = 0
    start = 0
    while True:
        if max_events is not None and n_done >= max_events:
            break
        stop = start + step
        if max_events is not None:
            stop = min(stop, max_events)
        arrays = tree.arrays(ALL_BRANCHES, entry_start=start, entry_stop=stop,
                             library="np")
        n_chunk = len(arrays["Vertex_x"])
        if n_chunk == 0:
            break

        labels = vt.label_events(arrays, n_chunk)

        for i in range(n_chunk):
            ev = build_event(arrays, i, labels, diag)
            if ev is None:
                continue
            for k in ("feats", "aux", "mask", "ntype", "role", "ctx",
                      "ctx_cat", "m_chi2", "m_iso"):
                out[k].append(ev[k])
            out["n_vtx"].append(ev["n_vtx"])
            out["mva1"].append(_scalar(arrays, "EVT_MVA1", i))

            n_done += 1
            diag["events_built"] += 1

        start = stop
        if n_chunk < step:
            break

    # `ranges` is kept only for a stable 3-tuple return signature; both
    # features it used to track (thrust angle convention, cand_B units) are
    # now resolved and transformed at build time (see the THRUST_ANGLE /
    # CAND_B note above), so there is nothing left to accumulate.
    ranges = {}

    if not out["feats"]:
        return None, diag, ranges

    n = len(out["feats"])
    shard = {
        "node_feats": torch.from_numpy(np.stack(out["feats"]).astype(np.float32)),
        "node_aux":   torch.from_numpy(np.stack(out["aux"]).astype(np.float32)),
        "node_mask":  torch.from_numpy(np.stack(out["mask"])),
        "node_type":  torch.from_numpy(np.stack(out["ntype"]).astype(np.int8)),
        "role":       torch.from_numpy(np.stack(out["role"]).astype(np.int8)),
        # int32, not int16: PDG codes such as 100443 overflow int16
        "context":    torch.from_numpy(np.stack(out["ctx"]).astype(np.int32)),
        "ctx_cat":    torch.from_numpy(np.stack(out["ctx_cat"]).astype(np.int8)),
        "match_chi2": torch.from_numpy(np.stack(out["m_chi2"]).astype(np.float32)),
        "match_iso":  torch.from_numpy(np.stack(out["m_iso"]).astype(np.float32)),
        "y":          torch.full((n,), int(y), dtype=torch.int8),
        "channel_id": torch.full((n,), int(channel_id), dtype=torch.int8),
        "evt_mva1":   torch.tensor(out["mva1"], dtype=torch.float32),
        "n_vtx":      torch.tensor(out["n_vtx"], dtype=torch.int16),
    }
    return shard, diag, ranges


def main():
    ap = argparse.ArgumentParser(
        description="Build padded graph tensor shards from stage-1 ntuples.")
    ap.add_argument("--channel", required=True,
                    help="channel name, 'all', or 'probe:<name>' for a held-out "
                         f"exclusive tau mode. Known: {', '.join(CHANNEL_ORDER)}")
    ap.add_argument("--out", default="shards", help="output directory")
    ap.add_argument("--production", default="analysis",
                    choices=sorted(PRODUCTIONS),
                    help="'analysis' = tight (EVT_MVA1>0.6), 'training' = loose")
    ap.add_argument("--max_events_per_file", type=int, default=None)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--step", type=int, default=20000,
                    help="events per uproot read (memory/speed tradeoff)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild shards that already exist")
    ap.add_argument("--dry_run", action="store_true",
                    help="build and report diagnostics, write nothing")
    args = ap.parse_args()

    base = PRODUCTIONS[args.production]

    if args.channel == "all":
        todo = [(c, CHANNELS[c][0], CHANNELS[c][1]) for c in CHANNEL_ORDER]
    elif args.channel.startswith("probe:"):
        nm = args.channel.split(":", 1)[1]
        if nm not in PROBE_CHANNELS:
            ap.error(f"unknown probe channel {nm}; "
                     f"known: {', '.join(sorted(PROBE_CHANNELS))}")
        # Probes are never trained on, so their y is meaningless. -1 makes any
        # accidental use in a loss loudly wrong rather than quietly plausible.
        todo = [(nm, PROBE_CHANNELS[nm], -1)]
    elif args.channel in CHANNELS:
        todo = [(args.channel, CHANNELS[args.channel][0],
                 CHANNELS[args.channel][1])]
    else:
        ap.error(f"unknown channel {args.channel}")

    grand = Counter()
    for cname, subdir, y in todo:
        files = sorted(glob.glob(os.path.join(base, subdir, "*.root")))
        if args.max_files:
            files = files[:args.max_files]
        if not files:
            print(f"!! no files for {cname} at {base}/{subdir}", flush=True)
            continue

        cid = CHANNEL_ORDER.index(cname) if cname in CHANNEL_ORDER else -1
        odir = os.path.join(args.out, cname)
        if not args.dry_run:
            os.makedirs(odir, exist_ok=True)

        print(f"\n=== {cname}  (y={y}, channel_id={cid}, {len(files)} files, "
              f"production={args.production}) ===", flush=True)

        for fi, f in enumerate(files):
            tag = os.path.splitext(os.path.basename(f))[0]
            opath = os.path.join(odir, f"{tag}.pt")
            if os.path.exists(opath) and not args.overwrite and not args.dry_run:
                print(f"  [{fi+1}/{len(files)}] {tag}: exists, skip", flush=True)
                continue

            t0 = time.time()
            diag = Counter()
            shard, diag, ranges = process_file(
                f, y, cid, max_events=args.max_events_per_file,
                step=args.step, diag=diag)
            if shard is None:
                print(f"  [{fi+1}/{len(files)}] {tag}: 0 events built", flush=True)
                continue

            n = shard["node_feats"].shape[0]
            if not args.dry_run:
                shard["meta"] = {
                    "channel": cname, "channel_id": cid, "y": y,
                    "production": args.production, "src_file": f,
                    "feature_names": FEATURE_NAMES,
                    "feature_blocks": {k: v for k, v in FEAT_BLOCKS},
                    "node_type_names": NODE_TYPE_NAMES,
                    "channels": CHANNEL_ORDER,
                    "max_nodes": MAX_NODES,
                    "role_names": vt.ROLE_NAMES,
                    "role_not_a_vertex": ROLE_NOT_A_VERTEX,
                    "ctx_cat_names": vt.CTX_CAT_NAMES,
                    "match_chi2_max": vt.CHI2_MAX, "match_iso_min": vt.ISO_MIN,
                    "diagnostics": dict(diag),
                    "raw_ranges": {k: v for k, v in ranges.items()},
                }
                torch.save(shard, opath)

            dt = time.time() - t0
            print(f"  [{fi+1}/{len(files)}] {tag}: {n} events, {dt:.1f}s "
                  f"({n/max(dt,1e-9):.0f} ev/s)", flush=True)
            grand.update(diag)

    # ---- summary ---------------------------------------------------------
    print("\n=== diagnostics (summed over all files) ===", flush=True)
    if not grand:
        print("  (nothing built)")
        return
    for k in sorted(grand):
        print(f"  {k:42s} {grand[k]}")

    built = grand.get("events_built", 0)
    print("\n  interpretation:")
    print("    skip_no_vertices / skip_no_PV should be ~0 (stage 1 filters on")
    print("      EVT_hasPV==1, so an event without a PV would mean the branch")
    print("      does not carry what we think it does)")
    print("    truncated_events should be 0 (measured max Vertex_n = 11 <= "
          f"{MAX_VERTICES})")
    print("    DV_d0_per_nonPV_vertex should equal events_built (the measured")
    print("      convention). DV_d0_unresolved_length > 0 means a third length")
    print("      convention exists and those features are being zeroed")
    print("    any nonfinite_* or neg_* count > 0 wants explaining before "
          "training")
    if built:
        print(f"\n  events built: {built}")


if __name__ == "__main__":
    main()
