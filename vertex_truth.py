#!/usr/bin/env python3


import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import uproot

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------
# PDG code sets -- ROLE (weakly-decaying species that form their own vertex)
# --------------------------------------------------------------------------
# Absolute values throughout; sign is particle/antiparticle and irrelevant to
# the role. Only *weakly* decaying species are listed here: a strongly or
# electromagnetically decaying resonance (D*, Sigma_c, Lambda_b* ...) decays at
# its production point and never forms a distinct displaced vertex, so it can
# only ever appear as a context (see below), never as a role.

TAU_PDG = {15}

CHARM_HADRON_PDG = {
    411, 421, 431,                    # D+, D0, Ds+
    4122, 4132, 4232, 4332,           # Lambda_c+, Xi_c0, Xi_c+, Omega_c0
    4412, 4422, 4432,                 # doubly-charmed (negligible rate)
}

BOTTOM_HADRON_PDG = {
    511, 521, 531, 541,               # B0, B+, Bs0, Bc+
    5122,                             # Lambda_b0
    5132, 5232, 5332,                 # Xi_b-, Xi_b0, Omega_b-
}

# Long-lived strange hadrons. These produce real displaced vertices and are a
# large fraction of the displaced-vertex population in inclusive Zbb/Zuds.
# Previously fell into "other", which both polluted that class and discarded
# the most useful negative control available: displaced, multi-prong, and
# carrying no heavy flavour whatsoever.
STRANGE_LL_PDG = {
    310,                              # K0_S
    3122,                             # Lambda0
    3112, 3222,                       # Sigma-, Sigma+
    3312, 3322, 3334,                 # Xi-, Xi0, Omega-
}

ROLE_OTHER, ROLE_TAU, ROLE_CHARM, ROLE_BOTTOM, ROLE_STRANGE, ROLE_PV = range(6)
ROLE_UNMATCHED = -1

ROLE_NAMES = {
    ROLE_UNMATCHED: "unmatched",
    ROLE_OTHER:     "other/frag",
    ROLE_TAU:       "tau",
    ROLE_CHARM:     "charm",
    ROLE_BOTTOM:    "bottom",
    ROLE_STRANGE:   "strange-LL",
    ROLE_PV:        "PV",
}
N_ROLE_CLASSES = 6

# Priority for resolving a vertex whose mother list contains several species.
# Explicit table: an earlier implementation used max() over class indices,
# which silently inverted the documented tau > charm > bottom ordering and let
# bottom (class 3) beat tau (class 1).
_ROLE_PRIORITY = {
    ROLE_TAU:     5,
    ROLE_CHARM:   4,
    ROLE_BOTTOM:  3,
    ROLE_STRANGE: 2,
    ROLE_OTHER:   1,
}

# Matching cuts. See module docstring for the measured eff/purity behaviour.
CHI2_MAX = 30.0     # essentially inert with calibrated errors; guards pathologies
ISO_MIN = 3.0       # second-best chi2 / assigned chi2; this is the real cut
SIGMA_FLOOR = 1e-3  # mm, guards against zero/absent quoted errors


# --------------------------------------------------------------------------
# PDG code sets -- CONTEXT (grandmother categories, 2nd generation only)
# --------------------------------------------------------------------------
CTX_BC       = {541, 543, 545, 10541, 10543}
CTX_B_MESON  = {511, 513, 515, 521, 523, 525, 531, 533, 535,
                10511, 10513, 10521, 10523, 10531, 10533,
                20513, 20523, 20533}
CTX_B_BARYON = {5112, 5114, 5122, 5132, 5142, 5212, 5214, 5222, 5224,
                5232, 5242, 5314, 5324, 5332, 5334, 5342}
CTX_C_MESON  = {411, 413, 415, 421, 423, 425, 431, 433, 435,
                10411, 10413, 10421, 10423, 10431, 10433,
                20413, 20423, 20433, 441, 443, 445, 100443}
CTX_C_BARYON = {4112, 4114, 4122, 4124, 4132, 4212, 4214, 4222, 4224,
                4232, 4322, 4324, 4332, 4334, 4412, 4422, 4432}

CTX_CAT_NONE, CTX_CAT_BC, CTX_CAT_BMES, CTX_CAT_BBAR, \
    CTX_CAT_CMES, CTX_CAT_CBAR, CTX_CAT_TAU, CTX_CAT_STRANGE = range(8)
CTX_CAT_NAMES = {
    CTX_CAT_NONE:    "none/frag",
    CTX_CAT_BC:      "Bc",
    CTX_CAT_BMES:    "B-meson",
    CTX_CAT_BBAR:    "b-baryon",
    CTX_CAT_CMES:    "c-meson",
    CTX_CAT_CBAR:    "c-baryon",
    CTX_CAT_TAU:     "tau",
    CTX_CAT_STRANGE: "strange",
}

# Ordered (group, category) search list, most specific first. Bc is checked
# ahead of the generic B-meson set even though the sets don't overlap, purely
# so Bc-context vertices are easy to spot in printouts.
_CTX_GROUPS = (
    (CTX_BC,           CTX_CAT_BC),
    (CTX_B_MESON,      CTX_CAT_BMES),
    (CTX_B_BARYON,     CTX_CAT_BBAR),
    (CTX_C_MESON,      CTX_CAT_CMES),
    (CTX_C_BARYON,     CTX_CAT_CBAR),
    (TAU_PDG,          CTX_CAT_TAU),
    (STRANGE_LL_PDG,   CTX_CAT_STRANGE),
)

BRANCHES = [
    "Vertex_x", "Vertex_y", "Vertex_z",
    "Vertex_xErr", "Vertex_yErr", "Vertex_zErr",
    "Vertex_isPV", "Vertex_ntrk",
    "MC_Vertex_x", "MC_Vertex_y", "MC_Vertex_z", "MC_Vertex_ntrk",
    "MC_Vertex_PDGmother", "MC_Vertex_PDGgmother",
]
VALIDATION_BRANCHES = [
    "Tau23PiCandidates_vertex", "Tau23PiCandidates_mcvertex",
    "TrueTau23PiBc_vertex", "TrueTau23PiBu_vertex",
]


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def _abs_codes(seq):
    """Nested-branch element -> set of abs PDG ints, tolerating np/ak/list."""
    out = set()
    for p in seq:
        try:
            out.add(abs(int(p)))
        except (TypeError, ValueError):
            continue
    return out


def classify_role(pdg_mothers):
    """Role of the MC vertex whose outgoing particles have these mothers.

    The mother of the particles *at* a vertex is the particle that decayed
    *to make* that vertex, which is exactly the object we want to name.
    """
    codes = _abs_codes(pdg_mothers)
    hits = []
    if codes & TAU_PDG:
        hits.append(ROLE_TAU)
    if codes & CHARM_HADRON_PDG:
        hits.append(ROLE_CHARM)
    if codes & BOTTOM_HADRON_PDG:
        hits.append(ROLE_BOTTOM)
    if codes & STRANGE_LL_PDG:
        hits.append(ROLE_STRANGE)
    if not hits:
        return ROLE_OTHER
    return max(hits, key=lambda r: _ROLE_PRIORITY[r])


def parent_context(pdg_gmothers, role):
    """Two-generation parent context of a vertex, as (raw_pdg, category).

    raw_pdg: the specific grandmother PDG code driving the invariance test
        (e.g. 521 vs 541 vs 5122 vs 431 for the tau role). 0 if none resolved.
    category: raw_pdg folded onto a flavour class (see CTX_CAT_* / _CTX_GROUPS
        above), so that excited intermediate states (D*, B**, ...) don't
        fragment the context into many near-empty bins.

    For role in {BOTTOM, PV, UNMATCHED} the grandmother is a quark/string
    rather than a hadron (or the concept is not meaningful), so (0, NONE) is
    returned unconditionally.
    """
    if role in (ROLE_BOTTOM, ROLE_PV, ROLE_UNMATCHED):
        return 0, CTX_CAT_NONE
    codes = _abs_codes(pdg_gmothers)
    for group, cat in _CTX_GROUPS:
        inter = codes & group
        if inter:
            return min(inter), cat   # deterministic pick if >1 candidate
    return 0, CTX_CAT_NONE


# --------------------------------------------------------------------------
# geometric matching
# --------------------------------------------------------------------------
def _greedy_assignment(C):
    """Fallback for linear_sum_assignment when scipy is unavailable.

    Repeatedly takes the globally smallest remaining cost. Slightly worse than
    optimal but never double-assigns; graphs here are <= ~12 x ~18.
    """
    C = C.copy()
    rows, cols = [], []
    for _ in range(min(C.shape)):
        r, c = np.unravel_index(np.argmin(C), C.shape)
        if not np.isfinite(C[r, c]):
            break
        rows.append(r)
        cols.append(c)
        C[r, :] = np.inf
        C[:, c] = np.inf
    return np.array(rows, int), np.array(cols, int)


def match_event(reco_xyz, reco_sigma, mc_xyz, reco_ntrk=None, mc_ntrk=None):
    """Match reco vertices to MC vertices for one event.

    Returns (mc_idx, chi2, iso), each of length n_reco. mc_idx is -1 where no
    assignment was made at all. Cuts are applied by the caller so that the
    quality variables stay inspectable.
    """
    n_r, n_m = len(reco_xyz), len(mc_xyz)
    mc_idx = np.full(n_r, -1, dtype=np.int64)
    chi2 = np.full(n_r, np.inf, dtype=np.float64)
    iso = np.zeros(n_r, dtype=np.float64)
    if n_r == 0 or n_m == 0:
        return mc_idx, chi2, iso

    sig = np.maximum(reco_sigma, SIGMA_FLOOR)
    # (n_r, n_m) chi2 with 3 dof
    C = (((reco_xyz[:, None, :] - mc_xyz[None, :, :]) / sig[:, None, :]) ** 2).sum(-1)

    # Track-count veto: a reco vertex cannot be fitted from more tracks than
    # its MC vertex had. ~97-98% of true pairs satisfy this; migration in the
    # other direction (reco < mc) is expected and must NOT be vetoed
    # (arXiv:2105.13330 fig 1b: ~7% of 3-track reco vertices come from 4-track
    # MC vertices).
    if reco_ntrk is not None and mc_ntrk is not None:
        bad = np.asarray(reco_ntrk)[:, None] > np.asarray(mc_ntrk)[None, :]
        C = np.where(bad, np.inf, C)

    assign = linear_sum_assignment if _HAVE_SCIPY else _greedy_assignment
    finite = np.isfinite(C)
    if not finite.any():
        return mc_idx, chi2, iso
    Cw = np.where(finite, C, 1e12)   # linear_sum_assignment dislikes inf
    ri, mi = assign(Cw)

    for r, m in zip(ri, mi):
        if not np.isfinite(C[r, m]):
            continue
        mc_idx[r] = m
        chi2[r] = C[r, m]
        row = np.sort(C[r][np.isfinite(C[r])])
        second = row[1] if len(row) > 1 else np.inf
        iso[r] = second / max(C[r, m], 1e-12)
    return mc_idx, chi2, iso


def _get(arrays, key, i):
    return np.asarray(arrays[key][i])


def label_events(arrays, n_events, chi2_max=CHI2_MAX, iso_min=ISO_MIN):
    """Label every reco vertex in a block of events.

    Returns lists (one entry per event) of equal-length arrays:
        role      int64  ROLE_* ; ROLE_UNMATCHED where match quality fails
        context   int64  raw grandmother PDG (2-gen), 0 if n/a
        ctx_cat   int64  CTX_CAT_* category of `context`
        mc_idx    int64  matched MC vertex index, -1 if none
        chi2      f64
        iso       f64
    """
    out = defaultdict(list)
    for i in range(n_events):
        R = np.stack([_get(arrays, "Vertex_x", i),
                      _get(arrays, "Vertex_y", i),
                      _get(arrays, "Vertex_z", i)], axis=1)
        S = np.stack([_get(arrays, "Vertex_xErr", i),
                      _get(arrays, "Vertex_yErr", i),
                      _get(arrays, "Vertex_zErr", i)], axis=1)
        M = np.stack([_get(arrays, "MC_Vertex_x", i),
                      _get(arrays, "MC_Vertex_y", i),
                      _get(arrays, "MC_Vertex_z", i)], axis=1)
        is_pv = _get(arrays, "Vertex_isPV", i)
        r_ntrk = _get(arrays, "Vertex_ntrk", i)
        m_ntrk = _get(arrays, "MC_Vertex_ntrk", i)

        mc_idx, chi2, iso = match_event(R, S, M, r_ntrk, m_ntrk)

        moth = arrays["MC_Vertex_PDGmother"][i]
        gmoth = arrays["MC_Vertex_PDGgmother"][i]
        n_mc = len(moth)

        role = np.full(len(R), ROLE_UNMATCHED, dtype=np.int64)
        ctx = np.zeros(len(R), dtype=np.int64)
        ctx_cat = np.full(len(R), CTX_CAT_NONE, dtype=np.int64)
        for r in range(len(R)):
            # The PV is identified by an observable flag, not by truth, so it
            # is always labeled regardless of match quality.
            if int(is_pv[r]) == 1:
                role[r] = ROLE_PV
                continue
            m = mc_idx[r]
            if m < 0 or m >= n_mc:
                continue
            if not (chi2[r] < chi2_max and iso[r] > iso_min):
                continue          # stays ROLE_UNMATCHED
            role[r] = classify_role(moth[m])
            ctx[r], ctx_cat[r] = parent_context(gmoth[m], role[r])

        out["role"].append(role)
        out["context"].append(ctx)
        out["ctx_cat"].append(ctx_cat)
        out["mc_idx"].append(mc_idx)
        out["chi2"].append(chi2)
        out["iso"].append(iso)
    return dict(out)


# --------------------------------------------------------------------------
# I/O helper
# --------------------------------------------------------------------------
def _resolve(pattern):
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*.root")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return files


def read_block(pattern, nev, extra=()):
    f = _resolve(pattern)[0]
    t = uproot.open(f)["events"]
    want = list(BRANCHES) + [b for b in extra if b in t.keys()]
    a = t.arrays(want, entry_stop=nev, library="np")
    return f, a, len(a["Vertex_x"])


# --------------------------------------------------------------------------
# mode: validate
# --------------------------------------------------------------------------
def mode_validate(pattern, nev):
    f, a, n = read_block(pattern, nev, extra=VALIDATION_BRANCHES)
    print(f"file: {f}\nevents: {n}\nscipy: {_HAVE_SCIPY}\n")

    has_cand = "Tau23PiCandidates_vertex" in a
    tot = 0
    grid = defaultdict(lambda: [0, 0, 0])   # (chi2,iso) -> [n_sel, n_truth, n_right]
    pulls = []

    for i in range(n):
        R = np.stack([_get(a, "Vertex_x", i), _get(a, "Vertex_y", i), _get(a, "Vertex_z", i)], 1)
        S = np.stack([_get(a, "Vertex_xErr", i), _get(a, "Vertex_yErr", i), _get(a, "Vertex_zErr", i)], 1)
        M = np.stack([_get(a, "MC_Vertex_x", i), _get(a, "MC_Vertex_y", i), _get(a, "MC_Vertex_z", i)], 1)
        if len(R) == 0 or len(M) == 0:
            continue
        truth = {}
        if has_cand:
            for rv, mv in zip(_get(a, "Tau23PiCandidates_vertex", i),
                              _get(a, "Tau23PiCandidates_mcvertex", i)):
                rv, mv = int(rv), int(mv)
                if 0 <= rv < len(R) and 0 <= mv < len(M):
                    truth[rv] = mv
                    pulls.append((R[rv] - M[mv]) / np.maximum(S[rv], SIGMA_FLOOR))

        mc_idx, chi2, iso = match_event(R, S, M, _get(a, "Vertex_ntrk", i), _get(a, "MC_Vertex_ntrk", i))
        tot += len(R)
        for cmax in (10, 30, 100, np.inf):
            for imin in (1.0, 3.0, 10.0):
                g = grid[(cmax, imin)]
                for r in range(len(R)):
                    if mc_idx[r] < 0 or not (chi2[r] < cmax and iso[r] > imin):
                        continue
                    g[0] += 1
                    if r in truth:
                        g[1] += 1
                        g[2] += int(mc_idx[r] == truth[r])

    if pulls:
        P = np.abs(np.asarray(pulls))
        print("pull |R-M|/sigma  median %.3f %.3f %.3f   q95 %.3f %.3f %.3f"
              % (*np.median(P, 0), *np.percentile(P, 95, 0)))
        print("  (expected for calibrated Gaussian errors: 0.674 and 1.960)\n")

    print(f"total reco vertices: {tot}")
    print("  chi2<   iso>  |    eff    purity   (n_truth)")
    for (cmax, imin), (nsel, ntru, nright) in sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        pur = nright / ntru if ntru else float("nan")
        print("  %6g %6g  |  %.3f    %.4f   (%d)" % (cmax, imin, nsel / max(tot, 1), pur, ntru))

    # Independent cross-check: the exclusive-signal truth branches index
    # MC_Vertex_*, not Vertex_*. For a Bc event, the tagged MC vertex must have
    # a tau mother and a Bc grandmother.
    for br, gm_code, tag in (("TrueTau23PiBc_vertex", 541, "Bc"),
                             ("TrueTau23PiBu_vertex", 521, "Bu")):
        if br not in a:
            continue
        ok = bad = oor = 0
        for i in range(n):
            moth = a["MC_Vertex_PDGmother"][i]
            gmoth = a["MC_Vertex_PDGgmother"][i]
            for v in np.atleast_1d(_get(a, br, i)):
                v = int(v)
                if not (0 <= v < len(moth)):
                    oor += 1
                    continue
                if (_abs_codes(moth[v]) & TAU_PDG) and (gm_code in _abs_codes(gmoth[v])):
                    ok += 1
                else:
                    bad += 1
        tot_b = ok + bad + oor
        if tot_b:
            print(f"\n{br}: tagged MC vertices with tau mother + {tag} gmother: "
                  f"{ok}/{tot_b} ({ok / tot_b:.3f})   wrong {bad}   out-of-range {oor}")
            print("  (out-of-range should be ~0 when indexing MC_Vertex_*; a large"
                  " count means the branch indexes something else)")


# --------------------------------------------------------------------------
# mode: yields
# --------------------------------------------------------------------------
def mode_yields(patterns, nev):
    grand = {}
    for pat in patterns:
        name = os.path.basename(os.path.normpath(pat.replace("/*.root", "")))
        try:
            f, a, n = read_block(pat, nev)
        except FileNotFoundError as e:
            print(f"!! {e}")
            continue
        lab = label_events(a, n)
        cnt = Counter()          # (role, raw context) -> count
        cnt_cat = Counter()      # (role, category)     -> count
        for role, ctx, cat in zip(lab["role"], lab["context"], lab["ctx_cat"]):
            for r, c, k in zip(role, ctx, cat):
                cnt[(int(r), int(c))] += 1
                cnt_cat[(int(r), int(k))] += 1
        grand[name] = (n, cnt, cnt_cat)

        total = sum(cnt.values())
        print(f"\n=== {name}   ({n} events, {total} vertices) ===")
        by_role = Counter()
        for (r, _), v in cnt.items():
            by_role[r] += v
        for r in sorted(by_role, key=lambda x: -by_role[x]):
            print("  %-12s %8d  (%.3f)" % (ROLE_NAMES[r], by_role[r], by_role[r] / total))

        print("  -- role x category (folds excited states; use this to judge cell health) --")
        rows_cat = [(r, k, v) for (r, k), v in cnt_cat.items() if v >= 50
                    and r in (ROLE_TAU, ROLE_CHARM, ROLE_STRANGE)]
        for r, k, v in sorted(rows_cat, key=lambda t: (t[0], -t[2])):
            print("     %-12s cat %-10s %8d" % (ROLE_NAMES[r], CTX_CAT_NAMES[k], v))

        print("  -- role x raw context (>=50 vertices; fine-grained, for the tau/charm invariance test) --")
        rows = [(r, c, v) for (r, c), v in cnt.items() if v >= 50 and c != 0
                and r in (ROLE_TAU, ROLE_CHARM, ROLE_STRANGE)]
        for r, c, v in sorted(rows, key=lambda t: (t[0], -t[2])):
            print("     %-12s ctx %-6d %8d" % (ROLE_NAMES[r], c, v))

    if len(grand) > 1:
        print("\n\n=== combined (role x raw context) across samples ===")
        agg = Counter()
        agg_cat = Counter()
        for _, cnt, cnt_cat in grand.values():
            agg.update(cnt)
            agg_cat.update(cnt_cat)
        for (r, c), v in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1])):
            if v >= 50 and r in (ROLE_TAU, ROLE_CHARM, ROLE_STRANGE):
                print("  %-12s ctx %-6d %9d" % (ROLE_NAMES[r], c, v))

        print("\n=== combined (role x category) across samples ===")
        for (r, k), v in sorted(agg_cat.items(), key=lambda kv: (kv[0][0], -kv[1])):
            if v >= 50 and r in (ROLE_TAU, ROLE_CHARM, ROLE_STRANGE):
                print("  %-12s cat %-10s %9d" % (ROLE_NAMES[r], CTX_CAT_NAMES[k], v))

        print("\nGATE: any (role, context) cell you intend to test the invariance")
        print("claim on needs enough vertices to make P(expert|role,context)")
        print("stable -- rule of thumb O(1e4) after train/test splitting. Thin")
        print("raw-context cells should fold up to the category level (e.g. all")
        print("excited B-meson states -> 'B-meson') before training, not after")
        print("seeing the routing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("mode", choices=["validate", "yields"])
    ap.add_argument("patterns", nargs="+", help="sample directory or root glob")
    ap.add_argument("--nev", type=int, default=20000, help="events per file to read")
    args = ap.parse_args()

    if not _HAVE_SCIPY:
        print("[warn] scipy unavailable -> greedy assignment fallback "
              "(slightly suboptimal, still no double-assignment)\n", file=sys.stderr)

    if args.mode == "validate":
        for p in args.patterns:
            mode_validate(p, args.nev)
    else:
        mode_yields(args.patterns, args.nev)


if __name__ == "__main__":
    main()
