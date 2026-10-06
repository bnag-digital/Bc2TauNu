#!/usr/bin/env python3
"""
USAGE
    python3 routing_analysis.py --run runs/n8_k2_s0_v2
    python3 routing_analysis.py --run runs/n8_k2_s0_v2 --figures
    # several seeds: expert indices are permuted between runs, so they are
    # Hungarian-matched to a reference before averaging
    python3 routing_analysis.py --run runs/n8_k2_s0 runs/n8_k2_s1 ... --seeds
"""

import argparse
import json
import os

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# from vertex_truth / graph_build
ROLE_NAMES = {-2: "not-a-vertex", -1: "unmatched", 0: "other/frag", 1: "tau",
              2: "charm", 3: "bottom", 4: "strange-LL", 5: "PV"}
# roles that are genuine reconstructed vertices with a trustworthy label
PHYSICS_ROLES = [5, 1, 2, 3, 4, 0]
# DISPLACED roles only: the PV is identifiable from the is_PV flag alone, so
# including it inflates the NMI baseline at initialisation and makes the trained
# gain look smaller than it is. Every NMI is reported both ways.
DISPLACED_ROLES = [1, 2, 3, 4, 0]
CTX_NAMES = {0: "none/frag", 541: "Bc+", 521: "B+", 511: "B0", 531: "Bs0",
             5122: "Lb0", 431: "Ds+", 411: "D+", 421: "D0", 4122: "Lc+",
             413: "D*+", 423: "D*0", 433: "Ds*+", 3122: "Lambda"}
NODE_TYPE_NAMES = {0: "pad", 1: "PV", 2: "SV", 3: "SV3pi", 4: "EVT"}


# --------------------------------------------------------------------------
# distribution helpers
# --------------------------------------------------------------------------
def entropy(p, base=None):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -float((p * np.log(p)).sum())
    return h / np.log(base) if base else h


def normalised_entropy(p, n):
    """Entropy in [0, 1], 1 = uniform over n experts."""
    return entropy(p) / np.log(n) if n > 1 else 0.0


def jsd(p, q):
    """Jensen-Shannon DISTANCE (sqrt of the divergence), in [0, 1] with log2."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))).sum())
    div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(div, 0.0)))


def mutual_information(labels, experts, n_labels=None, n_experts=None):
    """MI and normalised MI between a label array and an expert-index array."""
    lu, li = np.unique(labels, return_inverse=True)
    eu, ei = np.unique(experts, return_inverse=True)
    if lu.size < 2 or eu.size < 2:
        return 0.0, 0.0
    joint = np.zeros((lu.size, eu.size), np.float64)
    np.add.at(joint, (li, ei), 1.0)
    joint /= joint.sum()
    pl = joint.sum(1, keepdims=True)
    pe = joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pl @ pe)[nz])).sum())
    hl = entropy(pl.ravel())
    he = entropy(pe.ravel())
    nmi = 2.0 * mi / (hl + he) if (hl + he) > 0 else 0.0
    return mi, nmi


# --------------------------------------------------------------------------
# routing file access
# --------------------------------------------------------------------------
class Routing:
    """One routing_*.npz, with convenience views."""

    def __init__(self, path, n_experts=None):
        z = np.load(path)
        self.path = path
        self.role = z["role"]
        self.context = z["context"]
        self.ctx_cat = z["ctx_cat"]
        self.node_type = z["node_type"]
        # candidate flag recorded before any feature mask collapsed the node
        # type; older runs predate it, so fall back to the node type
        self.is_cand = (z["is_cand"].astype(bool) if "is_cand" in z.files
                        else (self.node_type == 3))
        self.channel_id = z["channel_id"]
        self.y = z["y"]
        self.idx = z["expert_idx"]        # (n_nodes, n_layers, k)
        self.gate = z["expert_gate"]
        self.n_nodes, self.n_layers, self.k = self.idx.shape
        self.n_experts = int(n_experts or (self.idx.max() + 1))

    def top1(self, layer):
        return self.idx[:, layer, 0]

    def topset(self, layer):
        """Unordered top-k set encoded as a single integer per node.

        With k=2 the ordering of the pair is an artefact of the gate weights
        being close; the SET is what identifies "this role uses experts {1,4}".
        """
        s = np.sort(self.idx[:, layer, :], axis=1)
        code = np.zeros(self.n_nodes, np.int64)
        for j in range(self.k):
            code = code * self.n_experts + s[:, j]
        return code

    def dist_over_experts(self, sel, layer, weighted=False):
        """P(expert) for the selected nodes, over top-k with optional gate weights."""
        if sel.sum() == 0:
            return np.zeros(self.n_experts)
        out = np.zeros(self.n_experts, np.float64)
        idx = self.idx[sel, layer, :]
        if weighted:
            g = self.gate[sel, layer, :]
            np.add.at(out, idx.ravel(), g.ravel())
        else:
            np.add.at(out, idx.ravel(), 1.0)
        return out / max(out.sum(), 1e-12)

    def dist_top1(self, sel, layer):
        if sel.sum() == 0:
            return np.zeros(self.n_experts)
        c = np.bincount(self.top1(layer)[sel], minlength=self.n_experts).astype(float)
        return c / c.sum()


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def report_usage(rt, layer, label):
    used_1 = len(np.unique(rt.top1(layer)))
    counts = np.bincount(rt.idx[:, layer, :].ravel(), minlength=rt.n_experts)
    frac = counts / counts.sum()
    print(f"  [{label}] layer {layer}: {used_1}/{rt.n_experts} experts as top-1; "
          f"top-k load " + " ".join(f"{f:.3f}" for f in frac))
    print(f"           load CV = {frac.std() / max(frac.mean(), 1e-12):.3f} "
          f"(0 = perfectly balanced)")


def report_role_tables(init, test, layer, roles=PHYSICS_ROLES):
    """P(expert | role) for init and trained, with the entropy change."""
    print(f"\n  P(expert | role), layer {layer}, top-k weighted"
          f"   [{init.n_experts} experts]")
    hdr = "    " + "role".ljust(14) + "n".rjust(9) + "   " + \
        "  ".join(f"e{e}" for e in range(init.n_experts)) + "   H_norm"
    print("    --- at initialisation ---")
    print(hdr)
    rows = {}
    for r in roles:
        s_i = init.role == r
        s_t = test.role == r
        if s_t.sum() < 100:
            continue
        d_i = init.dist_over_experts(s_i, layer, weighted=True)
        d_t = test.dist_over_experts(s_t, layer, weighted=True)
        rows[r] = (d_i, d_t, int(s_i.sum()), int(s_t.sum()))
        print(f"    {ROLE_NAMES.get(r, r):<14}{int(s_i.sum()):>9}   "
              + "  ".join(f"{v:.2f}" for v in d_i)
              + f"   {normalised_entropy(d_i, init.n_experts):.3f}")
    print("    --- after training ---")
    print(hdr)
    for r, (d_i, d_t, n_i, n_t) in rows.items():
        h_i = normalised_entropy(d_i, init.n_experts)
        h_t = normalised_entropy(d_t, test.n_experts)
        arrow = "sharper" if h_t < h_i - 0.02 else \
                ("diffuser" if h_t > h_i + 0.02 else "~same")
        print(f"    {ROLE_NAMES.get(r, r):<14}{n_t:>9}   "
              + "  ".join(f"{v:.2f}" for v in d_t)
              + f"   {h_t:.3f}  ({h_i:.3f} -> {h_t:.3f}, {arrow})")

    # exclusivity: experts a role used at init and drove to ~zero
    print("\n    exclusivity (experts abandoned: init >= 0.05 -> trained < 0.01)")
    for r, (d_i, d_t, _, _) in rows.items():
        dropped = [e for e in range(init.n_experts)
                   if d_i[e] >= 0.05 and d_t[e] < 0.01]
        gained = [e for e in range(init.n_experts)
                  if d_i[e] < 0.05 and d_t[e] >= 0.20]
        print(f"      {ROLE_NAMES.get(r, r):<14} abandoned {dropped or '-'}"
              f"   newly dominant {gained or '-'}")
    return rows


def report_expert_identity(rows, n_experts):
    """Which role dominates each expert, from P(role | expert)."""
    print("\n  expert -> role identity (by lift over the role's overall share)")
    roles = list(rows.keys())
    # P(expert | role) matrix
    M = np.stack([rows[r][1] for r in roles])            # (n_roles, n_experts)
    n = np.array([rows[r][3] for r in roles], np.float64)
    joint = M * n[:, None]                               # counts
    per_expert = joint.sum(0)
    prior = n / n.sum()
    print("    " + "expert".ljust(8) + "share".rjust(7) + "   dominant role(s) "
          "with P(role|expert) and lift")
    for e in range(n_experts):
        if per_expert[e] <= 0:
            print(f"    e{e:<7}{0.0:>7.3f}   (unused)")
            continue
        p_role = joint[:, e] / per_expert[e]
        order = np.argsort(-p_role)
        bits = []
        for j in order[:3]:
            if p_role[j] < 0.05:
                continue
            lift = p_role[j] / max(prior[j], 1e-12)
            bits.append(f"{ROLE_NAMES.get(roles[j], roles[j])} "
                        f"{p_role[j]:.2f} (x{lift:.1f})")
        print(f"    e{e:<7}{per_expert[e]/per_expert.sum():>7.3f}   "
              + ";  ".join(bits))


def report_nmi(init, test, layer, seed=0):
    """NMI(expert, role) for init, trained, and a label-shuffled null.

    Reported over all physics roles AND over displaced roles only. The PV is
    identifiable from a single input flag, so it is separated for free at
    initialisation and drags the baseline up; excluding it isolates the part of
    the association that actually had to be learned.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for scope, roles in (("all-roles", PHYSICS_ROLES),
                         ("displaced-only", DISPLACED_ROLES)):
        print(f"\n  NMI(expert, role), layer {layer}, {scope}")
        keep_i = np.isin(init.role, roles)
        keep_t = np.isin(test.role, roles)
        for name, view in (("top-1", "top1"), ("top-k set", "topset")):
            ei = getattr(init, view)(layer)[keep_i]
            et = getattr(test, view)(layer)[keep_t]
            _, nmi_i = mutual_information(init.role[keep_i], ei)
            _, nmi_t = mutual_information(test.role[keep_t], et)
            shuf = rng.permutation(test.role[keep_t])
            _, nmi_n = mutual_information(shuf, et)
            out[f"{scope}/{name}"] = (nmi_i, nmi_t, nmi_n)
            print(f"    {name:<10} init {nmi_i:.4f}   trained {nmi_t:.4f}   "
                  f"shuffled-null {nmi_n:.4f}   "
                  f"gain over init {nmi_t - nmi_i:+.4f}")
    return out


def report_candidate_confound(test, layer, min_n=200):
    """Is the routing keyed on the 3pi-candidate FLAG rather than on the role?

    The diagnostic that exposed the confound in the unmasked run: if experts E
    are "candidate experts" rather than "tau experts", then every role's weight
    on E should track that role's CANDIDATE FRACTION rather than anything about
    its physics. Measured on the first run, charm placed 0.23 of its weight on
    the tau-associated pair and 23.3% of charm vertices were candidates -- equal
    to two decimals.
    """
    print(f"\n  === candidate-flag confound check, layer {layer} ===")
    # experts most associated with tau
    tau = test.role == 1
    if tau.sum() < min_n:
        print("    too few tau vertices")
        return None
    d_tau = test.dist_over_experts(tau, layer, weighted=True)
    tau_experts = [e for e in range(test.n_experts) if d_tau[e] >= 0.15]
    if not tau_experts:
        print("    no dominant tau experts")
        return None
    print(f"    tau-associated experts: {tau_experts} "
          f"(tau weight {d_tau[tau_experts].sum():.3f})")
    print("    " + "role".ljust(13) + "cand.frac".rjust(10)
          + "w(tau-experts)".rjust(15) + "   ratio")
    out = {"tau_experts": tau_experts}
    for r in DISPLACED_ROLES:
        s = test.role == r
        if s.sum() < min_n:
            continue
        cf = float(test.is_cand[s].mean())
        w = float(test.dist_over_experts(s, layer, weighted=True)[tau_experts].sum())
        # A role with essentially no candidates cannot "track the flag": both
        # the ratio and the |w - cf| test are meaningless there (strange-LL has
        # cand.frac = 0.000 by construction, since K0S/Lambda are 2-prong).
        meaningful = cf > 0.05
        ratio = (w / cf) if meaningful else float("nan")
        flag = ""
        if r != 1 and meaningful and abs(w - cf) < 0.05:
            flag = "  <- tracks the flag, not the role"
        rat_s = f"{ratio:>5.2f}" if meaningful else "    -"
        print(f"    {ROLE_NAMES.get(r, r):<13}{cf:>10.3f}{w:>15.3f}"
              f"   {rat_s}{flag}")
        out[ROLE_NAMES.get(r, r)] = {"cand_frac": cf, "w_tau_experts": w,
                                     "ratio": ratio}
    print("    reading: for a genuine tau expert, tau's weight should greatly")
    print("    exceed its candidate fraction while other roles' weights should")
    print("    NOT equal theirs. w ~= cand.frac for several roles means the")
    print("    experts are selecting on the reconstruction flag.")

    # and the same split within candidates only, for tau vs charm
    for r in (2, 3):
        a = test.role == 1
        b = test.role == r
        for lbl, m in (("candidates", test.is_cand), ("non-candidates", ~test.is_cand)):
            sa, sb = a & m, b & m
            if sa.sum() < min_n or sb.sum() < min_n:
                continue
            v = jsd(test.dist_over_experts(sa, layer, weighted=True),
                    test.dist_over_experts(sb, layer, weighted=True))
            print(f"    JSD(tau, {ROLE_NAMES[r]}) among {lbl:<15} "
                  f"= {v:.4f}   (n={int(sa.sum())}, {int(sb.sum())})")
            out[f"jsd_tau_{ROLE_NAMES[r]}_{lbl}"] = float(v)
    return out


def report_context_invariance(test, init, layer, role=1, min_n=300,
                              use_cat=False, n_perm=100):
    """THE CORE TEST: for one role, does routing depend on the parent hadron?"""
    key = test.ctx_cat if use_cat else test.context
    key_i = init.ctx_cat if use_cat else init.context
    label = "ctx_cat" if use_cat else "context"
    sel_role = test.role == role
    rname = ROLE_NAMES.get(role, role)
    print(f"\n  === context invariance for role = {rname}  ({label}) ===")
    ctxs = [c for c in np.unique(key[sel_role])
            if (sel_role & (key == c)).sum() >= min_n]
    if len(ctxs) < 2:
        print(f"    fewer than 2 contexts with >= {min_n} nodes; nothing to test")
        return None

    dists, dists_i, names, ns = [], [], [], []
    for c in ctxs:
        s = sel_role & (key == c)
        dists.append(test.dist_over_experts(s, layer, weighted=True))
        si = (init.role == role) & (key_i == c)
        dists_i.append(init.dist_over_experts(si, layer, weighted=True))
        nm = CTX_NAMES.get(int(c), str(int(c))) if not use_cat else str(int(c))
        names.append(nm)
        ns.append(int(s.sum()))

    print(f"    P(expert | role={rname}, {label}), trained:")
    print("      " + "context".ljust(12) + "n".rjust(8) + "   "
          + "  ".join(f"e{e}" for e in range(test.n_experts)))
    for nm, n, d in zip(names, ns, dists):
        print(f"      {nm:<12}{n:>8}   " + "  ".join(f"{v:.2f}" for v in d))

    # pairwise cross-context JSD
    m = len(dists)
    J = np.zeros((m, m))
    Ji = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            J[a, b] = jsd(dists[a], dists[b])
            Ji[a, b] = jsd(dists_i[a], dists_i[b])
    off = ~np.eye(m, dtype=bool)

    # PERMUTATION NULL: the sampling-noise floor. A JSD between two
    # distributions ESTIMATED from finite samples is nonzero even when the true
    # distributions are identical, and the small cells here (Bs0 has n ~ 840)
    # make that floor non-negligible. Shuffling the context labels among this
    # role's nodes preserves every cell size while destroying any real context
    # dependence, so the resulting JSD is exactly the floor. A trained value at
    # the floor means invariance to within statistics; clearly above it means a
    # real residual dependence on the parent.
    rng = np.random.default_rng(0)
    ctx_of_role = key[sel_role]
    idx_of_role = np.flatnonzero(sel_role)
    null_means = []
    for _ in range(n_perm):
        shuffled = rng.permutation(ctx_of_role)
        nd = []
        for c in ctxs:
            pick = idx_of_role[shuffled == c]
            s = np.zeros(len(test.role), bool)
            s[pick] = True
            nd.append(test.dist_over_experts(s, layer, weighted=True))
        Jn = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                Jn[a, b] = jsd(nd[a], nd[b])
        null_means.append(Jn[off].mean())
    null_means = np.asarray(null_means)
    null_mu = float(null_means.mean())
    null_sd = float(null_means.std())

    print(f"\n    cross-context JSD (same role, different parent):")
    print("      " + "".ljust(12) + "  ".join(f"{nm[:8]:>8}" for nm in names))
    for a in range(m):
        print(f"      {names[a]:<12}"
              + "  ".join(f"{J[a, b]:8.3f}" for b in range(m)))
    obs = float(J[off].mean())
    print(f"    mean off-diagonal: trained {obs:.4f}   "
          f"init {Ji[off].mean():.4f}   max trained {J[off].max():.4f}")
    excess = obs - null_mu
    # Proper permutation test: each shuffle is one realisation directly
    # comparable to the observed value, so the percentile of the observed within
    # the null distribution IS the p-value. Reporting a sigma against the
    # standard error of the null MEAN would treat the observed value as exact,
    # which it is not -- it carries sampling noise of the same size.
    pct = float((null_means < obs).mean() * 100.0)
    z = excess / null_sd if null_sd > 0 else float("inf")
    print(f"    permutation null (sampling-noise floor): {null_mu:.4f} "
          f"+- {null_sd:.4f}   [{n_perm} shuffles]")
    print(f"    observed sits at the {pct:.1f}th percentile of the null   "
          f"(excess {excess:+.4f}, {z:+.1f} null-sd)")
    if pct < 97.5:
        print(f"      -> INVARIANT to within sampling statistics "
              f"(cell sizes alone explain the spread)")
    else:
        print(f"      -> residual parent dependence beyond sampling noise; "
              f"judge it against the cross-role scale, not against 0")
    return {"contexts": names, "n": ns, "jsd_mean": obs,
            "jsd_max": float(J[off].max()),
            "jsd_mean_init": float(Ji[off].mean()),
            "jsd_null_mean": null_mu, "jsd_null_std": null_sd,
            "excess_over_null": float(excess), "excess_sigma": float(z),
            "null_percentile": pct, "n_perm": n_perm,
            "dists": [d.tolist() for d in dists]}


def report_cross_role_jsd(test, layer, roles=(1, 2, 3, 4), min_n=300):
    """Cross-ROLE JSD, the scale against which cross-context JSD is read."""
    rs = [r for r in roles if (test.role == r).sum() >= min_n]
    if len(rs) < 2:
        return None
    d = {r: test.dist_over_experts(test.role == r, layer, weighted=True)
         for r in rs}
    print(f"\n  cross-role JSD (different role) -- the contrast scale")
    print("      " + "".ljust(12)
          + "  ".join(f"{ROLE_NAMES.get(r, r)[:8]:>8}" for r in rs))
    vals = []
    for a in rs:
        row = []
        for b in rs:
            v = jsd(d[a], d[b])
            row.append(v)
            if a != b:
                vals.append(v)
        print(f"      {ROLE_NAMES.get(a, a):<12}"
              + "  ".join(f"{v:8.3f}" for v in row))
    print(f"    mean off-diagonal: {np.mean(vals):.4f}")
    return {"jsd_mean": float(np.mean(vals))}


def report_negative_control(test, layer, min_n=200):
    """Within 3pi-candidate vertices, tau vs charm: same object, different truth.

    If the tau-associated experts fire equally on charm 3pi candidates, the model
    has learned "displaced 3-prong vertex", not tau kinematics.
    """
    print(f"\n  === negative control: 3pi-candidate vertices only ===")
    # Must use is_cand, NOT node_type == 3: a masked run collapses SV3pi into SV
    # so the node type carries no candidate information by design. is_cand is
    # recorded before that collapse for exactly this reason.
    cand = test.is_cand
    if cand.sum() < 2 * min_n:
        print(f"    too few candidate vertices ({int(cand.sum())})")
        return None
    out = {}
    dists = {}
    for r in (1, 2, 3):
        s = cand & (test.role == r)
        if s.sum() < min_n:
            continue
        dists[r] = test.dist_over_experts(s, layer, weighted=True)
        out[ROLE_NAMES[r]] = {"n": int(s.sum()), "dist": dists[r].tolist()}
        print(f"    {ROLE_NAMES[r]:<12} n={int(s.sum()):>7}   "
              + "  ".join(f"{v:.2f}" for v in dists[r]))
    if 1 in dists and 2 in dists:
        v = jsd(dists[1], dists[2])
        print(f"    JSD(tau, charm) among 3pi candidates = {v:.4f}")
        print(f"      large  -> the model separates tau from charm on the SAME "
              f"reconstructed object")
        print(f"      ~0     -> it only learned 'displaced 3-prong'")
        out["jsd_tau_charm"] = float(v)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(init, test, layer, outdir, role=1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable, skipping figures)")
        return
    os.makedirs(outdir, exist_ok=True)

    roles = [r for r in PHYSICS_ROLES if (test.role == r).sum() >= 100]
    Mi = np.stack([init.dist_over_experts(init.role == r, layer, weighted=True)
                   for r in roles])
    Mt = np.stack([test.dist_over_experts(test.role == r, layer, weighted=True)
                   for r in roles])
    labels = [ROLE_NAMES.get(r, str(r)) for r in roles]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for a, (M, ttl) in zip(ax, ((Mi, "at initialisation"), (Mt, "after training"))):
        im = a.imshow(M, cmap="viridis", vmin=0, vmax=max(Mt.max(), Mi.max()),
                      aspect="auto")
        a.set_xticks(range(test.n_experts))
        a.set_xticklabels([f"e{e}" for e in range(test.n_experts)])
        a.set_yticks(range(len(labels)))
        a.set_yticklabels(labels)
        a.set_title(f"P(expert | role), layer {layer}, {ttl}")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.10:
                    a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                           color="w", fontsize=7)
        fig.colorbar(im, ax=a, shrink=0.85)
    p = os.path.join(outdir, f"role_expert_layer{layer}.pdf")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")

    # context panels for the chosen role -- the money plot
    sel = test.role == role
    ctxs = [c for c in np.unique(test.context[sel])
            if (sel & (test.context == c)).sum() >= 300]
    if len(ctxs) >= 2:
        fig, axes = plt.subplots(1, len(ctxs), figsize=(3 * len(ctxs), 3),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for a, c in zip(axes, ctxs):
            s = sel & (test.context == c)
            d = test.dist_over_experts(s, layer, weighted=True)
            a.bar(range(test.n_experts), d, color="steelblue")
            a.set_title(f"{CTX_NAMES.get(int(c), int(c))}\nn={int(s.sum())}",
                        fontsize=9)
            a.set_xticks(range(test.n_experts))
            a.set_xticklabels([str(e) for e in range(test.n_experts)], fontsize=7)
            a.set_ylim(0, 1)
        axes[0].set_ylabel("P(expert)")
        fig.suptitle(f"role = {ROLE_NAMES.get(role, role)}: routing by parent "
                     f"hadron (layer {layer})", fontsize=10)
        p = os.path.join(outdir, f"context_invariance_role{role}_layer{layer}.pdf")
        fig.savefig(p)
        plt.close(fig)
        print(f"  wrote {p}")


# --------------------------------------------------------------------------
def analyse_run(run, figures=False, layer=None):
    ip = os.path.join(run, "routing_init.npz")
    tp = os.path.join(run, "routing_test.npz")
    for p in (ip, tp):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    cfg = {}
    cp = os.path.join(run, "config.json")
    if os.path.exists(cp):
        cfg = json.load(open(cp))
    n_exp = cfg.get("n_experts")

    init = Routing(ip, n_experts=n_exp)
    test = Routing(tp, n_experts=n_exp)
    print(f"=== {run} ===")
    print(f"  init: {init.n_nodes} nodes, test: {test.n_nodes} nodes, "
          f"{test.n_layers} layers, k={test.k}, {test.n_experts} experts")
    if cfg:
        print(f"  config: n_experts={cfg.get('n_experts')} k={cfg.get('k')} "
              f"balance_weight={cfg.get('balance_weight')} seed={cfg.get('seed')}")

    results = {"run": run, "config": cfg, "layers": {}}
    layers = [layer] if layer is not None else list(range(test.n_layers))
    for L in layers:
        print(f"\n{'=' * 70}\nLAYER {L}\n{'=' * 70}")
        report_usage(init, L, "init")
        report_usage(test, L, "trained")
        rows = report_role_tables(init, test, L)
        report_expert_identity(rows, test.n_experts)
        nmi = report_nmi(init, test, L)
        conf = report_candidate_confound(test, L)
        ctx = report_context_invariance(test, init, L, role=1)
        ctx_cat = report_context_invariance(test, init, L, role=1, use_cat=True)
        ctx_charm = report_context_invariance(test, init, L, role=2)
        xrole = report_cross_role_jsd(test, L)
        neg = report_negative_control(test, L)

        if ctx and xrole:
            print(f"\n  >>> HEADLINE, layer {L}")
            print(f"      cross-context JSD (tau, same role)   {ctx['jsd_mean']:.4f}")
            print(f"      cross-role JSD (different roles)     {xrole['jsd_mean']:.4f}")
            ratio = xrole["jsd_mean"] / max(ctx["jsd_mean"], 1e-9)
            print(f"      sampling-noise floor                 "
                  f"{ctx['jsd_null_mean']:.4f}")
            print(f"      ratio {ratio:.1f}x  -- routing varies "
                  f"{ratio:.1f}x more with ROLE than with PARENT")
            if ratio < 2:
                print(f"      NOT a clean separation; treat with caution")

        results["layers"][L] = {"nmi": {k: list(v) for k, v in nmi.items()},
                                "candidate_confound": conf,
                                "context_tau": ctx, "context_tau_cat": ctx_cat,
                                "context_charm": ctx_charm,
                                "cross_role": xrole, "negative_control": neg}
        if figures:
            make_figures(init, test, L, os.path.join(run, "figures"))

    op = os.path.join(run, "routing_analysis.json")
    with open(op, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {op}")
    return results


def analyse_seeds(runs, layer=0):
    """Aggregate several seeds; expert indices are permuted between runs, so
    P(expert|role) matrices are Hungarian-matched to the first run first."""
    print(f"\n{'=' * 70}\nACROSS {len(runs)} SEEDS (layer {layer})\n{'=' * 70}")
    mats, nmis, ctxs, xroles = [], [], [], []
    ref = None
    for r in runs:
        ip, tp = (os.path.join(r, "routing_init.npz"),
                  os.path.join(r, "routing_test.npz"))
        if not (os.path.exists(ip) and os.path.exists(tp)):
            print(f"  skip {r} (missing routing files)")
            continue
        cfg = json.load(open(os.path.join(r, "config.json"))) \
            if os.path.exists(os.path.join(r, "config.json")) else {}
        init, test = Routing(ip, cfg.get("n_experts")), Routing(tp, cfg.get("n_experts"))
        roles = [x for x in PHYSICS_ROLES if (test.role == x).sum() >= 100]
        M = np.stack([test.dist_over_experts(test.role == x, layer, weighted=True)
                      for x in roles])
        if ref is None:
            ref = M
            perm = np.arange(test.n_experts)
        else:
            # match this run's experts to the reference by maximising overlap
            cost = -(ref.T @ M)          # (n_experts_ref, n_experts_this)
            if _HAVE_SCIPY:
                _, perm = linear_sum_assignment(cost)
            else:
                perm = np.argsort(cost.min(0))
            M = M[:, perm]
        mats.append(M)
        keep = np.isin(test.role, PHYSICS_ROLES)
        _, n_t = mutual_information(test.role[keep], test.top1(layer)[keep])
        keep_i = np.isin(init.role, PHYSICS_ROLES)
        _, n_i = mutual_information(init.role[keep_i], init.top1(layer)[keep_i])
        nmis.append((n_i, n_t))
        c = report_context_invariance(test, init, layer, role=1)
        x = report_cross_role_jsd(test, layer)
        if c:
            ctxs.append(c["jsd_mean"])
        if x:
            xroles.append(x["jsd_mean"])

    if not mats:
        return
    A = np.stack(mats)
    roles = [x for x in PHYSICS_ROLES]
    print(f"\n  mean +- std P(expert | role) over {len(mats)} seeds "
          f"(Hungarian-matched):")
    mu, sd = A.mean(0), A.std(0)
    for i in range(mu.shape[0]):
        print("    " + " ".join(f"{mu[i, j]:.2f}+-{sd[i, j]:.2f}"
                                for j in range(mu.shape[1])))
    ni = np.array([a for a, _ in nmis])
    nt = np.array([b for _, b in nmis])
    print(f"\n  NMI(expert, role) top-1: init {ni.mean():.4f}+-{ni.std():.4f}   "
          f"trained {nt.mean():.4f}+-{nt.std():.4f}   "
          f"gain {(nt - ni).mean():+.4f}+-{(nt - ni).std():.4f}")
    if ctxs and xroles:
        c, x = np.array(ctxs), np.array(xroles)
        print(f"  cross-context JSD {c.mean():.4f}+-{c.std():.4f}   "
              f"cross-role JSD {x.mean():.4f}+-{x.std():.4f}   "
              f"ratio {x.mean() / max(c.mean(), 1e-9):.1f}x")


def main():
    ap = argparse.ArgumentParser(description="MoE routing interpretability analysis.")
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--seeds", action="store_true",
                    help="aggregate across the given runs (expert indices are "
                         "Hungarian-matched before averaging)")
    args = ap.parse_args()

    if not _HAVE_SCIPY:
        print("[warn] scipy unavailable: seed matching falls back to a greedy "
              "heuristic\n")
    for r in args.run:
        analyse_run(r, figures=args.figures, layer=args.layer)
    if args.seeds and len(args.run) > 1:
        analyse_seeds(args.run, layer=args.layer or 0)


if __name__ == "__main__":
    main()
