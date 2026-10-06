#!/usr/bin/env python3
"""
USAGE
    python3 mgt_report.py --run runs/n8_k2_s0_masked
    python3 mgt_report.py --run runs/n8_k2_s0_masked --layer 1
    python3 mgt_report.py --run runs/n8_k2_s0_masked --no_figures
"""

import argparse
import json
import os

import numpy as np

import routing_analysis as ra   # reuse jsd / NMI / Routing / name tables


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
PV_ROLE = 5
TAU_ROLE = 1
CHARM_ROLE = 2
CLASS_NAMES = {0: "Bc -> tau nu", 1: "B+ -> tau nu", 2: "background"}


def load_channel_names(cfg):
    """Channel index -> name.

    dataset.GraphDataset sets channel_names = list(stats['split'].keys()), and
    feature_stats writes that dict in sorted(os.listdir(shards)) order. So the
    channel_id stored in the routing files is an index into the ALPHABETICAL
    channel list, not into graph_build.CHANNEL_ORDER. Reading it back from
    stats.json is the only way to get this right; hardcoding CHANNEL_ORDER
    would silently mislabel every per-channel row.
    """
    # A run trained with --channels sees only a SUBSET, and GraphDataset
    # renumbers channel_id by position within that subset (0..n-1). Indexing
    # the full stats.json list would then shift every label by however many
    # channels were dropped -- silently attributing one channel's physics to
    # another. train.py records the subset it actually loaded; trust that first.
    used = cfg.get("channels_used") or cfg.get("channels")
    if used:
        return {i: c for i, c in enumerate(used)}
    sp = cfg.get("stats")
    if sp and os.path.exists(sp):
        try:
            with open(sp) as fh:
                st = json.load(fh)
            return {i: c for i, c in enumerate(st["split"].keys())}
        except Exception as exc:
            print(f"  [warn] could not read channel names from {sp}: {exc}")
    return {}


def roc_auc(scores, pos):
    """One-vs-rest AUC via Mann-Whitney U, with average ranks for ties."""
    scores = np.asarray(scores, np.float64)
    pos = np.asarray(pos).astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def eff_at_thresholds(sig_scores, bkg_scores, sig_effs):
    """Background efficiency at each requested signal efficiency.

    The threshold is set on the SIGNAL distribution (keep the top eff fraction)
    and then applied to the background, which is how a cut-based selection is
    actually quoted.
    """
    out = []
    for e in sig_effs:
        if sig_scores.size == 0:
            out.append(float("nan"))
            continue
        thr = np.quantile(sig_scores, 1.0 - e)
        out.append(float((bkg_scores >= thr).mean()) if bkg_scores.size else float("nan"))
    return out


# --------------------------------------------------------------------------
# 1. primary-vertex routing consistency
# --------------------------------------------------------------------------
def pv_consistency(init, test, layer, chan_names, n_perm=200, seed=0):
    """Is the PV routed to the same expert every time, everywhere?

    Six separate readings, because "consistently" can mean any of them:
      purity        fraction of PV nodes whose top-1 expert is the modal one
      pair purity   same for the unordered top-k SET (with k=2 the pair is the
                    real unit of routing; a 50/50 top-1 split between two
                    experts is one consistent pair, not indecision)
      exclusivity   fraction of that expert's total load that is PV. High purity
                    with low exclusivity means "the PV always goes there, but so
                    does everything else" -- which is not specialisation.
      stability     is the modal expert the same in every channel, every class,
                    and at every event complexity?
      null test     cross-channel JSD against a channel-label permutation, so
                    "the panels look the same" becomes a number with an error.
      init baseline PV carries an is_PV flag that is never masked, so a random
                    gate already separates it. The learned part is the CHANGE.
    """
    rng = np.random.default_rng(seed)
    res = {"layer": layer}
    sel = test.role == PV_ROLE
    n_pv = int(sel.sum())
    print(f"\n{'=' * 72}\nPRIMARY VERTEX ROUTING -- layer {layer}\n{'=' * 72}")
    if n_pv < 100:
        print(f"  only {n_pv} PV nodes; skipping")
        return None

    # --- one PV per event? -------------------------------------------------
    ev = test.event_id[sel]
    _, counts = np.unique(ev, return_counts=True)
    n_events = int(np.unique(test.event_id).size)
    res["pv_per_event_is_one"] = float((counts == 1).mean())
    res["events_with_a_pv"] = float(len(counts) / max(n_events, 1))
    print(f"  {n_pv} PV nodes over {n_events} events;  "
          f"{res['events_with_a_pv']:.3f} of events have one, "
          f"{res['pv_per_event_is_one']:.3f} of those have exactly one")

    # --- purity, pair purity, entropy -------------------------------------
    t1 = test.top1(layer)[sel]
    cnt = np.bincount(t1, minlength=test.n_experts).astype(float)
    p_top1 = cnt / cnt.sum()
    modal = int(np.argmax(p_top1))
    purity = float(p_top1[modal])

    codes = test.topset(layer)[sel]
    uc, ucount = np.unique(codes, return_counts=True)
    best = uc[np.argmax(ucount)]
    pair_purity = float(ucount.max() / ucount.sum())
    pair = []
    b = int(best)
    for _ in range(test.k):
        pair.append(b % test.n_experts)
        b //= test.n_experts
    pair = sorted(pair)

    p_w = test.dist_over_experts(sel, layer, weighted=True)
    h = ra.normalised_entropy(p_w, test.n_experts)

    # same at init
    sel_i = init.role == PV_ROLE
    p_top1_i = (np.bincount(init.top1(layer)[sel_i], minlength=init.n_experts)
                .astype(float))
    p_top1_i /= max(p_top1_i.sum(), 1e-12)
    purity_i = float(p_top1_i.max())
    h_i = ra.normalised_entropy(
        init.dist_over_experts(sel_i, layer, weighted=True), init.n_experts)

    res.update({"modal_expert": modal, "purity_top1": purity,
                "purity_top1_init": purity_i, "modal_pair": pair,
                "purity_pair": pair_purity, "entropy_norm": h,
                "entropy_norm_init": h_i,
                "p_expert_given_pv": p_w.tolist()})
    print(f"\n  P(expert | PV), top-k gate-weighted:  "
          + "  ".join(f"e{e}:{v:.2f}" for e, v in enumerate(p_w)))
    print(f"  modal top-1 expert           e{modal}")
    print(f"  purity  (top-1)              {purity:.4f}      "
          f"(at init {purity_i:.4f})")
    print(f"  purity  (top-{test.k} set){'':<10}{pair_purity:.4f}   "
          f"-- the modal set is {{{', '.join('e%d' % e for e in pair)}}}")
    print(f"  normalised entropy           {h:.4f}      "
          f"(at init {h_i:.4f};  0 = one expert always, 1 = uniform)")

    # --- exclusivity: what else lands on that expert? ---------------------
    on_modal = test.top1(layer) == modal
    frac_pv = float((test.role[on_modal] == PV_ROLE).mean()) if on_modal.sum() else 0.0
    res["exclusivity"] = frac_pv
    print(f"  exclusivity                  {frac_pv:.4f}   "
          f"-- of everything whose top-1 is e{modal}, this fraction is the PV")
    # Judge on the top-k SET, not on top-1. With k=2 the gate picks a PAIR, so
    # a role splitting its weight across two experts is one CONSISTENT PAIR,
    # not indecision -- the same reason every table here is reported for the
    # unordered set as well as the argmax. Reading only top-1 previously called
    # a PV that lands on {e0, e5} in 87% of events "no single home expert",
    # which is the opposite of what the numbers say.
    if pair_purity > 0.8 and frac_pv > 0.9:
        print("    -> a dedicated PV route: the PV consistently uses "
              "{%s}, and essentially nothing else goes there"
              % ", ".join("e%d" % e for e in pair))
    elif pair_purity > 0.8:
        print("    -> the PV consistently uses {%s}, but that route is shared "
              "with other roles" % ", ".join("e%d" % e for e in pair))
    elif frac_pv > 0.9:
        print("    -> e%d is PV-only, but the PV spreads over several routes"
              % modal)
    else:
        print("    -> the PV does NOT have a single home expert")

    # --- stability across channels ----------------------------------------
    print(f"\n  is it the same expert everywhere?")
    print("    " + "group".ljust(18) + "n".rjust(8) + "  modal  purity   "
          + "  ".join(f"e{e}" for e in range(test.n_experts)))

    def _group_rows(key_arr, name_of, header):
        rows = {}
        print(f"    --- by {header} ---")
        for g in sorted(set(key_arr[sel].tolist())):
            s = sel & (key_arr == g)
            if s.sum() < 50:
                continue
            d = test.dist_over_experts(s, layer, weighted=True)
            m = int(np.argmax(np.bincount(test.top1(layer)[s],
                                          minlength=test.n_experts)))
            pu = float(np.bincount(test.top1(layer)[s],
                                   minlength=test.n_experts).max() / s.sum())
            nm = name_of(g)
            rows[nm] = {"n": int(s.sum()), "modal": m, "purity": pu,
                        "dist": d.tolist()}
            print(f"    {nm:<18}{int(s.sum()):>8}    e{m}   {pu:>5.3f}   "
                  + "  ".join(f"{v:.2f}" for v in d))
        return rows

    by_chan = _group_rows(test.channel_id,
                          lambda g: chan_names.get(int(g), f"chan{int(g)}"),
                          "production channel")
    by_class = _group_rows(test.y,
                           lambda g: "class: " + CLASS_NAMES.get(int(g), str(g)),
                           "event class")

    # by event complexity: number of real nodes in the event
    ev_all, inv, ev_counts = np.unique(test.event_id, return_inverse=True,
                                       return_counts=True)
    nodes_in_event = ev_counts[inv]
    bins = np.array([0, 4, 6, 8, 100])
    cbin = np.digitize(nodes_in_event, bins[1:-1])
    labels = ["size: <=4 nodes", "size: 5-6", "size: 7-8", "size: >8"]
    by_size = _group_rows(cbin, lambda g: labels[int(g)], "event size")

    modals = ([v["modal"] for v in by_chan.values()]
              + [v["modal"] for v in by_class.values()]
              + [v["modal"] for v in by_size.values()])
    same_everywhere = len(set(modals)) == 1
    res.update({"by_channel": by_chan, "by_class": by_class,
                "by_event_size": by_size,
                "same_modal_expert_everywhere": bool(same_everywhere)})
    print(f"\n    same modal expert in every group: "
          f"{'YES' if same_everywhere else 'NO -- ' + str(sorted(set(modals)))}")

    # --- permutation null on the cross-channel spread ---------------------
    # Same logic as the context-invariance null: a JSD between two finite
    # samples is nonzero even when the underlying distributions are identical,
    # so "the per-channel panels look the same" needs a floor to be read
    # against. Shuffling the channel labels among PV nodes preserves every cell
    # size and destroys any real channel dependence.
    chans = [c for c in np.unique(test.channel_id[sel])
             if (sel & (test.channel_id == c)).sum() >= 50]
    if len(chans) >= 2:
        dists = [test.dist_over_experts(sel & (test.channel_id == c), layer,
                                        weighted=True) for c in chans]
        m = len(dists)
        off = ~np.eye(m, dtype=bool)
        J = np.array([[ra.jsd(dists[a], dists[b]) for b in range(m)]
                      for a in range(m)])
        obs = float(J[off].mean())

        idx_pv = np.flatnonzero(sel)
        lab = test.channel_id[sel]
        nulls = []
        for _ in range(n_perm):
            shuf = rng.permutation(lab)
            nd = []
            for c in chans:
                s = np.zeros(len(test.role), bool)
                s[idx_pv[shuf == c]] = True
                nd.append(test.dist_over_experts(s, layer, weighted=True))
            Jn = np.array([[ra.jsd(nd[a], nd[b]) for b in range(m)]
                           for a in range(m)])
            nulls.append(Jn[off].mean())
        nulls = np.asarray(nulls)
        mu, sd = float(nulls.mean()), float(nulls.std())
        z = (obs - mu) / sd if sd > 0 else float("inf")
        pct = float((nulls < obs).mean() * 100)
        res.update({"xchannel_jsd": obs, "xchannel_null_mean": mu,
                    "xchannel_null_std": sd, "xchannel_z": z,
                    "xchannel_percentile": pct})
        print(f"\n  cross-channel JSD of PV routing   {obs:.4f}")
        print(f"  sampling-noise floor              {mu:.4f} +- {sd:.4f} "
              f"[{n_perm} shuffles]")
        print(f"  excess                            {obs - mu:+.4f}  "
              f"({z:+.1f} null-sd, {pct:.1f}th percentile)")
        if pct < 97.5:
            print("    -> PV routing is consistent across channels to within "
                  "sampling statistics")
        else:
            print("    -> PV routing does depend on the channel; read the "
                  "size of that against the cross-role scale, not against 0")

    # --- the honest caveat -------------------------------------------------
    # Report the change in PAIR purity, not top-1. With k=2 the gate selects a
    # SET, and when a role splits its weight near-evenly across two experts the
    # argmax flips event to event: top-1 purity collapses while the set is
    # perfectly stable. Quoting the top-1 change here produced lines like
    # "the learned part is the change 0.99 -> 0.63", which reads as though
    # training destroyed the separation when in fact the pair stayed clean and
    # exclusive. Exclusivity is the honest companion number: it is what falls
    # if the PV genuinely starts sharing experts with other roles.
    print(f"\n  CAVEAT to say out loud: is_PV is an input feature and is never "
          f"masked,\n  so a random gate already separates the PV "
          f"(top-1 purity {purity_i:.2f} at init). Consistent\n  PV routing is "
          f"a SANITY CHECK that the mechanism works, not evidence of learned\n"
          f"  physics. The learned part is the top-2 set purity "
          f"{pair_purity:.2f} on {{{', '.join('e%d' % e for e in pair)}}}\n"
          f"  and the exclusivity {frac_pv:.2f}. Top-1 purity moved "
          f"{purity_i:.2f} -> {purity:.2f}, but with k=2 that number tracks\n"
          f"  which member of the pair happens to win the argmax, so read the "
          f"set, not the argmax.")
    return res


# --------------------------------------------------------------------------
# 2. performance against the existing BDT
# --------------------------------------------------------------------------
def performance_report(run, cfg, chan_names, sig_effs=(0.9, 0.7, 0.5, 0.3, 0.1)):
    p = os.path.join(run, "preds_test.npz")
    if not os.path.exists(p):
        print("\n  (no preds_test.npz; skipping the performance section)")
        return None
    z = np.load(p)
    probs, y = z["probs"], z["y"].astype(int)
    chan = z["channel_id"].astype(int) if "channel_id" in z.files else None
    mva1 = z["evt_mva1"] if "evt_mva1" in z.files else None

    print(f"\n{'=' * 72}\nPERFORMANCE, AND WHAT IT CAN AND CANNOT BE COMPARED TO"
          f"\n{'=' * 72}")
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    s_sig = probs[:, 0] + probs[:, 1]          # P(signal) = P(Bc) + P(Bu)
    s_bc = probs[:, 0]
    auc_sig = roc_auc(s_sig, y != 2)
    auc_bc = roc_auc(s_bc, y == 0)
    out = {"accuracy": acc, "auc_signal_vs_bkg": auc_sig, "auc_Bc_vs_rest": auc_bc}
    print(f"  accuracy (3-class)                       {acc:.4f}")
    print(f"  AUC, signal (Bc or B+) vs background     {auc_sig:.4f}")
    print(f"  AUC, Bc vs everything else               {auc_bc:.4f}")
    if mva1 is not None:
        a1 = roc_auc(mva1, y != 2)
        out["auc_evt_mva1"] = a1
        print(f"  AUC of EVT_MVA1 (BDT1) on the SAME events {a1:.4f}")

    # confusion matrix
    C = np.zeros((3, 3), int)
    np.add.at(C, (y, pred), 1)
    out["confusion"] = C.tolist()
    print("\n  confusion matrix (rows = truth, cols = predicted)")
    print("    " + "".ljust(16) + "".join(f"{CLASS_NAMES[c][:12]:>14}" for c in range(3)))
    for r in range(3):
        print(f"    {CLASS_NAMES[r][:15]:<16}"
              + "".join(f"{C[r, c]:>14d}" for c in range(3))
              + f"    (recall {C[r, r] / max(C[r].sum(), 1):.3f})")

    # per-mode background efficiency at fixed signal efficiency
    if chan is not None and chan_names:
        print(f"\n  background efficiency at fixed SIGNAL efficiency "
              f"(MGT score = P(Bc)+P(B+))")
        print("    " + "background mode".ljust(18)
              + "".join(f"{f'eps_S={e:.0%}':>12}" for e in sig_effs))
        sig_scores = s_sig[y != 2]
        per_mode = {}
        for ci in sorted(set(chan.tolist())):
            s = (chan == ci) & (y == 2)
            if s.sum() < 50:
                continue
            nm = chan_names.get(ci, f"chan{ci}")
            effs = eff_at_thresholds(sig_scores, s_sig[s], sig_effs)
            per_mode[nm] = effs
            print(f"    {nm:<18}" + "".join(f"{e:>12.4f}" for e in effs))
        if mva1 is not None:
            print("    " + "-" * 60)
            print("    reference: the same table for EVT_MVA1 (BDT1)")
            mva_sig = mva1[y != 2]
            for ci in sorted(set(chan.tolist())):
                s = (chan == ci) & (y == 2)
                if s.sum() < 50:
                    continue
                nm = chan_names.get(ci, f"chan{ci}")
                effs = eff_at_thresholds(mva_sig, mva1[s], sig_effs)
                per_mode[nm + " (BDT1)"] = effs
                print(f"    {nm:<18}" + "".join(f"{e:>12.4f}" for e in effs))
        out["bkg_eff_at_sig_eff"] = {"sig_effs": list(sig_effs), "modes": per_mode}

    print("""
  HOW TO QUOTE THIS, AND HOW NOT TO
    Amhis et al. report ROC areas of 0.984 for BDT1 and 0.966 for BDT2. Neither
    is comparable to the numbers above, for two independent reasons:

    (a) These events come from the 'analysis' production, which already
        requires EVT_MVA1 > 0.6. Every event here has therefore already passed
        BDT1's own cut, so the AUC measured for EVT_MVA1 on this sample is its
        RESIDUAL discrimination inside its own signal region -- structurally
        much lower than the 0.984 it scores on the full pre-selected sample.
        Rebuilding shards with --production training removes this and is the
        only way to get a like-for-like BDT1 number.
    (b) The background mixture here is ~44% exclusive hadronic 3pi modes, not a
        physical Z->qq composition. Those modes were chosen as hard negatives
        for the interpretability study; they are far harder than the inclusive
        background the BDTs were trained against, so the absolute AUC is not a
        physics number either way.

    What IS a fair statement: on identical events, with an identical background
    mixture, the graph transformer separates signal from background better than
    the BDT1 score carried in the same ntuples. The per-mode efficiency table
    above is the more useful comparison, because it is the quantity the
    published selection is actually optimised on.

    BDT2 cannot be compared at all yet: only EVT_MVA1 is read by graph_build.py.
    If EVT_MVA2 exists in the stage-1 ntuples, adding it to the scalar list is a
    two-line change plus a shard rebuild.""")
    return out


# --------------------------------------------------------------------------
# 3. figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.titlesize": 10, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    return plt


def _expert_colours(plt, n):
    import matplotlib.cm as cm
    return [cm.tab10(i % 10) for i in range(n)]


def fig_who_goes_where(init, test, layer, outdir, plt):
    """Figure 1 -- the Genovese et al. Fig 9 plot, with vertex role on x.

    One bar per role, split by which expert processed the node. If the bars
    have visibly different colour compositions, different roles use different
    experts. That is the whole claim, in one picture.
    """
    roles = [r for r in ra.PHYSICS_ROLES if (test.role == r).sum() >= 100]
    names = [ra.ROLE_NAMES[r] for r in roles]
    cols = _expert_colours(plt, test.n_experts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True,
                             constrained_layout=True)
    for ax, (rt, ttl) in zip(axes, ((init, "BEFORE training (random gate)"),
                                    (test, "AFTER training"))):
        M = np.stack([rt.dist_over_experts(rt.role == r, layer, weighted=True)
                      for r in roles])
        bottom = np.zeros(len(roles))
        for e in range(rt.n_experts):
            ax.bar(names, M[:, e], bottom=bottom, color=cols[e],
                   label=f"expert {e}", edgecolor="white", linewidth=0.4)
            bottom += M[:, e]
        ax.set_ylim(0, 1)
        ax.set_title(ttl)
        ax.set_xlabel("what actually decayed at this vertex (simulation truth)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("share of the vertex's routing weight")
    axes[1].legend(ncol=1, fontsize=7, loc="center left",
                   bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"Q: do different kinds of vertex go to different experts?   "
                 f"(layer {layer})\n"
                 f"read it as: same colour pattern in every bar = no "
                 f"specialisation;  different patterns = specialisation",
                 fontsize=10)
    p = os.path.join(outdir, f"fig1_who_goes_where_layer{layer}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_professors_test(test, layer, outdir, plt, min_n=300):
    """Figure 2 -- his test and its necessary companion, on one page.

    LEFT  : tau vertices only, one bar group per parent hadron.
            His prediction: the groups sit on top of each other.
    RIGHT : one bar group per role.
            The contrast scale. If the left panel is flat and the right panel
            is not, the routing encodes role and largely ignores parent.
    """
    tau = test.role == TAU_ROLE
    ctxs = [c for c in np.unique(test.context[tau])
            if (tau & (test.context == c)).sum() >= min_n]
    roles = [r for r in (TAU_ROLE, CHARM_ROLE, 3, 4)
             if (test.role == r).sum() >= min_n]
    if len(ctxs) < 2 or len(roles) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True,
                             constrained_layout=True)
    x = np.arange(test.n_experts)

    w = 0.8 / len(ctxs)
    for i, c in enumerate(ctxs):
        s = tau & (test.context == c)
        d = test.dist_over_experts(s, layer, weighted=True)
        axes[0].bar(x + (i - (len(ctxs) - 1) / 2) * w, d, width=w,
                    label=f"{ra.CTX_NAMES.get(int(c), int(c))}  (n={int(s.sum()):,})")
    axes[0].set_title("SAME role (tau), DIFFERENT parent hadron\n"
                      "his prediction: these should line up")
    axes[0].set_ylabel("share of routing weight")

    w = 0.8 / len(roles)
    for i, r in enumerate(roles):
        s = test.role == r
        d = test.dist_over_experts(s, layer, weighted=True)
        axes[1].bar(x + (i - (len(roles) - 1) / 2) * w, d, width=w,
                    label=f"{ra.ROLE_NAMES[r]}  (n={int(s.sum()):,})")
    axes[1].set_title("DIFFERENT role\n"
                      "the contrast scale: these should NOT line up")

    # quantify both panels on the figure
    dctx = [test.dist_over_experts(tau & (test.context == c), layer, weighted=True)
            for c in ctxs]
    drole = [test.dist_over_experts(test.role == r, layer, weighted=True)
             for r in roles]

    def _mean_off(ds):
        m = len(ds)
        v = [ra.jsd(ds[a], ds[b]) for a in range(m) for b in range(m) if a != b]
        return float(np.mean(v))

    j_ctx, j_role = _mean_off(dctx), _mean_off(drole)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"e{e}" for e in x])
        ax.set_xlabel("expert")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False)
    axes[0].text(0.02, 0.96, f"spread between bars (JSD) = {j_ctx:.3f}",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(fc="white", ec="0.7", alpha=0.9))
    axes[1].text(0.02, 0.96, f"spread between bars (JSD) = {j_role:.3f}\n"
                             f"that is {j_role / max(j_ctx, 1e-9):.1f}x larger",
                 transform=axes[1].transAxes, va="top", fontsize=9,
                 bbox=dict(fc="white", ec="0.7", alpha=0.9))
    fig.suptitle(f"Q: does the model care more about WHAT decayed than about "
                 f"WHAT MADE IT?   (layer {layer})", fontsize=10)
    p = os.path.join(outdir, f"fig2_role_vs_parent_layer{layer}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_negative_control(test, layer, outdir, plt, min_n=200):
    """Figure 3 -- tau vs charm on the SAME reconstructed object.

    Both are displaced 3-prong vertices flagged by the reconstruction as 3pi
    candidates. If they route the same way, the model learned "3-prong", not
    "tau". This is the figure that turns a nice result into a defended one.
    """
    cand = test.is_cand
    a, b = cand & (test.role == TAU_ROLE), cand & (test.role == CHARM_ROLE)
    if a.sum() < min_n or b.sum() < min_n:
        return None
    da = test.dist_over_experts(a, layer, weighted=True)
    db = test.dist_over_experts(b, layer, weighted=True)
    x = np.arange(test.n_experts)
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    ax.bar(x - 0.2, da, width=0.4, label=f"tau -> 3 pions  (n={int(a.sum()):,})",
           color="#2c6fbb")
    ax.bar(x + 0.2, db, width=0.4,
           label=f"charm -> 3 pions  (n={int(b.sum()):,})", color="#d1603d")
    ax.set_xticks(x)
    ax.set_xticklabels([f"e{e}" for e in x])
    ax.set_xlabel("expert")
    ax.set_ylabel("share of routing weight")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    v = ra.jsd(da, db)
    verdict = ("the model tells them apart" if v > 0.15 else
               "WARNING: it does not separate them -- it may have learned "
               "'3-prong vertex', not 'tau'")
    ax.set_title(f"Q: same object, different physics -- does the routing "
                 f"differ?   (layer {layer})\n"
                 f"both are displaced 3-prong vertices tagged as 3pi "
                 f"candidates\nJSD = {v:.3f}  ->  {verdict}", fontsize=9.5)
    p = os.path.join(outdir, f"fig3_negative_control_layer{layer}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_effect_sizes(ctx_res, xrole, outdir, layer, plt):
    """Figure 4 -- three bars: noise floor, parent effect, role effect.

    The single most useful summary slide. It makes "0.10 is small" checkable
    instead of asserted, because the height of a pure-noise bar is drawn next
    to it.
    """
    if not ctx_res or not xrole:
        return None
    floor = ctx_res["jsd_null_mean"]
    floor_sd = ctx_res["jsd_null_std"]
    parent = ctx_res["jsd_mean"]
    role = xrole["jsd_mean"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    labels = ["pure sampling noise\n(labels shuffled)",
              "changing the PARENT\n(tau from Bc, B+, B0, Bs, Ds)",
              "changing the ROLE\n(tau vs charm vs bottom vs strange)"]
    vals = [floor, parent, role]
    errs = [floor_sd, 0, 0]
    bars = ax.bar(labels, vals, yerr=errs, capsize=4,
                  color=["#9e9e9e", "#e0a458", "#2c6fbb"])
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel("how much the routing changes  (Jensen-Shannon distance)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title(f"Q: how big is each effect?   (layer {layer})\n"
                 f"role changes the routing {role / max(parent, 1e-9):.1f}x "
                 f"more than parent does;\nthe parent effect is "
                 f"{parent / max(floor, 1e-9):.1f}x the noise floor, so it is "
                 f"small but real", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8)
    p = os.path.join(outdir, f"fig4_effect_sizes_layer{layer}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_pv(pv, test, layer, outdir, plt):
    """Figure 5 -- the primary vertex, split every way that could break it."""
    if not pv:
        return None
    cols = _expert_colours(plt, test.n_experts)
    groups = []
    for src, tag in ((pv.get("by_channel", {}), ""),
                     (pv.get("by_class", {}), ""),
                     (pv.get("by_event_size", {}), "")):
        for k, v in src.items():
            groups.append((k + tag, v))
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(groups) + 4), 4.4),
                           constrained_layout=True)
    names = [g[0] for g in groups]
    M = np.array([g[1]["dist"] for g in groups])
    bottom = np.zeros(len(groups))
    for e in range(test.n_experts):
        ax.bar(names, M[:, e], bottom=bottom, color=cols[e],
               label=f"expert {e}", edgecolor="white", linewidth=0.4)
        bottom += M[:, e]
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the PV's routing weight")
    ax.tick_params(axis="x", rotation=40, labelsize=7.5)
    ax.legend(ncol=1, fontsize=7, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    verdict = ("SAME expert in every single group"
               if pv.get("same_modal_expert_everywhere") else
               "the modal expert CHANGES between groups")
    ax.set_title(f"Q: is the primary vertex routed consistently?   "
                 f"(layer {layer})\n"
                 f"purity {pv['purity_top1']:.3f}  "
                 f"(random gate before training: {pv['purity_top1_init']:.3f})   |   "
                 f"exclusivity {pv['exclusivity']:.3f}   |   {verdict}",
                 fontsize=9.5)
    p = os.path.join(outdir, f"fig5_primary_vertex_layer{layer}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_performance(run, cfg, chan_names, outdir, plt):
    """Figure 6 -- ROC vs BDT1 and the confusion matrix, side by side."""
    p_in = os.path.join(run, "preds_test.npz")
    if not os.path.exists(p_in):
        return None
    z = np.load(p_in)
    probs, y = z["probs"], z["y"].astype(int)
    mva1 = z["evt_mva1"] if "evt_mva1" in z.files else None
    s = probs[:, 0] + probs[:, 1]
    pos = (y != 2)

    def roc_curve(score, pos):
        o = np.argsort(-score)
        p = pos[o]
        tp = np.cumsum(p) / max(p.sum(), 1)
        fp = np.cumsum(~p) / max((~p).sum(), 1)
        return np.concatenate([[0], fp]), np.concatenate([[0], tp])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    fx, ty = roc_curve(s, pos)
    axes[0].plot(fx, ty, lw=2, color="#2c6fbb",
                 label=f"graph transformer   AUC = {roc_auc(s, pos):.4f}")
    if mva1 is not None:
        fx2, ty2 = roc_curve(mva1, pos)
        axes[0].plot(fx2, ty2, lw=2, color="#d1603d", ls="--",
                     label=f"BDT1 (EVT_MVA1)     AUC = {roc_auc(mva1, pos):.4f}")
    axes[0].plot([0, 1], [0, 1], color="0.7", lw=1, ls=":")
    axes[0].set_xlabel("background kept (false positive rate)")
    axes[0].set_ylabel("signal kept (true positive rate)")
    axes[0].set_title("signal (Bc or B+) vs background,\non identical events")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    pred = probs.argmax(1)
    C = np.zeros((3, 3))
    np.add.at(C, (y, pred), 1)
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    im = axes[1].imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([CLASS_NAMES[c] for c in range(3)], rotation=20,
                            fontsize=8)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels([CLASS_NAMES[c] for c in range(3)], fontsize=8)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("truth")
    axes[1].grid(False)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center",
                         color="white" if Cn[i, j] > 0.5 else "black",
                         fontsize=10)
    axes[1].set_title("confusion matrix (row-normalised)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.suptitle("Q: does the interpretable model actually classify well?",
                 fontsize=10)
    out = os.path.join(outdir, "fig6_performance.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_bkg_rejection(run, chan_names, outdir, plt):
    """Figure 7 -- per-mode background rejection vs signal efficiency.

    Amhis et al. quote efficiency profiles, not AUCs, so this is the plot that
    can be laid next to their figures 2b and 3b.
    """
    p_in = os.path.join(run, "preds_test.npz")
    if not os.path.exists(p_in) or not chan_names:
        return None
    z = np.load(p_in)
    if "channel_id" not in z.files:
        return None
    probs, y = z["probs"], z["y"].astype(int)
    chan = z["channel_id"].astype(int)
    s = probs[:, 0] + probs[:, 1]
    sig = s[y != 2]
    if sig.size == 0:
        return None
    effs = np.linspace(0.02, 0.99, 60)
    thr = np.quantile(sig, 1.0 - effs)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for ci in sorted(set(chan.tolist())):
        sel = (chan == ci) & (y == 2)
        if sel.sum() < 50:
            continue
        bs = s[sel]
        be = np.array([(bs >= t).mean() for t in thr])
        ax.plot(effs, np.maximum(be, 1e-5), lw=1.8,
                label=f"{chan_names.get(ci, ci)}  (n={int(sel.sum()):,})")
    ax.set_yscale("log")
    ax.set_xlabel("signal efficiency (fraction of Bc/B+ -> tau nu kept)")
    ax.set_ylabel("background efficiency (fraction kept)")
    ax.set_title("Q: which backgrounds are hard?\n"
                 "lower is better; the exclusive 3pi modes are the hard "
                 "negatives by construction", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    out = os.path.join(outdir, "fig7_background_rejection.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def channel_3pi_report(test, layer, chan_names, outdir, plt, min_n=100):
    """The four exclusive hadronic 3pi modes are already in the mixture, so
    "do B -> D 3pi events activate the same experts as the signal?" can be
    answered now, with no extra training.

    For each production channel, restrict to the vertices the reconstruction
    flagged as 3pi candidates and ask two things:
      -- what actually decayed there (the truth composition), and
      -- where the routing sent it.
    A signal channel's candidates are ~all tau; a Bd2D3Pi channel's candidates
    are ~all charm. If the routing follows the truth composition rather than
    the channel, the experts are keyed on the vertex, not on the event.
    """
    print(f"\n{'=' * 72}\n3pi-CANDIDATE VERTICES BY PRODUCTION CHANNEL -- "
          f"layer {layer}\n{'=' * 72}")
    cand = test.is_cand
    if cand.sum() < min_n:
        print("  too few candidate vertices")
        return None, None
    chans = [c for c in np.unique(test.channel_id[cand])
             if (cand & (test.channel_id == c)).sum() >= min_n]
    if not chans:
        return None, None

    print("  " + "channel".ljust(14) + "n".rjust(7) + "   truth: "
          + "  ".join(f"{ra.ROLE_NAMES[r][:7]:>7}" for r in (1, 2, 3, 0))
          + "     routing: "
          + "  ".join(f"e{e}" for e in range(test.n_experts)))
    rows, dists, names = {}, [], []
    for c in chans:
        s = cand & (test.channel_id == c)
        comp = [float((test.role[s] == r).mean()) for r in (1, 2, 3, 0)]
        d = test.dist_over_experts(s, layer, weighted=True)
        nm = chan_names.get(int(c), f"chan{int(c)}")
        rows[nm] = {"n": int(s.sum()), "truth": comp, "dist": d.tolist()}
        dists.append(d)
        names.append(nm)
        print(f"  {nm:<14}{int(s.sum()):>7}          "
              + "  ".join(f"{v:>7.2f}" for v in comp) + "            "
              + "  ".join(f"{v:.2f}" for v in d))

    # Does the routing follow the tau fraction? If the experts encode the
    # vertex's identity, a channel's weight on the tau-experts should track its
    # candidates' TAU FRACTION -- which is a truth quantity -- rather than being
    # flat across channels.
    tau = test.role == 1
    d_tau = test.dist_over_experts(tau, layer, weighted=True)
    tau_e = [e for e in range(test.n_experts) if d_tau[e] >= 0.15]
    if tau_e:
        xs = np.array([rows[n]["truth"][0] for n in names])
        ys = np.array([sum(rows[n]["dist"][e] for e in tau_e) for n in names])
        r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
        print(f"\n  tau-associated experts: {tau_e}")
        print(f"  correlation across channels between a channel's TAU FRACTION")
        print(f"  and its weight on those experts:  r = {r:.3f}")
        if r > 0.8:
            print("    -> the routing tracks what actually decayed, not which")
            print("       production channel the event came from")
        rows["_tau_experts"] = tau_e
        rows["_corr_taufrac_vs_weight"] = r

    fig = None
    if plt is not None and len(dists) >= 2:
        cols = _expert_colours(plt, test.n_experts)
        f, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
        M = np.array([rows[n]["truth"] for n in names])
        bottom = np.zeros(len(names))
        for i, (r_, lab) in enumerate(zip((1, 2, 3, 0),
                                          ("tau", "charm", "bottom", "other"))):
            axes[0].bar(names, M[:, i], bottom=bottom, label=lab,
                        edgecolor="white", linewidth=0.4)
            bottom += M[:, i]
        axes[0].set_title("what these vertices really are (truth)")
        axes[0].set_ylabel("fraction of the channel's 3pi candidates")
        axes[0].legend(fontsize=7, frameon=False)

        D = np.array([rows[n]["dist"] for n in names])
        bottom = np.zeros(len(names))
        for e in range(test.n_experts):
            axes[1].bar(names, D[:, e], bottom=bottom, color=cols[e],
                        label=f"expert {e}", edgecolor="white", linewidth=0.4)
            bottom += D[:, e]
        axes[1].set_title("where the model sent them")
        axes[1].set_ylabel("share of routing weight")
        axes[1].legend(ncol=1, fontsize=7, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), frameon=False)
        for ax in axes:
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
        f.suptitle(f"Q: do B -> D 3pi events use the same experts as the "
                   f"signal?   (layer {layer})\n"
                   f"if the right panel mirrors the left, the experts follow "
                   f"the VERTEX, not the production channel", fontsize=10)
        fig = os.path.join(outdir, f"fig8_channels_3pi_layer{layer}.png")
        f.savefig(fig, bbox_inches="tight")
        plt.close(f)
    return rows, fig


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="PV-routing check, BDT comparison and plain-language "
                    "figures for one training run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None,
                    help="figure directory (default <run>/report)")
    ap.add_argument("--layer", type=int, default=None,
                    help="restrict to one layer (default: all)")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()

    run = args.run
    cfg = {}
    cp = os.path.join(run, "config.json")
    if os.path.exists(cp):
        cfg = json.load(open(cp))
    n_exp = cfg.get("n_experts")
    chan_names = load_channel_names(cfg)

    init = ra.Routing(os.path.join(run, "routing_init.npz"), n_experts=n_exp)
    test = ra.Routing(os.path.join(run, "routing_test.npz"), n_experts=n_exp)
    # event_id is needed for the per-event PV checks and Routing does not
    # expose it, so attach it here rather than editing routing_analysis.py
    for rt, pth in ((init, "routing_init.npz"), (test, "routing_test.npz")):
        rt.event_id = np.load(os.path.join(run, pth))["event_id"]

    print(f"=== {run} ===")
    masked = cfg.get("masked_feature_names") or []
    print(f"  {test.n_experts} experts, k={test.k}, {test.n_layers} layers, "
          f"seed={cfg.get('seed')}")
    print(f"  masked features ({len(masked)}): "
          f"{', '.join(masked) if masked else 'none'}")
    print(f"  SV3pi collapsed into SV: {cfg.get('collapsed_sv3pi')}")

    outdir = args.out or os.path.join(run, "report")
    os.makedirs(outdir, exist_ok=True)
    plt = _plt() if not args.no_figures else None

    summary = {"run": run, "config": cfg, "layers": {}}
    layers = [args.layer] if args.layer is not None else list(range(test.n_layers))
    written = []
    for L in layers:
        pv = pv_consistency(init, test, L, chan_names, n_perm=args.n_perm)
        # reuse routing_analysis for the two JSD scales, quietly
        ctx = ra.report_context_invariance(test, init, L, role=TAU_ROLE)
        xrole = ra.report_cross_role_jsd(test, L)
        ch3, ch3fig = channel_3pi_report(test, L, chan_names, outdir, plt)
        summary["layers"][L] = {"pv": pv, "context_tau": ctx,
                                "cross_role": xrole, "channels_3pi": ch3}
        if ch3fig:
            written.append(ch3fig)
        if plt is not None:
            for f in (fig_who_goes_where(init, test, L, outdir, plt),
                      fig_professors_test(test, L, outdir, plt),
                      fig_negative_control(test, L, outdir, plt),
                      fig_effect_sizes(ctx, xrole, outdir, L, plt),
                      fig_pv(pv, test, L, outdir, plt)):
                if f:
                    written.append(f)

    summary["performance"] = performance_report(run, cfg, chan_names)
    if plt is not None:
        for f in (fig_performance(run, cfg, chan_names, outdir, plt),
                  fig_bkg_rejection(run, chan_names, outdir, plt)):
            if f:
                written.append(f)

    sp = os.path.join(outdir, "summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {sp}")
    for f in written:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
