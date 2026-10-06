#!/usr/bin/env python3
"""
dataset.py

Stage 2 of the MoE-graph-transformer pipeline: a torch Dataset (plus a collate
function and mixture-aware sampler) that turns the padded shards from
graph_build.py into standardised, augmented, batched model inputs, using the
split and statistics frozen in stats.json by feature_stats.py.

WHAT IT DOES, PER EVENT
    1. loads the (13, 50) node feature table and its mask from a shard
    2. STANDARDISES each feature using the train-only statistics for the node
       type it lives on (PV / SV / EVT group), leaving structural zeros and
       binary flags untouched, and clipping to +-CLIP so the tails of the
       heavy features cannot dominate
    3. SHUFFLES the order of the real nodes (padding stays at the end). The
       audit found Vertex_isPV == slot 0 in 5000/5000 events -- an ordering
       leak. The model must read "this is the PV" from the is_PV flag, never
       from position, so the order is randomised every time the event is drawn.
    4. builds the 6 EDGE features on the fly from the raw displacement vectors
       (node_aux), which is cheaper than storing a (13,13,6) tensor per event
       and lets the shuffle in step 3 permute edges consistently for free
    5. returns tensors the model consumes, plus -- separately -- the truth
       labels (role / context / ctx_cat) for the routing analysis. The model
       must never receive those; they travel in a distinct dict key so a
       training loop cannot pick them up by accident.

WHY STANDARDISATION LIVES HERE, NOT IN THE SHARD
    Standardising at build time would bake one particular choice of statistics
    into the data. Doing it here, from stats.json, means the split and the
    scaling can be changed and the shards reused, and -- crucially -- it is
    structurally impossible to standardise with anything other than the
    train-only statistics, because that is all stats.json exposes.

EDGE FEATURES (all symmetric except #2, which is antisymmetric)
    0  log1p( ||v_i - v_j|| )              3D separation of the two vertices
    1  d2PV_i - d2PV_j                       signed: which is further downstream
    2  cos angle( v_i - PV , v_j - PV )      alignment of the two displacements
    3  same signal hemisphere (0/1)          from is_signal_hemisphere
    4  cos angle( v_j - v_i , v_i - PV )      "is i on the path from PV to j" --
                                             the soft parent->child signal that
                                             lets the model represent B->D->K
                                             decay chains
    5  either endpoint is the EVT node (0/1) so the global node is distinguishable
    Displacement vectors come from node_aux; the PV's is ~0 by construction, so
    edges touching the PV use its position (origin of the d2PV frame) directly.

USAGE (as a library)
    from dataset import GraphDataset, collate, make_loader
    tr = GraphDataset("stats.json", split="train")
    va = GraphDataset("stats.json", split="val")
    loader = make_loader(tr, batch_size=256, shuffle=True, mixture="physical")
    for batch in loader:
        logits = model(batch["node_feats"], batch["node_mask"],
                       batch["node_type"], batch["edge_feats"])
        loss = crossentropy(logits, batch["y"])
        # batch["role"], batch["context"], batch["ctx_cat"] exist but are for
        # evaluation only -- never feed them to the model or the loss

SELF-TEST
    python3 dataset.py --stats stats.json --selftest
        Loads a few batches, checks shapes, mask invariants, that standardised
        features have ~unit spread on train, that padding and structural zeros
        stayed zero, and that the node shuffle actually permutes.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# node-type codes, mirrored from graph_build; verified against stats.json meta
NODE_PAD, NODE_PV, NODE_SV, NODE_SV3PI, NODE_EVT = 0, 1, 2, 3, 4
N_EDGE_FEATS = 6

# roles that are genuinely vertices (for the routing-analysis denominator).
# ROLE_NOT_A_VERTEX (-2, pad/EVT) and ROLE_UNMATCHED (-1) are excluded there,
# but ALL nodes are always fed to the model regardless of role.
ROLE_NOT_A_VERTEX = -2
ROLE_UNMATCHED = -1


# --------------------------------------------------------------------------
# standardiser
# --------------------------------------------------------------------------
class Standardizer:
    """Per-(node-type-group, feature) centering and scaling from stats.json.

    Precomputes, for each of the three groups, a (n_feat,) center and scale
    vector plus a boolean "standardise this column" mask, so applying it to a
    shard is three array ops per group rather than a Python loop over features.
    """

    def __init__(self, stats, method="robust", clip=5.0):
        self.clip = float(clip)
        self.feature_names = list(stats["feature_names"])
        self.nfeat = len(self.feature_names)
        self.no_std = set(stats["no_standardize"])
        # feature -> set of node types it is applicable on
        self.applicable = {f: set(ts) for f, ts in stats["applicable_types"].items()}
        # node type -> group name
        self.group_of = {int(k): v for k, v in stats["norm_group_of_node_type"].items()}

        if method not in ("robust", "standard"):
            raise ValueError(method)
        cen_key = "median" if method == "robust" else "mean"
        scl_key = "robust_std" if method == "robust" else "std"

        groups = sorted(set(self.group_of.values()))
        self.center = {g: np.zeros(self.nfeat, np.float32) for g in groups}
        self.scale = {g: np.ones(self.nfeat, np.float32) for g in groups}
        self.do = {g: np.zeros(self.nfeat, bool) for g in groups}

        st = stats["stats"]
        for g in groups:
            for j, f in enumerate(self.feature_names):
                if f in self.no_std:
                    continue
                d = st.get(g, {}).get(f)
                if d is None:
                    continue                    # feature n/a for this group
                c = d.get(cen_key)
                s = d.get(scl_key, 0.0)
                if c is None:
                    continue
                if not s or s <= 0.0:
                    # zero spread: a robust scale can be 0 when >50% of values
                    # are identical (IQR=0) even though std>0. Fall back to std,
                    # then to leaving the feature centred-only, never dividing
                    # by zero.
                    s = d.get("std", 0.0)
                    if not s or s <= 0.0:
                        self.center[g][j] = c
                        self.do[g][j] = True     # centre, unit scale
                        continue
                self.center[g][j] = c
                self.scale[g][j] = s
                self.do[g][j] = True

        # group id per node type, as an array for fast lookup
        self.group_id = {g: i for i, g in enumerate(groups)}
        self.groups = groups
        self.type_to_group = {t: self.group_of[t] for t in self.group_of}

        # Per NODE TYPE, which columns to actually transform: a column is
        # transformed on a node only if (a) it is standardisable in that node's
        # group AND (b) the feature is applicable to that specific node type.
        # This second condition matters because SV and SV3pi share the "SV"
        # group, but the candidate block is applicable only to SV3pi -- without
        # it, standardising would centre cand_* structural zeros on plain SV
        # nodes and turn them into a nonzero constant.
        name_to_type = {}  # not needed; build the mask directly
        self.do_type = {}
        for t, g in self.type_to_group.items():
            col = np.zeros(self.nfeat, bool)
            for j, f in enumerate(self.feature_names):
                if self.do[g][j] and (t in self.applicable.get(f, set())):
                    col[j] = True
            self.do_type[t] = col

    def apply_batch(self, feats, node_type, node_mask):
        """Standardise a whole (N, MAX_NODES, F) block in place, vectorised.

        Equivalent to calling apply() per event, but done once at load time.
        This is the single biggest throughput win in the pipeline: the per-event
        version ran ~150 small fancy-indexing operations per event, and at 570k
        events per epoch that dominated everything else, including edge
        construction. Standardisation is safe to precompute because it depends
        only on each node's TYPE, which the per-event node shuffle preserves.
        """
        for t, g in self.type_to_group.items():
            sel = node_mask & (node_type == t)          # (N, MAX_NODES) bool
            if not sel.any():
                continue
            do = self.do_type[t]
            if not do.any():
                continue
            cols = np.where(do)[0]
            block = feats[sel][:, cols]                  # (M, n_selected_cols)
            block = (block - self.center[g][cols]) / self.scale[g][cols]
            np.clip(block, -self.clip, self.clip, out=block)
            # scatter back: index the boolean rows, then the column subset
            rows = np.where(sel)
            tmp = feats[rows[0], rows[1]]                # (M, F)
            tmp[:, cols] = block
            feats[rows[0], rows[1]] = tmp
        return feats

    def apply(self, feats, node_type, node_mask):
        """feats (N, F) float32 for ONE event -> standardised copy.

        Only real nodes are touched; padding stays exactly zero. Within a real
        node, only columns applicable to that node's type and marked for
        standardisation are transformed -- structural zeros and flags are left
        as-is.
        """
        out = feats.copy()
        for t, g in self.type_to_group.items():
            rows = np.where(node_mask & (node_type == t))[0]
            if rows.size == 0:
                continue
            do = self.do_type[t]        # group-standardisable AND type-applicable
            if not do.any():
                continue
            c = self.center[g]
            s = self.scale[g]
            block = out[rows][:, do]
            block = (block - c[do]) / s[do]
            np.clip(block, -self.clip, self.clip, out=block)
            out[np.ix_(rows, np.where(do)[0])] = block
        # padding untouched (was zero, stays zero)
        return out


# --------------------------------------------------------------------------
# edge features
# --------------------------------------------------------------------------
def build_edges(aux, node_type, node_mask, sig_hemi):
    """(N, N, N_EDGE_FEATS) float32 edge tensor for one event.

    aux      (N, 3)  raw displacement-from-PV vector per node (0 for PV/EVT/pad)
    sig_hemi (N,)    is_signal_hemisphere flag per node

    Fully vectorised: every edge quantity is a pairwise operation over whole
    arrays, so there is no Python loop over the ~150 node pairs. This matters --
    at 570k events per epoch the interpreted version starves the GPU.

    Agreement with the reference loop is to 1-2 ULP in float32 (<= 2.4e-7), not
    bit-exact: features 2 and 4 are dot products, and scalar np.dot on a
    3-vector accumulates differently from any vectorised formulation (matmul,
    einsum and explicit multiply-then-sum all differ from it, and from each
    other, at the same magnitude). The vectorised form is if anything the more
    accurate one. Checked by test_edges_equivalence(), and shown not to move
    model outputs in test_integration.py.

    Padded rows/cols and the diagonal are left at zero. The attention mask
    already ignores padding, but zeroing keeps it out of anything that sums or
    averages over edges.
    """
    N = aux.shape[0]
    E = np.zeros((N, N, N_EDGE_FEATS), np.float32)
    if node_mask.sum() < 2:
        return E

    d2pv = np.linalg.norm(aux, axis=1)                        # (N,)
    diff = aux[:, None, :] - aux[None, :, :]                   # (N,N,3) a - b
    sep = np.linalg.norm(diff, axis=2)                         # (N,N)
    unit = aux / np.maximum(d2pv, 1e-9)[:, None]               # (N,3)
    hemi = sig_hemi > 0.5
    is_evt = (node_type == NODE_EVT)

    # 0: 3D separation
    E[:, :, 0] = np.log1p(sep)
    # 1: signed radial ordering, which vertex is further downstream
    E[:, :, 1] = d2pv[:, None] - d2pv[None, :]
    # 2: alignment of the two displacement directions
    E[:, :, 2] = unit @ unit.T
    # 3: both in the signal hemisphere
    E[:, :, 3] = (hemi[:, None] & hemi[None, :]).astype(np.float32)
    # 4: collinearity of (b - a) with (a - PV) -- is a on the PV->b path.
    #    ab = -diff. The vector is normalised BEFORE the dot product, matching
    #    the reference loop's operation order exactly; dotting first and
    #    dividing after is mathematically identical but differs by ~1 ULP in
    #    float32, and exact equivalence is worth more than saving one divide.
    #    Guarded as the reference was: skip coincident vertices, and skip when a
    #    sits at the PV (no displacement direction to compare against).
    ok = (sep > 1e-9) & (d2pv[:, None] > 1e-9)
    ab_unit = np.zeros_like(diff)
    np.divide(-diff, sep[:, :, None], out=ab_unit,
              where=ok[:, :, None])
    E[:, :, 4] = np.einsum("abd,ad->ab", ab_unit, unit)
    E[:, :, 4][~ok] = 0.0
    # 5: either endpoint is the global EVT node
    E[:, :, 5] = (is_evt[:, None] | is_evt[None, :]).astype(np.float32)

    # keep only real-real off-diagonal pairs
    pair = node_mask[:, None] & node_mask[None, :]
    np.fill_diagonal(pair, False)
    E *= pair[:, :, None]
    return E


def test_edges_equivalence(n_trials=500, tol=1e-6, verbose=True):
    """Vectorised build_edges vs the reference loop, incl. degenerate geometry.

    Covers coincident vertices, vertices sitting exactly at the PV, and
    single-real-node events, since those are what the guards in feature 4 exist
    for. Tolerance is 1e-6: the true disagreement is <= 2.4e-7 (1-2 float32
    ULP, from dot-product accumulation order), and anything materially larger
    would mean a genuine logic difference.
    """
    rng = np.random.default_rng(0)
    worst = np.zeros(N_EDGE_FEATS)
    for trial in range(n_trials):
        N = 13
        nv = int(rng.integers(1, 12))
        mask = np.zeros(N, bool)
        mask[:nv + 1] = True
        aux = np.zeros((N, 3), np.float32)
        if nv > 1:
            aux[1:nv] = rng.normal(0, 2, (nv - 1, 3))
        if trial % 7 == 0 and nv > 3:      # coincident vertices
            aux[2] = aux[1]
        if trial % 11 == 0 and nv > 2:     # a vertex exactly at the PV
            aux[1] = 0.0
        ntype = np.zeros(N, np.int64)
        ntype[0] = NODE_PV
        ntype[1:nv] = NODE_SV
        ntype[nv] = NODE_EVT
        sh = (rng.random(N) > 0.5).astype(np.float32)

        fast = build_edges(aux, ntype, mask, sh)
        slow = _build_edges_reference(aux, ntype, mask, sh)
        assert fast.shape == slow.shape
        for f in range(N_EDGE_FEATS):
            worst[f] = max(worst[f], float(np.abs(fast[:, :, f] - slow[:, :, f]).max()))

    if verbose:
        for f in range(N_EDGE_FEATS):
            print(f"    edge feature {f}: max |vec - loop| = {worst[f]:.3e}")
    bad = [f for f in range(N_EDGE_FEATS) if worst[f] > tol]
    assert not bad, f"edge features {bad} disagree beyond {tol}: {worst}"
    return worst


def _build_edges_reference(aux, node_type, node_mask, sig_hemi):
    """Original explicit-loop implementation, kept only as a test oracle."""
    N = aux.shape[0]
    E = np.zeros((N, N, N_EDGE_FEATS), np.float32)
    real = np.where(node_mask)[0]
    if real.size < 2:
        return E
    d2pv = np.linalg.norm(aux, axis=1)
    sep = np.linalg.norm(aux[:, None, :] - aux[None, :, :], axis=2)
    is_evt = (node_type == NODE_EVT)
    unit = aux / np.maximum(d2pv, 1e-9)[:, None]
    for a in real:
        for b in real:
            if a == b:
                continue
            E[a, b, 0] = np.log1p(sep[a, b])
            E[a, b, 1] = d2pv[a] - d2pv[b]
            E[a, b, 2] = float(np.dot(unit[a], unit[b]))
            E[a, b, 3] = 1.0 if (sig_hemi[a] > 0.5 and sig_hemi[b] > 0.5) else 0.0
            ab = aux[b] - aux[a]
            nab = np.linalg.norm(ab)
            if nab > 1e-9 and d2pv[a] > 1e-9:
                E[a, b, 4] = float(np.dot(ab / nab, unit[a]))
            E[a, b, 5] = 1.0 if (is_evt[a] or is_evt[b]) else 0.0
    return E


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class GraphDataset(Dataset):
    """Events from one split, standardised and augmented on access.

    All shards for the split are loaded into contiguous arrays at construction
    and standardised once, so the only per-event work is the node shuffle and
    edge construction. Resident size is ~1.5 GB for the 570k train split.
    """

    def __init__(self, stats_path, split="train", method="robust", clip=5.0,
                 shuffle_nodes=True, channels=None, seed=0, verbose=True,
                 mask_features=None, evt_mode="node"):
        with open(stats_path) as fh:
            self.stats = json.load(fh)
        self.shard_dir = self.stats["shard_dir"]
        self.split = split
        self.shuffle_nodes = shuffle_nodes and (split == "train")
        self.std = Standardizer(self.stats, method=method, clip=clip)
        self.feature_names = self.std.feature_names
        self.max_nodes = int(self.stats["max_nodes"])
        self._rng = np.random.default_rng(seed)
        # cached once: this was a list.index() scan on every __getitem__
        self._sig_col = self.feature_names.index("is_signal_hemisphere")

        chans = list(self.stats["split"].keys())
        if channels is not None:
            chans = [c for c in chans if c in channels]
        self.channel_names = chans

        # ---- load every shard for this split into contiguous arrays --------
        # One concatenated block rather than a per-shard cache. Same total
        # memory (~1.5 GB for the 570k train split), but contiguous, free of
        # per-item dict lookups and tensor->numpy conversions, and shareable
        # with DataLoader workers through fork copy-on-write instead of being
        # duplicated once per worker.
        F, A, M, T, R, C, K, Y, CH, V = [], [], [], [], [], [], [], [], [], []
        for ci, ch in enumerate(chans):
            for rel in self.stats["split"][ch][split]:
                sh = torch.load(os.path.join(self.shard_dir, rel),
                                map_location="cpu", weights_only=False)
                F.append(sh["node_feats"].numpy().astype(np.float32))
                A.append(sh["node_aux"].numpy().astype(np.float32))
                M.append(sh["node_mask"].numpy().astype(bool))
                T.append(sh["node_type"].numpy().astype(np.int8))
                R.append(sh["role"].numpy().astype(np.int8))
                C.append(sh["context"].numpy().astype(np.int32))
                K.append(sh["ctx_cat"].numpy().astype(np.int8))
                Y.append(sh["y"].numpy().astype(np.int8))
                V.append(sh["evt_mva1"].numpy().astype(np.float32))
                CH.append(np.full(sh["y"].shape[0], ci, np.int16))
                del sh
        if not F:
            raise RuntimeError(f"no shards found for split '{split}'")

        self.feats = np.concatenate(F);  del F
        self.aux = np.concatenate(A);    del A
        self.mask = np.concatenate(M);   del M
        self.ntype = np.concatenate(T);  del T
        self.role = np.concatenate(R);   del R
        self.ctx = np.concatenate(C);    del C
        self.ctx_cat = np.concatenate(K)
        self.y = np.concatenate(Y)
        self.mva1 = np.concatenate(V)
        self.event_channel = np.concatenate(CH)
        self.n_events = self.feats.shape[0]

        # unstandardised hemisphere flag, kept for edge construction
        self.sig_hemi = self.feats[:, :, self._sig_col].copy()

        # ANALYSIS-ONLY: which nodes were 3pi candidates, recorded from the
        # original node type BEFORE any masking collapses it. Without this the
        # negative control (tau vs charm among candidate vertices) could not be
        # computed on a masked run at all.
        self.is_cand = (self.ntype == 3)

        # ---- standardise once, for the whole split -------------------------
        self.std.apply_batch(self.feats, self.ntype.astype(np.int64), self.mask)

        # ---- optional feature masking --------------------------------------
        # Applied AFTER standardisation so masked columns are exactly zero;
        # masking first would let the standardiser centre them onto a nonzero
        # constant, which is not masking at all.
        self.mask_features = None
        self.masked_cols = []
        if mask_features:
            self.mask_features = list(mask_features)
            self.masked_cols = self._resolve_mask(self.mask_features)
            self.feats[:, :, self.masked_cols] = 0.0

            # CRITICAL: node_type distinguishes SV (2) from SV3pi (3), and the
            # model embeds it. Zeroing the candidate feature columns while
            # leaving the type embedding intact would leave the router perfectly
            # able to identify candidates -- the same leak as the zero-pattern
            # of the candidate block, one level down. So if candidate identity
            # is being masked, the node type must be collapsed too.
            cand_block = set(self.stats["feature_blocks"].get("cand3pi", []))
            cand_block.add("is_3pi_candidate")
            masked_names = {self.feature_names[c] for c in self.masked_cols}
            self.collapsed_sv3pi = bool(masked_names & cand_block)
            if self.collapsed_sv3pi:
                n_before = int((self.ntype == 3).sum())
                self.ntype[self.ntype == 3] = 2       # SV3pi -> SV
                if verbose:
                    print(f"    [{split}] masked {len(self.masked_cols)} features "
                          f"and collapsed {n_before} SV3pi nodes to SV "
                          f"(type embedding would otherwise leak the flag)",
                          flush=True)
            elif verbose:
                print(f"    [{split}] masked {len(self.masked_cols)} features",
                      flush=True)
        else:
            self.collapsed_sv3pi = False

        # ---- EVT node handling ---------------------------------------------
        # Three ways the event-level scalars can reach the model:
        #   node : the original design. The EVT node sits in the graph, takes
        #          part in attention both ways, and is one term in the pooled
        #          mean -- so its share of the readout is 1/(n_vertices+1),
        #          i.e. ~1/4 in a 3-vertex event but ~1/13 in a 12-vertex one.
        #   pool : EVT stays in the graph and in attention, but is removed from
        #          the pooled mean and handed to the head separately, so its
        #          weight in the readout no longer depends on multiplicity.
        #          Isolates the pooling quirk from everything else.
        #   mlp  : EVT is removed from the graph entirely (masked out, so it is
        #          never attended to or from, and edge feature 5 goes
        #          identically zero) and only reaches the classifier through
        #          the head. Isolates whether EVT belongs in attention at all.
        #
        # The scalars are lifted out AFTER standardise() so they carry exactly
        # the statistics already recorded in stats.json for the EVT node-type
        # group. Recomputing them here would risk a second, inconsistent
        # normalisation and a train/test leak.
        if evt_mode not in ("node", "pool", "mlp"):
            raise ValueError(f"evt_mode must be node|pool|mlp, got {evt_mode!r}")
        self.evt_mode = evt_mode
        self.evt_cols = [self.feature_names.index(f)
                         for f in self.stats["feature_blocks"]["event"]]
        self.n_global = len(self.evt_cols) if evt_mode != "node" else 0

        if evt_mode != "node":
            is_evt = (self.ntype == NODE_EVT) & self.mask          # (N, MAX)
            # exactly one EVT node per event by construction; assert it, since
            # silently taking the first would hide a shard-building bug
            n_per_event = is_evt.sum(axis=1)
            if not (n_per_event == 1).all():
                bad = int((n_per_event != 1).sum())
                raise RuntimeError(
                    f"[{split}] {bad} events do not have exactly one EVT node; "
                    f"evt_mode={evt_mode} assumes one per event")
            rows = np.argmax(is_evt, axis=1)                        # (N,)
            self.evt_scalars = self.feats[np.arange(self.n_events), rows][
                :, self.evt_cols].copy()                            # (N, 21)
            if evt_mode == "mlp":
                # Drop it from the graph. Zero the row as well as unsetting the
                # mask: the mask alone is enough for the model, but leaving
                # stale values in a masked slot makes every later assertion
                # about padding being zero harder to trust.
                self.mask[np.arange(self.n_events), rows] = False
                self.feats[np.arange(self.n_events), rows] = 0.0
                self.aux[np.arange(self.n_events), rows] = 0.0
                self.ntype[np.arange(self.n_events), rows] = NODE_PAD
                self.role[np.arange(self.n_events), rows] = ROLE_NOT_A_VERTEX
            if verbose:
                what = ("kept in attention, split out of the pooled mean"
                        if evt_mode == "pool" else
                        "removed from the graph, head-only")
                print(f"    [{split}] evt_mode={evt_mode}: {self.n_global} "
                      f"event scalars {what}", flush=True)
        else:
            self.evt_scalars = None

        if verbose:
            gb = (self.feats.nbytes + self.aux.nbytes) / 1e9
            print(f"    [{split}] {self.n_events} events from "
                  f"{len(chans)} channels, standardised, {gb:.2f} GB resident",
                  flush=True)

    def _resolve_mask(self, spec):
        """Resolve names of features and/or feature blocks to column indices."""
        blocks = self.stats["feature_blocks"]
        cols = set()
        for item in spec:
            if item in blocks:                       # a whole block
                for f in blocks[item]:
                    cols.add(self.feature_names.index(f))
            elif item in self.feature_names:         # a single feature
                cols.add(self.feature_names.index(item))
            else:
                raise ValueError(
                    f"--mask_features: '{item}' is neither a feature nor a "
                    f"block. Blocks: {sorted(blocks)}")
        return sorted(cols)

    def __len__(self):
        return self.n_events

    def channel_name_of(self, k):
        return self.channel_names[int(self.event_channel[k])]

    def __getitem__(self, k):
        feats = self.feats[k]
        aux = self.aux[k]
        ntype = self.ntype[k].astype(np.int64)
        nmask = self.mask[k]
        role = self.role[k].astype(np.int64)
        ctx = self.ctx[k].astype(np.int64)
        ctx_cat = self.ctx_cat[k].astype(np.int64)
        sig_hemi = self.sig_hemi[k]
        is_cand = self.is_cand[k]

        # shuffle real-node order (train only). One permutation applied
        # consistently to every per-node array, so the graph is identical up to
        # relabelling; edges are built afterwards so they inherit it for free.
        if self.shuffle_nodes:
            order = np.arange(self.max_nodes)
            real = np.where(nmask)[0]
            order[:real.size] = self._rng.permutation(real)
            feats = feats[order]
            aux = aux[order]
            ntype = ntype[order]
            nmask = nmask[order]
            role = role[order]
            ctx = ctx[order]
            ctx_cat = ctx_cat[order]
            sig_hemi = sig_hemi[order]
            is_cand = is_cand[order]
        else:
            feats = feats.copy()

        # (3) edges, built AFTER the shuffle so indices already agree
        edges = build_edges(aux, ntype, nmask, sig_hemi)

        return {
            "node_feats": torch.from_numpy(feats),
            "node_mask": torch.from_numpy(nmask),
            "node_type": torch.from_numpy(ntype.astype(np.int64)),
            "edge_feats": torch.from_numpy(edges),
            "y": int(self.y[k]),
            # eval-only truth; kept in a clearly separate set of keys
            "role": torch.from_numpy(role.astype(np.int64)),
            "context": torch.from_numpy(ctx.astype(np.int64)),
            "ctx_cat": torch.from_numpy(ctx_cat.astype(np.int64)),
            # analysis-only: recorded before any mask collapsed the node type,
            # so the negative control stays computable on masked runs
            "is_cand": torch.from_numpy(is_cand.astype(np.int64)),
            "channel_id": int(self.event_channel[k]),
            # BENCHMARK ONLY. EVT_MVA1 is the existing BDT1 score, carried
            # through so its AUC can be quoted on exactly the same events as
            # the model's. It must never be used as a model input -- that would
            # be training on the output of the classifier we are comparing to.
            "evt_mva1": float(self.mva1[k]),
            # MODEL INPUT when evt_mode != "node": the standardised event-level
            # scalars, routed to the head instead of (or as well as) travelling
            # through the graph. Empty in the default mode so the batch dict
            # keeps a stable shape either way.
            "evt_scalars": (torch.from_numpy(self.evt_scalars[k])
                            if self.evt_scalars is not None
                            else torch.zeros(0, dtype=torch.float32)),
        }


# --------------------------------------------------------------------------
# collate + loaders
# --------------------------------------------------------------------------
def collate(batch):
    """Stack per-event dicts into batched tensors. All events already share
    MAX_NODES, so this is a plain stack -- no ragged handling needed."""
    out = {
        "node_feats": torch.stack([b["node_feats"] for b in batch]),
        "node_mask": torch.stack([b["node_mask"] for b in batch]),
        "node_type": torch.stack([b["node_type"] for b in batch]),
        "edge_feats": torch.stack([b["edge_feats"] for b in batch]),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "role": torch.stack([b["role"] for b in batch]),
        "context": torch.stack([b["context"] for b in batch]),
        "ctx_cat": torch.stack([b["ctx_cat"] for b in batch]),
        "is_cand": torch.stack([b["is_cand"] for b in batch]),
        "channel_id": torch.tensor([b["channel_id"] for b in batch], dtype=torch.long),
        "evt_mva1": torch.tensor([b["evt_mva1"] for b in batch],
                                 dtype=torch.float32),
        "evt_scalars": torch.stack([b["evt_scalars"] for b in batch]),
    }
    return out


# Physical Z->bb composition is dominated by non-signal; the hard-negative 3pi
# hadronic modes are individually rare. These relative weights are a starting
# point for a mixture sampler, to be tuned as a systematic -- NOT physics truth.
MIXTURES = {
    "as_built": None,                      # use shards as they are
    "balanced": {"Bc2TauNu": 1.0, "Bu2TauNu": 1.0, "Zbb_incl": 1.0,
                 "Zcc_incl": 1.0, "Bd2D3Pi": 1.0, "Bu2D03Pi": 1.0,
                 "Bs2Ds3Pi": 1.0, "Lb2Lc3Pi": 1.0},
}


def make_sampler(ds, mixture):
    """WeightedRandomSampler giving each channel a target sampling weight.

    Weights are per-event = target_channel_weight / n_events_in_channel, so a
    channel's expected draws are proportional to its target weight regardless
    of how many events it contributed.
    """
    spec = MIXTURES.get(mixture, mixture) if isinstance(mixture, str) else mixture
    if spec is None:
        return None
    # count events per channel in this split
    names = np.array(ds.channel_names)
    ch_idx = ds.event_channel
    counts = np.bincount(ch_idx, minlength=len(names)).astype(np.float64)
    per_channel = np.array([spec.get(n, 0.0) for n in names], np.float64)
    w = np.where(counts[ch_idx] > 0,
                 per_channel[ch_idx] / np.maximum(counts[ch_idx], 1.0), 0.0)
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(ds),
                                 replacement=True)


def make_loader(ds, batch_size=256, shuffle=True, mixture="as_built",
                num_workers=0, drop_last=False):
    sampler = make_sampler(ds, mixture)
    if sampler is not None:
        shuffle = False       # sampler and shuffle are mutually exclusive
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, collate_fn=collate,
                      num_workers=num_workers, drop_last=drop_last)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _selftest(stats_path):
    print("=== dataset self-test ===")

    print("  vectorised build_edges vs reference loop:")
    test_edges_equivalence(n_trials=300)

    ds = GraphDataset(stats_path, split="train", shuffle_nodes=True)
    print(f"train events: {len(ds)}")
    F = len(ds.feature_names)

    b = ds[0]
    assert b["node_feats"].shape == (ds.max_nodes, F), b["node_feats"].shape
    assert b["edge_feats"].shape == (ds.max_nodes, ds.max_nodes, N_EDGE_FEATS)
    assert b["node_mask"].dtype == torch.bool
    # padding stays exactly zero
    pad = ~b["node_mask"].numpy()
    assert not b["node_feats"].numpy()[pad].any(), "padding not zero after standardise"
    print("  shapes and padding ok")

    # shuffle actually permutes: two draws of the same event should (almost
    # always) differ in node order but describe the same multiset of nodes
    orders = set()
    for _ in range(8):
        bb = ds[0]
        types = tuple(bb["node_type"][bb["node_mask"]].tolist())
        orders.add(types)
    assert len(orders) > 1, "node shuffle is not permuting"
    print(f"  node shuffle permutes ({len(orders)} orderings in 8 draws)")

    # standardised spread ~1 on train, per group, for a heavy feature
    loader = make_loader(ds, batch_size=512, shuffle=True, mixture="as_built")
    batch = next(iter(loader))
    feats = batch["node_feats"].numpy()
    ntype = batch["node_type"].numpy()
    nmask = batch["node_mask"].numpy()
    j = ds.feature_names.index("log1p_d2PV_significance")
    sv = (ntype == NODE_SV) | (ntype == NODE_SV3PI)
    vals = feats[:, :, j][sv & nmask]
    print(f"  log1p_d2PV_significance on SV: mean {vals.mean():+.3f} "
          f"std {vals.std():.3f} (expect ~0, ~1; clipped at 5)")
    assert abs(vals.mean()) < 0.3 and 0.5 < vals.std() < 1.5

    # a flag must be untouched (still 0/1)
    jf = ds.feature_names.index("is_3pi_candidate")
    fv = feats[:, :, jf][nmask]
    assert set(np.unique(fv).tolist()) <= {0.0, 1.0}, "flag was standardised"
    print("  flags left untouched")

    # candidate block: standardised only where is_3pi_candidate, still 0 elsewhere
    jc = ds.feature_names.index("cand_m3pi")
    is_cand = feats[:, :, jf] > 0.5
    non_cand_vals = feats[:, :, jc][nmask & ~is_cand]
    assert not non_cand_vals.any(), "candidate feature nonzero on non-candidate node"
    print("  candidate block stays zero off-candidate")

    # edge sanity: symmetric #0, antisymmetric #1
    e = batch["edge_feats"].numpy()[0]
    m = batch["node_mask"].numpy()[0]
    idx = np.where(m)[0]
    if len(idx) >= 2:
        a, c = idx[0], idx[1]
        assert abs(e[a, c, 0] - e[c, a, 0]) < 1e-5, "sep not symmetric"
        assert abs(e[a, c, 1] + e[c, a, 1]) < 1e-5, "d2PV diff not antisymmetric"
    print("  edge symmetry ok")

    # eval-only truth present, and roles look sane (PV role present per event)
    roles = batch["role"].numpy()[nmask]
    print(f"  role values present: {sorted(set(roles.tolist()))}")

    # mixture sampler runs
    ml = make_loader(ds, batch_size=256, mixture="balanced")
    _ = next(iter(ml))
    print("  balanced mixture sampler ok")

    # val split has shuffle disabled
    vds = GraphDataset(stats_path, split="val", shuffle_nodes=True)
    o = set()
    for _ in range(5):
        o.add(tuple(vds[0]["node_type"][vds[0]["node_mask"]].tolist()))
    assert len(o) == 1, "val split should not shuffle nodes"
    print(f"  val split deterministic (no shuffle); val events: {len(vds)}")

    print("\nALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description="Graph dataset / loader.")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.stats)
    else:
        ds = GraphDataset(args.stats, split="train")
        print(f"train events: {len(ds)}   features: {len(ds.feature_names)}   "
              f"max_nodes: {ds.max_nodes}")


if __name__ == "__main__":
    main()
