#!/usr/bin/env python3

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASSES = 3
N_NODE_TYPES = 5          # pad, PV, SV, SV3pi, EVT  (graph_build.NODE_*)
N_EDGE_FEATS = 6
NODE_EVT = 4              # must match graph_build.NODE_EVT


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------
class MaskedEdgeAttention(nn.Module):
    """Multi-head self-attention with an additive per-head edge bias and
    key-side padding mask."""

    def __init__(self, d_model, n_heads, edge_dim=N_EDGE_FEATS, dropout=0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # edge features -> one bias scalar per head
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 2 * n_heads),
            nn.GELU(),
            nn.Linear(2 * n_heads, n_heads),
        )

    def forward(self, x, edge_feats, mask, return_attn=False):
        """x (B,N,d); edge_feats (B,N,N,E); mask (B,N) bool, True = real."""
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)   # (B,h,N,dk)
        k = self.k(x).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.dk).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)     # (B,h,N,N)

        # edge bias, (B,N,N,h) -> (B,h,N,N)
        logits = logits + self.edge_mlp(edge_feats).permute(0, 3, 1, 2)

        # mask keys: a padded node is never attended TO
        key_mask = mask[:, None, None, :]                            # (B,1,1,N)
        logits = logits.masked_fill(~key_mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        # A padded QUERY row has all-(-inf) logits -> softmax gives NaN. Those
        # rows are discarded downstream by the pooling mask, but a NaN in the
        # residual stream would spread to real nodes through the next layer's
        # LayerNorm statistics, so zero them explicitly.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.o(out)
        out = out * mask[..., None]          # zero padded query rows
        return (out, attn) if return_attn else (out, None)


# --------------------------------------------------------------------------
# mixture of experts
# --------------------------------------------------------------------------
class MoE(nn.Module):
    """Noisy top-k gated mixture of MLP experts, routed per node."""

    def __init__(self, d_model, d_ff, n_experts=8, k=2, dropout=0.0,
                 noise_eps=1e-2):
        super().__init__()
        if k > n_experts:
            raise ValueError(f"k={k} > n_experts={n_experts}")
        self.n_experts = n_experts
        self.k = k
        self.noise_eps = noise_eps
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        self.w_noise = nn.Linear(d_model, n_experts, bias=False)
        # experts are a batched pair of linear maps; equivalent to n separate
        # 2-layer MLPs but computed without a Python loop over experts
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        for w in (self.w1, self.w2):
            for e in range(n_experts):
                nn.init.xavier_uniform_(w[e])
        self.drop = nn.Dropout(dropout)
        # zero-init the noise head so routing starts noise-free and symmetric
        nn.init.zeros_(self.w_noise.weight)

    def forward(self, x, mask):
        """x (B,N,d); mask (B,N) bool. Returns (out, aux_loss, routing)."""
        B, N, d = x.shape
        flat = x.reshape(-1, d)                      # (BN, d)
        m = mask.reshape(-1)                         # (BN,)

        clean = self.w_gate(flat)                    # (BN, n_experts)
        if self.training:
            # exploration noise, scaled per-expert by a learned magnitude
            sigma = F.softplus(self.w_noise(flat)) + self.noise_eps
            logits = clean + torch.randn_like(clean) * sigma
        else:
            logits = clean

        # padding is routed nowhere: force it to a uniform, ignored gate so it
        # can never enter the top-k statistics or the balance terms
        logits = logits.masked_fill(~m[:, None], 0.0)

        topv, topi = logits.topk(self.k, dim=-1)      # (BN, k)
        gates = torch.softmax(topv, dim=-1)           # renormalised over the k
        gates = gates * m[:, None]                    # zero for padding

        # dense expert application. n_experts * d_ff is small here (8 * 4d), so
        # computing all experts and combining is faster and simpler than a
        # scatter/gather dispatch, and keeps the whole thing differentiable.
        hidden = torch.einsum("nd,edf->nef", flat, self.w1) + self.b1
        hidden = F.gelu(hidden)
        outs = torch.einsum("nef,efd->ned", hidden, self.w2) + self.b2  # (BN,e,d)
        outs = self.drop(outs)

        # combine only the selected k experts
        sel = torch.zeros(flat.shape[0], self.n_experts, device=x.device,
                          dtype=gates.dtype)
        sel.scatter_(1, topi, gates)                  # (BN, n_experts)
        out = torch.einsum("ne,ned->nd", sel, outs)
        out = out * m[:, None]
        out = out.view(B, N, d)

        aux = self._balance_loss(clean, topi, m)
        routing = {"top_idx": topi.view(B, N, self.k).detach(),
                   "top_gate": gates.view(B, N, self.k).detach(),
                   "mask": mask.detach()}
        return out, aux, routing

    def _balance_loss(self, clean_logits, topi, m):
        """CV^2 of expert importance + CV^2 of expert load, over real nodes."""
        if m.sum() == 0:
            return clean_logits.sum() * 0.0
        probs = torch.softmax(clean_logits, dim=-1) * m[:, None]
        importance = probs.sum(0)                     # (n_experts,) differentiable
        # load: how many real nodes put each expert in their top-k
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, topi, 1.0)
        load = (onehot * m[:, None]).sum(0)

        def cv2(t):
            mu = t.mean()
            if mu.abs() < 1e-12:
                return t.sum() * 0.0
            return (t.var(unbiased=False) / (mu ** 2))

        return cv2(importance) + cv2(load)


# --------------------------------------------------------------------------
# block
# --------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm attention + pre-norm MoE, both residual."""

    def __init__(self, d_model, n_heads, d_ff, n_experts, k, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = MaskedEdgeAttention(d_model, n_heads, dropout=dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, d_ff, n_experts=n_experts, k=k, dropout=dropout)

    def forward(self, x, edge_feats, mask, return_attn=False):
        a, attn = self.attn(self.n1(x), edge_feats, mask, return_attn=return_attn)
        x = x + a
        m, aux, routing = self.moe(self.n2(x), mask)
        x = x + m
        x = x * mask[..., None]      # keep padded rows identically zero
        return x, aux, routing, attn


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------
class MoEGraphTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=None, n_experts=8, k=2, dropout=0.1,
                 n_classes=N_CLASSES, use_moe=True, n_global=0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.n_features = n_features
        self.n_experts = n_experts
        self.k = k
        self.use_moe = use_moe
        # n_global > 0 routes that many event-level scalars straight to the
        # classifier head instead of through the graph. Zero by default, so the
        # EVT node keeps travelling through attention and pooling exactly as
        # before and old commands reproduce bit-for-bit.
        self.n_global = int(n_global)

        self.in_proj = nn.Linear(n_features, d_model)
        self.type_emb = nn.Embedding(N_NODE_TYPES, d_model)

        if use_moe:
            self.blocks = nn.ModuleList([
                Block(d_model, n_heads, d_ff, n_experts, k, dropout)
                for _ in range(n_layers)])
        else:
            # plain-FFN ablation: identical everywhere except the MoE is a
            # single expert, so "does the MoE cost accuracy" is answerable
            self.blocks = nn.ModuleList([
                Block(d_model, n_heads, d_ff, 1, 1, dropout)
                for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)
        # The globals are projected to d_model before being concatenated rather
        # than pasted on raw, so the two halves of the head's input arrive on a
        # comparable scale and the head does not have to learn to rescale 21
        # standardised scalars against a LayerNormed pooled vector.
        if self.n_global:
            self.global_proj = nn.Sequential(
                nn.Linear(self.n_global, d_model), nn.GELU(),
                nn.LayerNorm(d_model))
            head_in = 2 * d_model
        else:
            self.global_proj = None
            head_in = d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

        self._record = False
        self._routing = None
        self._attn = None

    # -- routing capture -------------------------------------------------
    def record_routing(self, on=True):
        self._record = bool(on)
        return self

    def get_routing(self):
        """list over layers of {top_idx (B,N,k), top_gate, mask}."""
        return self._routing

    def get_attention(self):
        """list over layers of (B, heads, N, N), only if recording."""
        return self._attn

    def forward(self, node_feats, node_mask, node_type, edge_feats,
                evt_scalars=None):
        h = self.in_proj(node_feats) + self.type_emb(node_type)
        h = h * node_mask[..., None]

        aux_total = h.new_zeros(())
        routing, attns = [], []
        for blk in self.blocks:
            h, aux, r, a = blk(h, edge_feats, node_mask,
                               return_attn=self._record)
            aux_total = aux_total + aux
            if self._record:
                routing.append(r)
                attns.append(a)

        h = self.norm(h) * node_mask[..., None]
        # masked mean over REAL nodes only; dividing by N would make the pooled
        # representation depend on how many padding slots an event happens to
        # have, i.e. leak the vertex count through the denominator
        # When the globals reach the head directly, the EVT node must come OUT
        # of the pooled mean -- otherwise the event-level information is counted
        # twice, and the whole point of the change (removing the
        # 1/(n_vertices+1) multiplicity weighting) is defeated. In "mlp" mode
        # the dataset has already unset its mask, so this is a no-op there; in
        # "pool" mode the node is still in the graph and in attention, and this
        # is the line that takes it out of the readout.
        pool_mask = node_mask
        if self.global_proj is not None:
            pool_mask = node_mask & (node_type != NODE_EVT)
        denom = pool_mask.sum(1, keepdim=True).clamp(min=1).to(h.dtype)
        pooled = (h * pool_mask[..., None]).sum(1) / denom
        if self.global_proj is not None:
            if evt_scalars is None or evt_scalars.shape[-1] != self.n_global:
                raise ValueError(
                    f"model built with n_global={self.n_global} needs "
                    f"evt_scalars of that width, got "
                    f"{None if evt_scalars is None else tuple(evt_scalars.shape)}")
            pooled = torch.cat([pooled, self.global_proj(evt_scalars)], dim=-1)
        logits = self.head(pooled)

        self._routing = routing if self._record else None
        self._attn = attns if self._record else None
        return logits, aux_total


def build_model(stats_or_nfeat, **kw):
    """Convenience: accepts either a feature count or a loaded stats dict."""
    if isinstance(stats_or_nfeat, dict):
        n = len(stats_or_nfeat["feature_names"])
    else:
        n = int(stats_or_nfeat)
    return MoEGraphTransformer(n, **kw)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _fake_batch(B=8, N=13, F_=50, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, N, F_, generator=g)
    mask = torch.zeros(B, N, dtype=torch.bool)
    ntype = torch.zeros(B, N, dtype=torch.long)
    for b in range(B):
        nv = int(torch.randint(3, N, (1,), generator=g))
        mask[b, :nv] = True
        ntype[b, 0] = 1                       # PV
        ntype[b, 1:nv - 1] = 2                # SV
        if nv > 3:
            ntype[b, 2] = 3                   # one SV3pi
        ntype[b, nv - 1] = 4                  # EVT
    feats = feats * mask[..., None]
    edges = torch.randn(B, N, N, N_EDGE_FEATS, generator=g)
    edges = edges * (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1)
    y = torch.randint(0, N_CLASSES, (B,), generator=g)
    return feats.to(device), mask.to(device), ntype.to(device), edges.to(device), y.to(device)


def _selftest():
    torch.manual_seed(0)
    print("=== model self-test ===")
    B, N, Fn = 8, 13, 50
    feats, mask, ntype, edges, y = _fake_batch(B, N, Fn)

    m = MoEGraphTransformer(Fn, d_model=64, n_heads=4, n_layers=2,
                            n_experts=8, k=2, dropout=0.0)
    npar = sum(p.numel() for p in m.parameters())
    print(f"  parameters: {npar:,}")

    logits, aux = m(feats, mask, ntype, edges)
    assert logits.shape == (B, N_CLASSES), logits.shape
    assert torch.isfinite(logits).all(), "non-finite logits"
    assert torch.isfinite(aux).all()
    print(f"  forward ok: logits {tuple(logits.shape)}, aux {aux.item():.4f}")

    # --- padding cannot influence real outputs -------------------------
    m.eval()
    with torch.no_grad():
        base, _ = m(feats, mask, ntype, edges)
        f2 = feats.clone()
        e2 = edges.clone()
        f2[~mask] = 999.0                              # garbage in padded rows
        pad_pair = ~(mask[:, :, None] & mask[:, None, :])
        e2[pad_pair] = -777.0
        alt, _ = m(f2, mask, ntype, e2)
    delta = (base - alt).abs().max().item()
    assert delta < 1e-5, f"padding leaked into output: max delta {delta}"
    print(f"  padding invariance ok (max delta {delta:.2e})")

    # --- permutation equivariance of the pooled prediction --------------
    # the dataset shuffles node order every draw, so the graph-level output
    # must not depend on it
    with torch.no_grad():
        pf, pm, pt, pe = feats.clone(), mask.clone(), ntype.clone(), edges.clone()
        for b in range(B):
            idx = torch.arange(N)
            real = idx[mask[b]]
            perm = real[torch.randperm(real.numel())]
            new = idx.clone()
            new[:real.numel()] = perm
            pf[b] = feats[b][new]
            pt[b] = ntype[b][new]
            pm[b] = mask[b][new]
            pe[b] = edges[b][new][:, new]
        permuted, _ = m(pf, pm, pt, pe)
    dperm = (base - permuted).abs().max().item()
    assert dperm < 1e-4, f"not permutation invariant: {dperm}"
    print(f"  permutation invariance ok (max delta {dperm:.2e})")

    # --- gradients reach everything -------------------------------------
    m.train()
    logits, aux = m(feats, mask, ntype, edges)
    loss = F.cross_entropy(logits, y) + 1.0 * aux
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"no/!finite grad for: {missing[:6]}"
    gn = sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None).sqrt()
    print(f"  gradients ok for all {sum(1 for _ in m.parameters())} tensors "
          f"(global norm {gn:.3f})")

    # --- routing capture -------------------------------------------------
    m.eval().record_routing(True)
    with torch.no_grad():
        m(feats, mask, ntype, edges)
    r = m.get_routing()
    assert len(r) == 2, f"expected 2 layers of routing, got {len(r)}"
    assert r[0]["top_idx"].shape == (B, N, 2), r[0]["top_idx"].shape
    # gates over the selected k sum to 1 on real nodes, 0 on padding
    gs = r[0]["top_gate"].sum(-1)
    assert torch.allclose(gs[mask], torch.ones_like(gs[mask]), atol=1e-5)
    assert gs[~mask].abs().max() < 1e-6, "padding received gate weight"
    used = torch.unique(r[0]["top_idx"][mask]).numel()
    print(f"  routing captured: layer0 uses {used}/8 experts on this batch, "
          f"gates normalised, padding unrouted")

    a = m.get_attention()
    assert a[0].shape == (B, 4, N, N)
    # attention rows over real queries sum to 1 and put no weight on padding
    arow = a[0].sum(-1)
    assert torch.allclose(arow[:, :, :][mask[:, None, :].expand(-1, 4, -1)],
                          torch.ones(1), atol=1e-4)
    pad_w = a[0].masked_select(~mask[:, None, None, :].expand(-1, 4, N, -1))
    assert pad_w.abs().max() < 1e-6, "attention placed weight on padding"
    print("  attention masked correctly (no weight on padded keys)")

    # --- eval determinism (noise must be off) ---------------------------
    with torch.no_grad():
        o1, _ = m(feats, mask, ntype, edges)
        o2, _ = m(feats, mask, ntype, edges)
    assert torch.equal(o1, o2), "eval mode is not deterministic"
    r1 = m.get_routing()[0]["top_idx"].clone()
    with torch.no_grad():
        m(feats, mask, ntype, edges)
    assert torch.equal(r1, m.get_routing()[0]["top_idx"]), "routing not deterministic"
    print("  eval deterministic (gate noise disabled, routing reproducible)")

    # --- train mode DOES vary (noise on) --------------------------------
    m.train()
    m.moe_noise_check = True
    outs = [m(feats, mask, ntype, edges)[0] for _ in range(2)]
    # with dropout=0 the only stochasticity is the gate noise; w_noise is
    # zero-initialised so noise = noise_eps > 0, enough to vary routing
    print(f"  train mode stochastic: max |d| = "
          f"{(outs[0] - outs[1]).abs().max().item():.2e}")

    # --- load-balance term responds to collapse -------------------------
    # Exercised on _balance_loss directly with constructed logits. Driving it
    # through the module by editing w_gate does not work: a weight row of
    # constants makes the logit proportional to sum(x), whose sign flips with
    # random input, and an all-zero gate produces exactly tied logits, where
    # topk deterministically returns 0..k-1 for every node -- an artefact of
    # tie-breaking rather than a property of the loss.
    mm = MoE(32, 128, n_experts=8, k=2).eval()
    nE, nTok = 8, 256
    g = torch.Generator().manual_seed(7)
    real = torch.ones(nTok, dtype=torch.bool)

    # collapsed: every node overwhelmingly prefers expert 0
    lg_col = torch.zeros(nTok, nE)
    lg_col[:, 0] = 12.0
    lg_col += 0.01 * torch.randn(nTok, nE, generator=g)   # break exact ties
    ti_col = lg_col.topk(2, dim=-1).indices
    aux_collapse = mm._balance_loss(lg_col, ti_col, real)

    # spread: independent random preferences per node
    lg_spr = torch.randn(nTok, nE, generator=g)
    ti_spr = lg_spr.topk(2, dim=-1).indices
    aux_spread = mm._balance_loss(lg_spr, ti_spr, real)

    print(f"  balance loss: collapsed {aux_collapse.item():.3f} vs "
          f"spread {aux_spread.item():.3f}")
    assert aux_collapse > aux_spread, "balance loss does not penalise collapse"
    assert aux_spread < 0.5, f"balanced routing should score near 0, got {aux_spread}"

    # padding must not enter the statistics: appending padded tokens that all
    # want expert 3 should leave the loss unchanged
    pad_lg = torch.cat([lg_spr, torch.zeros(64, nE)])
    pad_lg[nTok:, 3] = 20.0
    pad_ti = pad_lg.topk(2, dim=-1).indices
    pad_mask = torch.cat([real, torch.zeros(64, dtype=torch.bool)])
    aux_padded = mm._balance_loss(pad_lg, pad_ti, pad_mask)
    assert abs(aux_padded.item() - aux_spread.item()) < 1e-5, \
        f"padding changed the balance loss: {aux_padded.item()} vs {aux_spread.item()}"
    print("  balance loss ignores padded nodes")

    # --- expert-count scan constructs -----------------------------------
    for n_e in (6, 8, 10, 12):
        mk = MoEGraphTransformer(Fn, n_experts=n_e, k=2, d_model=32, n_heads=2)
        lg, ax = mk(feats, mask, ntype, edges)
        assert lg.shape == (B, N_CLASSES)
    print("  n_experts in {6,8,10,12} all construct and run")

    # --- no-MoE ablation -------------------------------------------------
    plain = MoEGraphTransformer(Fn, use_moe=False, d_model=32, n_heads=2)
    lg, ax = plain(feats, mask, ntype, edges)
    assert lg.shape == (B, N_CLASSES)
    print("  use_moe=False ablation runs (single-expert FFN)")

    print("\nALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description="MoE graph transformer.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n_features", type=int, default=50)
    ap.add_argument("--n_experts", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        m = MoEGraphTransformer(args.n_features, n_experts=args.n_experts, k=args.k)
        print(m)
        print(f"parameters: {sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    main()
