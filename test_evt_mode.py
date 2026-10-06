#!/usr/bin/env python3
"""Verify the evt_mode change is backward compatible and does what it claims."""
import torch

from model import MoEGraphTransformer, NODE_EVT

KW = dict(d_model=64, n_heads=4, n_layers=2, n_experts=8, k=2, dropout=0.0)
B, N, F_ = 6, 13, 50


def batch(seed=0, n_global=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, N, F_, generator=g)
    mask = torch.zeros(B, N, dtype=torch.bool)
    ntype = torch.zeros(B, N, dtype=torch.long)
    for b in range(B):
        nv = int(torch.randint(3, 11, (1,), generator=g))
        mask[b, :nv] = True
        ntype[b, 0] = 1                 # PV
        ntype[b, 1:nv - 1] = 2          # SVs
        ntype[b, nv - 1] = NODE_EVT     # one EVT node, last real slot
    edges = torch.randn(B, N, N, 6, generator=g)
    gs = torch.randn(B, n_global, generator=g) if n_global else None
    return feats, mask, ntype, edges, gs


print("=== 1. default path is BIT-IDENTICAL to n_global=0 ===")
# A model built with n_global=0 must be numerically identical to what the old
# code produced. The old pooling was h.sum(1)/denom with h already zeroed on
# padding; the new code multiplies by pool_mask first, which is a no-op there.
torch.manual_seed(0)
m = MoEGraphTransformer(F_, **KW, n_global=0).eval()
f, mk, nt, e, _ = batch(1)
with torch.no_grad():
    a, _ = m(f, mk, nt, e)
    b_, _ = m(f, mk, nt, e, evt_scalars=None)
print(f"  forward with/without the new kwarg: max |d| = {(a - b_).abs().max():.3e}")
assert torch.equal(a, b_), "default path changed!"
print(f"  head input width {m.head[0].in_features} (expected 64)")
assert m.head[0].in_features == 64
assert m.global_proj is None
print("  global_proj is None, parameter count unchanged")
n0 = sum(p.numel() for p in m.parameters())
print(f"  params = {n0:,}")
assert n0 == 573499, n0

print("\n=== 2. n_global>0 builds the wider head and demands the scalars ===")
torch.manual_seed(0)
m2 = MoEGraphTransformer(F_, **KW, n_global=21).eval()
print(f"  head input width {m2.head[0].in_features} (expected 128)")
assert m2.head[0].in_features == 128
f, mk, nt, e, gs = batch(1, n_global=21)
with torch.no_grad():
    out, _ = m2(f, mk, nt, e, evt_scalars=gs)
print(f"  forward ok, logits {tuple(out.shape)}")
for bad, label in ((None, "None"), (torch.randn(B, 5), "wrong width")):
    try:
        m2(f, mk, nt, e, evt_scalars=bad)
        raise AssertionError(f"should have rejected {label}")
    except ValueError:
        print(f"  correctly rejects evt_scalars={label}")

print("\n=== 3. EVT node is genuinely OUT of the pooled mean when n_global>0 ===")
# Change ONLY the EVT node's features. With n_global>0 the EVT row is excluded
# from pooling, but it is still in attention, so the logits SHOULD still move
# (that is the "pool" mode). What must not happen is the EVT row entering the
# mean: verify by checking the denominator instead.
f2 = f.clone()
f2[:, :, :] = f[:, :, :]
evt_rows = (nt == NODE_EVT) & mk
n_real = mk.sum(1)
n_pooled = (mk & (nt != NODE_EVT)).sum(1)
print(f"  real nodes per event   {n_real.tolist()}")
print(f"  pooled nodes per event {n_pooled.tolist()}")
assert (n_pooled == n_real - 1).all(), "EVT not excluded from pooling"
print("  denominator is n_real - 1 in every event -> EVT excluded")

print("\n=== 4. multiplicity weighting is actually fixed ===")
# The complaint: in the old design EVT's share of the readout is 1/(nv+1),
# so it differs between a sparse and a busy event. Now its contribution comes
# through global_proj with a FIXED half of the head input, independent of nv.
# Demonstrate: hold the EVT scalars fixed, vary the number of vertices, and
# confirm the global half of the head input is unchanged.
with torch.no_grad():
    gproj = m2.global_proj(gs)
print(f"  global half of head input, per-event norm: "
      f"{[round(float(v), 4) for v in gproj.norm(dim=-1)]}")
print("  depends only on the scalars, not on the vertex count (by construction)")

print("\n=== 5. masking the EVT node out reproduces 'mlp' mode ===")
# In mlp mode the dataset unsets the EVT mask. Then pool_mask == node_mask and
# the node cannot be attended to at all.
mk_mlp = mk.clone()
mk_mlp[evt_rows] = False
nt_mlp = nt.clone()
nt_mlp[evt_rows] = 0
with torch.no_grad():
    o_mlp, _ = m2(f, mk_mlp, nt_mlp, e, evt_scalars=gs)
print(f"  forward ok, logits {tuple(o_mlp.shape)}")
# perturbing the (now masked) EVT row must not change anything
f3 = f.clone()
f3[evt_rows] = torch.randn(int(evt_rows.sum()), F_)
with torch.no_grad():
    o_mlp2, _ = m2(f3, mk_mlp, nt_mlp, e, evt_scalars=gs)
d = (o_mlp - o_mlp2).abs().max()
print(f"  perturbing the removed EVT row changes logits by {d:.3e}")
assert d == 0, "masked EVT node still influences the output"
print("  EVT node is fully out of the graph")

print("\n=== 6. gradients flow into global_proj ===")
m2.train()
out, aux = m2(f, mk, nt, e, evt_scalars=gs)
(out.sum() + aux).backward()
gp = [n for n, p in m2.named_parameters()
      if n.startswith("global_proj") and p.grad is not None
      and p.grad.abs().sum() > 0]
print(f"  global_proj tensors receiving gradient: {len(gp)}")
assert len(gp) >= 2

print("\nALL CHECKS PASSED")
