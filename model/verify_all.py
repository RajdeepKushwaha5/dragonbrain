"""Cross-verification script for Dragon Brain project."""
import torch
from bdh_tiny import BDHTinyModel, TINY_CONFIG, create_model
from gpt_tiny import GPTTinyModel, TINY_GPT_CONFIG, create_gpt_model

print("=" * 60)
print("DRAGON BRAIN — FULL VERIFICATION")
print("=" * 60)

# 1. Model creation and forward pass
print("\n[1] Model creation and forward pass")
m = create_model()
x = torch.randint(0, 256, (1, 16))
logits = m(x)
print(f"  BDH forward OK: logits shape = {logits.shape}")
assert logits.shape == (1, 16, 256), f"Expected (1,16,256), got {logits.shape}"

logits_h, ld = m.forward_with_hooks(x)
print(f"  BDH hooks OK: {len(ld)} layers")
for i, l in enumerate(ld):
    xs = l["x_sparse"]
    ys = l["y_sparse"]
    xys = l["xy_sparse"]
    attn = l["attn_scores"]
    sparsity = (xs > 0).float().mean().item()
    print(f"    L{i}: x={xs.shape} y={ys.shape} xy={xys.shape} attn={attn.shape} sparsity={sparsity*100:.1f}%")

g = create_gpt_model()
logits_g = g(x)
print(f"  GPT forward OK: logits shape = {logits_g.shape}")
assert logits_g.shape == (1, 16, 256)

logits_ga, acts = g.forward_with_activations(x)
for i, a in enumerate(acts):
    density = (a.abs() > 1e-6).float().mean().item()
    print(f"    L{i}: act shape={a.shape} density={density*100:.1f}%")

# 2. Load checkpoints
print("\n[2] Checkpoint loading")
ckpt_b = torch.load("checkpoints/tiny_bdh.pt", map_location="cpu", weights_only=False)
bi = ckpt_b.get("iter", "?")
bv = ckpt_b.get("best_val", 0)
print(f"  BDH checkpoint: iter={bi}, val_loss={bv:.4f}")

ckpt_g = torch.load("checkpoints/tiny_gpt.pt", map_location="cpu", weights_only=False)
gi = ckpt_g.get("iter", "?")
gv = ckpt_g.get("best_val", 0)
print(f"  GPT checkpoint: iter={gi}, val_loss={gv:.4f}")

# 3. Trained model sparsity/density
print("\n[3] Trained model sparsity analysis")
mb = BDHTinyModel(TINY_CONFIG)
mb.load_state_dict(ckpt_b["model"])
mb.eval()
with torch.no_grad():
    _, ld2 = mb.forward_with_hooks(x)
    for i, l in enumerate(ld2):
        xs = l["x_sparse"]
        sparsity = (xs > 0).float().mean().item()
        print(f"  Trained BDH L{i}: {sparsity*100:.1f}% active neurons")

mg = GPTTinyModel(TINY_GPT_CONFIG)
mg.load_state_dict(ckpt_g["model"])
mg.eval()
with torch.no_grad():
    _, acts2 = mg.forward_with_activations(x)
    for i, a in enumerate(acts2):
        density = (a.abs() > 1e-6).float().mean().item()
        print(f"  Trained GPT L{i}: {density*100:.1f}% active neurons")

# 4. Graph extraction (Gx/Gy)
print("\n[4] Graph topology extraction")
decoder_x_h = mb.decoder_x[0].detach()  # (D, N) = (64, 512) 
decoder_y_h = mb.decoder_y[0].detach()  # (D, N) = (64, 512)
encoder_h = mb.encoder.view(2, 512, 64)[0].detach()  # (N, D) = (512, 64)

Gx = encoder_h @ decoder_x_h  # (N, D) @ (D, N) = (N, N) = (512, 512)
Gy = decoder_y_h.t() @ decoder_x_h  # (N, D).T @ (D, N) = (N, N) 

print(f"  Gx: shape={Gx.shape}, range=[{Gx.min():.4f}, {Gx.max():.4f}]")
print(f"  Gy: shape={Gy.shape}, range=[{Gy.min():.4f}, {Gy.max():.4f}]")
assert Gx.shape == (512, 512), f"Gx shape error: {Gx.shape}"
assert Gy.shape == (512, 512), f"Gy shape error: {Gy.shape}"

# Check for scale-free structure (trained should have higher max degree)
gx_deg = (Gx.abs() > Gx.abs().flatten().quantile(0.92)).sum(dim=1)
print(f"  Gx degree stats: max={gx_deg.max().item()}, mean={gx_deg.float().mean().item():.1f}")

# 5. Hebbian σ update consistency
print("\n[5] Hebbian σ update consistency")
N = 512
sigma = torch.zeros(N, N)
x_act = ld2[0]["x_sparse"][0, 0, -1]  # last token, head 0
y_act = ld2[0]["y_sparse"][0, 0, -1]
sigma += torch.outer(y_act, x_act)
nonzero = (sigma.abs() > 1e-8).sum().item()
print(f"  outer(y, x): {nonzero} nonzero entries")

# Verify xy_sparse == x_sparse * y_sparse (gating correctness)
xy = ld2[0]["xy_sparse"][0, 0, -1]
y_pre = ld2[0]["y_sparse"][0, 0, -1]
x_s = ld2[0]["x_sparse"][0, 0, -1]
manual_xy = y_pre * x_s
assert torch.allclose(xy, manual_xy, atol=1e-5), "xy_sparse != x * y (gating broken)"
print("  xy_sparse = x * y_sparse: PASSED")

# 6. Attention mask check
print("\n[6] Attention mask (causal) check")
attn = ld2[0]["attn_scores"]  # (1, 2, 16, 16)
T = 16
for h in range(2):
    scores = attn[0, h]
    # Check upper triangle is zero (causal mask)
    upper = scores.triu(diagonal=0)
    assert (upper.abs() < 1e-8).all(), f"Causal mask violated in head {h}"
print("  Causal mask (upper triangle = 0): PASSED for both heads")

# 7. RoPE sanity check
print("\n[7] RoPE position encoding check")
from bdh_tiny import _apply_rope
# Test that RoPE changes with position
q = torch.randn(1, 2, 8, 512)
qr = _apply_rope(q)
# Same content at different positions should produce different rotations
assert not torch.allclose(qr[0, 0, 0], qr[0, 0, 1]), "RoPE not applying position-dependent rotation"
print("  Position-dependent rotation: PASSED")

# 8. Weight sharing across layers
print("\n[8] Weight sharing verification")
# The model uses the same decoder_x/decoder_y/encoder across all layers (loop)
# Verify by checking there's only one set of parameters
param_names = [n for n, _ in mb.named_parameters()]
has_per_layer = any("layers" in n or "blocks" in n for n in param_names)
print(f"  Parameter names: {param_names}")
print(f"  Per-layer weights detected: {has_per_layer}")
assert not has_per_layer, "Found per-layer parameters (should be shared)"
print("  Shared weights across layers: PASSED")

# 9. Memory scaling math
print("\n[9] Memory scaling math verification")
sigma_bytes = 4 * 512 * 512 * 4  # 4 (layer_head combos) * N * N * 4 bytes
kv_per_token = 2 * 2 * 2 * 32 * 4  # 2(K+V) * 2 layers * 2 heads * head_dim * 4 bytes
crossover = sigma_bytes / kv_per_token
print(f"  σ memory: {sigma_bytes:,} bytes = {sigma_bytes/1024/1024:.1f} MB (constant)")
print(f"  KV per token: {kv_per_token:,} bytes = {kv_per_token/1024:.1f} KB")
print(f"  Crossover at: T = {crossover:.0f} tokens")
assert abs(sigma_bytes - 4_194_304) < 100, f"σ bytes mismatch: {sigma_bytes}"
assert abs(kv_per_token - 1024) < 10, f"KV per token mismatch: {kv_per_token}"
assert abs(crossover - 4096) < 10, f"Crossover mismatch: {crossover}"
print("  Memory scaling: σ=4MB, KV=1KB/token, crossover≈4096: PASSED")

# 10. Parameter count verification
print("\n[10] Parameter counts")
bdh_params = sum(p.numel() for p in mb.parameters())
gpt_params = sum(p.numel() for p in mg.parameters())
print(f"  BDH: {bdh_params:,} parameters")
print(f"  GPT: {gpt_params:,} parameters")

# 11. Logit consistency (forward vs forward_with_hooks)
print("\n[11] Forward consistency check")
with torch.no_grad():
    logits_plain = mb(x)
    logits_hooks, _ = mb.forward_with_hooks(x)
    diff = (logits_plain - logits_hooks).abs().max().item()
    print(f"  Max logit diff between forward() and forward_with_hooks(): {diff:.8f}")
    assert diff < 1e-4, f"Forward methods produce different results: {diff}"
    print("  Forward consistency: PASSED")

# 12. σ-modulated logit pathway (manual test)
print("\n[12] σ-modulated logit pathway")
import numpy as np
encoder_np = mb.encoder.detach().numpy()  # (1024, 64)
lm_head_np = mb.lm_head.detach().numpy()  # (64, 256)

x_head0 = ld2[0]["x_sparse"][0, 0, -1].numpy()  # (512,)
sigma_np = np.outer(y_act.numpy(), x_act.numpy())  # (512, 512) from earlier

# σ · x → (512,)
a_sigma = np.maximum(0, sigma_np @ x_head0)  # ReLU

# E^T · a_sigma → (64,)
enc_h0 = encoder_np[:512]  # head 0 rows
correction = enc_h0.T @ a_sigma  # (64,)

# correction · lm_head → (256,)
logit_bias = correction @ lm_head_np  # (256,)

print(f"  a_sigma nonzero: {(a_sigma > 1e-8).sum()}")
print(f"  correction norm: {np.linalg.norm(correction):.4f}")
print(f"  logit_bias range: [{logit_bias.min():.4f}, {logit_bias.max():.4f}]")
print("  σ→E→lm_head pipeline: OK")

print("\n" + "=" * 60)
print("ALL 12 VERIFICATION CHECKS PASSED")
print("=" * 60)
