"""
Baby Dragon Hatchling (BDH) — Tiny configuration for browser visualization.

Based on the official implementation at github.com/pathwaycom/bdh
Adapted to a tiny config (n_layer=2, n_embd=64, n_head=2, N=512 neurons/head)
that preserves all three BDH phenomena: sparsity, graph emergence, Hebbian memory.

Reference: Kosowski et al. 2025, arXiv:2509.26507
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BDHConfig:
    n_layer: int = 2
    n_embd: int = 64          # D — embedding dimension
    n_head: int = 2           # nh — number of heads
    mlp_internal_dim_multiplier: int = 16  # N = D * mult // nh = 64*16//2 = 512
    vocab_size: int = 256     # byte-level tokenization
    block_size: int = 256     # max sequence length
    dropout: float = 0.1

    @property
    def N(self):
        """Number of neurons per head."""
        return self.n_embd * self.mlp_internal_dim_multiplier // self.n_head


def _apply_rope(x):
    """Apply Rotary Position Embedding (RoPE) to Q or K activations.

    Following Su et al. 2021 (RoFormer) as used in the BDH paper (Appendix E).
    Applies position-dependent rotation so that (Qr @ Kr^T) encodes relative
    position, even when Q=K=x (same sparse activation vector).

    Args:
        x: (B, H, T, d) tensor — Q or K activations
    Returns:
        (B, H, T, d) tensor with position-dependent rotation applied
    """
    B, H, T, d = x.shape
    assert d % 2 == 0, f"RoPE requires even dimension, got {d}"
    half_d = d // 2

    # Frequency bands: θ_i = 10000^{-2i/d}
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d))
    # Position indices
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    # Angles: (T, d//2)
    angles = torch.outer(t, freqs)
    cos_vals = angles.cos()  # (T, d//2)
    sin_vals = angles.sin()  # (T, d//2)

    # Split into even/odd pairs and apply 2D rotation
    x1 = x[..., 0::2]  # (B, H, T, d//2)
    x2 = x[..., 1::2]  # (B, H, T, d//2)
    out = torch.stack([
        x1 * cos_vals - x2 * sin_vals,
        x1 * sin_vals + x2 * cos_vals,
    ], dim=-1).flatten(-2)  # interleave back → (B, H, T, d)
    return out


class LinearCausalAttention(nn.Module):
    """BDH's linear causal attention with Rotary Position Embeddings (RoPE).

    Following the paper exactly (Appendix E, Definition 4):
        Qr = RoPE(Q)
        Kr = RoPE(K)
        scores = (Qr @ Kr^T) · tril(-1)
        output = scores @ V

    RoPE provides position-dependent rotation so that attention patterns
    encode relative position, even when Q=K=x (same sparse activation vector).
    No softmax — this is truly linear attention, O(T²) in parallel form.
    """

    def forward(self, Q, K, V):
        """
        Q, K: (B, nh, T, N) — sparse neuron activations
        V:    (B, 1,  T, D) — token embeddings (broadcasts over heads)

        Returns:
            out:    (B, nh, T, D) — attended values
            scores: (B, nh, T, T) — attention weights (detached, for viz)
        """
        Qr = _apply_rope(Q)
        Kr = _apply_rope(K)

        # Linear attention scores with causal mask (lower-triangular, zero diagonal)
        scores = torch.matmul(Qr, Kr.transpose(-1, -2))  # (B, nh, T, T)
        scores = scores.tril(diagonal=-1)

        # Weighted sum of values
        out = torch.matmul(scores, V)  # V broadcasts: (B,1,T,D) → (B,nh,T,D)
        return out, scores.detach()


class BDHTinyModel(nn.Module):
    """Tiny BDH model for interactive visualization.

    Architecture per layer (matching paper Appendix E notation):
        x = ReLU(v_ast @ decoder_x)        → (B, H, T, N//H) ~5% nonzero
        a_ast = attn(Q=x, K=x, V=v_ast)   → (B, H, T, D)
        y = ReLU(LN(a_ast) @ decoder_y) * x → (B, H, T, N//H) gated output
        v_ast += LN(y @ encoder)           → (B, 1, T, D) residual update
        v_ast = LN(v_ast)

    Parameter naming follows the paper:
        decoder_x (H, D, N//H) — projects D→N to produce x activations
        decoder_y (H, D, N//H) — projects D→N to produce y activations
        encoder   (N, D)       — projects N→D to return to embedding space
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        D = config.n_embd
        N = config.N
        nh = config.n_head

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, D)

        # Shared weight matrices (paper Appendix E — tied across all L layers)
        self.decoder_x = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))  # D→N (x activations)
        self.decoder_y = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))  # D→N (y activations)
        self.encoder = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))   # N→D (back to embed)

        # Attention (RoPE-based linear causal attention)
        self.attn = LinearCausalAttention()

        # Layer norm without learnable affine (paper: elementwise_affine=False, bias=False)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)

        # Language model head
        self.lm_head = nn.Parameter(torch.zeros(D, config.vocab_size).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        """Standard forward pass for training.

        Args:
            idx: (B, T) token indices

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.size()
        D = self.config.n_embd
        N = self.config.N
        nh = self.config.n_head

        v_ast = self.ln(self.embed(idx).unsqueeze(1))  # (B, 1, T, D)

        for _ in range(self.config.n_layer):
            x = F.relu(v_ast @ self.decoder_x)        # (B, nh, T, N) — sparse

            a_ast = self.attn(Q=x, K=x, V=v_ast)[0]   # (B, nh, T, D)

            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x  # (B, nh, T, N)
            y = y.transpose(1, 2).reshape(B, 1, T, -1)       # (B, 1, T, N_total)
            y = self.drop(y)

            v_ast = v_ast + self.ln(y @ self.encoder)  # (B, 1, T, D)
            v_ast = self.ln(v_ast)

        logits = v_ast.squeeze(1) @ self.lm_head       # (B, T, vocab_size)
        return logits

    def forward_with_hooks(self, idx):
        """Forward pass that returns per-layer activation data for visualization.

        Returns:
            logits: (B, T, vocab_size)
            layer_data: list of dicts, each containing:
                x_sparse    (B, nh, T, N)  — input neuron activations
                y_sparse    (B, nh, T, N)  — readout neuron activations
                xy_sparse   (B, nh, T, N)  — element-wise Hebbian co-activation
                attn_scores (B, nh, T, T)  — linear attention weights (causal)

        Key insight (paper Section 6.4): sparsity of x_sparse tracks uncertainty.
        For repeated/boring tokens, sparsity drops to ~2.5%; novel tokens push ~7%.
        """
        B, T = idx.size()
        layer_data = []

        v_ast = self.ln(self.embed(idx).unsqueeze(1))  # (B, 1, T, D)

        for _ in range(self.config.n_layer):
            x = F.relu(v_ast @ self.decoder_x)                # (B, nh, T, N)
            x_sparse = x                                      # alias for clarity

            a_ast, attn_scores = self.attn(Q=x, K=x, V=v_ast) # (B, nh, T, D)

            y_pre_gate = F.relu(self.ln(a_ast) @ self.decoder_y)  # (B, nh, T, N)
            y = y_pre_gate * x                                    # gated by x
            y_sparse = y_pre_gate                                 # y before gating
            xy_sparse = y                                         # == x * y_pre_gate

            layer_data.append({
                "x_sparse": x_sparse.detach(),
                "y_sparse": y_sparse.detach(),
                "xy_sparse": xy_sparse.detach(),
                "attn_scores": attn_scores,     # (B, nh, T, T)
            })

            y_flat = y.transpose(1, 2).reshape(B, 1, T, -1)  # (B, 1, T, N_total)
            y_flat = self.drop(y_flat)

            v_ast = v_ast + self.ln(y_flat @ self.encoder)    # (B, 1, T, D)
            v_ast = self.ln(v_ast)

        logits = v_ast.squeeze(1) @ self.lm_head
        return logits, layer_data


# Tiny config used throughout the project
TINY_CONFIG = BDHConfig(
    n_layer=2,
    n_embd=64,
    n_head=2,
    mlp_internal_dim_multiplier=16,
    vocab_size=256,
    block_size=256,
    dropout=0.1,
)


def create_model(config=None):
    """Create a BDH model with the given config (defaults to TINY_CONFIG)."""
    if config is None:
        config = TINY_CONFIG
    model = BDHTinyModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BDH model created: {n_params:,} parameters, N={config.N} neurons/head")
    return model
