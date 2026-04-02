"""
Tiny GPT (Transformer) baseline — matched config for fair BDH comparison.

Same D=64, 2 layers, byte-level vocab=256, block_size=256.
Uses standard GELU MLP + softmax attention — the classic transformer design.
Provides MLP hidden activations as outputs so the frontend can show the
real ~97% dense activation pattern vs BDH's ~5% sparse pattern.

Reference architecture: nanoGPT (Karpathy), adapted to byte-level tiny config.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 2
    vocab_size: int = 256
    block_size: int = 256
    dropout: float = 0.1
    # MLP hidden dim = mlp_ratio*D. Default=4 → 256 hidden neurons.
    # GPT uses GELU (~97% density) vs BDH's ReLU (~5% density).
    mlp_ratio: int = 4


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention with softmax."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with softmax
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerMLP(nn.Module):
    """Standard GELU MLP — the component that produces dense activations."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.n_embd * config.mlp_ratio
        self.fc1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        h = F.gelu(self.fc1(x))  # GELU → nearly all neurons fire (~97%)
        return self.drop(self.fc2(h)), h  # return both output and hidden activations


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = TransformerMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        mlp_out, mlp_hidden = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x, mlp_hidden


class GPTTinyModel(nn.Module):
    """Tiny GPT for comparison with BDH.

    Outputs logits + per-layer MLP hidden activations so the browser
    can display real dense activation patterns.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        """Standard forward — returns only logits (for training)."""
        B, T = idx.shape
        x = self.embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x, _ = block(x)
        return self.lm_head(self.ln_f(x))

    def forward_with_activations(self, idx):
        """Forward pass returning per-layer MLP hidden activations.

        Returns:
            logits: (B, T, vocab_size)
            activations: list of (B, T, hidden_dim) tensors — one per layer
        """
        B, T = idx.shape
        x = self.embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        layer_acts = []
        for block in self.blocks:
            x, mlp_hidden = block(x)
            layer_acts.append(mlp_hidden.detach())
        logits = self.lm_head(self.ln_f(x))
        return logits, layer_acts


TINY_GPT_CONFIG = GPTConfig(
    n_layer=2,
    n_embd=64,
    n_head=2,
    vocab_size=256,
    block_size=256,
    dropout=0.1,
    mlp_ratio=4,
)


def create_gpt_model(config=None):
    if config is None:
        config = TINY_GPT_CONFIG
    model = GPTTinyModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT model created: {n_params:,} parameters, mlp_hidden={config.n_embd * config.mlp_ratio}")
    return model
