"""
Memory Benchmark: BDH σ vs GPT KV-Cache

Computes and compares memory footprint of BDH's fixed-size Hebbian memory (σ)
against a GPT transformer's linearly-growing KV-cache at various context lengths.

Usage:
    python memory_benchmark.py
"""

import sys

# ── BDH Architecture Constants ──
BDH_N_LAYER = 2
BDH_N_HEAD = 2
BDH_N = 512       # neurons per head
BDH_D = 64        # embedding dim
BDH_VOCAB = 256
BDH_PARAMS = 229_000  # approximate

# σ memory: n_layer × n_head × N × N × 4 bytes (float32)
BDH_SIGMA_BYTES = BDH_N_LAYER * BDH_N_HEAD * BDH_N * BDH_N * 4


# ── GPT Architecture Constants ──
GPT_N_LAYER = 2
GPT_N_HEAD = 2
GPT_D = 64         # embedding dim
GPT_HIDDEN = 256   # MLP hidden dim
GPT_VOCAB = 256
GPT_PARAMS = 148_000  # approximate

def gpt_kv_bytes(context_length):
    """KV-cache: 2 (K+V) × n_layer × n_head × (D/n_head) × T × 4 bytes."""
    head_dim = GPT_D // GPT_N_HEAD
    return 2 * GPT_N_LAYER * GPT_N_HEAD * head_dim * context_length * 4


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    else:
        return f"{b / (1024 * 1024):.2f} MB"


def main():
    print("=" * 72)
    print("  MEMORY BENCHMARK: BDH Hebbian σ vs GPT KV-Cache")
    print("=" * 72)
    print()

    print(f"  BDH: {BDH_N_LAYER}L × {BDH_N_HEAD}H × N={BDH_N}, D={BDH_D}  ({BDH_PARAMS:,} params)")
    print(f"  GPT: {GPT_N_LAYER}L × {GPT_N_HEAD}H, D={GPT_D}, MLP={GPT_HIDDEN}  ({GPT_PARAMS:,} params)")
    print()

    bdh_mem = BDH_SIGMA_BYTES

    print(f"  BDH σ memory (constant): {format_bytes(bdh_mem)}")
    print(f"    = {BDH_N_LAYER} layers × {BDH_N_HEAD} heads × {BDH_N}×{BDH_N} × 4 bytes")
    print()

    context_lengths = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # Find crossover
    crossover = None
    for t in range(1, 100000):
        if gpt_kv_bytes(t) >= bdh_mem:
            crossover = t
            break

    print(f"  {'Context':>10}  {'BDH σ':>12}  {'GPT KV':>12}  {'Ratio':>8}  {'Winner':>6}")
    print(f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 8}  {'─' * 6}")

    for T in context_lengths:
        gpt_mem = gpt_kv_bytes(T)
        ratio = gpt_mem / bdh_mem if bdh_mem > 0 else 0
        winner = "BDH" if gpt_mem > bdh_mem else "GPT" if gpt_mem < bdh_mem else "TIE"
        marker = " ←" if T == crossover else ""
        print(f"  {T:>10,}  {format_bytes(bdh_mem):>12}  {format_bytes(gpt_mem):>12}  {ratio:>7.2f}×  {winner:>6}{marker}")

    print()
    if crossover:
        print(f"  ★ Crossover at T = {crossover:,} tokens")
        print(f"    Beyond this point, BDH uses LESS memory than GPT.")
    print()

    print("  Key Insight:")
    print("    BDH's Hebbian memory (σ) is O(1) in context length.")
    print("    GPT's KV-cache grows O(T) — linearly with each token.")
    print("    For long-context scenarios, BDH's fixed memory is a")
    print("    fundamental architectural advantage.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
