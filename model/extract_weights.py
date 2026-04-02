"""
Extract encoder and lm_head weight matrices from the trained BDH model
and save as a compact binary file for browser-side σ-modulated inference.

Output format (Float32, little-endian):
  encoder:  (nh * N, D)  = (1024, 64)  → 65,536 floats
  lm_head:  (D, vocab)   = (64, 256)   → 16,384 floats
  Total: 81,920 floats × 4 bytes = 327,680 bytes (~320 KB)
"""

import argparse
import struct

import numpy as np
import torch

from bdh_tiny import BDHTinyModel, TINY_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Extract BDH weight matrices for browser inference")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_bdh.pt", help="Path to trained model checkpoint")
    parser.add_argument("--output", default="../frontend/public/bdh_weights.bin", help="Output binary file")
    args = parser.parse_args()

    # Load model
    model = BDHTinyModel(TINY_CONFIG)
    torch.serialization.add_safe_globals([TINY_CONFIG.__class__])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    # Extract weight matrices
    encoder = model.encoder.detach().numpy()   # (nh*N, D) = (1024, 64)
    lm_head = model.lm_head.detach().numpy()   # (D, vocab) = (64, 256)

    print(f"encoder shape: {encoder.shape}, dtype: {encoder.dtype}")
    print(f"lm_head shape: {lm_head.shape}, dtype: {lm_head.dtype}")

    # Flatten and write as raw float32 (little-endian)
    encoder_flat = encoder.astype(np.float32).flatten()
    lm_head_flat = lm_head.astype(np.float32).flatten()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "wb") as f:
        f.write(encoder_flat.tobytes())
        f.write(lm_head_flat.tobytes())

    total_bytes = (encoder_flat.size + lm_head_flat.size) * 4
    print(f"\nSaved {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB) to {args.output}")
    print(f"  encoder: {encoder_flat.size} floats")
    print(f"  lm_head: {lm_head_flat.size} floats")


if __name__ == "__main__":
    main()
