"""
Export trained GPT baseline to ONNX for browser-side activation comparison.

Outputs:
  - logits         (B, T, 256)
  - mlp_act_0      (B, T, hidden_dim)  — Layer 0 MLP hidden activations
  - mlp_act_1      (B, T, hidden_dim)  — Layer 1 MLP hidden activations

The frontend uses mlp_act to show real dense activation patterns (~97% nonzero)
alongside BDH's sparse pattern (~5% nonzero).

Usage:
    python export_gpt_onnx.py
    python export_gpt_onnx.py --checkpoint checkpoints/tiny_gpt.pt --output ../frontend/public/transformer.onnx
"""

import argparse
import os

import torch
import torch.nn as nn

from gpt_tiny import GPTTinyModel, TINY_GPT_CONFIG


class GPTOnnxWrapper(nn.Module):
    """Wraps GPT to expose MLP activations as named ONNX outputs."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, idx):
        logits, layer_acts = self.m.forward_with_activations(idx)
        return (logits,) + tuple(layer_acts)


def export(model, path, seq_len=32):
    wrapper = GPTOnnxWrapper(model).eval()
    dummy = torch.zeros(1, seq_len, dtype=torch.long)

    n_layers = model.config.n_layer
    output_names = ["logits"] + [f"mlp_act_{i}" for i in range(n_layers)]

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        path,
        input_names=["tokens"],
        output_names=output_names,
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=17,
    )

    file_size = os.path.getsize(path) / (1024 * 1024)
    print(f"Exported ONNX: {path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Outputs: {output_names}")


def verify_export(onnx_path, seq_len=16):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return

    import numpy as np
    session = ort.InferenceSession(onnx_path)
    dummy = np.zeros((1, seq_len), dtype=np.int64)
    outputs = session.run(None, {"tokens": dummy})

    print(f"\nVerification (seq_len={seq_len}):")
    for out_meta, arr in zip(session.get_outputs(), outputs):
        density = (np.abs(arr) > 1e-6).mean()
        print(f"  {out_meta.name}: shape={arr.shape}, density={density*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--output", default="../frontend/public/transformer.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", TINY_GPT_CONFIG)
    model = GPTTinyModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded GPT checkpoint from iter {ckpt.get('iter', '?')}")

    export(model, args.output)
    verify_export(args.output)


if __name__ == "__main__":
    main()
