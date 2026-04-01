"""
Export trained BDH model to ONNX with multi-output for browser inference.

Per layer exports 4 tensors:
  - x_sparse    (B, nh, T, N) — input neuron activations (~5% nonzero)
  - y_sparse    (B, nh, T, N) — readout neuron activations (~3-5% nonzero)
  - xy_sparse   (B, nh, T, N) — element-wise Hebbian co-activation (x * y)
  - attn_scores (B, nh, T, T) — causal linear attention pattern

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint checkpoints/tiny_bdh.pt --output ../frontend/public/model.onnx
"""

import argparse
import os

import torch
import torch.nn as nn

from bdh_tiny import BDHTinyModel, TINY_CONFIG


class BDHOnnxWrapper(nn.Module):
    """Wraps BDH model to expose layer activations as named ONNX outputs."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, idx):
        logits, layer_data = self.m.forward_with_hooks(idx)
        outputs = [logits]
        for ld in layer_data:
            outputs.append(ld["x_sparse"])
            outputs.append(ld["y_sparse"])
            outputs.append(ld["xy_sparse"])
            outputs.append(ld["attn_scores"])
        return tuple(outputs)


def export(model, path, seq_len=32):
    """Export model to ONNX format.

    Args:
        model: trained BDHTinyModel
        path: output .onnx file path
        seq_len: dummy sequence length for tracing
    """
    wrapper = BDHOnnxWrapper(model).eval()
    dummy = torch.zeros(1, seq_len, dtype=torch.long)

    n_layers = model.config.n_layer
    output_names = ["logits"]
    for i in range(n_layers):
        output_names += [
            f"layer_{i}_x_sparse",
            f"layer_{i}_y_sparse",
            f"layer_{i}_xy_sparse",
            f"layer_{i}_attn_scores",
        ]

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
    print(f"  Outputs: {len(output_names)} ({output_names})")


def verify_export(onnx_path, seq_len=16):
    """Verify ONNX model produces correct output shapes."""
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
    output_names = [o.name for o in session.get_outputs()]
    for name, arr in zip(output_names, outputs):
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny_bdh.pt")
    parser.add_argument("--output", default="../frontend/public/model.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", TINY_CONFIG)
    model = BDHTinyModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt.get('iter', '?')}")

    export(model, args.output)
    verify_export(args.output)


if __name__ == "__main__":
    main()
