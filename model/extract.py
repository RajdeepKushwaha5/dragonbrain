"""
Extract graph topologies (Gx, Gy) and save as JSON for frontend.

Gx = encoder_h @ decoder_x_h   (N×N) — feedforward causal circuit (N→D→N)
Gy = decoder_y_h.T @ decoder_x_h (N×N) — attention readout graph

Both exhibit emergent modular, scale-free structure.

Usage:
    python extract.py                                # default settings
    python extract.py --checkpoint checkpoints/tiny_bdh.pt --top_k 80
"""

import argparse
import json
import os

import torch

from bdh_tiny import BDHTinyModel, TINY_CONFIG


def _extract_topology(G, top_k, threshold_quantile):
    """Generic hub extraction for any N×N graph G (already on CPU).

    Args:
        G: (N, N) tensor — neuron-to-neuron interaction matrix
        top_k: number of hub neurons to include
        threshold_quantile: quantile for edge thresholding

    Returns:
        dict with 'nodes' and 'links' for D3 force-directed graph
    """
    threshold = G.abs().flatten().quantile(threshold_quantile).item()
    out_degree = (G.abs() > threshold).sum(dim=1)
    top_idx = out_degree.topk(min(top_k, G.shape[0])).indices

    G_sub = G[top_idx][:, top_idx]
    nodes = [
        {"id": int(top_idx[i]), "degree": int(out_degree[top_idx[i]])}
        for i in range(len(top_idx))
    ]
    links = []
    for si in range(len(top_idx)):
        for ti in range(len(top_idx)):
            w = float(G_sub[si, ti])
            if abs(w) > threshold * 0.4:
                links.append({
                    "source": int(si),
                    "target": int(ti),
                    "weight": round(w, 4),
                    "excitatory": w > 0,
                })

    return {"nodes": nodes, "links": links, "threshold": round(threshold, 6)}


def extract_graph_topologies(model, head=0, top_k=80, threshold_quantile=0.92):
    """Extract both Gx and Gy graph topologies from trained model.

    Args:
        model: trained BDHTinyModel
        head: which attention head to extract
        top_k: number of hub neurons to include
        threshold_quantile: edge threshold as quantile of absolute weights

    Returns:
        dict with 'gx' and 'gy' topology data
    """
    config = model.config
    nh = config.n_head
    N = config.N
    D = config.n_embd

    decoder_x_h = model.decoder_x[head].detach().cpu()         # (D, N)
    decoder_y_h = model.decoder_y[head].detach().cpu()         # (D, N)
    encoder_h = model.encoder.view(nh, N, D)[head].detach().cpu()  # (N, D)

    Gx = encoder_h @ decoder_x_h         # (N, N) feedforward causal circuit
    Gy = decoder_y_h.t() @ decoder_x_h   # (N, N) attention readout graph

    print(f"Gx shape: {Gx.shape}, range: [{Gx.min():.3f}, {Gx.max():.3f}]")
    print(f"Gy shape: {Gy.shape}, range: [{Gy.min():.3f}, {Gy.max():.3f}]")

    return {
        "gx": _extract_topology(Gx, top_k, threshold_quantile),
        "gy": _extract_topology(Gy, top_k, threshold_quantile),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny_bdh.pt")
    parser.add_argument("--output", default="../frontend/src/data/graph_topology.json")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=80)
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", TINY_CONFIG)
    model = BDHTinyModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt.get('iter', '?')}")

    # Extract topologies
    topologies = extract_graph_topologies(model, head=args.head, top_k=args.top_k)

    gx_nodes = len(topologies["gx"]["nodes"])
    gx_links = len(topologies["gx"]["links"])
    gy_nodes = len(topologies["gy"]["nodes"])
    gy_links = len(topologies["gy"]["links"])
    print(f"Gx: {gx_nodes} nodes, {gx_links} links")
    print(f"Gy: {gy_nodes} nodes, {gy_links} links")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(topologies, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
