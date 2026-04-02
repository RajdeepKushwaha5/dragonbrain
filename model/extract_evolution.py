"""
Extract graph evolution data: random init → trained BDH.

Shows how scale-free hub structure emerges from random initialization
during training. Outputs a JSON with two snapshots for the frontend
evolution slider.

Usage:
    python extract_evolution.py
"""

import json
import os

import torch

from bdh_tiny import BDHTinyModel, TINY_CONFIG, create_model
from extract import _extract_topology


def extract_topologies(model, head=0, top_k=80, threshold_quantile=0.92):
    """Extract Gx and Gy graph topologies from a BDH model."""
    config = model.config
    nh = config.n_head
    N = config.N
    D = config.n_embd

    decoder_x_h = model.decoder_x[head].detach().cpu()
    decoder_y_h = model.decoder_y[head].detach().cpu()
    encoder_h = model.encoder.view(nh, N, D)[head].detach().cpu()

    Gx = encoder_h @ decoder_x_h
    Gy = decoder_y_h.t() @ decoder_x_h

    return {
        "gx": _extract_topology(Gx, top_k, threshold_quantile),
        "gy": _extract_topology(Gy, top_k, threshold_quantile),
    }


def degree_stats(topo):
    """Compute degree distribution stats for a topology."""
    degrees = [n["degree"] for n in topo["nodes"]]
    if not degrees:
        return {"max_degree": 0, "avg_degree": 0, "nodes": 0, "edges": 0}
    return {
        "max_degree": max(degrees),
        "avg_degree": round(sum(degrees) / len(degrees), 1),
        "nodes": len(topo["nodes"]),
        "edges": len(topo["links"]),
    }


def main():
    output_path = "../frontend/src/data/graph_evolution.json"

    # 1. Random init (untrained model)
    print("Extracting random init topology...")
    torch.manual_seed(42)
    model_random = create_model(TINY_CONFIG)
    random_topo = extract_topologies(model_random)

    # 2. Trained model
    print("Extracting trained topology...")
    ckpt = torch.load("checkpoints/tiny_bdh.pt", map_location="cpu", weights_only=False)
    config = ckpt.get("config", TINY_CONFIG)
    model_trained = BDHTinyModel(config)
    model_trained.load_state_dict(ckpt["model"])
    model_trained.eval()
    iter_num = ckpt.get("iter", 0)

    trained_topo = extract_topologies(model_trained)

    result = {
        "snapshots": [
            {
                "label": "Random Init",
                "iter": 0,
                "stats": {
                    **degree_stats(random_topo["gx"]),
                    "gy_edges": len(random_topo["gy"]["links"]),
                },
                "gx": random_topo["gx"],
                "gy": random_topo["gy"],
            },
            {
                "label": f"Trained (iter {iter_num})",
                "iter": iter_num,
                "stats": {
                    **degree_stats(trained_topo["gx"]),
                    "gy_edges": len(trained_topo["gy"]["links"]),
                },
                "gx": trained_topo["gx"],
                "gy": trained_topo["gy"],
            },
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)

    # Print summary
    rs = result["snapshots"][0]["stats"]
    ts = result["snapshots"][1]["stats"]
    print(f"\nRandom Init — max_degree={rs['max_degree']}, avg={rs['avg_degree']}, edges={rs['edges']}")
    print(f"Trained     — max_degree={ts['max_degree']}, avg={ts['avg_degree']}, edges={ts['edges']}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
