"""
Generate mock/placeholder data files for frontend development.

Creates realistic-looking graph_topology.json and synapse_labels.json
so the frontend can be developed and tested without a trained model.

Usage:
    python generate_mock_data.py
"""

import json
import math
import os
import random

random.seed(42)


def generate_mock_graph(n_nodes=80, n_links_approx=200):
    """Generate a mock scale-free-ish graph topology."""
    # Create nodes with power-law degree distribution
    nodes = []
    for i in range(n_nodes):
        # Power-law-ish degree: most nodes low degree, a few hubs
        degree = max(1, int(random.paretovariate(1.5)))
        degree = min(degree, 60)
        nodes.append({"id": i * 6 + random.randint(0, 5), "degree": degree})

    # Sort by degree descending (hubs first)
    nodes.sort(key=lambda n: n["degree"], reverse=True)

    # Generate links with preferential attachment bias
    links = []
    for _ in range(n_links_approx):
        # Bias toward high-degree nodes
        weights = [n["degree"] ** 1.5 for n in nodes]
        total = sum(weights)
        probs = [w / total for w in weights]

        si = random.choices(range(n_nodes), weights=probs, k=1)[0]
        ti = random.choices(range(n_nodes), weights=probs, k=1)[0]
        if si == ti:
            continue

        w = random.gauss(0, 0.3)
        if abs(w) < 0.05:
            continue

        links.append({
            "source": si,
            "target": ti,
            "weight": round(w, 4),
            "excitatory": w > 0,
        })

    return {"nodes": nodes, "links": links, "threshold": 0.15}


def generate_mock_synapse_labels():
    """Generate mock synapse labels for development."""
    return {
        "currency": {
            "pairs": [
                {"i": 47, "j": 312, "strength": 2.34},
                {"i": 128, "j": 55, "strength": 1.98},
                {"i": 91, "j": 203, "strength": 1.76},
                {"i": 312, "j": 47, "strength": 1.65},
                {"i": 22, "j": 189, "strength": 1.43},
                {"i": 405, "j": 88, "strength": 1.21},
                {"i": 167, "j": 334, "strength": 1.08},
                {"i": 256, "j": 12, "strength": 0.95},
                {"i": 78, "j": 401, "strength": 0.87},
                {"i": 199, "j": 267, "strength": 0.79},
            ],
            "color": "#f7c948",
            "label": "Currency Synapse",
        },
        "proper_noun": {
            "pairs": [
                {"i": 33, "j": 445, "strength": 2.11},
                {"i": 211, "j": 67, "strength": 1.87},
                {"i": 156, "j": 389, "strength": 1.54},
                {"i": 400, "j": 23, "strength": 1.32},
                {"i": 89, "j": 278, "strength": 1.18},
                {"i": 345, "j": 112, "strength": 1.05},
                {"i": 67, "j": 490, "strength": 0.93},
                {"i": 234, "j": 56, "strength": 0.84},
                {"i": 178, "j": 301, "strength": 0.76},
                {"i": 412, "j": 145, "strength": 0.69},
            ],
            "color": "#48f78a",
            "label": "Proper Noun Synapse",
        },
        "punctuation": {
            "pairs": [
                {"i": 15, "j": 501, "strength": 1.89},
                {"i": 288, "j": 44, "strength": 1.56},
                {"i": 102, "j": 367, "strength": 1.23},
                {"i": 456, "j": 78, "strength": 1.01},
                {"i": 201, "j": 333, "strength": 0.88},
            ],
            "color": "#f74848",
            "label": "Punctuation Synapse",
        },
    }


def main():
    # Generate graph topologies (Gx and Gy)
    gx = generate_mock_graph(n_nodes=80, n_links_approx=250)
    gy = generate_mock_graph(n_nodes=80, n_links_approx=180)
    topologies = {"gx": gx, "gy": gy}

    out_dir = os.path.join("..", "frontend", "src", "data")
    os.makedirs(out_dir, exist_ok=True)

    graph_path = os.path.join(out_dir, "graph_topology.json")
    with open(graph_path, "w") as f:
        json.dump(topologies, f, indent=2)
    print(f"Created {graph_path}")
    print(f"  Gx: {len(gx['nodes'])} nodes, {len(gx['links'])} links")
    print(f"  Gy: {len(gy['nodes'])} nodes, {len(gy['links'])} links")

    synapse_path = os.path.join(out_dir, "synapse_labels.json")
    synapse_labels = generate_mock_synapse_labels()
    with open(synapse_path, "w") as f:
        json.dump(synapse_labels, f, indent=2)
    print(f"Created {synapse_path}")
    for concept, data in synapse_labels.items():
        print(f"  {concept}: {len(data['pairs'])} pairs")


if __name__ == "__main__":
    main()
