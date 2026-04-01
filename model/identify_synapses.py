"""
Identify monosemantic synapse pairs in the trained BDH model.

Runs inference on concept-specific text vs. neutral text and finds (i,j) neuron
pairs where the σ matrix changes most specifically for each concept.

Output: synapse_labels.json with concept labels for specific neuron pairs.

Usage:
    python identify_synapses.py
    python identify_synapses.py --checkpoint checkpoints/tiny_bdh.pt
"""

import argparse
import json
import os

import torch

from bdh_tiny import BDHTinyModel, TINY_CONFIG


def find_concept_synapses(model, concept_texts, other_texts, head=0, layer=0, top_n=10):
    """Find neuron pairs that respond specifically to concept text.

    Args:
        model: trained BDHTinyModel
        concept_texts: list of strings containing the target concept
        other_texts: list of neutral/generic strings
        head: attention head to analyze
        layer: which layer to examine
        top_n: number of top synapse pairs to return

    Returns:
        list of (i, j, strength) tuples
    """
    N = model.config.N

    def compute_sigma(text):
        tokens = torch.tensor(
            [[b for b in text.encode("utf-8")[:256]]], dtype=torch.long
        )
        with torch.no_grad():
            _, layer_data = model.forward_with_hooks(tokens)
        ld = layer_data[layer]
        x = ld["x_sparse"][0, head]  # (T, N)
        y = ld["y_sparse"][0, head]  # (T, N)
        sigma = torch.zeros(N, N)
        for t in range(x.shape[0]):
            sigma += torch.outer(y[t], x[t])
        return sigma

    # Accumulate sigma for concept and neutral texts
    sigma_concept = sum(compute_sigma(t) for t in concept_texts)
    sigma_other = sum(compute_sigma(t) for t in other_texts)

    # Synapses that respond specifically to concept (high contrast)
    delta = sigma_concept - sigma_other
    flat = delta.abs().flatten()
    top_indices = flat.topk(top_n).indices

    pairs = []
    for idx in top_indices:
        i = int(idx // N)
        j = int(idx % N)
        strength = float(delta[i, j])
        pairs.append({"i": i, "j": j, "strength": round(strength, 4)})

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny_bdh.pt")
    parser.add_argument("--output", default="../frontend/src/data/synapse_labels.json")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", TINY_CONFIG)
    model = BDHTinyModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt.get('iter', '?')}")

    # Define concept categories and their text samples
    concepts = {
        "currency": {
            "concept_texts": [
                "dollar euro pound yen USD EUR GBP JPY",
                "the price rose to fifty dollars and sixty euros",
                "currency exchange rates pound sterling yen",
            ],
            "other_texts": [
                "the cat sat on the mat",
                "it was a bright and warm summer day",
                "he walked slowly through the forest path",
            ],
            "color": "#f7c948",
            "label": "Currency Synapse",
        },
        "proper_noun": {
            "concept_texts": [
                "London Paris Tokyo New York Berlin",
                "King Richard Duke of York Henry",
                "Romeo Juliet Hamlet Macbeth Othello",
            ],
            "other_texts": [
                "and but or the a is was were",
                "quickly slowly carefully gently softly",
                "small large old new bright dark",
            ],
            "color": "#48f78a",
            "label": "Proper Noun Synapse",
        },
        "punctuation": {
            "concept_texts": [
                "Hello! What? Yes. No! Really? Fine.",
                "Stop! Wait! Listen! Go! Come! Help!",
            ],
            "other_texts": [
                "the morning sun was warm and gentle",
                "birds sang sweetly in the tall trees",
            ],
            "color": "#f74848",
            "label": "Punctuation Synapse",
        },
    }

    result = {}
    for concept_name, config_data in concepts.items():
        pairs = find_concept_synapses(
            model,
            concept_texts=config_data["concept_texts"],
            other_texts=config_data["other_texts"],
        )
        result[concept_name] = {
            "pairs": pairs,
            "color": config_data["color"],
            "label": config_data["label"],
        }
        print(
            f"{concept_name}: top synapse pairs = "
            + ", ".join(f"({p['i']},{p['j']})" for p in pairs[:5])
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved synapse labels to {args.output}")


if __name__ == "__main__":
    main()
