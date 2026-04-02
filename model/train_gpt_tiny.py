"""
Training script for tiny GPT baseline on Tiny Shakespeare.

Mirrors train_tiny.py but trains the standard transformer model.
Expected training time: ~10-15 min on CPU (tiny model).

Usage:
    python train_gpt_tiny.py
    python train_gpt_tiny.py --max_iters 2000
"""

import argparse
import os
import time

import numpy as np
import requests
import torch

from gpt_tiny import TINY_GPT_CONFIG, create_gpt_model

MAX_ITERS = 5000
BATCH_SIZE = 64
BLOCK_SIZE = 256
LR = 3e-3
EVAL_INTERVAL = 250
EVAL_ITERS = 50
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tiny_shakespeare.txt"
CKPT_PATH = "checkpoints/tiny_gpt.pt"


def download_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        return
    print("Downloading Tiny Shakespeare...")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"Saved {len(resp.text):,} characters")


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    n = len(data)
    split = int(n * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Data: {n:,} bytes, train={len(train_data):,}, val={len(val_data):,}")
    return train_data, val_data


def get_batch(data, batch_size, block_size, device):
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].copy()).long() for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].copy()).long() for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, device):
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            total += loss.item()
        losses[name] = total / eval_iters
    model.train()
    return losses


def train(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    download_data()
    train_data, val_data = load_data()

    model = create_gpt_model(TINY_GPT_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    best_val = float("inf")
    t0 = time.time()

    for it in range(args.max_iters):
        if it % EVAL_INTERVAL == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, EVAL_ITERS, device)
            elapsed = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {elapsed:.1f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": TINY_GPT_CONFIG,
                     "iter": it, "best_val": best_val},
                    CKPT_PATH,
                )
                print(f"  → saved (val loss {best_val:.4f})")

        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.1f}s, best val loss: {best_val:.4f}")

    # Show activation density stats
    model.eval()
    with torch.no_grad():
        sample_x, _ = get_batch(train_data, 1, BLOCK_SIZE, device)
        _, layer_acts = model.forward_with_activations(sample_x)
        for i, act in enumerate(layer_acts):
            density = (act.abs() > 1e-6).float().mean().item()
            print(f"Layer {i} MLP activation density: {density*100:.1f}% active")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()
    train(args)
