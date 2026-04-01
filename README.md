# 🐉 Dragon Brain

**Interactive explorer of the Baby Dragon Hatchling (BDH) architecture — a post-transformer neural network inspired by neuroscience.**

Built for the **IIT Ropar × Pathway — Beyond Transformers Hackathon**

> Type text → watch sparse neurons fire → see graphs emerge → observe Hebbian memory build — all in real time, entirely in your browser.

---

## Live Demo

👉 **[dragonbrain.vercel.app](https://dragonbrain.vercel.app/)**

---

## What is BDH?

The **Baby Dragon Hatchling** (Kosowski et al. 2025) is a novel neural architecture that replaces the standard transformer MLP with a biologically-inspired sparse activation system:

| Property | Transformer | BDH |
|---|---|---|
| Activation | ~99% neurons fire (GELU/SwiGLU) | ~5% neurons fire at scale (ReLU) |
| Memory | KV-cache (grows linearly with T) | σ matrix (fixed O(N²), Hebbian) |
| Structure | Dense, homogeneous weights | Emergent scale-free graph with hubs |
| Interpretability | Requires SAE probing (post-hoc) | Natively monosemantic neurons |
| Attention | Softmax O(T²d), separate Q/K/V | Linear + RoPE, Q=K=x, O(T) recurrent |
| Parameters | Per-layer weights | Shared encoder/decoder across all layers |
| Tokenization | BPE (32k–128k vocab) | Byte-level (256 vocab) |

**Paper:** Kosowski et al. 2025 — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)

---

## Features

### 4 Interactive Visualization Panels

**Sparse Activation Panel** — Side-by-side 32×32 neuron grids comparing BDH's sparse activations vs a transformer's dense pattern. Real-time sparsity percentage and active neuron count.

**Emergent Graph Panel** — Force-directed D3.js graph of the model's internal wiring:
- **Gx (Thought Flow)** — `Encoder @ Decoder_x` — feedforward causal circuit
- **Gy (Memory Echo)** — `Decoder_y^T @ Decoder_x` — Hebbian memory readout graph
- 80 hub neurons extracted from trained weights, self-loops filtered. Active neurons highlight yellow in real time. Zoom & drag supported.

**Hebbian Memory (σ) Panel** — 64×64 heatmap of the co-activation matrix with viridis color scale. Synapse labels identify concept-specific neuron pairs (currency, proper nouns, punctuation). Clear and rebuild to watch memory form from scratch.

**Attention Pattern Panel** — Token×Token heatmap of raw causal attention scores (no softmax). Switch between heads to compare attention behaviors. Strictly lower-triangular due to causal mask.

### Additional Features

- **Next-token predictions** — top-5 predicted next bytes with probabilities shown after each keystroke
- **Inference timer** — per-token latency displayed in milliseconds
- **Quick Guide** — overlay guide accessible from the header explaining how to use each panel
- **About page** — deep-dive into BDH architecture with formulas, math, and comparisons
- **Byte-level tokenization** — every character is a token (0–255), visible as a token stream
- **Layer & head switching** — L1/L2 and H1/H2 buttons to explore different model components
- **Insight badges** — contextual explanations that appear based on data patterns
- **Fully client-side** — zero server calls, runs entirely via ONNX Runtime WebAssembly
- **Mock mode** — works without a trained model, generating realistic sparse patterns for development

---

## Architecture

### The BDH Layer Pipeline

```
Input v* (D=64) → Decoder_x → ReLU → x (sparse, N=512/head)
                                        ↓
                            Attention(Q=x, K=x, V=v*) → a* (linear, no softmax)
                                        ↓
                            Decoder_y → ReLU → y ⊙ x (gated readout)
                                        ↓
                            σ += y ⊗ x (Hebbian memory update)
                                        ↓
                            Encoder → residual add → LayerNorm → v* (next layer)
```

### Key Formulas

| Step | Formula | Description |
|---|---|---|
| Sparse activation | `x = ReLU(v* @ Decoder_x)` | Only ~5% nonzero at scale |
| Linear attention | `a* = RoPE(x) @ RoPE(x)^T · v*` | No softmax, Q=K=x |
| Gated readout | `y = ReLU(LN(a*) @ Decoder_y) ⊙ x` | Only fires where x is active |
| Hebbian update | `σ += y ⊗ x` | Fixed-size memory, doesn't grow |
| Residual | `v* = LN(v* + LN(y @ Encoder))` | Layer output |

---

## Project Structure

```
dragonbrain/
├── frontend/                  # Svelte + Vite + D3.js (deployable)
│   ├── src/
│   │   ├── App.svelte         # Root layout, routing, guide overlay, header/footer
│   │   ├── app.css            # Design system (true black/white theme)
│   │   ├── main.js            # Entry point
│   │   ├── lib/
│   │   │   ├── BDHModel.js          # ONNX inference + logits extraction + mock fallback
│   │   │   ├── tokenizer.js         # Byte-level tokenizer (UTF-8)
│   │   │   ├── stores.js            # Svelte stores + derived computations
│   │   │   └── activation_math.js   # Sparsity/activation utilities
│   │   ├── components/
│   │   │   ├── AboutPage.svelte     # Comprehensive project explainer
│   │   │   ├── TokenInput.svelte    # Text input with token visualization
│   │   │   ├── LayerSelector.svelte # L1/L2/H1/H2 buttons + stats
│   │   │   ├── SparsePanel.svelte   # BDH vs Transformer comparison
│   │   │   ├── NeuronGrid.svelte    # 32×32 neuron activation grid
│   │   │   ├── GraphBrain.svelte    # D3 force-directed graph (Gx/Gy)
│   │   │   ├── HebbianHeatmap.svelte # σ matrix heatmap with tooltips
│   │   │   ├── AttentionPanel.svelte # Causal attention heatmap
│   │   │   ├── InsightBadge.svelte  # Contextual insight messages
│   │   │   └── StatsBar.svelte      # Progress bar component
│   │   └── data/
│   │       ├── graph_topology.json  # Pre-extracted Gx/Gy graph data
│   │       └── synapse_labels.json  # Concept-specific synapse labels
│   ├── public/
│   │   ├── model.onnx             # Trained ONNX model (~10 KB)
│   │   ├── model.onnx.data        # Model weights (~0.9 MB)
│   │   └── .nojekyll              # GitHub Pages compatibility
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── svelte.config.js
│
├── model/                     # Python — training & export (offline only)
│   ├── bdh_tiny.py            # BDH model definition (~229K params)
│   ├── train_tiny.py          # Training on Tiny Shakespeare
│   ├── extract.py             # Gx/Gy graph topology extraction
│   ├── export_onnx.py         # ONNX export for browser inference
│   ├── identify_synapses.py   # Concept-specific synapse identification
│   ├── generate_mock_data.py  # Mock data generation for frontend dev
│   └── requirements.txt
│
├── vercel.json                # Vercel deployment config (COOP/COEP headers)
└── README.md
```

---

## Quick Start

### Frontend (runs immediately with mock data)

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. The app runs in **mock mode** if no ONNX model is present — generating realistic sparse activation patterns. All 4 panels work immediately.

### Training the Model (optional)

```bash
cd model
pip install -r requirements.txt
python train_tiny.py              # Train on Tiny Shakespeare (~5000 iters)
python extract.py                 # Extract Gx/Gy graph topologies
python identify_synapses.py       # Find concept-specific synapses
python export_onnx.py --output ../frontend/public/model.onnx
```

After training, rebuild the frontend — real ONNX inference replaces mock mode automatically.

### Production Build

```bash
cd frontend
npm run build       # Output in dist/
npm run preview     # Preview locally
```

---

## Deployment

### Vercel (Recommended)

The repo includes a [`vercel.json`](vercel.json) that handles everything automatically:

1. Push to GitHub
2. Import the repo on [vercel.com](https://vercel.com)
3. Leave **Root Directory** blank (the `vercel.json` at repo root handles build commands)
4. Deploy

The `vercel.json` configures:
- **Build:** `cd frontend && npm run build`
- **Output:** `frontend/dist`
- **COOP/COEP headers** for ONNX Runtime WASM threading
- **SPA rewrites** for client-side routing

> **Note:** The Python files in `model/` are **not needed at runtime**. The entire app is a static site served from `frontend/dist/`.

### GitHub Pages

```bash
cd frontend
npm run deploy    # Uses gh-pages to publish dist/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch (BDH, ~229K params) |
| Inference | ONNX Runtime Web (WASM) |
| Frontend | Svelte 4.2, Vite 5.4 |
| Visualization | D3.js v7.9 (force graphs, heatmaps) |
| Design | Custom CSS design system (Inter + Orbitron + JetBrains Mono) |
| Theme | True black (#000) + white text + blue accent palette |
| Deployment | Vercel (static, with COOP/COEP headers) |

---

## Model Configuration

| Parameter | Value |
|---|---|
| Layers | 2 |
| Embedding dim (D) | 64 |
| Heads (n_h) | 2 |
| Neurons per head (N) | 512 |
| Total neurons | 1,024 |
| MLP multiplier | 16× |
| Vocab size | 256 (byte-level) |
| Total parameters | ~229K |
| Training data | Tiny Shakespeare |
| Best val loss | 1.5311 (iter 4750) |
| Runtime | ONNX + WebAssembly |
| Max sequence | 256 tokens |

---

## How It Works

1. **Type text** in the input box — each character becomes a byte token (0–255)
2. **Panels update instantly** — sparse activations, graph highlights, Hebbian memory, and attention patterns all respond to each keystroke
3. **Switch layers/heads** — use L1/L2 and H1/H2 to explore different model components; different heads specialise in different patterns
4. **Explore the graph** — toggle Gx (Thought Flow) / Gy (Memory Echo), drag and zoom to inspect hub neurons
5. **Clear memory** — reset the Hebbian σ matrix, then retype to watch memory rebuild from scratch

---

## References

- Kosowski et al. 2025, *"The Dragon Hatchling"* — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
- Official BDH repository — [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- Pathway — [pathway.com](https://pathway.com)

---

## Author

**[Rajdeep Singh](https://rajdeep-singh.vercel.app/)** — Built for the Beyond Transformers Hackathon (Pathway A)
