# 🐉 Dragon Brain

**Interactive explorer of the Baby Dragon Hatchling (BDH) architecture — a post-transformer neural network inspired by neuroscience.**

Built for the **IIT Ropar × Pathway — Beyond Transformers Hackathon**

> Type text → watch sparse neurons fire → see graphs emerge → observe Hebbian memory build — all in real time, entirely in your browser.

---

## Live Demo

👉 **[Try Dragon Brain](https://dragonbrain.vercel.app/)**

---

## What is BDH?

The **Baby Dragon Hatchling** (Kosowski et al. 2025) is a novel neural architecture that replaces the standard transformer MLP with a biologically-inspired sparse activation system:

| Property | Transformer | BDH |
|---|---|---|
| Activation | ~99% neurons fire (GELU/SwiGLU) | ~5% neurons fire (ReLU) |
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

**Sparse Activation Panel** — Side-by-side 32×32 neuron grids comparing BDH's sparse activations (~5% active) vs a transformer's dense pattern (~97% active). Real-time sparsity percentage and active neuron count.

**Emergent Graph Panel** — Force-directed D3.js graph visualization of the model's internal wiring:
- **Gx (Thought Flow)** — `Encoder @ Decoder_x` — feedforward causal circuit
- **Gy (Memory Echo)** — `Decoder_y^T @ Decoder_x` — Hebbian memory readout graph
- Hub neurons (large circles) emerge spontaneously from random initialization. Active neurons highlight yellow in real time.

**Hebbian Memory (σ) Panel** — 64×64 heatmap of the co-activation matrix with viridis color scale. Synapse labels identify concept-specific neuron pairs (currency, proper nouns, punctuation). Clear and rebuild to watch memory form from scratch.

**Attention Pattern Panel** — Token×Token heatmap of raw causal attention scores (no softmax). Switch between heads to compare attention behaviors. Strictly lower-triangular due to causal mask.

### Additional Features

- **Byte-level tokenization** — every character is a token (0–255), no tokenizer needed
- **Layer & head switching** — L1/L2 buttons and H1/H2 buttons to explore different model components
- **Token counter & sparsity stats** — real-time statistics as you type
- **Insight badges** — contextual explanations that appear based on the data patterns
- **Comprehensive About page** — deep-dive into BDH architecture with formulas, comparisons, and usage guide
- **Fully client-side** — zero server calls, runs entirely via ONNX Runtime WebAssembly
- **Mock mode** — works immediately without a trained model, generating realistic sparse patterns

---

## Architecture

### The BDH Layer Pipeline

```
Input v* (D=64) → Decoder_x → ReLU → x (sparse, N=512/head, ~5% nonzero)
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
| Sparse activation | `x = ReLU(v* @ Decoder_x)` | Only ~5% nonzero |
| Linear attention | `a* = RoPE(x) @ RoPE(x)^T · v*` | No softmax, Q=K=x |
| Gated readout | `y = ReLU(LN(a*) @ Decoder_y) ⊙ x` | Only fires where x is active |
| Hebbian update | `σ += y ⊗ x` | Fixed-size memory, doesn't grow |
| Residual | `v* = LN(v* + LN(y @ Encoder))` | Layer output |

---

## Project Structure

```
dragon-brain/
├── model/                     # Python — training & export (offline only)
│   ├── bdh_tiny.py            # BDH model definition (~230K params)
│   ├── train_tiny.py          # Training on Tiny Shakespeare
│   ├── extract.py             # Gx/Gy graph topology extraction
│   ├── export_onnx.py         # ONNX export for browser inference
│   ├── identify_synapses.py   # Concept-specific synapse identification
│   ├── generate_mock_data.py  # Mock data generation for frontend dev
│   └── requirements.txt
│
├── frontend/                  # Svelte + Vite + D3.js (deployable)
│   ├── src/
│   │   ├── App.svelte         # Root layout, routing, header/footer
│   │   ├── app.css            # Design system (true black/white theme)
│   │   ├── main.js            # Entry point
│   │   ├── lib/
│   │   │   ├── BDHModel.js          # ONNX inference + mock fallback
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
│   │   └── .nojekyll              # GitHub Pages compatibility
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── svelte.config.js
│
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

Opens at `http://localhost:5173`. The app runs in **mock mode** by default — generating realistic sparse activation patterns without a trained model. All 4 panels work immediately.

### Training the Model (optional)

```bash
cd model
pip install -r requirements.txt
python train_tiny.py              # Train on Tiny Shakespeare
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

1. Push to GitHub
2. Import repo on [vercel.com](https://vercel.com)
3. Settings:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Deploy

The Python files in `model/` are **not needed at runtime** — they only train and export the model offline. The entire app is a static site served from `dist/`.

### GitHub Pages

```bash
cd frontend
npm run deploy    # Uses gh-pages to publish dist/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch (BDH, ~230K params) |
| Inference | ONNX Runtime Web (WASM, single-threaded) |
| Frontend | Svelte 4.2, Vite 5.4 |
| Visualization | D3.js v7.9 (force graphs, heatmaps) |
| Design | Custom CSS design system (Inter + Orbitron + JetBrains Mono) |
| Theme | True black (#000) + white text + Tailwind-like accent palette |
| Deployment | Vercel / GitHub Pages (static) |

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
| Total parameters | ~230K |
| Runtime | ONNX + WebAssembly |
| Max sequence | 256 tokens |

---

## References

- Kosowski et al. 2025, *"The Dragon Hatchling"* — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
- Official BDH repository — [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- Pathway — [pathway.com](https://pathway.com)

---

## Author

**[Rajdeep Singh](https://rajdeep-singh.vercel.app/)** — Built for the Beyond Transformers Hackathon (Pathway A)
