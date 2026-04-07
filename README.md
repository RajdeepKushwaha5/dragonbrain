# 🐉 Dragon Brain

**Interactive explorer of the Baby Dragon Hatchling (BDH) architecture — a post-transformer neural network that learns during inference, uses fixed-size memory, and organizes into interpretable brain-like graphs.**

Built for the **IIT Ropar × Pathway — Beyond Transformers Hackathon** (Path A: Visualization & Inner Worlds)

> Type text → watch sparse neurons fire → see graphs self-organize → observe Hebbian memory strengthen → compare against a real GPT transformer — all in real time, entirely in your browser.

---

## Live Demo

👉 **[dragonbrain.vercel.app](https://dragonbrain.vercel.app/)**

---

## What Insight This Reveals About BDH

Dragon Brain makes **five architectural breakthroughs** viscerally visible:

1. **Sparsity is real and dramatic.** BDH fires ~5–15% of neurons per token while a real GPT transformer (trained on the same data, running side-by-side in the same browser) fires ~97–100%. This isn't a simulated comparison — both models run live ONNX inference on every keystroke.

2. **Memory stays constant.** BDH's Hebbian σ matrix is 4 MB regardless of how many tokens you process. The KV-cache of an equivalent transformer grows by 1 KB per token. The Memory Scaling panel charts this divergence in real time with a crossover marker at T≈4,096.

3. **The brain builds itself.** Toggle the Evolution view in the Graph panel to see the network before training (random noise, no hubs, max degree 93) vs after training (scale-free graph, hub neurons, max degree 208). No one programmed this structure — it emerged.

4. **Synapses are interpretable.** Hover over coloured borders on the Hebbian heatmap to see what each synapse encodes (currency, punctuation, proper nouns). These aren't post-hoc probes — they're direct σ matrix entries that reliably activate for specific concepts.

5. **The model learns as you type.** Watch the Δσ indicator on the Hebbian panel — it reports how many synapses strengthened with each keystroke. The sparsity sparkline tracks the model's "surprise" over time: novel text activates more neurons, predictable text activates fewer — an uncertainty signal baked into the architecture.

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
| Merging | Requires fine-tuning tricks | Concatenate independently trained models |
| Tokenization | BPE (32k–128k vocab) | Byte-level (256 vocab) |

**Paper:** Kosowski et al. 2025 — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)

---

## BDH Key Differentiators Demonstrated

| # | Capability | How We Demonstrate It |
|---|---|---|
| 1 | **Constant-size memory** | Memory Scaling panel: σ flat at 4MB vs KV-cache growing at 1KB/token |
| 2 | **Native sparsity** | Real GPT ONNX model runs side-by-side — ~97% density vs BDH's ~5–15% |
| 3 | **Monosemantic synapses** | Concept-labeled synapse borders on the Hebbian heatmap (currency, nouns, punctuation) |
| 4 | **Inference-time learning** | Δσ indicator + sparsity sparkline showing live Hebbian updates per keystroke |
| 5 | **σ-Learned predictions** | Three prediction rows (BDH Raw / σ-Learned / GPT) — ⇄ shifted indicator when σ changes the top prediction |
| 6 | **Composable merging** | Explained in the About page (Section 7.1 of the paper — requires separate training) |

---

## Features

### 5 Interactive Visualization Panels

**Sparse Activation Panel** — Side-by-side neuron grids: BDH (32×32, ~1024 neurons, sparse ReLU) vs a real GPT transformer (16×16, 256 neurons, dense GELU). Both models run live ONNX inference on the same input. Sparsity sparkline tracks activation density over the last 60 keystrokes. Insight badges explain what the sparsity level means.

**Emergent Graph Panel** — D3.js force-directed graph of the model's internal wiring:
- **Gx (Thought Flow)** — `Encoder @ Decoder_x` — feedforward causal circuit
- **Gy (Memory Echo)** — `Decoder_y^T @ Decoder_x` — Hebbian memory readout graph
- **Evolution toggle** — switch between random initialization (untrained) and trained state to see hub emergence
- 80 hub neurons, active ones highlight yellow in real time. Zoom & drag supported.

**Hebbian Memory (σ) Panel** — 64×64 heatmap of the co-activation matrix with viridis color scale. Synapse concept labels on hover. **Δσ indicator** shows how many synapses strengthened per token and the max change magnitude. Clear and rebuild to watch memory form from scratch.

**Memory Scaling Panel** — SVG chart comparing BDH's constant σ memory (flat blue line at 4 MB) against a GPT KV-cache (red diagonal growing at 1 KB/token). Crossover marker at T≈4,096 tokens. Current position updates with each keystroke.

**Attention Pattern Panel** — T×T causal attention heatmap of raw dot products (no softmax). Switch between heads. Strictly lower-triangular (causal mask).

### Additional Features

- **Real GPT comparison** — a separately trained GPT transformer runs in the same browser via ONNX
- **Three prediction rows** — BDH Raw, σ-Learned (Hebbian-corrected), and GPT predictions with ⇄ shifted indicator
- **Demo mode** — one-click Shakespeare auto-typing with all panels animating in sync
- **Teach mode** — feed repeated phrases to build σ memory and watch predictions shift toward learned patterns
- **Inference timer** — per-token latency in milliseconds
- **Sparsity-as-uncertainty sparkline** — novel insight: BDH activation density correlates with input novelty (Section 6.4)
- **Graph evolution** — random init vs trained topology with stats (max degree, avg degree, edge count)
- **Side-by-side token view** — input text and byte tokens displayed together with full token visibility
- **Quick Guide** — 7-step overlay tutorial covering all features including Teach mode and predictions
- **About page** — deep-dive with formulas, all 5 BDH pillars, architecture step-by-step, key concepts
- **Layer & head switching** — L1/L2, H1/H2 exploration
- **Fully client-side** — zero server calls, ONNX Runtime WebAssembly
- **Byte-level tokenization** — every character visible (0–255)
- **Responsive design** — works on desktop and tablet

---
<img width="1795" height="645" alt="image" src="https://github.com/user-attachments/assets/34275c00-4761-4a26-b44d-63688f229414" />
<img width="1847" height="975" alt="image" src="https://github.com/user-attachments/assets/4f73c726-77ed-4a83-b6a3-178bc9dae66f" />
<img width="1870" height="974" alt="image" src="https://github.com/user-attachments/assets/012e690d-421e-470d-9d51-b400775e72ae" />
<img width="1782" height="522" alt="image" src="https://github.com/user-attachments/assets/19a61fa4-5c5a-4f06-b49c-b8073b40e13b" />

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
│   │   ├── App.svelte         # Root: dual-model loading, inference pipeline, layout
│   │   ├── app.css            # Design system (dark theme, CSS custom properties)
│   │   ├── main.js            # Entry point
│   │   ├── lib/
│   │   │   ├── BDHModel.js          # BDH ONNX inference + Hebbian σ management
│   │   │   ├── GPTModel.js          # GPT ONNX inference (shared ort runtime)
│   │   │   ├── tokenizer.js         # Byte-level tokenizer (UTF-8)
│   │   │   ├── stores.js            # Svelte stores (BDH + GPT + derived)
│   │   │   └── activation_math.js   # Sparsity/activation utilities
│   │   ├── components/
│   │   │   ├── AboutPage.svelte     # 10-section interactive architecture guide
│   │   │   ├── SparsePanel.svelte   # BDH vs GPT real comparison + sparkline
│   │   │   ├── GraphBrain.svelte    # Force graph + evolution toggle
│   │   │   ├── HebbianHeatmap.svelte # σ heatmap + Δσ indicator
│   │   │   ├── MemoryPanel.svelte   # σ vs KV-cache scaling chart
│   │   │   ├── AttentionPanel.svelte # Causal attention heatmap
│   │   │   ├── NeuronGrid.svelte    # Reusable activation grid (any size)
│   │   │   ├── TokenInput.svelte    # Text input with token stream
│   │   │   ├── LayerSelector.svelte # L1/L2/H1/H2 + stats
│   │   │   ├── InsightBadge.svelte  # Contextual insight messages
│   │   │   └── StatsBar.svelte      # Progress bar component
│   │   └── data/
│   │       ├── graph_topology.json  # Pre-extracted Gx/Gy (80 hubs)
│   │       ├── graph_evolution.json # Random init vs trained snapshots
│   │       └── synapse_labels.json  # Concept-specific synapse labels
│   ├── public/
│   │   ├── model.onnx               # Trained BDH weights (~0.9 MB, self-contained)
│   │   ├── transformer.onnx         # Trained GPT weights (~1.1 MB, self-contained)
│   │   ├── bdh_weights.bin          # Encoder + lm_head for σ-modulated inference
│   │   └── .nojekyll
│   └── package.json
│
├── model/                     # Python — training & export (offline only)
│   ├── bdh_tiny.py            # BDH model definition (~229K params)
│   ├── gpt_tiny.py            # GPT baseline definition (~148K params)
│   ├── train_tiny.py          # BDH training on Tiny Shakespeare
│   ├── train_gpt_tiny.py      # GPT training (matched hyperparams)
│   ├── extract.py             # Gx/Gy graph topology extraction
│   ├── extract_evolution.py   # Graph evolution snapshots
│   ├── export_onnx.py         # BDH ONNX export (self-contained)
│   ├── export_gpt_onnx.py     # GPT ONNX export (self-contained)
│   ├── extract_weights.py     # Extract encoder/lm_head for σ inference
│   ├── identify_synapses.py   # Concept-specific synapse identification
│   └── requirements.txt
│
├── vercel.json                # Vercel config (COOP/COEP + SPA rewrites)
└── README.md
```

---

## Quick Start

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`.

### Training Both Models

```bash
cd model
pip install -r requirements.txt

# Train BDH
python train_tiny.py              # ~5000 iters, best val loss 1.5311

# Train GPT baseline (same data, same hyperparams)
python train_gpt_tiny.py          # ~5000 iters, val loss ~1.648

# Export both to ONNX (self-contained, no .data files)
python export_onnx.py --output ../frontend/public/model.onnx
python export_gpt_onnx.py

# Extract weights for σ-modulated inference
python extract_weights.py

# Extract graph data
python extract.py
python extract_evolution.py
python identify_synapses.py
```

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
2. Import on [vercel.com](https://vercel.com) — leave Root Directory blank
3. Deploy

The `vercel.json` configures build commands, COOP/COEP headers (required for ONNX WASM threading), and SPA rewrites.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Models | PyTorch (BDH ~229K params, GPT ~148K params) |
| Inference | ONNX Runtime Web (WASM backend) |
| Frontend | Svelte 4.2, Vite 5.4 |
| Visualization | D3.js v7.9 (force graphs, heatmaps, grids) |
| Design | Custom CSS (Inter + JetBrains Mono) |
| Deployment | Vercel (static, COOP/COEP headers) |

---

## Model Configuration

### BDH (Primary)

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
| Best val loss | 1.5311 (iter 4750) |

### GPT Baseline (Comparison)

| Parameter | Value |
|---|---|
| Layers | 2 |
| Embedding dim (D) | 64 |
| Heads (n_h) | 2 |
| MLP hidden dim | 256 (4× D) |
| Activation | GELU (~97–100% density) |
| Vocab size | 256 (byte-level) |
| Total parameters | ~148K |
| Val loss | ~1.648 (iter 4999) |

---

## How It Works

1. **Type text** — each character becomes a byte token (0–255), shown side-by-side with the input
2. **Both models run** — BDH and GPT process the same tokens via ONNX in parallel
3. **Read predictions** — three rows: BDH Raw, σ-Learned (Hebbian-corrected), and GPT. A ⇄ shifted indicator appears when σ changes the top prediction
4. **5 panels update** — sparse grids, graph highlights, Hebbian heatmap, memory chart, attention
5. **Switch layers/heads** — L1/L2 and H1/H2 show different model internals
6. **Watch the graph evolve** — toggle Evolution to compare random init vs trained topology
7. **Clear memory** — reset σ, then retype to watch Hebbian memory rebuild in real time
8. **Demo mode** — click ▶ for auto-typing Shakespeare with all panels animating
9. **Teach mode** — click ✏ to feed repeated phrases and watch σ-Learned predictions shift

---

## Limitations & Future Scope

### Current Limitations

- **Model scale:** 229K parameters is too small for BDH to fully exhibit paper-scale properties (the paper uses models with millions of parameters). Sparsity may be higher than the reported ~5% at full scale.
- **Memory chart is theoretical:** The Memory Scaling panel computes σ size and KV-cache growth from known constants — it doesn't measure actual GPU/browser memory usage.
- **Composable merging is explained, not demonstrated:** True model merging requires separately training two domain-specific models and concatenating them. This is documented in the About page but not implemented as a live demo.
- **No long-context demo:** The model's block size is 256 tokens. Demonstrating BDH's infinite context advantage requires larger models and longer sequences.
- **Byte-level tokenizer:** Limits the vocabulary to 256 characters, making the text generation less meaningful than BPE-tokenized models.

### Future Scope

- **Scale up:** Train larger BDH models (1M+ params) to show true 5% sparsity
- **Long-context demo:** Implement recurrent inference mode to process sequences beyond block size
- **Composable merging demo:** Train two domain-specific models, merge them live, show combined capabilities
- **3D visualization:** Three.js walkthrough of the full computation graph
- **Memory profiling:** Measure actual WASM memory usage to validate the σ vs KV-cache comparison empirically

---

## References

- Kosowski et al. 2025, *"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"* — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
- Official BDH repository — [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- Pathway — [pathway.com](https://pathway.com)
- Transformer Explainer (Georgia Tech) — [poloclub.github.io/transformer-explainer](https://poloclub.github.io/transformer-explainer) (inspiration)

---

## Author

**[Rajdeep Singh](https://rajdeep-singh.vercel.app/)** — Built for the Beyond Transformers Hackathon (Path A: Visualization & Inner Worlds)
