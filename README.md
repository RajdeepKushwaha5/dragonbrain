# Dragon Brain

**An interactive, in-browser explorer that lets you see inside a post-transformer neural network (BDH) and compare it against a standard GPT — in real time.**

Built for the **IIT Ropar x Pathway — Beyond Transformers Hackathon** (Path A: Visualization & Inner Worlds)

**Live demo:** [dragonbrain.vercel.app](https://dragonbrain.vercel.app/)

---

## What Is This?

Dragon Brain is a web app that runs two tiny neural networks (BDH and GPT) entirely in your browser. As you type, it shows you what's happening inside each model — which neurons fire, how attention flows, how memory forms, and what each model predicts next.

Everything updates live. No server. No API calls. Both models run via ONNX WebAssembly on your machine.

---

## What Is BDH?

**Baby Dragon Hatchling** (Kosowski et al. 2025) is a new neural architecture that works differently from a transformer:

- **Sparse activations** — Only ~5-15% of neurons fire per token (transformers fire ~97-100%)
- **Fixed-size memory** — Uses a Hebbian matrix (constant 4 MB) instead of a KV-cache (grows with every token)
- **Self-organizing structure** — During training, the network forms a brain-like graph with hub neurons — no one programs this
- **Interpretable synapses** — Individual matrix entries reliably activate for specific concepts (currency, punctuation, proper nouns)
- **Learns during inference** — The memory matrix updates as you type, not just during training

**Paper:** [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)

---

## Screenshots

<img width="1795" height="645" alt="Main interface showing sparse activation comparison" src="https://github.com/user-attachments/assets/34275c00-4761-4a26-b44d-63688f229414" />
<img width="1847" height="975" alt="Graph brain and Hebbian heatmap panels" src="https://github.com/user-attachments/assets/4f73c726-77ed-4a83-b6a3-178bc9dae66f" />
<img width="1870" height="974" alt="Memory scaling and attention panels" src="https://github.com/user-attachments/assets/012e690d-421e-470d-9d51-b400775e72ae" />
<img width="1782" height="522" alt="Training curves and text generation" src="https://github.com/user-attachments/assets/19a61fa4-5c5a-4f06-b49c-b8073b40e13b" />

---

## Features

### Visualization Panels

| Panel | What It Shows |
|---|---|
| **Sparse Activation** | Side-by-side neuron grids — BDH (1024 neurons, sparse) vs GPT (256 neurons, dense). Includes a sparsity sparkline over the last 60 keystrokes. |
| **Emergent Graph** | D3.js force-directed graph of the model's internal wiring (Gx: feedforward circuit, Gy: memory readout). Toggle between untrained and trained states to see hub emergence. |
| **Hebbian Memory** | 64x64 heatmap of the co-activation matrix. Hover to see synapse concept labels. Shows how many synapses strengthened per keystroke. |
| **Memory Scaling** | Chart comparing BDH's constant memory (flat line) vs GPT's KV-cache (grows linearly). Crossover at ~4,096 tokens. |
| **Attention Pattern** | Token-by-token causal attention heatmap. Switch between heads. |
| **Synapse Tracer** | Real-time concept-level firing decomposition. Shows which named synapse pairs (currency, proper noun, punctuation) are actively driving predictions, with firing direction and strength. |
| **Training Curves** | Validation loss over training for both models, rendered on canvas with hover tooltips. |

### Interactive Tools

| Tool | What It Does |
|---|---|
| **Text Generation** | Type a prompt and watch both models generate text side-by-side. Each character is color-coded by prediction loss (green = confident, red = surprised). |
| **Teach Experiment** | Quantified test: clears BDH memory, measures baseline loss, feeds repeated phrases to build memory, then measures again. Shows the exact improvement percentage. |
| **Demo Mode** | One-click auto-typing of Shakespeare with all panels animating in sync. |
| **Teach Mode** | Feed repeated phrases and watch predictions shift as the model memorizes. |
| **Three Prediction Rows** | BDH Raw, BDH with memory correction, and GPT — with an indicator when memory changes the top prediction. |
| **Guided Tour** | 12-step interactive tour that walks through every panel — expands collapsed sections automatically, highlights targets, and explains what to look for. |
| **Quick Guide** | One-click overlay with step-by-step instructions. Escape to close. |
| **About Page** | Full architecture deep-dive with glossary of all terms, BDH vs Transformer comparison table, layer pipeline formulas, configuration details, and references. |

### Other

- Layer and head switching (L1/L2, H1/H2)
- Per-token inference timer (milliseconds)
- Collapsible panel sections (progressive disclosure)
- Cross-session memory (σ persists via IndexedDB between browser sessions)
- Synapse concept labels on the Hebbian heatmap with activation timeline
- Model transparency bar (parameter counts, training data, σ approximation disclaimer)
- Byte-level tokenization (every character visible)
- Fully client-side (zero server calls)

---

## How It Works

1. **You type text** — each character becomes a byte token (0-255)
2. **Both models run** — BDH and GPT process the same tokens via ONNX in parallel
3. **Panels update** — sparse grids highlight, graph nodes glow, heatmap shifts, memory chart extends
4. **Predictions appear** — three rows showing what each model thinks comes next
5. **Memory learns** — BDH's Hebbian matrix updates with each keystroke (the model remembers patterns)

---

## What This Demonstrates About BDH

| Claim from the Paper | How We Show It |
|---|---|
| BDH neurons are sparse | Real GPT model runs side-by-side: ~97% density vs BDH's ~5-15% |
| Memory is constant-size | Memory Scaling panel: flat line at 4 MB vs KV-cache growing at 1 KB/token |
| Synapses are interpretable | Concept labels on the heatmap (hover to see what each synapse encodes) |
| The model learns during inference | Per-keystroke memory update indicator + Teach Experiment with measured improvement |
| Network structure is self-organizing | Graph Evolution toggle: random init vs trained (hub neurons, scale-free topology) |

---

## Quick Start

### Run Locally

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. Both ONNX models load automatically in the browser.

### Production Build

```bash
cd frontend
npm run build       # Output in dist/
npm run preview     # Preview locally
```

### Train the Models (Optional)

The pre-trained ONNX models are already included. If you want to retrain:

```bash
cd model
pip install -r requirements.txt

# Train BDH (~229K params, ~5000 iterations)
python train_tiny.py

# Train GPT baseline (~148K params, same data)
python train_gpt_tiny.py

# Export to ONNX
python export_onnx.py --output ../frontend/public/model.onnx
python export_gpt_onnx.py

# Extract supporting data
python extract_weights.py       # Encoder/lm_head for memory-corrected inference
python extract.py               # Graph topology
python extract_evolution.py     # Graph evolution snapshots
python identify_synapses.py     # Synapse concept labels
```

### Deploy to Vercel

1. Push to GitHub
2. Import on [vercel.com](https://vercel.com)
3. Deploy (root directory = repo root)

The `vercel.json` handles build commands, COOP/COEP headers (required for ONNX WASM threading), and SPA rewrites.

---

## Project Structure

```
dragonbrain/
├── frontend/                      Svelte + Vite + D3.js
│   ├── src/
│   │   ├── App.svelte             Root component: model loading, inference, layout
│   │   ├── components/
│   │   │   ├── SparsePanel        BDH vs GPT neuron activation grids
│   │   │   ├── GraphBrain         Force-directed graph + evolution toggle
│   │   │   ├── HebbianHeatmap     Memory heatmap + synapse labels
│   │   │   ├── MemoryPanel        Memory scaling comparison chart
│   │   │   ├── AttentionPanel     Causal attention heatmap
│   │   │   ├── GeneratePanel      Side-by-side text generation
│   │   │   ├── TeachExperiment    Quantified memory learning test
│   │   │   ├── TrainingCurves     Validation loss curves
│   │   │   ├── TokenInput         Text input with token display
│   │   │   ├── LayerSelector      Layer/head switching
│   │   │   ├── GuidedTour         Interactive tutorial overlay
│   │   │   ├── AboutPage          Architecture deep-dive
│   │   │   └── ...                Supporting components
│   │   ├── lib/
│   │   │   ├── BDHModel.js        BDH ONNX inference + Hebbian memory
│   │   │   ├── GPTModel.js        GPT ONNX inference
│   │   │   ├── tokenizer.js       Byte-level tokenizer
│   │   │   ├── stores.js          Svelte state management
│   │   │   └── activation_math.js Sparsity/activation utilities
│   │   └── data/                  Pre-extracted model data (JSON)
│   └── public/
│       ├── model.onnx             Trained BDH weights (~0.9 MB)
│       └── transformer.onnx       Trained GPT weights (~1.1 MB)
│
├── model/                         Python: training & export (offline)
│   ├── bdh_tiny.py                BDH model definition
│   ├── gpt_tiny.py                GPT model definition
│   ├── train_tiny.py              BDH training script
│   ├── train_gpt_tiny.py          GPT training script
│   ├── export_onnx.py             BDH ONNX export
│   ├── export_gpt_onnx.py         GPT ONNX export
│   └── ...                        Data extraction scripts
│
├── vercel.json                    Deployment config
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Models | PyTorch (BDH ~229K params, GPT ~148K params) |
| Inference | ONNX Runtime Web (WebAssembly) |
| Frontend | Svelte 4.2, Vite |
| Visualization | D3.js v7.9, Canvas |
| Design | Custom CSS, dark theme (Inter + JetBrains Mono) |
| Deployment | Vercel (static site, COOP/COEP headers) |

---

## Model Details

### BDH (Primary Model)

| Parameter | Value |
|---|---|
| Layers | 2 |
| Embedding dim | 64 |
| Heads | 2 |
| Neurons per head | 512 (1,024 total) |
| Activation | ReLU (sparse) |
| Vocab | 256 (byte-level) |
| Parameters | ~229K |
| Val loss | 1.5309 |

### GPT (Baseline)

| Parameter | Value |
|---|---|
| Layers | 2 |
| Embedding dim | 64 |
| Heads | 2 |
| MLP hidden | 256 |
| Activation | GELU (dense) |
| Vocab | 256 (byte-level) |
| Parameters | ~148K |
| Val loss | 1.6236 |

Both models are trained on Tiny Shakespeare (the same dataset, same training setup) so the comparison is fair.

---

## The BDH Layer Pipeline

```
Input (D=64) → Decoder_x → ReLU → x (sparse, ~5-15% active)
                                     ↓
                         Attention(Q=x, K=x, V=input) → linear, no softmax
                                     ↓
                         Decoder_y → ReLU → y gated by x
                                     ↓
                         Memory update: σ += y ⊗ x (Hebbian, fixed size)
                                     ↓
                         Encoder → residual add → LayerNorm → next layer
```

---

## Limitations

- **Small scale:** 229K parameters is educational-scale. The paper's full models are millions of parameters, where sparsity is more dramatic.
- **Memory chart is computed, not measured:** The scaling panel uses known constants (σ size, KV-cache growth rate), not actual browser memory usage.
- **Model merging is explained, not demoed:** True composable merging requires separately trained domain-specific models.
- **Block size is 256 tokens:** Can't demonstrate BDH's infinite-context advantage at this scale.
- **Byte tokenizer:** Limits output quality compared to BPE-based models.

---

## References

- Kosowski et al. 2025, *"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"* — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
- Official BDH repository — [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- Pathway — [pathway.com](https://pathway.com)

---

## Author

**[Rajdeep Singh](https://rajdeep-singh.vercel.app/)** — Built for the Beyond Transformers Hackathon (Path A: Visualization & Inner Worlds)
