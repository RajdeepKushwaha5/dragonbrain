# 🐉 Dragon Brain — Demo Video Script

**Duration:** 3–4 minutes  
**Format:** Screen recording with voiceover  
**URL:** [dragonbrain.vercel.app](https://dragonbrain.vercel.app/)

---

## Full Forms & Terminology Reference

| Term | Full Form / Meaning |
|---|---|
| BDH | Baby Dragon Hatchling (novel neural architecture) |
| GPT | Generative Pre-trained Transformer (baseline comparison) |
| σ (sigma) | Hebbian co-activation memory matrix — fixed-size, doesn't grow |
| Δσ (delta sigma) | Change in σ per token — how many synapses strengthened |
| ReLU | Rectified Linear Unit — `max(0, x)` — produces sparse activations |
| GELU | Gaussian Error Linear Unit — smooth activation used in GPT — nearly all neurons fire |
| RoPE | Rotary Position Embedding — encodes relative token position via rotation |
| Gx | `Encoder × Decoder_x` — feedforward causal circuit (Thought Flow) |
| Gy | `Decoder_y^T × Decoder_x` — Hebbian memory readout graph (Memory Echo) |
| KV-cache | Key-Value cache — a transformer's memory that grows linearly with every token |
| ONNX | Open Neural Network Exchange — format for running models in the browser via WebAssembly |
| Q, K, V | Query, Key, Value — attention components. In BDH, Q = K = x (self-correlation) |
| T | Token count (sequence length) |
| N | Neuron count per head (512 in this model) |
| D | Embedding dimension (64 in this model) |
| Dx | Decoder_x weight matrix — projects embedding into sparse neuron space |
| SAE | Sparse AutoEncoder — a post-hoc tool transformers need for interpretability |
| MRO | Method Resolution Order |
| O(1) | Constant time/space — doesn't grow with input |
| O(T) | Linear in sequence length |
| O(T²) | Quadratic in sequence length |

---

## Scene 1 — Opening (15 seconds)

**[Screen: dragonbrain.vercel.app — page loads, BDH model initializes]**

> "This is Dragon Brain — an interactive explorer for the Baby Dragon Hatchling architecture, or BDH.
>
> BDH is a post-transformer neural network from the Pathway research paper by Kosowski and colleagues. It replaces the standard transformer MLP with a biologically-inspired system that uses sparse activations, fixed-size Hebbian memory, and emergent brain-like graph structure.
>
> Everything you're about to see runs entirely in your browser — two real neural networks, BDH and a standard GPT transformer, are loaded as ONNX models and execute via WebAssembly. No server calls."

---

## Scene 2 — Typing & Token Input (25 seconds)

**[Action: Type "The dollar rose against the euro" slowly]**

> "Let's start by typing. Every character becomes a byte token — a number from 0 to 255. You can see the raw byte tokens appearing on the right side, next to the input box. The 'T' is byte 84, 'h' is byte 104, and so on.
>
> Both models — BDH and GPT — process exactly the same tokens simultaneously via ONNX inference. The inference timer in the header shows per-token latency in milliseconds.
>
> Now look at what happens across all five visualization panels as I type."

---

## Scene 3 — Sparse Activation Panel (35 seconds)

**[Action: Point to the Sparse Activation panel. Both grids are visible.]**

> "This is the Sparse Activation panel. Each small square represents one neuron. A lit pixel means that neuron fired — it's active.
>
> On the left is BDH with 1,024 neurons arranged in a 32 by 32 grid. BDH uses ReLU — Rectified Linear Unit — which outputs zero for negative inputs, creating genuine sparsity. Only about 5 to 15 percent of neurons light up. The percentage and count update live.
>
> On the right is a real GPT transformer with 256 neurons in a 16 by 16 grid. GPT uses GELU — Gaussian Error Linear Unit — a smooth activation where nearly every neuron fires. You can see 97 to 100 percent density. This isn't a simulation — it's a separately trained transformer running the same input.
>
> Below the grids is the Sparsity Over Time sparkline. This line chart tracks BDH's activation density over the last 60 keystrokes. When the text is predictable, fewer neurons fire — the line dips. When the input is novel or surprising, more neurons activate — the line rises. The paper calls this 'sparsity as uncertainty' — Section 6.4. It's an uncertainty signal baked into the architecture, not bolted on."

---

## Scene 4 — Emergent Graph Panel (45 seconds)

**[Action: Show the Emergent Graph panel. Toggle between Gx and Gy. Show Evolution.]**

> "This is the Emergent Graph panel — a force-directed network visualization built with D3.js showing how 80 hub neurons are wired together.
>
> There are two graph modes:
>
> **Thought Flow, or Gx** — computed as Encoder times Decoder_x. This is the feedforward causal circuit. It shows how neurons propagate computation through the equation x equals ReLU of v-star times Decoder_x. Blue edges are excitatory connections — they amplify signal. Pink edges are inhibitory — they suppress it. Gold-highlighted nodes are neurons that are currently active as you type.
>
> **Memory Echo, or Gy** — computed as Decoder_y-transpose times Decoder_x. This is the Hebbian memory readout graph — how σ memory gets decoded back into y activations to influence the model's output.
>
> The stats show '80 hubs, 3,702 edges.' This power-law degree distribution — a few hub neurons with many connections, most neurons with few — is called a scale-free network. It mirrors biological brains, the internet, and social networks. Critically, nobody designed this structure. It self-organized from random weights during training.
>
> Now watch what happens when I click **Evolution**. Toggle to 'Random Init' — this is the network before training. It's random noise: no hubs, no structure. Max degree around 93. Toggle to 'Trained, iteration 4,750' — hub neurons have emerged. Max degree jumps to 208, average degree to 100. That's the scale-free graph the paper describes in Section 5.
>
> The **Communities** toggle colors neurons by detected community — clusters of tightly interconnected neurons that may specialize in different linguistic functions."

---

## Scene 5 — Hebbian Memory Panel (40 seconds)

**[Action: Point to the Hebbian heatmap. Show Δσ indicator. Click synapse concepts.]**

> "This is the Hebbian Memory panel, labeled sigma or σ. The core equation is σ plus-equals outer product of y and x — neurons that fire together wire together. This is Donald Hebb's 1949 principle implemented as a running co-activation matrix.
>
> The 64 by 64 heatmap shows the top hub neurons from σ, which lives in R to the power N times N — 512 by 512 in the full model. Brighter cells mean stronger connections; the color scale runs from zero (dark) through weak to strong.
>
> Watch the Δσ indicator as I type — it flashes to show how many synapses strengthened with each token and the maximum change magnitude. This is real-time learning with no gradient updates, no backpropagation.
>
> Now, the colored borders on certain cells — these are **discovered synapse concepts**. Click 'Currency Synapse' — these are neuron pairs that co-activate specifically when the model processes currency-related characters like dollar signs and numbers. 'Proper Noun Synapse' fires for capitalized words. 'Punctuation Synapse' activates on periods, commas, and spaces. These aren't post-hoc probes from a Sparse AutoEncoder — they're direct σ matrix entries with interpretable meaning. That's native monosemanticity.
>
> The key architectural advantage: unlike a transformer's KV-cache, which grows by about 1 kilobyte per token, σ is **fixed-size** regardless of context length. You could process a million tokens and σ would still be 4 megabytes."

**[Action: Click "Clear" button, then retype a few characters]**

> "Watch — I clear σ, and the heatmap goes dark. As I retype, the memory rebuilds from scratch. Synapses strengthen in real time."

---

## Scene 6 — Attention & Memory Scaling Panels (30 seconds)

**[Action: Show the Attention Pattern panel. Switch heads. Show Memory Scaling.]**

> "The **Attention Pattern** panel shows a T by T heatmap of raw attention scores. This is linear causal attention with RoPE — Rotary Position Embedding — and no softmax. Since Q equals K equals x in BDH, the heatmap shows raw pairwise inner products between token positions.
>
> The strictly lower-triangular structure comes from the causal mask — each token can only attend to previous tokens. Brighter cells near the diagonal mean stronger attention to recent context. Switch between H1 and H2 heads — different heads often specialize in different patterns.
>
> Below that is the **Memory Scaling** chart. The flat blue line labeled σ is BDH's memory — fixed at 4 megabytes regardless of sequence length. The red diagonal line labeled KV is a transformer's key-value cache growing at 1 kilobyte per token. The crossover marker at T approximately 4,000 shows where the transformer starts using more memory than BDH. At the paper's full scale with 32,768 neurons, that crossover happens at just 400 tokens."

---

## Scene 7 — Layer & Head Switching (15 seconds)

**[Action: Click L1/L2 and H1/H2 buttons in the Layer Selector]**

> "The Layer Selector lets you explore different depths. L1 and L2 switch between layer 1 and layer 2. H1 and H2 switch attention heads. The stats update live — sparsity percentage, active neuron count, and total tokens processed. Different layers and heads capture different aspects of the input."

---

## Scene 8 — Predictions & σ-Learned Row (30 seconds)

**[Action: Point to the three prediction rows below the input. Type slowly to show predictions updating.]**

> "Below the input are three rows of next-character predictions. Each shows the top predicted next bytes with their probabilities.
>
> **BDH Raw** is the base model output — just the BDH forward pass, no σ correction.
>
> **σ-Learned** is where inference-time learning becomes visible. This row blends in the accumulated Hebbian memory using the formula: adjusted logits equals base logits plus alpha times sigma-logits, where alpha is 0.01 times the natural log of 1 plus the total token count. As σ builds up more context, the correction grows logarithmically. When σ is strong enough to change the top prediction, a '⇄ shifted' indicator appears — showing the prediction changed from one byte to another. No gradient updates needed — this is pure co-activation memory.
>
> **GPT** shows the standard transformer's prediction as a baseline comparison.
>
> The sigma indicator below shows how many total tokens have been learned into σ."

---

## Scene 9 — Demo Mode (15 seconds)

**[Action: Click the Demo button in the header]**

> "Click **Demo** in the header to watch a Shakespeare passage — 'To be, or not to be, that is the question' — type itself automatically. All five panels animate in sync. The sparkline rises during novel phrases and dips during predictable ones. Graph nodes light up, synapses strengthen, attention patterns fill in. It's a hands-free walkthrough of the entire architecture."

**[Let it run for a few seconds, then click Stop]**

---

## Scene 10 — Teach Mode (35 seconds)

**[Action: Click the Teach button in the header. Watch the progress bar and phase indicators.]**

> "Now the most powerful feature — **Teach mode**. Click the Teach button. The app clears σ memory, then feeds the phrase 'the cat sat on the mat' three times to build up Hebbian memory.
>
> Watch the progress bar and phase indicator:  
> — Repetition 1 of 3: building σ memory  
> — Repetition 2 of 3: σ strengthening  
> — Repetition 3 of 3: σ patterns consolidating  
>
> Then it types the test phrase: 'the cat sat on the ' — and stops.
>
> Now look at the σ-Learned prediction row. After three repetitions, the Hebbian memory has accumulated enough co-activation patterns that the σ-Learned row shifts its top prediction toward 'mat' — the model **learned the pattern** with zero gradient updates, purely through the outer-product memory rule σ plus-equals y outer-product x.
>
> The '⇄ shifted' indicator confirms it: the prediction changed. This is inference-time learning — the core BDH breakthrough — demonstrated live in your browser."

---

## Scene 11 — Closing (15 seconds)

**[Action: Show the About page briefly, then return to main view]**

> "The About page has a deep dive into all five BDH pillars, the full architecture pipeline, key concepts like RoPE and Hebbian learning, and the model configuration.
>
> Dragon Brain is fully open source, runs entirely client-side with zero server dependencies, and is deployed on Vercel. Built by Rajdeep Singh for the IIT Ropar times Pathway — Beyond Transformers Hackathon, Path A: Visualization and Inner Worlds.
>
> Try it yourself at dragonbrain.vercel.app."

---

## Recording Tips

1. **Browser:** Use Chrome or Edge for best ONNX WASM performance
2. **Resolution:** Record at 1920×1080 or higher
3. **Tab:** Close all other tabs to avoid WASM memory pressure
4. **Typing speed:** Slow and deliberate — let panels update between keystrokes
5. **Mouse:** Hover over panels before speaking about them so the viewer follows
6. **Demo mode:** Let it run for at least 5 seconds to show the sparkline pattern
7. **Teach mode:** Wait for the "Done!" message before explaining the prediction shift
8. **About page:** Just flash it briefly at the end — 2-3 seconds is enough
