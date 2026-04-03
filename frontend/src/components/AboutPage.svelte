<script>
  import { createEventDispatcher } from 'svelte';
  const dispatch = createEventDispatcher();

  const tocItems = [
    { id: 'glossary', label: 'Glossary' },
    { id: 'what', label: 'What is this?' },
    { id: 'why', label: 'Why BDH?' },
    { id: 'pillars', label: '5 Pillars' },
    { id: 'arch', label: 'Architecture' },
    { id: 'compare', label: 'BDH vs Transformers' },
    { id: 'panels', label: 'Visualizer Panels' },
    { id: 'config', label: 'Model Config' },
    { id: 'concepts', label: 'Key Concepts' },
    { id: 'usage', label: 'How to Use' },
    { id: 'refs', label: 'References' },
  ];
</script>

<div class="about">
  <!-- ── Back Button ── -->
  <button class="back-btn" on:click={() => dispatch('back')} aria-label="Back to visualizer">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
    Back to Visualizer
  </button>

  <!-- ── Hero Header ── -->
  <header class="hero">
    <div class="hero-badge">Deep Dive</div>
    <h1 class="hero-title">Understanding<br/><span class="hero-gradient">Dragon Brain</span></h1>
    <p class="hero-subtitle">
      A complete guide to the Baby Dragon Hatchling architecture,<br class="hide-mobile"/>
      its neuroscience roots, and how this interactive visualizer works.
    </p>
    <div class="hero-meta">
      <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener" class="hero-link">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        Read the Paper
      </a>
      <span class="hero-sep">/</span>
      <span class="hero-author">Kosowski et al. 2025</span>
    </div>
  </header>

  <!-- ── Table of Contents ── -->
  <nav class="toc" aria-label="Table of contents">
    <span class="toc-label">Contents</span>
    <div class="toc-items">
      {#each tocItems as item}
        <a href="#{item.id}" class="toc-link">{item.label}</a>
      {/each}
    </div>
  </nav>

  <!-- ── Section 0: Glossary ── -->
  <section class="section glossary-section" id="glossary">
    <h2><span class="section-num">00</span> Quick Reference — All Terms</h2>
    <div class="glossary-scroll">
      <table class="glossary-table">
        <thead>
          <tr><th>Term</th><th>Full Form / Meaning</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>BDH</strong></td><td>Baby Dragon Hatchling — the novel neural architecture</td></tr>
          <tr><td><strong>GPT</strong></td><td>Generative Pre-trained Transformer — the standard baseline</td></tr>
          <tr><td><strong>σ (sigma)</strong></td><td>Hebbian co-activation memory matrix — fixed-size, named after Greek letter sigma</td></tr>
          <tr><td><strong>Δσ (delta sigma)</strong></td><td>Change in σ per token — how many synapses strengthened and by how much</td></tr>
          <tr><td><strong>ReLU</strong></td><td>Rectified Linear Unit — <code>max(0, x)</code> — outputs zero for negatives → sparse activations</td></tr>
          <tr><td><strong>GELU</strong></td><td>Gaussian Error Linear Unit — smooth activation used in GPT → nearly all neurons fire</td></tr>
          <tr><td><strong>RoPE</strong></td><td>Rotary Position Embedding — encodes relative token position via rotation instead of adding position vectors</td></tr>
          <tr><td><strong>Gx</strong></td><td><code>Encoder × Decoder_x</code> — feedforward causal circuit, called "Thought Flow"</td></tr>
          <tr><td><strong>Gy</strong></td><td><code>Decoder_y<sup>T</sup> × Decoder_x</code> — Hebbian memory readout graph, called "Memory Echo"</td></tr>
          <tr><td><strong>KV-cache</strong></td><td>Key-Value cache — stores past keys and values in Transformers, grows by ~1 KB per token</td></tr>
          <tr><td><strong>Q, K, V</strong></td><td>Query, Key, Value — the three projections in attention. In BDH: Q = K = x (no separate projections)</td></tr>
          <tr><td><strong>ONNX</strong></td><td>Open Neural Network Exchange — model format for cross-platform inference</td></tr>
          <tr><td><strong>WASM</strong></td><td>WebAssembly — binary instruction format that lets native code run in browsers</td></tr>
          <tr><td><strong>T</strong></td><td>Token count / sequence length</td></tr>
          <tr><td><strong>N</strong></td><td>Neuron count per attention head (512 in this model, 1024 total across 2 heads)</td></tr>
          <tr><td><strong>D</strong></td><td>Embedding dimension (64 in this model)</td></tr>
          <tr><td><strong>D<sub>x</sub> (Decoder_x)</strong></td><td>Weight matrix that projects embedding v* into sparse neuron space x</td></tr>
          <tr><td><strong>D<sub>y</sub> (Decoder_y)</strong></td><td>Weight matrix that projects attention output into gated readout y</td></tr>
          <tr><td><strong>Encoder</strong></td><td>Weight matrix that projects y back to embedding dimension — shared across all layers</td></tr>
          <tr><td><strong>SAE</strong></td><td>Sparse AutoEncoder — post-hoc tool Transformers need for interpretability; BDH doesn't need this</td></tr>
          <tr><td><strong>Monosemantic</strong></td><td>One neuron = one concept. BDH gets this naturally via sparsity; Transformers are polysemantic</td></tr>
          <tr><td><strong>Scale-free network</strong></td><td>A graph where a few "hub" nodes have many connections and most have few — follows a power law</td></tr>
          <tr><td><strong>Outer product</strong></td><td>y ⊗ x — element (i,j) = y[i] × x[j]. This is how σ records which neurons co-activated</td></tr>
          <tr><td><strong>Causal mask</strong></td><td>Enforces that token at position t can only attend to tokens 0…t−1, never future tokens</td></tr>
          <tr><td><strong>Softmax</strong></td><td>Converts raw scores into probabilities summing to 1. Used in Transformer attention; BDH does not use it</td></tr>
          <tr><td><strong>O(T²)</strong></td><td>Quadratic complexity — grows with the square of sequence length (standard attention)</td></tr>
          <tr><td><strong>O(T)</strong></td><td>Linear complexity — grows proportionally to sequence length (BDH's recurrent form)</td></tr>
          <tr><td><strong>O(1)</strong></td><td>Constant — doesn't grow with input (σ memory size)</td></tr>
          <tr><td><strong>α (alpha)</strong></td><td>Blending coefficient for σ-Learned: α = 0.01 × log(1 + T) — grows logarithmically</td></tr>
          <tr><td><strong>D3.js</strong></td><td>Data-Driven Documents — JavaScript library for interactive visualizations</td></tr>
          <tr><td><strong>Hub neurons</strong></td><td>Neurons with many connections in the emergent graph — organizational centers</td></tr>
          <tr><td><strong>Excitatory edge</strong></td><td>Connection where one neuron amplifies another's signal (positive weight)</td></tr>
          <tr><td><strong>Inhibitory edge</strong></td><td>Connection where one neuron suppresses another's signal (negative weight)</td></tr>
          <tr><td><strong>Synapse</strong></td><td>Connection between two neurons. σ(i,j) records how strongly neurons i and j co-activate</td></tr>
          <tr><td><strong>Viridis</strong></td><td>Perceptually-uniform color scale (dark → green → yellow) used for the Hebbian heatmap</td></tr>
          <tr><td><strong>Byte token</strong></td><td>Character encoded as raw byte value (0–255). 'A' = 65, 'a' = 97, space = 32</td></tr>
          <tr><td><strong>Inference</strong></td><td>Running a trained model on new input to get predictions (vs training)</td></tr>
          <tr><td><strong>Inference-time learning</strong></td><td>BDH's ability to learn (update σ) during inference, without retraining or gradients</td></tr>
          <tr><td><strong>IIT Ropar</strong></td><td>Indian Institute of Technology Ropar</td></tr>
          <tr><td><strong>Pathway</strong></td><td>The company behind the BDH research (pathway.com)</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- ── Section 1: What is this project? ── -->
  <section class="section" id="what">
    <h2><span class="section-num">01</span> What is This Project?</h2>
    <p>
      <strong>Dragon Brain</strong> is an interactive visualizer that lets you explore a working neural network
      built on the <strong>Baby Dragon Hatchling (BDH)</strong> architecture — a post-transformer design proposed
      by Kosowski et al. in their 2025 paper <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener">"The Dragon Hatchling"</a>.
    </p>
    <p>
      Instead of just reading about BDH, you can <em>type text</em> and watch the architecture respond in real time:
      see which neurons fire, how attention flows between tokens, how the Hebbian memory matrix builds up,
      and how the emergent graph structure organizes computation.
    </p>
    <div class="callout">
      <div class="callout-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
      </div>
      <p>The model runs <strong>entirely in your browser</strong> using ONNX Runtime (WebAssembly). No server calls,
      no API keys — the neural network weights are loaded locally and inference happens on your machine.</p>
    </div>
  </section>

  <!-- ── Section 2: Why BDH matters ── -->
  <section class="section" id="why">
    <h2><span class="section-num">02</span> Why Does BDH Matter?</h2>
    <p>
      Modern transformers (GPT, LLaMA, etc.) are incredibly powerful but have fundamental limitations:
    </p>
    <div class="problem-grid">
      <div class="problem-card">
        <div class="problem-icon problem-red">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
        </div>
        <h4>Dense Computation</h4>
        <p>Nearly every neuron fires on every token, wasting energy on irrelevant features</p>
      </div>
      <div class="problem-card">
        <div class="problem-icon problem-red">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        </div>
        <h4>Growing Memory</h4>
        <p>KV-cache grows linearly with context length, making long sequences expensive</p>
      </div>
      <div class="problem-card">
        <div class="problem-icon problem-red">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <h4>Opaque Internals</h4>
        <p>Understanding what neurons represent requires expensive post-hoc probing (SAEs)</p>
      </div>
      <div class="problem-card">
        <div class="problem-icon problem-red">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
        </div>
        <h4>Quadratic Attention</h4>
        <p>Softmax attention is O(T&sup2;d) per layer, limiting sequence length</p>
      </div>
    </div>
    <p>
      BDH addresses <strong>all four</strong> by drawing inspiration from neuroscience — specifically, how biological brains
      use sparse coding, Hebbian learning, and modular organization.
    </p>
  </section>

  <!-- ── Section 2.5: The 5 Pillars ── -->
  <section class="section" id="pillars">
    <h2><span class="section-num">03</span> The 5 Pillars of BDH</h2>
    <div class="pillar-grid">
      <div class="pillar-card pillar-blue">
        <div class="pillar-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        </div>
        <h3>Sparse Activation</h3>
        <p>Only a small fraction of neurons fire per token (~5% in the paper's large models, vs ~99% in transformers). Each neuron represents <strong>one concept</strong> — natively monosemantic.</p>
      </div>
      <div class="pillar-card pillar-green">
        <div class="pillar-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="19" r="2"/><line x1="7" y1="7" x2="11" y2="17"/><line x1="17" y1="7" x2="13" y2="17"/><line x1="7" y1="6" x2="17" y2="6"/></svg>
        </div>
        <h3>Emergent Graph</h3>
        <p>Hub neurons and modular communities emerge <strong>from random initialization</strong> during training — no one programs the structure.</p>
      </div>
      <div class="pillar-card pillar-gold">
        <div class="pillar-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
        </div>
        <h3>Hebbian Memory (σ)</h3>
        <p>"Neurons that fire together wire together." The σ matrix is a <strong>fixed-size</strong> memory that doesn't grow with context length.</p>
      </div>
      <div class="pillar-card pillar-rose">
        <div class="pillar-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 22 8.5 12 15 2 8.5"/><polyline points="2 15.5 12 22 22 15.5"/></svg>
        </div>
        <h3>Linear Attention</h3>
        <p>No softmax. Uses RoPE rotation with a causal mask. Q=K=x (same vector!), making attention a <strong>self-correlation</strong> readout.</p>
      </div>
      <div class="pillar-card" style="border-color: rgba(155, 126, 240, 0.25); --pillar-accent: #9b7ef0;">
        <div class="pillar-icon" style="color: #9b7ef0;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="8" height="10" rx="1"/><rect x="14" y="7" width="8" height="10" rx="1"/><line x1="10" y1="12" x2="14" y2="12"/></svg>
        </div>
        <h3>Composable Merging</h3>
        <p>BDH's scale-free architecture enables <strong>concatenating independently trained models</strong>. A French translator + Spanish translator, merged into one multilingual model — no retraining. This composability is architecturally native (Section 7.1 of the paper).</p>
      </div>
    </div>
  </section>

  <!-- ── Section 3: Architecture deep dive ── -->
  <section class="section" id="arch">
    <h2><span class="section-num">04</span> The Architecture — Step by Step</h2>
    <p>
      Each layer of BDH processes token embeddings through a pipeline that is deliberately different from a transformer.
      Here is the full computation for one layer:
    </p>

    <div class="step-grid">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-content">
          <h3>Sparse Activation (x)</h3>
          <div class="formula">
            <span class="formula-label">Decode</span>
            <code>x = ReLU( v* @ Decoder<sub>x</sub> )</code>
          </div>
          <p>
            The token embedding <code>v*</code> (dimension D=64) is projected into a much larger space (N=512 neurons per head)
            through the <code>Decoder<sub>x</sub></code> matrix. ReLU zeros out all negative values, leaving only a sparse subset of neurons
            with positive activations (~5% in the paper's large models). Each active neuron corresponds to a specific feature the model has learned.
          </p>
        </div>
      </div>

      <div class="step">
        <div class="step-num">2</div>
        <div class="step-content">
          <h3>Linear Attention</h3>
          <div class="formula">
            <span class="formula-label">Attend</span>
            <code>a* = Attention( Q=x, K=x, V=v* )</code>
          </div>
          <p>
            BDH uses the <em>same</em> sparse activation as both Query and Key (Q=K=x). No separate projections needed.
            Scores are <code>RoPE(x) @ RoPE(x)<sup>T</sup></code> with a causal mask. <strong>No softmax</strong> — raw dot products
            after RoPE rotation. Attention becomes a self-correlation: how similar are sparse patterns across tokens?
          </p>
          <p>
            The parallel form is O(T&sup2;), but the paper describes a recurrent form using
            cumulative sums that achieves <strong>O(T) per step</strong> during inference.
          </p>
        </div>
      </div>

      <div class="step">
        <div class="step-num">3</div>
        <div class="step-content">
          <h3>Gated Readout (y)</h3>
          <div class="formula">
            <span class="formula-label">Gate</span>
            <code>y = ReLU( LN(a*) @ Decoder<sub>y</sub> ) &odot; x</code>
          </div>
          <p>
            The attended value is projected through a second decoder, then element-wise multiplied (&odot;) by x.
            This gating ensures y is nonzero only where x was already active —
            forming <strong>(x, y)</strong> co-activation pairs that drive Hebbian learning.
          </p>
        </div>
      </div>

      <div class="step">
        <div class="step-num">4</div>
        <div class="step-content">
          <h3>Hebbian Memory Update (σ)</h3>
          <div class="formula formula-gold">
            <span class="formula-label">Hebb</span>
            <code>σ &nbsp;+=&nbsp; y &otimes; x</code>
          </div>
          <p>
            After each token, the outer product (&otimes;) of y and x is accumulated into a fixed-size matrix
            σ &isin; ℝ<sup>N&times;N</sup>.
            This follows Hebb's rule: "neurons that fire together wire together." Unlike a transformer's KV-cache
            that grows with every new token, σ has a <strong>fixed size</strong> regardless of context length.
          </p>
        </div>
      </div>

      <div class="step">
        <div class="step-num">5</div>
        <div class="step-content">
          <h3>Residual Update</h3>
          <div class="formula formula-green">
            <span class="formula-label">Encode</span>
            <code>v* = LN( v* + LN( y @ Encoder ) )</code>
          </div>
          <p>
            The y activations are mapped back to the D-dimensional embedding space
            using the <code>Encoder</code> matrix and added as a residual. Layer normalization (without learnable
            affine parameters) is applied before and after. The updated v* feeds into the next layer.
          </p>
        </div>
      </div>
    </div>
  </section>

  <!-- ── Section 4: Key Differences ── -->
  <section class="section" id="compare">
    <h2><span class="section-num">05</span> BDH vs Transformers</h2>

    <div class="comparison-table">
      <table>
        <thead>
          <tr>
            <th>Property</th>
            <th>Transformer</th>
            <th class="bdh-col">BDH</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="prop-name">Activation</td>
            <td>GELU/SwiGLU — nearly all neurons fire (~99%)</td>
            <td class="bdh-val">ReLU — sparse activation (~5% at scale)</td>
          </tr>
          <tr>
            <td class="prop-name">Memory</td>
            <td>KV-cache: grows O(T) with sequence length</td>
            <td class="bdh-val">σ matrix: fixed O(N&sup2;), Hebbian outer product</td>
          </tr>
          <tr>
            <td class="prop-name">Structure</td>
            <td>Dense, homogeneous — all weights equally important</td>
            <td class="bdh-val">Emergent scale-free graph with hub neurons</td>
          </tr>
          <tr>
            <td class="prop-name">Interpretability</td>
            <td>Requires Sparse AutoEncoders (post-hoc, expensive)</td>
            <td class="bdh-val">Natively monosemantic — each neuron = one concept</td>
          </tr>
          <tr>
            <td class="prop-name">Attention</td>
            <td>Softmax over Q@K<sup>T</sup>/&radic;d, O(T&sup2;d)</td>
            <td class="bdh-val">No softmax, Q=K=x, RoPE, O(T) recurrent</td>
          </tr>
          <tr>
            <td class="prop-name">Params</td>
            <td>Each layer has its own parameters</td>
            <td class="bdh-val">Decoder<sub>x</sub>, Decoder<sub>y</sub>, Encoder shared across all layers</td>
          </tr>
          <tr>
            <td class="prop-name">Tokens</td>
            <td>Sub-word (BPE, 32k-128k vocab)</td>
            <td class="bdh-val">Byte-level (256 vocab), no tokenizer needed</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- ── Section 5: Visualizer Panels ── -->
  <section class="section" id="panels">
    <h2><span class="section-num">06</span> Understanding the Visualizer Panels</h2>

    <div class="panel-guide">
      <div class="guide-card guide-blue">
        <div class="guide-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          <h3>Sparse Activation Panel</h3>
        </div>
        <p>
          Shows a 32&times;32 grid of neurons (1,024 total = 2 heads &times; 512 neurons/head). Each pixel is one neuron.
          Lit pixels = active neurons after ReLU. The left grid shows real BDH activations, the right shows
          a dense transformer reference for comparison.
        </p>
        <div class="guide-tip">
          <strong>Tip:</strong> Predictable text (repeated characters) fires ~2-3% of neurons. Novel input pushes it to ~6-7%.
          This is BDH's built-in uncertainty indicator.
        </div>
      </div>

      <div class="guide-card guide-green">
        <div class="guide-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="19" r="2"/><line x1="7" y1="7" x2="11" y2="17"/><line x1="17" y1="7" x2="13" y2="17"/><line x1="7" y1="6" x2="17" y2="6"/></svg>
          <h3>Emergent Graph Panel</h3>
        </div>
        <p>Visualizes the internal wiring structure using D3 force-directed layout. Two modes:</p>
        <div class="guide-modes">
          <div class="guide-mode">
            <div class="formula formula-sm">
              <span class="formula-label">Gx</span>
              <code>Encoder @ Decoder<sub>x</sub></code>
            </div>
            <span class="guide-mode-desc"><strong>Thought Flow</strong> — feedforward causal circuit</span>
          </div>
          <div class="guide-mode">
            <div class="formula formula-sm formula-green">
              <span class="formula-label">Gy</span>
              <code>Decoder<sub>y</sub><sup>T</sup> @ Decoder<sub>x</sub></code>
            </div>
            <span class="guide-mode-desc"><strong>Memory Echo</strong> — Hebbian memory readout graph</span>
          </div>
        </div>
        <div class="guide-tip">
          <strong>Tip:</strong> Hub neurons (large circles) emerge spontaneously from random init.
          Yellow nodes show currently active neurons for your input.
        </div>
      </div>

      <div class="guide-card guide-gold">
        <div class="guide-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
          <h3>Hebbian Memory (σ) Panel</h3>
        </div>
        <p>
          Displays a 64&times;64 heatmap of the most active rows/columns of the σ matrix. Viridis color scale
          maps connection strength from 0 (dark) to strong (yellow).
        </p>
        <div class="formula formula-gold formula-sm">
          <span class="formula-label">Hebb</span>
          <code>σ += y &otimes; x</code>
        </div>
        <div class="guide-tip">
          <strong>Tip:</strong> Press "Clear" to reset, then retype to watch σ rebuild. The matrix is fixed-size
          regardless of how much text you type — O(1) memory for unlimited context.
        </div>
      </div>

      <div class="guide-card guide-rose">
        <div class="guide-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 22 8.5 12 15 2 8.5"/><polyline points="2 15.5 12 22 22 15.5"/></svg>
          <h3>Attention Pattern Panel</h3>
        </div>
        <p>
          Shows a T&times;T heatmap of raw attention scores. The causal mask enforces strictly lower-triangular
          structure — each token can only attend to <em>previous</em> tokens.
        </p>
        <div class="guide-tip">
          <strong>Tip:</strong> Since Q=K=x, attention scores are inner products of sparse vectors. No softmax —
          what you see are raw dot products with RoPE position encoding. Brighter cells near the diagonal = recent focus.
        </div>
      </div>

      <div class="guide-card guide-violet">
        <div class="guide-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><line x1="6" y1="10" x2="6" y2="14"/><line x1="10" y1="10" x2="10" y2="14"/></svg>
          <h3>Memory Scaling Panel</h3>
        </div>
        <p>
          An SVG chart comparing BDH's constant σ memory (flat blue line at 4 MB) against a GPT KV-cache
          (pink line growing at ~1 KB per token). The two lines cross at T≈4,096 — after which a transformer
          uses more memory than BDH for the same context length.
        </p>
        <div class="guide-tip">
          <strong>Tip:</strong> At the paper's full scale (N=32,768), the crossover happens at just T≈400 tokens.
          The dots on each line track the current token count as you type.
        </div>
      </div>

      <div class="guide-card guide-gold">
        <div class="guide-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          <h3>Prediction Bar</h3>
        </div>
        <p>
          Three rows of next-character predictions appear below the input. <strong>BDH Raw</strong> shows the
          model's base output. <strong>σ-Learned</strong> adds a correction from accumulated Hebbian memory —
          when σ is strong enough to change the top prediction, a <em>⇄ shifted</em> indicator appears.
          <strong>GPT</strong> shows a standard transformer's prediction as a baseline.
        </p>
        <div class="guide-tip">
          <strong>Tip:</strong> Use Teach mode (the ✏ button) to build up σ quickly. After 3 repetitions of
          "the cat sat on the mat", the σ-Learned row will visibly shift predictions toward the learned pattern.
        </div>
      </div>
    </div>
  </section>

  <!-- ── Section 6: Model Configuration ── -->
  <section class="section" id="config">
    <h2><span class="section-num">07</span> Model Configuration</h2>
    <p>
      This visualizer uses a <strong>tiny</strong> BDH model — small enough for your browser
      while preserving all three key phenomena: sparsity, graph emergence, and Hebbian memory.
    </p>

    <div class="config-grid">
      <div class="config-item"><span class="config-label">Layers (L)</span><span class="config-value">2</span></div>
      <div class="config-item"><span class="config-label">Embedding (D)</span><span class="config-value">64</span></div>
      <div class="config-item"><span class="config-label">Heads (n<sub>h</sub>)</span><span class="config-value">2</span></div>
      <div class="config-item"><span class="config-label">Neurons/Head (N)</span><span class="config-value">512</span></div>
      <div class="config-item"><span class="config-label">Total Neurons</span><span class="config-value">1,024</span></div>
      <div class="config-item"><span class="config-label">Vocab Size</span><span class="config-value">256</span></div>
      <div class="config-item"><span class="config-label">Max Sequence</span><span class="config-value">128</span></div>
      <div class="config-item"><span class="config-label">Runtime</span><span class="config-value">ONNX+WASM</span></div>
    </div>

    <div class="callout">
      <div class="callout-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      </div>
      <p>
        The weight matrices <code>Decoder<sub>x</sub></code>, <code>Decoder<sub>y</sub></code>, and <code>Encoder</code> are
        <strong>shared across all layers</strong> — the same projection is reused in every layer, a distinctive BDH design choice.
        Note: the paper reports ~5% neuron sparsity for models with 32k+ neurons. This tiny model (1,024 neurons)
        may show higher activation rates — sparsity increases with model scale.
      </p>
    </div>
  </section>

  <!-- ── Section 7: Key Concepts ── -->
  <section class="section" id="concepts">
    <h2><span class="section-num">08</span> Key Concepts Explained</h2>

    <div class="concept-list">
      <div class="concept concept-blue">
        <h3>Monosemantic Neurons</h3>
        <p>
          In transformers, individual neurons often respond to multiple unrelated concepts (polysemantic).
          BDH's sparse ReLU naturally forces each neuron to specialize in <strong>one concept</strong>
          (monosemantic). When neuron #247 fires, it means one specific thing — you don't need a Sparse
          AutoEncoder to decode it.
        </p>
      </div>

      <div class="concept concept-rose">
        <h3>RoPE (Rotary Position Embedding)</h3>
        <p>
          Instead of adding positional embeddings, RoPE applies a rotation encoding relative position.
        </p>
        <div class="formula formula-sm">
          <span class="formula-label">RoPE</span>
          <code>θ<sub>i</sub> = 10000<sup>−2i/d</sup></code>
        </div>
        <p>
          When computing Q@K<sup>T</sup>, the rotations compose so the score depends only on <em>relative distance</em>
          between tokens, not absolute positions. This works even when Q=K=x in BDH.
        </p>
      </div>

      <div class="concept concept-green">
        <h3>Scale-Free Networks</h3>
        <p>
          The emergent graph topology follows a power law degree distribution — a few hub neurons have
          many connections while most neurons have few. This mirrors biological neural networks, the internet,
          and social networks. This structure was <strong>not designed</strong> — it emerged from random initialization.
        </p>
      </div>

      <div class="concept concept-gold">
        <h3>Hebbian Learning</h3>
        <p>
          Donald Hebb's 1949 principle: "When an axon of cell A repeatedly takes part in firing cell B,
          A's efficiency in firing B is increased."
        </p>
        <div class="formula formula-gold formula-sm">
          <span class="formula-label">Hebb</span>
          <code>σ += y &otimes; x</code>
        </div>
        <p>
          When neurons x and y co-activate, their connection in σ strengthens.
          This accumulates a long-term memory that <strong>doesn't grow</strong> with context length.
        </p>
      </div>

      <div class="concept concept-violet">
        <h3>Inference-Time Learning</h3>
        <p>
          Unlike standard transformers whose weights are frozen at inference, BDH keeps learning through σ.
          This visualizer demonstrates it with the σ-Learned prediction row:
        </p>
        <div class="formula formula-sm">
          <span class="formula-label">α</span>
          <code>α = 0.01 &times; log(1 + T)</code>
        </div>
        <div class="formula formula-sm">
          <span class="formula-label">blend</span>
          <code>logits′ = logits + α &times; σ·logits</code>
        </div>
        <p>
          As σ accumulates more context, the correction grows logarithmically. No gradient updates needed —
          the model adapts purely through co-activation memory. Try Teach mode to see this in action.
        </p>
      </div>
    </div>
  </section>

  <!-- ── Section 8: How to Use ── -->
  <section class="section" id="usage">
    <h2><span class="section-num">09</span> How to Use This Visualizer</h2>
    <ol class="usage-steps">
      <li>
        <strong>Type text</strong> in the input box at the top. The model processes each character as a byte token (0-255).
        Try "The dollar rose against the euro" or "Hello world hello world" to see sparsity shift.
      </li>
      <li>
        <strong>Watch the panels</strong> update in real time — sparse activation grid, Hebbian heatmap,
        attention pattern, and active graph nodes all respond instantly.
      </li>
      <li>
        <strong>Read the predictions</strong> — three rows appear below the input: <em>BDH Raw</em> (base output),
        <em>σ-Learned</em> (Hebbian-corrected), and <em>GPT</em> (transformer baseline). When σ shifts the
        top prediction, a <em>⇄ shifted</em> indicator appears.
      </li>
      <li>
        <strong>Switch layers and heads</strong> using L1/L2 and H1/H2 buttons. Different heads often specialize
        in different patterns.
      </li>
      <li>
        <strong>Explore the graph</strong> — switch between Gx (Thought Flow) and Gy (Memory Echo). Drag and zoom
        to inspect hub neurons.
      </li>
      <li>
        <strong>Clear memory</strong> with the Clear button on the Hebbian panel, then retype to watch σ rebuild
        from scratch.
      </li>
      <li>
        <strong>Try Demo mode</strong> — click the ▶ button in the header to watch a Shakespeare passage type
        itself automatically while all panels animate.
      </li>
      <li>
        <strong>Try Teach mode</strong> — click the ✏ button to feed a repeated phrase (e.g. "the cat sat on the mat")
        three times. After a few repetitions, the σ-Learned predictions begin to shift toward the taught pattern.
      </li>
    </ol>
  </section>

  <!-- ── Section 9: References ── -->
  <section class="section" id="refs">
    <h2><span class="section-num">10</span> References &amp; Credits</h2>
    <div class="ref-grid">
      <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener" class="ref-card">
        <div class="ref-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </div>
        <div>
          <h4>Research Paper</h4>
          <p>Kosowski et al. 2025 — "The Dragon Hatchling"</p>
        </div>
      </a>
      <a href="https://github.com/pathwaycom/bdh" target="_blank" rel="noopener" class="ref-card">
        <div class="ref-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
        </div>
        <div>
          <h4>Official Code</h4>
          <p>github.com/pathwaycom/bdh</p>
        </div>
      </a>
      <a href="https://pathway.com" target="_blank" rel="noopener" class="ref-card">
        <div class="ref-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
        </div>
        <div>
          <h4>Pathway</h4>
          <p>pathway.com</p>
        </div>
      </a>
      <a href="https://rajdeep-singh.vercel.app/" target="_blank" rel="noopener" class="ref-card">
        <div class="ref-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        </div>
        <div>
          <h4>Built by Rajdeep Singh</h4>
          <p>For the Beyond Transformers Hackathon (Pathway A)</p>
        </div>
      </a>
    </div>
  </section>
</div>

<style>
  .about {
    max-width: 780px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    animation: fadeIn 0.35s ease;
  }

  /* ── Back Button ── */
  .back-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.3rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-full);
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-base);
    margin-bottom: 2.5rem;
  }

  .back-btn:hover {
    background: var(--bg-elevated);
    border-color: var(--border-hover);
    color: var(--text-primary);
    transform: translateX(-2px);
  }

  /* ── Hero ── */
  .hero {
    text-align: center;
    padding: 3rem 0 3rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .hero-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--accent);
    background: var(--accent-glow);
    border: 1px solid rgba(91, 141, 239, 0.2);
    border-radius: var(--radius-full);
    padding: 0.35rem 1.1rem;
    margin-bottom: 1.5rem;
  }

  .hero-title {
    font-family: var(--font-sans);
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 1.2rem;
  }

  .hero-gradient {
    background: linear-gradient(135deg, #7da8f5 0%, #9b7ef0 40%, #f06292 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .hero-subtitle {
    font-size: 1.12rem;
    color: var(--text-secondary);
    line-height: 1.8;
    max-width: 560px;
    margin: 0 auto 1.5rem;
  }

  .hero-meta {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    font-size: 0.88rem;
  }

  .hero-link {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: var(--accent-bright);
    font-weight: 500;
    text-decoration: none;
    transition: color var(--transition-fast);
  }

  .hero-link:hover { color: #fff; }

  .hero-sep { color: var(--text-dim); }

  .hero-author {
    color: var(--text-dim);
    font-style: italic;
  }

  .hide-mobile { display: inline; }

  /* ── Table of Contents ── */
  .toc {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 1rem 1.2rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    margin-bottom: 3rem;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
  }

  .toc::-webkit-scrollbar { height: 5px; }
  .toc::-webkit-scrollbar-track { background: transparent; margin: 0 0.8rem; }
  .toc::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: var(--radius-full);
  }
  .toc::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  .toc-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .toc-items {
    display: flex;
    gap: 0.25rem;
    flex-wrap: nowrap;
  }

  .toc-link {
    font-size: 0.84rem;
    color: var(--text-muted);
    text-decoration: none;
    padding: 0.35rem 0.7rem;
    border-radius: var(--radius-sm);
    white-space: nowrap;
    transition: all var(--transition-fast);
  }

  .toc-link:hover {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.06);
  }

  /* ── Sections ── */
  .section {
    margin-bottom: 4rem;
    scroll-margin-top: 2rem;
  }

  /* ── Glossary Table ── */
  .glossary-scroll {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  .glossary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    line-height: 1.5;
  }

  .glossary-table thead tr {
    border-bottom: 2px solid var(--accent);
  }

  .glossary-table th {
    text-align: left;
    padding: 0.55rem 0.75rem;
    font-weight: 700;
    color: var(--accent);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    white-space: nowrap;
  }

  .glossary-table th:first-child {
    width: 22%;
    min-width: 140px;
  }

  .glossary-table td {
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    vertical-align: top;
  }

  .glossary-table td:first-child {
    color: var(--text-primary);
    white-space: nowrap;
  }

  .glossary-table tbody tr:hover {
    background: rgba(91, 141, 239, 0.04);
  }

  .glossary-table code {
    font-family: var(--font-mono);
    font-size: 0.82em;
    background: rgba(91, 141, 239, 0.1);
    padding: 0.1em 0.35em;
    border-radius: 3px;
    color: var(--accent);
  }

  .section h2 {
    font-family: var(--font-sans);
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 1.4rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .section-num {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent);
    background: var(--accent-glow);
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
    letter-spacing: 0.02em;
  }

  .section p {
    font-size: 1.02rem;
    color: var(--text-secondary);
    line-height: 1.9;
    margin-bottom: 1rem;
  }

  .section ol {
    margin: 1rem 0 1.2rem 1.5rem;
    color: var(--text-secondary);
    font-size: 1.02rem;
    line-height: 1.9;
  }

  .section li { margin-bottom: 0.5rem; }

  .section a {
    color: var(--accent-bright);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: rgba(125, 168, 245, 0.3);
    transition: text-decoration-color var(--transition-fast), color var(--transition-fast);
  }

  .section a:hover {
    color: #fff;
    text-decoration-color: var(--accent-bright);
  }

  .section code {
    font-family: var(--font-mono);
    font-size: 0.88em;
    padding: 0.18rem 0.5rem;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 5px;
    color: var(--accent-bright);
  }

  .section strong {
    color: var(--text-primary);
    font-weight: 600;
  }

  .section em {
    color: var(--text-muted);
    font-style: italic;
  }

  /* ── Callout ── */
  .callout {
    display: flex;
    gap: 1rem;
    padding: 1.2rem 1.4rem;
    background: var(--accent-glow);
    border: 1px solid rgba(91, 141, 239, 0.18);
    border-radius: var(--radius-md);
    margin: 1.2rem 0;
  }

  .callout-icon {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(91, 141, 239, 0.12);
    border-radius: var(--radius-sm);
    color: var(--accent-bright);
  }

  .callout p {
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.75;
  }

  /* ── Problem Grid ── */
  .problem-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
    margin: 1.2rem 0;
  }

  .problem-card {
    padding: 1rem;
    background: rgba(240, 98, 146, 0.04);
    border: 1px solid rgba(240, 98, 146, 0.12);
    border-radius: var(--radius-md);
    transition: border-color var(--transition-base);
  }

  .problem-card:hover {
    border-color: rgba(240, 98, 146, 0.3);
  }

  .problem-icon {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    margin-bottom: 0.5rem;
  }

  .problem-red {
    background: rgba(240, 98, 146, 0.1);
    color: var(--rose);
  }

  .problem-card h4 {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
  }

  .problem-card p {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.6;
  }

  /* ── Pillar Grid ── */
  .pillar-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1rem;
  }

  .pillar-card {
    padding: 1.3rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    transition: border-color var(--transition-base), transform var(--transition-base), box-shadow var(--transition-base);
  }

  .pillar-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
  }

  .pillar-blue { border-top: 3px solid var(--accent); }
  .pillar-blue:hover { border-color: var(--accent); }
  .pillar-green { border-top: 3px solid var(--green); }
  .pillar-green:hover { border-color: var(--green); }
  .pillar-gold { border-top: 3px solid var(--gold); }
  .pillar-gold:hover { border-color: var(--gold); }
  .pillar-rose { border-top: 3px solid var(--rose); }
  .pillar-rose:hover { border-color: var(--rose); }

  .pillar-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-md);
    margin-bottom: 0.7rem;
  }

  .pillar-blue .pillar-icon { background: var(--accent-glow); color: var(--accent); }
  .pillar-green .pillar-icon { background: var(--green-glow); color: var(--green); }
  .pillar-gold .pillar-icon { background: var(--gold-glow); color: var(--gold); }
  .pillar-rose .pillar-icon { background: var(--rose-glow); color: var(--rose); }

  .pillar-card h3 {
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.45rem;
  }

  .pillar-card p {
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin: 0;
  }

  /* ── Formulas ── */
  .formula {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.65rem 1rem;
    background: linear-gradient(135deg, rgba(91, 141, 239, 0.08) 0%, rgba(91, 141, 239, 0.03) 100%);
    border: 1px solid rgba(91, 141, 239, 0.2);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
    margin: 0.6rem 0 0.8rem;
    width: fit-content;
    max-width: 100%;
  }

  .formula code {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 500;
    color: #7da8f5;
    background: none;
    border: none;
    padding: 0;
    letter-spacing: 0.02em;
  }

  .formula-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    background: rgba(91, 141, 239, 0.12);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    flex-shrink: 0;
  }

  .formula-gold {
    background: linear-gradient(135deg, rgba(240, 194, 70, 0.08) 0%, rgba(240, 194, 70, 0.02) 100%);
    border-color: rgba(240, 194, 70, 0.2);
    border-left-color: var(--gold);
  }

  .formula-gold code { color: #f5dfa0; }
  .formula-gold .formula-label {
    color: var(--gold);
    background: rgba(240, 194, 70, 0.12);
  }

  .formula-green {
    background: linear-gradient(135deg, rgba(61, 214, 140, 0.08) 0%, rgba(61, 214, 140, 0.02) 100%);
    border-color: rgba(61, 214, 140, 0.2);
    border-left-color: var(--green);
  }

  .formula-green code { color: #7ee8b4; }
  .formula-green .formula-label {
    color: var(--green);
    background: rgba(61, 214, 140, 0.12);
  }

  .formula-sm {
    padding: 0.45rem 0.8rem;
  }

  .formula-sm code {
    font-size: 0.85rem;
  }

  /* ── Steps ── */
  .step-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1.2rem;
  }

  .step {
    display: flex;
    gap: 1.2rem;
    padding: 1.6rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  .step:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
  }

  .step-num {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--accent) 0%, #9b7ef0 100%);
    color: #fff;
    font-family: var(--font-mono);
    font-size: 0.88rem;
    font-weight: 700;
    border-radius: 50%;
    box-shadow: 0 2px 12px rgba(91, 141, 239, 0.25);
  }

  .step-content h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
  }

  .step-content p {
    font-size: 0.95rem;
    line-height: 1.8;
    margin-bottom: 0.6rem;
  }

  /* ── Comparison Table ── */
  .comparison-table {
    overflow-x: auto;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-default);
    margin-top: 1rem;
    background: var(--bg-card);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }

  th, td {
    padding: 0.85rem 1.1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-subtle);
  }

  th {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    background: rgba(255, 255, 255, 0.02);
  }

  td {
    color: var(--text-secondary);
    line-height: 1.6;
  }

  .prop-name {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.9rem;
  }

  .bdh-val {
    color: var(--green);
    font-weight: 500;
  }

  tr:last-child td { border-bottom: none; }

  tr:hover td { background: rgba(255, 255, 255, 0.02); }

  /* ── Panel Guide ── */
  .panel-guide {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
  }

  .guide-card {
    padding: 1.5rem 1.8rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    border-left: 3px solid var(--border-default);
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  .guide-card:hover {
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
  }

  .guide-blue { border-left-color: var(--accent); }
  .guide-blue:hover { border-color: rgba(91, 141, 239, 0.3); border-left-color: var(--accent); }
  .guide-green { border-left-color: var(--green); }
  .guide-green:hover { border-color: rgba(61, 214, 140, 0.3); border-left-color: var(--green); }
  .guide-gold { border-left-color: var(--gold); }
  .guide-gold:hover { border-color: rgba(240, 194, 70, 0.3); border-left-color: var(--gold); }
  .guide-rose { border-left-color: var(--rose); }
  .guide-rose:hover { border-color: rgba(240, 98, 146, 0.3); border-left-color: var(--rose); }
  .guide-violet { border-left-color: var(--violet, #9b7ef0); }
  .guide-violet:hover { border-color: rgba(155, 126, 240, 0.3); border-left-color: var(--violet, #9b7ef0); }

  .guide-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
  }

  .guide-blue .guide-header { color: var(--accent); }
  .guide-green .guide-header { color: var(--green); }
  .guide-gold .guide-header { color: var(--gold); }
  .guide-rose .guide-header { color: var(--rose); }
  .guide-violet .guide-header { color: var(--violet, #9b7ef0); }

  .guide-card h3 {
    font-size: 1.08rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .guide-card p {
    font-size: 0.95rem;
    line-height: 1.75;
    margin-bottom: 0.7rem;
  }

  .guide-tip {
    font-size: 0.88rem;
    color: var(--text-muted);
    padding: 0.7rem 0.9rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: var(--radius-sm);
    margin-top: 0.7rem;
    line-height: 1.65;
  }

  .guide-tip strong { color: var(--text-secondary); }

  .guide-modes {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 0.5rem 0;
  }

  .guide-mode {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
  }

  .guide-mode-desc {
    font-size: 0.92rem;
    color: var(--text-secondary);
  }

  /* ── Config Grid ── */
  .config-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin: 1.2rem 0;
  }

  .config-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
    padding: 0.8rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    text-align: center;
    transition: border-color var(--transition-base);
  }

  .config-item:hover { border-color: var(--border-hover); }

  .config-label {
    font-size: 0.8rem;
    color: var(--text-dim);
    font-weight: 500;
  }

  .config-value {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent-bright);
  }

  /* ── Concept List ── */
  .concept-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
  }

  .concept {
    padding: 1.4rem 1.6rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    border-left: 3px solid var(--accent);
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  .concept:hover {
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
  }

  .concept-blue { border-left-color: var(--accent); }
  .concept-blue:hover { border-color: rgba(91, 141, 239, 0.3); border-left-color: var(--accent); }
  .concept-rose { border-left-color: var(--rose); }
  .concept-rose:hover { border-color: rgba(240, 98, 146, 0.3); border-left-color: var(--rose); }
  .concept-green { border-left-color: var(--green); }
  .concept-green:hover { border-color: rgba(61, 214, 140, 0.3); border-left-color: var(--green); }
  .concept-gold { border-left-color: var(--gold); }
  .concept-gold:hover { border-color: rgba(240, 194, 70, 0.3); border-left-color: var(--gold); }
  .concept-violet { border-left-color: var(--violet, #9b7ef0); }
  .concept-violet:hover { border-color: rgba(155, 126, 240, 0.3); border-left-color: var(--violet, #9b7ef0); }

  .concept h3 {
    font-size: 1.08rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.55rem;
  }

  .concept p {
    font-size: 0.95rem;
    line-height: 1.8;
    margin-bottom: 0.6rem;
  }

  .concept p:last-child { margin-bottom: 0; }

  /* ── Usage Steps ── */
  .usage-steps {
    counter-reset: step;
    list-style: none;
    margin-left: 0 !important;
    padding: 0;
  }

  .usage-steps li {
    counter-increment: step;
    padding: 1.1rem 1.3rem 1.1rem 3.4rem;
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    margin-bottom: 0.6rem;
    font-size: 0.95rem;
    line-height: 1.75;
    transition: border-color var(--transition-base);
  }

  .usage-steps li:hover {
    border-color: var(--border-hover);
  }

  .usage-steps li::before {
    content: counter(step);
    position: absolute;
    left: 0.9rem;
    top: 1rem;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--accent) 0%, #9b7ef0 100%);
    color: #fff;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(91, 141, 239, 0.2);
  }

  /* ── References Grid ── */
  .ref-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
    margin-top: 1rem;
  }

  .ref-card {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 1rem 1.2rem;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    text-decoration: none;
    transition: all var(--transition-base);
  }

  .ref-card:hover {
    border-color: var(--border-hover);
    background: var(--bg-elevated);
    transform: translateY(-2px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
  }

  .ref-icon {
    flex-shrink: 0;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-glow);
    border-radius: var(--radius-md);
    color: var(--accent-bright);
  }

  .ref-card h4 {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
  }

  .ref-card p {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.5;
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .about { padding: 1.5rem 1rem 3rem; }
    .hero-title { font-size: 2rem; }
    .hero-subtitle { font-size: 1rem; }
    .section h2 { font-size: 1.2rem; }
    .hide-mobile { display: none; }
    .step { flex-direction: column; }
    .config-grid { grid-template-columns: repeat(2, 1fr); }
    .pillar-grid { grid-template-columns: 1fr; }
    .problem-grid { grid-template-columns: 1fr; }
    .ref-grid { grid-template-columns: 1fr; }
    .toc-items { gap: 0; }
    .guide-mode { flex-direction: column; gap: 0.3rem; align-items: flex-start; }
  }

  @media (max-width: 480px) {
    .about { padding: 1rem 0.75rem 2rem; }
    .hero-title { font-size: 1.6rem; }
    .section h2 { font-size: 1.1rem; }
    .hero-meta { flex-direction: column; gap: 0.3rem; }
    .hero-sep { display: none; }
    .config-grid { grid-template-columns: repeat(2, 1fr); }
    table { font-size: 0.76rem; }
    th, td { padding: 0.5rem 0.6rem; }
  }
</style>
