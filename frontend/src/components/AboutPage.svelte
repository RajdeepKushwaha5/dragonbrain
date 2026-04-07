<script>
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  const dispatch = createEventDispatcher();

  const tocItems = [
    { id: 'glossary', label: 'Glossary' },
    { id: 'what', label: 'What This Is' },
    { id: 'why', label: 'Why BDH' },
    { id: 'pillars', label: 'Five Pillars' },
    { id: 'arch', label: 'Architecture' },
    { id: 'compare', label: 'BDH vs Transformers' },
    { id: 'panels', label: 'Panels' },
    { id: 'config', label: 'Configuration' },
    { id: 'concepts', label: 'Concepts' },
    { id: 'usage', label: 'Usage' },
    { id: 'refs', label: 'Credits' },
  ];

  let activeSection = 'glossary';
  let sidebarOpen = false;
  let observer;

  function handleNavClick(id) {
    activeSection = id;
    sidebarOpen = false;
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  onMount(() => {
    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            activeSection = entry.target.id;
          }
        }
      },
      { rootMargin: '-10% 0px -70% 0px', threshold: 0 }
    );
    for (const item of tocItems) {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    }
  });

  onDestroy(() => {
    if (observer) observer.disconnect();
  });
</script>

<div class="about-layout">
  <!-- Sidebar -->
  <button class="sidebar-toggle" on:click={() => sidebarOpen = !sidebarOpen} aria-label="Toggle navigation">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      {#if sidebarOpen}
        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
      {:else}
        <line x1="3" y1="7" x2="21" y2="7"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="17" x2="21" y2="17"/>
      {/if}
    </svg>
  </button>

  <!-- svelte-ignore a11y-click-events-have-key-events -->
  {#if sidebarOpen}
    <div class="sidebar-backdrop" on:click={() => sidebarOpen = false}></div>
  {/if}

  <nav class="sidebar" class:open={sidebarOpen} aria-label="Table of contents">
    <span class="sidebar-label">Contents</span>
    {#each tocItems as item}
      <button
        class="sidebar-link"
        class:active={activeSection === item.id}
        on:click={() => handleNavClick(item.id)}
      >
        {item.label}
      </button>
    {/each}
  </nav>

  <div class="about">
    <button class="back-btn" on:click={() => dispatch('back')} aria-label="Back to visualizer">
      &larr; Back to Visualizer
    </button>

    <header class="hero">
      <h1>Dragon Brain</h1>
      <p class="hero-sub">
        An interactive visualizer for the Baby Dragon Hatchling architecture —
        a post-transformer neural network design proposed by Kosowski et al. (2025).
      </p>
      <div class="hero-meta">
        <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener">Read the Paper</a>
        <span class="sep">&middot;</span>
        <span>Kosowski et al. 2025</span>
      </div>
    </header>

    <section class="section" id="glossary">
      <h2>Quick Reference — All Terms</h2>
      <div class="glossary-scroll">
        <table class="glossary-table">
          <thead>
            <tr><th>Term</th><th>Full Form / Meaning</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>BDH</strong></td><td>Baby Dragon Hatchling — the novel neural architecture</td></tr>
            <tr><td><strong>GPT</strong></td><td>Generative Pre-trained Transformer — the standard baseline</td></tr>
            <tr><td><strong>&sigma; (sigma)</strong></td><td>Hebbian co-activation memory matrix — fixed-size, named after Greek letter sigma</td></tr>
            <tr><td><strong>&Delta;&sigma; (delta sigma)</strong></td><td>Change in &sigma; per token — how many synapses strengthened and by how much</td></tr>
            <tr><td><strong>ReLU</strong></td><td>Rectified Linear Unit — <code>max(0, x)</code> — outputs zero for negatives &rarr; sparse activations</td></tr>
            <tr><td><strong>GELU</strong></td><td>Gaussian Error Linear Unit — smooth activation used in GPT &rarr; nearly all neurons fire</td></tr>
            <tr><td><strong>RoPE</strong></td><td>Rotary Position Embedding — encodes relative token position via rotation instead of adding position vectors</td></tr>
            <tr><td><strong>Gx</strong></td><td><code>Encoder &times; Decoder_x</code> — feedforward causal circuit, called "Thought Flow"</td></tr>
            <tr><td><strong>Gy</strong></td><td><code>Decoder_y<sup>T</sup> &times; Decoder_x</code> — Hebbian memory readout graph, called "Memory Echo"</td></tr>
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
            <tr><td><strong>Outer product</strong></td><td>y &otimes; x — element (i,j) = y[i] &times; x[j]. This is how &sigma; records which neurons co-activated</td></tr>
            <tr><td><strong>Causal mask</strong></td><td>Enforces that token at position t can only attend to tokens 0&hellip;t&minus;1, never future tokens</td></tr>
            <tr><td><strong>Softmax</strong></td><td>Converts raw scores into probabilities summing to 1. Used in Transformer attention; BDH does not use it</td></tr>
            <tr><td><strong>O(T&sup2;)</strong></td><td>Quadratic complexity — grows with the square of sequence length (standard attention)</td></tr>
            <tr><td><strong>O(T)</strong></td><td>Linear complexity — grows proportionally to sequence length (BDH's recurrent form)</td></tr>
            <tr><td><strong>O(1)</strong></td><td>Constant — doesn't grow with input (&sigma; memory size)</td></tr>
            <tr><td><strong>&alpha; (alpha)</strong></td><td>Blending coefficient for &sigma;-Learned: &alpha; = 0.05 &times; log(1 + T) — grows logarithmically</td></tr>
            <tr><td><strong>D3.js</strong></td><td>Data-Driven Documents — JavaScript library for interactive visualizations</td></tr>
            <tr><td><strong>Hub neurons</strong></td><td>Neurons with many connections in the emergent graph — organizational centers</td></tr>
            <tr><td><strong>Excitatory edge</strong></td><td>Connection where one neuron amplifies another's signal (positive weight)</td></tr>
            <tr><td><strong>Inhibitory edge</strong></td><td>Connection where one neuron suppresses another's signal (negative weight)</td></tr>
            <tr><td><strong>Synapse</strong></td><td>Connection between two neurons. &sigma;(i,j) records how strongly neurons i and j co-activate</td></tr>
            <tr><td><strong>Viridis</strong></td><td>Perceptually-uniform color scale (dark &rarr; green &rarr; yellow) used for the Hebbian heatmap</td></tr>
            <tr><td><strong>Byte token</strong></td><td>Character encoded as raw byte value (0&ndash;255). 'A' = 65, 'a' = 97, space = 32</td></tr>
            <tr><td><strong>Inference</strong></td><td>Running a trained model on new input to get predictions (vs training)</td></tr>
            <tr><td><strong>Inference-time learning</strong></td><td>BDH's ability to learn (update &sigma;) during inference, without retraining or gradients</td></tr>
            <tr><td><strong>IIT Ropar</strong></td><td>Indian Institute of Technology Ropar</td></tr>
            <tr><td><strong>Pathway</strong></td><td>The company behind the BDH research (pathway.com)</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section" id="what">
      <h2>What This Is</h2>
      <p>
        <strong>Dragon Brain</strong> lets you type text and watch a neural network respond
        in real time. The architecture is <strong>BDH</strong> (Baby Dragon Hatchling) —
        a fundamentally different approach to sequence modeling proposed in
        <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener">"The Dragon Hatchling"</a>
        by Kosowski et al. (2025).
      </p>
      <p>
        This isn't a diagram or a simulation. The actual model runs inference in your browser
        using ONNX Runtime (WebAssembly) — no server calls, no API keys. The neural network
        weights are loaded locally and computation happens on your machine.
      </p>
      <p>
        You can see which neurons fire, how attention flows between tokens, how the Hebbian
        memory matrix builds across a conversation, and how graph structure organizes computation
        — all from a single text input.
      </p>
    </section>

    <section class="section" id="why">
      <h2>Why BDH</h2>
      <p>
        Using transformers for language modeling works. What makes them imperfect is four things:
        <strong>dense computation</strong>, <strong>growing memory</strong>,
        <strong>opaque internals</strong>, and <strong>quadratic attention</strong>.
      </p>
      <p>
        <strong>Dense computation</strong> means nearly every neuron fires on every token.
        A GPT layer activates ~99% of its neurons regardless of input — most of that
        computation is irrelevant to the current token.
      </p>
      <p>
        <strong>Growing memory</strong> is the KV-cache problem. Every new token adds to
        a linearly growing cache. Long contexts get expensive fast.
      </p>
      <p>
        <strong>Opaque internals</strong> make interpretability hard. Individual neurons
        respond to multiple unrelated concepts (polysemantic). Understanding what a
        transformer has learned requires expensive post-hoc tools like Sparse AutoEncoders.
      </p>
      <p>
        <strong>Quadratic attention</strong> — softmax over Q@K<sup>T</sup> — scales
        as O(T&sup2;d) per layer, limiting practical sequence length.
      </p>
      <p>
        BDH addresses all four by drawing on neuroscience: sparse coding, Hebbian learning,
        and modular network organization.
      </p>
    </section>

    <section class="section" id="pillars">
      <h2>The Five Pillars of BDH</h2>
      <p>
        BDH combines five architectural principles. Each addresses a specific transformer limitation.
      </p>
      <p>
        <strong>Sparse activation.</strong> Only a small fraction of neurons fire per token —
        roughly 5% in the paper's large models, vs ~99% in transformers. Each neuron represents
        one concept. This is natively monosemantic — no SAE needed.
      </p>
      <p>
        <strong>Emergent graph.</strong> Hub neurons and modular communities emerge from random
        initialization during training. Nobody programs the structure. It forms spontaneously,
        following a scale-free power law like biological neural networks.
      </p>
      <p>
        <strong>Hebbian memory (&sigma;).</strong> "Neurons that fire together wire together."
        The &sigma; matrix is a fixed-size memory that records co-activation patterns. Unlike a KV-cache,
        it doesn't grow with context length.
      </p>
      <p>
        <strong>Linear attention.</strong> No softmax. BDH uses RoPE rotation with a causal mask.
        Q = K = x (the same sparse activation vector), making attention a self-correlation readout.
        The recurrent form runs in O(T) per step.
      </p>
      <p>
        <strong>Composable merging.</strong> The scale-free architecture enables concatenating
        independently trained models — a French translator and a Spanish translator merged into
        one multilingual model, no retraining. This composability is architecturally native
        (Section 7.1 of the paper).
      </p>
    </section>

    <section class="section" id="arch">
      <h2>Architecture — Step by Step</h2>
      <p>Each BDH layer processes token embeddings through five stages.</p>

      <h3>1. Sparse Activation (x)</h3>
      <p>
        The token embedding v* (dimension D=64) is projected into a larger neuron space
        (N=512 per head) through <code>Decoder<sub>x</sub></code>, then passed through ReLU.
        Negative values become zero, leaving only a sparse subset of active neurons.
      </p>
      <div class="formula">
        <code>x = ReLU( v* @ Decoder<sub>x</sub> )</code>
      </div>

      <h3>2. Linear Attention</h3>
      <p>
        The same sparse activation serves as both Query and Key — no separate Q/K projections.
        Scores are <code>RoPE(x) @ RoPE(x)<sup>T</sup></code> with a causal mask.
        <strong>No softmax</strong> — raw dot products after RoPE rotation. Attention becomes
        a self-correlation: how similar are the sparse patterns across tokens?
      </p>
      <div class="formula">
        <code>a* = Attention( Q=x, K=x, V=v* )</code>
      </div>
      <p>
        The parallel form is O(T&sup2;), but the paper describes a recurrent form using cumulative sums
        that achieves O(T) per step during inference.
      </p>

      <h3>3. Gated Readout (y)</h3>
      <p>
        The attended value is projected through <code>Decoder<sub>y</sub></code>,
        then element-wise multiplied (&odot;) by x. This gating ensures y is nonzero only where x
        was already active — forming (x, y) co-activation pairs that drive Hebbian learning.
      </p>
      <div class="formula">
        <code>y = ReLU( LN(a*) @ Decoder<sub>y</sub> ) &odot; x</code>
      </div>

      <h3>4. Hebbian Memory Update (&sigma;)</h3>
      <p>
        After each token, the outer product (&otimes;) of y and x accumulates into a fixed-size matrix
        &sigma; &isin; &#8477;<sup>N&times;N</sup>. Unlike a KV-cache that grows with every new token,
        &sigma; has a fixed size regardless of context length.
      </p>
      <div class="formula">
        <code>&sigma; += y &otimes; x</code>
      </div>

      <h3>5. Residual Update</h3>
      <p>
        y is mapped back to the D-dimensional embedding space using the <code>Encoder</code>
        matrix and added as a residual. Layer normalization is applied before and after.
        The updated v* feeds into the next layer.
      </p>
      <div class="formula">
        <code>v* = LN( v* + LN( y @ Encoder ) )</code>
      </div>
    </section>

    <section class="section" id="compare">
      <h2>BDH vs Transformers</h2>
      <div class="table-wrap">
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
              <td class="bdh-val">&sigma; matrix: fixed O(N&sup2;), Hebbian outer product</td>
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

    <section class="section" id="panels">
      <h2>Understanding the Visualizer Panels</h2>
      <p>Each panel shows a different aspect of BDH computation.</p>

      <h3>Sparse Activation</h3>
      <p>
        A 32&times;32 grid of 1,024 neurons (2 heads &times; 512 each). Lit pixels are active neurons
        after ReLU. The left grid shows BDH activations, the right shows a dense GPT reference.
        Predictable text fires ~2&ndash;3% of neurons. Novel input pushes it to ~6&ndash;7% —
        BDH's built-in uncertainty indicator.
      </p>

      <h3>Emergent Graph</h3>
      <p>
        Visualizes internal wiring using a D3 force-directed layout. Two modes:
        <strong>Gx</strong> (<code>Encoder @ Decoder<sub>x</sub></code>) shows the feedforward
        circuit — "Thought Flow." <strong>Gy</strong> (<code>Decoder<sub>y</sub><sup>T</sup> @ Decoder<sub>x</sub></code>)
        shows the Hebbian readout — "Memory Echo." Hub neurons (large circles) emerge spontaneously.
        Yellow nodes are currently active.
      </p>

      <h3>Hebbian Memory (&sigma;)</h3>
      <p>
        A 64&times;64 heatmap of the most active rows and columns of &sigma;. The Viridis color scale maps
        strength from dark (zero) to yellow (strong). Clear and retype to watch it rebuild — the
        matrix stays the same size regardless of input length.
      </p>

      <h3>Attention Pattern</h3>
      <p>
        A T&times;T heatmap of raw attention scores. The causal mask enforces lower-triangular structure —
        each token attends only to previous tokens. Since Q=K=x, scores are inner products of sparse
        vectors. No softmax — what you see are raw dot products with RoPE encoding. Brighter cells near
        the diagonal indicate recent focus.
      </p>

      <h3>Memory Scaling</h3>
      <p>
        Compares BDH's constant &sigma; memory (flat line at 4 MB) against a GPT KV-cache (growing at
        ~1 KB per token). The lines cross at T &asymp; 4,096. At the paper's full scale (N=32,768),
        crossover happens at just T &asymp; 400 tokens.
      </p>

      <h3>Prediction Bar</h3>
      <p>
        Three rows of next-character predictions. <strong>BDH Raw</strong> is the base output.
        <strong>&sigma;-Learned</strong> adds a correction from accumulated Hebbian memory — when &sigma;
        changes the top prediction, a &ldquo;&lrarr; shifted&rdquo; indicator appears. <strong>GPT</strong> shows
        the transformer baseline. Use Teach mode to build up &sigma; quickly and see visible prediction shifts.
      </p>
    </section>

    <section class="section" id="config">
      <h2>Model Configuration</h2>
      <p>
        This visualizer uses a tiny BDH model — small enough for your browser while preserving
        all three key phenomena: sparsity, graph emergence, and Hebbian memory.
      </p>
      <div class="table-wrap">
        <table class="config-table">
          <tbody>
            <tr><td>Layers (L)</td><td>2</td></tr>
            <tr><td>Embedding (D)</td><td>64</td></tr>
            <tr><td>Heads (n<sub>h</sub>)</td><td>2</td></tr>
            <tr><td>Neurons / Head (N)</td><td>512</td></tr>
            <tr><td>Total Neurons</td><td>1,024</td></tr>
            <tr><td>Vocab Size</td><td>256 (byte-level)</td></tr>
            <tr><td>Max Sequence</td><td>128</td></tr>
            <tr><td>Runtime</td><td>ONNX + WASM</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        The weight matrices <code>Decoder<sub>x</sub></code>, <code>Decoder<sub>y</sub></code>,
        and <code>Encoder</code> are shared across all layers — the same projection is reused in
        every layer, a distinctive BDH design choice. The paper reports ~5% neuron sparsity for
        models with 32k+ neurons. This tiny model (1,024 neurons) may show higher activation
        rates — sparsity increases with model scale.
      </p>
    </section>

    <section class="section" id="concepts">
      <h2>Key Concepts</h2>

      <h3>Monosemantic Neurons</h3>
      <p>
        In transformers, individual neurons respond to multiple unrelated concepts (polysemantic).
        BDH's sparse ReLU forces each neuron to specialize in one concept. When neuron #247 fires,
        it means one specific thing — no Sparse AutoEncoder needed to decode it.
      </p>

      <h3>RoPE (Rotary Position Embedding)</h3>
      <p>
        Instead of adding positional embeddings, RoPE applies a rotation. When computing
        Q@K<sup>T</sup>, the rotations compose so the score depends only on relative distance
        between tokens. This works even when Q=K=x in BDH.
      </p>
      <div class="formula">
        <code>&theta;<sub>i</sub> = 10000<sup>&minus;2i/d</sup></code>
      </div>

      <h3>Scale-Free Networks</h3>
      <p>
        The emergent graph follows a power law — a few hub neurons have many connections, most have few.
        This mirrors biological neural networks, the internet, and social networks. The structure was
        not designed — it emerged from random initialization.
      </p>

      <h3>Hebbian Learning</h3>
      <p>
        Donald Hebb's 1949 principle: "When an axon of cell A repeatedly takes part in firing cell B,
        A's efficiency in firing B is increased." When neurons x and y co-activate, their connection
        in &sigma; strengthens. This accumulates a long-term memory that doesn't grow with context length.
      </p>
      <div class="formula">
        <code>&sigma; += y &otimes; x</code>
      </div>

      <h3>Inference-Time Learning</h3>
      <p>
        Unlike transformers whose weights are frozen at inference, BDH keeps learning through &sigma;.
        The correction blends in logarithmically — no gradient updates needed. The model adapts purely
        through co-activation memory.
      </p>
      <div class="formula">
        <code>&alpha; = 0.05 &times; log(1 + T)</code>
      </div>
      <div class="formula">
        <code>a* = ReLU(&sigma; &middot; x)</code>
      </div>
      <div class="formula">
        <code>logits&prime; = logits + &alpha; &times; scale &times; (E<sup>T</sup> &middot; a*) &middot; W<sub>lm</sub></code>
      </div>
      <p>
        The correction flows through the same encoder and language-model head used in training.
        <code>scale</code> normalizes &sigma;-logits to the same range as raw logits so they blend cleanly.
        Try Teach mode to see this in action.
      </p>
    </section>

    <section class="section" id="usage">
      <h2>How to Use This Visualizer</h2>
      <ol>
        <li><strong>Type text</strong> in the input box. The model processes each character as a byte token (0&ndash;255).</li>
        <li><strong>Watch the panels</strong> update in real time — sparse activation, Hebbian heatmap, attention, and graph all respond instantly.</li>
        <li><strong>Read the predictions</strong> — three rows below the input: BDH Raw, &sigma;-Learned, and GPT baseline.</li>
        <li><strong>Switch layers and heads</strong> with L1/L2 and H1/H2.</li>
        <li><strong>Explore the graph</strong> — toggle between Gx (Thought Flow) and Gy (Memory Echo). Drag and zoom to inspect hubs.</li>
        <li><strong>Clear memory</strong> with the Clear button, then retype to watch &sigma; rebuild from scratch.</li>
        <li><strong>Try Demo</strong> (&blacktriangleright;) — watch a Shakespeare passage type itself while all panels animate.</li>
        <li><strong>Try Teach</strong> (&phone;) — feed a repeated phrase three times, then watch &sigma;-Learned shift predictions.</li>
      </ol>
    </section>

    <section class="section" id="refs">
      <h2>References &amp; Credits</h2>
      <ul class="ref-list">
        <li>
          <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener">Kosowski et al. 2025 — "The Dragon Hatchling"</a>
          <span class="ref-desc">Research paper</span>
        </li>
        <li>
          <a href="https://github.com/pathwaycom/bdh" target="_blank" rel="noopener">github.com/pathwaycom/bdh</a>
          <span class="ref-desc">Official BDH code</span>
        </li>
        <li>
          <a href="https://pathway.com" target="_blank" rel="noopener">pathway.com</a>
          <span class="ref-desc">Pathway — the company behind BDH research</span>
        </li>
        <li>
          <a href="https://rajdeep-singh.vercel.app/" target="_blank" rel="noopener">Built by Rajdeep Singh</a>
          <span class="ref-desc">For the Beyond Transformers Hackathon (Pathway A)</span>
        </li>
      </ul>
    </section>
  </div>
</div>

<style>
  /* ── Layout ── */
  .about-layout {
    display: flex;
    min-height: 100vh;
  }

  .about {
    max-width: 740px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    animation: fadeIn 0.3s ease;
    flex: 1;
    min-width: 0;
  }

  /* ── Sidebar ── */
  .sidebar {
    position: sticky;
    top: 0;
    height: 100vh;
    width: 200px;
    flex-shrink: 0;
    padding: 1.5rem 0.6rem 2rem 1rem;
    background: var(--bg-card);
    border-right: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
    z-index: 50;
  }

  .sidebar-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    padding: 0.5rem 0.65rem 0.8rem;
    flex-shrink: 0;
  }

  .sidebar-link {
    display: block;
    width: 100%;
    text-align: left;
    font-family: var(--font-sans);
    font-size: 0.82rem;
    color: var(--text-muted);
    background: none;
    border: none;
    border-left: 2px solid transparent;
    padding: 0.45rem 0.65rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .sidebar-link:hover {
    color: var(--text-primary);
    background: rgba(255,255,255,0.04);
  }

  .sidebar-link.active {
    color: var(--accent);
    border-left-color: var(--accent);
    background: rgba(91,141,239,0.08);
    font-weight: 600;
  }

  .sidebar-toggle {
    display: none;
    position: fixed;
    bottom: 1.25rem;
    left: 1.25rem;
    z-index: 60;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
    color: #fff;
    cursor: pointer;
    box-shadow: 0 2px 12px rgba(91,141,239,0.4);
    align-items: center;
    justify-content: center;
    transition: transform 0.2s ease;
  }

  .sidebar-toggle:hover { transform: scale(1.08); }
  .sidebar-backdrop { display: none; }

  /* ── Back Button ── */
  .back-btn {
    display: inline-block;
    padding: 0.5rem 0;
    background: none;
    border: none;
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 0.88rem;
    cursor: pointer;
    transition: color 0.15s ease;
    margin-bottom: 1.5rem;
  }

  .back-btn:hover { color: var(--accent); }

  /* ── Hero ── */
  .hero {
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .hero h1 {
    font-family: var(--font-sans);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.75rem;
  }

  .hero-sub {
    font-size: 1.05rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 0.75rem;
  }

  .hero-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.88rem;
    color: var(--text-dim);
  }

  .hero-meta a {
    color: var(--accent-bright);
    text-decoration: none;
    font-weight: 500;
  }

  .hero-meta a:hover { color: var(--text-primary); }
  .hero-meta .sep { color: var(--text-dim); }

  /* ── Sections ── */
  .section {
    margin-bottom: 3rem;
    scroll-margin-top: 1rem;
  }

  .section h2 {
    font-family: var(--font-sans);
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .section h3 {
    font-family: var(--font-sans);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
  }

  .section h3:first-of-type {
    margin-top: 0.8rem;
  }

  .section p {
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.85;
    margin-bottom: 0.9rem;
  }

  .section p:last-child { margin-bottom: 0; }

  .section ol, .section ul {
    margin: 0.8rem 0 1rem 1.5rem;
    color: var(--text-secondary);
    font-size: 1rem;
    line-height: 1.85;
  }

  .section li { margin-bottom: 0.4rem; }

  .section a {
    color: var(--accent-bright);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: rgba(125, 168, 245, 0.3);
    transition: color 0.15s ease;
  }

  .section a:hover {
    color: #fff;
    text-decoration-color: var(--accent-bright);
  }

  .section code {
    font-family: var(--font-mono);
    font-size: 0.88em;
    padding: 0.15rem 0.4rem;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
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

  /* ── Formula blocks ── */
  .formula {
    display: flex;
    align-items: center;
    padding: 0.6rem 1rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
    margin: 0.5rem 0 0.8rem;
    width: fit-content;
    max-width: 100%;
  }

  .formula code {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--accent-bright);
    background: none;
    border: none;
    padding: 0;
    letter-spacing: 0.01em;
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

  /* ── Tables (comparison, config) ── */
  .table-wrap {
    overflow-x: auto;
    border-radius: var(--radius-md, 8px);
    border: 1px solid var(--border-default);
    margin: 1rem 0;
    background: var(--bg-card);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }

  th, td {
    padding: 0.75rem 1rem;
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

  .config-table td:first-child {
    color: var(--text-muted);
    font-size: 0.9rem;
    width: 40%;
  }

  .config-table td:last-child {
    font-family: var(--font-mono);
    color: var(--accent-bright);
    font-weight: 600;
  }

  /* ── Reference List ── */
  .ref-list {
    list-style: none;
    margin: 0 !important;
    padding: 0;
  }

  .ref-list li {
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .ref-list li:last-child { border-bottom: none; }

  .ref-desc {
    font-size: 0.82rem;
    color: var(--text-dim);
  }

  /* ── Responsive ── */
  @media (max-width: 1024px) {
    .sidebar {
      position: fixed;
      left: 0;
      top: 0;
      height: 100vh;
      transform: translateX(-100%);
      transition: transform 0.25s ease;
      border-right: 1px solid var(--border-default);
      box-shadow: 4px 0 24px rgba(0,0,0,0.4);
    }
    .sidebar.open { transform: translateX(0); }
    .sidebar-toggle { display: flex; }
    .sidebar-backdrop {
      display: block;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      z-index: 49;
    }
  }

  @media (max-width: 768px) {
    .about { padding: 1.5rem 1rem 3rem; }
    .hero h1 { font-size: 1.8rem; }
    .section h2 { font-size: 1.2rem; }
  }

  @media (max-width: 480px) {
    .about { padding: 1rem 0.75rem 2rem; }
    .hero h1 { font-size: 1.5rem; }
    .section h2 { font-size: 1.1rem; }
    .hero-meta { flex-direction: column; gap: 0.3rem; }
    table { font-size: 0.78rem; }
    th, td { padding: 0.5rem 0.6rem; }
  }
</style>
