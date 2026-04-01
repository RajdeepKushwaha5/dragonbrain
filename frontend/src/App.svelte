<script>
  import { onMount } from 'svelte';
  import { BDHModel } from './lib/BDHModel.js';
  import { tokenize, detokenize, tokenToChar } from './lib/tokenizer.js';
  import {
    inputText, tokenBuffer, inferenceData, sigmaData,
    modelReady, flatActivations, activeNeuronIds,
    selectedLayer, selectedHead, tokenCount, inferring,
  } from './lib/stores.js';

  import TokenInput from './components/TokenInput.svelte';
  import LayerSelector from './components/LayerSelector.svelte';
  import SparsePanel from './components/SparsePanel.svelte';
  import GraphBrain from './components/GraphBrain.svelte';
  import HebbianHeatmap from './components/HebbianHeatmap.svelte';
  import AttentionPanel from './components/AttentionPanel.svelte';
  import AboutPage from './components/AboutPage.svelte';

  const model = new BDHModel();

  let sigmaFlat = null;
  let totalTokens = 0;
  let currentPage = 'visualizer';
  let inferenceMs = 0;
  let lastLogits = null;
  let topPredictions = [];
  let generating = false;
  let showGuide = false;

  onMount(async () => {
    await model.load('./model.onnx');
    modelReady.set(true);
  });

  let inferenceSeq = 0;

  async function handleInput(event) {
    const { tokens } = event.detail;
    if (!tokens || tokens.length === 0) {
      inferenceData.set(null);
      sigmaFlat = null;
      lastLogits = null;
      topPredictions = [];
      return;
    }
    if (!model.ready) return;

    const seq = ++inferenceSeq;
    inferring.set(true);

    try {
      const t0 = performance.now();
      const result = await model.runToken(tokens);
      const t1 = performance.now();
      if (seq !== inferenceSeq) return; // stale result from earlier keystroke

      inferenceMs = Math.round(t1 - t0);
      inferenceData.set(result);

      // Store logits for generation
      lastLogits = result.logits || null;
      topPredictions = model.topKPredictions(lastLogits, 5);

      const layer = result.layers[0];
      model.updateSigma(layer.x_last, layer.y_last, 0);

      if (result.layers.length > 1) {
        model.updateSigma(result.layers[1].x_last, result.layers[1].y_last, 1);
      }

      totalTokens++;
      tokenCount.set(totalTokens);

      // .slice() creates a new reference so Svelte detects the change
      sigmaFlat = model.getSigma($selectedLayer, $selectedHead).slice();
      sigmaData.set({ ...model.sigma });
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      if (seq === inferenceSeq) inferring.set(false);
    }
  }

  async function handleGenerate() {
    if (!model.ready || generating) return;
    generating = true;

    const GENERATE_COUNT = 32;
    let currentTokens = [...$tokenBuffer];
    if (currentTokens.length === 0) {
      // Start with a space
      currentTokens = [32];
    }

    for (let step = 0; step < GENERATE_COUNT; step++) {
      if (!generating) break;

      const t0 = performance.now();
      const result = await model.runToken(currentTokens);
      const t1 = performance.now();
      inferenceMs = Math.round(t1 - t0);

      if (!result.logits) break;

      const nextToken = model.sampleFromLogits(result.logits, 0.8);
      currentTokens.push(nextToken);
      // Truncate to last 128
      if (currentTokens.length > 128) {
        currentTokens = currentTokens.slice(-128);
      }

      // Update all stores so panels animate
      inferenceData.set(result);
      lastLogits = result.logits;
      topPredictions = model.topKPredictions(lastLogits, 5);
      tokenBuffer.set(currentTokens);
      inputText.set(detokenize(currentTokens));

      const layer = result.layers[0];
      model.updateSigma(layer.x_last, layer.y_last, 0);
      if (result.layers.length > 1) {
        model.updateSigma(result.layers[1].x_last, result.layers[1].y_last, 1);
      }

      totalTokens++;
      tokenCount.set(totalTokens);
      sigmaFlat = model.getSigma($selectedLayer, $selectedHead).slice();
      sigmaData.set({ ...model.sigma });

      // Small delay so user can see each step
      await new Promise(r => setTimeout(r, 60));
    }

    generating = false;
  }

  function stopGeneration() {
    generating = false;
  }

  function handleClearMemory() {
    model.resetMemory();
    totalTokens = 0;
    tokenCount.set(0);
    sigmaFlat = null;
    sigmaData.set(null);
    inferenceData.set(null);
    lastLogits = null;
    topPredictions = [];
    generating = false;
  }

  $: sigmaFlat = model.getSigma($selectedLayer, $selectedHead).slice();
</script>

<main>
  <!-- ── Hero Header ── -->
  <header>
    <div class="header-left">
      <div class="logo-row">
        <span class="logo-icon" aria-hidden="true">🐉</span>
        <div>
          <h1>Dragon Brain</h1>
          <p class="tagline">Interactive BDH Architecture Explorer</p>
        </div>
      </div>
    </div>
    <div class="header-right">
      <button class="about-btn" on:click={() => showGuide = !showGuide}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span class="about-btn-text">Guide</span>
      </button>
      <button class="about-btn" on:click={() => { currentPage = 'about'; window.scrollTo(0, 0); }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        <span class="about-btn-text">About</span>
      </button>
    </div>
  </header>

  <!-- ── Quick Guide Overlay ── -->
  {#if showGuide}
    <div class="guide-backdrop" on:click={() => showGuide = false} on:keydown={e => e.key === 'Escape' && (showGuide = false)} role="presentation"></div>
    <div class="guide-panel" role="dialog" aria-label="Quick guide">
      <div class="guide-header">
        <h2 class="guide-title">
          <span class="guide-icon">💡</span>
          Quick Guide
        </h2>
        <button class="guide-close" on:click={() => showGuide = false} aria-label="Close guide">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <ol class="guide-steps">
        <li><strong>Type text</strong> in the input box. The model processes each character as a byte token (0–255).</li>
        <li><strong>Watch the panels</strong> update in real time — sparse activation grid, Hebbian heatmap, attention pattern, and active graph nodes all respond instantly.</li>
        <li><strong>Switch layers and heads</strong> using L1/L2 and H1/H2 buttons. Different heads often specialise in different patterns.</li>
        <li><strong>Explore the graph</strong> — switch between Gx (Thought Flow) and Gy (Memory Echo). Drag and zoom to inspect hub neurons.</li>
        <li><strong>Generate text</strong> — click Generate to auto-complete 32 tokens and watch every panel animate step-by-step.</li>
        <li><strong>Clear memory</strong> with the Clear button on the Hebbian panel, then retype to watch σ rebuild from scratch.</li>
      </ol>
    </div>
  {/if}

  {#if currentPage === 'about'}
    <AboutPage on:back={() => currentPage = 'visualizer'} />
  {:else}

  <!-- ── Loading State ── -->
  {#if !$modelReady}
    <div class="loading-state">
      <div class="spinner-ring">
        <div class="spinner-inner"></div>
      </div>
      <p class="loading-text">Initialising BDH neural model…</p>
      <p class="loading-sub">Loading ONNX weights into WebAssembly</p>
    </div>
  {:else}
    <!-- ── Controls ── -->
    <section class="controls-section">
      <TokenInput on:input={handleInput} />
      <div class="controls-row">
        <LayerSelector />
        <div class="gen-controls">
          {#if generating}
            <button class="gen-btn gen-stop" on:click={stopGeneration}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>
              Stop
            </button>
          {:else}
            <button class="gen-btn" on:click={handleGenerate} disabled={!$modelReady}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              Generate
            </button>
          {/if}
          {#if inferenceMs > 0}
            <span class="inference-time">{inferenceMs}ms</span>
          {/if}
        </div>
      </div>
    </section>

    <!-- ── Predictions Bar ── -->
    {#if topPredictions.length > 0}
      <section class="predictions-bar" aria-label="Next token predictions">
        <span class="pred-label">Next:</span>
        {#each topPredictions as pred}
          <span class="pred-token" title="Byte {pred.token} — {(pred.prob * 100).toFixed(1)}%">
            <span class="pred-char">{tokenToChar(pred.token)}</span>
            <span class="pred-prob">{(pred.prob * 100).toFixed(0)}%</span>
          </span>
        {/each}
      </section>
    {/if}

    <!-- ── Panel Grid ── -->
    <section class="panels" aria-label="Visualization panels">
      <div class="panel-row top-row">
        <SparsePanel />
        <GraphBrain activeNeuronIds={$activeNeuronIds} />
      </div>

      <div class="panel-row bottom-row">
        <HebbianHeatmap
          sigmaFlat={sigmaFlat}
          tokenCount={totalTokens}
          on:clear={handleClearMemory}
        />
        <AttentionPanel />
      </div>
    </section>
  {/if}

  {/if}

  <!-- ── Footer ── -->
  <footer>
    <div class="footer-inner">
      <p class="footer-credit">
        Built by <a href="https://rajdeep-singh.vercel.app/" target="_blank" rel="noopener"><strong>Rajdeep Singh</strong></a> for the Beyond Transformers Hackathon
      </p>
      <div class="footer-links">
        <a href="https://arxiv.org/abs/2509.26507" target="_blank" rel="noopener">Paper</a>
        <span class="footer-sep">·</span>
        <a href="https://github.com/pathwaycom/bdh" target="_blank" rel="noopener">Repo</a>
        <span class="footer-sep">·</span>
        <a href="https://pathway.com" target="_blank" rel="noopener">Pathway</a>
      </div>
    </div>
  </footer>
</main>

<style>
  /* ── Layout ── */
  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1rem 1.5rem 2rem;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 0 1.2rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    gap: 0.8rem;
  }

  .logo-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }

  .logo-icon {
    font-size: 2.4rem;
    animation: float 3s ease-in-out infinite;
    line-height: 1;
  }

  h1 {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.08em;
    line-height: 1.2;
    background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .tagline {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 400;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .about-btn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.9rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-family: var(--font-sans);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition-base);
    letter-spacing: 0.01em;
  }

  .about-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: var(--border-hover);
    color: var(--text-primary);
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.06);
    transform: translateY(-1px);
  }

  /* ── Guide Overlay ── */
  .guide-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(4px);
    z-index: 90;
    animation: fadeIn 0.2s ease;
  }

  .guide-panel {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(520px, 92vw);
    max-height: 80vh;
    overflow-y: auto;
    background: var(--bg-secondary);
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.8rem;
    z-index: 100;
    box-shadow: 0 0 40px rgba(59, 130, 246, 0.08), 0 8px 32px rgba(0, 0, 0, 0.5);
    animation: slideUp 0.25s ease;
  }

  .guide-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .guide-title {
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
  }

  .guide-icon {
    font-size: 1.2rem;
  }

  .guide-close {
    background: none;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-dim);
    cursor: pointer;
    padding: 0.3rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition-fast);
  }

  .guide-close:hover {
    color: var(--text-primary);
    border-color: var(--border-hover);
    background: rgba(255, 255, 255, 0.06);
  }

  .guide-steps {
    list-style: none;
    counter-reset: guide;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
  }

  .guide-steps li {
    counter-increment: guide;
    position: relative;
    padding: 0.7rem 0.9rem 0.7rem 2.8rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    font-size: 0.84rem;
    color: var(--text-secondary);
    line-height: 1.55;
    transition: border-color var(--transition-fast);
  }

  .guide-steps li:hover {
    border-color: var(--border-hover);
  }

  .guide-steps li::before {
    content: counter(guide);
    position: absolute;
    left: 0.75rem;
    top: 0.65rem;
    width: 1.5rem;
    height: 1.5rem;
    background: var(--accent);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--font-mono);
  }

  .guide-steps li strong {
    color: var(--text-primary);
  }

  /* ── Loading ── */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6rem 0;
    flex: 1;
  }

  .spinner-ring {
    width: 48px;
    height: 48px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
    margin-bottom: 1.2rem;
  }

  .loading-text {
    font-family: var(--font-display);
    font-size: 0.9rem;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
  }

  .loading-sub {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
  }

  /* ── Controls ── */
  .controls-section {
    margin-bottom: 1.2rem;
  }

  .controls-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
  }

  .gen-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-left: auto;
  }

  .gen-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.8rem;
    font-size: 0.78rem;
    font-family: var(--font-mono);
    font-weight: 500;
    color: var(--accent);
    background: transparent;
    border: 1px solid var(--accent);
    border-radius: 6px;
    cursor: pointer;
    transition: background var(--transition-fast), box-shadow var(--transition-fast);
    letter-spacing: 0.02em;
  }

  .gen-btn:hover:not(:disabled) {
    background: rgba(59, 130, 246, 0.1);
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.15);
  }

  .gen-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .gen-stop {
    color: #fb7185;
    border-color: #fb7185;
  }

  .gen-stop:hover {
    background: rgba(251, 113, 133, 0.1);
    box-shadow: 0 0 12px rgba(251, 113, 133, 0.15);
  }

  .inference-time {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.03em;
  }

  /* ── Predictions Bar ── */
  .predictions-bar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.7rem;
    margin-bottom: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow-x: auto;
    flex-wrap: wrap;
  }

  .pred-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .pred-token {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.5rem;
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    transition: background var(--transition-fast);
    cursor: default;
  }

  .pred-token:hover {
    background: rgba(59, 130, 246, 0.15);
  }

  .pred-char {
    color: var(--text-primary);
    font-weight: 600;
    min-width: 0.8em;
    text-align: center;
  }

  .pred-prob {
    color: var(--text-dim);
    font-size: 0.68rem;
  }

  /* ── Panels ── */
  .panels {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    flex: 1;
    animation: slideUp 0.4s ease;
  }

  .panel-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
  }

  /* ── Footer ── */
  footer {
    margin-top: auto;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-subtle);
  }

  .footer-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
  }

  .footer-credit {
    font-size: 0.82rem;
    color: var(--text-muted);
  }

  .footer-credit strong {
    color: var(--text-secondary);
    font-weight: 600;
  }

  .footer-links {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
  }

  .footer-links a {
    color: var(--text-secondary);
    transition: color var(--transition-fast);
  }

  .footer-links a:hover {
    color: var(--accent-bright);
  }

  .footer-sep {
    color: var(--text-dim);
  }

  /* ── Responsive ── */
  @media (max-width: 1024px) {
    main {
      padding: 0.8rem 1rem 1.5rem;
    }
  }

  @media (max-width: 768px) {
    main {
      padding: 0.6rem 0.8rem 1.5rem;
    }

    h1 {
      font-size: 1.3rem;
    }

    .logo-icon {
      font-size: 1.8rem;
    }

    .about-btn-text {
      display: none;
    }

    .panel-row {
      grid-template-columns: 1fr;
    }

    .footer-inner {
      flex-direction: column;
      text-align: center;
    }
  }
</style>
