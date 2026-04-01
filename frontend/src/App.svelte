<script>
  import { onMount } from 'svelte';
  import { BDHModel } from './lib/BDHModel.js';
  import { tokenize } from './lib/tokenizer.js';
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

  onMount(async () => {
    await model.load('./model.onnx');
    modelReady.set(true);
  });

  async function handleInput(event) {
    const { tokens } = event.detail;
    if (!tokens || tokens.length === 0 || !model.ready) return;

    inferring.set(true);

    try {
      const result = await model.runToken(tokens);
      inferenceData.set(result);

      const layer = result.layers[0];
      model.updateSigma(layer.x_last, layer.y_last, 0);

      if (result.layers.length > 1) {
        model.updateSigma(result.layers[1].x_last, result.layers[1].y_last, 1);
      }

      totalTokens++;
      tokenCount.set(totalTokens);

      sigmaFlat = model.getSigma($selectedLayer, $selectedHead);
      sigmaData.set({ ...model.sigma });
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      inferring.set(false);
    }
  }

  function handleClearMemory() {
    model.resetMemory();
    totalTokens = 0;
    tokenCount.set(0);
    sigmaFlat = null;
    sigmaData.set(null);
    inferenceData.set(null);
  }

  $: sigmaFlat = model.getSigma($selectedLayer, $selectedHead);
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
      <button class="about-btn" on:click={() => { currentPage = 'about'; window.scrollTo(0, 0); }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        <span class="about-btn-text">About</span>
      </button>
    </div>
  </header>

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
      <LayerSelector />
    </section>

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
