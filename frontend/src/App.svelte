<script>
  import { onMount } from 'svelte';
  import { BDHModel } from './lib/BDHModel.js';
  import { GPTModel } from './lib/GPTModel.js';
  import { tokenize, tokenToChar } from './lib/tokenizer.js';
  import {
    inputText, tokenBuffer, inferenceData, sigmaData,
    modelReady, flatActivations, activeNeuronIds,
    selectedLayer, selectedHead, tokenCount, inferring,
    gptReady, gptData, sparsityHistory, sigmaDelta,
  } from './lib/stores.js';

  import TokenInput from './components/TokenInput.svelte';
  import LayerSelector from './components/LayerSelector.svelte';
  import SparsePanel from './components/SparsePanel.svelte';
  import GraphBrain from './components/GraphBrain.svelte';
  import HebbianHeatmap from './components/HebbianHeatmap.svelte';
  import AttentionPanel from './components/AttentionPanel.svelte';
  import MemoryPanel from './components/MemoryPanel.svelte';
  import AboutPage from './components/AboutPage.svelte';

  const model = new BDHModel();
  const gptModel = new GPTModel();

  let sigmaFlat = null;
  let sigmaDeltaLocal = null;
  let totalTokens = 0;
  let currentPage = 'visualizer';
  let inferenceMs = 0;
  let lastLogits = null;
  let topPredictions = [];
  let sigmaTopPredictions = [];
  let gptTopPredictions = [];
  let predictionShift = null;  // { from, to } when σ changes top-1
  let showGuide = false;

  // ── Demo Mode ──
  let demoRunning = false;
  let demoTimer = null;
  let demoText = null;
  const DEMO_SCRIPT = "To be, or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.";

  function startDemo() {
    if (demoRunning || teachRunning) { stopDemo(); return; }
    handleClearMemory();
    demoRunning = true;
    let idx = 0;

    demoTimer = setInterval(() => {
      if (idx >= DEMO_SCRIPT.length) { stopDemo(); return; }
      idx++;
      const partial = DEMO_SCRIPT.slice(0, idx);
      demoText = partial;

      const bytes = tokenize(partial);
      const tail = Array.from(bytes.slice(-128));
      inputText.set(partial);
      tokenBuffer.set(tail);

      handleInput({ detail: { tokens: tail } });
    }, 120);
  }

  function stopDemo() {
    demoRunning = false;
    teachRunning = false;
    teachPhase = '';
    if (demoTimer) clearInterval(demoTimer);
    demoTimer = null;
    demoText = null;
  }

  // ── Teach Mode ──
  let teachRunning = false;
  let teachPhase = '';  // 'repeat-1', 'repeat-2', 'repeat-3', 'test'
  const TEACH_PHRASE = "the cat sat on the mat. ";
  const TEACH_TEST = "the cat sat on the ";

  function startTeach() {
    if (teachRunning || demoRunning) { stopDemo(); return; }
    handleClearMemory();
    teachRunning = true;
    teachPhase = 'repeat-1';

    let fullScript = TEACH_PHRASE + TEACH_PHRASE + TEACH_PHRASE + TEACH_TEST;
    let idx = 0;
    let phaseLen1 = TEACH_PHRASE.length;
    let phaseLen2 = phaseLen1 * 2;
    let phaseLen3 = phaseLen1 * 3;

    demoTimer = setInterval(() => {
      if (idx >= fullScript.length) { clearInterval(demoTimer); demoTimer = null; teachRunning = false; teachPhase = 'done'; demoText = null; return; }
      idx++;

      if (idx <= phaseLen1) teachPhase = 'repeat-1';
      else if (idx <= phaseLen2) teachPhase = 'repeat-2';
      else if (idx <= phaseLen3) teachPhase = 'repeat-3';
      else teachPhase = 'test';

      const partial = fullScript.slice(0, idx);
      demoText = partial;

      const bytes = tokenize(partial);
      const tail = Array.from(bytes.slice(-128));
      inputText.set(partial);
      tokenBuffer.set(tail);

      handleInput({ detail: { tokens: tail } });
    }, 100);
  }

  onMount(async () => {
    await model.load('./model.onnx');
    await model.loadWeights('./bdh_weights.bin');
    modelReady.set(true);

    // Load GPT model using shared ort runtime
    if (model._ort) {
      await gptModel.load('./transformer.onnx', model._ort);
      gptReady.set(gptModel.ready);
    }
  });

  let inferenceSeq = 0;

  async function handleInput(event) {
    const { tokens } = event.detail;
    if (!tokens || tokens.length === 0) {
      inferenceData.set(null);
      gptData.set(null);
      sigmaFlat = null;
      sigmaDeltaLocal = null;
      lastLogits = null;
      topPredictions = [];
      sigmaTopPredictions = [];
      gptTopPredictions = [];
      predictionShift = null;
      return;
    }
    if (!model.ready) return;

    const seq = ++inferenceSeq;
    inferring.set(true);

    try {
      // Run BDH inference
      const t0 = performance.now();
      const result = await model.runToken(tokens);
      const t1 = performance.now();
      if (seq !== inferenceSeq) return;

      inferenceMs = Math.round(t1 - t0);
      inferenceData.set(result);

      lastLogits = result.logits || null;
      topPredictions = model.topKPredictions(lastLogits, 5);

      // Snapshot σ before update for delta tracking
      const prevSigma = model.getSigma($selectedLayer, $selectedHead).slice();

      const layer = result.layers[0];
      model.updateSigma(layer.x_last, layer.y_last, 0);

      if (result.layers.length > 1) {
        model.updateSigma(result.layers[1].x_last, result.layers[1].y_last, 1);
      }

      totalTokens++;
      tokenCount.set(totalTokens);

      // Compute σ-modulated predictions (inference-time learning)
      const sigmaLogits = model.computeSigmaLogits(
        result.layers[$selectedLayer].x_last, $selectedLayer
      );
      if (sigmaLogits && lastLogits) {
        // α grows with experience: more tokens → stronger σ influence
        const alpha = 0.01 * Math.log(1 + totalTokens);
        const adjustedLogits = new Float32Array(lastLogits.length);
        for (let i = 0; i < lastLogits.length; i++) {
          adjustedLogits[i] = lastLogits[i] + alpha * sigmaLogits[i];
        }
        sigmaTopPredictions = model.topKPredictions(adjustedLogits, 5);

        // Track prediction shift
        if (topPredictions.length > 0 && sigmaTopPredictions.length > 0) {
          const rawTop = topPredictions[0].token;
          const sigmaTop = sigmaTopPredictions[0].token;
          predictionShift = rawTop !== sigmaTop ? { from: rawTop, to: sigmaTop } : null;
        }
      } else {
        sigmaTopPredictions = [];
        predictionShift = null;
      }

      sigmaFlat = model.getSigma($selectedLayer, $selectedHead).slice();
      sigmaData.set({ ...model.sigma });

      // Compute σ delta for inference-time learning visualization
      let deltaMax = 0;
      let deltaChanged = 0;
      for (let i = 0; i < sigmaFlat.length; i++) {
        const d = Math.abs(sigmaFlat[i] - prevSigma[i]);
        if (d > 1e-8) {
          deltaChanged++;
          if (d > deltaMax) deltaMax = d;
        }
      }
      sigmaDeltaLocal = { norm: 0, maxVal: deltaMax, changedCells: deltaChanged };
      sigmaDelta.set(sigmaDeltaLocal);

      // Track sparsity over time for sparkline
      const actCount = result.layers[$selectedLayer]
        ? result.layers[$selectedLayer].x_last.reduce(
            (sum, head) => sum + head.filter(v => v > 1e-6).length, 0
          )
        : 0;
      const actTotal = result.layers[$selectedLayer]
        ? result.layers[$selectedLayer].x_last.reduce((sum, head) => sum + head.length, 0)
        : 1;
      const pct = (actCount / actTotal) * 100;
      sparsityHistory.update(h => [...h.slice(-59), { pct }]);

      // Run GPT inference (non-blocking relative to BDH)
      if (gptModel.ready) {
        try {
          const gptResult = await gptModel.runToken(tokens);
          if (seq === inferenceSeq) {
            gptData.set(gptResult);
            if (gptResult && gptResult.logits) {
              gptTopPredictions = gptModel.topKPredictions(gptResult.logits, 5);
            } else {
              gptTopPredictions = [];
            }
          }
        } catch (err) {
          console.warn('GPT inference error:', err);
        }
      }
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      if (seq === inferenceSeq) inferring.set(false);
    }
  }

  function handleClearMemory() {
    inferenceSeq++;             // cancel any in-flight inference
    model.resetMemory();
    totalTokens = 0;
    tokenCount.set(0);
    sigmaFlat = null;
    sigmaDeltaLocal = null;
    sigmaData.set(null);
    sigmaDelta.set(null);
    inferenceData.set(null);
    gptData.set(null);
    sparsityHistory.set([]);
    lastLogits = null;
    topPredictions = [];
    sigmaTopPredictions = [];
    gptTopPredictions = [];
    predictionShift = null;
    teachPhase = '';
    demoText = null;
  }

  function autoFocus(node) { requestAnimationFrame(() => node.focus()); }

  $: if ($selectedLayer !== undefined || $selectedHead !== undefined) {
    sigmaFlat = model.getSigma($selectedLayer, $selectedHead).slice();
  }
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
      <button class="about-btn demo-btn" class:demo-active={demoRunning} on:click={startDemo}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span class="about-btn-text">{demoRunning ? 'Stop' : 'Demo'}</span>
      </button>
      <button class="about-btn teach-btn" class:teach-active={teachRunning} on:click={startTeach}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>
        <span class="about-btn-text">{teachRunning ? 'Stop' : 'Teach'}</span>
      </button>
      <button class="about-btn" on:click={() => { currentPage = 'about'; window.scrollTo(0, 0); }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        <span class="about-btn-text">About</span>
      </button>
    </div>
  </header>

  <!-- ── Quick Guide Overlay ── -->
  {#if showGuide}
    <div class="guide-backdrop" on:click={() => showGuide = false} role="presentation"></div>
    <div class="guide-panel" role="dialog" aria-label="Quick guide" tabindex="-1" on:keydown={e => e.key === 'Escape' && (showGuide = false)} use:autoFocus>
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
        <li><strong>Type text</strong> in the input box. The model processes each character as a byte token (0–255). Byte tokens appear on the right.</li>
        <li><strong>Watch the panels</strong> update in real time — sparse activation grid, Hebbian heatmap, attention pattern, and active graph nodes all respond instantly.</li>
        <li><strong>Read the predictions</strong> — three rows appear below the input: <em>BDH Raw</em> (base model output), <em>σ-Learned</em> (adjusted by Hebbian memory — watch for ⇄ shift indicators), and <em>GPT</em> (transformer baseline comparison).</li>
        <li><strong>Switch layers and heads</strong> using L1/L2 and H1/H2 buttons. Different heads often specialize in different patterns.</li>
        <li><strong>Explore the graph</strong> — switch between Gx (Thought Flow) and Gy (Memory Echo). Drag and zoom to inspect hub neurons.</li>
        <li><strong>Clear memory</strong> with the Clear button on the Hebbian panel, then retype to watch σ rebuild from scratch.</li>
        <li><strong>Try Teach mode</strong> — click the Teach button in the header. The model sees "the cat sat on the mat" three times, building σ memory. Then watch σ-Learned predictions shift toward "mat" — the model learned the pattern with no gradient updates.</li>
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
      <TokenInput on:input={handleInput} externalText={demoText} />
      <div class="controls-row">
        <LayerSelector />
        {#if inferenceMs > 0}
          <span class="inference-time">{inferenceMs}ms</span>
        {/if}
      </div>
    </section>

    <!-- ── Predictions Bar ── -->
    {#if topPredictions.length > 0}
      <section class="predictions-bar" aria-label="Next token predictions">
        <div class="pred-row">
          <span class="pred-label">BDH Raw:</span>
          {#each topPredictions as pred}
            <span class="pred-token" title="Byte {pred.token} — {(pred.prob * 100).toFixed(1)}%">
              <span class="pred-char">{tokenToChar(pred.token)}</span>
              <span class="pred-prob">{(pred.prob * 100).toFixed(0)}%</span>
            </span>
          {/each}
        </div>
        {#if sigmaTopPredictions.length > 0}
          <div class="pred-row sigma-row">
            <span class="pred-label sigma-label">σ-Learned:</span>
            {#each sigmaTopPredictions as pred}
              <span class="pred-token sigma-token" title="Byte {pred.token} — {(pred.prob * 100).toFixed(1)}% (after σ modulation)">
                <span class="pred-char">{tokenToChar(pred.token)}</span>
                <span class="pred-prob">{(pred.prob * 100).toFixed(0)}%</span>
              </span>
            {/each}
            {#if predictionShift}
              <span class="shift-indicator" title="σ memory shifted the top prediction">
                ⇄ shifted: <strong>{tokenToChar(predictionShift.from)}</strong> → <strong>{tokenToChar(predictionShift.to)}</strong>
              </span>
            {/if}
            <span class="sigma-indicator" title="Predictions modulated by accumulated Hebbian memory">
              ⚡ {totalTokens} tokens learned
            </span>
          </div>
        {/if}
        {#if gptTopPredictions.length > 0}
          <div class="pred-row gpt-row">
            <span class="pred-label gpt-label">GPT:</span>
            {#each gptTopPredictions as pred}
              <span class="pred-token gpt-token" title="GPT byte {pred.token} — {(pred.prob * 100).toFixed(1)}%">
                <span class="pred-char">{tokenToChar(pred.token)}</span>
                <span class="pred-prob">{(pred.prob * 100).toFixed(0)}%</span>
              </span>
            {/each}
            <span class="gpt-indicator">Transformer baseline</span>
          </div>
        {/if}
      </section>
    {/if}

    <!-- ── Teach Phase Indicator ── -->
    {#if teachRunning || teachPhase === 'done'}
      <section class="teach-bar">
        <span class="teach-icon">🧠</span>
        {#if teachPhase === 'repeat-1'}
          <span class="teach-text">Teaching: <strong>Repetition 1/3</strong> — building σ memory...</span>
        {:else if teachPhase === 'repeat-2'}
          <span class="teach-text">Teaching: <strong>Repetition 2/3</strong> — σ strengthening...</span>
        {:else if teachPhase === 'repeat-3'}
          <span class="teach-text">Teaching: <strong>Repetition 3/3</strong> — σ patterns consolidating...</span>
        {:else if teachPhase === 'test'}
          <span class="teach-text"><strong>Testing:</strong> Watch σ-Learned predictions shift toward "mat" — the model <em>remembers</em>!</span>
        {:else if teachPhase === 'done'}
          <span class="teach-text"><strong>Done!</strong> σ memory learned the pattern. Try typing yourself to see predictions.</span>
        {/if}
        <div class="teach-progress">
          <div class="teach-progress-fill" style="width: {teachPhase === 'repeat-1' ? '25' : teachPhase === 'repeat-2' ? '50' : teachPhase === 'repeat-3' ? '75' : '100'}%"></div>
        </div>
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
          sigmaDelta={sigmaDeltaLocal}
          on:clear={handleClearMemory}
        />
        <AttentionPanel />
      </div>

      <div class="panel-row insights-row">
        <MemoryPanel />
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
    padding: 1.2rem 2rem 2rem;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0 1.2rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.6rem;
    flex-wrap: wrap;
    gap: 0.8rem;
  }

  .logo-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
  }

  .logo-icon {
    font-size: 2rem;
    animation: float 3s ease-in-out infinite;
    line-height: 1;
  }

  h1 {
    font-family: var(--font-sans);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.2;
  }

  .tagline {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 400;
    letter-spacing: -0.01em;
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
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-family: var(--font-sans);
    font-size: 0.84rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-base);
    letter-spacing: -0.01em;
  }

  .about-btn:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: var(--border-hover);
    color: var(--text-primary);
    transform: translateY(-1px);
  }

  .demo-btn {
    border-color: var(--green);
    color: var(--green);
  }

  .demo-active {
    background: rgba(61, 214, 140, 0.12) !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
    animation: demoPulse 1.5s ease-in-out infinite;
  }

  @keyframes demoPulse {
    0%, 100% { box-shadow: 0 0 6px rgba(61, 214, 140, 0.15); }
    50% { box-shadow: 0 0 14px rgba(61, 214, 140, 0.35); }
  }

  .teach-btn {
    border-color: var(--cyan, #4dd0e1);
    color: var(--cyan, #4dd0e1);
  }

  .teach-active {
    background: rgba(77, 208, 225, 0.12) !important;
    border-color: var(--cyan, #4dd0e1) !important;
    color: var(--cyan, #4dd0e1) !important;
    animation: demoPulse 1.5s ease-in-out infinite;
  }

  .teach-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.9rem;
    background: rgba(77, 208, 225, 0.06);
    border: 1px solid rgba(77, 208, 225, 0.2);
    border-radius: var(--radius-md);
    margin-bottom: 0.8rem;
    animation: slideUp 0.3s ease;
  }

  .teach-icon { font-size: 1.1rem; }

  .teach-text {
    font-size: 0.86rem;
    color: var(--text-secondary);
    flex: 1;
  }

  .teach-text strong { color: var(--cyan, #4dd0e1); }
  .teach-text em { color: var(--gold); font-style: normal; }

  .teach-progress {
    width: 60px;
    height: 4px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 2px;
    overflow: hidden;
    flex-shrink: 0;
  }

  .teach-progress-fill {
    height: 100%;
    background: var(--cyan, #4dd0e1);
    border-radius: 2px;
    transition: width 0.3s ease;
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
    box-shadow: 0 0 40px rgba(91, 141, 239, 0.08), 0 8px 32px rgba(0, 0, 0, 0.5);
    animation: slideUp 0.25s ease;
  }

  .guide-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .guide-title {
    font-family: var(--font-sans);
    font-size: 1.15rem;
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
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.6;
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
    font-family: var(--font-sans);
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: -0.01em;
  }

  .loading-sub {
    font-size: 0.85rem;
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

  .inference-time {
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.03em;
  }

  /* ── Predictions Bar ── */
  .predictions-bar {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    padding: 0.45rem 0.7rem;
    margin-bottom: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow-x: auto;
  }

  .pred-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }

  .pred-label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.03em;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .pred-token {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.55rem;
    background: rgba(91, 141, 239, 0.08);
    border: 1px solid rgba(91, 141, 239, 0.18);
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    transition: background var(--transition-fast);
    cursor: default;
  }

  .pred-token:hover {
    background: rgba(91, 141, 239, 0.14);
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

  .sigma-row {
    border-top: 1px solid rgba(240, 194, 70, 0.12);
    padding-top: 0.3rem;
  }

  .sigma-label {
    color: var(--gold) !important;
  }

  .sigma-token {
    background: rgba(240, 194, 70, 0.08) !important;
    border-color: rgba(240, 194, 70, 0.25) !important;
  }

  .sigma-token:hover {
    background: rgba(240, 194, 70, 0.15) !important;
  }

  .sigma-indicator {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--gold);
    opacity: 0.85;
    margin-left: auto;
  }

  .shift-indicator {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--rose);
    background: rgba(240, 98, 146, 0.08);
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    border: 1px solid rgba(240, 98, 146, 0.2);
    animation: shiftPulse 0.5s ease;
  }

  .shift-indicator strong {
    color: var(--text-primary);
  }

  @keyframes shiftPulse {
    0% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }

  .gpt-row {
    border-top: 1px solid rgba(155, 126, 240, 0.12);
    padding-top: 0.3rem;
  }

  .gpt-label {
    color: var(--violet, #9b7ef0) !important;
  }

  .gpt-token {
    background: rgba(155, 126, 240, 0.08) !important;
    border-color: rgba(155, 126, 240, 0.25) !important;
  }

  .gpt-token:hover {
    background: rgba(155, 126, 240, 0.15) !important;
  }

  .gpt-indicator {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--violet, #9b7ef0);
    opacity: 0.85;
    margin-left: auto;
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

  .insights-row {
    grid-template-columns: 1fr;
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
    font-size: 0.85rem;
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
    font-size: 0.82rem;
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
      padding: 0.6rem 1rem 1.5rem;
    }

    h1 {
      font-size: 1.25rem;
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
