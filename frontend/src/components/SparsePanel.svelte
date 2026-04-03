<script>
  import NeuronGrid from './NeuronGrid.svelte';
  import StatsBar from './StatsBar.svelte';
  import InsightBadge from './InsightBadge.svelte';
  import { flatActivations, sparsityStats, gptActivations, gptSparsityStats, gptReady, sparsityHistory } from '../lib/stores.js';
  import { TOTAL_NEURONS } from '../lib/activation_math.js';

  // Real GPT activations from ONNX inference (256 neurons, ~97-100% density)
  $: gptActs = $gptActivations || new Float32Array(256);
  $: gptStats = $gptSparsityStats;

  $: sparsityRatio = parseFloat(gptStats.pct) > 0 && parseFloat($sparsityStats.pct) > 0
    ? (parseFloat(gptStats.pct) / parseFloat($sparsityStats.pct)).toFixed(0)
    : '0';

  $: insightText = $sparsityStats.active > 0 && gptStats.active > 0
    ? `BDH fires <strong>${$sparsityStats.pct}%</strong> of ${$sparsityStats.total} neurons vs GPT's <strong>${gptStats.pct}%</strong> of ${gptStats.total} — same input, real models, ${sparsityRatio}× difference.`
    : '';

  // Sparkline drawing
  let sparkCanvas;
  let sparkContainer;
  const SPARK_H = 48;

  $: if (sparkCanvas && $sparsityHistory.length > 1) {
    drawSparkline($sparsityHistory);
  }

  function drawSparkline(history) {
    // Match canvas resolution to actual displayed size
    const rect = sparkContainer?.getBoundingClientRect();
    const w = rect ? Math.round(rect.width * (window.devicePixelRatio || 1)) : 600;
    const dpr = window.devicePixelRatio || 1;
    const displayW = rect ? rect.width : 600;
    sparkCanvas.width = w;
    sparkCanvas.height = SPARK_H * dpr;
    const ctx = sparkCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayW, SPARK_H);

    const pts = history.slice(-60);
    if (pts.length < 2) return;

    const minPct = Math.min(...pts.map(p => p.pct));
    const maxPct = Math.max(...pts.map(p => p.pct));
    const range = Math.max(maxPct - minPct, 2); // at least 2% range to show variation
    const pad = 6;
    const stepX = displayW / (pts.length - 1);

    const toY = (pct) => pad + (1 - (pct - minPct) / range) * (SPARK_H - pad * 2);

    // Fill area
    ctx.beginPath();
    ctx.moveTo(0, SPARK_H);
    pts.forEach((p, i) => ctx.lineTo(i * stepX, toY(p.pct)));
    ctx.lineTo((pts.length - 1) * stepX, SPARK_H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(91, 141, 239, 0.1)';
    ctx.fill();

    // Line
    ctx.beginPath();
    pts.forEach((p, i) => {
      if (i === 0) ctx.moveTo(0, toY(p.pct));
      else ctx.lineTo(i * stepX, toY(p.pct));
    });
    ctx.strokeStyle = '#5b8def';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Current dot
    const lastPt = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc((pts.length - 1) * stepX, toY(lastPt.pct), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#7da8f5';
    ctx.fill();
  }
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h3 class="panel-title">
        <svg class="title-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        Sparse Activation
      </h3>
      <p class="panel-desc">
        Each square = 1 neuron. Lit pixels = active after {$gptReady ? 'ReLU (BDH) vs GELU (GPT)' : 'ReLU'}.
        Both models run on the same input — the density difference is real.
      </p>
    </div>
  </div>

  <div class="comparison">
    <div class="side">
      <NeuronGrid activations={$flatActivations} label="BDH · {$sparsityStats.total} neurons" accentColor="var(--accent)" />
      <StatsBar
        active={$sparsityStats.active}
        total={$sparsityStats.total}
        pct={$sparsityStats.pct}
        label="ReLU sparse (~5% at scale)"
        color="var(--accent)"
      />
    </div>

    <div class="vs-divider">
      <div class="vs-line"></div>
      <span class="vs-text">VS</span>
      <div class="vs-line"></div>
    </div>

    <div class="side">
      {#if $gptReady}
        <NeuronGrid activations={gptActs} label="GPT · {gptStats.total} neurons" accentColor="var(--rose)" cols={16} rows={16} />
        <StatsBar
          active={gptStats.active}
          total={gptStats.total}
          pct={gptStats.pct}
          label="GELU dense (real GPT)"
          color="var(--rose)"
        />
      {:else}
        <div class="gpt-loading">
          <span class="gpt-loading-text">Loading transformer…</span>
        </div>
      {/if}
    </div>
  </div>

  <InsightBadge type="key" text={insightText} />

  {#if parseFloat($sparsityStats.pct) > 0 && parseFloat($sparsityStats.pct) < 3}
    <InsightBadge
      type="wow"
      text="Sparsity below 3% — the model is <em>less surprised</em> by this input. BDH literally uses fewer neurons on predictable text (Section 6.4 of the paper)."
    />
  {:else if parseFloat($sparsityStats.pct) > 6}
    <InsightBadge
      type="wow"
      text="Sparsity above 6% — the model is <em>more surprised</em>! Novel input activates more neurons. This is BDH's uncertainty indicator."
    />
  {/if}

  <!-- Sparsity Sparkline — Novel insight: sparsity tracks uncertainty -->
  {#if $sparsityHistory.length > 2}
    <div class="sparkline-section" bind:this={sparkContainer}>
      <div class="sparkline-header">
        <span class="sparkline-label">Sparsity Over Time</span>
        <span class="sparkline-hint">↑ novel input &nbsp; ↓ predictable text</span>
      </div>
      <canvas
        bind:this={sparkCanvas}
        class="sparkline-canvas"
        aria-label="Sparsity trend over keystrokes"
      ></canvas>
    </div>
  {/if}
</div>

<style>
  .panel {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    overflow: hidden;
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  .panel:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
  }

  .panel-header {
    margin-bottom: 1rem;
  }

  .panel-title {
    font-family: var(--font-sans);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    letter-spacing: -0.02em;
  }

  .title-icon {
    flex-shrink: 0;
    opacity: 0.8;
  }

  .panel-desc {
    font-size: 0.87rem;
    color: var(--text-muted);
    margin: 0.25rem 0 0;
    line-height: 1.65;
    max-width: 55ch;
  }

  .comparison {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0.8rem;
    flex-wrap: wrap;
  }

  .side {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    min-width: 200px;
  }

  .vs-divider {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    align-self: center;
    padding: 0 0.3rem;
  }

  .vs-line {
    width: 1px;
    height: 40px;
    background: var(--border-subtle);
  }

  .vs-text {
    font-family: var(--font-sans);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-dim);
    letter-spacing: 0.1em;
  }

  .gpt-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 180px;
    min-width: 180px;
  }

  .gpt-loading-text {
    font-size: 0.8rem;
    color: var(--text-dim);
    font-style: italic;
  }

  .sparkline-section {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border-subtle);
  }

  .sparkline-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.4rem;
  }

  .sparkline-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: -0.01em;
  }

  .sparkline-hint {
    font-size: 0.72rem;
    color: var(--text-dim);
  }

  .sparkline-canvas {
    width: 100%;
    height: 48px;
    border-radius: var(--radius-sm);
    background: var(--bg-secondary);
  }

  @media (max-width: 600px) {
    .comparison {
      flex-direction: column;
      align-items: center;
    }

    .vs-divider {
      flex-direction: row;
      gap: 0.6rem;
      padding: 0.3rem 0;
    }

    .vs-line {
      width: 40px;
      height: 1px;
    }
  }
</style>
