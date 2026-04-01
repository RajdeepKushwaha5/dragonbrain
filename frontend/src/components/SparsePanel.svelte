<script>
  import NeuronGrid from './NeuronGrid.svelte';
  import StatsBar from './StatsBar.svelte';
  import InsightBadge from './InsightBadge.svelte';
  import { flatActivations, sparsityStats, tokenBuffer } from '../lib/stores.js';
  import { generateDenseReference, countActive, TOTAL_NEURONS } from '../lib/activation_math.js';

  $: denseActivations = $tokenBuffer.length > 0
    ? generateDenseReference(new Uint8Array($tokenBuffer))
    : new Float32Array(TOTAL_NEURONS);
  $: denseStats = countActive(denseActivations);

  $: sparsityRatio = parseFloat($sparsityStats.pct) > 0
    ? (denseStats.pct / parseFloat($sparsityStats.pct)).toFixed(0)
    : '0';
  $: insightText = $sparsityStats.active > 0
    ? `BDH uses <strong>${$sparsityStats.pct}%</strong> of neurons vs <strong>${denseStats.pct.toFixed(1)}%</strong> in transformers — a <strong>${sparsityRatio}×</strong> reduction in compute.`
    : '';
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h3 class="panel-title">
        <svg class="title-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        Sparse Activation
      </h3>
      <p class="panel-desc">
        Each square = 1 neuron. Lit pixels = active neurons after ReLU. BDH achieves significant sparsity vs ~99% dense activation in transformers (~5% in the paper's large models).
      </p>
    </div>
  </div>

  <div class="comparison">
    <div class="side">
      <NeuronGrid activations={$flatActivations} label="BDH (Sparse)" accentColor="var(--accent)" />
      <StatsBar
        active={$sparsityStats.active}
        total={$sparsityStats.total}
        pct={$sparsityStats.pct}
        label="sparse (paper: ~5% at scale)"
        color="var(--accent)"
      />
    </div>

    <div class="vs-divider">
      <div class="vs-line"></div>
      <span class="vs-text">VS</span>
      <div class="vs-line"></div>
    </div>

    <div class="side">
      <NeuronGrid activations={denseActivations} label="Transformer (Dense)" accentColor="var(--rose)" />
      <StatsBar
        active={denseStats.count}
        total={denseStats.total}
        pct={denseStats.pct.toFixed(1)}
        label="~97% typical"
        color="var(--rose)"
      />
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
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .title-icon {
    flex-shrink: 0;
    opacity: 0.8;
  }

  .panel-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0.2rem 0 0;
    line-height: 1.6;
    max-width: 50ch;
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
    font-family: var(--font-display);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-dim);
    letter-spacing: 0.1em;
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
