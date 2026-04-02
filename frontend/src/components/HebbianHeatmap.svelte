<script>
  import { onMount, createEventDispatcher } from 'svelte';
  import * as d3 from 'd3';
  import { N } from '../lib/activation_math.js';
  import { topNeuronsBySigma, extractSubmatrix } from '../lib/activation_math.js';
  import synapseData from '../data/synapse_labels.json';

  const dispatch = createEventDispatcher();

  export let sigmaFlat = null;  // Float32Array(N*N) for current layer/head
  export let tokenCount = 0;
  export let sigmaDelta = null; // { norm, maxVal, changedCells }

  let svgEl;
  const SIZE = 64;
  const cellSize = 4;
  const gap = 0;
  const W = SIZE * (cellSize + gap);
  const H = SIZE * (cellSize + gap);

  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

  // Build a lookup of synapse positions
  const synapseLookup = {};
  for (const [concept, data] of Object.entries(synapseData)) {
    for (const pair of data.pairs) {
      synapseLookup[`${pair.i}_${pair.j}`] = {
        concept,
        color: data.color,
        label: data.label,
      };
    }
  }

  let topIndices = [];
  let submatrix = new Float32Array(SIZE * SIZE);
  let maxVal = 1;

  $: if (sigmaFlat) {
    topIndices = topNeuronsBySigma(sigmaFlat, SIZE);
    submatrix = extractSubmatrix(sigmaFlat, topIndices);
    const absMax = Math.max(...Array.from(submatrix).map(Math.abs));
    maxVal = absMax > 0 ? absMax : 1;
    renderHeatmap();
  } else {
    submatrix = new Float32Array(SIZE * SIZE);
    maxVal = 1;
    renderHeatmap();
  }

  function renderHeatmap() {
    if (!svgEl) return;

    const svg = d3.select(svgEl);

    const cells = svg.selectAll('rect').data(Array.from(submatrix));

    cells.enter()
      .append('rect')
      .merge(cells)
      .attr('x', (_, i) => (i % SIZE) * (cellSize + gap))
      .attr('y', (_, i) => Math.floor(i / SIZE) * (cellSize + gap))
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('fill', v => {
        const norm = Math.abs(v) / maxVal;
        return norm > 0.01 ? colorScale(norm) : '#050505';
      })
      .attr('stroke', (_, i) => {
        const row = Math.floor(i / SIZE);
        const col = i % SIZE;
        if (row < topIndices.length && col < topIndices.length) {
          const key = `${topIndices[row]}_${topIndices[col]}`;
          const syn = synapseLookup[key];
          if (syn) return syn.color;
        }
        return 'none';
      })
      .attr('stroke-width', (_, i) => {
        const row = Math.floor(i / SIZE);
        const col = i % SIZE;
        if (row < topIndices.length && col < topIndices.length) {
          const key = `${topIndices[row]}_${topIndices[col]}`;
          if (synapseLookup[key]) return 1.5;
        }
        return 0;
      });

    cells.exit().remove();
  }

  // Tooltip
  let tooltipText = '';
  let tooltipX = 0;
  let tooltipY = 0;
  let showTooltip = false;

  function handleMouseMove(e) {
    if (!svgEl) return;
    const rect = svgEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / (cellSize + gap));
    const row = Math.floor(y / (cellSize + gap));

    if (row >= 0 && row < SIZE && col >= 0 && col < SIZE) {
      const val = submatrix[row * SIZE + col];
      const ni = topIndices[row];
      const nj = topIndices[col];
      let extra = '';
      const syn = synapseLookup[`${ni}_${nj}`];
      if (syn) extra = ` — ${syn.label}`;

      tooltipText = `σ(${ni}, ${nj}) = ${val.toFixed(4)}${extra}`;
      tooltipX = e.clientX;
      tooltipY = e.clientY;
      showTooltip = true;
    } else {
      showTooltip = false;
    }
  }

  function handleMouseLeave() { showTooltip = false; }

  function handleClear() {
    dispatch('clear');
  }

  onMount(renderHeatmap);
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h3 class="panel-title">
        <svg class="title-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
        Hebbian Memory (σ)
      </h3>
      <p class="panel-desc">
        Synaptic state updated via <code>σ += outer(y, x)</code> — neurons that fire together wire together.
        Unlike a transformer's KV-cache, σ is <strong>fixed-size</strong> regardless of context length.
      </p>
    </div>
    <button class="clear-btn" on:click={handleClear} aria-label="Clear Hebbian memory">
      <span class="clear-icon" aria-hidden="true">✕</span> Clear
    </button>
  </div>

  <div class="heatmap-container">
    <svg
      bind:this={svgEl}
      width={W}
      height={H}
      role="img"
      aria-label="Hebbian memory heatmap"
      on:mousemove={handleMouseMove}
      on:mouseleave={handleMouseLeave}
      class="heatmap-svg"
    ></svg>

    <div class="legend">
      <div class="legend-bar"></div>
      <div class="legend-labels">
        <span>strong</span>
        <span>weak</span>
        <span>0</span>
      </div>
    </div>
  </div>

  <div class="synapse-key">
    <span class="synapse-label">Synapse Concepts:</span>
    {#each Object.entries(synapseData) as [concept, data]}
      <span class="synapse-tag" style="border-color: {data.color}; color: {data.color}">
        {data.label}
      </span>
    {/each}
  </div>

  {#if sigmaDelta && sigmaDelta.changedCells > 0}
    <div class="delta-bar">
      <span class="delta-icon" aria-hidden="true">⚡</span>
      <span class="delta-text">
        Δσ: <strong>{sigmaDelta.changedCells}</strong> synapses strengthened
        <span class="delta-stat">(max Δ = {sigmaDelta.maxVal.toFixed(4)})</span>
      </span>
      <div class="delta-indicator">
        <div
          class="delta-fill"
          style="width: {Math.min(100, sigmaDelta.changedCells / 10)}%"
        ></div>
      </div>
    </div>
  {/if}

  <p class="footnote">
    The 64×64 heatmap shows the top hub neurons from σ ∈ ℝ<sup>N×N</sup>.
    Coloured outlines mark <strong>discovered synapse concepts</strong> — semantically meaningful
    connection clusters (Section 6 of the paper).
    {#if tokenCount > 0}
      <span class="token-counter">{tokenCount} tokens encoded</span>
    {/if}
  </p>
</div>

{#if showTooltip}
  <div class="tooltip" style="left: {tooltipX + 12}px; top: {tooltipY - 8}px;">
    {tooltipText}
  </div>
{/if}

<style>
  .panel {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  .panel:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.8rem;
    gap: 0.8rem;
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

  .title-icon { flex-shrink: 0; opacity: 0.8; }

  .panel-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0.2rem 0 0;
    line-height: 1.6;
    max-width: 40ch;
  }

  .panel-desc code {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.08rem 0.35rem;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 4px;
    color: var(--cyan);
  }

  .panel-desc strong {
    color: var(--gold);
  }

  .clear-btn {
    padding: 0.3rem 0.7rem;
    background: transparent;
    border: 1px solid var(--rose);
    border-radius: var(--radius-sm);
    color: var(--rose);
    font-family: var(--font-sans);
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .clear-btn:hover {
    background: var(--rose);
    color: #fff;
    box-shadow: 0 2px 8px rgba(244, 63, 94, 0.3);
  }

  .clear-icon { font-size: 0.65rem; }

  .heatmap-container {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
  }

  .heatmap-svg {
    border-radius: var(--radius-md);
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid var(--border-subtle);
    display: block;
  }

  .legend {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    flex-shrink: 0;
  }

  .legend-bar {
    width: 14px;
    height: 200px;
    background: linear-gradient(to bottom, #fde725, #21918c, #440154);
    border-radius: 4px;
    border: 1px solid var(--border-subtle);
  }

  .legend-labels {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 200px;
    font-size: 0.55rem;
    font-family: var(--font-mono);
    color: var(--text-dim);
  }

  .synapse-key {
    display: flex;
    gap: 0.4rem;
    margin-top: 0.7rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .synapse-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .synapse-tag {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    padding: 0.12rem 0.45rem;
    border: 1px solid;
    border-radius: 999px;
    opacity: 0.85;
  }

  .footnote {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0.6rem 0 0;
    line-height: 1.6;
  }

  .footnote strong {
    color: var(--gold);
  }

  .token-counter {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    color: var(--accent);
    font-weight: 600;
    margin-left: 0.3rem;
  }

  .tooltip {
    position: fixed;
    background: var(--bg-elevated);
    color: var(--text-primary);
    padding: 0.35rem 0.65rem;
    border-radius: var(--radius-sm);
    font-size: 0.76rem;
    font-family: var(--font-mono);
    pointer-events: none;
    z-index: 100;
    border: 1px solid var(--border-default);
    white-space: nowrap;
    box-shadow: var(--shadow-elevated);
  }

  .delta-bar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.5rem;
    padding: 0.4rem 0.6rem;
    background: rgba(250, 204, 21, 0.04);
    border: 1px solid rgba(250, 204, 21, 0.15);
    border-radius: var(--radius-sm);
    flex-wrap: wrap;
  }

  .delta-icon { font-size: 0.75rem; }

  .delta-text {
    font-size: 0.75rem;
    color: var(--text-secondary);
  }

  .delta-text strong {
    color: var(--gold);
    font-weight: 700;
  }

  .delta-stat {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
  }

  .delta-indicator {
    flex: 1;
    min-width: 40px;
    height: 4px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 2px;
    overflow: hidden;
  }

  .delta-fill {
    height: 100%;
    background: var(--gold);
    border-radius: 2px;
    transition: width 0.15s ease;
  }

  @media (max-width: 600px) {
    .panel-header {
      flex-direction: column;
    }

    .panel-desc {
      max-width: none;
    }

    .heatmap-container {
      flex-direction: column;
    }

    .legend {
      flex-direction: row;
      align-items: center;
    }

    .legend-bar {
      width: 160px;
      height: 12px;
      background: linear-gradient(to right, #440154, #21918c, #fde725);
    }

    .legend-labels {
      flex-direction: row;
      height: auto;
      width: 160px;
    }
  }
</style>
