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

  let tooltipText = '';
  let tooltipX = 0;
  let tooltipY = 0;
  let showTooltip = false;

  // Clickable synapse detail state
  let selectedConcept = null;
  let selectedConceptData = null;

  // Synapse activation timeline — per-concept σ magnitude history
  const synapseHistory = {};  // concept → Array<{ token, avgSigma }>
  let historyToken = 0;

  // Clear synapse history when memory is reset
  $: if (!sigmaFlat || tokenCount === 0) {
    for (const key of Object.keys(synapseHistory)) {
      delete synapseHistory[key];
    }
    historyToken = 0;
  }

  // Track synapse timeline on each sigma update
  $: if (sigmaFlat && tokenCount > 0) {
    for (const [concept, data] of Object.entries(synapseData)) {
      if (!synapseHistory[concept]) synapseHistory[concept] = [];
      let totalAbs = 0;
      let count = 0;
      for (const pair of data.pairs) {
        if (pair.i < N && pair.j < N) {
          totalAbs += Math.abs(sigmaFlat[pair.i * N + pair.j]);
          count++;
        }
      }
      const avgSigma = count > 0 ? totalAbs / count : 0;
      const hist = synapseHistory[concept];
      if (hist.length === 0 || hist[hist.length - 1].token !== tokenCount) {
        hist.push({ token: tokenCount, avgSigma });
        if (hist.length > 60) hist.shift();
      }
    }
    historyToken = tokenCount;  // trigger reactivity
  }

  function handleConceptClick(concept) {
    if (selectedConcept === concept) {
      selectedConcept = null;
      selectedConceptData = null;
    } else {
      selectedConcept = concept;
      const data = synapseData[concept];
      // Compute live σ values for each synapse pair
      const pairs = data.pairs.map(pair => {
        let sigmaVal = 0;
        if (sigmaFlat && pair.i < N && pair.j < N) {
          sigmaVal = sigmaFlat[pair.i * N + pair.j];
        }
        return { ...pair, sigmaVal };
      });
      selectedConceptData = { ...data, concept, pairs };
    }
  }

  // Update live σ values when sigma changes (or zero out when cleared)
  $: if (selectedConcept) {
    const data = synapseData[selectedConcept];
    if (data) {
      const pairs = data.pairs.map(pair => {
        let sigmaVal = 0;
        if (sigmaFlat && pair.i < N && pair.j < N) {
          sigmaVal = sigmaFlat[pair.i * N + pair.j];
        }
        return { ...pair, sigmaVal };
      });
      selectedConceptData = { ...data, concept: selectedConcept, pairs };
    }
  }

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
      <button
        class="synapse-tag"
        class:synapse-tag-active={selectedConcept === concept}
        style="border-color: {data.color}; color: {data.color}"
        on:click={() => handleConceptClick(concept)}
        title="Click to inspect {data.label} neuron pairs"
      >
        {data.label}
      </button>
    {/each}
  </div>

  {#if selectedConceptData}
    <div class="synapse-detail" style="border-color: {selectedConceptData.color}">
      <div class="synapse-detail-header">
        <h4 class="synapse-detail-title" style="color: {selectedConceptData.color}">
          {selectedConceptData.label}
        </h4>
        <button class="synapse-detail-close" on:click={() => { selectedConcept = null; selectedConceptData = null; }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <p class="synapse-detail-desc">
        Neuron pairs that co-activate for <strong>{selectedConceptData.concept}</strong>-related patterns.
        Live σ values update as you type.
      </p>
      <div class="synapse-pairs-grid">
        {#each selectedConceptData.pairs as pair, idx}
          <div class="synapse-pair" class:synapse-pair-active={Math.abs(pair.sigmaVal) > 0.001}>
            <span class="pair-index">#{idx + 1}</span>
            <span class="pair-neurons">σ({pair.i}, {pair.j})</span>
            <span class="pair-value" class:pair-value-positive={pair.sigmaVal > 0.001} class:pair-value-negative={pair.sigmaVal < -0.001}>
              {pair.sigmaVal.toFixed(2)}
            </span>
            <div class="pair-bar-track">
              <div
                class="pair-bar-fill"
                style="width: {Math.min(100, Math.abs(pair.sigmaVal) / (Math.abs(pair.strength) + 1) * 100)}%; background: {selectedConceptData.color}"
              ></div>
            </div>
          </div>
        {/each}
      </div>
      <p class="synapse-detail-note">
        {selectedConceptData.pairs.filter(p => Math.abs(p.sigmaVal) > 0.001).length} of {selectedConceptData.pairs.length} pairs currently active
      </p>

      <!-- Synapse Activation Timeline -->
      {#if synapseHistory[selectedConcept] && synapseHistory[selectedConcept].length > 1}
        {@const hist = synapseHistory[selectedConcept]}
        {@const maxH = Math.max(...hist.map(h => h.avgSigma), 0.001)}
        {@const sparkW = 220}
        {@const sparkH = 36}
        <div class="synapse-timeline">
          <span class="timeline-label">σ Timeline</span>
          <svg width={sparkW} height={sparkH} class="timeline-svg">
            <polyline
              fill="none"
              stroke={selectedConceptData.color}
              stroke-width="1.5"
              stroke-linejoin="round"
              points={hist.map((h, i) => `${(i / (hist.length - 1)) * (sparkW - 4) + 2},${sparkH - 2 - (h.avgSigma / maxH) * (sparkH - 4)}`).join(' ')}
            />
            <circle
              cx={sparkW - 2}
              cy={sparkH - 2 - (hist[hist.length - 1].avgSigma / maxH) * (sparkH - 4)}
              r="2.5"
              fill={selectedConceptData.color}
            />
          </svg>
          <span class="timeline-val">{hist[hist.length - 1].avgSigma.toFixed(3)}</span>
        </div>
      {/if}
    </div>
  {/if}

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
    Colored outlines mark <strong>discovered synapse concepts</strong> — semantically meaningful
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

  .title-icon { flex-shrink: 0; opacity: 0.8; }

  .panel-desc {
    font-size: 0.87rem;
    color: var(--text-muted);
    margin: 0.25rem 0 0;
    line-height: 1.65;
    max-width: 45ch;
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
    box-shadow: 0 2px 8px rgba(240, 98, 146, 0.3);
  }

  .clear-icon { font-size: 0.65rem; }

  .heatmap-container {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
  }

  .heatmap-svg {
    border-radius: var(--radius-md);
    background: rgba(0, 0, 0, 0.2);
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
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--text-secondary);
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
    cursor: pointer;
    background: transparent;
    transition: all 0.15s ease;
  }

  .synapse-tag:hover {
    opacity: 1;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(255, 255, 255, 0.06);
  }

  .synapse-tag-active {
    opacity: 1;
    background: rgba(255, 255, 255, 0.06);
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.08);
  }

  .synapse-detail {
    margin-top: 0.6rem;
    padding: 0.7rem 0.9rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid;
    border-radius: var(--radius-md);
    animation: slideDown 0.2s ease;
  }

  @keyframes slideDown {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .synapse-detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
  }

  .synapse-detail-title {
    font-family: var(--font-sans);
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0;
  }

  .synapse-detail-close {
    background: none;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-dim);
    cursor: pointer;
    padding: 0.2rem;
    display: flex;
    align-items: center;
    transition: all 0.1s ease;
  }

  .synapse-detail-close:hover {
    color: var(--text-primary);
    border-color: var(--border-hover);
  }

  .synapse-detail-desc {
    font-size: 0.74rem;
    color: var(--text-muted);
    margin: 0 0 0.5rem;
    line-height: 1.5;
  }

  .synapse-detail-desc strong {
    color: var(--text-secondary);
  }

  .synapse-pairs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.3rem;
    max-height: 180px;
    overflow-y: auto;
  }

  .synapse-pair {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.4rem;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    transition: background 0.1s ease;
  }

  .synapse-pair-active {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-secondary);
  }

  .pair-index {
    font-size: 0.68rem;
    color: var(--text-muted);
    min-width: 1.5em;
  }

  .pair-neurons {
    white-space: nowrap;
  }

  .pair-value {
    margin-left: auto;
    font-weight: 600;
    min-width: 3.5em;
    text-align: right;
  }

  .pair-value-positive { color: var(--green); }
  .pair-value-negative { color: var(--rose); }

  .pair-bar-track {
    width: 30px;
    height: 3px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 2px;
    overflow: hidden;
    flex-shrink: 0;
  }

  .pair-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.15s ease;
  }

  .synapse-detail-note {
    font-size: 0.68rem;
    color: var(--text-dim);
    margin: 0.4rem 0 0;
    font-family: var(--font-mono);
  }

  .synapse-timeline {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    padding: 0.35rem 0.5rem;
    background: rgba(0, 0, 0, 0.25);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
  }

  .timeline-label {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }

  .timeline-svg {
    flex-shrink: 0;
  }

  .timeline-val {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    font-weight: 600;
    min-width: 3em;
    text-align: right;
  }

  .footnote {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0.6rem 0 0;
    line-height: 1.65;
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
    background: rgba(240, 194, 70, 0.04);
    border: 1px solid rgba(240, 194, 70, 0.15);
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
