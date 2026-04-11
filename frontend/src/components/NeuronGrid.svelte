<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  export let activations = new Float32Array(1024);
  export let label = 'BDH';
  export let accentColor = '#4a6cf7';
  export let cols = 32;
  export let rows = 32;

  let svgEl;
  let initialized = false;

  const cellSize = 10;
  const gap = 1;
  const W = cols * (cellSize + gap);
  const H = rows * (cellSize + gap);

  function getColorScale(acts) {
    let maxVal = 0.5;
    for (let i = 0; i < acts.length; i++) {
      const a = Math.abs(acts[i]);
      if (a > maxVal) maxVal = a;
    }
    return d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxVal]);
  }

  function renderGrid() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    const colorScale = getColorScale(activations);

    if (!initialized) {
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('preserveAspectRatio', 'xMidYMid meet');
      svg.selectAll('rect')
        .data(Array.from(activations))
        .enter()
        .append('rect')
        .attr('x', (_, i) => (i % cols) * (cellSize + gap))
        .attr('y', (_, i) => Math.floor(i / cols) * (cellSize + gap))
        .attr('width', cellSize)
        .attr('height', cellSize)
        .attr('rx', 1.5);
      initialized = true;
    }

    svg.selectAll('rect')
      .data(Array.from(activations))
      .attr('fill', v => (Math.abs(v) > 1e-6 ? colorScale(Math.abs(v)) : 'rgba(255,255,255,0.03)'))
      .attr('opacity', v => (Math.abs(v) > 1e-6 ? 1 : 0.4));
  }

  onMount(renderGrid);

  $: if (svgEl && activations) {
    renderGrid();
  }

  let tooltipText = '';
  let tooltipX = 0;
  let tooltipY = 0;
  let showTooltip = false;

  function handleMouseMove(e) {
    const rect = svgEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / (cellSize + gap));
    const row = Math.floor(y / (cellSize + gap));
    const idx = row * cols + col;

    if (idx >= 0 && idx < activations.length) {
      const val = activations[idx];
      tooltipText = `#${idx} — ${Math.abs(val) > 1e-6 ? val.toFixed(4) : 'inactive'}`;
      tooltipX = e.clientX;
      tooltipY = e.clientY;
      showTooltip = true;
    } else {
      showTooltip = false;
    }
  }

  function handleMouseLeave() {
    showTooltip = false;
  }
</script>

<div class="grid-wrap">
  <div class="grid-label" style="color: {accentColor};">{label}</div>
  <div class="grid-container">
    <svg
      bind:this={svgEl}
      on:mousemove={handleMouseMove}
      on:mouseleave={handleMouseLeave}
      class="neuron-grid"
      role="img"
      aria-label="{label} neuron activation grid showing {cols * rows} neurons"
    ></svg>
  </div>
</div>

{#if showTooltip}
  <div class="tooltip" style="left: {tooltipX + 14}px; top: {tooltipY - 10}px;">
    {tooltipText}
  </div>
{/if}

<style>
  .grid-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
  }

  .grid-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .grid-container {
    border-radius: var(--radius-sm);
    background: rgba(0, 0, 0, 0.3);
    padding: 4px;
    border: 1px solid var(--border-default);
    width: 100%;
    max-width: 352px;
  }

  .neuron-grid {
    display: block;
    border-radius: 4px;
    width: 100%;
    height: auto;
  }

  .tooltip {
    position: fixed;
    background: var(--bg-tooltip);
    color: var(--text-primary);
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.68rem;
    font-family: var(--font-mono);
    pointer-events: none;
    z-index: 100;
    border: 1px solid var(--border-default);
    white-space: nowrap;
    box-shadow: var(--shadow-elevated);
  }
</style>
