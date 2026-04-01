<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import graphData from '../data/graph_topology.json';

  export let activeNeuronIds = new Set();

  let container;
  let simulation = null;
  let graphMode = 'gx';

  function cloneGraph(g) {
    return {
      nodes: g.nodes.map(n => ({ ...n })),
      links: g.links.map(l => ({ ...l })),
      threshold: g.threshold,
    };
  }

  let currentData = cloneGraph(graphData.gx);

  function switchMode(mode) {
    if (mode === graphMode) return;
    graphMode = mode;
    currentData = cloneGraph(graphData[mode]);
    rebuildGraph();
  }

  const W = 580, H = 420;
  const sizeScale = d3.scaleSqrt().domain([1, 60]).range([2.5, 16]);

  function rebuildGraph() {
    if (!container) return;
    const svg = d3.select(container);
    svg.selectAll('*').remove();
    buildGraph(svg, currentData);
  }

  function buildGraph(svg, data) {
    svg.attr('viewBox', `0 0 ${W} ${H}`);

    // Gradient defs
    const defs = svg.append('defs');

    const grad = defs.append('radialGradient').attr('id', 'node-glow');
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.6);
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0);

    const g = svg.append('g');

    svg.call(
      d3.zoom()
        .scaleExtent([0.3, 5])
        .on('zoom', e => g.attr('transform', e.transform))
    );

    // Links
    const link = g.selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('stroke', d => d.excitatory ? 'rgba(59,130,246,0.4)' : 'rgba(244,63,94,0.3)')
      .attr('stroke-width', d => Math.max(0.3, Math.abs(d.weight) * 3));

    // Node glow
    g.selectAll('circle.glow')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('class', 'glow')
      .attr('r', d => sizeScale(d.degree) * 2.5)
      .attr('fill', 'url(#node-glow)')
      .attr('opacity', d => Math.min(0.4, d.degree / 80));

    // Nodes
    const node = g.selectAll('circle.node')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('class', 'node')
      .attr('r', d => sizeScale(d.degree))
      .attr('fill', '#3b82f6')
      .attr('stroke', 'rgba(255,255,255,0.15)')
      .attr('stroke-width', 0.6)
      .attr('cursor', 'grab')
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active && simulation) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => {
            if (!event.active && simulation) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    node.append('title').text(d => `Neuron #${d.id} · degree ${d.degree}`);

    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links)
        .id((_, i) => i)
        .distance(35)
        .strength(0.35))
      .force('charge', d3.forceManyBody().strength(-45))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(d => sizeScale(d.degree) + 3))
      .on('tick', () => {
        link
          .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        g.selectAll('circle.glow')
          .attr('cx', d => d.x).attr('cy', d => d.y);
        node
          .attr('cx', d => d.x).attr('cy', d => d.y);
      });
  }

  onMount(() => {
    if (container) buildGraph(d3.select(container), currentData);
  });

  $: if (container && currentData) {
    d3.select(container).selectAll('circle.node')
      .attr('fill', d => {
        if (!d) return '#3b82f6';
        return activeNeuronIds.has(d.id) ? '#facc15' : '#3b82f6';
      })
      .attr('stroke', d => {
        if (!d) return 'rgba(255,255,255,0.15)';
        return activeNeuronIds.has(d.id) ? 'rgba(250,204,21,0.5)' : 'rgba(255,255,255,0.15)';
      });
  }

  const modeInfo = {
    gx: {
      name: 'Thought Flow',
      formula: 'Gx = Encoder × Decoder_x',
      desc: 'Feedforward causal circuit — how neurons propagate computation through x = ReLU(v* @ Dx).',
    },
    gy: {
      name: 'Memory Echo',
      formula: 'Gy = Decoder_y^T × Decoder_x',
      desc: 'Memory readout graph — how Hebbian memory σ is decoded into y activations.',
    },
  };
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h3 class="panel-title">
        <svg class="title-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="19" r="2"/><line x1="7" y1="7" x2="11" y2="17"/><line x1="17" y1="7" x2="13" y2="17"/><line x1="7" y1="6" x2="17" y2="6"/></svg>
        Emergent Graph
      </h3>
      <p class="panel-desc">{modeInfo[graphMode].desc}</p>
    </div>
  </div>

  <div class="mode-bar">
    <div class="mode-btns">
      <button
        class="mode-btn"
        class:active={graphMode === 'gx'}
        on:click={() => switchMode('gx')}
      >
        Thought Flow (Gx)
      </button>
      <button
        class="mode-btn"
        class:active={graphMode === 'gy'}
        on:click={() => switchMode('gy')}
      >
        Memory Echo (Gy)
      </button>
    </div>
    <span class="node-count">{currentData.nodes.length} hubs · {currentData.links.length} edges</span>
  </div>

  <div class="graph-container">
    <svg bind:this={container} class="graph-svg" role="img" aria-label="Emergent graph topology"></svg>

    <div class="legend">
      <div class="legend-item">
        <span class="legend-circle" style="background: #3b82f6;"></span>
        <span>Excitatory</span>
      </div>
      <div class="legend-item">
        <span class="legend-circle" style="background: var(--rose);"></span>
        <span>Inhibitory</span>
      </div>
      <div class="legend-item">
        <span class="legend-circle" style="background: var(--gold);"></span>
        <span>Active now</span>
      </div>
    </div>
  </div>

  <p class="footnote">
    This structure <strong>self-organized from random weights</strong> during training.
    Hub neurons act as real organizational centres — the graph is scale-free (Section 5 of the paper).
    <span class="formula">{modeInfo[graphMode].formula}</span>
  </p>
</div>

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
    margin-bottom: 0.6rem;
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
  }

  .mode-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .mode-btns {
    display: flex;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 2px;
    gap: 2px;
  }

  .mode-btn {
    padding: 0.3rem 0.7rem;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .mode-btn:hover {
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.04);
  }

  .mode-btn.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
  }

  .node-count {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
  }

  .graph-container {
    position: relative;
  }

  .graph-svg {
    width: 100%;
    height: 400px;
    border-radius: var(--radius-md);
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid var(--border-default);
    display: block;
  }

  .legend {
    position: absolute;
    bottom: 10px;
    right: 10px;
    display: flex;
    gap: 0.6rem;
    padding: 0.3rem 0.6rem;
    background: rgba(0, 0, 0, 0.8);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.68rem;
    color: var(--text-dim);
  }

  .legend-circle {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
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

  .formula {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    display: inline-block;
    margin-left: 0.3rem;
    padding: 0.1rem 0.4rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
  }

  @media (max-width: 768px) {
    .graph-svg {
      height: 320px;
    }
  }
</style>
