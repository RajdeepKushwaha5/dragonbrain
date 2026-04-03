<script>
  import * as d3 from 'd3';
  import { onMount, onDestroy } from 'svelte';
  import graphData from '../data/graph_topology.json';
  import evolutionData from '../data/graph_evolution.json';

  export let activeNeuronIds = new Set();

  let container;
  let simulation = null;
  let graphMode = 'gx';
  let communityMap = {};  // nodeId → communityId
  let showCommunities = true;

  // Community color palette
  const COMMUNITY_PALETTE = [
    '#5b8def', '#f0c246', '#3dd68c', '#f06292',
    '#9b7ef0', '#e879a8', '#4dd0e1', '#f58b42',
    '#2ec4b6', '#b07ef0', '#d4a832', '#7b8394',
  ];

  function detectCommunities(nodes, links) {
    const comm = {};
    nodes.forEach(n => comm[n.id] = n.id);
    const adj = {};
    nodes.forEach(n => adj[n.id] = []);
    links.forEach(l => {
      // Raw data uses array indices for source/target; D3-mutated data uses node objects
      const s = typeof l.source === 'object' ? l.source.id : (nodes[l.source] ? nodes[l.source].id : l.source);
      const t = typeof l.target === 'object' ? l.target.id : (nodes[l.target] ? nodes[l.target].id : l.target);
      if (s === t) return;
      if (!adj[s] || !adj[t]) return;
      adj[s].push({ id: t, weight: Math.abs(l.weight) });
      adj[t].push({ id: s, weight: Math.abs(l.weight) });
    });
    for (let iter = 0; iter < 30; iter++) {
      let changed = false;
      const order = [...nodes];
      // Deterministic shuffle via index-based swap
      for (let i = order.length - 1; i > 0; i--) {
        const j = ((i * 2654435761) >>> 0) % (i + 1);
        [order[i], order[j]] = [order[j], order[i]];
      }
      for (const node of order) {
        const neighbors = adj[node.id];
        if (!neighbors || neighbors.length === 0) continue;
        const votes = {};
        for (const nb of neighbors) {
          const c = comm[nb.id];
          votes[c] = (votes[c] || 0) + nb.weight;
        }
        let bestC = comm[node.id];
        let bestW = -Infinity;
        for (const c of Object.keys(votes)) {
          if (votes[c] > bestW) { bestW = votes[c]; bestC = parseInt(c); }
        }
        if (bestC !== comm[node.id]) { comm[node.id] = bestC; changed = true; }
      }
      if (!changed) break;
    }
    return comm;
  }

  function getCommunityColor(nodeId) {
    if (!showCommunities || communityMap[nodeId] === undefined) return '#5b8def';
    const ids = [...new Set(Object.values(communityMap))].sort((a,b) => a - b);
    const idx = ids.indexOf(communityMap[nodeId]);
    return COMMUNITY_PALETTE[idx % COMMUNITY_PALETTE.length];
  }

  $: communityCount = [...new Set(Object.values(communityMap))].length;
  $: communityLegend = showCommunities && communityCount > 1
    ? [...new Set(Object.values(communityMap))].sort((a,b) => a - b).map((id, i) => ({
        color: COMMUNITY_PALETTE[i % COMMUNITY_PALETTE.length],
        label: `C${i + 1}`,
      }))
    : [];

  // Evolution toggle
  let showEvolution = false;
  let snapshotIdx = evolutionData.snapshots.length - 1; // default to trained
  $: snapshot = evolutionData.snapshots[snapshotIdx];
  $: evoStats = snapshot ? snapshot.stats : null;

  function cloneGraph(g) {
    return {
      nodes: g.nodes.map(n => ({ ...n })),
      links: g.links.filter(l => l.source !== l.target).map(l => ({ ...l })),
      threshold: g.threshold,
    };
  }

  let currentData = cloneGraph(graphData.gx);

  function getActiveSource() {
    return showEvolution ? evolutionData.snapshots[snapshotIdx] : graphData;
  }

  function switchMode(mode) {
    if (mode === graphMode) return;
    graphMode = mode;
    const src = getActiveSource();
    currentData = cloneGraph(src[mode]);
    rebuildGraph();
  }

  function toggleEvolution() {
    showEvolution = !showEvolution;
    if (showEvolution) {
      snapshotIdx = evolutionData.snapshots.length - 1;
    }
    const src = getActiveSource();
    currentData = cloneGraph(src[graphMode]);
    rebuildGraph();
  }

  function switchSnapshot(idx) {
    snapshotIdx = idx;
    const src = evolutionData.snapshots[idx];
    currentData = cloneGraph(src[graphMode]);
    rebuildGraph();
  }

  const W = 580, H = 420;
  const sizeScale = d3.scaleSqrt().domain([1, 60]).range([2.5, 16]);

  function rebuildGraph() {
    if (!container) return;
    communityMap = detectCommunities(currentData.nodes, currentData.links);
    const svg = d3.select(container);
    svg.selectAll('*').remove();
    buildGraph(svg, currentData);
  }

  function buildGraph(svg, data) {
    svg.attr('viewBox', `0 0 ${W} ${H}`);

    // Gradient defs
    const defs = svg.append('defs');

    // Create glow gradient per community color
    const usedColors = new Set(data.nodes.map(n => getCommunityColor(n.id)));
    usedColors.add('#f0c246'); // active gold
    for (const color of usedColors) {
      const gid = `node-glow-${color.replace('#', '')}`;
      const grad = defs.append('radialGradient').attr('id', gid);
      grad.append('stop').attr('offset', '0%').attr('stop-color', color).attr('stop-opacity', 0.6);
      grad.append('stop').attr('offset', '100%').attr('stop-color', color).attr('stop-opacity', 0);
    }

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
      .attr('stroke', d => d.excitatory ? 'rgba(91,141,239,0.4)' : 'rgba(240,98,146,0.3)')
      .attr('stroke-width', d => Math.max(0.3, Math.abs(d.weight) * 3));

    // Node glow
    g.selectAll('circle.glow')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('class', 'glow')
      .attr('r', d => sizeScale(d.degree) * 2.5)
      .attr('fill', d => {
        const c = getCommunityColor(d.id);
        return `url(#node-glow-${c.replace('#', '')})`;
      })
      .attr('opacity', d => Math.min(0.4, d.degree / 80));

    // Nodes
    const node = g.selectAll('circle.node')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('class', 'node')
      .attr('r', d => sizeScale(d.degree))
      .attr('fill', d => getCommunityColor(d.id))
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
    if (container) {
      communityMap = detectCommunities(currentData.nodes, currentData.links);
      buildGraph(d3.select(container), currentData);
    }
  });

  onDestroy(() => {
    if (simulation) simulation.stop();
  });

  $: if (container && currentData) {
    d3.select(container).selectAll('circle.node')
      .attr('fill', d => {
        if (!d) return '#5b8def';
        return activeNeuronIds.has(d.id) ? '#f0c246' : getCommunityColor(d.id);
      })
      .attr('stroke', d => {
        if (!d) return 'rgba(255,255,255,0.15)';
        return activeNeuronIds.has(d.id) ? 'rgba(240,194,70,0.5)' : 'rgba(255,255,255,0.15)';
      });
    d3.select(container).selectAll('circle.glow')
      .attr('fill', d => {
        if (!d) return 'url(#node-glow-5b8def)';
        const c = activeNeuronIds.has(d.id) ? '#f0c246' : getCommunityColor(d.id);
        return `url(#node-glow-${c.replace('#', '')})`;
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
    <div class="mode-right">
      <button class="evo-btn" class:active={showCommunities} on:click={() => { showCommunities = !showCommunities; rebuildGraph(); }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="8" r="2"/><circle cx="20" cy="8" r="2"/><circle cx="4" cy="16" r="2"/><circle cx="20" cy="16" r="2"/></svg>
        Communities{#if showCommunities && communityCount > 1}&nbsp;({communityCount}){/if}
      </button>
      <button class="evo-btn" class:active={showEvolution} on:click={toggleEvolution}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        Evolution
      </button>
      <span class="node-count">{currentData.nodes.length} hubs · {currentData.links.length} edges</span>
    </div>
  </div>

  {#if showEvolution}
    <div class="evo-bar">
      {#each evolutionData.snapshots as snap, idx}
        <button
          class="snap-btn"
          class:active={snapshotIdx === idx}
          on:click={() => switchSnapshot(idx)}
        >
          {snap.label}
        </button>
      {/each}
      {#if evoStats}
        <span class="evo-stats">
          max°{evoStats.max_degree} · avg°{evoStats.avg_degree} · {evoStats.gx_edges} edges
        </span>
      {/if}
    </div>
  {/if}

  <div class="graph-container">
    <svg bind:this={container} class="graph-svg" role="img" aria-label="Emergent graph topology"></svg>

    <div class="legend">
      {#if communityLegend.length > 1}
        {#each communityLegend as c}
          <div class="legend-item">
            <span class="legend-circle" style="background: {c.color};"></span>
            <span>{c.label}</span>
          </div>
        {/each}
      {:else}
        <div class="legend-item">
          <span class="legend-circle" style="background: #5b8def;"></span>
          <span>Excitatory</span>
        </div>
        <div class="legend-item">
          <span class="legend-circle" style="background: var(--rose);"></span>
          <span>Inhibitory</span>
        </div>
      {/if}
      <div class="legend-item">
        <span class="legend-circle" style="background: var(--gold);"></span>
        <span>Active now</span>
      </div>
    </div>
  </div>

  <p class="footnote">
    {#if showEvolution && snapshotIdx === 0}
      <strong>Random initialization</strong> — weights are untrained noise. No hub structure has formed yet.
      Toggle to "Trained" to see the scale-free graph emerge.
    {:else}
      This structure <strong>self-organized from random weights</strong> during training.
      {#if showCommunities && communityCount > 1}
        <strong>{communityCount} communities</strong> detected via label propagation — neurons cluster into functional modules.
      {:else}
        Hub neurons act as real organizational centers — the graph is scale-free (Section 5 of the paper).
      {/if}
      {#if !showEvolution}Click <em>Evolution</em> to compare with random init.{/if}
    {/if}
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

  .mode-right {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .evo-btn {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.55rem;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 0.72rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .evo-btn:hover { color: var(--text-secondary); border-color: var(--border-hover); }
  .evo-btn.active { background: rgba(240, 194, 70, 0.1); border-color: var(--gold); color: var(--gold); }

  .evo-bar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
  }

  .snap-btn {
    padding: 0.2rem 0.6rem;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 0.72rem;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .snap-btn:hover { border-color: var(--border-hover); color: var(--text-secondary); }
  .snap-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }

  .evo-stats {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-left: auto;
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
    box-shadow: 0 2px 8px rgba(91, 141, 239, 0.3);
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
    background: rgba(0, 0, 0, 0.2);
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
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0.6rem 0 0;
    line-height: 1.65;
  }

  .footnote strong {
    color: var(--gold);
  }

  .formula {
    font-family: var(--font-mono);
    font-size: 0.72rem;
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
