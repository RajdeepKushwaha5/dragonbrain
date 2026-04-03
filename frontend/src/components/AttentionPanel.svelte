<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { NH } from '../lib/activation_math.js';
  import { attentionScores, selectedHead, tokenBuffer } from '../lib/stores.js';
  import { tokenToChar } from '../lib/tokenizer.js';

  let svgEl;
  const maxT = 32; // Show last 32 tokens
  const cellSize = 10;
  const gap = 1;
  const marginLeft = 28;
  const marginTop = 28;

  const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

  function renderHeatmap(scores, T, head, tokens) {
    if (!svgEl || !scores) return;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const dispT = Math.min(T, maxT);
    const startIdx = Math.max(0, T - maxT);
    const W = marginLeft + dispT * (cellSize + gap);
    const H = marginTop + dispT * (cellSize + gap);
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', null).attr('height', null);

    // Extract TxT scores for selected head
    const matrix = [];
    for (let row = 0; row < dispT; row++) {
      for (let col = 0; col < dispT; col++) {
        const r = row + startIdx;
        const c = col + startIdx;
        const flatIdx = head * T * T + r * T + c;
        matrix.push({
          row, col,
          value: r < c ? 0 : (scores[flatIdx] || 0),  // lower-tri only (causal mask)
        });
      }
    }

    // Normalize
    const maxVal = Math.max(...matrix.map(d => Math.abs(d.value)), 0.001);

    // Draw cells
    svg.selectAll('rect.cell')
      .data(matrix)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => marginLeft + d.col * (cellSize + gap))
      .attr('y', d => marginTop + d.row * (cellSize + gap))
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('fill', d => d.value > 0.001 ? colorScale(Math.abs(d.value) / maxVal) : '#050505');

    // Column labels (tokens)
    const dispTokens = tokens.slice(-maxT);
    svg.selectAll('text.col-label')
      .data(dispTokens)
      .enter()
      .append('text')
      .attr('class', 'col-label')
      .attr('x', (_, i) => marginLeft + i * (cellSize + gap) + cellSize / 2)
      .attr('y', marginTop - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#aaa')
      .attr('font-size', '8px')
      .text(b => tokenToChar(b));

    // Row labels
    svg.selectAll('text.row-label')
      .data(dispTokens)
      .enter()
      .append('text')
      .attr('class', 'row-label')
      .attr('x', marginLeft - 5)
      .attr('y', (_, i) => marginTop + i * (cellSize + gap) + cellSize / 2 + 3)
      .attr('text-anchor', 'end')
      .attr('fill', '#aaa')
      .attr('font-size', '8px')
      .text(b => tokenToChar(b));
  }

  $: if ($attentionScores && $tokenBuffer.length > 0) {
    renderHeatmap(
      $attentionScores.flat,
      $attentionScores.T,
      $attentionScores.head,
      $tokenBuffer
    );
  } else if (!$attentionScores && svgEl) {
    d3.select(svgEl).selectAll('*').remove();
  }

  onMount(() => {
    if ($attentionScores) {
      renderHeatmap(
        $attentionScores.flat,
        $attentionScores.T,
        $attentionScores.head,
        $tokenBuffer
      );
    }
  });
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h3 class="panel-title">
        <svg class="title-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="12 2 22 8.5 12 15 2 8.5"/><polyline points="2 15.5 12 22 22 15.5"/></svg>
        Attention Pattern
      </h3>
      <p class="panel-desc">
        Linear causal attention with RoPE — no softmax needed.
        Q = K = x, so the heatmap shows raw pairwise correlations between token positions.
      </p>
    </div>
    <div class="head-btns">
      {#each Array(NH) as _, h}
        <button
          class="head-btn"
          class:active={$selectedHead === h}
          on:click={() => selectedHead.set(h)}
          aria-label="Select head {h + 1}"
        >
          H{h + 1}
        </button>
      {/each}
    </div>
  </div>

  <div class="heatmap-container">
    <svg
      bind:this={svgEl}
      class="attn-svg"
      role="img"
      aria-label="Attention pattern heatmap"
    ></svg>
  </div>

  <p class="footnote">
    Since <strong>Q = K = x</strong>, attention becomes a simple inner product rotated by RoPE.
    No key/value projection — the recurrent form runs in <em>O(T)</em> via cumulative sum; the parallel form shown here is O(T²) (Section 4.1 of the paper).
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
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.6rem;
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

  .head-btns {
    display: flex;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 2px;
    gap: 2px;
  }

  .head-btn {
    padding: 0.3rem 0.6rem;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.76rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .head-btn:hover {
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.04);
  }

  .head-btn.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(91, 141, 239, 0.3);
  }

  .heatmap-container {
    overflow-x: auto;
  }

  .attn-svg {
    border-radius: var(--radius-md);
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-default);
    display: block;
    width: 100%;
    max-width: 500px;
    height: auto;
  }

  .footnote {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0.6rem 0 0;
    line-height: 1.65;
  }

  .footnote strong {
    color: var(--cyan);
  }

  .footnote em {
    color: var(--text-muted);
    font-style: normal;
  }

  @media (max-width: 600px) {
    .panel-header {
      flex-direction: column;
    }

    .panel-desc {
      max-width: none;
    }
  }
</style>
