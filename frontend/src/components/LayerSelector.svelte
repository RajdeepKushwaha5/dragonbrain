<script>
  import { selectedLayer, selectedHead, sparsityStats, tokenCount } from '../lib/stores.js';
  import { NH } from '../lib/activation_math.js';

  const layers = [0, 1];
  const heads = Array.from({ length: NH }, (_, i) => i);
</script>

<div class="selector-bar">
  <div class="selector-group">
    <span class="group-label">Layer</span>
    <div class="btn-group">
      {#each layers as layer}
        <button
          class="sel-btn"
          class:active={$selectedLayer === layer}
          on:click={() => selectedLayer.set(layer)}
          aria-pressed={$selectedLayer === layer}
        >
          L{layer + 1}
        </button>
      {/each}
    </div>
  </div>

  <div class="selector-group">
    <span class="group-label">Head</span>
    <div class="btn-group">
      {#each heads as head}
        <button
          class="sel-btn"
          class:active={$selectedHead === head}
          on:click={() => selectedHead.set(head)}
          aria-pressed={$selectedHead === head}
        >
          H{head + 1}
        </button>
      {/each}
    </div>
  </div>

  <div class="stats-row">
    {#if $sparsityStats.active > 0}
      <div class="stat">
        <span class="stat-value">{$sparsityStats.pct}%</span>
        <span class="stat-label">sparsity</span>
      </div>
      <div class="stat-sep"></div>
      <div class="stat">
        <span class="stat-value">{$sparsityStats.active}</span>
        <span class="stat-label">/ {$sparsityStats.total} active</span>
      </div>
    {/if}
    {#if $tokenCount > 0}
      <div class="stat-sep"></div>
      <div class="stat">
        <span class="stat-value">{$tokenCount}</span>
        <span class="stat-label">tokens</span>
      </div>
    {/if}
  </div>
</div>

<style>
  .selector-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .selector-group {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .group-label {
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .btn-group {
    display: flex;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 2px;
    gap: 2px;
  }

  .sel-btn {
    padding: 0.3rem 0.7rem;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .sel-btn:hover {
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.04);
  }

  .sel-btn.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
  }

  .stats-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-left: auto;
  }

  .stat {
    display: flex;
    align-items: baseline;
    gap: 0.25rem;
  }

  .stat-value {
    font-family: var(--font-mono);
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--gold);
  }

  .stat-label {
    font-size: 0.72rem;
    color: var(--text-dim);
  }

  .stat-sep {
    width: 1px;
    height: 14px;
    background: var(--border-subtle);
  }

  @media (max-width: 600px) {
    .stats-row {
      margin-left: 0;
      width: 100%;
    }
  }
</style>
