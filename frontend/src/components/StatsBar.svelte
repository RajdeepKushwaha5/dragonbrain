<script>
  export let active = 0;
  export let total = 1024;
  export let pct = '0.0';
  export let label = '';
  export let color = 'var(--accent)';

  $: barWidth = Math.min(100, parseFloat(pct));
</script>

<div class="stats-bar">
  <div class="bar-track">
    <div class="bar-fill" style="width: {barWidth}%; background: {color};"></div>
  </div>
  <div class="stats-text">
    <span class="stat-main" style="color: {color};">{pct}%</span>
    <span class="stat-detail">{active} / {total}</span>
    {#if label}
      <span class="stat-badge">{label}</span>
    {/if}
  </div>
</div>

<style>
  .stats-bar {
    width: 100%;
    padding: 0.3rem 0;
  }

  .bar-track {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: var(--radius-full);
    overflow: hidden;
    margin-bottom: 0.3rem;
  }

  .bar-fill {
    height: 100%;
    border-radius: var(--radius-full);
    transition: width 0.4s ease;
  }

  .stats-text {
    display: flex;
    align-items: baseline;
    gap: 0.35rem;
    font-size: 0.72rem;
  }

  .stat-main {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 700;
  }

  .stat-detail {
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 0.68rem;
  }

  .stat-badge {
    margin-left: auto;
    padding: 0.1rem 0.5rem;
    background: rgba(255, 255, 255, 0.04);
    border-radius: var(--radius-full);
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 500;
  }
</style>
