<script>
  import { tokenCount } from '../lib/stores.js';

  // BDH σ memory: 4 keys × 512² × 4 bytes = 4,194,304 bytes ≈ 4MB (constant)
  const SIGMA_BYTES = 4 * 512 * 512 * 4;
  // GPT KV per token: 2(K+V) × 2 layers × 2 heads × 32 head_dim × 4 bytes = 1024 bytes
  const KV_PER_TOKEN = 2 * 2 * 2 * 32 * 4;
  // Crossover point
  const CROSSOVER = SIGMA_BYTES / KV_PER_TOKEN; // 4096 tokens

  // Chart dimensions
  const W = 340, H = 150;
  const pad = { top: 18, right: 20, bottom: 28, left: 50 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Project to 6000 tokens to show crossover
  const maxT = 6000;
  const maxBytes = Math.max(SIGMA_BYTES * 1.4, KV_PER_TOKEN * maxT);

  function formatBytes(b) {
    if (b >= 1048576) return (b / 1048576).toFixed(1) + 'MB';
    if (b >= 1024) return Math.round(b / 1024) + 'KB';
    return b + 'B';
  }

  function xScale(t) { return pad.left + (t / maxT) * plotW; }
  function yScale(b) { return pad.top + plotH - (b / maxBytes) * plotH; }

  $: currentKV = $tokenCount * KV_PER_TOKEN;
  $: sigmaLabel = formatBytes(SIGMA_BYTES);
  $: kvLabel = formatBytes(currentKV);

  // σ line (horizontal)
  $: sigmaY = yScale(SIGMA_BYTES);

  // KV line points
  $: kvLine = `${xScale(0)},${yScale(0)} ${xScale(maxT)},${yScale(KV_PER_TOKEN * maxT)}`;

  // Current position marker
  $: cursorX = xScale(Math.min($tokenCount, maxT));
  $: cursorKvY = yScale(Math.min(currentKV, maxBytes));

  // Crossover marker
  $: crossoverX = xScale(CROSSOVER);
  $: crossoverY = sigmaY;

  // X-axis ticks
  const xTicks = [0, 2000, 4000, 6000];
  // Y-axis ticks (in MB)
  const yTicks = [0, 2, 4, 6];
</script>

<div class="panel">
  <div class="panel-header">
    <h3 class="panel-title">
      <svg class="title-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="6" width="20" height="12" rx="2"/><line x1="6" y1="10" x2="6" y2="14"/><line x1="10" y1="10" x2="10" y2="14"/></svg>
      Memory Scaling
    </h3>
    <p class="panel-desc">
      BDH σ stays <strong>fixed at {sigmaLabel}</strong> regardless of sequence length.
      A transformer's KV-cache grows <strong>linearly</strong> — already {kvLabel} at {$tokenCount} tokens.
    </p>
  </div>

  <svg viewBox="0 0 {W} {H}" class="chart-svg" role="img" aria-label="Memory scaling comparison chart">
    <!-- Grid lines -->
    {#each yTicks as mb}
      <line
        x1={pad.left} y1={yScale(mb * 1048576)}
        x2={W - pad.right} y2={yScale(mb * 1048576)}
        stroke="rgba(255,255,255,0.04)"
        stroke-width="0.5"
      />
      <text x={pad.left - 6} y={yScale(mb * 1048576) + 3} class="axis-label" text-anchor="end">{mb}MB</text>
    {/each}
    {#each xTicks as t}
      <text x={xScale(t)} y={H - 6} class="axis-label" text-anchor="middle">
        {t >= 1000 ? (t / 1000) + 'K' : '0'}
      </text>
    {/each}

    <!-- Crossover vertical dashed line -->
    <line
      x1={crossoverX} y1={pad.top}
      x2={crossoverX} y2={pad.top + plotH}
      stroke="rgba(250,204,21,0.25)"
      stroke-width="1"
      stroke-dasharray="3,3"
    />
    <text x={crossoverX} y={pad.top - 4} class="crossover-label" text-anchor="middle">T≈4K ×</text>

    <!-- σ line (constant blue) -->
    <line
      x1={pad.left} y1={sigmaY}
      x2={W - pad.right} y2={sigmaY}
      stroke="#3b82f6" stroke-width="2"
    />
    <text x={W - pad.right + 3} y={sigmaY + 3} class="line-label sigma-label">σ</text>

    <!-- KV-cache line (growing red) -->
    <polyline
      points={kvLine}
      fill="none" stroke="var(--rose)" stroke-width="2"
    />
    <text x={W - pad.right + 3} y={yScale(KV_PER_TOKEN * maxT) + 3} class="line-label kv-label">KV</text>

    <!-- Current token marker -->
    {#if $tokenCount > 0}
      <line
        x1={cursorX} y1={pad.top}
        x2={cursorX} y2={pad.top + plotH}
        stroke="rgba(255,255,255,0.2)"
        stroke-width="1"
        stroke-dasharray="2,2"
      />
      <!-- σ dot -->
      <circle cx={cursorX} cy={sigmaY} r="3.5" fill="#3b82f6" stroke="#fff" stroke-width="0.5" />
      <!-- KV dot -->
      <circle cx={cursorX} cy={cursorKvY} r="3.5" fill="var(--rose)" stroke="#fff" stroke-width="0.5" />
    {/if}

    <!-- Axis label -->
    <text x={xScale(maxT / 2)} y={H - 1} class="axis-title" text-anchor="middle">tokens processed</text>
  </svg>

  <div class="stats-row">
    <div class="stat">
      <span class="stat-dot sigma-dot"></span>
      <span class="stat-label">BDH σ:</span>
      <span class="stat-value">{sigmaLabel}</span>
      <span class="stat-note">(constant)</span>
    </div>
    <div class="stat">
      <span class="stat-dot kv-dot"></span>
      <span class="stat-label">GPT KV:</span>
      <span class="stat-value">{kvLabel}</span>
      <span class="stat-note">(+1KB/token)</span>
    </div>
  </div>

  {#if $tokenCount > CROSSOVER}
    <p class="crossover-note">
      The KV-cache has crossed <strong>the σ threshold</strong> — a transformer would now use more memory than BDH. At paper scale (N=32K), this happens at T≈400.
    </p>
  {/if}
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

  .panel-header { margin-bottom: 0.8rem; }

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
  }

  .panel-desc strong { color: var(--gold); }

  .chart-svg {
    width: 100%;
    max-width: 360px;
    display: block;
    margin: 0 auto;
  }

  .axis-label {
    font-family: var(--font-mono);
    font-size: 7px;
    fill: var(--text-dim);
  }

  .axis-title {
    font-family: var(--font-mono);
    font-size: 7px;
    fill: var(--text-muted);
  }

  .crossover-label {
    font-family: var(--font-mono);
    font-size: 7px;
    fill: var(--gold);
    font-weight: 600;
  }

  .line-label {
    font-family: var(--font-mono);
    font-size: 8px;
    font-weight: 700;
  }

  .sigma-label { fill: #3b82f6; }
  .kv-label { fill: var(--rose); }

  .stats-row {
    display: flex;
    gap: 1.2rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
  }

  .stat {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
  }

  .stat-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .sigma-dot { background: #3b82f6; }
  .kv-dot { background: var(--rose); }

  .stat-label {
    color: var(--text-secondary);
    font-weight: 500;
  }

  .stat-value {
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-note {
    font-size: 0.68rem;
    color: var(--text-dim);
  }

  .crossover-note {
    font-size: 0.75rem;
    color: var(--gold);
    margin: 0.5rem 0 0;
    line-height: 1.5;
  }

  .crossover-note strong {
    color: var(--text-primary);
  }
</style>
