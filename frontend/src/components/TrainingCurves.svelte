<script>
  import { onMount, onDestroy } from 'svelte';
  import curvesData from '../data/training_curves.json';

  let canvas;
  let hoverInfo = null;
  let resizeOb;

  const COLORS = {
    bdh_val: '#5b8def',
    gpt_val: '#9b7ef0',
  };

  onMount(() => {
    draw();
    resizeOb = new ResizeObserver(() => draw());
    if (canvas?.parentElement) resizeOb.observe(canvas.parentElement);
  });
  onDestroy(() => resizeOb?.disconnect());
  $: if (canvas) draw();

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    const pad = { top: 24, right: 20, bottom: 36, left: 48 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Data ranges
    const maxIter = 5000;
    const allLosses = [...curvesData.bdh.val_loss, ...curvesData.gpt.val_loss];
    const minLoss = Math.floor(Math.min(...allLosses) * 10) / 10;
    const maxLoss = Math.ceil(Math.max(...allLosses) * 10) / 10;

    const xScale = (iter) => pad.left + (iter / maxIter) * plotW;
    const yScale = (loss) => pad.top + (1 - (loss - minLoss) / (maxLoss - minLoss)) * plotH;

    // Background
    ctx.fillStyle = 'transparent';
    ctx.clearRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    const yTicks = 6;
    for (let i = 0; i <= yTicks; i++) {
      const loss = minLoss + (maxLoss - minLoss) * (i / yTicks);
      const y = yScale(loss);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(loss.toFixed(1), pad.left - 6, y + 3);
    }

    // X-axis labels
    const xTicks = [0, 1000, 2000, 3000, 4000, 5000];
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'center';
    ctx.font = '10px monospace';
    for (const iter of xTicks) {
      ctx.fillText(iter.toString(), xScale(iter), H - pad.bottom + 16);
    }

    // Draw curves
    function drawLine(iters, losses, color, dashed = false) {
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 1 : 2;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.beginPath();
      for (let i = 0; i < iters.length; i++) {
        const x = xScale(iters[i]);
        const y = yScale(losses[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Val lines (primary)
    drawLine(curvesData.bdh.iters, curvesData.bdh.val_loss, COLORS.bdh_val);
    drawLine(curvesData.gpt.iters, curvesData.gpt.val_loss, COLORS.gpt_val);

    // Best point markers
    const bdhBest = curvesData.bdh.best_iter;
    const bdhBestIdx = curvesData.bdh.iters.indexOf(bdhBest);
    if (bdhBestIdx >= 0) {
      ctx.fillStyle = COLORS.bdh_val;
      ctx.beginPath();
      ctx.arc(xScale(bdhBest), yScale(curvesData.bdh.val_loss[bdhBestIdx]), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    const gptBest = curvesData.gpt.best_iter;
    const gptBestIdx = curvesData.gpt.iters.indexOf(gptBest);
    if (gptBestIdx >= 0) {
      ctx.fillStyle = COLORS.gpt_val;
      ctx.beginPath();
      ctx.arc(xScale(gptBest), yScale(curvesData.gpt.val_loss[gptBestIdx]), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Training Iteration', W / 2, H - 2);

    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Validation Loss', 0, 0);
    ctx.restore();
  }

  function handleMouseMove(e) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const W = rect.width;
    const pad = { left: 48, right: 20 };
    const plotW = W - pad.left - pad.right;
    const pct = (x - pad.left) / plotW;
    if (pct < 0 || pct > 1) { hoverInfo = null; return; }

    const iter = Math.round(pct * 5000);

    // Find nearest datapoint in each series
    function findNearest(iters, losses, targetIter) {
      let bestIdx = 0, bestDist = Infinity;
      for (let i = 0; i < iters.length; i++) {
        const d = Math.abs(iters[i] - targetIter);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      return losses[bestIdx];
    }

    hoverInfo = {
      iter,
      bdhVal: findNearest(curvesData.bdh.iters, curvesData.bdh.val_loss, iter),
      gptVal: findNearest(curvesData.gpt.iters, curvesData.gpt.val_loss, iter),
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }
</script>

<div class="curves-panel">
  <div class="curves-header">
    <h3 class="curves-title">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      Training Loss Curves
    </h3>
    <div class="curves-legend">
      <span class="legend-item"><span class="legend-swatch bdh-swatch"></span> BDH val</span>
      <span class="legend-item"><span class="legend-swatch gpt-swatch"></span> GPT val</span>
    </div>
  </div>

  <div class="curves-canvas-wrap"
    on:mousemove={handleMouseMove}
    on:mouseleave={() => hoverInfo = null}
    role="img"
    aria-label="Training loss curves for BDH and GPT models"
  >
    <canvas bind:this={canvas}></canvas>
    {#if hoverInfo}
      <div class="curves-tooltip" style="left: {hoverInfo.x + 10}px; top: {hoverInfo.y - 40}px;">
        <span class="tooltip-iter">iter {hoverInfo.iter}</span>
        <span class="tooltip-bdh">BDH: {hoverInfo.bdhVal.toFixed(3)}</span>
        <span class="tooltip-gpt">GPT: {hoverInfo.gptVal.toFixed(3)}</span>
      </div>
    {/if}
  </div>

  <div class="curves-summary">
    <div class="summary-item">
      <span class="summary-label">BDH best val</span>
      <span class="summary-value bdh-value">{curvesData.bdh.final_val}</span>
      <span class="summary-detail">@ iter {curvesData.bdh.best_iter}</span>
    </div>
    <div class="summary-item">
      <span class="summary-label">GPT best val</span>
      <span class="summary-value gpt-value">{curvesData.gpt.final_val}</span>
      <span class="summary-detail">@ iter {curvesData.gpt.best_iter}</span>
    </div>
    <div class="summary-item">
      <span class="summary-label">BDH advantage</span>
      <span class="summary-value advantage-value">{((curvesData.gpt.final_val - curvesData.bdh.final_val) / curvesData.gpt.final_val * 100).toFixed(1)}%</span>
      <span class="summary-detail">lower val loss</span>
    </div>
  </div>
  <div class="curves-note">Endpoints verified from saved checkpoints (BDH iter 4750, GPT iter 4999). Intermediate points interpolated. Both trained on Tiny Shakespeare, 5K iters.</div>
</div>

<style>
  .curves-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md, 10px);
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
  }

  .curves-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .curves-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: var(--font-sans);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .curves-legend {
    display: flex;
    align-items: center;
    gap: 0.7rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
  }

  .legend-swatch {
    width: 16px;
    height: 3px;
    border-radius: 2px;
  }

  .bdh-swatch { background: #5b8def; }
  .gpt-swatch { background: #9b7ef0; }

  .curves-canvas-wrap {
    position: relative;
    width: 100%;
    height: 220px;
    background: rgba(255, 255, 255, 0.015);
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    overflow: hidden;
  }

  canvas {
    width: 100%;
    height: 100%;
  }

  .curves-tooltip {
    position: absolute;
    pointer-events: none;
    background: var(--bg-elevated, #1c1d26);
    border: 1px solid var(--border-hover);
    border-radius: 6px;
    padding: 0.35rem 0.55rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10;
    white-space: nowrap;
  }

  .tooltip-iter { color: var(--text-dim); font-size: 0.65rem; }
  .tooltip-bdh { color: #5b8def; }
  .tooltip-gpt { color: #9b7ef0; }

  .curves-summary {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
  }

  .summary-item {
    flex: 1;
    min-width: 100px;
    text-align: center;
    padding: 0.4rem 0.5rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }

  .summary-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .summary-value {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
  }

  .bdh-value { color: #5b8def; }
  .gpt-value { color: #9b7ef0; }
  .advantage-value { color: var(--green, #3dd68c); }

  .summary-detail {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-dim);
  }

  .curves-note {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-dim);
    opacity: 0.7;
    line-height: 1.5;
    padding: 0.2rem 0;
  }
</style>
