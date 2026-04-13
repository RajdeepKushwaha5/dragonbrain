<script>
  import { createEventDispatcher } from 'svelte';
  import { tokenToChar } from '../lib/tokenizer.js';

  export let bdhModel = null;
  export let gptModel = null;
  export let modelReady = false;

  const dispatch = createEventDispatcher();

  let prompt = 'To be, or not to be';
  let generating = false;
  let maxTokens = 150;
  let temperature = 0.8;

  // Generated outputs
  let bdhChars = [];
  let gptChars = [];
  let bdhDone = false;
  let gptDone = false;

  // Loss tracking per character
  let bdhLosses = [];
  let gptLosses = [];

  function softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp(logits[i] - max);
      sum += exps[i];
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  function sampleFromLogits(logits, temp) {
    // Apply temperature
    const scaled = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      scaled[i] = logits[i] / temp;
    }
    const probs = softmax(scaled);

    // Sample from distribution
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) return i;
    }
    return probs.length - 1;
  }

  function crossEntropy(logits, targetToken) {
    const probs = softmax(logits);
    const p = Math.max(probs[targetToken], 1e-10);
    return -Math.log(p);
  }

  async function generate() {
    if (!modelReady || generating) return;
    generating = true;
    bdhChars = [];
    gptChars = [];
    bdhLosses = [];
    gptLosses = [];
    bdhDone = false;
    gptDone = false;

    const encoder = new TextEncoder();
    const promptBytes = Array.from(encoder.encode(prompt));

    // Generate BDH and GPT in interleaved fashion for visual effect
    let bdhBuffer = [...promptBytes];
    let gptBuffer = [...promptBytes];

    for (let step = 0; step < maxTokens; step++) {
      if (!generating) break;

      // BDH generation
      if (!bdhDone) {
        try {
          const bdhTail = bdhBuffer.slice(-128);
          const bdhResult = await bdhModel.runToken(bdhTail);
          const bdhLogits = bdhResult?.logits;
          if (bdhLogits) {
            const nextToken = sampleFromLogits(bdhLogits, temperature);
            const loss = crossEntropy(bdhLogits, nextToken);
            bdhBuffer.push(nextToken);
            bdhChars = [...bdhChars, { token: nextToken, char: tokenToChar(nextToken), loss }];
            bdhLosses = [...bdhLosses, loss];

            // Update sigma for inference-time learning
            if (bdhResult.layers && bdhResult.layers[0]) {
              bdhModel.updateSigma(bdhResult.layers[0].x_last, bdhResult.layers[0].y_last, 0);
              if (bdhResult.layers.length > 1) {
                bdhModel.updateSigma(bdhResult.layers[1].x_last, bdhResult.layers[1].y_last, 1);
              }
            }
          } else {
            bdhDone = true;
          }
        } catch (e) {
          console.warn('BDH generation error:', e);
          bdhDone = true;
        }
      }

      // GPT generation
      if (!gptDone) {
        try {
          const gptTail = gptBuffer.slice(-128);
          const gptResult = await gptModel.runToken(gptTail);
          const gptLogits = gptResult?.logits;
          if (gptLogits) {
            const nextToken = sampleFromLogits(gptLogits, temperature);
            const loss = crossEntropy(gptLogits, nextToken);
            gptBuffer.push(nextToken);
            gptChars = [...gptChars, { token: nextToken, char: tokenToChar(nextToken), loss }];
            gptLosses = [...gptLosses, loss];
          } else {
            gptDone = true;
          }
        } catch (e) {
          console.warn('GPT generation error:', e);
          gptDone = true;
        }
      }

      // Small delay for visual streaming effect
      await new Promise(r => setTimeout(r, 15));
    }

    generating = false;
  }

  function stopGeneration() {
    generating = false;
  }

  function avgLoss(losses) {
    if (!losses.length) return 0;
    return losses.reduce((a, b) => a + b, 0) / losses.length;
  }

  // Color: green (low loss) → yellow → red (high loss)
  function lossColor(loss) {
    const t = Math.min(loss / 6, 1); // clamp at loss=6
    if (t < 0.5) {
      // green → yellow
      const r = Math.round(t * 2 * 255);
      const g = 200;
      const b = Math.round((1 - t * 2) * 80);
      return `rgb(${r},${g},${b})`;
    } else {
      // yellow → red
      const r = 255;
      const g = Math.round((1 - (t - 0.5) * 2) * 200);
      const b = Math.round((1 - (t - 0.5) * 2) * 40);
      return `rgb(${r},${g},${b})`;
    }
  }
</script>

<div class="gen-panel">
  <div class="gen-header">
    <h3 class="gen-title">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>
      Text Generation — BDH vs GPT
    </h3>
    <span class="gen-subtitle">Side-by-side autoregressive generation with loss coloring</span>
  </div>

  <div class="gen-controls">
    <div class="gen-input-row">
      <label class="gen-label" for="gen-prompt-input">Prompt</label>
      <input
        id="gen-prompt-input"
        class="gen-input"
        type="text"
        bind:value={prompt}
        placeholder="Type a prompt..."
        disabled={generating}
      />
    </div>
    <div class="gen-params">
      <label class="gen-param">
        <span>Tokens</span>
        <input type="number" min="10" max="500" bind:value={maxTokens} disabled={generating} />
      </label>
      <label class="gen-param">
        <span>Temp</span>
        <input type="number" min="0.1" max="2.0" step="0.1" bind:value={temperature} disabled={generating} />
      </label>
      {#if !generating}
        <button class="gen-btn gen-start" on:click={generate} disabled={!modelReady}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Generate
        </button>
      {:else}
        <button class="gen-btn gen-stop" on:click={stopGeneration}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"/></svg>
          Stop
        </button>
      {/if}
    </div>
  </div>

  <div class="gen-outputs">
    <!-- BDH Output -->
    <div class="gen-output bdh-output">
      <div class="gen-output-header">
        <span class="gen-model-label bdh-label">BDH</span>
        <span class="gen-model-tag">229K · Sparse · σ-memory</span>
        {#if bdhLosses.length > 0}
          <span class="gen-loss-avg">avg loss: {avgLoss(bdhLosses).toFixed(2)}</span>
        {/if}
      </div>
      <div class="gen-text">
        <span class="gen-prompt-text">{prompt}</span>{#each bdhChars as c}<span
          class="gen-char"
          style="color: {lossColor(c.loss)}"
          title="'{c.char}' loss={c.loss.toFixed(2)}"
        >{c.char}</span>{/each}{#if generating && !bdhDone}<span class="gen-cursor">▌</span>{/if}
      </div>
    </div>

    <!-- GPT Output -->
    <div class="gen-output gpt-output">
      <div class="gen-output-header">
        <span class="gen-model-label gpt-label">GPT</span>
        <span class="gen-model-tag">148K · Dense · Transformer</span>
        {#if gptLosses.length > 0}
          <span class="gen-loss-avg">avg loss: {avgLoss(gptLosses).toFixed(2)}</span>
        {/if}
      </div>
      <div class="gen-text">
        <span class="gen-prompt-text">{prompt}</span>{#each gptChars as c}<span
          class="gen-char"
          style="color: {lossColor(c.loss)}"
          title="'{c.char}' loss={c.loss.toFixed(2)}"
        >{c.char}</span>{/each}{#if generating && !gptDone}<span class="gen-cursor">▌</span>{/if}
      </div>
    </div>
  </div>

  <!-- Loss Legend -->
  {#if bdhChars.length > 0 || gptChars.length > 0}
    <div class="gen-legend">
      <span class="gen-legend-label">Character color = model surprise:</span>
      <span class="gen-legend-gradient">
        <span style="color: rgb(0,200,80)">■</span> low loss
        <span style="color: rgb(255,200,40)">■</span> medium
        <span style="color: rgb(255,0,0)">■</span> high loss
      </span>
    </div>
  {/if}
</div>

<style>
  .gen-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md, 10px);
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }

  .gen-header {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .gen-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: var(--font-sans);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .gen-subtitle {
    font-size: 0.75rem;
    color: var(--text-dim);
    font-family: var(--font-mono);
  }

  .gen-controls {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .gen-input-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .gen-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    flex-shrink: 0;
  }

  .gen-input {
    flex: 1;
    background: var(--bg-input, #101118);
    border: 1px solid var(--border-default);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 0.85rem;
    outline: none;
    transition: border-color 0.15s;
  }

  .gen-input:focus {
    border-color: var(--accent);
  }

  .gen-params {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    flex-wrap: wrap;
  }

  .gen-param {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
  }

  .gen-param input {
    width: 60px;
    background: var(--bg-input, #101118);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 0.25rem 0.35rem;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    text-align: center;
    outline: none;
  }

  .gen-btn {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.85rem;
    border-radius: 6px;
    font-family: var(--font-sans);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    border: 1px solid;
    margin-left: auto;
  }

  .gen-start {
    background: rgba(61, 214, 140, 0.1);
    border-color: var(--green, #3dd68c);
    color: var(--green, #3dd68c);
  }

  .gen-start:hover:not(:disabled) {
    background: rgba(61, 214, 140, 0.18);
  }

  .gen-start:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .gen-stop {
    background: rgba(240, 98, 146, 0.1);
    border-color: var(--rose, #f06292);
    color: var(--rose, #f06292);
  }

  .gen-stop:hover {
    background: rgba(240, 98, 146, 0.2);
  }

  .gen-outputs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
  }

  .gen-output {
    background: var(--bg-secondary, #0f1015);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.7rem 0.85rem;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .gen-output-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .gen-model-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
  }

  .bdh-label {
    color: var(--accent);
    background: rgba(91, 141, 239, 0.12);
    border: 1px solid rgba(91, 141, 239, 0.25);
  }

  .gpt-label {
    color: var(--violet, #9b7ef0);
    background: rgba(155, 126, 240, 0.12);
    border: 1px solid rgba(155, 126, 240, 0.25);
  }

  .gen-model-tag {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-dim);
  }

  .gen-loss-avg {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--gold, #f0c246);
    margin-left: auto;
  }

  .gen-text {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    line-height: 1.65;
    color: var(--text-secondary);
    word-break: break-word;
    white-space: pre-wrap;
    max-height: 250px;
    overflow-y: auto;
  }

  .gen-prompt-text {
    color: var(--text-dim);
    opacity: 0.7;
  }

  .gen-char {
    transition: color 0.1s;
    cursor: default;
  }

  .gen-cursor {
    color: var(--accent);
    animation: blink 0.8s step-end infinite;
  }

  @keyframes blink {
    50% { opacity: 0; }
  }

  .gen-legend {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.5rem;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
  }

  .gen-legend-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
  }

  .gen-legend-gradient {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    display: flex;
    gap: 0.4rem;
  }

  @media (max-width: 768px) {
    .gen-outputs {
      grid-template-columns: 1fr;
    }
  }
</style>
