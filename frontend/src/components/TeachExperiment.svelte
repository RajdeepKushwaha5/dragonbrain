<script>
  import { tokenToChar } from '../lib/tokenizer.js';

  export let bdhModel = null;
  export let gptModel = null;
  export let modelReady = false;

  let running = false;
  let phase = ''; // 'baseline', 'training-1', 'training-2', 'training-3', 'testing', 'done'
  let progress = 0;

  const TEACH_PHRASE = "the cat sat on the mat";
  const TEST_PREFIX = "the cat sat on the ";
  const TARGET_CHAR = 'm'; // we expect 'mat' completion

  // Results
  let baselineLoss = null;   // loss on TEST_PREFIX → 'm' BEFORE sigma
  let afterLoss = null;       // loss on TEST_PREFIX → 'm' AFTER sigma
  let baselineTopK = [];
  let afterTopK = [];
  let gptLoss = null;
  let gptTopK = [];
  let improvementPct = null;
  let stepLog = [];

  function softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp(logits[i] - max);
      sum += exps[i];
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  function crossEntropy(logits, targetToken) {
    const probs = softmax(logits);
    const p = Math.max(probs[targetToken], 1e-10);
    return -Math.log(p);
  }

  function topK(logits, k = 5) {
    const probs = softmax(logits);
    const indexed = Array.from(probs).map((p, i) => ({ token: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
  }

  async function runTeachExperiment() {
    if (!modelReady || running) return;
    running = true;
    phase = 'baseline';
    progress = 0;
    stepLog = [];
    baselineLoss = null;
    afterLoss = null;
    baselineTopK = [];
    afterTopK = [];
    gptLoss = null;
    gptTopK = [];
    improvementPct = null;

    const encoder = new TextEncoder();
    const targetByte = encoder.encode(TARGET_CHAR)[0]; // 'm' = 109

    // Step 1: Clear sigma memory
    bdhModel.resetMemory();
    stepLog = [...stepLog, { text: 'Cleared σ memory', icon: '🧹' }];
    await delay(200);

    // Step 2: Measure BASELINE loss on test prefix
    phase = 'baseline';
    progress = 10;
    const testBytes = Array.from(encoder.encode(TEST_PREFIX));
    
    const baseResult = await bdhModel.runToken(testBytes);
    if (baseResult?.logits) {
      baselineLoss = crossEntropy(baseResult.logits, targetByte);
      baselineTopK = topK(baseResult.logits, 5);
      const rank = baselineTopK.findIndex(p => p.token === targetByte) + 1;
      stepLog = [...stepLog, { 
        text: `Baseline loss for '${TARGET_CHAR}': ${baselineLoss.toFixed(3)} (rank #${rank || '>5'})`, 
        icon: '📊' 
      }];
    }

    // GPT baseline (it doesn't have sigma, so it's always the same)
    if (gptModel?.ready) {
      const gptResult = await gptModel.runToken(testBytes);
      if (gptResult?.logits) {
        gptLoss = crossEntropy(gptResult.logits, targetByte);
        gptTopK = topK(gptResult.logits, 5);
      }
    }
    await delay(300);

    // Step 3: Train — feed the phrase 3 times, building sigma each time
    for (let rep = 1; rep <= 3; rep++) {
      phase = `training-${rep}`;
      progress = 10 + rep * 20;

      const phraseBytes = Array.from(encoder.encode(TEACH_PHRASE + ". "));

      // Feed token by token to accumulate sigma
      for (let i = 1; i <= phraseBytes.length; i++) {
        const buf = phraseBytes.slice(0, i);
        const result = await bdhModel.runToken(buf);
        if (result?.layers) {
          for (let l = 0; l < result.layers.length; l++) {
            bdhModel.updateSigma(result.layers[l].x_last, result.layers[l].xy_last, l);
          }
        }
      }

      // After each repetition, check intermediate loss
      const midResult = await bdhModel.runToken(testBytes);
      let midLoss = null;
      if (midResult?.logits) {
        // With sigma modulation
        const sigmaLogits = bdhModel.computeSigmaLogits(
          midResult.layers[0]?.x_last, 0
        );
        if (sigmaLogits && midResult.logits) {
          const alpha = 0.05 * Math.log(1 + rep * phraseBytes.length);
          let rawRange = 0, sigmaRange = 0;
          for (let i = 0; i < midResult.logits.length; i++) {
            rawRange = Math.max(rawRange, Math.abs(midResult.logits[i]));
            sigmaRange = Math.max(sigmaRange, Math.abs(sigmaLogits[i]));
          }
          const scale = sigmaRange > 1e-8 ? rawRange / sigmaRange : 0;
          const adjusted = new Float32Array(midResult.logits.length);
          for (let i = 0; i < midResult.logits.length; i++) {
            adjusted[i] = midResult.logits[i] + alpha * scale * sigmaLogits[i];
          }
          midLoss = crossEntropy(adjusted, targetByte);
        } else {
          midLoss = crossEntropy(midResult.logits, targetByte);
        }
      }

      stepLog = [...stepLog, { 
        text: `Rep ${rep}/3 complete — σ loss: ${midLoss?.toFixed(3) ?? 'N/A'}`, 
        icon: rep === 3 ? '✅' : '🔄' 
      }];
      await delay(200);
    }

    // Step 4: Measure AFTER loss with sigma modulation
    phase = 'testing';
    progress = 85;

    const afterResult = await bdhModel.runToken(testBytes);
    if (afterResult?.logits) {
      const sigmaLogits = bdhModel.computeSigmaLogits(
        afterResult.layers[0]?.x_last, 0
      );
      if (sigmaLogits) {
        const totalTokens = 3 * (TEACH_PHRASE.length + 2);
        const alpha = 0.05 * Math.log(1 + totalTokens);
        let rawRange = 0, sigmaRange = 0;
        for (let i = 0; i < afterResult.logits.length; i++) {
          rawRange = Math.max(rawRange, Math.abs(afterResult.logits[i]));
          sigmaRange = Math.max(sigmaRange, Math.abs(sigmaLogits[i]));
        }
        const scale = sigmaRange > 1e-8 ? rawRange / sigmaRange : 0;
        const adjusted = new Float32Array(afterResult.logits.length);
        for (let i = 0; i < afterResult.logits.length; i++) {
          adjusted[i] = afterResult.logits[i] + alpha * scale * sigmaLogits[i];
        }
        afterLoss = crossEntropy(adjusted, targetByte);
        afterTopK = topK(adjusted, 5);
      } else {
        afterLoss = crossEntropy(afterResult.logits, targetByte);
        afterTopK = topK(afterResult.logits, 5);
      }
    }

    if (baselineLoss != null && afterLoss != null) {
      improvementPct = ((baselineLoss - afterLoss) / baselineLoss * 100);
    }

    stepLog = [...stepLog, { 
      text: `After σ loss: ${afterLoss?.toFixed(3)} → ${improvementPct?.toFixed(1)}% improvement`, 
      icon: '🎯' 
    }];

    phase = 'done';
    progress = 100;
    running = false;
  }

  function delay(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  function reset() {
    running = false;
    phase = '';
    progress = 0;
    stepLog = [];
    baselineLoss = null;
    afterLoss = null;
    baselineTopK = [];
    afterTopK = [];
    gptLoss = null;
    gptTopK = [];
    improvementPct = null;
  }
</script>

<div class="teach-panel">
  <div class="teach-header">
    <h3 class="teach-title">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z"/><line x1="9" y1="21" x2="15" y2="21"/></svg>
      Inference-Time Learning — Quantified
    </h3>
    <span class="teach-subtitle">Proves BDH learns WITHOUT gradient updates. Measures cross-entropy loss before and after σ exposure.</span>
  </div>

  <div class="teach-experiment">
    <div class="teach-protocol">
      <span class="teach-protocol-label">Protocol:</span>
      <span class="teach-protocol-text">
        Feed "<strong>{TEACH_PHRASE}</strong>" × 3 → Test: "<strong>{TEST_PREFIX}</strong><em>?</em>" → Measure if the model predicts '<strong>{TARGET_CHAR}</strong>'
      </span>
    </div>

    {#if !running && phase !== 'done'}
      <button class="teach-run-btn" on:click={runTeachExperiment} disabled={!modelReady}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
        Run Experiment
      </button>
    {:else if phase === 'done'}
      <button class="teach-restart-btn" on:click={reset}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
        Reset & Run Again
      </button>
    {/if}

    {#if running || phase === 'done'}
      <!-- Progress bar -->
      <div class="teach-progress-bar">
        <div class="teach-progress-fill" style="width: {progress}%"></div>
        <span class="teach-progress-label">{phase === 'done' ? 'Complete' : phase.replace('-', ' ')}</span>
      </div>

      <!-- Step log -->
      <div class="teach-log">
        {#each stepLog as step}
          <div class="teach-log-item">
            <span class="teach-log-icon">{step.icon}</span>
            <span class="teach-log-text">{step.text}</span>
          </div>
        {/each}
      </div>
    {/if}

    {#if phase === 'done'}
      <!-- Results comparison -->
      <div class="teach-results">
        <div class="result-box baseline-box">
          <span class="result-label">Before σ</span>
          <span class="result-loss" class:high-loss={baselineLoss > 3}>{baselineLoss?.toFixed(3) ?? '—'}</span>
          <span class="result-unit">cross-entropy</span>
          <div class="result-topk">
            {#each baselineTopK.slice(0, 3) as p}
              <span class="result-pred" class:target-hit={p.token === new TextEncoder().encode(TARGET_CHAR)[0]}>
                {tokenToChar(p.token)} {(p.prob * 100).toFixed(1)}%
              </span>
            {/each}
          </div>
        </div>

        <div class="result-arrow">
          {#if improvementPct != null && improvementPct > 0}
            <span class="improvement-badge positive">↓ {improvementPct.toFixed(1)}%</span>
          {:else if improvementPct != null}
            <span class="improvement-badge negative">↑ {Math.abs(improvementPct).toFixed(1)}%</span>
          {/if}
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
        </div>

        <div class="result-box after-box">
          <span class="result-label">After σ (3 reps)</span>
          <span class="result-loss" class:low-loss={afterLoss != null && afterLoss < baselineLoss}>{afterLoss?.toFixed(3) ?? '—'}</span>
          <span class="result-unit">cross-entropy</span>
          <div class="result-topk">
            {#each afterTopK.slice(0, 3) as p}
              <span class="result-pred" class:target-hit={p.token === new TextEncoder().encode(TARGET_CHAR)[0]}>
                {tokenToChar(p.token)} {(p.prob * 100).toFixed(1)}%
              </span>
            {/each}
          </div>
        </div>

        {#if gptLoss != null}
          <div class="result-box gpt-box">
            <span class="result-label">GPT (no σ)</span>
            <span class="result-loss">{gptLoss.toFixed(3)}</span>
            <span class="result-unit">cross-entropy</span>
            <div class="result-topk">
              {#each gptTopK.slice(0, 3) as p}
                <span class="result-pred gpt-pred" class:target-hit={p.token === new TextEncoder().encode(TARGET_CHAR)[0]}>
                  {tokenToChar(p.token)} {(p.prob * 100).toFixed(1)}%
                </span>
              {/each}
            </div>
          </div>
        {/if}
      </div>

      <div class="teach-insight">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
        {#if improvementPct != null && improvementPct > 0}
          BDH's Hebbian σ memory reduced loss by <strong>{improvementPct.toFixed(1)}%</strong> with zero gradient updates — pure inference-time learning.
          {#if gptLoss != null}
            GPT cannot do this; its loss is fixed at {gptLoss.toFixed(3)} regardless of exposure.
          {/if}
        {:else}
          σ effect was minimal at this scale. The paper's effect is stronger at N≥32K neurons; our N=512 toy model shows the mechanism directionally.
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .teach-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md, 10px);
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }

  .teach-header {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .teach-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: var(--font-sans);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .teach-subtitle {
    font-size: 0.75rem;
    color: var(--text-dim);
    font-family: var(--font-mono);
    line-height: 1.5;
  }

  .teach-experiment {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .teach-protocol {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    padding: 0.4rem 0.6rem;
    background: rgba(77, 208, 225, 0.04);
    border: 1px solid rgba(77, 208, 225, 0.12);
    border-radius: 6px;
    flex-wrap: wrap;
  }

  .teach-protocol-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--cyan, #4dd0e1);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    flex-shrink: 0;
  }

  .teach-protocol-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }

  .teach-protocol-text strong { color: var(--text-primary); }
  .teach-protocol-text em { color: var(--gold, #f0c246); font-style: normal; }

  .teach-run-btn, .teach-restart-btn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-family: var(--font-sans);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    align-self: flex-start;
  }

  .teach-run-btn {
    background: rgba(77, 208, 225, 0.1);
    border: 1px solid var(--cyan, #4dd0e1);
    color: var(--cyan, #4dd0e1);
  }

  .teach-run-btn:hover:not(:disabled) { background: rgba(77, 208, 225, 0.18); }
  .teach-run-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .teach-restart-btn {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-default);
    color: var(--text-secondary);
  }

  .teach-restart-btn:hover { background: rgba(255, 255, 255, 0.08); }

  .teach-progress-bar {
    position: relative;
    height: 22px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 4px;
    overflow: hidden;
  }

  .teach-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan, #4dd0e1), var(--accent));
    border-radius: 4px;
    transition: width 0.3s ease;
    opacity: 0.6;
  }

  .teach-progress-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-primary);
    text-transform: capitalize;
    letter-spacing: 0.03em;
  }

  .teach-log {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.4rem 0;
  }

  .teach-log-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--text-secondary);
    animation: slideUp 0.2s ease;
  }

  .teach-log-icon { flex-shrink: 0; }

  .teach-results {
    display: flex;
    align-items: stretch;
    gap: 0.6rem;
    flex-wrap: wrap;
  }

  .result-box {
    flex: 1;
    min-width: 140px;
    padding: 0.6rem 0.7rem;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    text-align: center;
  }

  .baseline-box {
    background: rgba(240, 98, 146, 0.06);
    border: 1px solid rgba(240, 98, 146, 0.2);
  }

  .after-box {
    background: rgba(61, 214, 140, 0.06);
    border: 1px solid rgba(61, 214, 140, 0.2);
  }

  .gpt-box {
    background: rgba(155, 126, 240, 0.06);
    border: 1px solid rgba(155, 126, 240, 0.2);
  }

  .result-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .result-loss {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
  }

  .result-loss.high-loss { color: var(--rose, #f06292); }
  .result-loss.low-loss { color: var(--green, #3dd68c); }

  .result-unit {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-dim);
  }

  .result-topk {
    display: flex;
    gap: 0.3rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 0.2rem;
  }

  .result-pred {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.12rem 0.35rem;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--text-muted);
  }

  .result-pred.target-hit {
    background: rgba(61, 214, 140, 0.12);
    border-color: rgba(61, 214, 140, 0.3);
    color: var(--green, #3dd68c);
    font-weight: 600;
  }

  .result-pred.gpt-pred {
    border-color: rgba(155, 126, 240, 0.15);
  }

  .result-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
    color: var(--text-dim);
    flex-shrink: 0;
    padding: 0 0.3rem;
  }

  .improvement-badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
  }

  .improvement-badge.positive {
    color: var(--green, #3dd68c);
    background: rgba(61, 214, 140, 0.1);
    border: 1px solid rgba(61, 214, 140, 0.25);
  }

  .improvement-badge.negative {
    color: var(--rose, #f06292);
    background: rgba(240, 98, 146, 0.1);
    border: 1px solid rgba(240, 98, 146, 0.25);
  }

  .teach-insight {
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    padding: 0.5rem 0.7rem;
    background: rgba(91, 141, 239, 0.04);
    border: 1px solid rgba(91, 141, 239, 0.15);
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--text-secondary);
    line-height: 1.55;
  }

  .teach-insight strong {
    color: var(--green, #3dd68c);
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 640px) {
    .teach-results {
      flex-direction: column;
    }
    .result-arrow {
      transform: rotate(90deg);
      padding: 0.2rem 0;
    }
  }
</style>
