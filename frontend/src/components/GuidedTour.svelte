<script>
  import { createEventDispatcher, onDestroy } from 'svelte';

  const dispatch = createEventDispatcher();
  export let active = false;

  const STEPS = [
    {
      target: null,
      title: 'Welcome to Dragon Brain',
      description: 'Explore a BDH neural network that learns at inference time \u2014 no gradient updates needed. This tour walks through each visualization panel.',
      icon: 'welcome',
      section: null,
    },
    {
      target: '[data-tour="input"]',
      title: 'Feed the Brain',
      description: 'Type any text here. Each character is a byte token (0\u2013255) fed into the BDH model running in your browser via WebAssembly. Try typing a few words.',
      icon: 'input',
      section: null,
    },
    {
      target: '[data-tour="predictions"]',
      title: 'Predictions Diverge',
      description: 'Three rows: BDH Raw (base output), \u03C3-Learned (adjusted by Hebbian memory), and GPT (transformer baseline). As \u03C3 builds, watch the learned row shift.',
      icon: 'predictions',
      section: null,
    },
    {
      target: '[data-tour="sparse"]',
      title: 'Sparse vs Dense',
      description: 'BDH activates ~5% of neurons while GPT lights up ~97%. This extreme sparsity means individual neurons carry interpretable meaning.',
      icon: 'sparse',
      section: 'activations',
    },
    {
      target: '[data-tour="graph"]',
      title: 'Network Topology',
      description: 'Force-directed graph of all 512 neurons. Bright nodes are currently active. Toggle Gx (thought flow) and Gy (memory echo) for different views.',
      icon: 'graph',
      section: 'activations',
    },
    {
      target: '[data-tour="heatmap"]',
      title: 'Hebbian Memory (\u03C3)',
      description: 'The \u03C3 matrix shows accumulated co-activation patterns. Bright cells = neurons that fire together. Colored borders mark named concept synapses.',
      icon: 'heatmap',
      section: 'internals',
    },
    {
      target: '[data-tour="attention"]',
      title: 'Causal Attention',
      description: 'Standard attention pattern showing token-to-token focus. Attention is ephemeral, while \u03C3 memory persists across the full sequence.',
      icon: 'attention',
      section: 'internals',
    },
    {
      target: '[data-tour="memory"]',
      title: 'Memory Efficiency',
      description: 'BDH\u2019s \u03C3 matrix is fixed O(1) memory regardless of length. GPT\u2019s KV-cache grows linearly. For long sequences, BDH uses orders of magnitude less.',
      icon: 'memory',
      section: 'insights',
    },
    {
      target: '[data-tour="tracer"]',
      title: 'Synapse Tracer',
      description: 'See which concept-synapses drive prediction shifts. Each row: a named synapse pair, its firing strength, and direction \u2014 making reasoning transparent.',
      icon: 'tracer',
      section: 'insights',
    },
  ];

  let currentStep = 0;
  let prevHighlight = null;

  function clearHighlight() {
    if (prevHighlight) {
      prevHighlight.classList.remove('tour-highlight');
      prevHighlight = null;
    }
  }

  function highlightTarget(step) {
    clearHighlight();
    if (step && step.section) {
      dispatch('expandSection', step.section);
    }
    // Wait for DOM to update after section expansion
    setTimeout(() => {
      if (step && step.target) {
        const el = document.querySelector(step.target);
        if (el) {
          el.classList.add('tour-highlight');
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          prevHighlight = el;
        }
      }
    }, 150);
  }

  $: if (active) {
    currentStep = 0;
    // Delay highlight so DOM is ready
    setTimeout(() => highlightTarget(STEPS[0]), 100);
  }

  function next() {
    if (currentStep < STEPS.length - 1) {
      currentStep++;
      highlightTarget(STEPS[currentStep]);
    } else {
      close();
    }
  }

  function prev() {
    if (currentStep > 0) {
      currentStep--;
      highlightTarget(STEPS[currentStep]);
    }
  }

  function close() {
    clearHighlight();
    dispatch('close');
  }

  function handleKeydown(e) {
    if (!active) return;
    if (e.key === 'Escape') close();
    if (e.key === 'ArrowRight') next();
    if (e.key === 'ArrowLeft') prev();
  }

  onDestroy(clearHighlight);

  $: step = STEPS[currentStep] || STEPS[0];
  $: progress = ((currentStep + 1) / STEPS.length * 100).toFixed(0);
</script>

<svelte:window on:keydown={handleKeydown} />

{#if active}
  <div class="tour-backdrop" on:click={close} role="presentation"></div>
  <div class="tour-card" role="dialog" aria-label="Guided tour">
    <div class="tour-progress-bar">
      <div class="tour-progress-fill" style="width: {progress}%"></div>
    </div>
    <div class="tour-content">
      <div class="tour-step-badge">
        <span class="tour-icon" aria-hidden="true">
          {#if step.icon === 'welcome'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          {:else if step.icon === 'input'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8h.01"/><path d="M10 8h.01"/><path d="M14 8h.01"/><rect x="6" y="12" width="12" height="4" rx="1"/></svg>
          {:else if step.icon === 'predictions'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
          {:else if step.icon === 'sparse'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          {:else if step.icon === 'graph'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><line x1="8.5" y1="7.5" x2="15.5" y2="16.5"/><line x1="15.5" y1="7.5" x2="8.5" y2="16.5"/></svg>
          {:else if step.icon === 'heatmap'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
          {:else if step.icon === 'attention'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          {:else if step.icon === 'memory'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
          {:else if step.icon === 'tracer'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          {/if}
        </span>
        <span class="tour-step-count">{currentStep + 1} / {STEPS.length}</span>
      </div>
      <h3 class="tour-title">{step.title}</h3>
      <p class="tour-desc">{step.description}</p>
    </div>
    <div class="tour-actions">
      <button class="tour-btn tour-skip" on:click={close}>Skip tour</button>
      <div class="tour-nav">
        {#if currentStep > 0}
          <button class="tour-btn tour-prev" on:click={prev}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
            Prev
          </button>
        {/if}
        <button class="tour-btn tour-next" on:click={next}>
          {currentStep < STEPS.length - 1 ? 'Next' : 'Finish'}
          {#if currentStep < STEPS.length - 1}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .tour-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 900;
    animation: fadeIn 0.2s ease;
  }

  .tour-card {
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    width: min(460px, 90vw);
    background: var(--bg-elevated);
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-elevated), 0 0 40px rgba(91, 141, 239, 0.1);
    z-index: 1000;
    overflow: hidden;
    animation: slideUp 0.3s ease;
  }

  .tour-progress-bar {
    height: 3px;
    background: rgba(255, 255, 255, 0.06);
  }

  .tour-progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s ease;
    border-radius: 0 3px 3px 0;
  }

  .tour-content {
    padding: 1.2rem 1.4rem 0.8rem;
  }

  .tour-step-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .tour-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background: rgba(91, 141, 239, 0.12);
    color: var(--accent);
    flex-shrink: 0;
  }

  .tour-step-count {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
  }

  .tour-title {
    font-family: var(--font-sans);
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.35rem;
    letter-spacing: -0.02em;
  }

  .tour-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0;
  }

  .tour-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.4rem;
    border-top: 1px solid var(--border-subtle);
    background: rgba(0, 0, 0, 0.12);
  }

  .tour-nav {
    display: flex;
    gap: 0.45rem;
  }

  .tour-btn {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.4rem 0.85rem;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-secondary);
    font-family: var(--font-sans);
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .tour-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: var(--border-hover);
    color: var(--text-primary);
  }

  .tour-skip {
    border: none;
    background: none;
    color: var(--text-dim);
    font-size: 0.76rem;
    padding: 0.4rem 0.5rem;
  }

  .tour-skip:hover {
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.04);
  }

  .tour-next {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .tour-next:hover {
    background: var(--accent-bright);
    border-color: var(--accent-bright);
  }

  .tour-prev {
    color: var(--text-muted);
  }

  @media (max-width: 600px) {
    .tour-card {
      bottom: 0.75rem;
      width: min(460px, 95vw);
    }

    .tour-content {
      padding: 0.9rem 1rem 0.6rem;
    }

    .tour-title {
      font-size: 0.92rem;
    }

    .tour-desc {
      font-size: 0.8rem;
    }

    .tour-actions {
      padding: 0.6rem 1rem;
    }

    .tour-btn {
      padding: 0.4rem 0.7rem;
      min-height: 36px;
      font-size: 0.76rem;
    }

    .tour-skip {
      min-height: 36px;
    }
  }

  @media (max-width: 480px) {
    .tour-card {
      bottom: 0.5rem;
      border-radius: 12px;
    }

    .tour-actions {
      flex-wrap: wrap;
      gap: 0.4rem;
    }

    .tour-nav {
      width: 100%;
      justify-content: space-between;
    }
  }
</style>
