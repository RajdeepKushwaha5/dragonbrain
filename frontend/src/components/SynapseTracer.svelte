<script>
  import { N } from '../lib/activation_math.js';
  import synapseData from '../data/synapse_labels.json';

  export let sigmaFlat = null;
  export let xActivations = null;
  export let tokenCount = 0;

  let topContributions = [];

  const allPairs = [];
  for (const [concept, data] of Object.entries(synapseData)) {
    for (const pair of data.pairs) {
      allPairs.push({
        concept,
        label: data.label,
        color: data.color,
        i: pair.i,
        j: pair.j,
        baseStrength: pair.strength,
      });
    }
  }

  $: if (sigmaFlat && xActivations && tokenCount > 0) {
    const contributions = [];
    for (const pair of allPairs) {
      const sigma_ij = sigmaFlat[pair.i * N + pair.j];
      const x_j = xActivations[pair.j] || 0;
      const firing = sigma_ij * x_j;

      if (Math.abs(firing) > 1e-8) {
        contributions.push({
          concept: pair.concept,
          label: pair.label,
          color: pair.color,
          i: pair.i,
          j: pair.j,
          sigma_ij,
          x_j,
          firing,
          absFiring: Math.abs(firing),
        });
      }
    }
    contributions.sort((a, b) => b.absFiring - a.absFiring);
    topContributions = contributions.slice(0, 6);
  } else {
    topContributions = [];
  }

  $: maxFiring = topContributions.length > 0 ? topContributions[0].absFiring : 1;
</script>

<div class="tracer-panel" data-tour="tracer">
  <div class="tracer-header">
    <div class="tracer-title-row">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      <h3 class="tracer-title">Synapse Reasoning Tracer</h3>
    </div>
    <span class="tracer-subtitle">concept-level firing decomposition</span>
  </div>

  {#if topContributions.length > 0}
    <div class="contributions">
      {#each topContributions as item, idx (item.concept + item.i + '_' + item.j)}
        <div class="contrib-row" style="animation-delay: {idx * 40}ms">
          <span class="concept-tag" style="border-color: {item.color}; color: {item.color}">
            {item.concept.replace('_', ' ')}
          </span>
          <span class="neuron-pair">
            n{item.i}&thinsp;&rarr;&thinsp;n{item.j}
          </span>
          <div class="firing-bar-container">
            <div
              class="firing-bar"
              style="width: {(item.absFiring / maxFiring * 100).toFixed(1)}%; background: {item.firing > 0 ? item.color : 'var(--rose)'}"
            ></div>
          </div>
          <span class="firing-value" class:pos={item.firing > 0} class:neg={item.firing < 0}>
            {item.firing > 0 ? '+' : ''}{item.firing.toFixed(3)}
          </span>
        </div>
      {/each}
    </div>
    <div class="tracer-legend">
      <span class="legend-item">
        <span class="legend-dot" style="background: var(--green)"></span>
        excitatory
      </span>
      <span class="legend-item">
        <span class="legend-dot" style="background: var(--rose)"></span>
        inhibitory
      </span>
      <span class="legend-sep">&middot;</span>
      <span class="legend-info">firing = &sigma;[i,j] &times; x[j]</span>
    </div>
  {:else}
    <div class="tracer-empty">
      <p>Type text and build &sigma; memory to see synapse-level reasoning.</p>
      <p class="tracer-hint">Active synapses will appear here once patterns emerge.</p>
    </div>
  {/if}
</div>

<style>
  .tracer-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
  }

  .tracer-header {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .tracer-title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-secondary);
  }

  .tracer-title {
    font-family: var(--font-sans);
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .tracer-subtitle {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .contributions {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .contrib-row {
    display: grid;
    grid-template-columns: minmax(5rem, 7.5rem) minmax(4rem, 5.5rem) 1fr minmax(3rem, 4.5rem);
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.55rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    animation: slideUp 0.2s ease both;
    transition: border-color var(--transition-fast);
  }

  .contrib-row:hover {
    border-color: var(--border-hover);
  }

  .concept-tag {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: capitalize;
    padding: 0.12rem 0.45rem;
    border: 1px solid;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
    white-space: nowrap;
    text-align: center;
  }

  .neuron-pair {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    white-space: nowrap;
  }

  .firing-bar-container {
    height: 6px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 3px;
    overflow: hidden;
    min-width: 40px;
  }

  .firing-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
    min-width: 2px;
  }

  .firing-value {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-align: right;
  }

  .firing-value.pos { color: var(--green); }
  .firing-value.neg { color: var(--rose); }

  .tracer-legend {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding-top: 0.3rem;
    border-top: 1px solid var(--border-subtle);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
  }

  .legend-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .legend-sep {
    color: var(--text-dim);
    font-size: 0.7rem;
  }

  .legend-info {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    color: var(--text-dim);
    margin-left: auto;
  }

  .tracer-empty {
    padding: 0.8rem 0;
    text-align: center;
  }

  .tracer-empty p {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0;
  }

  .tracer-hint {
    font-size: 0.76rem !important;
    color: var(--text-dim) !important;
    margin-top: 0.3rem !important;
  }

  @media (max-width: 600px) {
    .tracer-panel {
      padding: 0.7rem 0.8rem;
    }

    .contrib-row {
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto auto;
      gap: 0.3rem;
    }

    .firing-bar-container {
      grid-column: 1 / -1;
    }

    .concept-tag {
      font-size: 0.65rem;
    }

    .neuron-pair {
      font-size: 0.66rem;
    }

    .tracer-legend {
      flex-wrap: wrap;
      gap: 0.4rem;
    }
  }
</style>
