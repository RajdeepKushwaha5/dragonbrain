/**
 * Svelte stores — reactive state for the Dragon Brain app.
 */
import { writable, derived } from 'svelte/store';
import { NH, N, TOTAL_NEURONS } from './activation_math.js';

/** Raw text typed by the user */
export const inputText = writable('');

/** Token buffer (last ≤128 bytes) */
export const tokenBuffer = writable([]);

/** Currently selected layer index (0 or 1) */
export const selectedLayer = writable(0);

/** Currently selected head index for sigma display */
export const selectedHead = writable(0);

/** Whether the ONNX model is loaded and ready */
export const modelReady = writable(false);

/** Whether inference is currently running */
export const inferring = writable(false);

/** Last inference result from BDHModel.runToken() */
export const inferenceData = writable(null);

/** Current sigma data { "layer_head": Float32Array(N*N) } */
export const sigmaData = writable(null);

/** Token count since memory was last reset */
export const tokenCount = writable(0);

/**
 * Derived: flat 1024-element activation array for NeuronGrid (BDH side).
 * Merges both heads' x_sparse into one array.
 */
export const flatActivations = derived(
  [inferenceData, selectedLayer],
  ([$data, $layer]) => {
    if (!$data || !$data.layers[$layer]) return new Float32Array(TOTAL_NEURONS);
    const result = new Float32Array(TOTAL_NEURONS);
    $data.layers[$layer].x_last.forEach((head, h) => {
      result.set(head, h * N);
    });
    return result;
  }
);

/**
 * Derived: set of active neuron IDs for Graph Brain highlighting.
 */
export const activeNeuronIds = derived(
  [inferenceData, selectedLayer],
  ([$data, $layer]) => {
    if (!$data || !$data.layers[$layer]) return new Set();
    const ids = new Set();
    $data.layers[$layer].x_last.forEach((head, h) => {
      head.forEach((val, i) => {
        if (val > 1e-6) ids.add(h * N + i);
      });
    });
    return ids;
  }
);

/**
 * Derived: active neuron count and percentage.
 */
export const sparsityStats = derived(flatActivations, ($acts) => {
  let count = 0;
  for (let i = 0; i < $acts.length; i++) {
    if ($acts[i] > 1e-6) count++;
  }
  return {
    active: count,
    total: $acts.length,
    pct: ((count / $acts.length) * 100).toFixed(1),
  };
});

/**
 * Derived: attention scores matrix for the selected layer/head.
 */
export const attentionScores = derived(
  [inferenceData, selectedLayer, selectedHead],
  ([$data, $layer, $head]) => {
    if (!$data || !$data.layers[$layer]) return null;
    const ld = $data.layers[$layer];
    return {
      flat: ld.attn_scores,
      T: ld.T,
      head: $head,
    };
  }
);
