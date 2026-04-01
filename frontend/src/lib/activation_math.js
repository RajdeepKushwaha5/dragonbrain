/**
 * Math utilities for BDH activation processing.
 * Runs entirely in the browser — no server needed.
 */

/** Model constants for tiny BDH config */
export const NH = 2;      // number of heads
export const N = 512;     // neurons per head
export const D = 64;      // embedding dimension
export const TOTAL_NEURONS = NH * N;  // 1024

/**
 * Extract the last token's activations from a flat ONNX output tensor.
 *
 * ONNX output is flattened (B * nh * T * N). We want the last timestep.
 *
 * @param {Float32Array} flat - Flattened (1, nh, T, N) tensor
 * @param {number} T - Sequence length
 * @returns {Float32Array[]} Array of per-head activation vectors [head0(N), head1(N)]
 */
export function extractLastToken(flat, T) {
  const result = [];
  for (let h = 0; h < NH; h++) {
    const offset = (h * T + (T - 1)) * N;
    result.push(flat.slice(offset, offset + N));
  }
  return result;
}

/**
 * Merge per-head activations into a single flat array for grid display.
 * @param {Float32Array[]} heads - [head0(N), head1(N)]
 * @returns {Float32Array} Flat array of TOTAL_NEURONS values
 */
export function mergeHeads(heads) {
  const result = new Float32Array(TOTAL_NEURONS);
  heads.forEach((head, h) => {
    result.set(head, h * N);
  });
  return result;
}

/**
 * Count active (nonzero) neurons.
 * @param {Float32Array} activations
 * @returns {{ count: number, total: number, pct: number }}
 */
export function countActive(activations) {
  let count = 0;
  for (let i = 0; i < activations.length; i++) {
    if (activations[i] > 1e-6) count++;
  }
  return {
    count,
    total: activations.length,
    pct: (count / activations.length) * 100,
  };
}

/**
 * Update the Hebbian σ matrix with a new outer product: σ += outer(y, x).
 * Optimized to skip zero y rows (exploiting sparsity).
 *
 * @param {Float32Array} sigma - N*N flat sigma matrix (mutated in place)
 * @param {Float32Array} x - (N,) x_sparse activations for one head
 * @param {Float32Array} y - (N,) y_sparse activations for one head
 */
export function updateSigma(sigma, x, y) {
  for (let i = 0; i < N; i++) {
    if (y[i] < 1e-6) continue; // skip zero rows (sparsity optimization)
    const rowOffset = i * N;
    for (let j = 0; j < N; j++) {
      sigma[rowOffset + j] += y[i] * x[j];
    }
  }
}

/**
 * Extract top-k neuron indices by cumulative sigma activity.
 * Used to select which neurons to display in the 64×64 heatmap.
 *
 * @param {Float32Array} sigma - N*N flat sigma matrix
 * @param {number} topK - Number of neurons to select
 * @returns {number[]} Indices of top-k most active neurons
 */
export function topNeuronsBySigma(sigma, topK = 64) {
  // Compute total absolute sigma per row
  const rowSums = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    const base = i * N;
    for (let j = 0; j < N; j++) {
      sum += Math.abs(sigma[base + j]);
    }
    rowSums[i] = sum;
  }

  // Get top-k indices
  const indices = Array.from({ length: N }, (_, i) => i);
  indices.sort((a, b) => rowSums[b] - rowSums[a]);
  return indices.slice(0, topK);
}

/**
 * Extract submatrix for heatmap display.
 * @param {Float32Array} sigma - N*N flat sigma matrix
 * @param {number[]} indices - Selected neuron indices
 * @returns {Float32Array} k*k flat submatrix
 */
export function extractSubmatrix(sigma, indices) {
  const k = indices.length;
  const sub = new Float32Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      sub[i * k + j] = sigma[indices[i] * N + indices[j]];
    }
  }
  return sub;
}

/**
 * Generate a fake "dense transformer" activation pattern for comparison.
 * In a real transformer, ~97% of neurons are active after GELU.
 * We simulate this with a deterministic pattern seeded by token values.
 *
 * @param {Uint8Array} tokenBuffer - Input tokens
 * @returns {Float32Array} 1024-element activation array (~97% nonzero)
 */
export function generateDenseReference(tokenBuffer) {
  const result = new Float32Array(TOTAL_NEURONS);
  // Use last token as seed for deterministic pattern
  const seed = tokenBuffer.length > 0 ? tokenBuffer[tokenBuffer.length - 1] : 0;

  for (let i = 0; i < TOTAL_NEURONS; i++) {
    // ~97% of neurons active, with varied magnitudes
    const hash = ((seed * 2654435761 + i * 340573321) >>> 0) / 4294967296;
    if (hash < 0.97) {
      result[i] = 0.1 + hash * 0.6; // range [0.1, 0.7]
    }
    // else: stays at 0 (~3% inactive)
  }
  return result;
}

/**
 * Extract attention scores for the last token from flat ONNX output.
 * @param {Float32Array} flat - Flattened (1, nh, T, T) tensor
 * @param {number} T - Sequence length
 * @param {number} head - Head index
 * @returns {Float32Array} T-length array of attention scores to previous tokens
 */
export function extractAttentionRow(flat, T, head) {
  // attn_scores shape: (1, nh, T, T)
  // We want [0, head, T-1, :] which is the last token's attention to all past tokens
  const offset = head * T * T + (T - 1) * T;
  return flat.slice(offset, offset + T);
}

/**
 * Extract full T×T scores matrix for a head.
 * @param {Float32Array} flat - Flattened (1, nh, T, T) tensor
 * @param {number} T - Sequence length
 * @param {number} head - Head index
 * @returns {Float32Array} T*T flat matrix
 */
export function extractScoresMatrix(flat, T, head) {
  const offset = head * T * T;
  return flat.slice(offset, offset + T * T);
}
