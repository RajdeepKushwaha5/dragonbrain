/**
 * BDH ONNX inference wrapper for browser.
 *
 * Loads model.onnx via onnxruntime-web (WASM backend).
 * Provides:
 *   - runToken()     → sparse activations + attention scores
 *   - updateSigma()  → accumulate Hebbian memory
 *   - resetMemory()  → clear σ
 *
 * Falls back to mock mode if ONNX loading fails (for development).
 */

import {
  NH, N, TOTAL_NEURONS,
  extractLastToken, mergeHeads, updateSigma as updateSigmaMatrix,
  generateDenseReference, extractScoresMatrix,
} from './activation_math.js';

export class BDHModel {
  constructor() {
    this.session = null;
    this.sigma = {};              // { "layer_head": Float32Array(N*N) }
    this.nLayers = 2;
    this.mockMode = false;
    this.ready = false;
  }

  /**
   * Load the ONNX model. Falls back to mock mode on failure.
   * @param {string} modelPath - Path to model.onnx
   */
  async load(modelPath) {
    try {
      const ort = await import('onnxruntime-web');

      // Configure WASM backend
      ort.env.wasm.numThreads = 1;

      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
      });

      // Determine number of layers from output names
      const outputNames = this.session.outputNames;
      this.nLayers = (outputNames.length - 1) / 4; // logits + 4 per layer
      this.ready = true;
      console.log(`BDH model loaded: ${outputNames.length} outputs, ${this.nLayers} layers`);
    } catch (err) {
      console.warn('ONNX model not available, using mock mode:', err.message);
      this.mockMode = true;
      this.ready = true;
    }
  }

  /**
   * Run inference on a token buffer.
   * @param {number[]} tokenBuffer - Array of byte tokens (max ~128)
   * @returns {object} Parsed inference results
   */
  async runToken(tokenBuffer) {
    if (this.mockMode) return this._mockInference(tokenBuffer);

    const ort = await import('onnxruntime-web');
    const T = tokenBuffer.length;
    const tokens = new ort.Tensor(
      'int64',
      BigInt64Array.from(tokenBuffer.map(BigInt)),
      [1, T]
    );

    const results = await this.session.run({ tokens });
    return this._parseOutputs(results, T);
  }

  /**
   * Parse raw ONNX outputs into structured data.
   * @private
   */
  _parseOutputs(results, T) {
    const data = { layers: [], T };

    for (let l = 0; l < this.nLayers; l++) {
      const xFlat = results[`layer_${l}_x_sparse`].data;
      const yFlat = results[`layer_${l}_y_sparse`].data;
      const xyFlat = results[`layer_${l}_xy_sparse`].data;
      const attnFlat = results[`layer_${l}_attn_scores`].data;

      data.layers.push({
        x_last: extractLastToken(xFlat, T),
        y_last: extractLastToken(yFlat, T),
        xy_last: extractLastToken(xyFlat, T),
        attn_scores: attnFlat,
        T,
      });
    }

    return data;
  }

  /**
   * Generate mock inference data for development without ONNX.
   * Produces realistic-looking sparse activation patterns.
   * @private
   */
  _mockInference(tokenBuffer) {
    const T = tokenBuffer.length;
    const lastByte = T > 0 ? tokenBuffer[T - 1] : 0;
    const data = { layers: [], T };

    for (let l = 0; l < this.nLayers; l++) {
      const x_last = [];
      const y_last = [];
      const xy_last = [];

      for (let h = 0; h < NH; h++) {
        const xHead = new Float32Array(N);
        const yHead = new Float32Array(N);
        const xyHead = new Float32Array(N);

        // Generate sparse activations (~5% nonzero for x, ~3% for y)
        for (let i = 0; i < N; i++) {
          const hash = ((lastByte * 2654435761 + i * 340573321 + h * 123456789 + l * 987654321) >>> 0) / 4294967296;
          if (hash < 0.05) {
            xHead[i] = hash * 10;
          }
          const hashY = ((lastByte * 1597334677 + i * 789456123 + h * 456789123 + l * 321654987) >>> 0) / 4294967296;
          if (hashY < 0.03) {
            yHead[i] = hashY * 8;
          }
          xyHead[i] = xHead[i] * yHead[i];
        }

        x_last.push(xHead);
        y_last.push(yHead);
        xy_last.push(xyHead);
      }

      // Mock attention scores (T×T per head)
      const attnSize = NH * T * T;
      const attn_scores = new Float32Array(attnSize);
      for (let h = 0; h < NH; h++) {
        for (let row = 0; row < T; row++) {
          for (let col = 0; col < row; col++) {
            const idx = h * T * T + row * T + col;
            const dist = row - col;
            attn_scores[idx] = Math.max(0, Math.random() * (1 / (dist + 1)));
          }
        }
      }

      data.layers.push({ x_last, y_last, xy_last, attn_scores, T });
    }

    return data;
  }

  /**
   * Accumulate outer product into σ: sigma += outer(y, x) for each head.
   * @param {Float32Array[]} x_heads - Per-head x activations
   * @param {Float32Array[]} y_heads - Per-head y activations
   * @param {number} layer - Layer index
   */
  updateSigma(x_heads, y_heads, layer = 0) {
    for (let h = 0; h < x_heads.length; h++) {
      const key = `${layer}_${h}`;
      if (!this.sigma[key]) {
        this.sigma[key] = new Float32Array(N * N);
      }
      updateSigmaMatrix(this.sigma[key], x_heads[h], y_heads[h]);
    }
  }

  /**
   * Get sigma matrix for a given layer and head.
   * @param {number} layer
   * @param {number} head
   * @returns {Float32Array} N*N flat sigma matrix
   */
  getSigma(layer = 0, head = 0) {
    const key = `${layer}_${head}`;
    return this.sigma[key] || new Float32Array(N * N);
  }

  /** Clear all accumulated Hebbian memory. */
  resetMemory() {
    this.sigma = {};
  }
}
