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
  NH, N, D, TOTAL_NEURONS,
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
    this._ort = null;            // cached onnxruntime-web module
    this.weights = null;         // { encoder: Float32Array, lm_head: Float32Array }
  }

  /**
   * Load the ONNX model. Falls back to mock mode on failure.
   * @param {string} modelPath - Path to model.onnx
   */
  async load(modelPath) {
    try {
      this._ort = await import('onnxruntime-web');
      const ort = this._ort;

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
   * Load extracted weight matrices (encoder, lm_head) for σ-modulated inference.
   * Binary format: encoder(1024*64 floats) + lm_head(64*256 floats), float32 LE.
   * @param {string} weightsPath - Path to bdh_weights.bin
   */
  async loadWeights(weightsPath) {
    try {
      const resp = await fetch(weightsPath);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const all = new Float32Array(buf);

      const encoderSize = TOTAL_NEURONS * D;  // 1024 * 64 = 65536
      const lmHeadSize = D * 256;             // 64 * 256 = 16384

      this.weights = {
        encoder: all.slice(0, encoderSize),
        lm_head: all.slice(encoderSize, encoderSize + lmHeadSize),
      };
      console.log(`BDH weights loaded: encoder(${encoderSize}), lm_head(${lmHeadSize})`);
    } catch (err) {
      console.warn('Could not load BDH weights for σ-modulated inference:', err.message);
    }
  }

  /**
   * Compute σ-modulated logit corrections using accumulated Hebbian memory.
   *
   * Implements the paper's key equation:
   *   a*_t = σ · x_t          → (N,) σ-weighted activation
   *   correction_D = E^T · a* → (D,) back to embedding space
   *   logit_bias = correction · W_lm → (vocab,) logit adjustment
   *
   * This is the mechanism by which accumulated σ influences predictions —
   * the core "inference-time learning" property of BDH.
   *
   * @param {Float32Array[]} x_heads - Per-head x activations for last token
   * @param {number} layer - Layer index
   * @returns {Float32Array|null} Logit bias (256), or null if weights not loaded
   */
  computeSigmaLogits(x_heads, layer = 0) {
    if (!this.weights) return null;

    const { encoder, lm_head } = this.weights;
    const correction = new Float32Array(D);

    for (let h = 0; h < NH; h++) {
      const x = x_heads[h];
      const sigma = this.getSigma(layer, h);

      // a_sigma = σ · x → (N,) : how strongly past co-activations match current x
      const a_sigma = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let sum = 0;
        const row = i * N;
        for (let j = 0; j < N; j++) {
          sum += sigma[row + j] * x[j];
        }
        // ReLU — paper requires positive activations
        a_sigma[i] = sum > 0 ? sum : 0;
      }

      // Project through encoder: correction += E[h*N:(h+1)*N, :]^T · a_sigma
      // encoder is (nh*N, D) row-major, so encoder[(h*N+i)*D + d]
      const hOffset = h * N;
      for (let d = 0; d < D; d++) {
        let sum = 0;
        for (let i = 0; i < N; i++) {
          sum += encoder[(hOffset + i) * D + d] * a_sigma[i];
        }
        correction[d] += sum;
      }
    }

    // Project to vocab: logit_bias = correction · lm_head
    // lm_head is (D, 256) row-major
    const logitBias = new Float32Array(256);
    for (let v = 0; v < 256; v++) {
      let sum = 0;
      for (let d = 0; d < D; d++) {
        sum += correction[d] * lm_head[d * 256 + v];
      }
      logitBias[v] = sum;
    }

    return logitBias;
  }

  /**
   * Run inference on a token buffer.
   * @param {number[]} tokenBuffer - Array of byte tokens (max ~128)
   * @returns {object} Parsed inference results
   */
  async runToken(tokenBuffer) {
    if (this.mockMode) return this._mockInference(tokenBuffer);

    const ort = this._ort;
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

    // Extract logits for text generation (last timestep only)
    const logitsRaw = results['logits']?.data;
    if (logitsRaw) {
      const vocabSize = 256;
      const start = (T - 1) * vocabSize;
      data.logits = logitsRaw.slice(start, start + vocabSize);
    }

    for (let l = 0; l < this.nLayers; l++) {
      const xFlat = results[`layer_${l}_x_sparse`]?.data;
      const yFlat = results[`layer_${l}_y_sparse`]?.data;
      const xyFlat = results[`layer_${l}_xy_sparse`]?.data;
      const attnFlat = results[`layer_${l}_attn_scores`]?.data;

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
   * Get top-k predicted next tokens from logits.
   * @param {Float32Array} logits - Raw logits (vocab_size=256)
   * @param {number} k - Number of top predictions
   * @returns {Array<{token: number, prob: number}>}
   */
  topKPredictions(logits, k = 5) {
    if (!logits || logits.length === 0) return [];

    // Softmax
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) maxVal = logits[i];
    }
    const probs = new Float32Array(logits.length);
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - maxVal);
      sumExp += probs[i];
    }
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
    }

    // Find top-k
    const indexed = Array.from(probs).map((p, i) => ({ token: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
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
