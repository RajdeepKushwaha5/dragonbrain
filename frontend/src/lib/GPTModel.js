/**
 * GPT (Transformer) ONNX inference wrapper for browser.
 *
 * Loads transformer.onnx via onnxruntime-web (WASM backend).
 * Provides real MLP hidden activations for side-by-side comparison
 * with BDH's sparse activations.
 *
 * Architecture: 2-layer GPT with GELU MLP (hidden_dim=256).
 * Expected density: ~97-100% (GELU activates nearly all neurons).
 */

export class GPTModel {
  constructor() {
    this.session = null;
    this.ready = false;
    this._ort = null;
    this.nLayers = 2;
    this.hiddenDim = 256; // D=64 × mlp_ratio=4
  }

  /**
   * Load the transformer ONNX model.
   * Fetches model + external data explicitly to avoid ort URL-resolution issues
   * when creating a second InferenceSession in the same runtime.
   * @param {string} modelPath - Path to transformer.onnx
   * @param {object} ort - Cached onnxruntime-web module (shared with BDH)
   */
  async load(modelPath, ort) {
    try {
      this._ort = ort;

      // Fetch model proto and external weights as ArrayBuffers
      const [modelResp, dataResp] = await Promise.all([
        fetch(modelPath),
        fetch(modelPath + '.data'),
      ]);
      if (!modelResp.ok) throw new Error(`Failed to fetch ${modelPath}: ${modelResp.status}`);
      if (!dataResp.ok) throw new Error(`Failed to fetch ${modelPath}.data: ${dataResp.status}`);

      const modelBuffer = await modelResp.arrayBuffer();
      const dataBuffer = await dataResp.arrayBuffer();

      // Derive the external data filename from the path (e.g. "transformer.onnx.data")
      const dataFileName = modelPath.split('/').pop() + '.data';

      this.session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        externalData: [{ data: dataBuffer, path: dataFileName }],
      });

      // Detect hidden dim from output shape metadata
      const outputNames = this.session.outputNames;
      this.nLayers = outputNames.length - 1; // logits + 1 per layer
      this.ready = true;
      console.log(`GPT model loaded: ${outputNames.length} outputs, ${this.nLayers} layers`);
    } catch (err) {
      console.warn('GPT model not available:', err.message);
      this.ready = false;
    }
  }

  /**
   * Run inference and extract MLP hidden activations.
   * @param {number[]} tokenBuffer - Array of byte tokens
   * @returns {object} { activations: { 0: Float32Array, 1: Float32Array }, T }
   */
  async runToken(tokenBuffer) {
    if (!this.ready) return null;

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
   * Parse ONNX outputs — extract last-timestep MLP activations and logits.
   * @private
   */
  _parseOutputs(results, T) {
    const activations = {};

    for (let l = 0; l < this.nLayers; l++) {
      const key = `mlp_act_${l}`;
      const actRaw = results[key]?.data;
      if (actRaw) {
        // Extract last timestep: shape is (1, T, hidden_dim)
        const start = (T - 1) * this.hiddenDim;
        activations[l] = actRaw.slice(start, start + this.hiddenDim);
      }
    }

    // Extract logits for prediction comparison
    let logits = null;
    const logitsRaw = results['logits']?.data;
    if (logitsRaw) {
      const vocabSize = 256;
      const start = (T - 1) * vocabSize;
      logits = logitsRaw.slice(start, start + vocabSize);
    }

    return { activations, logits, T };
  }

  topKPredictions(logits, k = 5) {
    if (!logits || logits.length === 0) return [];
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
    for (let i = 0; i < probs.length; i++) probs[i] /= sumExp;
    const indexed = Array.from(probs).map((p, i) => ({ token: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
  }
}
