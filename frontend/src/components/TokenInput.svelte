<script>
  import { createEventDispatcher } from 'svelte';
  import { inputText, tokenBuffer, inferring } from '../lib/stores.js';
  import { tokenize, tokenToChar } from '../lib/tokenizer.js';

  const dispatch = createEventDispatcher();

  /** External text set by demo mode — synced to textarea */
  export let externalText = null;

  let textValue = '';

  // Sync from external (demo mode)
  $: if (externalText !== null && externalText !== textValue) {
    textValue = externalText;
  }

  function handleInput() {
    inputText.set(textValue);
    const bytes = tokenize(textValue);
    // ONNX model trained with max_seq_len=128; limit to avoid WASM OOM on attention tensor
    const tail = Array.from(bytes.slice(-128));
    tokenBuffer.set(tail);
    dispatch('input', { text: textValue, tokens: tail });
  }

  function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleInput();
    }
  }

  $: visibleTokens = textValue ? Array.from(tokenize(textValue)) : [];
  $: byteCount = visibleTokens.length;
</script>

<div class="input-card">
  <div class="input-header">
    <div class="input-title">
      <span>Input</span>
    </div>
    <div class="input-meta">
      {#if byteCount > 0}
        <span class="byte-count">{byteCount} bytes</span>
      {/if}
      {#if $inferring}
        <span class="infer-badge">
          <span class="infer-dot"></span>
          Processing…
        </span>
      {/if}
    </div>
  </div>

  <div class="input-body">
    <div class="input-left">
      <textarea
        id="text-input"
        bind:value={textValue}
        on:input={handleInput}
        on:keydown={handleKeydown}
        placeholder="Type text to explore BDH internals — e.g. The dollar rose against the euro..."
        rows="3"
        spellcheck="false"
        aria-label="Input text for BDH inference"
      ></textarea>
    </div>

    <div class="input-right">
      {#if visibleTokens.length > 0}
        <div class="token-stream">
          <span class="token-label">Byte tokens</span>
          <div class="token-list">
            {#each visibleTokens as byte, i}
              <span class="token" title="byte {byte} (0x{byte.toString(16).padStart(2, '0')})">
                {tokenToChar(byte)}
              </span>
            {/each}
          </div>
        </div>
      {:else}
        <div class="token-empty">
          <span class="token-label">Byte tokens</span>
          <span class="empty-hint">Type something to see byte-level tokenization…</span>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .input-card {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  .input-card:focus-within {
    border-color: var(--border-hover);
  }

  .input-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
  }

  .input-title {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
  }

  .input-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .byte-count {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    color: var(--text-dim);
    padding: 0.15rem 0.5rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
  }

  .infer-badge {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.74rem;
    color: var(--accent);
    font-weight: 500;
  }

  .infer-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
    animation: spin 1s linear infinite;
  }

  .input-body {
    display: flex;
    gap: 1rem;
    align-items: stretch;
  }

  .input-left {
    flex: 1;
    min-width: 0;
  }

  .input-right {
    flex: 1;
    min-width: 0;
  }

  textarea {
    width: 100%;
    height: 100%;
    min-height: 5.5rem;
    padding: 0.75rem 1rem;
    background: var(--bg-input);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 0.95rem;
    line-height: 1.6;
    resize: vertical;
    outline: none;
    box-sizing: border-box;
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
  }

  textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }

  textarea::placeholder {
    color: var(--text-muted);
    font-size: 0.9rem;
  }

  .token-stream, .token-empty {
    padding: 0.5rem 0.7rem;
    background: rgba(255, 255, 255, 0.025);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
  }

  .token-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 0.3rem;
    flex-shrink: 0;
  }

  .token-list {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    flex: 1;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border-hover) transparent;
  }

  .token-list::-webkit-scrollbar {
    width: 4px;
  }

  .token-list::-webkit-scrollbar-track {
    background: transparent;
  }

  .token-list::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 2px;
  }

  .token {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    background: rgba(91, 141, 239, 0.1);
    color: var(--accent-bright);
    padding: 1px 6px;
    border-radius: 4px;
    cursor: default;
    border: 1px solid rgba(74, 108, 247, 0.15);
    transition: background var(--transition-fast);
  }

  .token:hover {
    background: rgba(74, 108, 247, 0.2);
  }

  .empty-hint {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-style: italic;
  }

  @media (max-width: 640px) {
    .input-body {
      flex-direction: column;
    }

    .input-card {
      padding: 0.8rem;
    }

    textarea {
      font-size: 0.85rem;
      min-height: auto;
    }

    .token-stream, .token-empty {
      height: auto;
      max-height: 6rem;
    }
  }
</style>
