/**
 * Byte-level tokenizer for BDH (vocab_size=256).
 * Each character maps to its UTF-8 byte value(s).
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * Convert text to byte-level token array.
 * @param {string} text
 * @returns {Uint8Array}
 */
export function tokenize(text) {
  return encoder.encode(text);
}

/**
 * Convert byte tokens back to text.
 * @param {Uint8Array|number[]} tokens
 * @returns {string}
 */
export function detokenize(tokens) {
  return decoder.decode(new Uint8Array(tokens));
}

/**
 * Get the display character for a token byte.
 * Common control characters get readable symbols; others show hex.
 * @param {number} byte
 * @returns {string}
 */
export function tokenToChar(byte) {
  if (byte >= 32 && byte < 127) return String.fromCharCode(byte);
  // Readable symbols for common control characters
  switch (byte) {
    case 0:   return '\u2400';  // ␀ NUL
    case 9:   return '\u21E5';  // ⇥ TAB
    case 10:  return '\u23CE';  // ⏎ LF
    case 13:  return '\u240D';  // ␍ CR
    case 32:  return '\u00B7';  // · SPACE (already handled above, but just in case)
    case 127: return '\u2421';  // ␡ DEL
  }
  return `0x${byte.toString(16).padStart(2, '0')}`;
}
