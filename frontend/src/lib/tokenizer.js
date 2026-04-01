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
 * Non-printable characters are shown as their hex code.
 * @param {number} byte
 * @returns {string}
 */
export function tokenToChar(byte) {
  if (byte >= 32 && byte < 127) return String.fromCharCode(byte);
  return `\\x${byte.toString(16).padStart(2, '0')}`;
}
