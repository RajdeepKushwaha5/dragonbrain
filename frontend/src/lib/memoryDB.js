/**
 * IndexedDB persistence layer for cross-session Hebbian memory.
 * Stores σ matrices, token counts, and session metadata so the
 * BDH model's learned patterns survive page reloads.
 */

const DB_NAME = 'DragonBrainMemory';
const DB_VERSION = 1;
const STORE_NAME = 'brain_state';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Save the model's current σ state to IndexedDB.
 * @param {object} sigmaObj - { "layer_head": Float32Array(N*N) }
 * @param {number} totalTokens - Cumulative token count
 * @param {number} sessionCount - Number of sessions
 */
export async function saveBrainState(sigmaObj, totalTokens, sessionCount) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  const serialized = {};
  for (const [key, arr] of Object.entries(sigmaObj)) {
    serialized[key] = Array.from(arr);
  }

  store.put({
    sigma: serialized,
    totalTokens,
    sessionCount,
    timestamp: Date.now(),
  }, 'current');

  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Load saved σ state from IndexedDB.
 * @returns {object|null} { sigma, totalTokens, sessionCount, timestamp } or null
 */
export async function loadBrainState() {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.get('current');

    return new Promise((resolve) => {
      req.onsuccess = () => {
        const data = req.result;
        if (!data) return resolve(null);

        const sigma = {};
        for (const [key, arr] of Object.entries(data.sigma)) {
          sigma[key] = new Float32Array(arr);
        }

        resolve({
          sigma,
          totalTokens: data.totalTokens,
          sessionCount: data.sessionCount,
          timestamp: data.timestamp,
        });
      };
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

/**
 * Clear all saved brain state.
 */
export async function clearBrainState() {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).delete('current');
    return new Promise((resolve) => {
      tx.oncomplete = resolve;
      tx.onerror = () => resolve();
    });
  } catch {
    // Ignore errors when clearing
  }
}

/**
 * Compute a scalar "strength" metric for the σ state.
 * Returns the RMS of all σ entries across all heads/layers.
 */
export function computeSigmaStrength(sigmaObj) {
  let sumSq = 0;
  let n = 0;
  for (const arr of Object.values(sigmaObj)) {
    for (let i = 0; i < arr.length; i++) {
      sumSq += arr[i] * arr[i];
      n++;
    }
  }
  return n > 0 ? Math.sqrt(sumSq / n) : 0;
}

/**
 * Format a timestamp as a human-readable relative time.
 */
export function formatTimeAgo(timestamp) {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
