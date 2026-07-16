// src/privacy-image.js
//
// PrivacyScreen — the IMAGE tier. This file ships the INTERFACE + the fail-closed lifecycle only; it
// NEVER imports onnxruntime-web or any model. The multi-MB quantized multi-label CNN is a
// product-supplied artifact, exactly like the pose landmarker (src/pose.js) and the Silero VAD slot
// (src/vad.js): percept-edge stays zero-runtime-dependency, and the model asset lives app-side.
//
// The app implements this adapter and hands it to PrivacyScreen ({ image }):
//
//   privacyImageAdapter = {
//     async load(): Promise<void>,                       // init the model (weights, WASM/onnxruntime-web)
//     async infer(frame): Promise<{                      // frame = { width, height, data }
//       categories: string[],                            // image-relevant privacy categories present
//       scores: number[],                                // per-category confidence, aligned to categories
//       boxes?: Array<{ start:{x,y}, end:{x,y} }>,       // OPTIONAL pixel boxes for redaction
//     }>,
//     version?: string,                                  // surfaced to the eval scorecard for attribution
//   }
//
// Because inference is async and only runs on egress-candidate frames (~0.25 fps), it has a
// few-hundred-ms budget. On timeout OR error we FAIL CLOSED: return a sensitive verdict so a model that
// hangs can never leak a frame. Over-masking is acceptable; a leaked frame is not.

import { PRIVACY_CATEGORIES } from "./privacy-constants.js";

/**
 * Documented adapter slot — the twin of SileroVadAdapter. It intentionally throws so `npm test` needs no
 * model download and a build that never screens images ships nothing extra. Products subclass or supply
 * their own object with the same shape.
 */
export class PrivacyImageAdapter {
  constructor(opts = {}) {
    this.opts = opts;
    this.version = opts.version ?? null;
  }
  // eslint-disable-next-line no-unused-vars
  async load() {
    throw new Error(
      "PrivacyImageAdapter is a documented adapter slot — wire onnxruntime-web + your privacy CNN to enable, or pass your own { load, infer } to PrivacyScreen({ image })."
    );
  }
  // eslint-disable-next-line no-unused-vars
  async infer(_frame) {
    throw new Error("PrivacyImageAdapter.infer not implemented (adapter slot).");
  }
}

/** The conservative verdict-partial returned whenever the image tier cannot produce a trustworthy answer. */
export function failClosedImage() {
  return { sensitive: true, categories: [], score: 1, spans: [], failed: true };
}

/**
 * Run the image tier under a timeout, failing closed on error/timeout.
 *
 * @param {object} adapter  the product-supplied { infer } (already load()'ed)
 * @param {object} frame    { width, height, data }
 * @param {object} opts
 * @param {number} opts.budgetMs  deadline for infer() (default 300)
 * @param {() => number} opts.clock  monotonic ms source; used ONLY for the deadline, never in the verdict
 * @returns {Promise<{sensitive:boolean,categories:string[],score:number,spans:Array,failed?:boolean}>}
 */
export async function runImageTier(adapter, frame, { budgetMs = 300, clock = null } = {}) {
  if (!adapter || typeof adapter.infer !== "function") {
    // No model wired → fail closed. The app must supply an adapter to screen images at all.
    return failClosedImage();
  }
  const now = clock || (() => Date.now());
  const started = now();
  let timer = null;
  const timeout = new Promise((resolve) => {
    // Wall-clock timeout guards a hung model. setTimeout ordering is deterministic under the budget.
    timer = setTimeout(() => resolve({ __timeout: true }), budgetMs);
  });
  try {
    const raw = await Promise.race([adapter.infer(frame), timeout]);
    if (raw && raw.__timeout) return failClosedImage();
    void started; // (latency is measured by the eval harness, not needed for the verdict)
    return normalizeImageResult(raw);
  } catch {
    // Any adapter throw (bad weights, OOM, corrupt frame) → fail closed.
    return failClosedImage();
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Normalize + validate an adapter result into a verdict-partial. A malformed result is treated as a
 * failure (fail closed) rather than trusted — the SDK cannot vouch for an asset it never shipped.
 */
function normalizeImageResult(raw) {
  if (!raw || !Array.isArray(raw.categories) || !Array.isArray(raw.scores)) {
    return failClosedImage();
  }
  const { categories, scores } = raw;
  if (categories.length !== scores.length) return failClosedImage();
  const cats = [];
  let score = 0;
  for (let i = 0; i < categories.length; i++) {
    const c = categories[i];
    const s = scores[i];
    if (!PRIVACY_CATEGORIES.has(c)) return failClosedImage(); // unknown label → don't trust the model
    if (typeof s !== "number" || !Number.isFinite(s) || s < 0 || s > 1) return failClosedImage();
    cats.push(c);
    if (s > score) score = s;
  }
  const uniqueCats = [...new Set(cats)].sort();
  // Pixel boxes are optional and app-redacted; pass them through as spans when present + well-formed.
  const spans = [];
  if (Array.isArray(raw.boxes)) {
    for (const b of raw.boxes) {
      if (b && b.start && b.end) spans.push({ start: b.start, end: b.end });
    }
  }
  return {
    sensitive: uniqueCats.length > 0,
    categories: uniqueCats,
    score: uniqueCats.length > 0 ? score : 0,
    spans,
  };
}
