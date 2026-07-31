// src/privacy-audio.js
//
// PrivacyScreen — the AUDIO tier. Honestly bounded: there is NO on-device STT (transcription happens
// cloud-side, in the same call the audio streams to), so this head CANNOT catch exact spoken digits.
// What it produces is a coarse ACOUSTIC-CONTEXT signal — "steady, articulated speech that could be a
// read-out number sequence / financial framing" — that can only ever raise a `confirm`/hold, never a
// hard drop and never a category assertion. It is the weakest head on day one, behind the same interface
// so it can improve (a real acoustic model, later) without an API change.
//
// Deterministic: pure DSP over the PCM buffer (energy + zero-crossing rate + a flatness proxy), no model,
// no clock, no randomness. The score is hard-capped so audio alone can never escalate past `confirm`.

import { frameEnergy } from "./vad.js";
import { finiteNumber } from "./numeric.js";

/** Score ceiling for the coarse head — audio alone never escalates past `confirm` (see privacy.js bands). */
export const AUDIO_SCORE_CAP = 0.4;

/** The one tuple every non-scoring path returns. Built fresh each time — callers mutate the arrays. */
function silent() {
  return { sensitive: false, categories: [], score: 0, spans: [] };
}

/** A PCM buffer this head can read: a typed array or a plain array. Not a string, not `{length}`. */
function isPcmBuffer(pcm) {
  return Array.isArray(pcm) || (ArrayBuffer.isView(pcm) && !(pcm instanceof DataView));
}

/**
 * Triangular membership of `zcr` in the open band `(lo, hi)`, peaked at the centre — in [0,1], total.
 *
 * Two guards, and NEITHER is redundant:
 *
 *   * `!(hi > lo)` — a degenerate or inverted band makes the half-width 0 (or negative). `1 - d/0`
 *     is NaN at the centre point and -Infinity elsewhere; the NaN then survives `Math.min`, reads as
 *     `sensitive:false` (NaN > 0 is false) and reaches the verdict as a null score. A degenerate band
 *     must disable the tier LOUDLY (the constructor now rejects one) or score 0 — never NaN.
 *   * the edge short-circuit — at a band EDGE the triangle is mathematically 0, but in float64
 *     `1 - Math.abs(0.05 - 0.15) / 0.1` is 1.11e-16 rather than 0, so `score > 0` reported a
 *     detection on an out-of-band buffer; for other ordinary bands the same residual comes out
 *     NEGATIVE ([0.1,0.3], [0.3,0.7], [0.01,0.03], [0.02,0.06], [0.17,0.51] all do), and a negative
 *     score makes `makePrivacyVerdict` throw on a perfectly valid PCM buffer.
 *
 * The final clamp is the third: no float path may leave [0,1].
 */
function bandMembership(zcr, lo, hi) {
  if (!(hi > lo)) return 0;
  if (zcr <= lo || zcr >= hi) return 0;
  const m = 1 - Math.abs(zcr - (lo + hi) / 2) / ((hi - lo) / 2);
  return m <= 0 ? 0 : m >= 1 ? 1 : m;
}

/** Zero-crossing rate of a PCM buffer (proxy for voiced/unvoiced + articulation rate). */
function zeroCrossingRate(pcm) {
  const n = pcm.length;
  if (n < 2) return 0;
  let crossings = 0;
  for (let i = 1; i < n; i++) {
    if ((pcm[i - 1] < 0 && pcm[i] >= 0) || (pcm[i - 1] >= 0 && pcm[i] < 0)) crossings++;
  }
  return crossings / (n - 1);
}

/**
 * Coarse acoustic-context head. Bounded, category-free, and deterministic.
 *
 * The heuristic: sustained mid-energy speech with a moderate, regular zero-crossing rate is the acoustic
 * shape of someone articulating a sequence (reading a payment number or SSN aloud). We can't verify
 * the content, so we
 * only nudge toward `confirm`. Silence and noise-like buffers score ~0.
 */
export class PrivacyAudioHead {
  /**
   * @param {object} [opts]
   * @param {number} [opts.minEnergy=0.005] energy floor below which the buffer is treated as silence
   * @param {[number,number]} [opts.zcrBand=[0.05,0.25]] ZCR range typical of voiced articulated speech
   * @throws {RangeError} on a knob that cannot produce a score in [0, AUDIO_SCORE_CAP]
   */
  constructor({ minEnergy = 0.005, zcrBand = [0.05, 0.25] } = {}) {
    // Validate the knobs the way PrivacyScreen validates its bands (RangeError on confirm > drop):
    // an unvalidated knob here does not error, it silently produces a NON-FINITE or NEGATIVE score
    // that the tier then reads as "not sensitive", so the head disables itself without a sound.
    if (!finiteNumber(minEnergy) || minEnergy < 0) {
      throw new RangeError("PrivacyAudioHead: minEnergy must be a finite number >= 0");
    }
    if (!Array.isArray(zcrBand) || zcrBand.length !== 2) {
      throw new RangeError("PrivacyAudioHead: zcrBand must be a [lo, hi] pair");
    }
    const [lo, hi] = zcrBand;
    if (!finiteNumber(lo) || !finiteNumber(hi) || !(hi > lo)) {
      throw new RangeError("PrivacyAudioHead: require a finite zcrBand with lo < hi");
    }
    this.minEnergy = minEnergy;
    this.zcrBand = [lo, hi]; // copied: the band is read on every frame, so it is not the caller's to mutate
  }

  /**
   * @param {Float32Array|number[]} pcm
   * @returns {{ sensitive:boolean, categories:string[], score:number, spans:Array }}
   *   categories is ALWAYS [] (the head asserts no category); score ∈ [0, AUDIO_SCORE_CAP], always
   *   a finite number — an unreadable buffer is silent, never a NaN score.
   */
  screen(pcm) {
    // A real type check, not `!pcm || pcm.length === 0`: that was a falsy-and-length test, so a bare
    // number, a string, or any `{length: n}` object walked straight into frameEnergy and came back
    // with a NaN score. An input this head cannot read is silent, exactly like an empty one.
    if (!isPcmBuffer(pcm) || pcm.length === 0) return silent();
    const energy = frameEnergy(pcm);
    // A buffer carrying NaN/Infinity samples yields a non-finite energy, which would survive
    // `Math.min(1, energy / …)` as NaN. Unreadable, so: silent.
    if (!finiteNumber(energy) || energy <= 0 || energy < this.minEnergy) return silent();
    const [lo, hi] = this.zcrBand;
    // Triangular membership in the voiced-articulation ZCR band → [0,1], peak at the band center.
    const inBand = bandMembership(zeroCrossingRate(pcm), lo, hi);
    // Energy confidence saturates quickly once above the floor (sustained speech, not a transient).
    // `energy > 0` above keeps a zero floor from making this 0/0.
    const energyConf = Math.min(1, energy / (this.minEnergy * 10));
    const raw = inBand * energyConf;
    const score = Math.min(AUDIO_SCORE_CAP, raw * AUDIO_SCORE_CAP);
    return { sensitive: score > 0, categories: [], score, spans: [] };
  }
}
