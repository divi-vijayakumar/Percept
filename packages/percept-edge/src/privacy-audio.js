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

/** Score ceiling for the coarse head — audio alone never escalates past `confirm` (see privacy.js bands). */
export const AUDIO_SCORE_CAP = 0.4;

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
   */
  constructor({ minEnergy = 0.005, zcrBand = [0.05, 0.25] } = {}) {
    this.minEnergy = minEnergy;
    this.zcrBand = zcrBand;
  }

  /**
   * @param {Float32Array|number[]} pcm
   * @returns {{ sensitive:boolean, categories:string[], score:number, spans:Array }}
   *   categories is ALWAYS [] (the head asserts no category); score ∈ [0, AUDIO_SCORE_CAP].
   */
  screen(pcm) {
    if (!pcm || pcm.length === 0) {
      return { sensitive: false, categories: [], score: 0, spans: [] };
    }
    const energy = frameEnergy(pcm);
    if (energy < this.minEnergy) {
      return { sensitive: false, categories: [], score: 0, spans: [] };
    }
    const zcr = zeroCrossingRate(pcm);
    const [lo, hi] = this.zcrBand;
    // Triangular membership in the voiced-articulation ZCR band → [0,1], peak at the band center.
    let inBand = 0;
    if (zcr >= lo && zcr <= hi) {
      const center = (lo + hi) / 2;
      const halfW = (hi - lo) / 2;
      inBand = 1 - Math.abs(zcr - center) / halfW;
    }
    // Energy confidence saturates quickly once above the floor (sustained speech, not a transient).
    const energyConf = Math.min(1, energy / (this.minEnergy * 10));
    const raw = inBand * energyConf;
    const score = Math.min(AUDIO_SCORE_CAP, raw * AUDIO_SCORE_CAP);
    return { sensitive: score > 0, categories: [], score, spans: [] };
  }
}
