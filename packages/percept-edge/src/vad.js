// src/vad.js
//
// Voice-activity endpointer. The shipped, tested path is a pure energy-
// threshold endpointer: it detects when speech starts (energy rises above a
// threshold) and emits an *endpoint* after a run of low-energy frames, i.e.
// the speaker has stopped. No model, no native deps, deterministic.
//
// Silero-VAD (ONNX, via onnxruntime-web / transformers.js) is a documented
// adapter slot — see SileroVadAdapter below. It is NOT exercised by the test
// suite and requires no model download for the energy fallback to work.

/**
 * Compute short-time energy (mean of squares) of a frame of PCM samples.
 * Samples may be float (-1..1) or already squared/energy values >= 0; we just
 * take mean of squares, so float PCM in [-1, 1] yields energy in [0, 1].
 * @param {Float32Array|number[]} samples
 * @returns {number}
 */
export function frameEnergy(samples) {
  const n = samples.length;
  if (n === 0) return 0;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const s = samples[i];
    acc += s * s;
  }
  return acc / n;
}

/**
 * Streaming energy-threshold endpointer.
 *
 * Feed it per-frame energy (or raw PCM frames via `pushSamples`). It tracks
 * whether speech is active and fires an endpoint once `hangoverFrames`
 * consecutive low-energy frames follow a speech run.
 *
 * Options:
 *   threshold      energy above this counts as speech         (default 0.01)
 *   hangoverFrames silence frames after speech before endpoint (default 8)
 *   minSpeechFrames minimum speech frames to count as an utterance (default 3)
 */
export class EnergyVad {
  constructor({
    threshold = 0.01,
    hangoverFrames = 8,
    minSpeechFrames = 3,
  } = {}) {
    this.threshold = threshold;
    this.hangoverFrames = hangoverFrames;
    this.minSpeechFrames = minSpeechFrames;
    this.reset();
  }

  reset() {
    this._speaking = false;
    this._speechRun = 0; // consecutive speech frames in current utterance
    this._silenceRun = 0; // consecutive silence frames since last speech
    this._frameIndex = -1;
    return this;
  }

  get speaking() {
    return this._speaking;
  }

  /**
   * Push one frame's energy. Returns an event object:
   *   { speech: boolean, started: boolean, endpoint: boolean, frame: number }
   * `started` is true on the rising edge (silence -> speech); `endpoint` is
   * true on the frame the utterance is considered finished.
   */
  pushEnergy(energy) {
    this._frameIndex += 1;
    const isSpeech = energy >= this.threshold;
    let started = false;
    let endpoint = false;

    if (isSpeech) {
      if (!this._speaking) {
        this._speaking = true;
        this._speechRun = 0;
        started = true;
      }
      this._speechRun += 1;
      this._silenceRun = 0;
    } else if (this._speaking) {
      this._silenceRun += 1;
      // Endpoint only if we accumulated a real utterance and have now seen
      // enough trailing silence (hangover).
      if (
        this._silenceRun >= this.hangoverFrames &&
        this._speechRun >= this.minSpeechFrames
      ) {
        this._speaking = false;
        endpoint = true;
        this._speechRun = 0;
        this._silenceRun = 0;
      } else if (this._silenceRun >= this.hangoverFrames) {
        // Too short to be an utterance: drop it silently, re-arm.
        this._speaking = false;
        this._speechRun = 0;
        this._silenceRun = 0;
      }
    }

    return {
      speech: isSpeech,
      started,
      endpoint,
      frame: this._frameIndex,
    };
  }

  /**
   * Push one frame of raw PCM samples (Float32Array). Convenience wrapper that
   * computes energy then delegates to pushEnergy.
   */
  pushSamples(samples) {
    return this.pushEnergy(frameEnergy(samples));
  }
}

/**
 * Run the energy endpointer over a whole sequence of per-frame energies and
 * return the frame indices at which an endpoint fired. Convenience / batch API
 * used by fixtures and tests.
 * @param {number[]} energies
 * @param {object} [opts] forwarded to EnergyVad
 * @returns {number[]} endpoint frame indices
 */
export function endpointEnergySequence(energies, opts = {}) {
  const vad = new EnergyVad(opts);
  const endpoints = [];
  for (const e of energies) {
    const ev = vad.pushEnergy(e);
    if (ev.endpoint) endpoints.push(ev.frame);
  }
  return endpoints;
}

/**
 * Documented adapter slot for Silero-VAD (ONNX). Not used by the tested path
 * and intentionally unimplemented here — wiring it requires onnxruntime-web /
 * transformers.js plus the shipped Silero model. Implement `process(frame)` to
 * return a speech probability and feed that (thresholded) into the same
 * started/endpoint state machine as EnergyVad.
 *
 * Keeping it as a stub means `npm test` needs no model download or native dep.
 */
export class SileroVadAdapter {
  constructor(opts = {}) {
    this.opts = opts;
  }
  // eslint-disable-next-line no-unused-vars
  async load(modelUrl) {
    throw new Error(
      "SileroVadAdapter is a documented adapter slot — wire onnxruntime-web + the Silero model to enable. Use EnergyVad for the standalone path."
    );
  }
  // eslint-disable-next-line no-unused-vars
  async process(_frameSamples) {
    throw new Error("SileroVadAdapter.process not implemented (adapter slot).");
  }
}
