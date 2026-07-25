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
 * Mean-square energy of signed 16-bit PCM, normalized to [0, 1].
 * This is the byte-domain twin of Python runtime/audio.py `pcm_energy`.
 * @param {Int16Array|number[]} samples
 * @returns {number}
 */
export function pcm16Energy(samples) {
  const n = samples.length;
  if (n === 0) return 0;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const s = samples[i] / 32768;
    acc += s * s;
  }
  return Math.min(1, acc / n);
}

/**
 * Bounded ordered buffer for transport-neutral audio units.
 *
 * A unit is `{seq,t_start,t_end,energy,sample_rate,channels,direction,participant_id,boundaries}`.
 * `push` explicitly reports duplicate/reorder/missing/gap/overlap/format-change outcomes, matching
 * Python `AudioWindow.push_unit`; no image/frame carrier is involved.
 */
export class OrderedAudioBuffer {
  constructor({ seconds = 4, maxUnits = 4096 } = {}) {
    this.seconds = Math.max(0.5, seconds);
    this.maxUnits = Math.max(1, Math.trunc(maxUnits));
    this._units = [];
    this._maxSeq = null;
  }

  push(unit) {
    if (!Number.isInteger(unit.seq) || unit.seq < 0) {
      throw new Error("audio unit seq must be a non-negative integer");
    }
    if (!Number.isFinite(unit.t_start) || !Number.isFinite(unit.t_end)) {
      throw new Error("audio unit times must be finite");
    }
    if (unit.t_end < unit.t_start) throw new Error("audio unit t_end must be >= t_start");
    if (
      unit.energy != null &&
      (!Number.isFinite(unit.energy) || unit.energy < 0 || unit.energy > 1)
    ) {
      throw new Error("audio unit energy must be finite and within [0, 1]");
    }
    if (
      unit.sample_rate != null &&
      (!Number.isInteger(unit.sample_rate) || unit.sample_rate <= 0)
    ) {
      throw new Error("audio unit sample_rate must be positive when known");
    }
    if (unit.channels != null && (!Number.isInteger(unit.channels) || unit.channels <= 0)) {
      throw new Error("audio unit channels must be positive when known");
    }
    if (
      unit.direction != null &&
      !["input", "output", "mixed", "unknown"].includes(unit.direction)
    ) {
      throw new Error("audio unit direction is invalid");
    }
    if (unit.participant_id != null && typeof unit.participant_id !== "string") {
      throw new Error("audio unit participant_id must be a string when known");
    }
    if (this._units.some((existing) => existing.seq === unit.seq)) {
      return this._result({ accepted: false, duplicate: true });
    }

    const previousMaxSeq = this._maxSeq;
    const reordered = previousMaxSeq !== null && unit.seq < previousMaxSeq;
    const missing =
      previousMaxSeq !== null && unit.seq > previousMaxSeq
        ? Math.max(0, unit.seq - previousMaxSeq - 1)
        : 0;

    let index = 0;
    while (
      index < this._units.length &&
      (this._units[index].t_start < unit.t_start ||
        (this._units[index].t_start === unit.t_start && this._units[index].seq < unit.seq))
    ) {
      index++;
    }
    const previous = index > 0 ? this._units[index - 1] : null;
    const following = index < this._units.length ? this._units[index] : null;
    let gap_s = 0;
    let overlap_s = 0;
    if (previous) {
      if (unit.t_start > previous.t_end) gap_s = unit.t_start - previous.t_end;
      else if (unit.t_start < previous.t_end) overlap_s = previous.t_end - unit.t_start;
    }
    if (following) {
      if (unit.t_end < following.t_start) gap_s = Math.max(gap_s, following.t_start - unit.t_end);
      else if (unit.t_end > following.t_start) {
        overlap_s = Math.max(overlap_s, unit.t_end - following.t_start);
      }
    }
    const formatDiff = (other) =>
      Boolean(
        other &&
          ((unit.sample_rate != null &&
            other.sample_rate != null &&
            unit.sample_rate !== other.sample_rate) ||
            (unit.channels != null &&
              other.channels != null &&
              unit.channels !== other.channels))
      );
    const format_changed = formatDiff(previous) || formatDiff(following);

    this._units.splice(index, 0, {
      direction: "unknown",
      participant_id: null,
      boundaries: [],
      ...unit,
    });
    this._maxSeq = previousMaxSeq === null ? unit.seq : Math.max(previousMaxSeq, unit.seq);

    const newestEnd = Math.max(...this._units.map((row) => row.t_end));
    const cutoff = newestEnd - this.seconds;
    let expired = 0;
    while (this._units.length > 1) {
      const staleIndex = this._units.findIndex((retained) => retained.t_end < cutoff);
      if (staleIndex < 0) break;
      this._units.splice(staleIndex, 1);
      expired++;
    }
    let evicted = 0;
    while (this._units.length > 1 && this._units.length > this.maxUnits) {
      this._units.shift();
      evicted++;
    }
    return this._result({
      accepted: true,
      reordered,
      missing,
      gap_s,
      overlap_s,
      format_changed,
      expired,
      evicted,
    });
  }

  _result(overrides = {}) {
    return {
      accepted: true,
      duplicate: false,
      reordered: false,
      missing: 0,
      gap_s: 0,
      overlap_s: 0,
      format_changed: false,
      expired: 0,
      evicted: 0,
      ...overrides,
    };
  }

  units() {
    return this._units.map((unit) => ({ ...unit, boundaries: [...(unit.boundaries ?? [])] }));
  }

  window(tStart, tEnd) {
    if (tEnd < tStart) throw new Error("audio window t_end must be >= t_start");
    return this.units().filter((unit) => unit.t_end >= tStart && unit.t_start <= tEnd);
  }

  boundaries(tStart = null, tEnd = null) {
    return this._units
      .flatMap((unit) => unit.boundaries ?? [])
      .filter((boundary) => tStart === null || boundary.t >= tStart)
      .filter((boundary) => tEnd === null || boundary.t <= tEnd)
      .sort((a, b) => a.t - b.t || a.kind.localeCompare(b.kind));
  }
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
