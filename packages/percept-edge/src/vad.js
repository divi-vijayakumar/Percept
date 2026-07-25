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
    if (!Number.isInteger(samples[i]) || samples[i] < -32768 || samples[i] > 32767) {
      throw new Error("pcm16Energy samples must be signed 16-bit integers");
    }
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
  constructor({ seconds = 4, maxUnits = 4096, maxBytes = 16 * 1024 * 1024 } = {}) {
    if (!Number.isFinite(seconds) || seconds <= 0) {
      throw new Error("seconds must be finite and positive");
    }
    if (!Number.isSafeInteger(maxUnits) || maxUnits <= 0) {
      throw new Error("maxUnits must be a positive safe integer");
    }
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new Error("maxBytes must be a positive safe integer");
    }
    this.seconds = Math.max(0.5, seconds);
    this.maxUnits = maxUnits;
    this.maxBytes = maxBytes;
    this._units = [];
    this._bytes = 0;
    this._maxSeq = null;
    this._seenLimit = Math.max(4096, maxUnits * 2);
    if (!Number.isSafeInteger(this._seenLimit)) {
      throw new Error("maxUnits is too large for bounded duplicate history");
    }
    this._seenOrder = [];
    this._seenSeq = new Set();
  }

  push(unit) {
    if (!unit || typeof unit !== "object") throw new Error("audio unit must be an object");
    if (!Number.isSafeInteger(unit.seq) || unit.seq < 0) {
      throw new Error("audio unit seq must be a non-negative safe integer");
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
      (!Number.isSafeInteger(unit.sample_rate) || unit.sample_rate <= 0)
    ) {
      throw new Error("audio unit sample_rate must be positive when known");
    }
    if (
      unit.channels != null &&
      (!Number.isSafeInteger(unit.channels) || unit.channels <= 0)
    ) {
      throw new Error("audio unit channels must be positive when known");
    }
    if (
      unit.sample_format != null &&
      !["s16le", "f32le", "u8"].includes(unit.sample_format)
    ) {
      throw new Error("audio unit sample_format is invalid");
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
    if (unit.event_t != null && !Number.isFinite(unit.event_t)) {
      throw new Error("audio unit event_t must be finite when known");
    }
    if (unit.boundaries != null && !Array.isArray(unit.boundaries)) {
      throw new Error("audio unit boundaries must be an array");
    }
    const boundaries = (unit.boundaries ?? []).map((boundary) => {
      if (
        !boundary ||
        !["onset", "vad_start", "endpoint"].includes(boundary.kind) ||
        !Number.isFinite(boundary.t)
      ) {
        throw new Error("audio unit boundary is invalid");
      }
      return Object.freeze({ kind: boundary.kind, t: boundary.t });
    });
    if (unit.frame != null) {
      for (const key of ["sample_rate", "channels", "sample_format"]) {
        if (unit[key] != null && unit.frame[key] != null && unit[key] !== unit.frame[key]) {
          throw new Error(`audio unit ${key} disagrees with its frame`);
        }
      }
      for (const key of ["t_start", "t_end"]) {
        if (unit.frame[key] != null && !Number.isFinite(unit.frame[key])) {
          throw new Error(`audio frame ${key} must be finite when known`);
        }
        if (unit.frame[key] != null && unit.frame[key] !== unit[key]) {
          throw new Error(`audio unit ${key} disagrees with its frame`);
        }
      }
    }
    const byteLength = audioUnitBytes(unit);
    if (byteLength > this.maxBytes) {
      throw new Error("audio unit exceeds the buffer byte cap");
    }
    const pcm = unit.pcm ?? unit.frame?.pcm;
    if (pcm != null && unit.sample_format != null && unit.channels != null) {
      const bytesPerSample = { s16le: 2, f32le: 4, u8: 1 }[unit.sample_format];
      if (byteLength % (bytesPerSample * unit.channels) !== 0) {
        throw new Error("audio unit PCM is not aligned to its format/channels");
      }
    }
    if (this._seenSeq.has(unit.seq)) {
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
    if (overlap_s) {
      return this._result({
        accepted: false,
        reordered,
        missing,
        gap_s,
        overlap_s,
      });
    }
    const formatDiff = (other) =>
      Boolean(
        other &&
          ((unit.sample_rate != null &&
            other.sample_rate != null &&
            unit.sample_rate !== other.sample_rate) ||
          (unit.channels != null &&
            other.channels != null &&
            unit.channels !== other.channels) ||
          (unit.sample_format != null &&
            other.sample_format != null &&
            unit.sample_format !== other.sample_format))
      );
    const format_changed = formatDiff(previous) || formatDiff(following);

    const retainedFrame =
      unit.frame == null ? undefined : Object.freeze({ ...unit.frame });
    const retained = Object.freeze({
      direction: "unknown",
      participant_id: null,
      sample_format: null,
      ...unit,
      ...(retainedFrame === undefined ? {} : { frame: retainedFrame }),
      byte_length: byteLength,
      boundaries: Object.freeze(boundaries),
    });
    this._units.splice(index, 0, retained);
    this._bytes += byteLength;
    this._maxSeq = previousMaxSeq === null ? unit.seq : Math.max(previousMaxSeq, unit.seq);
    this._seenSeq.add(unit.seq);
    this._seenOrder.push(unit.seq);
    while (this._seenOrder.length > this._seenLimit) {
      this._seenSeq.delete(this._seenOrder.shift());
    }

    const newestEnd = Math.max(...this._units.map((row) => row.t_end));
    const cutoff = newestEnd - this.seconds;
    let expired = 0;
    let expired_bytes = 0;
    while (this._units.length > 1) {
      const staleIndex = this._units.findIndex((retained) => retained.t_end < cutoff);
      if (staleIndex < 0) break;
      const [dropped] = this._units.splice(staleIndex, 1);
      this._bytes -= dropped.byte_length;
      expired++;
      expired_bytes += dropped.byte_length;
    }
    let evicted = 0;
    let evicted_bytes = 0;
    while (
      this._units.length > 1 &&
      (this._units.length > this.maxUnits || this._bytes > this.maxBytes)
    ) {
      const dropped = this._units.shift();
      this._bytes -= dropped.byte_length;
      evicted++;
      evicted_bytes += dropped.byte_length;
    }
    return this._result({
      accepted: true,
      reordered,
      missing,
      gap_s,
      overlap_s,
      format_changed,
      expired,
      expired_bytes,
      evicted,
      evicted_bytes,
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
      expired_bytes: 0,
      evicted: 0,
      evicted_bytes: 0,
      ...overrides,
    };
  }

  units() {
    return this._units.map((unit) => ({
      ...unit,
      boundaries: unit.boundaries.map((boundary) => ({ ...boundary })),
      ...(unit.frame == null ? {} : { frame: { ...unit.frame } }),
    }));
  }

  window(tStart, tEnd) {
    if (!Number.isFinite(tStart) || !Number.isFinite(tEnd)) {
      throw new Error("audio window times must be finite");
    }
    if (tEnd < tStart) throw new Error("audio window t_end must be >= t_start");
    return this.units().filter(
      (unit) =>
        (unit.t_end > tStart && unit.t_start < tEnd) ||
        (unit.t_start === unit.t_end && tStart <= unit.t_start && unit.t_start < tEnd)
    );
  }

  boundaries(tStart = null, tEnd = null) {
    if (tStart != null && !Number.isFinite(tStart)) {
      throw new Error("boundary window start must be finite");
    }
    if (tEnd != null && !Number.isFinite(tEnd)) {
      throw new Error("boundary window end must be finite");
    }
    if (tStart != null && tEnd != null && tEnd < tStart) {
      throw new Error("boundary window end must be >= start");
    }
    return this._units
      .flatMap((unit) => unit.boundaries ?? [])
      .filter((boundary) => tStart === null || boundary.t >= tStart)
      .filter((boundary) => tEnd === null || boundary.t < tEnd)
      .sort((a, b) => a.t - b.t || a.kind.localeCompare(b.kind));
  }
}

function audioUnitBytes(unit) {
  const payload = unit.pcm ?? unit.frame?.pcm;
  const actual =
    ArrayBuffer.isView(payload)
      ? payload.byteLength
      : payload instanceof ArrayBuffer
        ? payload.byteLength
        : null;
  if (unit.byte_length != null) {
    if (!Number.isSafeInteger(unit.byte_length) || unit.byte_length < 0) {
      throw new Error("audio unit byte_length must be a non-negative safe integer");
    }
    if (actual !== null && unit.byte_length !== actual) {
      throw new Error("audio unit byte_length disagrees with its PCM payload");
    }
    return unit.byte_length;
  }
  return actual ?? 0;
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
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      throw new Error("threshold must be finite and within [0, 1]");
    }
    if (!Number.isSafeInteger(hangoverFrames) || hangoverFrames <= 0) {
      throw new Error("hangoverFrames must be a positive safe integer");
    }
    if (!Number.isSafeInteger(minSpeechFrames) || minSpeechFrames <= 0) {
      throw new Error("minSpeechFrames must be a positive safe integer");
    }
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
    if (!Number.isFinite(energy) || energy < 0 || energy > 1) {
      throw new Error("energy must be finite and within [0, 1]");
    }
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
