// test/privacy-audio.test.js
//
// The dedicated suite for the AUDIO tier (src/privacy-audio.js): PrivacyAudioHead and
// AUDIO_SCORE_CAP.
//
// Why this file exists separately from privacy.test.js. The aggregate suite reaches this head
// through PrivacyScreen.screenAsync and asserts the *action* ("allow" / "<= confirm"). That
// mapping is deeply lossy for this tier in particular: PrivacyScreen._verdict overwrites the
// head's own `sensitive` with `action !== "allow"`, and the default confirm band is 0.35 — so
// every head score below 0.35 — including, before the G-03-0x fixes, a NaN and a float-noise
// 4e-17 — arrives at the caller as an indistinguishable `{sensitive:false, action:"allow"}`. The
// score-contract assertions at the bottom of this file are invisible from the aggregate suite for
// that reason. This file asserts the head's own return value whole, before any mapping.
//
// There is no Python twin of this module in the repository — `percept/` has no acoustic privacy
// head at all — so "the two runtimes agree" cannot be asserted against anything, which is what
// spec row JS-7 ("privacy-audio.js / Same, for audio / Covers: NONE dedicated") records. What
// stands in for a parity vector set is a frozen synthetic PCM battery: the exact numbers below are
// the contract a second implementation would have to reproduce.
//
// The contract, frozen:
//   * Four keys, always, in this order: {sensitive, categories, score, spans}. No `failed` key —
//     privacy.js `_decide` keys fail-closed off `part.failed`, so a port that adds `failed:false`
//     is harmless but a port that adds `failed:true` on a bad buffer silently turns every such
//     buffer into a hard `drop`.
//   * `categories` and `spans` are ALWAYS []. This head asserts no category by design (no
//     on-device STT exists to justify one) and localizes nothing. A port that emits a category
//     would move audio onto privacy.js's severity ladder, where HARD categories drop.
//   * The early returns (a buffer that is not a readable PCM array, an empty one, a non-finite or
//     zero energy, energy below the floor) all produce the byte-same silent tuple.
//   * `score` is `min(CAP, inBand * energyConf * CAP)` — triangular ZCR membership times a
//     saturating energy confidence. Amplitude above 10x the floor does not change the score.
//   * `sensitive` is exactly `score > 0`, and the score is ALWAYS a finite number in [0, CAP] —
//     which is what the two G-03-0x defects broke, from opposite ends (a NaN, and a negative).
//   * The ZCR band is HALF-OPEN in effect: membership at either edge is exactly 0, on both sides.
//
// Deliberate choices in the vectors:
//   * Amplitudes are 0.5 / 1.0 — exactly representable in float32 — so the Float32Array and
//     number[] paths are compared on values that cannot differ by storage rounding. A test using
//     0.9 would compare float32-rounded against float64 and prove nothing about the code.
//   * ZCR is set by construction (a chosen number of sign flips over 101 samples) rather than by
//     picking a sine period, because the band edges and the band centre have to be hit EXACTLY
//     for the membership assertions to mean anything.
//   * No Math.random and no clock anywhere: the module docstring claims "no model, no clock, no
//     randomness", and a fuzz vector would make the determinism test unfalsifiable.

import { describe, it, expect } from "vitest";
import { PrivacyAudioHead, AUDIO_SCORE_CAP } from "../src/privacy-audio.js";
import { PrivacyScreen } from "../src/privacy.js";
import { frameEnergy } from "../src/vad.js";

// --- vectors ---------------------------------------------------------------

/** The one tuple every non-scoring path must produce. toStrictEqual, so an extra key fails. */
const SILENT = { sensitive: false, categories: [], score: 0, spans: [] };

/**
 * A square-ish buffer of `n` samples at amplitude `A` with EXACTLY `crossings` sign changes, so
 * its zero-crossing rate is exactly `crossings / (n - 1)`. With n = 101 the rate is `crossings/100`
 * — every hundredth is exactly representable, which is what lets the band-edge tests be exact.
 * Energy is exactly A*A (uniform magnitude), so the energy floor is exactly controllable too.
 */
function pcmWithCrossings(crossings, A = 0.5, n = 101) {
  const buf = new Float32Array(n);
  let sign = 1;
  let flipped = 0;
  for (let i = 0; i < n; i++) {
    if (i > 0 && flipped < crossings) {
      sign = -sign;
      flipped += 1;
    }
    buf[i] = A * sign;
  }
  return buf;
}

/** zcr 0.05 — the LOWER band edge of the shipped [0.05, 0.25] default. */
const PCM_ZCR_LO_EDGE = pcmWithCrossings(5);
/** zcr 0.15 — the band CENTRE, peak triangular membership. */
const PCM_ZCR_CENTRE = pcmWithCrossings(15);
/** zcr 0.25 — the UPPER band edge. */
const PCM_ZCR_HI_EDGE = pcmWithCrossings(25);
/** zcr 0.10 / 0.20 — symmetric half-membership points either side of the centre. */
const PCM_ZCR_HALF_LO = pcmWithCrossings(10);
const PCM_ZCR_HALF_HI = pcmWithCrossings(20);
/** zcr 0.50 — well outside the band on the high side. */
const PCM_ZCR_OUT_HI = pcmWithCrossings(50);

/** A deterministic LCG "noise" buffer — no Math.random, so the determinism claim stays testable. */
function lcgNoise(n, seed = 12345) {
  const buf = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (1103515245 * s + 12345) >>> 0;
    buf[i] = (s / 4294967296) * 2 - 1;
  }
  return buf;
}

/** The signal battery the score range is swept over. Names are used in failure messages. */
const BATTERY = [
  ["digital silence (all zeros)", new Float32Array(400)],
  ["DC offset, high energy", new Float32Array(400).fill(0.9)],
  ["low-frequency sine (period 64)", (() => {
    const b = new Float32Array(400);
    for (let i = 0; i < 400; i++) b[i] = 0.8 * Math.sin((2 * Math.PI * i) / 64);
    return b;
  })()],
  ["speech-rate sine (period 13.5)", (() => {
    const b = new Float32Array(1600);
    for (let i = 0; i < 1600; i++) b[i] = 0.9 * Math.sin((2 * Math.PI * i) / 13.5);
    return b;
  })()],
  ["loud clipping square at speech rate", (() => {
    const b = new Float32Array(1600);
    for (let i = 0; i < 1600; i++) b[i] = Math.sign(Math.sin((2 * Math.PI * i) / 13.5)) || 1;
    return b;
  })()],
  ["full-scale Nyquist square (zcr 1.0)", (() => {
    const b = new Float32Array(400);
    for (let i = 0; i < 400; i++) b[i] = i % 2 ? -1 : 1;
    return b;
  })()],
  ["deterministic LCG noise", lcgNoise(400)],
  ["single sample, full scale", new Float32Array([1])],
  ["band centre, in-band", PCM_ZCR_CENTRE],
  ["lower band edge", PCM_ZCR_LO_EDGE],
  ["upper band edge", PCM_ZCR_HI_EDGE],
];

// --- the cap ---------------------------------------------------------------

describe("AUDIO_SCORE_CAP — the ceiling that keeps audio out of `drop`", () => {
  it("is 0.4", () => {
    expect(AUDIO_SCORE_CAP).toBe(0.4);
  });

  it("sits strictly below PrivacyScreen's default `drop` band, so the cap alone bounds the tier", () => {
    // privacy.js caps the audio ACTION at `confirm` via capAction(), but that is the second of two
    // independent guards. The first is arithmetic: a head that can never exceed 0.4 can never reach
    // the 0.6 drop band in _decide, whatever capAction does. Freezing the inequality means a future
    // widening of the cap (or a narrowing of the default drop band) trips here rather than silently
    // making the action cap the only thing standing between audio and a hard drop.
    const bands = new PrivacyScreen().bands;
    expect(bands.drop).toBe(0.6);
    expect(AUDIO_SCORE_CAP).toBeLessThan(bands.drop);
  });
});

// --- the three silent early returns ----------------------------------------

describe("PrivacyAudioHead.screen — the silent early returns are byte-identical", () => {
  const head = new PrivacyAudioHead();

  it("a falsy or empty buffer returns exactly the silent tuple", () => {
    // Four keys, not five: the guard `!pcm || pcm.length === 0` returns a literal, so a port must
    // not add `failed` here. An audio buffer that never arrived is NOT a fail-closed condition for
    // this head — privacy.js reserves that for a missing head entirely (screenAsync case "audio").
    for (const empty of [null, undefined, new Float32Array(0), [], ""]) {
      expect(head.screen(empty)).toStrictEqual(SILENT);
    }
  });

  it("energy below minEnergy returns the same tuple, whatever the ZCR", () => {
    // The floor is checked BEFORE the ZCR, so a perfectly in-band articulation rate at whisper
    // energy is silent. That ordering is the reason a quiet room cannot accumulate score.
    const quiet = pcmWithCrossings(15, 0.01); // energy 1e-4, floor is 5e-3
    expect(frameEnergy(quiet)).toBeLessThan(0.005);
    expect(head.screen(quiet)).toStrictEqual(SILENT);
  });

  it("the floor is strict `<`: energy exactly equal to minEnergy is admitted, at 1/10 confidence", () => {
    // energy == minEnergy => energyConf = min(1, e / (10e)) = 0.1 => score = 0.1 * 1 * 0.4 = 0.04.
    // Exact, not approximate: 0.5*0.5 = 0.25 and 101 * 0.25 / 101 are both exact in float64.
    expect(frameEnergy(PCM_ZCR_CENTRE)).toBe(0.25);
    const atFloor = new PrivacyAudioHead({ minEnergy: 0.25 });
    expect(atFloor.screen(PCM_ZCR_CENTRE).score).toBeCloseTo(0.04, 12);
    expect(atFloor.screen(PCM_ZCR_CENTRE).sensitive).toBe(true);
    // One ulp-ish above the same energy and it is silent instead.
    const justAbove = new PrivacyAudioHead({ minEnergy: 0.2500001 });
    expect(justAbove.screen(PCM_ZCR_CENTRE)).toStrictEqual(SILENT);
  });

  it("a one-sample buffer is never sensitive, however loud", () => {
    // zeroCrossingRate returns 0 for n < 2, and 0 is outside the default band — so the shortest
    // possible non-empty buffer takes the scoring path and still scores 0. Worth freezing because
    // the alternative implementation (crossings / n, or a guard that returns NaN) would not.
    const one = new Float32Array([1]);
    expect(frameEnergy(one)).toBe(1);
    expect(head.screen(one)).toStrictEqual(SILENT);
  });
});

// --- the category-free invariant -------------------------------------------

describe("PrivacyAudioHead.screen — asserts no category, localizes nothing, ever", () => {
  const head = new PrivacyAudioHead();

  it("categories and spans are empty for every vector in the battery", () => {
    // This is the honesty bound, not an implementation detail: there is no on-device STT, so the
    // head cannot know WHAT was said. A port that emitted e.g. "payment" would put audio on
    // privacy.js's HARD-severity rung, where _decide escalates to "drop" — and a coarse ZCR
    // heuristic would then be dropping payloads on the strength of an articulation rate.
    for (const [name, pcm] of BATTERY) {
      const r = head.screen(pcm);
      expect(r.categories, name).toStrictEqual([]);
      expect(r.spans, name).toStrictEqual([]);
    }
  });

  it("returns exactly four keys and no `failed` discriminator", () => {
    for (const [name, pcm] of BATTERY) {
      expect(Object.keys(head.screen(pcm)), name).toStrictEqual([
        "sensitive",
        "categories",
        "score",
        "spans",
      ]);
    }
  });

  it("hands back a fresh object each call — a caller mutating categories cannot poison the next", () => {
    const a = head.screen(PCM_ZCR_CENTRE);
    a.categories.push("payment");
    a.spans.push({ start: 0, end: 1 });
    const b = head.screen(PCM_ZCR_CENTRE);
    expect(b.categories).toStrictEqual([]);
    expect(b.spans).toStrictEqual([]);
    expect(b).not.toBe(a);
  });
});

// --- the score range -------------------------------------------------------

describe("PrivacyAudioHead.screen — score stays inside [0, AUDIO_SCORE_CAP]", () => {
  const head = new PrivacyAudioHead();

  it("holds across the whole signal battery", () => {
    for (const [name, pcm] of BATTERY) {
      const s = head.screen(pcm).score;
      expect(Number.isFinite(s), `${name}: score must be finite`).toBe(true);
      expect(s, name).toBeGreaterThanOrEqual(0);
      expect(s, name).toBeLessThanOrEqual(AUDIO_SCORE_CAP);
    }
  });

  it("`sensitive` is exactly `score > 0` across the battery", () => {
    for (const [name, pcm] of BATTERY) {
      const r = head.screen(pcm);
      expect(r.sensitive, name).toBe(r.score > 0);
    }
  });
});

// --- the ZCR triangular membership -----------------------------------------

describe("PrivacyAudioHead.screen — triangular ZCR membership over the shipped [0.05, 0.25] band", () => {
  const head = new PrivacyAudioHead();

  it("a high-energy DC buffer (zcr 0, below the band) scores exactly 0", () => {
    // Not "approximately 0" — the `zcr >= lo` guard means inBand is never assigned, so raw is a
    // literal 0 and min(0.4, 0) is exactly 0. A port that computed the triangle unconditionally
    // and clamped afterwards would produce a small positive number here and flip `sensitive`.
    const dc = new Float32Array(400).fill(0.9);
    expect(frameEnergy(dc)).toBeGreaterThan(0.005); // it is loud; only the ZCR excludes it
    expect(head.screen(dc)).toStrictEqual(SILENT);
  });

  it("zcr well above the band (0.50) scores exactly 0", () => {
    expect(head.screen(PCM_ZCR_OUT_HI)).toStrictEqual(SILENT);
  });

  it("the band centre (zcr 0.15) with saturated energy gives exactly the cap", () => {
    // membership 1 * energyConf 1 * 0.4 = 0.4, exactly. This is the single input that reaches the
    // ceiling, and it is the only place AUDIO_SCORE_CAP is observable as a value rather than a bound.
    expect(head.screen(PCM_ZCR_CENTRE).score).toBe(AUDIO_SCORE_CAP);
    expect(head.screen(PCM_ZCR_CENTRE).sensitive).toBe(true);
  });

  it("half-membership points either side of the centre are symmetric at half the cap", () => {
    // zcr 0.10 and 0.20 are both |0.05| from the centre over a half-width of 0.10 => membership 0.5.
    expect(head.screen(PCM_ZCR_HALF_LO).score).toBeCloseTo(AUDIO_SCORE_CAP / 2, 12);
    expect(head.screen(PCM_ZCR_HALF_HI).score).toBeCloseTo(AUDIO_SCORE_CAP / 2, 12);
  });

  it("the UPPER band edge (zcr 0.25) has membership 0 — exactly the silent tuple", () => {
    // 0.25 - 0.15 divided by the 0.1 half-width comes out as exactly 1.0 in float64, so
    // 1 - 1 = 0 and the tuple is byte-identical to silence. The LOWER edge does not — see the
    // failing case at the bottom of this file. The asymmetry is the defect, not the design.
    expect(head.screen(PCM_ZCR_HI_EDGE)).toStrictEqual(SILENT);
  });
});

// --- energy confidence -----------------------------------------------------

describe("PrivacyAudioHead.screen — energy confidence saturates at 10x the floor", () => {
  it("reaches 1 exactly at energy == 10 * minEnergy and does not grow past it", () => {
    // Energy is pinned at exactly 0.25 by construction, so the ratio is exact and these are `toBe`,
    // not `toBeCloseTo`. minEnergy 0.025 puts the saturation point exactly at the buffer's energy.
    expect(frameEnergy(PCM_ZCR_CENTRE)).toBe(0.25);
    expect(new PrivacyAudioHead({ minEnergy: 0.025 }).screen(PCM_ZCR_CENTRE).score).toBe(0.4);
    // Ten times more headroom: still saturated, identical score. Loudness beyond the knee is
    // deliberately not evidence — that is what stops a slammed door outscoring a spoken sequence.
    expect(new PrivacyAudioHead({ minEnergy: 0.0025 }).screen(PCM_ZCR_CENTRE).score).toBe(0.4);
  });

  it("below the knee the score is linear in energy", () => {
    // minEnergy 0.05 puts saturation at 0.5 while the buffer sits at 0.25 => conf 0.5 => score 0.2.
    // Exact for the halving; the fifth is `toBeCloseTo` because 0.2 * 0.4 is not exact in float64
    // (it lands on 0.08000000000000002) — that is IEEE-754, not a divergence worth freezing.
    expect(new PrivacyAudioHead({ minEnergy: 0.05 }).screen(PCM_ZCR_CENTRE).score).toBe(0.2);
    expect(new PrivacyAudioHead({ minEnergy: 0.125 }).screen(PCM_ZCR_CENTRE).score).toBeCloseTo(
      0.08,
      12
    );
  });

  it("amplitude above the knee is invisible: a sine and a clipping square at the same rate tie", () => {
    // Energy 0.405 vs energy 1.0 — a 2.5x difference that the saturated confidence erases. If a
    // port dropped the saturation, the clipped buffer would outscore the clean one and loudness
    // would become the dominant term.
    const head = new PrivacyAudioHead();
    const n = 1600;
    const sine = new Float32Array(n);
    const square = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = Math.sin((2 * Math.PI * i) / 13.5);
      sine[i] = 0.9 * v;
      square[i] = Math.sign(v) || 1;
    }
    expect(frameEnergy(square)).toBe(1);
    expect(frameEnergy(sine)).toBeLessThan(0.5);
    expect(head.screen(sine).score).toBe(head.screen(square).score);
    expect(head.screen(sine).score).toBeGreaterThan(0);
  });
});

// --- constructor overrides -------------------------------------------------

describe("PrivacyAudioHead — the two constructor knobs actually take effect", () => {
  it("defaults are minEnergy 0.005 and zcrBand [0.05, 0.25]", () => {
    const head = new PrivacyAudioHead();
    expect(head.minEnergy).toBe(0.005);
    expect(head.zcrBand).toStrictEqual([0.05, 0.25]);
    // An empty options object and no argument at all must land in the same place.
    expect(new PrivacyAudioHead({}).minEnergy).toBe(0.005);
    expect(new PrivacyAudioHead({}).zcrBand).toStrictEqual([0.05, 0.25]);
  });

  it("a raised minEnergy silences a buffer the default admits", () => {
    expect(new PrivacyAudioHead().screen(PCM_ZCR_CENTRE).score).toBe(0.4);
    expect(new PrivacyAudioHead({ minEnergy: 0.5 }).screen(PCM_ZCR_CENTRE)).toStrictEqual(SILENT);
  });

  it("a shifted zcrBand moves the whole triangle, not just its width", () => {
    // zcr 0.50 is silent under the default band and peaks under a band centred on it — and the
    // previously-peaking 0.15 buffer goes silent. Both directions, so a constructor that stored
    // the band but never read it cannot pass.
    const shifted = new PrivacyAudioHead({ zcrBand: [0.4, 0.6] });
    expect(new PrivacyAudioHead().screen(PCM_ZCR_OUT_HI).score).toBe(0);
    expect(shifted.screen(PCM_ZCR_OUT_HI).score).toBe(AUDIO_SCORE_CAP);
    expect(shifted.screen(PCM_ZCR_CENTRE).score).toBe(0);
  });
});

// --- determinism and input-type parity -------------------------------------

describe("PrivacyAudioHead.screen — deterministic and storage-agnostic", () => {
  it("the same buffer twice serializes byte-identically, to distinct objects", () => {
    // The module claims "no model, no clock, no randomness". Nothing here injects a clock, so the
    // only way this fails is if the head grew hidden state between calls.
    const head = new PrivacyAudioHead();
    for (const [name, pcm] of BATTERY) {
      const first = head.screen(pcm);
      const second = head.screen(pcm);
      expect(JSON.stringify(second), name).toBe(JSON.stringify(first));
      expect(second, name).not.toBe(first);
    }
  });

  it("two separately constructed heads agree — no module-level state", () => {
    const a = new PrivacyAudioHead();
    const b = new PrivacyAudioHead();
    a.screen(PCM_ZCR_CENTRE);
    a.screen(PCM_ZCR_OUT_HI);
    expect(JSON.stringify(b.screen(PCM_ZCR_CENTRE))).toBe(JSON.stringify(a.screen(PCM_ZCR_CENTRE)));
  });

  it("Float32Array and plain number[] carrying the same values give the same verdict", () => {
    // Amplitudes are 0.5, exactly representable in float32, so any difference here would be a
    // code path difference (indexed access, .length, a typed-array-only method) and not rounding.
    const head = new PrivacyAudioHead();
    for (const [name, typed] of [
      ["centre", PCM_ZCR_CENTRE],
      ["lower edge", PCM_ZCR_LO_EDGE],
      ["out of band", PCM_ZCR_OUT_HI],
      ["half membership", PCM_ZCR_HALF_LO],
    ]) {
      const plain = Array.from(typed);
      expect(JSON.stringify(head.screen(plain)), name).toBe(JSON.stringify(head.screen(typed)));
    }
  });

  it("the four-key shape survives an input the guard was never written for", () => {
    // `!pcm || pcm.length === 0` is a falsy-and-length guard, not a type check, so a bare number
    // walks straight past it into frameEnergy. The SHAPE still holds — four keys, no category, no
    // span — which is all this test claims. The `score` it carries on that path is not finite; see
    // the failing case below, which is the same missing guarantee reached by a supported input.
    const r = new PrivacyAudioHead().screen(5);
    expect(Object.keys(r)).toStrictEqual(["sensitive", "categories", "score", "spans"]);
    expect(r.categories).toStrictEqual([]);
    expect(r.spans).toStrictEqual([]);
  });
});

// --- the score contract, at the knobs and edges that used to break it -------

describe("PrivacyAudioHead — the documented score contract holds at every knob", () => {
  it("BUG-G-03-01: a degenerate zcrBand is rejected outright, not silently scored NaN", () => {
    // The constructor used to accept any [lo, hi]. With lo === hi the triangular membership is
    // `1 - |zcr - centre| / 0`; a ZCR exactly on that point made it 0/0 = NaN, which flowed
    // through raw and survived Math.min. The JSDoc on screen() promises "score in
    // [0, AUDIO_SCORE_CAP]" and the verdict shape in privacy.js requires a finite number in [0,1]
    // — makePrivacyVerdict would have thrown on it, except that _decide reads `part.sensitive`
    // first and NaN > 0 is false, so the buffer was routed to "allow" before the validator ever
    // saw the score: the band did not error and did not score, it disabled the tier silently.
    // Now it is a construction-time RangeError, matching PrivacyScreen's own confirm>drop guard.
    expect(() => new PrivacyAudioHead({ zcrBand: [0.15, 0.15] })).toThrow(RangeError);
    expect(() => new PrivacyAudioHead({ zcrBand: [0.3, 0.1] })).toThrow(/lo < hi/);
    expect(() => new PrivacyAudioHead({ zcrBand: [NaN, 0.3] })).toThrow(RangeError);
    expect(() => new PrivacyAudioHead({ zcrBand: [0.05] })).toThrow(/\[lo, hi\] pair/);
    expect(() => new PrivacyAudioHead({ minEnergy: NaN })).toThrow(RangeError);
    expect(() => new PrivacyAudioHead({ minEnergy: -1 })).toThrow(/>= 0/);
  });

  it("BUG-G-03-01: the second guard — a band mutated after construction scores 0, not NaN", () => {
    // `zcrBand` is a public field, so the constructor check is necessary but not sufficient. The
    // membership helper carries its own `hi > lo` guard, and this is the only way to reach it.
    const head = new PrivacyAudioHead();
    head.zcrBand = [0.15, 0.15];
    const r = head.screen(PCM_ZCR_CENTRE);
    expect(Number.isFinite(r.score)).toBe(true);
    expect(r.score).toBeGreaterThanOrEqual(0);
    expect(r.score).toBeLessThanOrEqual(AUDIO_SCORE_CAP);
    expect(r).toStrictEqual(SILENT);
  });

  it("BUG-G-03-02: the LOWER band edge is membership 0, symmetric with the upper edge", () => {
    // zcr is exactly 0.05, the band's lower bound, where the triangle is mathematically zero.
    // `1 - |0.05 - 0.15| / 0.1` evaluates to 1.1102230246251565e-16 in float64 rather than 0, so
    // score was 4.4408920985006264e-17 and `sensitive: score > 0` was TRUE, while the upper edge
    // (0.25) evaluated to exactly 0. The same boundary answered differently depending on which
    // side it was approached from. Both edges are now short-circuited before the arithmetic.
    expect(new PrivacyAudioHead().screen(PCM_ZCR_LO_EDGE)).toStrictEqual(SILENT);
    expect(new PrivacyAudioHead().screen(PCM_ZCR_HI_EDGE)).toStrictEqual(SILENT);
  });

  it("BUG-G-03-02: no ordinary band leaves a NEGATIVE residual, which threw on a valid buffer", () => {
    // The float residual at a band edge is positive for the shipped band and NEGATIVE for others
    // ([0.1,0.3], [0.3,0.7], [0.01,0.03], [0.02,0.06], [0.17,0.51] all measured negative). A
    // negative score is not merely wrong: makePrivacyVerdict rejects it, so PrivacyScreen.screenAsync
    // threw a TypeError on a perfectly valid PCM buffer.
    for (const [lo, hi] of [
      [0.05, 0.25], [0.1, 0.3], [0.3, 0.7], [0.01, 0.03],
      [0.02, 0.06], [0.17, 0.51], [0.2, 0.4], [0.15, 0.45],
    ]) {
      const head = new PrivacyAudioHead({ zcrBand: [lo, hi] });
      for (let crossings = 0; crossings <= 100; crossings++) {
        const s = head.screen(pcmWithCrossings(crossings)).score;
        expect(Number.isFinite(s), `band [${lo},${hi}] at zcr ${crossings / 100}`).toBe(true);
        expect(s, `band [${lo},${hi}] at zcr ${crossings / 100}`).toBeGreaterThanOrEqual(0);
        expect(s, `band [${lo},${hi}] at zcr ${crossings / 100}`).toBeLessThanOrEqual(AUDIO_SCORE_CAP);
      }
    }
  });

  it("BUG-G-03-02: screenAsync returns a verdict for every band, where it used to throw", async () => {
    for (const [lo, hi] of [[0.1, 0.3], [0.3, 0.7], [0.01, 0.03], [0.02, 0.06], [0.17, 0.51]]) {
      const screen = new PrivacyScreen({ audio: new PrivacyAudioHead({ zcrBand: [lo, hi] }) });
      const v = await screen.screenAsync({ modality: "audio", pcm: pcmWithCrossings(lo * 100) });
      expect(Number.isFinite(v.score), `band [${lo},${hi}]`).toBe(true);
      expect(v.score, `band [${lo},${hi}]`).toBeGreaterThanOrEqual(0);
    }
  });

  it("BUG-G-03-01: an input the guard was never written for is silent, not NaN-scored", () => {
    // `!pcm || pcm.length === 0` was a falsy-and-length test with no type check, so a bare number,
    // a string, or any `{length: n}` object walked into frameEnergy and came back with a NaN
    // score. All of them now take the same silent early return an empty buffer takes.
    for (const bad of [5, "abc", { length: 4 }, true, {}, () => {}, new Map()]) {
      expect(new PrivacyAudioHead().screen(bad), String(bad)).toStrictEqual(SILENT);
    }
    // A buffer of the right TYPE carrying non-finite samples is the same story via frameEnergy.
    expect(new PrivacyAudioHead().screen([NaN, NaN, NaN])).toStrictEqual(SILENT);
    expect(new PrivacyAudioHead().screen(new Float32Array([Infinity, -Infinity]))).toStrictEqual(
      SILENT
    );
  });
});
