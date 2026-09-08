import { describe, it, expect } from "vitest";
import { Edge } from "../src/edge.js";
import { makeFrame } from "../src/motion.js";
import { parseTier0Signal } from "../src/contract.js";

const PY_WATCHSPEC =
  '{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null}],"version":1}';
// The same spec with the goal OPTING IN to the wake-policy fusion. A configured policy no longer
// overrides the channel a goal asked for, so a fused goal has to say so (BUG-A-01-04).
const FUSED_WATCHSPEC =
  '{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null,"gate_on":"fused"}],"version":1}';

function makeEdge(opts = {}) {
  const signals = [];
  // deterministic clock for stable ts in assertions
  let t = 1000;
  const edge = new Edge({
    onSignal: (s) => signals.push(s),
    clock: () => t++,
    ...opts,
  });
  return { edge, signals };
}

describe("Edge — WatchSpec down / Tier0Signal up wired loop", () => {
  it("applyWatchSpec consumes the exact Python WatchSpec JSON", () => {
    const { edge } = makeEdge();
    const spec = edge.applyWatchSpec(PY_WATCHSPEC);
    expect(spec.conditions[0].goal_id).toBe("g1");
    expect(edge.activeThreshold).toBeCloseTo(0.3, 10);
  });

  it("still frames -> low motion, no escalation (no cloud call)", () => {
    const { edge, signals } = makeEdge();
    edge.applyWatchSpec(PY_WATCHSPEC);
    const still = makeFrame(8, 8, 120);

    expect(edge.processFrame(still)).toBeNull(); // first frame: no history
    expect(edge.processFrame(still)).toBeNull(); // still scene: gated out
    expect(edge.processFrame(still)).toBeNull();
    expect(signals).toHaveLength(0);
  });

  it("a moving frame clears threshold -> emits a well-formed Tier0Signal", () => {
    const { edge, signals } = makeEdge();
    edge.applyWatchSpec(PY_WATCHSPEC); // threshold 0.3
    const still = makeFrame(8, 8, 0);
    const moved = makeFrame(8, 8, 255); // full inversion -> motion ~1.0

    expect(edge.processFrame(still)).toBeNull(); // prime history
    const sig = edge.processFrame(moved); // big motion -> escalate

    expect(sig).not.toBeNull();
    // Field set is exactly the frozen contract (energy joined in 1.3, null on the frame path).
    expect(Object.keys(sig).sort()).toEqual([
      "energy",
      "motion",
      "semantic",
      "structural",
      "ts",
    ]);
    expect(sig.motion).toBeGreaterThan(0.3);
    expect(sig.structural).toBe(0);
    expect(sig.semantic).toBe(0);
    expect(sig.energy).toBe(null); // frame path carries no audio energy
    expect(typeof sig.ts).toBe("number");

    // It was emitted up the channel and is a valid wire Tier0Signal.
    expect(signals).toHaveLength(1);
    expect(signals[0]).toEqual(sig);
    expect(() => parseTier0Signal(JSON.stringify(sig))).not.toThrow();
  });

  it("motion below threshold does NOT escalate", () => {
    const { edge, signals } = makeEdge();
    edge.applyWatchSpec(PY_WATCHSPEC); // threshold 0.3
    const base = makeFrame(8, 8, 100);
    const tiny = makeFrame(8, 8, 105); // ~0.02 motion < 0.3

    edge.processFrame(base);
    expect(edge.processFrame(tiny)).toBeNull();
    expect(signals).toHaveLength(0);
  });

  it("falls back to defaultThreshold when no WatchSpec has ever been applied", () => {
    // RETITLED: "/ no conditions" was in this name and is no longer true — an APPLIED spec carrying
    // zero conditions never escalates now (see the FIND-PARITY-03 suite below). Only the
    // never-configured edge keeps the fallback, which is the case this test actually drives.
    const { edge, signals } = makeEdge();
    // no applyWatchSpec at all -> defaultThreshold 0.05
    expect(edge.activeThreshold).toBeCloseTo(0.05, 10);
    const a = makeFrame(8, 8, 0);
    const b = makeFrame(8, 8, 255);
    edge.processFrame(a);
    const sig = edge.processFrame(b);
    expect(sig).not.toBeNull();
    expect(signals).toHaveLength(1);
  });

  it("uses the minimum threshold across conditions (any goal can escalate)", () => {
    const { edge } = makeEdge();
    edge.applyWatchSpec({
      conditions: [
        { goal_id: "g1", anchor_text: "a", threshold: 0.8 },
        { goal_id: "g2", anchor_text: "b", threshold: 0.2 },
      ],
      version: 1,
    });
    expect(edge.activeThreshold).toBeCloseTo(0.2, 10);
  });

  it("a wired semantic adapter can escalate a semantic-gated goal even when motion is 0", () => {
    const { edge, signals } = makeEdge();
    // A goal that gates on the semantic channel (gate_on:"semantic") — mirrors the Python reference,
    // where semantic only arms escalation for a goal that asked for it (a motion goal stays motion-only;
    // a hot semantic channel does NOT silently escalate it — the opt-in invariant).
    edge.applyWatchSpec({
      conditions: [
        { goal_id: "g1", anchor_text: "the user is drinking coffee", threshold: 0.3, gate_on: "semantic" },
      ],
      version: 1,
    });
    edge.setSemanticAdapter(() => 0.9); // pretend CLIP fired
    const still = makeFrame(8, 8, 120);
    // First frame has no motion history (motion 0), but semantic 0.9 clears the 0.3 threshold.
    const sig = edge.processFrame(still);
    expect(sig).not.toBeNull();
    expect(sig.motion).toBe(0);
    expect(sig.semantic).toBeCloseTo(0.9, 10);
    expect(signals).toHaveLength(1);
    // It keeps escalating on every still frame while the semantic gate is hot.
    edge.processFrame(still);
    expect(signals).toHaveLength(2);
  });

  it("fixed cadence escalates on a still stream", () => {
    const { edge, signals } = makeEdge();
    edge.applyWatchSpec({
      conditions: [
        {
          goal_id: "screen",
          anchor_text: "the shared screen needs help",
          threshold: 0.3,
          cadence_s: 2,
          gate_on: "fixed",
        },
      ],
      version: 1,
    });
    const still = makeFrame(8, 8, 120);
    expect(edge.processFrame(still)).not.toBeNull(); // t=1000
    expect(edge.processFrame(still)).toBeNull(); // t=1001, cadence suppressed
    expect(edge.processFrame(still)).not.toBeNull(); // t=1002
    expect(signals).toHaveLength(2);
  });

  it("a wake policy can escalate from an audio event, on a goal that asked to be fused", () => {
    // Rewritten. This used to author no `gate_on` at all and pass, because a present wake policy
    // overrode every condition's channel. Fusion is now a channel a goal OPTS IN to, and `audio` is
    // the channel a microphone reading populates (`audio_event` names an acoustic-event CLASS, which
    // nothing on this edge classifies).
    const { edge, signals } = makeEdge({ wakePolicy: { weights: { audio: 1.0 } } });
    edge.applyWatchSpec(FUSED_WATCHSPEC); // threshold 0.3, gate_on "fused"
    const still = makeFrame(8, 8, 120);
    const sig = edge.processFrame(still, { energy: 0.5 });
    expect(sig).not.toBeNull();
    expect(sig.motion).toBe(0);
    expect(signals).toHaveLength(1);
  });

  it("a wake policy no longer overrides the channel a goal asked to gate on", () => {
    // BUG-A-01-04's JS twin. The same policy, the same loud audio, on a goal gated on `motion`:
    // the goal's own channel decides, and a still frame has no motion. Before, the policy replaced
    // the motion branch for every condition and this escalated.
    const { edge, signals } = makeEdge({ wakePolicy: { weights: { audio: 1.0 } } });
    edge.applyWatchSpec(PY_WATCHSPEC); // threshold 0.3, gate_on defaults to "motion"
    const still = makeFrame(8, 8, 120);
    expect(edge.processFrame(still, { energy: 0.5 })).toBeNull();
    expect(edge.processFrame(still, { energy: 0.9 })).toBeNull();
    expect(signals).toHaveLength(0);
  });

  it("a fused goal with no wake policy wired scores zero rather than falling back to motion", () => {
    const { edge } = makeEdge(); // no wakePolicy
    edge.applyWatchSpec(FUSED_WATCHSPEC);
    const still = makeFrame(8, 8, 0);
    const moved = makeFrame(8, 8, 255); // motion ~1.0, well over the 0.3 bar
    expect(edge.processFrame(still)).toBeNull();
    expect(edge.processFrame(moved)).toBeNull();
  });

  it("a wake policy can combine weak presence and semantic signals", () => {
    // `semantic_relevance` was an ALIAS of `semantic`, so this used to fuse one adapter reading with
    // itself. The caller now overlays `presence` (a channel it measures) beside the gate's own
    // `semantic`, and the noisy-OR combines two independent cues: 1-(1-0.24)(1-0.24) = 0.4224.
    const { edge } = makeEdge({ wakePolicy: { weights: { presence: 0.8, semantic: 0.8 } } });
    edge.applyWatchSpec(FUSED_WATCHSPEC); // threshold 0.3
    edge.setSemanticAdapter(() => 0.3);
    const still = makeFrame(8, 8, 120);
    const sig = edge.processFrame(still, { signals: { presence: 0.3 } });
    expect(sig).not.toBeNull();
    expect(sig.motion).toBe(0);
    expect(sig.semantic).toBeCloseTo(0.3, 10);
  });

  it("invokes the publisher slot with the gated frame + signal on escalation", () => {
    const { edge } = makeEdge();
    edge.applyWatchSpec(PY_WATCHSPEC);
    const published = [];
    edge.setPublisher((frame, signal) => published.push({ frame, signal }));
    const a = makeFrame(8, 8, 0);
    const b = makeFrame(8, 8, 255);
    edge.processFrame(a);
    edge.processFrame(b);
    expect(published).toHaveLength(1);
    expect(published[0].signal.motion).toBeGreaterThan(0.3);
  });

  it("onSignal returns an unsubscribe fn", () => {
    const { edge } = makeEdge();
    const got = [];
    const off = edge.onSignal((s) => got.push(s));
    off();
    const a = makeFrame(4, 4, 0);
    const b = makeFrame(4, 4, 255);
    edge.processFrame(a);
    edge.processFrame(b);
    expect(got).toHaveLength(0);
  });
});

const SPEC_WITH_DETECTORS = JSON.stringify({
  conditions: [
    { goal_id: "reps", anchor_text: "one squat rep", threshold: 0.3, region_hint: null, detectors: ["motion-periodicity"] },
  ],
  version: 1,
});

// Frames whose consecutive motion oscillates 0,1,0,1,... (pairs of still/inverted) -> 4 reps.
const OSC_FILLS = [0, 0, 255, 255, 0, 0, 255, 255, 0, 0];

describe("Edge — skill banks + the per-event Boundary up-channel (Option 2)", () => {
  it("runs a goal's selected skill and emits goal-tagged boundaries", () => {
    const boundaries = [];
    let t = 1000;
    const edge = new Edge({ onBoundary: (batch) => boundaries.push(...batch), clock: () => t++ });
    edge.applyWatchSpec(SPEC_WITH_DETECTORS);
    for (const f of OSC_FILLS) edge.processFrame(makeFrame(8, 8, f));

    expect(boundaries).toHaveLength(4);
    expect(boundaries.every((e) => e.goal_id === "reps")).toBe(true);
    expect(boundaries.every((e) => e.boundary.detector === "motion-periodicity" && e.boundary.kind === "peak")).toBe(true);
  });

  it("the Boundary channel is decoupled from the Tier0Signal channel (both flow)", () => {
    const signals = [];
    const boundaries = [];
    let t = 1000;
    const edge = new Edge({
      onSignal: (s) => signals.push(s),
      onBoundary: (batch) => boundaries.push(...batch),
      clock: () => t++,
    });
    edge.applyWatchSpec(SPEC_WITH_DETECTORS); // threshold 0.3
    for (const f of OSC_FILLS) edge.processFrame(makeFrame(8, 8, f));

    expect(boundaries).toHaveLength(4); // sparse events (per rep)
    expect(signals.length).toBeGreaterThan(0); // the high-motion frames also cleared the gate
    expect(signals.every((s) => Object.keys(s).sort().join() === "energy,motion,semantic,structural,ts")).toBe(true);
  });

  it("advertises its registered skills (the handshake basis)", () => {
    const edge = new Edge({});
    expect(edge.capabilities()).toContain("motion-periodicity");
    expect(edge.capabilities()).toContain("acoustic-onset");
  });

  it("drops an unregistered detector gracefully — records it, never crashes", () => {
    const edge = new Edge({});
    edge.applyWatchSpec(
      JSON.stringify({
        conditions: [{ goal_id: "g", anchor_text: "x", threshold: 0.3, region_hint: null, detectors: ["no-such-skill"] }],
        version: 1,
      })
    );
    expect(edge.unsupportedDetectors).toEqual([{ goal_id: "g", detector: "no-such-skill" }]);
    const boundaries = [];
    edge.onBoundary((batch) => boundaries.push(...batch));
    edge.processFrame(makeFrame(8, 8, 0));
    edge.processFrame(makeFrame(8, 8, 255)); // would-be motion, but no bank -> no boundaries, no throw
    expect(boundaries).toHaveLength(0);
  });
});

describe("Edge.advertise — the EdgeCapabilities message", () => {
  it("returns the registered skill set as an EdgeCapabilities message", () => {
    const edge = new Edge({});
    const msg = edge.advertise();
    expect(Array.isArray(msg.capabilities)).toBe(true);
    expect(msg.capabilities).toContain("motion-periodicity");
    expect(msg.capabilities).toContain("acoustic-onset");
  });
});

describe("Edge — salience modes (per-condition) + energy on the signal", () => {
  function localized(w, h, size = 2, val = 255) {
    const data = new Uint8Array(w * h);
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) data[r * w + c] = val;
    return { width: w, height: h, data };
  }

  it("a block_max condition escalates a localized event a mean condition misses", () => {
    const bmaxSpec = {
      conditions: [{ goal_id: "g1", anchor_text: "sip", threshold: 0.1, salience_stat: "block_max" }],
      version: 1,
    };
    const meanSpec = {
      conditions: [{ goal_id: "g1", anchor_text: "sip", threshold: 0.1, salience_stat: "mean" }],
      version: 1,
    };
    const still = makeFrame(8, 8, 0);
    const loc = localized(8, 8);

    const mean = makeEdge();
    mean.edge.applyWatchSpec(meanSpec);
    mean.edge.processFrame(still); // prime history
    expect(mean.edge.processFrame(loc)).toBeNull(); // mean misses the localized event

    const bmax = makeEdge();
    bmax.edge.applyWatchSpec(bmaxSpec);
    bmax.edge.processFrame(still);
    expect(bmax.edge.processFrame(loc)).not.toBeNull(); // block_max catches it
  });

  it("an ego_compensate condition suppresses a pan a mean condition fires on", () => {
    const w = 16, h = 16;
    const base = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) base[y * w + x] = (x * 60 + y * 30) % 256;
    const panned = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) panned[y * w + x] = base[y * w + Math.min(x + 1, w - 1)];
    const prev = { width: w, height: h, data: base };
    const cur = { width: w, height: h, data: panned };

    const mean = makeEdge();
    mean.edge.applyWatchSpec({
      conditions: [{ goal_id: "g", anchor_text: "x", threshold: 0.1, salience_stat: "mean" }],
      version: 1,
    });
    mean.edge.processFrame(prev);
    expect(mean.edge.processFrame(cur)).not.toBeNull(); // mean false-fires on the pan

    const ego = makeEdge();
    ego.edge.applyWatchSpec({
      conditions: [{ goal_id: "g", anchor_text: "x", threshold: 0.1, ego_compensate: true }],
      version: 1,
    });
    ego.edge.processFrame(prev);
    expect(ego.edge.processFrame(cur)).toBeNull(); // ego suppresses it
  });

  it("an ego block_max condition compares an aligned block score to its block threshold", () => {
    const w = 16, h = 16;
    const base = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) base[y * w + x] = (x * 17 + y * 31) % 256;
    const before = new Uint8Array(base);
    const after = new Uint8Array(base);
    for (let y = 0; y < 2; y++)
      for (let x = 0; x < 2; x++) {
        before[y * w + x] = 0;
        after[y * w + x] = 255;
      }

    const ego = makeEdge();
    ego.edge.applyWatchSpec({
      conditions: [{
        goal_id: "g", anchor_text: "x", threshold: 0.2,
        salience_stat: "block_max", ego_compensate: true,
      }],
      version: 1,
    });
    ego.edge.processFrame({ width: w, height: h, data: before });
    expect(ego.edge.processFrame({ width: w, height: h, data: after })).not.toBeNull();
  });

  it("an ego_compensate condition still honours its watch-out zone", () => {
    // A watch-out zone is a promise that movement elsewhere is ignored. Turning ego compensation on
    // used to revoke it silently: `conditionMotion` returned the WHOLE-FRAME residual while the
    // condition still carried and still reported its `region_hint`. Twin of Python
    // `EgoResidualBoundaryTests.test_an_ego_goal_still_honours_its_watch_out_zone`.
    const w = 16, h = 16;
    const base = new Uint8Array(w * h);
    let s = 20260731;
    for (let i = 0; i < w * h; i++) {
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      base[i] = (s >>> 16) & 0xff;
    }
    const a = new Uint8Array(base);
    const b = new Uint8Array(base);
    for (let y = 10; y < 16; y++)
      for (let x = 10; x < 16; x++) {
        a[y * w + x] = 0;      // every changed pixel is in the BOTTOM-RIGHT quadrant
        b[y * w + x] = 255;
      }
    const prev = { width: w, height: h, data: a };
    const cur = { width: w, height: h, data: b };
    const topLeft = { x: 0, y: 0, w: 0.5, h: 0.5 };

    const noZone = makeEdge();
    noZone.edge.applyWatchSpec({
      conditions: [{ goal_id: "g", anchor_text: "x", threshold: 0.1, ego_compensate: true }],
      version: 1,
    });
    noZone.edge.processFrame(prev);
    expect(noZone.edge.processFrame(cur)).not.toBeNull(); // the control: unmasked, it fires

    const zoned = makeEdge();
    zoned.edge.applyWatchSpec({
      conditions: [
        { goal_id: "g", anchor_text: "x", threshold: 0.1, region_hint: topLeft, ego_compensate: true },
      ],
      version: 1,
    });
    zoned.edge.processFrame(prev);
    expect(zoned.edge.processFrame(cur)).toBeNull(); // nothing happened in-zone
  });

  it("carries audio energy through onto the emitted Tier0Signal", () => {
    const { edge } = makeEdge();
    edge.applyWatchSpec(PY_WATCHSPEC);
    const a = makeFrame(8, 8, 0);
    const b = makeFrame(8, 8, 255); // full motion -> escalates
    edge.processFrame(a, { energy: 0.42 });
    const sig = edge.processFrame(b, { energy: 0.42 });
    expect(sig).not.toBeNull();
    expect(sig.energy).toBeCloseTo(0.42, 10);
  });

  it("the motion channel is clamped before the threshold compare, like structural and semantic", () => {
    // `conditionMotion`'s result was compared `>= c.threshold` RAW, while the structural and
    // semantic adapters were both read through clamp01 — so the one channel that is always live was
    // the only one with no floor or ceiling under it. `data` is documented as `Uint8Array|number[]`
    // and a plain array is not range-checked, so a per-pixel diff above 255 pushes the salience
    // above 1. A threshold above 1 must be unsatisfiable on EVERY channel, or the three channels do
    // not share a scale.
    const outOfRange = (w, h, v) => ({ width: w, height: h, data: new Array(w * h).fill(v) });
    const spec = (threshold, gate_on) => ({
      conditions: [{ goal_id: "g", anchor_text: "x", threshold, gate_on }],
      version: 1,
    });

    const hot = makeEdge();
    hot.edge.applyWatchSpec(spec(2, "motion"));
    hot.edge.processFrame(outOfRange(4, 4, 0));
    expect(hot.edge.processFrame(outOfRange(4, 4, 100000))).toBeNull();

    // The control: the same frames clear any threshold that is actually inside the salience range.
    const ok = makeEdge();
    ok.edge.applyWatchSpec(spec(1, "motion"));
    ok.edge.processFrame(outOfRange(4, 4, 0));
    expect(ok.edge.processFrame(outOfRange(4, 4, 100000))).not.toBeNull();

    // And the two channels that already had the clamp, for the same threshold.
    for (const gate_on of ["structural", "semantic"]) {
      const e = makeEdge();
      e.edge.applyWatchSpec(spec(2, gate_on));
      e.edge.setStructuralAdapter(() => 5).setSemanticAdapter(() => 5);
      e.edge.processFrame(makeFrame(4, 4, 0));
      expect(e.edge.processFrame(makeFrame(4, 4, 255)), gate_on).toBeNull();
    }
  });

  it("a DEGENERATE region_hint is REFUSED at applyWatchSpec, not silently deployed (FIND-PARITY-01)", () => {
    // REWRITTEN. This test used to assert that a WatchSpec carrying `{x:0,y:0,w:0,h:1}` is accepted
    // and then escalates 0 times — "frozen here so the behaviour is at least documented and a future
    // decision to reject it shows up as a deliberate change". This is that change. A permanent
    // never-fire that is indistinguishable from a working watch is not a behaviour to freeze: the
    // same authored zone admitted EVERY frame in the Python harness (the bounds truncate the origin
    // and round the far bound, so a zero-extent zone can still be one real pixel column wide) and no
    // frame here. The spec is now refused at the door instead of deployed deaf.
    const still = makeFrame(32, 32, 0);
    const moved = makeFrame(32, 32, 255);
    const withRegion = (region_hint) => ({
      conditions: [{ goal_id: "g", anchor_text: "x", threshold: 0.1, region_hint }],
      version: 1,
    });

    const degenerate = makeEdge();
    expect(() => degenerate.edge.applyWatchSpec(withRegion({ x: 0, y: 0, w: 0, h: 1 }))).toThrow(
      /w and h must be > 0/
    );
    // The FIND-PARITY-01 zone itself — the one whose bounds do NOT collapse — is refused by the same
    // rule, so the runtimes can no longer be handed a goal they answer differently.
    expect(() => degenerate.edge.applyWatchSpec(withRegion({ x: 0.1, y: 0.1, w: 0, h: 0.5 }))).toThrow(
      /w and h must be > 0/
    );
    // A refused spec is not installed: the edge is left exactly as it was, not half-configured.
    expect(degenerate.edge.watchSpec).toBeNull();

    const control = makeEdge();
    control.edge.applyWatchSpec(withRegion({ x: 0, y: 0, w: 1, h: 1 }));
    control.edge.processFrame(still);
    control.edge.processFrame(moved);
    expect(control.signals).toHaveLength(1);
  });
});

describe("Edge — the two runtimes must admit the same frames (FIND-PARITY-02 / -03)", () => {
  // A 24x18 alternating-luma clip: every second frame is a total flip, so motion is 1.0 on frames
  // 1..5 and the ONLY thing that can suppress an admission is a cadence brake. These are the JS half
  // of the cross-runtime vectors in `parity/compare.py` (GATE_CASES) — pinned here too so a fork
  // fails the unit suite as well as the release gate.
  const alternating = [];
  for (let i = 0; i < 6; i++) alternating.push(makeFrame(24, 18, i % 2 === 0 ? 0 : 255));

  function marksFor(spec, { frames = alternating, apply = true } = {}) {
    const signals = [];
    let cursor = 0;
    const edge = new Edge({ onSignal: (s) => signals.push(s), clock: () => cursor });
    if (apply) edge.applyWatchSpec(spec);
    let marks = "";
    for (cursor = 0; cursor < frames.length; cursor++) {
      marks += edge.processFrame(frames[cursor]) ? "1" : ".";
    }
    return { marks, edge, signals };
  }

  const goal = (goal_id, extra = {}) => ({
    goal_id, anchor_text: "a hand", threshold: 0.2, cadence_s: 2.0, ...extra,
  });

  it("every goal that fires gets its cadence brake stamped, not just the first", () => {
    // The defect: `_shouldEscalate` returned on the first firing condition, so the second goal's
    // `_lastEscalate` stayed at -Infinity and it admitted again on the very next frame — inside the
    // cadence window the deployment authored. Python `MotionGate.evaluate` stamps every goal that
    // fired, and measured `.1.1.1` where the edge measured `.11111`.
    const two = marksFor({ version: 1, conditions: [goal("first"), goal("second", { anchor_text: "a cup" })] });
    expect(two.marks).toBe(".1.1.1");
    // The control that rules out a threshold/salience explanation: ONE goal on the same cadence over
    // the same clip admits the same frames. Two goals must not cost more than one.
    const one = marksFor({ version: 1, conditions: [goal("only")] });
    expect(one.marks).toBe(".1.1.1");
    // ...and the stamp is per goal, both of them, at the frame each last fired.
    expect(two.edge._lastEscalate.get("first")).toBe(5);
    expect(two.edge._lastEscalate.get("second")).toBe(5);
  });

  it("three goals sharing a window are all braked — the accumulation is not two-deep", () => {
    const three = marksFor({
      version: 1,
      conditions: [goal("a"), goal("b"), goal("c")],
    });
    expect(three.marks).toBe(".1.1.1");
    for (const id of ["a", "b", "c"]) expect(three.edge._lastEscalate.get(id)).toBe(5);
  });

  it("a goal below its threshold does not get a brake stamped by a goal that fired", () => {
    // The accumulating loop must not become "stamp everything once anything fires": only conditions
    // that actually cleared their own bar are braked. `deaf` can never fire (threshold 1.5), so it
    // must never appear in the throttle map.
    const mixed = marksFor({
      version: 1,
      conditions: [goal("loud"), goal("deaf", { threshold: 1.5 })],
    });
    expect(mixed.marks).toBe(".1.1.1");
    expect(mixed.edge._lastEscalate.has("deaf")).toBe(false);
  });

  it("an APPLIED spec with no conditions never escalates — the last watch was cancelled", () => {
    // The defect: an empty applied spec fell through to `salience >= defaultThreshold`, so the
    // device kept clearing 0.05 on ordinary motion and kept handing frames to the publisher for a
    // watch nobody has. Python's gate has always answered this input with "nothing to look for".
    const cancelled = marksFor({ version: 1, conditions: [] });
    expect(cancelled.marks).toBe("......");
    expect(cancelled.signals).toHaveLength(0);
    expect(cancelled.edge.activeThreshold).toBe(Infinity);
  });

  it("...but an edge that was NEVER given a spec keeps the motion-only fallback", () => {
    // The two states are not the same state, and only one of them changed. A bare `new Edge()` has
    // no instruction, so it still gates on defaultThreshold.
    const bare = marksFor(null, { apply: false });
    expect(bare.marks).toBe(".11111");
    expect(bare.edge.activeThreshold).toBeCloseTo(0.05, 10);
    // The distinction is reachable from the same edge: cancel everything and it goes quiet.
    bare.edge.applyWatchSpec({ version: 1, conditions: [] });
    expect(bare.edge.processFrame(alternating[0])).toBeNull();
  });

  it("a publisher is not invoked for a cancelled watch (the privacy consequence)", () => {
    const published = [];
    let cursor = 0;
    const edge = new Edge({ clock: () => cursor });
    edge.setPublisher((frame, signal) => published.push(signal.ts));
    edge.applyWatchSpec({ version: 1, conditions: [] });
    for (cursor = 0; cursor < alternating.length; cursor++) edge.processFrame(alternating[cursor]);
    expect(published).toHaveLength(0);
  });
});
