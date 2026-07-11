import { describe, it, expect } from "vitest";
import { Edge } from "../src/edge.js";
import { makeFrame } from "../src/motion.js";
import { parseTier0Signal } from "../src/contract.js";

const PY_WATCHSPEC =
  '{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null}],"version":1}';

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

  it("falls back to defaultThreshold when no WatchSpec / no conditions", () => {
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

  it("a wired semantic adapter can escalate even when motion is 0", () => {
    const { edge, signals } = makeEdge();
    edge.applyWatchSpec(PY_WATCHSPEC); // threshold 0.3
    edge.setSemanticAdapter(() => 0.9); // pretend CLIP fired
    const still = makeFrame(8, 8, 120);
    // First frame has no motion history (motion 0), but semantic 0.9 alone
    // clears the 0.3 threshold -> escalates anyway.
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

  it("a wake policy can escalate from an audio event without changing default behavior", () => {
    const { edge, signals } = makeEdge({
      wakePolicy: { weights: { audio_event: 1.0 } },
    });
    edge.applyWatchSpec(PY_WATCHSPEC); // threshold 0.3
    const still = makeFrame(8, 8, 120);
    const sig = edge.processFrame(still, { energy: 0.5 });
    expect(sig).not.toBeNull();
    expect(sig.motion).toBe(0);
    expect(signals).toHaveLength(1);
  });

  it("a wake policy can combine weak presence and semantic-relevance signals", () => {
    const { edge } = makeEdge({
      wakePolicy: { weights: { presence: 0.8, semantic_relevance: 0.8 } },
    });
    edge.applyWatchSpec(PY_WATCHSPEC); // threshold 0.3
    const still = makeFrame(8, 8, 120);
    const sig = edge.processFrame(still, {
      signals: { presence: 0.3, semantic_relevance: 0.3 },
    });
    expect(sig).not.toBeNull();
    expect(sig.motion).toBe(0);
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
