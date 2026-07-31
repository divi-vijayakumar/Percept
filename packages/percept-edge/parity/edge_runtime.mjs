// edge_runtime.mjs — run percept-edge detectors + gate on test cases read from stdin (JSON), emit
// results as JSON on stdout. Driven by parity/compare.py to prove the JS edge route ≡ the Python
// percept-harness route (same frozen contract, same algorithms), and to measure edge frame latency.
//
//   in : { motionSignals: number[][], energySignals: number[][],
//          framePairs: [{w,h,a:number[],b:number[]}],
//          saliencePairs: [{w,h,a,b,stat?,region?,blockGrid?}],   // blockGrid defaults to [8,6]
//          egoPairs: [{w,h,a,b,k?,region?}],                        // region defaults to null
//          wakeCases: [{signals:object, policy?:object|null}],       // policy defaults to null
//          latency?: {width,height,frames} }
//   out: { motionBoundaries: Boundary[][], onsetBoundaries: Boundary[][], motionScores: number[],
//          saliences: number[], egoScores: number[],
//          wakeScores: number[], wakeTopChannels: (string|null)[], wakeFlags: boolean[],
//          latency?: {frames,width,height,totalMs,perFrameUs,fps} }

import { MotionPeriodicity, AcousticOnset, wakeScore, shouldWake } from "../src/skills.js";
import { motionScore, egoResidualScore } from "../src/motion.js";
import { Edge } from "../src/edge.js";
import { OrderedAudioBuffer, pcm16Energy } from "../src/vad.js";

function readStdin() {
  return new Promise((resolve) => {
    let s = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (d) => (s += d));
    process.stdin.on("end", () => resolve(s));
  });
}

function runDetector(DetClass, signal) {
  const det = new DetClass();
  const out = [];
  signal.forEach((v, t) => out.push(...det.push(t, v)));
  return out;
}

const cases = JSON.parse(await readStdin());

const result = {
  motionBoundaries: (cases.motionSignals || []).map((s) => runDetector(MotionPeriodicity, s)),
  onsetBoundaries: (cases.energySignals || []).map((s) => runDetector(AcousticOnset, s)),
  motionScores: (cases.framePairs || []).map(({ w, h, a, b }) =>
    motionScore({ width: w, height: h, data: a }, { width: w, height: h, data: b })
  ),
  saliences: (cases.saliencePairs || []).map(({ w, h, a, b, stat, region, blockGrid }) =>
    motionScore(
      { width: w, height: h, data: a },
      { width: w, height: h, data: b },
      { stat: stat ?? "mean", region: region ?? null, blockGrid: blockGrid ?? [8, 6] }
    )
  ),
  egoScores: (cases.egoPairs || []).map(({ w, h, a, b, k, region }) =>
    egoResidualScore(
      { width: w, height: h, data: a },
      { width: w, height: h, data: b },
      { k: k ?? 4, region: region ?? null }
    )
  ),
  // The wake fusion. `wakeScore` and `shouldWake` are compared separately on purpose: the score and
  // the top channel used to be computed from two different populations, so a runtime could agree on
  // one and fork on the other.
  wakeScores: (cases.wakeCases || []).map(({ signals, policy }) => wakeScore(signals, policy ?? null)),
  wakeTopChannels: (cases.wakeCases || []).map(
    ({ signals, policy }) => shouldWake(signals, policy ?? null).top_channel,
  ),
  wakeFlags: (cases.wakeCases || []).map(({ signals, policy }) => shouldWake(signals, policy ?? null).wake),
  audioEnergies: (cases.audioEnergyVectors || []).map(({ samples }) => pcm16Energy(samples)),
  audioTimelines: (cases.audioTimelineVectors || []).map(({ units }) => {
    const timeline = new OrderedAudioBuffer({ seconds: 10 });
    const results = units.map((unit) => timeline.push(unit));
    return {
      results,
      order: timeline.units().map((unit) => unit.seq),
      directions: timeline.units().map((unit) => unit.direction),
      participants: timeline.units().map((unit) => unit.participant_id),
      boundaryKinds: timeline.boundaries().map((boundary) => boundary.kind),
    };
  }),
};

if (cases.latency) {
  const { width, height, frames } = cases.latency;
  let clk = 0;
  const edge = new Edge({ clock: () => clk++ });
  edge.applyWatchSpec({
    conditions: [{ goal_id: "reps", anchor_text: "rep", threshold: 0.3, region_hint: null, detectors: ["motion-periodicity"] }],
    version: 1,
  });
  const fs = [];
  for (let i = 0; i < frames; i++) {
    const fill = i % 4 < 2 ? 0 : 255; // oscillating motion so the skill + gate both do real work
    fs.push({ width, height, data: new Uint8Array(width * height).fill(fill) });
  }
  const t0 = process.hrtime.bigint();
  for (const f of fs) edge.processFrame(f);
  const t1 = process.hrtime.bigint();
  const totalMs = Number(t1 - t0) / 1e6;
  result.latency = {
    frames, width, height, totalMs,
    perFrameUs: (totalMs * 1000) / frames,
    fps: Math.round(frames / (totalMs / 1000)),
  };
}

process.stdout.write(JSON.stringify(result));
