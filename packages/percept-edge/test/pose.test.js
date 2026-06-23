import { describe, it, expect } from "vitest";
import { openness, PoseOpenness, registerPoseSkill } from "../src/pose.js";
import { SkillBank, registered, signalOf } from "../src/skills.js";

// A 33-landmark BlazePose set for an OPEN (legs apart, arms up) vs CLOSED (together, arms down) pose.
function poseFor(open) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
  lm[11] = { x: 0.4, y: 0.5 }; // L shoulder
  lm[12] = { x: 0.6, y: 0.5 }; // R shoulder  -> shoulder width 0.2
  if (open) {
    lm[27] = { x: 0.3, y: 0.9 }; // L ankle
    lm[28] = { x: 0.7, y: 0.9 }; // R ankle    -> spread 0.4
    lm[15] = { x: 0.4, y: 0.2 }; // L wrist up (y above shoulder)
    lm[16] = { x: 0.6, y: 0.2 }; // R wrist up -> wristUp 0.3+0.3
  } else {
    lm[27] = { x: 0.45, y: 0.9 };
    lm[28] = { x: 0.55, y: 0.9 }; // spread 0.1
    lm[15] = { x: 0.4, y: 0.6 }; // wrists down (y below shoulder)
    lm[16] = { x: 0.6, y: 0.6 }; // wristUp 0
  }
  return lm;
}

function fakeLandmarker() {
  let loads = 0;
  return {
    get loads() {
      return loads;
    },
    async load() {
      loads += 1;
    },
    detect(frame) {
      return frame && frame.person === false ? null : poseFor(frame.open);
    },
  };
}

describe("pose-openness — model-backed lazy skill", () => {
  it("openness = ankle-spread/shoulder-width + wrist-elevation", () => {
    expect(openness(poseFor(true))).toBeCloseTo(0.4 / 0.2 + 0.6, 5); // 2.6
    expect(openness(poseFor(false))).toBeCloseTo(0.1 / 0.2 + 0.0, 5); // 0.5
  });

  it("counts whole-body reps as one peak per open->close cycle", () => {
    const skill = new PoseOpenness({ landmarker: fakeLandmarker() });
    const frames = [false, true, false, true, false, true, false].map((open) => ({ open }));
    const out = [];
    frames.forEach((f, t) => out.push(...skill.push(t, f)));
    expect(out).toHaveLength(3); // three opens -> three reps
    expect(out.every((b) => b.detector === "pose-openness" && b.kind === "peak")).toBe(true);
  });

  it("loads the model lazily — on load(), not at construct", async () => {
    const lm = fakeLandmarker();
    const skill = new PoseOpenness({ landmarker: lm });
    expect(lm.loads).toBe(0); // constructing did not load the model
    await skill.load();
    expect(lm.loads).toBe(1);
  });

  it("holds the last openness when no person is detected (no spurious reset)", () => {
    const skill = new PoseOpenness({ landmarker: fakeLandmarker() });
    const out = [];
    // open, then a gap with no person, then close -> still one rep (the gap doesn't break the waveform)
    [{ open: true }, { person: false }, { open: false }].forEach((f, t) => out.push(...skill.push(t, f)));
    expect(out).toHaveLength(1);
  });
});

describe("registerPoseSkill — opt-in registration with a product model", () => {
  it("registers pose-openness as a frame-routed, brain-selectable skill", async () => {
    registerPoseSkill(fakeLandmarker());
    expect(registered()).toContain("pose-openness");
    expect(signalOf("pose-openness")).toBe("frame");

    const bank = new SkillBank(["pose-openness"]);
    await bank.load(); // lazy-loads the pose model via the bank
    const out = [];
    [false, true, false, true, false].forEach((open, t) => out.push(...bank.push(t, { frame: { open } })));
    expect(out.length).toBeGreaterThanOrEqual(1);
    expect(out.every((b) => b.detector === "pose-openness")).toBe(true);
  });
});
