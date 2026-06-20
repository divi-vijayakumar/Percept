import { describe, it, expect } from "vitest";
import { motionScore, makeFrame } from "../src/motion.js";

describe("motionScore", () => {
  it("returns 0 with no previous frame (first frame, no history)", () => {
    expect(motionScore(null, makeFrame(4, 4, 100))).toBe(0);
  });

  it("identical (still) frames score ~0", () => {
    const a = makeFrame(8, 8, 120);
    const b = makeFrame(8, 8, 120);
    expect(motionScore(a, b)).toBe(0);
  });

  it("a fully inverted frame scores ~1 (maximum motion)", () => {
    const a = makeFrame(8, 8, 0);
    const b = makeFrame(8, 8, 255);
    expect(motionScore(a, b)).toBeCloseTo(1, 10);
  });

  it("a large change scores high (> 0.4); a tiny change scores low (< 0.05)", () => {
    const base = makeFrame(8, 8, 100);
    const big = makeFrame(8, 8, 240);
    const tiny = makeFrame(8, 8, 103);
    const bigScore = motionScore(base, big);
    const tinyScore = motionScore(base, tiny);
    expect(bigScore).toBeGreaterThan(0.4);
    expect(tinyScore).toBeLessThan(0.05);
    expect(bigScore).toBeGreaterThan(tinyScore);
  });

  it("is deterministic (same inputs -> same number)", () => {
    const a = makeFrame(8, 8, 30);
    const b = makeFrame(8, 8, 200);
    expect(motionScore(a, b)).toBe(motionScore(a, b));
  });

  it("scores partial motion proportionally (half the pixels change)", () => {
    // 4x4 frame: top half = 0, bottom half flipped to 255 in the second frame.
    const data0 = new Uint8Array(16).fill(0);
    const data1 = new Uint8Array(16).fill(0);
    for (let i = 8; i < 16; i++) data1[i] = 255;
    const a = { width: 4, height: 4, data: data0 };
    const b = { width: 4, height: 4, data: data1 };
    // Half the pixels move by full range -> mean abs diff = 127.5 -> ~0.5.
    expect(motionScore(a, b)).toBeCloseTo(0.5, 2);
  });

  it("throws on a frame-size mismatch", () => {
    expect(() => motionScore(makeFrame(4, 4, 0), makeFrame(8, 8, 0))).toThrow();
  });
});
