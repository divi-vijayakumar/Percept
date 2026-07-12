import { describe, it, expect } from "vitest";
import { motionScore, egoResidualScore, makeFrame } from "../src/motion.js";

// A small `size x size` patch in the top-left corner on an otherwise still WxH frame.
function localized(w, h, size = 2, val = 255) {
  const data = new Uint8Array(w * h);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) data[r * w + c] = val;
  return { width: w, height: h, data };
}

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

describe("motionScore — localized statistic (Mode A)", () => {
  it("block_max recovers a localized event the whole-frame mean misses", () => {
    const still = makeFrame(8, 8, 0);
    const loc = localized(8, 8);
    const mean = motionScore(still, loc);
    const bmax = motionScore(still, loc, { stat: "block_max" });
    expect(mean).toBeLessThan(0.1); // below the shipped gate -> missed by mean
    expect(bmax).toBeGreaterThanOrEqual(0.1); // caught by block_max
    expect(bmax).toBeGreaterThan(mean);
  });

  it("stat 'mean' is the default path", () => {
    const a = makeFrame(8, 8, 30);
    const b = makeFrame(8, 8, 200);
    expect(motionScore(a, b, { stat: "mean" })).toBe(motionScore(a, b));
  });

  it("block_max >= p95 >= 0, and block_max >= mean", () => {
    const still = makeFrame(8, 8, 0);
    const loc = localized(8, 8, 3);
    const mean = motionScore(still, loc);
    const p95 = motionScore(still, loc, { stat: "p95" });
    const bmax = motionScore(still, loc, { stat: "block_max" });
    expect(bmax).toBeGreaterThanOrEqual(p95);
    expect(p95).toBeGreaterThanOrEqual(0);
    expect(bmax).toBeGreaterThanOrEqual(mean - 1e-12);
  });

  it("a region masks the block statistic to its zone", () => {
    // patch bottom-right, region = top-left quadrant -> nothing in-zone
    const data = new Uint8Array(64);
    for (let r = 6; r < 8; r++) for (let c = 6; c < 8; c++) data[r * 8 + c] = 255;
    const cur = { width: 8, height: 8, data };
    const roi = { x: 0, y: 0, w: 0.5, h: 0.5 };
    expect(motionScore(makeFrame(8, 8, 0), cur, { stat: "block_max", region: roi })).toBe(0);
  });

  it("throws on an unknown stat", () => {
    expect(() => motionScore(makeFrame(8, 8, 0), makeFrame(8, 8, 9), { stat: "median" })).toThrow();
  });
});

describe("egoResidualScore — ego-motion compensation (Mode B)", () => {
  it("returns 0 with no previous frame", () => {
    expect(egoResidualScore(null, makeFrame(8, 8, 100))).toBe(0);
  });

  it("never exceeds the full-frame mean (the (0,0) shift is in the search)", () => {
    const a = makeFrame(8, 8, 0);
    const b = localized(8, 8, 4);
    expect(egoResidualScore(a, b)).toBeLessThanOrEqual(motionScore(a, b) + 1e-12);
  });

  it("compensates a pure horizontal pan (mean high, residual ~0)", () => {
    const w = 16, h = 16;
    const base = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) base[y * w + x] = (x * 60 + y * 30) % 256;
    const panned = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) panned[y * w + x] = base[y * w + Math.min(x + 1, w - 1)];
    const prev = { width: w, height: h, data: base };
    const cur = { width: w, height: h, data: panned };
    expect(motionScore(prev, cur)).toBeGreaterThan(0.1); // a pan looks like heavy motion...
    expect(egoResidualScore(prev, cur, { k: 4 })).toBeLessThan(0.1); // ...but the residual is ~0
  });
});
