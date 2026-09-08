import { describe, it, expect } from "vitest";
import { motionScore, egoResidualScore, makeFrame } from "../src/motion.js";

// A small `size x size` patch in the top-left corner on an otherwise still WxH frame.
function localized(w, h, size = 2, val = 255) {
  const data = new Uint8Array(w * h);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) data[r * w + c] = val;
  return { width: w, height: h, data };
}

// A frame with exactly the listed flat indices set to 255, everything else 0.
function hotPixels(w, h, indices) {
  const data = new Uint8Array(w * h);
  for (const i of indices) data[i] = 255;
  return { width: w, height: h, data };
}

// A deterministic non-uniform frame — the same generator the parity vectors use.
function noise(w, h, mul, add) {
  const data = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) data[i] = (i * mul + add) % 256;
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

// ---------------------------------------------------------------------------
// Row JS-4 — cross-runtime bit-identity of the block-boundary and region math.
//
// Every expected value below is a hardcoded literal (or a hand-derived integer ratio), verified
// once against `percept.harness.tier0.motion_score` / `.ego_residual_score` at this commit. No
// expectation is read back out of the implementation — where one call is compared to another it is
// two DIFFERENT code paths being pinned together, and the value itself is pinned separately by a
// literal. An inequality or a `toBeCloseTo` cannot see a boundary fork; a self-referential
// expectation cannot see anything at all. If a value here changes, the two runtimes have diverged
// and the same media now admits differently on the edge and on the server.
//
// Each block below was mutation-checked: breaking the seam it names (half-to-even -> Math.round,
// floor-of-product -> Math.round, the integer rank -> a positional one, the clamp -> a no-op, the
// ego overlap origin -> off by dx) makes at least one assertion in it fail.
// ---------------------------------------------------------------------------

describe("motionScore — block boundaries on a non-divisible frame (row JS-4)", () => {
  // 13 x 11 divides evenly by neither grid axis of the shipped default [8, 6]. Floor-of-product
  // boundaries put block (0,0) at exactly one pixel wide and one pixel tall, so a single hot pixel
  // at the origin saturates it. A grid built by rounding, by ceil, or by a float step would give
  // that block a different pixel count and a different score.
  const W = 13;
  const H = 11;
  const still = makeFrame(W, H, 0);
  const oneHot = hotPixels(W, H, [0]);

  it("mean is exactly 1/143 — the whole-area denominator is the integer pixel count", () => {
    expect(motionScore(still, oneHot)).toBe(0.006993006993006993);
    // Same real number, written the other way; both are the correctly-rounded double.
    expect(motionScore(still, oneHot)).toBe(1 / 143);
  });

  it("block_max is exactly 1 — the origin block is a single pixel under floor boundaries", () => {
    // Column edges are (i * 13) // 8 = 0,1,3,4,6,8,9,11,13; row edges are (j * 11) // 6 =
    // 0,1,3,5,7,9,11. Block (0,0) therefore spans cols [0,1) x rows [0,1) = 1 px.
    expect(motionScore(still, oneHot, { stat: "block_max" })).toBe(1);
  });

  it("p95 is exactly 0 — one saturated block out of 48 sits above the nearest rank", () => {
    // n = 48 -> idx = (95*48 + 99) // 100 - 1 = 45. Sorted ascending that is still a zero block;
    // the lone 1.0 lands at index 47.
    expect(motionScore(still, oneHot, { stat: "p95" })).toBe(0);
  });

  it("the three statistics disagree by construction, so none can stand in for another", () => {
    const mean = motionScore(still, oneHot);
    const bmax = motionScore(still, oneHot, { stat: "block_max" });
    const p95 = motionScore(still, oneHot, { stat: "p95" });
    expect(bmax).toBe(1);
    expect(p95).toBe(0);
    expect(mean).toBeGreaterThan(p95);
    expect(mean).toBeLessThan(bmax);
  });

  it("a full-frame region reaches the identical float as the no-region mean path", () => {
    // Two different summation loops (motionScoreRegion vs the flat mean loop) over the same 143
    // pixels. Exact equality, not closeTo: a reordered accumulation would still be a fork.
    const a = noise(W, H, 7, 0);
    const b = noise(W, H, 5, 9);
    const full = { x: 0, y: 0, w: 1, h: 1 };
    expect(motionScore(a, b, { region: full })).toBe(motionScore(a, b));
  });

  it("the non-divisible noise pair sums to exactly 11007 over 143 pixels", () => {
    // Hand-derived, not read off the implementation: with a[i] = 7i mod 256 and
    // b[i] = (5i+9) mod 256, |a-b| is piecewise linear in i with breaks where either sequence
    // wraps (i = 37, 50, 74, 101, 110) and where 2i-265 changes sign (i = 133). Summing the six
    // arithmetic runs gives 1049 + 2327 + 2736 + 2457 + 1809 + 629 = 11007.
    const a = noise(W, H, 7, 0);
    const b = noise(W, H, 5, 9);
    expect(motionScore(a, b)).toBe(11007 / (143 * 255));
    expect(Math.round(motionScore(a, b) * 143 * 255)).toBe(11007);
  });
});

describe("motionScore — the banker's-rounding region bound (row JS-4)", () => {
  // This is the single line where a naive JS port forks. `_region_bounds` uses Python's round(),
  // which is round-half-to-even: round(2.5) == 2. JS Math.round(2.5) == 3. On an 8x5 frame a region
  // of half the height lands exactly on 2.5, so the two conventions include a different number of
  // rows — and a row of real motion falls on the wrong side of the boundary.
  const W = 8;
  const H = 5;
  const still = makeFrame(W, H, 0);
  const rowIdx = (r) => Array.from({ length: W }, (_, c) => r * W + c);
  const row1Moved = hotPixels(W, H, rowIdx(1));
  const row2Moved = hotPixels(W, H, rowIdx(2));
  const topHalf = { x: 0, y: 0, w: 1, h: 0.5 };

  it("the whole-frame mean of one moved row out of five is exactly 0.2", () => {
    expect(motionScore(still, row2Moved)).toBe(0.2);
    expect(motionScore(still, row1Moved)).toBe(0.2);
  });

  it("row 2 is OUTSIDE the top-half region: exactly 0, because half-to-even gives y1 = 2", () => {
    // A Math.round port would set y1 = 3, include row 2, and report 2040 / (24 * 255) =
    // 0.3333333333333333 here. That is the fork: same media, different admission verdict.
    expect(motionScore(still, row2Moved, { region: topHalf })).toBe(0);
  });

  it("row 1 IS inside the same region and scores exactly 0.5 — the zone is not merely empty", () => {
    // Without this the previous assertion would also pass if regionBounds collapsed to an empty
    // span for some unrelated reason. Two rows in the zone, one of them fully moved -> 1/2.
    expect(motionScore(still, row1Moved, { region: topHalf })).toBe(0.5);
  });

  it("widening the region to the full height brings row 2 back at exactly 0.2", () => {
    const whole = { x: 0, y: 0, w: 1, h: 1 };
    expect(motionScore(still, row2Moved, { region: whole })).toBe(0.2);
  });

  it("the same boundary governs the block path, not just the mean path", () => {
    // block_max over the top-half region with the shipped grid: rows [0,2) contain no motion.
    expect(motionScore(still, row2Moved, { region: topHalf, stat: "block_max" })).toBe(0);
    // and row 1, which is inside, saturates a block outright (blocks are 1 px tall there).
    expect(motionScore(still, row1Moved, { region: topHalf, stat: "block_max" })).toBe(1);
  });
});

describe("motionScore — the nearest-rank p95 index formula (row JS-4)", () => {
  // idx = floor((95*n + 99) / 100) - 1, integer-only, which is exactly ceil(0.95*n) - 1. For
  // n = 20 that is 18 — the SECOND largest block, not the largest. An index built as
  // floor(0.95*n) or Math.round(0.95*n) lands on 19 and reports the maximum instead, and an
  // interpolating percentile lands between the two.
  const W = 20;
  const H = 20;
  const still = makeFrame(W, H, 0);
  // grid [5,4] over 20x20 -> 20 blocks of 4 wide x 5 tall = 20 px each.
  // one hot pixel in block (0,0) -> 1/20 = 0.05; two hot pixels in block (1,0) -> 2/20 = 0.1.
  const crafted = hotPixels(W, H, [0, 4, 5]);
  const grid = { blockGrid: [5, 4] };

  it("block_max takes the top block: exactly 0.1", () => {
    expect(motionScore(still, crafted, { ...grid, stat: "block_max" })).toBe(0.1);
  });

  it("p95 over 20 blocks takes rank 18 — exactly 0.05, the runner-up, not the max", () => {
    expect(motionScore(still, crafted, { ...grid, stat: "p95" })).toBe(0.05);
  });

  it("the whole-frame mean of the same pair is exactly 0.0075", () => {
    expect(motionScore(still, crafted)).toBe(0.0075);
  });

  it("with a single block (grid [1,1]) rank 0 is that block, so p95 == block_max == mean", () => {
    const one = { blockGrid: [1, 1] };
    expect(motionScore(still, crafted, { ...one, stat: "p95" })).toBe(0.0075);
    expect(motionScore(still, crafted, { ...one, stat: "block_max" })).toBe(0.0075);
    expect(motionScore(still, crafted)).toBe(0.0075);
  });

  it("at n = 12 the rank is 11 — the max — where a positional percentile would say 10", () => {
    // n = 20 above pins the formula against floor(0.95*n), but a positional variant
    // (round(0.95*(n-1)), which is what an interpolating percentile uses) also lands on 18 there
    // and survives. n = 12 separates them: ceil(0.95*12) - 1 = 11, round(0.95*11) = 10.
    // 12x12 under grid [4,3] is 12 blocks of 3x4 = 12 px. Three neighbouring blocks in the top
    // band carry 1, 2 and 3 hot pixels -> 1/12, 2/12, 3/12 at sorted ranks 9, 10, 11.
    const s = makeFrame(12, 12, 0);
    const c = hotPixels(12, 12, [0, 3, 4, 6, 7, 8]);
    const grid12 = { blockGrid: [4, 3] };
    expect(motionScore(s, c, { ...grid12, stat: "p95" })).toBe(0.25);
    expect(motionScore(s, c, { ...grid12, stat: "block_max" })).toBe(0.25);
    // 2/12 is rank 10 — the answer an interpolating percentile gives. It must NOT appear.
    expect(motionScore(s, c, { ...grid12, stat: "p95" })).not.toBe(0.16666666666666666);
    expect(motionScore(s, c)).toBe(0.041666666666666664);
    expect(motionScore(s, c)).toBe(1 / 24);
  });
});

describe("motionScore — region bounds clamp to the frame (row JS-4)", () => {
  // `regionBounds` clamps with Math.trunc for the origin and pyRound for the far edge, then pins
  // both into [0, w] / [0, h] — the same max/min sandwich as `tier0._region_bounds`. An unclamped
  // port reads outside the buffer (undefined -> NaN) instead of shrinking the zone.
  const W = 8;
  const H = 5;
  const still = makeFrame(W, H, 0);
  const row2 = hotPixels(W, H, Array.from({ length: W }, (_, c) => 2 * W + c));

  it("a region running past the right and bottom edges shrinks instead of reading out of bounds", () => {
    // x0 = trunc(0.5*8) = 4, y0 = trunc(0.5*5) = 2, x1 = min(8, 12) = 8, y1 = min(5, 8) = 5.
    // Row 2 lies on the top edge of that zone: 4 moved pixels out of 12.
    const box = { x: 0.5, y: 0.5, w: 1, h: 1 };
    expect(motionScore(still, row2, { region: box })).toBe(0.3333333333333333);
    expect(motionScore(still, row2, { region: box })).toBe(1 / 3);
    expect(motionScore(still, row2, { region: box, stat: "block_max" })).toBe(1);
  });

  it("a negative origin clamps to 0, giving the whole frame, not a negative index", () => {
    const box = { x: -1, y: -1, w: 2, h: 2 };
    expect(motionScore(still, row2, { region: box })).toBe(0.2);
    expect(motionScore(still, row2, { region: box })).toBe(motionScore(still, row2));
  });

  it("an entirely off-frame or zero-extent region collapses to an empty span and scores 0", () => {
    // NOTE these two boxes collapse because their BOUNDS collapse (2*8 clamps to the right edge;
    // 0*8 truncates and rounds to the same 0) — not because they enclose no area. A zero-extent box
    // whose bounds do NOT collapse is the case below.
    for (const box of [{ x: 2, y: 0, w: 0.1, h: 1 }, { x: 0, y: 0, w: 0, h: 1 }]) {
      expect(motionScore(still, row2, { region: box })).toBe(0);
      expect(motionScore(still, row2, { region: box, stat: "block_max" })).toBe(0);
      expect(motionScore(still, row2, { region: box, stat: "p95" })).toBe(0);
    }
  });

  it("a zero-extent zone whose bounds do NOT collapse is measured, identically on every stat (FIND-PARITY-01)", () => {
    // The bounds are asymmetric on purpose: the ORIGIN truncates and the FAR bound rounds
    // half-to-even. At x=0.1 on a 7px raster that is trunc(0.7)=0 .. round(0.7)=1 — one real pixel
    // column — so a zero-EXTENT box is not a zero-PIXEL box. Python's `_region_bounds` computes
    // exactly that and `_motion_score_region` scored the column 1.0, while the mean path here
    // short-circuited on `!nonEmptyBox(region)` and answered 0: the same authored goal admitted
    // EVERY frame server-side and no frame on device. The block stats never had the short-circuit
    // and always agreed with Python; all three now do.
    const a = makeFrame(7, 3, 0);
    const b = makeFrame(7, 3, 255);
    const zone = { x: 0.1, y: 0.1, w: 0.0, h: 0.5 };
    expect(motionScore(a, b, { region: zone })).toBe(1);
    expect(motionScore(a, b, { region: zone, stat: "block_max" })).toBe(1);
    expect(motionScore(a, b, { region: zone, stat: "p95" })).toBe(1);
    // Half the frame moved, so the column being measured is a real measurement and not a saturated
    // "everything moved": the same zone over a partial change reads the column's own diff.
    const half = makeFrame(7, 3, [0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0]);
    // rows 0..1 of column 0 (y: trunc(0.3)=0 .. round(1.8)=2) — one of the two moved.
    expect(motionScore(a, half, { region: zone })).toBe(0.5);
    // ...and the control from the other side: 0.1*3 = 0.3 truncates AND rounds to 0, so THIS zone
    // really does collapse and 0 is the right answer.
    expect(motionScore(a, b, { region: { x: 0.1, y: 0.1, w: 0.5, h: 0.0 } })).toBe(0);
    // Both zones are refused at the authoring/wire boundary — `motionScore` takes a Box as a VALUE
    // (a derived zone may collapse), so it degrades to the number the other runtime computes rather
    // than throwing. See contract.test.js for the boundary that refuses them.
  });
});

describe("motionScore — degenerate grids and areas (row JS-4)", () => {
  const still = makeFrame(8, 8, 0);
  const moved = makeFrame(8, 8, 255);

  it("a non-positive blockGrid dimension is refused, not answered with a plausible 0", () => {
    // REWRITTEN (was "yields 0, it does not divide by zero or throw" — the defect as the contract).
    // `still` -> `moved` is a total flip; calling that 0 salience is a reading, not a guard. The
    // Python twin raises ValueError for the same input, so the two runtimes agree on what an input
    // even is.
    for (const blockGrid of [[0, 6], [8, 0], [0, 0], [-3, 6], [8, -1]]) {
      expect(() => motionScore(still, moved, { blockGrid, stat: "block_max" })).toThrow(
        /blockGrid dims must be positive/
      );
      expect(() => motionScore(still, moved, { blockGrid, stat: "p95" })).toThrow(
        /blockGrid dims must be positive/
      );
      // ...but `mean` never reads blockGrid on either side, so it stays unread here too.
      expect(motionScore(still, moved, { blockGrid })).toBe(1);
    }
  });

  it("a fractional blockGrid truncates toward zero, matching Python's int()", () => {
    // trunc(0.9) == 0 -> impossible, and refused; trunc(1.9) == 1 -> one block == the frame mean.
    expect(() => motionScore(still, moved, { blockGrid: [0.9, 6], stat: "block_max" })).toThrow(
      /blockGrid dims must be positive/
    );
    expect(motionScore(still, moved, { blockGrid: [1.9, 1.9], stat: "block_max" })).toBe(1);
  });

  it("a grid finer than the active area is CLAMPED to it, so no block is ever empty", () => {
    // REWRITTEN (was "leaves empty blocks scoring 0, not NaN", which pinned 42 phantom 0.0 samples
    // into the percentile population). 3x2 area under the shipped [8,6] grid now lays a 3x2 grid:
    // 6 blocks of 1 px, all real. block_max is unchanged; p95 MOVES, 0 -> 1, because the rank now
    // ranks blocks that exist. Twin: `tier0._partition`.
    const tinyStill = makeFrame(3, 2, 0);
    const tinyHot = hotPixels(3, 2, [0]);
    expect(motionScore(tinyStill, tinyHot, { stat: "block_max" })).toBe(1);
    expect(Number.isNaN(motionScore(tinyStill, tinyHot, { stat: "block_max" }))).toBe(false);
    expect(motionScore(tinyStill, tinyHot, { stat: "p95" })).toBe(1);
    expect(motionScore(tinyStill, tinyHot)).toBe(0.16666666666666666);
    expect(motionScore(tinyStill, tinyHot)).toBe(1 / 6);
  });

  it("the clamp bites per-dimension and stops exactly at the area's own size", () => {
    // 8x4 under [8,6]: the columns fit exactly and only the rows clamp, to 8x4 = 32 blocks of 1 px.
    // A hot pixel is one block, so p95 over 32 (rank 30) reads 0 while block_max reads 1.
    const s = makeFrame(8, 4, 0);
    const c = hotPixels(8, 4, [0]);
    expect(motionScore(s, c, { stat: "block_max" })).toBe(1);
    expect(motionScore(s, c, { stat: "p95" })).toBe(0);
    // 9x7 carries [8,6] untouched: 48 blocks, and the top-left one is 1x1 so the pixel is still 1.
    const s9 = makeFrame(9, 7, 0);
    const c9 = hotPixels(9, 7, [0]);
    expect(motionScore(s9, c9, { stat: "block_max" })).toBe(1);
    expect(motionScore(s9, c9, { stat: "p95" })).toBe(0);
  });

  it("a 0x0 frame scores 0 on every path", () => {
    // Edge-only guard: the Python Frame forbids a non-positive dimension outright
    // (`frame dims must be positive`), so there is no cross-runtime value to match here — the
    // contract is only that the JS side stays finite instead of returning NaN.
    const empty = makeFrame(0, 0, 0);
    expect(motionScore(empty, empty)).toBe(0);
    expect(motionScore(empty, empty, { stat: "block_max" })).toBe(0);
    expect(motionScore(empty, empty, { stat: "p95" })).toBe(0);
    expect(motionScore(empty, empty, { region: { x: 0, y: 0, w: 1, h: 1 } })).toBe(0);
  });

  it("throws when data is shorter than width*height", () => {
    const short = { width: 4, height: 4, data: new Uint8Array(8) };
    expect(() => motionScore(short, makeFrame(4, 4, 0))).toThrow(/shorter than width\*height/);
    expect(() => motionScore(makeFrame(4, 4, 0), short)).toThrow(/shorter than width\*height/);
  });

  it("throws when either argument is not a frame", () => {
    // The two messages are distinguishable: "prevFrame: expected" vs "frame: expected".
    expect(() => motionScore({}, makeFrame(4, 4, 0))).toThrow(/prevFrame: expected/);
    expect(() => motionScore(makeFrame(4, 4, 0), { width: 4, height: 4 })).toThrow(
      /(?:^|\s)frame: expected/
    );
    // A non-finite dimension is not a dimension. (The non-POSITIVE dimension case is deliberately
    // left alone — see the 0x0 test above — but NaN never had a defensible answer.)
    expect(() => motionScore({ width: NaN, height: 4, data: new Uint8Array(16) }, makeFrame(4, 4, 0)))
      .toThrow(/prevFrame: expected/);
  });
});

describe("motionScore — the opts are inputs too, and the return type is a number", () => {
  // The JSDoc says `@returns {number} in [0,1]`. Two knob values used to produce a value outside
  // that type entirely, and neither raised: a NaN region returned NaN, and `blockGrid:[NaN,NaN]`
  // returned a literal `undefined`. Both raise in the Python twin (`int(nan)` is a ValueError), so
  // raising is also what keeps the runtimes agreeing on what counts as an input at all.
  const still = makeFrame(8, 8, 0);
  const moved = makeFrame(8, 8, 255);

  it("a non-finite region raises instead of returning NaN", () => {
    for (const region of [
      { x: NaN, y: 0, w: 1, h: 1 },
      { x: 0, y: 0, w: Infinity, h: 1 },
      { x: 0, y: 0, w: 1 }, // h missing -> undefined
      { x: "0", y: 0, w: 1, h: 1 },
    ]) {
      expect(() => motionScore(still, moved, { region }), JSON.stringify(region)).toThrow(
        /region must be a Box/
      );
      expect(() => motionScore(still, moved, { region, stat: "block_max" })).toThrow(
        /region must be a Box/
      );
    }
    // `null` stays the documented "no zone" value and is not an error.
    expect(motionScore(still, moved, { region: null })).toBe(1);
  });

  it("a non-finite blockGrid raises instead of returning undefined", () => {
    for (const blockGrid of [[NaN, NaN], [8, NaN], [Infinity, 6], [8], 8, null, ["8", "6"]]) {
      expect(() => motionScore(still, moved, { blockGrid, stat: "block_max" })).toThrow(
        /blockGrid must be a \[Gx, Gy\] pair/
      );
      expect(() => motionScore(still, moved, { blockGrid, stat: "p95" })).toThrow(
        /blockGrid must be a \[Gx, Gy\] pair/
      );
    }
  });

  it("a DEGENERATE region whose bounds collapse scores 0 — measured, not short-circuited", () => {
    // The prose here used to say the answer was decided "in one place" by a `nonEmptyBox`
    // short-circuit, and that "rejecting a zero-extent box here would fork the contract". Both
    // claims are retired: the short-circuit is gone (it forked the two runtimes — FIND-PARITY-01),
    // and the zero-area box is refused at the authoring/wire boundary instead. These three still
    // score 0, and now for the reason the docstring always gave — on an 8x8 raster their BOUNDS
    // collapse (0*8 and -1*8 both land on an empty span), so the pixel loops measure nothing.
    for (const region of [
      { x: 0, y: 0, w: 0, h: 1 },
      { x: 0, y: 0, w: 1, h: 0 },
      { x: 0, y: 0, w: -1, h: 1 },
    ]) {
      expect(motionScore(still, moved, { region }), JSON.stringify(region)).toBe(0);
      expect(motionScore(still, moved, { region, stat: "block_max" })).toBe(0);
      expect(motionScore(still, moved, { region, stat: "p95" })).toBe(0);
    }
    // The control: the same frames with a non-empty region are maximally salient.
    expect(motionScore(still, moved, { region: { x: 0, y: 0, w: 1, h: 1 } })).toBe(1);
  });

  it("every returned value across the knob matrix is a finite number in [0,1]", () => {
    // The non-positive grids that used to sit in this matrix ([0,0], [-3,6], [0.9,6]) now THROW on
    // a block stat rather than returning a number — see the degenerate-grid suite above. They stay
    // in the matrix on the `mean` path, which never reads the grid.
    for (const stat of ["mean", "block_max", "p95"]) {
      for (const region of [null, { x: 0, y: 0, w: 1, h: 1 }, { x: 0.5, y: 0.5, w: 1, h: 1 }]) {
        const grids =
          stat === "mean"
            ? [[8, 6], [1, 1], [0, 0], [-3, 6], [0.9, 6], [64, 64]]
            : [[8, 6], [1, 1], [64, 64]];
        for (const blockGrid of grids) {
          const s = motionScore(still, moved, { stat, region, blockGrid });
          const why = `${stat} ${JSON.stringify(region)} ${JSON.stringify(blockGrid)}`;
          expect(typeof s, why).toBe("number");
          expect(Number.isFinite(s), why).toBe(true);
          expect(s >= 0 && s <= 1, why).toBe(true);
        }
      }
    }
  });
});

describe("makeFrame — the fill conversion (row JS-4)", () => {
  it("a numeric fill SATURATES to a byte, matching the Python helper — it does not wrap", () => {
    // The twin is `tier0._clamp_u8`: `0 if v < 0 else 255 if v > 255 else int(v)`. This used to be
    // `fill & 0xff`, which takes the LOW BYTE instead, and the two answers are not merely different
    // — they are inversions: 300 wrapped to 44 where Python saturates to 255, -1 wrapped to 255
    // where Python floors to 0, 256 wrapped to 0 where Python saturates to 255. motionScore never
    // calls this, so nothing scored wrong; but a parity vector authored through both frame
    // constructors was not the same frame on the two sides, which is worse — it makes the parity
    // gate compare the wrong pair of frames and report agreement.
    expect(Array.from(makeFrame(2, 2, 300).data)).toEqual([255, 255, 255, 255]);
    expect(makeFrame(1, 1, 256).data[0]).toBe(255);
    expect(makeFrame(1, 1, -1).data[0]).toBe(0);
    expect(makeFrame(1, 1, 255).data[0]).toBe(255);
    expect(makeFrame(1, 1, 0).data[0]).toBe(0);
    // Python's int() truncates toward zero; so does Math.trunc. Both give 12.
    expect(makeFrame(1, 1, 12.7).data[0]).toBe(12);
    expect(makeFrame(1, 1, 254.999).data[0]).toBe(254);
    // int(nan) RAISES in Python — there is no saturating answer — so a non-number maps to the same
    // "no signal" byte the default fill uses rather than to a wrapped 0-by-accident.
    expect(makeFrame(1, 1, NaN).data[0]).toBe(0);
    expect(makeFrame(1, 1, Infinity).data[0]).toBe(255);
    expect(makeFrame(1, 1, -Infinity).data[0]).toBe(0);
  });

  it("a Uint8Array fill is passed through by reference, uncopied and unvalidated", () => {
    const src = Uint8Array.from([1, 2, 3, 4]);
    const frame = makeFrame(2, 2, src);
    expect(frame.data).toBe(src);
    expect(frame.width).toBe(2);
    expect(frame.height).toBe(2);
    // No length check: the declared size is trusted, and the guard lives in motionScore instead.
    expect(makeFrame(8, 8, src).data.length).toBe(4);
    expect(() => motionScore(makeFrame(8, 8, src), makeFrame(8, 8, 0))).toThrow(
      /shorter than width\*height/
    );
  });

  it("a number[] fill is converted elementwise with the same SATURATION", () => {
    // `Uint8Array.from(fill)` wraps per element, so the array branch forked from Python in exactly
    // the way the scalar branch did — and this is the branch a hand-authored parity vector uses.
    const frame = makeFrame(2, 2, [0, 255, 300, -1]);
    expect(frame.data).toBeInstanceOf(Uint8Array);
    expect(Array.from(frame.data)).toEqual([0, 255, 255, 0]);
    expect(Array.from(makeFrame(1, 4, [256, 12.7, NaN, -0.5]).data)).toEqual([255, 12, 0, 0]);
  });

  it("the default fill is 0 and the frame is exactly width*height long", () => {
    const frame = makeFrame(3, 5);
    expect(frame.data.length).toBe(15);
    expect(Array.from(frame.data).every((v) => v === 0)).toBe(true);
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
    expect(egoResidualScore(prev, cur, { k: 4, stat: "block_max" })).toBeLessThan(0.1);
  });

  it("estimates the shift globally, then applies block_max to the aligned residual", () => {
    const w = 16, h = 16;
    const base = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) base[y * w + x] = (x * 17 + y * 31) % 256;
    const a = new Uint8Array(base);
    const b = new Uint8Array(base);
    for (let y = 0; y < 2; y++)
      for (let x = 0; x < 2; x++) {
        a[y * w + x] = 0;
        b[y * w + x] = 255;
      }
    const prev = { width: w, height: h, data: a };
    const cur = { width: w, height: h, data: b };
    expect(egoResidualScore(prev, cur)).toBe(0.015625);
    expect(egoResidualScore(prev, cur, { stat: "block_max" })).toBe(1);
  });
});

describe("egoResidualScore — the integer overlap sums (row JS-4)", () => {
  // 4x2: column 0 lit in prev, column 1 lit in frame. A one-pixel pan explains it entirely.
  const W = 4;
  const H = 2;
  const prev = hotPixels(W, H, [0, 4]);
  const cur = hotPixels(W, H, [1, 5]);

  it("k = 0 is exactly the uncompensated full-frame mean, to the bit", () => {
    // Both paths reduce to total / (w*h*255) over the identical integer sum, so this is `toBe`.
    expect(egoResidualScore(prev, cur, { k: 0 })).toBe(0.5);
    expect(egoResidualScore(prev, cur, { k: 0 })).toBe(motionScore(prev, cur));
  });

  it("a negative k clamps to 0 and yields the very same number, not an empty search", () => {
    const atZero = egoResidualScore(prev, cur, { k: 0 });
    expect(egoResidualScore(prev, cur, { k: -1 })).toBe(atZero);
    expect(egoResidualScore(prev, cur, { k: -7 })).toBe(atZero);
    expect(atZero).toBe(0.5);
  });

  it("k = 1 finds the (1,0) shift and the residual collapses to exactly 0", () => {
    // Overlap is 3x2 = 6 px, denominator 6*255, numerator 0. An off-by-one in the overlap origin
    // (fx0 / paBase) would leave a non-zero residual here.
    expect(egoResidualScore(prev, cur, { k: 1 })).toBe(0);
    expect(egoResidualScore(prev, cur, { k: 4 })).toBe(0);
  });

  it("a fractional k truncates toward zero, matching Python's int()", () => {
    expect(egoResidualScore(prev, cur, { k: 0.9 })).toBe(0.5);
    expect(egoResidualScore(prev, cur, { k: 1.9 })).toBe(0);
  });

  it("the default k is 4", () => {
    expect(egoResidualScore(prev, cur)).toBe(egoResidualScore(prev, cur, { k: 4 }));
  });

  it("the residual never exceeds the full-frame mean on non-divisible pairs", () => {
    // The (0,0) shift is always in the search, so this is an invariant of the search, not of the
    // data. Non-divisible sizes are the ones where an overlap-arithmetic fork would show.
    for (const [w, h] of [[13, 11], [7, 5], [11, 3], [3, 2], [5, 1]]) {
      const a = noise(w, h, 7, 0);
      const b = noise(w, h, 5, 9);
      const mean = motionScore(a, b);
      for (const k of [0, 1, 3, 4]) {
        expect(egoResidualScore(a, b, { k })).toBeLessThanOrEqual(mean);
      }
      expect(egoResidualScore(a, b, { k: 0 })).toBe(mean);
    }
  });

  it("the 13x11 non-divisible noise pair holds its exact residual at every k", () => {
    // Verified against percept.harness.tier0.ego_residual_score. Each residual is an integer sum
    // over an integer overlap denominator, so the two runtimes must land on the same double —
    // there is no place for a float step or a fused multiply-add to drift.
    const a = noise(13, 11, 7, 0);
    const b = noise(13, 11, 5, 9);
    expect(egoResidualScore(a, b, { k: 0 })).toBe(11007 / (143 * 255));
    expect(egoResidualScore(a, b, { k: 1 })).toBe(0.2904977375565611);
    expect(egoResidualScore(a, b, { k: 3 })).toBe(0.2904977375565611);
    expect(egoResidualScore(a, b, { k: 4 })).toBe(0.26971677559912854);
  });

  it("a shift wider than the frame contributes no overlap and is skipped, not counted as 0", () => {
    // k = 8 on a 4-wide frame: |dx| >= 4 has ow <= 0 and must be skipped entirely. Counting it as
    // a zero-pixel, zero-diff shift would make every residual 0 (or NaN).
    expect(egoResidualScore(prev, cur, { k: 8 })).toBe(0);
    const still = makeFrame(4, 2, 0);
    const allMoved = makeFrame(4, 2, 255);
    // Nothing can explain a global flip, so the residual stays at the ceiling for every k.
    expect(egoResidualScore(still, allMoved, { k: 8 })).toBe(1);
    expect(egoResidualScore(still, allMoved, { k: 0 })).toBe(1);
  });

  it("throws on a frame-size mismatch, naming its own function", () => {
    expect(() => egoResidualScore(makeFrame(4, 4, 0), makeFrame(8, 8, 0))).toThrow(
      /egoResidualScore: frame size mismatch 4x4 vs 8x8/
    );
  });

  it("throws when data is shorter than width*height", () => {
    const short = { width: 4, height: 4, data: new Uint8Array(8) };
    expect(() => egoResidualScore(short, makeFrame(4, 4, 0))).toThrow(
      /egoResidualScore: data shorter than width\*height/
    );
    expect(() => egoResidualScore(makeFrame(4, 4, 0), short)).toThrow(
      /egoResidualScore: data shorter than width\*height/
    );
  });

  it("throws when either argument is not a frame", () => {
    expect(() => egoResidualScore({}, makeFrame(4, 4, 0))).toThrow(/prevFrame: expected/);
    expect(() => egoResidualScore(makeFrame(4, 4, 0), { width: 4, height: 4 })).toThrow(
      /(?:^|\s)frame: expected/
    );
  });
});

describe("egoResidualScore — the watch-out zone masks what is MEASURED, not what is ESTIMATED", () => {
  // The residual had no region parameter at all, so a goal carrying BOTH ego compensation and a
  // watch-out zone escalated on motion entirely outside the zone while still reporting the zone.
  // Twin: `percept.harness.tier0.ego_residual_score(..., region=...)`.
  const W = 16;
  const H = 16;

  // A deterministic high-entropy luma field: only the RIGHT shift explains a pan of it. `Math.imul`
  // and the high bits, because a plain `s * 1103515245` overflows 2**53 and the low bits of an LCG
  // are near-periodic — either one yields a field flat enough for a WRONG shift to win the search.
  function texture(seed) {
    const out = new Uint8Array(W * H);
    let s = seed >>> 0;
    for (let i = 0; i < W * H; i++) {
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      out[i] = (s >>> 16) & 0xff;
    }
    return out;
  }

  // A textured background with a 6x6 patch lit in the BOTTOM-RIGHT quadrant only.
  function outOfZonePair() {
    const base = texture(20260731);
    const a = new Uint8Array(base);
    const b = new Uint8Array(base);
    for (let y = 10; y < 16; y++)
      for (let x = 10; x < 16; x++) {
        a[y * W + x] = 0;
        b[y * W + x] = 255;
      }
    return [{ width: W, height: H, data: a }, { width: W, height: H, data: b }];
  }

  it("a zone with none of the movement in it scores 0 where the whole frame does not", () => {
    const [prev, cur] = outOfZonePair();
    const topLeft = { x: 0, y: 0, w: 0.5, h: 0.5 };
    expect(motionScore(prev, cur, { region: topLeft })).toBe(0);
    expect(egoResidualScore(prev, cur)).toBeGreaterThan(0);
    expect(egoResidualScore(prev, cur, { region: topLeft })).toBe(0);
  });

  it("a full-frame zone is exactly the unmasked residual, to the bit", () => {
    const [prev, cur] = outOfZonePair();
    expect(egoResidualScore(prev, cur, { region: { x: 0, y: 0, w: 1, h: 1 } })).toBe(
      egoResidualScore(prev, cur)
    );
    expect(egoResidualScore(prev, cur, { region: null })).toBe(egoResidualScore(prev, cur));
  });

  it("the shift is still estimated globally, so every zone agrees a pan is a pan", () => {
    // A pan is a whole-frame property. If each zone picked its own shift, a flat or tiny zone would
    // report a private opinion of how the camera moved; a global estimate makes them agree.
    const base = texture(987654321);
    const panned = new Uint8Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) panned[y * W + x] = base[y * W + Math.min(x + 4, W - 1)];
    const prev = { width: W, height: H, data: base };
    const cur = { width: W, height: H, data: panned };
    for (const region of [
      null,
      { x: 0, y: 0, w: 1, h: 1 },
      { x: 0, y: 0, w: 0.25, h: 0.25 },
      { x: 0.5, y: 0.5, w: 0.5, h: 0.5 },
    ]) {
      expect(egoResidualScore(prev, cur, { k: 4, region }), JSON.stringify(region)).toBe(0);
    }
  });

  it("a zone the winning shift leaves no overlap in scores 0, not NaN", () => {
    const prev = makeFrame(8, 8, 0);
    const cur = makeFrame(8, 8, 255);
    // x0 = trunc(0) = 0, x1 = pyRound(0.05*8) = 0 -> an empty span, the same permanent never-fire
    // `motionScore` reports for one.
    const sliver = { x: 0, y: 0, w: 0.05, h: 1 };
    expect(egoResidualScore(prev, cur, { k: 2, region: sliver })).toBe(0);
    expect(motionScore(prev, cur, { region: sliver })).toBe(0);
    const off = { x: 2, y: 0, w: 0.5, h: 1 };
    expect(egoResidualScore(prev, cur, { k: 2, region: off })).toBe(0);
  });

  it("a non-finite region raises here too, naming its own function", () => {
    const prev = makeFrame(8, 8, 0);
    const cur = makeFrame(8, 8, 255);
    expect(() => egoResidualScore(prev, cur, { region: { x: NaN, y: 0, w: 1, h: 1 } })).toThrow(
      /egoResidualScore: region must be a Box/
    );
  });
});
