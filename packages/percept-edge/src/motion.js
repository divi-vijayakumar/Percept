// src/motion.js
//
// Coarse, on-device motion gate. Pure arithmetic over raw pixels (no model,
// no native deps, fully deterministic) — the "is anything moving, worth a
// cloud call?" reflex. This is the wired proof-point detector for the edge.
//
// Frame representation (deliberately minimal, you can adapt a canvas
// ImageData / video frame into this):
//
//   Frame = { width: number, height: number, data: Uint8Array }
//
// `data` is single-channel grayscale, length === width * height, one byte
// (0..255) per pixel in row-major order.

import { finiteNumber } from "./numeric.js";

/**
 * Absolute-difference motion salience between two grayscale frames, normalized to [0, 1].
 *
 * Identical frames (a still scene) score ~0; a fully inverted frame scores ~1.
 * Deterministic: same inputs always yield the same number.
 *
 * The Python twin is `percept.harness.tier0.motion_score`; every branch here is bit-identical to it
 * (a release-gated parity invariant — see `parity/compare.py`).
 *
 * @param {{width:number,height:number,data:Uint8Array|number[]}|null|undefined} prevFrame
 * @param {{width:number,height:number,data:Uint8Array|number[]}} frame
 * @param {{region?:{x:number,y:number,w:number,h:number}|null, stat?:"mean"|"block_max"|"p95", blockGrid?:[number,number]}} [opts]
 *   `region` masks the diff to a normalized watch-out zone (Box); `stat` selects a localizing
 *   statistic over a coarse `blockGrid` (`[Gx,Gy]`) partition of the active area, clamped down to
 *   the active area when finer than it (see `partition`). `stat:"mean"` with no `region` is the
 *   byte-for-byte legacy path and never reads `blockGrid` at all.
 * @returns {number} motion salience in [0, 1] — never NaN, never undefined
 * @throws {Error} on a malformed frame, a size mismatch, a non-finite `region` / `blockGrid`, or a
 *   non-positive `blockGrid` dimension on a block stat
 */
export function motionScore(
  prevFrame,
  frame,
  { region = null, stat = "mean", blockGrid = [8, 6] } = {}
) {
  // No history yet (first frame): nothing to diff against -> no motion.
  if (prevFrame == null) return 0;

  // The opts are as much an input as the pixels, and the documented return type is a number in
  // [0,1]. Unvalidated, a NaN in either knob produced a value OUTSIDE that type: a NaN `region`
  // made every bound NaN and returned NaN, and `blockGrid:[NaN,NaN]` walked past `gx <= 0` (NaN
  // fails every comparison), built an EMPTY block list, and returned a literal `undefined` from
  // `scores[0]`. Both raise in the Python twin (`int(nan)` is a ValueError), so raising is also
  // what keeps the two runtimes agreeing about which inputs are inputs at all.
  assertBoxOpt(region, "region");
  assertGrid(blockGrid);

  assertFrame(prevFrame, "prevFrame");
  assertFrame(frame, "frame");

  if (
    prevFrame.width !== frame.width ||
    prevFrame.height !== frame.height
  ) {
    throw new Error(
      `motionScore: frame size mismatch ${prevFrame.width}x${prevFrame.height} vs ${frame.width}x${frame.height}`
    );
  }

  const a = prevFrame.data;
  const b = frame.data;
  const w = frame.width;
  const h = frame.height;
  const n = w * h;
  if (a.length < n || b.length < n) {
    throw new Error("motionScore: data shorter than width*height");
  }
  // The block path FIRST, so an impossible `blockGrid` is refused whatever the region is.
  if (stat !== "mean") return motionScoreBlocks(a, b, w, h, region, blockGrid, stat);
  if (n === 0) return 0;
  // NO short-circuit on a zero-EXTENT zone. There used to be one — `if (!nonEmptyBox(region))
  // return 0`, commented as reaching "the same 0 the pixel loops below would reach via an empty
  // span". It does not: `regionBounds` is deliberately ASYMMETRIC (the origin truncates, the far
  // bound rounds half-to-even), so a zero-extent Box is not a zero-PIXEL box whenever that shared
  // product has a fractional part >= 0.5 — `Box(x=0.1, w=0)` on a 7px raster is trunc(0.7)=0 ..
  // round(0.7)=1, one real column. Python's `_region_bounds` computes exactly that and
  // `_motion_score_region` scored it 1.0 while this path answered 0, so the same authored zone
  // admitted every frame in the harness and no frame on device (FIND-PARITY-01). The block stats
  // never had the short-circuit and always agreed with Python; the mean path now does too. Zero
  // pixels is still 0 — decided by the bounds, in `motionScoreRegion`, on both sides of the wire.
  //
  // A zone that ENCLOSES no area is refused upstream instead, at the authoring/wire boundary
  // (`contract.makeBox`/`parseBox`), which is where an authored never-fire belongs. This function
  // takes a Box as a VALUE, including one a caller derived and watched collapse, so it degrades
  // rather than throwing — and degrades to the number the other runtime computes.
  if (region != null) return motionScoreRegion(a, b, w, h, region);

  let acc = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    acc += d < 0 ? -d : d;
  }
  // Mean absolute diff is in [0, 255]; normalize to [0, 1].
  return acc / (n * 255);
}

// Python's round() is round-half-to-even (banker's rounding); JS Math.round is round-half-up.
// They differ only at an exact .5 — replicate Python so region bounds match bit-for-bit.
function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

function clampInt(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// Normalized Box -> half-open integer pixel bounds [x0,y0,x1,y1], clamped to the frame.
// Mirrors `percept.harness.tier0._region_bounds`: int()==Math.trunc, round()==pyRound.
function regionBounds(region, w, h) {
  const x0 = clampInt(Math.trunc(region.x * w), 0, w);
  const y0 = clampInt(Math.trunc(region.y * h), 0, h);
  const x1 = clampInt(pyRound((region.x + region.w) * w), x0, w);
  const y1 = clampInt(pyRound((region.y + region.h) * h), y0, h);
  return [x0, y0, x1, y1];
}

// Mean abs-diff within `region` only (the watch-out zone). Empty/degenerate region -> 0.
function motionScoreRegion(a, b, w, h, region) {
  const [x0, y0, x1, y1] = regionBounds(region, w, h);
  const rw = x1 - x0;
  const rh = y1 - y0;
  const n = rw * rh;
  if (n === 0) return 0;
  let total = 0;
  for (let row = y0; row < y1; row++) {
    const base = row * w;
    for (let col = x0; col < x1; col++) {
      const d = a[base + col] - b[base + col];
      total += d < 0 ? -d : d;
    }
  }
  return total / (n * 255);
}

// The block grid actually laid on the active area, and whether the clamp bit. ONE guard point, TWO
// deliberately distinct rules — mirrors `percept.harness.tier0._partition`:
//
//   * a non-positive Gx/Gy is an IMPOSSIBLE partition and THROWS. It used to return 0, i.e. a
//     reading ("the scene is still") for a frame that may have inverted end to end. The Python twin
//     raises `ValueError`, and the sibling case here — an unknown `stat` — already throws.
//   * a grid FINER than the area is CLAMPED, not padded. `Gx x Gy` blocks over an `aw x ah` area
//     with `aw < Gx` used to emit `Gx - aw` columns of 0-px blocks, each pushing a literal 0 into
//     the percentile population. A 6x4-px `region_hint` is a valid authored input at the shipped
//     48x36 gate raster, so it must not throw — but its p95 must rank real blocks, not padding.
//
// Returns [effectiveGx, effectiveGy, degenerate].
function partition(x0, y0, x1, y1, gx, gy) {
  // Negated comparisons, so a non-finite grid that slipped past the caller-facing guard lands here
  // rather than falling through to an empty `scores` list (whose `scores[0]` is `undefined`).
  if (!(gx > 0) || !(gy > 0)) {
    throw new Error(`motionScore: blockGrid dims must be positive, got ${gx}x${gy}`);
  }
  const aw = x1 - x0;
  const ah = y1 - y0;
  const effectiveGx = gx < aw ? gx : aw;
  const effectiveGy = gy < ah ? gy : ah;
  return [effectiveGx, effectiveGy, effectiveGx !== gx || effectiveGy !== gy];
}

// Localized salience over a Gx x Gy block partition of the active area (full frame or `region`).
// Mirrors `percept.harness.tier0._motion_score_blocks`: floor-of-product boundaries, integer
// per-block denominators, `block_max` = top block, `p95` = nearest-rank 95th-percentile block.
// The grid comes from `partition`, so no block is ever empty: with Gx <= aw every column band spans
// at least `aw // Gx >= 1` px, and the same for rows.
function motionScoreBlocks(a, b, w, h, region, blockGrid, stat) {
  let x0 = 0;
  let y0 = 0;
  let x1 = w;
  let y1 = h;
  if (region != null) [x0, y0, x1, y1] = regionBounds(region, w, h);
  const aw = x1 - x0;
  const ah = y1 - y0;
  const [gx, gy] = partition(x0, y0, x1, y1, Math.trunc(blockGrid[0]), Math.trunc(blockGrid[1]));
  if (gx <= 0 || gy <= 0) return 0; // an empty / off-frame active area — nothing in-zone to diff.
  const scores = [];
  for (let j = 0; j < gy; j++) {
    const by0 = y0 + Math.floor((j * ah) / gy);
    const by1 = y0 + Math.floor(((j + 1) * ah) / gy);
    for (let i = 0; i < gx; i++) {
      const bx0 = x0 + Math.floor((i * aw) / gx);
      const bx1 = x0 + Math.floor(((i + 1) * aw) / gx);
      const blockPx = (bx1 - bx0) * (by1 - by0);
      let total = 0;
      for (let row = by0; row < by1; row++) {
        const base = row * w;
        for (let col = bx0; col < bx1; col++) {
          const d = a[base + col] - b[base + col];
          total += d < 0 ? -d : d;
        }
      }
      scores.push(total / (blockPx * 255));
    }
  }
  if (stat === "block_max") {
    if (scores.length === 0) return 0; // mirrors Python's `if not scores: return 0.0`
    let m = scores[0];
    for (let k = 1; k < scores.length; k++) if (scores[k] > m) m = scores[k];
    return m;
  }
  if (stat === "p95") return nearestRankP95(scores);
  throw new Error(`motionScore: unknown stat ${stat}; expected 'mean', 'block_max', or 'p95'`);
}

// 95th-percentile by nearest-rank, integer rank `ceil(0.95*n)-1 == (95*n+99)//100 - 1` so it is
// identical in Python and JS (no float percentile / interpolation, which would drift).
function nearestRankP95(scores) {
  const n = scores.length;
  if (n === 0) return 0;
  const s = [...scores].sort((p, q) => p - q);
  let idx = Math.floor((95 * n + 99) / 100) - 1;
  if (idx < 0) idx = 0;
  else if (idx >= n) idx = n - 1;
  return s[idx];
}

/**
 * Ego-motion-compensated salience: the minimum mean-abs-diff over an integer-pixel global shift.
 *
 * Removes whole-frame translation (camera pan / head turn) by searching integer shifts `(dx,dy)` in
 * `[-k,k]` and taking the *residual* — the smallest MAD between `prev` shifted by `(dx,dy)` and
 * `frame` over their overlap. A pure pan is explained by some shift (residual ~0, no false escalate);
 * a subject entering the frame cannot be shifted away (residual stays high). The `(0,0)` shift equals
 * the uncompensated mean, so the residual is always <= the full-frame `motionScore`. Mirrors the
 * Python `percept.harness.tier0.ego_residual_score` bit-for-bit.
 *
 * `region` (the watch-out zone) restricts WHAT IS MEASURED, never WHAT IS ESTIMATED. The shift
 * search still runs over the whole frame, because a pan is a whole-frame property: a zone can be too
 * small, too flat, or too busy to identify a translation, and letting it pick the shift would mean
 * each zone reporting its own private opinion of how the camera moved. Once the global shift is
 * chosen, the residual sum AND its denominator cover only `region ∩ overlap`, so movement outside
 * the zone can no longer escalate the goal. `region: null` (the default) is the whole-frame path,
 * byte-for-byte unchanged.
 *
 * @param {{width:number,height:number,data:Uint8Array|number[]}|null|undefined} prevFrame
 * @param {{width:number,height:number,data:Uint8Array|number[]}} frame
 * @param {{k?:number, region?:{x:number,y:number,w:number,h:number}|null}} [opts]
 * @returns {number} ego-compensated salience in [0, 1]
 */
export function egoResidualScore(prevFrame, frame, { k = 4, region = null } = {}) {
  if (prevFrame == null) return 0;
  assertBoxOpt(region, "region", "egoResidualScore");
  assertFrame(prevFrame, "prevFrame");
  assertFrame(frame, "frame");
  if (prevFrame.width !== frame.width || prevFrame.height !== frame.height) {
    throw new Error(
      `egoResidualScore: frame size mismatch ${prevFrame.width}x${prevFrame.height} vs ${frame.width}x${frame.height}`
    );
  }
  const a = prevFrame.data;
  const b = frame.data;
  const w = frame.width;
  const h = frame.height;
  if (a.length < w * h || b.length < w * h) {
    throw new Error("egoResidualScore: data shorter than width*height");
  }
  let kk = Math.trunc(k);
  if (kk < 0) kk = 0;
  let best = null;
  let bestDx = 0;
  let bestDy = 0;
  for (let dy = -kk; dy <= kk; dy++) {
    const oh = h - (dy >= 0 ? dy : -dy); // overlap height = h - |dy|
    if (oh <= 0) continue;
    const fy0 = dy > 0 ? dy : 0;
    for (let dx = -kk; dx <= kk; dx++) {
      const ow = w - (dx >= 0 ? dx : -dx); // overlap width = w - |dx|
      if (ow <= 0) continue;
      const fx0 = dx > 0 ? dx : 0;
      const npx = ow * oh;
      let total = 0;
      for (let fy = fy0; fy < fy0 + oh; fy++) {
        const fbBase = fy * w;
        const paBase = (fy - dy) * w;
        for (let fx = fx0; fx < fx0 + ow; fx++) {
          const d = a[paBase + (fx - dx)] - b[fbBase + fx];
          total += d < 0 ? -d : d;
        }
      }
      const score = total / (npx * 255);
      if (best === null || score < best) {
        best = score;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }
  if (best === null) return 0;
  if (region == null) return best;
  return egoRegionResidual(a, b, w, h, region, bestDx, bestDy);
}

// The residual of the ALREADY-CHOSEN shift `(dx, dy)`, measured over `region ∩ overlap`. Two
// rectangles in `frame` coordinates: the overlap the shift leaves behind, and the watch-out zone.
// Their intersection is the only place the zone's promise ("movement elsewhere is ignored") and the
// shift's validity ("both frames have a pixel here") both hold. Integer sum over an integer
// denominator, exactly like the unmasked path. Twin: `tier0._ego_region_residual`.
function egoRegionResidual(a, b, w, h, region, dx, dy) {
  const [rx0, ry0, rx1, ry1] = regionBounds(region, w, h);
  const ox0 = dx > 0 ? dx : 0;
  const oy0 = dy > 0 ? dy : 0;
  const ox1 = ox0 + w - (dx >= 0 ? dx : -dx);
  const oy1 = oy0 + h - (dy >= 0 ? dy : -dy);
  const mx0 = rx0 > ox0 ? rx0 : ox0;
  const my0 = ry0 > oy0 ? ry0 : oy0;
  const mx1 = rx1 < ox1 ? rx1 : ox1;
  const my1 = ry1 < oy1 ? ry1 : oy1;
  // Per DIMENSION, not on the product: a zone sitting outside the overlap on BOTH axes makes both
  // spans negative and their product positive again.
  const mw = mx1 - mx0;
  const mh = my1 - my0;
  if (mw <= 0 || mh <= 0) return 0;
  const npx = mw * mh;
  let total = 0;
  for (let fy = my0; fy < my1; fy++) {
    const fbBase = fy * w;
    const paBase = (fy - dy) * w;
    for (let fx = mx0; fx < mx1; fx++) {
      const d = a[paBase + (fx - dx)] - b[fbBase + fx];
      total += d < 0 ? -d : d;
    }
  }
  return total / (npx * 255);
}

/**
 * Luma byte conversion — the JS twin of `percept.harness.tier0._clamp_u8`
 * (`0 if v < 0 else 255 if v > 255 else int(v)`).
 *
 * SATURATES; it does not wrap. `Uint8Array`'s own implicit conversion (and the `& 0xff` this file
 * used to lean on) takes the low byte instead, so a 300-luma fill became 44, -1 became 255 and 256
 * became 0 — the exact inversions of the values the Python helper produces. Since a parity vector is
 * routinely authored through both frame constructors, that fork meant the two runtimes could be
 * handed *different frames* and then compared for bit-identical scores.
 *
 * Python's `int()` truncates toward zero, which is what `Math.trunc` does (12.7 -> 12 on both
 * sides). The infinities saturate exactly as Python's comparison chain does (+inf > 255 -> 255,
 * -inf < 0 -> 0). `int(nan)` RAISES in Python — there is no saturating answer — so NaN and every
 * non-number map to 0, the same "no signal" byte the default fill uses.
 */
function clampU8(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return 0;
  if (v < 0) return 0;
  if (v > 255) return 255;
  return Math.trunc(v);
}

/**
 * Convenience: build a grayscale Frame from a width/height and a data source.
 * Accepts a Uint8Array, a plain number[], or a fill value (constant frame). Numeric luma is
 * saturated to 0..255 exactly as `tier0.gray_frame` does — see `clampU8`.
 * @returns {{width:number,height:number,data:Uint8Array}}
 */
export function makeFrame(width, height, fill = 0) {
  const n = width * height;
  let data;
  if (fill instanceof Uint8Array) {
    data = fill;
  } else if (Array.isArray(fill)) {
    // Elementwise, NOT `Uint8Array.from(fill)`: the implicit conversion there wraps too, so the
    // array branch forked from Python in exactly the same way the scalar branch did.
    data = new Uint8Array(fill.length);
    for (let i = 0; i < fill.length; i++) data[i] = clampU8(fill[i]);
  } else {
    data = new Uint8Array(n).fill(clampU8(fill));
  }
  return { width, height, data };
}

// A Box-shaped OPTION (not a wire payload — contract.js owns that seam). `null` is the documented
// "no zone" value; anything present must carry four finite numbers or the bounds arithmetic silently
// produces NaN indices.
function assertBoxOpt(box, name, fn = "motionScore") {
  if (box == null) return;
  if (typeof box !== "object" || ["x", "y", "w", "h"].some((k) => !finiteNumber(box[k]))) {
    throw new Error(`${fn}: ${name} must be a Box { x, y, w, h } of finite numbers`);
  }
}

// `[Gx, Gy]`. NON-FINITE is not a grid at all — `int(nan)` raises in the Python twin, so it raises
// here. Non-positive is refused too, but by `partition` rather than here: the Python twin only ever
// reads `block_grid` on a block stat, so a `[0, 6]` passed alongside `stat: "mean"` is unread on
// both sides and must stay unread on both sides.
function assertGrid(grid) {
  if (!Array.isArray(grid) || grid.length < 2 || !finiteNumber(grid[0]) || !finiteNumber(grid[1])) {
    throw new Error("motionScore: blockGrid must be a [Gx, Gy] pair of finite numbers");
  }
}

function assertFrame(f, name) {
  if (
    f == null ||
    !finiteNumber(f.width) ||
    !finiteNumber(f.height) ||
    f.data == null ||
    typeof f.data.length !== "number"
  ) {
    throw new Error(
      `${name}: expected { width, height, data: Uint8Array } grayscale frame`
    );
  }
}
