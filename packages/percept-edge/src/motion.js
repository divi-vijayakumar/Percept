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
 *   statistic over a coarse `blockGrid` (`[Gx,Gy]`) partition of the active area. `stat:"mean"`
 *   with no `region` is the byte-for-byte legacy path.
 * @returns {number} motion salience in [0, 1]
 */
export function motionScore(
  prevFrame,
  frame,
  { region = null, stat = "mean", blockGrid = [8, 6] } = {}
) {
  // No history yet (first frame): nothing to diff against -> no motion.
  if (prevFrame == null) return 0;

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
  if (n === 0) return 0;

  if (stat !== "mean") return motionScoreBlocks(a, b, w, h, region, blockGrid, stat);
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

// Localized salience over a Gx x Gy block partition of the active area (full frame or `region`).
// Mirrors `percept.harness.tier0._motion_score_blocks`: floor-of-product boundaries, integer
// per-block denominators, `block_max` = top block, `p95` = nearest-rank 95th-percentile block.
function motionScoreBlocks(a, b, w, h, region, blockGrid, stat) {
  let x0 = 0;
  let y0 = 0;
  let x1 = w;
  let y1 = h;
  if (region != null) [x0, y0, x1, y1] = regionBounds(region, w, h);
  const aw = x1 - x0;
  const ah = y1 - y0;
  if (aw <= 0 || ah <= 0) return 0;
  const gx = Math.trunc(blockGrid[0]);
  const gy = Math.trunc(blockGrid[1]);
  if (gx <= 0 || gy <= 0) return 0;
  const scores = [];
  for (let j = 0; j < gy; j++) {
    const by0 = y0 + Math.floor((j * ah) / gy);
    const by1 = y0 + Math.floor(((j + 1) * ah) / gy);
    for (let i = 0; i < gx; i++) {
      const bx0 = x0 + Math.floor((i * aw) / gx);
      const bx1 = x0 + Math.floor(((i + 1) * aw) / gx);
      const blockPx = (bx1 - bx0) * (by1 - by0);
      if (blockPx === 0) {
        scores.push(0);
        continue;
      }
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
 * @param {{width:number,height:number,data:Uint8Array|number[]}|null|undefined} prevFrame
 * @param {{width:number,height:number,data:Uint8Array|number[]}} frame
 * @param {{k?:number}} [opts]
 * @returns {number} ego-compensated salience in [0, 1]
 */
export function egoResidualScore(prevFrame, frame, { k = 4 } = {}) {
  if (prevFrame == null) return 0;
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
      if (best === null || score < best) best = score;
    }
  }
  return best === null ? 0 : best;
}

/**
 * Convenience: build a grayscale Frame from a width/height and a data source.
 * Accepts a Uint8Array, a plain number[], or a fill value (constant frame).
 * @returns {{width:number,height:number,data:Uint8Array}}
 */
export function makeFrame(width, height, fill = 0) {
  const n = width * height;
  let data;
  if (fill instanceof Uint8Array) {
    data = fill;
  } else if (Array.isArray(fill)) {
    data = Uint8Array.from(fill);
  } else {
    data = new Uint8Array(n).fill(fill & 0xff);
  }
  return { width, height, data };
}

function assertFrame(f, name) {
  if (
    f == null ||
    typeof f.width !== "number" ||
    typeof f.height !== "number" ||
    f.data == null ||
    typeof f.data.length !== "number"
  ) {
    throw new Error(
      `${name}: expected { width, height, data: Uint8Array } grayscale frame`
    );
  }
}
