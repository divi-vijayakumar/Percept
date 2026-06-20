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
 * Mean-absolute-difference motion score between two grayscale frames,
 * normalized to [0, 1].
 *
 * Identical frames (a still scene) score ~0; a fully inverted frame scores ~1.
 * Deterministic: same inputs always yield the same number.
 *
 * @param {{width:number,height:number,data:Uint8Array|number[]}|null|undefined} prevFrame
 * @param {{width:number,height:number,data:Uint8Array|number[]}} frame
 * @returns {number} motion salience in [0, 1]
 */
export function motionScore(prevFrame, frame) {
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
  const n = frame.width * frame.height;
  if (a.length < n || b.length < n) {
    throw new Error("motionScore: data shorter than width*height");
  }
  if (n === 0) return 0;

  let acc = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    acc += d < 0 ? -d : d;
  }
  // Mean absolute diff is in [0, 255]; normalize to [0, 1].
  return acc / (n * 255);
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
