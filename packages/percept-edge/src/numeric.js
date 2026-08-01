// src/numeric.js
//
// The numeric predicates the edge validates with — one definition each, shared by every seam that
// takes a number from outside the package.
//
// Before this module the same `typeof v === "number" && Number.isFinite(v)` test was written out
// four separate times (contract.js, privacy.js, edge.js, skills.js) and the same [0,1] clamp twice
// (edge.js, skills.js). Duplicated predicates drift: `makeBox` simply never grew one, so it accepted
// a NaN box its own `parseBox` twin rejects, and the privacy constructors validated some knobs and
// not others. Keeping them here means a seam either uses the predicate or visibly does not.
//
// Every function is total and pure: no throws, no allocation, no clock.

/** A real number — excludes NaN, ±Infinity, and every non-number (strings, null, bigint). */
export function finiteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/** A finite number already inside [0,1] — a score/threshold that needs no clamping. */
export function unitInterval(v) {
  return finiteNumber(v) && v >= 0 && v <= 1;
}

/** A strictly positive finite number — a dimension, a budget, a window: something zero cannot be. */
export function positiveDim(v) {
  return finiteNumber(v) && v > 0;
}

/**
 * Coerce anything to a salience in [0,1]. Non-numbers and non-finite numbers collapse to 0 (an
 * unusable signal contributes nothing rather than poisoning a comparison — NaN >= threshold is
 * false, but NaN propagated into a max/sum silently disables every other channel).
 */
export function clamp01(v) {
  if (!finiteNumber(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

/**
 * A Box `{x,y,w,h}` of finite numbers that encloses actual area (`w > 0 && h > 0`).
 *
 * This is the AREA half of the Box contract `makeBox`/`parseBox` enforce at the authoring/wire
 * boundary (they add the [0,1] range check). A zero-area zone selects no region of any image, so a
 * goal carrying one is a permanent never-fire at every threshold — an authoring mistake that is
 * indistinguishable from a working watch once it is on the wire, which is why it is refused there
 * rather than interpreted.
 *
 * It is deliberately NOT applied to a Box used as a VALUE: `motionScore` takes whatever box a
 * caller derived, including one that collapsed, and measures the pixel span the bounds give it (see
 * `motion.regionBounds` — a zero-EXTENT box is not always a zero-PIXEL box). Enforcing area there
 * instead of here is what forked the two runtimes on the mean path (FIND-PARITY-01).
 */
export function nonEmptyBox(box) {
  return (
    box != null &&
    typeof box === "object" &&
    finiteNumber(box.x) &&
    finiteNumber(box.y) &&
    finiteNumber(box.w) &&
    finiteNumber(box.h) &&
    box.w > 0 &&
    box.h > 0
  );
}
