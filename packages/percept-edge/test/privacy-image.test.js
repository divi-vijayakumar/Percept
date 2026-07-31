// test/privacy-image.test.js
//
// The dedicated suite for the IMAGE tier (src/privacy-image.js): PrivacyImageAdapter, the
// failClosedImage() constant-shape, runImageTier's timeout/error lifecycle, and the private
// normalizeImageResult that runImageTier calls on every adapter result.
//
// Why this file exists separately from privacy.test.js. The aggregate suite drives the image tier
// through PrivacyScreen.screenAsync and asserts the *action* ("drop", "confirm", "allow"), which is
// the orchestrator's mapping of the tier's answer. That mapping is lossy: a fail-closed verdict and
// a genuine high-confidence credential detection both come out as "drop", so an entire class of
// normalization regressions is invisible from there. This file asserts the tier's own return value
// whole, before any mapping.
//
// The contract a second runtime has to reproduce, frozen below:
//   * Every rejection path returns EXACTLY {sensitive:true, categories:[], score:1, spans:[], failed:true}
//     -- five keys, no more. `failed` is the discriminator the orchestrator keys off; a port that
//     drops it turns a hung model into an ordinary low-score verdict, i.e. into "allow".
//   * A malformed *box* is silently dropped; a malformed *category or score* fails the whole frame
//     closed. Those are deliberately different severities and a port must not unify them.
//   * `score` is the MAX of the supplied scores, and is forced to 0 when there are no categories --
//     but the max is taken over the raw list, before dedupe, so duplicate labels cannot inflate it.
//   * The happy-path verdict has FOUR keys. It carries no `failed` key at all, not `failed: false`.
//
// There is no Python twin of this module in the repository, so "the two runtimes agree" cannot be
// asserted against anything. Freezing the JS side whole is what makes a future divergence show up
// as a failing vector rather than as a frame leaving the device.

import { describe, it, expect } from "vitest";
import {
  PrivacyImageAdapter,
  failClosedImage,
  runImageTier,
} from "../src/privacy-image.js";
import { PRIVACY_CATEGORIES } from "../src/privacy-constants.js";
import { PrivacyScreen } from "../src/privacy.js";

// --- vectors ---------------------------------------------------------------

/** The one shape every rejection path must produce. Asserted with toStrictEqual, so an extra or
 *  missing key is a failure -- `toEqual` would let `failed: undefined` through. */
const FAIL_CLOSED = { sensitive: true, categories: [], score: 1, spans: [], failed: true };

const box = (x1, y1, x2, y2) => ({ start: { x: x1, y: y1 }, end: { x: x2, y: y2 } });

/** An adapter that returns exactly the payload handed to it -- including a malformed one. */
const returning = (payload) => ({
  async infer() {
    return payload;
  },
});

/** A well-formed adapter over (categories, scores, boxes?). */
const okAdapter = (categories, scores, boxes) =>
  returning(boxes === undefined ? { categories, scores } : { categories, scores, boxes });

const FRAME = { width: 48, height: 36, data: new Uint8Array(48 * 36) };

// --- the documented adapter slot -------------------------------------------

describe("PrivacyImageAdapter — the documented, deliberately unimplemented slot", () => {
  it("load() rejects and names the two ways to enable the tier", async () => {
    const slot = new PrivacyImageAdapter();
    await expect(slot.load()).rejects.toThrow(
      /documented adapter slot/
    );
    // The message has to stay actionable: it is the only thing an integrator sees.
    await expect(slot.load()).rejects.toThrow(/onnxruntime-web/);
    await expect(slot.load()).rejects.toThrow(/PrivacyScreen\(\{ image \}\)/);
  });

  it("infer() rejects rather than returning an empty non-sensitive result", async () => {
    // A slot that returned {categories:[],scores:[]} would read as "screened, nothing found" --
    // i.e. a silent allow for every frame on a build that never wired a model.
    const slot = new PrivacyImageAdapter();
    await expect(slot.infer(FRAME)).rejects.toThrow(
      "PrivacyImageAdapter.infer not implemented (adapter slot)."
    );
  });

  it("surfaces opts.version for scorecard attribution, defaulting to null not undefined", () => {
    expect(new PrivacyImageAdapter({ version: "cnn-q8-2026-07" }).version).toBe("cnn-q8-2026-07");
    expect(new PrivacyImageAdapter().version).toBe(null);
    expect(new PrivacyImageAdapter().opts).toEqual({});
  });

  it("the raw slot, handed straight to runImageTier, fails closed instead of propagating", async () => {
    // It has an infer function, so it passes the wiring guard and is actually invoked; the reject is
    // then caught by the error branch. This is the state a half-finished integration ships in.
    expect(await runImageTier(new PrivacyImageAdapter(), FRAME)).toStrictEqual(FAIL_CLOSED);
  });
});

// --- failClosedImage -------------------------------------------------------

describe("failClosedImage — the conservative verdict-partial", () => {
  it("is the exact five-key shape, with score pinned at the ceiling", () => {
    expect(failClosedImage()).toStrictEqual(FAIL_CLOSED);
  });

  it("returns a FRESH object each call — mutating one result cannot poison the next", () => {
    // It is returned from a hot path (~0.25 fps, every egress candidate) and handed to caller code
    // that is free to mutate it. A shared frozen singleton here would let one caller's
    // `verdict.sensitive = false` disable the fail-closed path process-wide.
    const first = failClosedImage();
    first.sensitive = false;
    first.score = 0;
    first.failed = false;
    first.categories.push("credential");
    first.spans.push(box(0, 0, 1, 1));

    const second = failClosedImage();
    expect(second).toStrictEqual(FAIL_CLOSED);
    expect(second.categories).not.toBe(first.categories);
    expect(second.spans).not.toBe(first.spans);
  });

  it("hands out a fresh array pair on every call, not two views of one buffer", () => {
    const a = failClosedImage();
    const b = failClosedImage();
    expect(a).not.toBe(b);
    expect(a.categories).not.toBe(b.categories);
    expect(a.spans).not.toBe(b.spans);
  });
});

// --- runImageTier: the wiring guard ----------------------------------------

describe("runImageTier — no usable adapter", () => {
  it("fails closed for null, undefined, and an object with no infer", async () => {
    for (const adapter of [null, undefined, {}, { load: async () => {} }, { infer: "not-a-fn" }]) {
      expect(await runImageTier(adapter, FRAME)).toStrictEqual(FAIL_CLOSED);
    }
  });

  it("never calls a non-function infer — the guard is typeof, not truthiness", async () => {
    // `{ infer: 1 }` is truthy; a truthiness guard would reach the call and throw a TypeError that
    // the catch would swallow into the same answer, hiding the bug until someone removed the catch.
    expect(await runImageTier({ infer: 1 }, FRAME)).toStrictEqual(FAIL_CLOSED);
  });
});

// --- runImageTier: the happy path ------------------------------------------

describe("runImageTier — a trustworthy adapter result", () => {
  it("returns a FOUR-key verdict with no `failed` key at all", async () => {
    const r = await runImageTier(okAdapter(["sexual"], [0.8]), FRAME);
    expect(r).toStrictEqual({ sensitive: true, categories: ["sexual"], score: 0.8, spans: [] });
    expect("failed" in r).toBe(false);
  });

  it("dedupes and sorts categories, and takes the MAX score across the raw list", async () => {
    // Duplicates are dropped from `categories` but their scores still count toward the max, because
    // the max is accumulated before the Set. A port that deduped first would report 0.4 here.
    const r = await runImageTier(
      okAdapter(["sexual", "document-id", "sexual", "payment"], [0.4, 0.91, 0.7, 0.2]),
      FRAME
    );
    expect(r).toStrictEqual({
      sensitive: true,
      categories: ["document-id", "payment", "sexual"], // lexicographic, not input order
      score: 0.91,
      spans: [],
    });
  });

  it("score is the max, never the sum — adding a category must not inflate it past 1", async () => {
    const r = await runImageTier(okAdapter(["sexual", "payment", "credential"], [0.5, 0.5, 0.5]), FRAME);
    expect(r.score).toBe(0.5);
  });

  it("accepts the closed interval endpoints 0 and 1", async () => {
    expect(await runImageTier(okAdapter(["payment"], [1]), FRAME)).toStrictEqual({
      sensitive: true,
      categories: ["payment"],
      score: 1,
      spans: [],
    });
    // A zero-confidence hit is still `sensitive: true` with score 0. Frozen because it is a real
    // discontinuity: it is indistinguishable from a clean frame on score alone.
    expect(await runImageTier(okAdapter(["payment"], [0]), FRAME)).toStrictEqual({
      sensitive: true,
      categories: ["payment"],
      score: 0,
      spans: [],
    });
  });

  it("an empty category list is a clean frame — not sensitive, score forced to 0", async () => {
    expect(await runImageTier(okAdapter([], []), FRAME)).toStrictEqual({
      sensitive: false,
      categories: [],
      score: 0,
      spans: [],
    });
  });

  it("accepts every category in the shared taxonomy — the tier's vocabulary is not a subset", async () => {
    // privacy-image.js validates against the same PRIVACY_CATEGORIES the text tier uses. If the
    // image tier ever narrowed its vocabulary, an otherwise valid label would fail the frame closed.
    for (const c of PRIVACY_CATEGORIES) {
      const r = await runImageTier(okAdapter([c], [0.6]), FRAME);
      expect(r).toStrictEqual({ sensitive: true, categories: [c], score: 0.6, spans: [] });
    }
  });
});

// --- runImageTier: boxes → spans -------------------------------------------

describe("runImageTier — optional pixel boxes become spans", () => {
  it("passes a well-formed box through as a span", async () => {
    const r = await runImageTier(okAdapter(["document-id"], [0.9], [box(0, 0, 10, 10)]), FRAME);
    expect(r).toStrictEqual({
      sensitive: true,
      categories: ["document-id"],
      score: 0.9,
      spans: [{ start: { x: 0, y: 0 }, end: { x: 10, y: 10 } }],
    });
  });

  it("silently DROPS a malformed box — it does not fail the frame closed", async () => {
    // This is the deliberate asymmetry: boxes are advisory redaction hints, so a bad one costs
    // precision, not safety. A bad category or score is evidence the model itself is untrustworthy,
    // so it costs the whole frame. A port that unified the two would over-mask every frame from an
    // adapter with one sloppy box.
    const r = await runImageTier(
      okAdapter(
        ["sexual"],
        [0.5],
        [box(1, 2, 3, 4), { start: { x: 0, y: 0 } }, { end: { x: 1, y: 1 } }, null, undefined, {}, 7]
      ),
      FRAME
    );
    expect(r).toStrictEqual({
      sensitive: true,
      categories: ["sexual"],
      score: 0.5,
      spans: [{ start: { x: 1, y: 2 }, end: { x: 3, y: 4 } }],
    });
    expect(r.failed).toBeUndefined();
  });

  it("keeps spans even when there are no categories — sensitive false, score 0, spans populated", async () => {
    // The redaction hint survives the "nothing sensitive" verdict. This is what lets a caller redact
    // a region the model flagged spatially but did not label.
    const r = await runImageTier(okAdapter([], [], [box(1, 2, 3, 4)]), FRAME);
    expect(r).toStrictEqual({
      sensitive: false,
      categories: [],
      score: 0,
      spans: [{ start: { x: 1, y: 2 }, end: { x: 3, y: 4 } }],
    });
  });

  it("a non-array `boxes` is ignored, not fatal", async () => {
    for (const boxes of [null, "nope", 5, { start: {}, end: {} }]) {
      const r = await runImageTier(returning({ categories: ["sexual"], scores: [0.5], boxes }), FRAME);
      expect(r).toStrictEqual({ sensitive: true, categories: ["sexual"], score: 0.5, spans: [] });
    }
  });

  it("a span carries the adapter's own start/end objects by reference, not a deep copy", async () => {
    // Frozen because it is load-bearing for an adapter that reuses a scratch box across frames:
    // the verdict aliases that scratch, so the caller sees it mutate. A port that deep-copied would
    // be safer but not identical, and the difference is only visible to a test that checks identity.
    const shared = box(1, 2, 3, 4);
    const r = await runImageTier(okAdapter(["sexual"], [0.5], [shared]), FRAME);
    expect(r.spans[0]).not.toBe(shared); // the wrapper is new
    expect(r.spans[0].start).toBe(shared.start); // the corners are not
    expect(r.spans[0].end).toBe(shared.end);
  });
});

// --- runImageTier: every fail-closed branch of normalizeImageResult --------

describe("runImageTier — a result the SDK cannot vouch for fails closed", () => {
  it("a falsy raw result (null, undefined, false, 0, '')", async () => {
    for (const raw of [null, undefined, false, 0, ""]) {
      expect(await runImageTier(returning(raw), FRAME)).toStrictEqual(FAIL_CLOSED);
    }
  });

  it("non-array `categories`", async () => {
    for (const categories of ["sexual", { 0: "sexual" }, null, undefined, 5]) {
      expect(await runImageTier(returning({ categories, scores: [0.5] }), FRAME)).toStrictEqual(
        FAIL_CLOSED
      );
    }
  });

  it("non-array `scores`", async () => {
    for (const scores of [0.5, "0.5", { 0: 0.5 }, null, undefined]) {
      expect(await runImageTier(returning({ categories: ["sexual"], scores }), FRAME)).toStrictEqual(
        FAIL_CLOSED
      );
    }
  });

  it("categories/scores length mismatch, in both directions", async () => {
    expect(await runImageTier(okAdapter(["sexual"], [0.5, 0.6]), FRAME)).toStrictEqual(FAIL_CLOSED);
    expect(await runImageTier(okAdapter(["sexual", "payment"], [0.5]), FRAME)).toStrictEqual(
      FAIL_CLOSED
    );
    expect(await runImageTier(okAdapter([], [0.5]), FRAME)).toStrictEqual(FAIL_CLOSED);
    expect(await runImageTier(okAdapter(["sexual"], []), FRAME)).toStrictEqual(FAIL_CLOSED);
  });

  it("an unknown label — the model is speaking a vocabulary the SDK does not know", async () => {
    for (const label of ["not-a-category", "PII-ID", "pii_id", "", "face", null, 0]) {
      expect(await runImageTier(okAdapter([label], [0.9]), FRAME)).toStrictEqual(FAIL_CLOSED);
    }
  });

  it("one unknown label poisons an otherwise valid list", async () => {
    // The loop returns on the first bad entry, so a mostly-good result is still rejected whole.
    // Frozen deliberately: partial trust in a model asset the SDK never shipped is the failure mode.
    expect(
      await runImageTier(okAdapter(["sexual", "not-a-category"], [0.9, 0.1]), FRAME)
    ).toStrictEqual(FAIL_CLOSED);
  });

  it("NaN, Infinity and -Infinity scores", async () => {
    for (const s of [NaN, Infinity, -Infinity]) {
      expect(await runImageTier(okAdapter(["sexual"], [s]), FRAME)).toStrictEqual(FAIL_CLOSED);
    }
  });

  it("scores outside the closed unit interval — -0.1 and 1.1", async () => {
    expect(await runImageTier(okAdapter(["sexual"], [-0.1]), FRAME)).toStrictEqual(FAIL_CLOSED);
    expect(await runImageTier(okAdapter(["sexual"], [1.1]), FRAME)).toStrictEqual(FAIL_CLOSED);
    // -0 is inside the interval and must NOT be rejected (-0 < 0 is false).
    expect(await runImageTier(okAdapter(["sexual"], [-0]), FRAME)).toStrictEqual({
      sensitive: true,
      categories: ["sexual"],
      score: 0,
      spans: [],
    });
  });

  it("non-number scores — the typeof guard runs before the range guard", async () => {
    // "0.5" would pass `s < 0 || s > 1` by coercion. Only the typeof check stops it.
    for (const s of ["0.5", null, undefined, true, {}, [0.5], BigInt(1)]) {
      expect(await runImageTier(okAdapter(["sexual"], [s]), FRAME)).toStrictEqual(FAIL_CLOSED);
    }
  });

  it("one bad score poisons the list even when the good scores came first", async () => {
    expect(
      await runImageTier(okAdapter(["sexual", "payment"], [0.9, "high"]), FRAME)
    ).toStrictEqual(FAIL_CLOSED);
  });

  it("every rejection is the SAME object shape — a caller can branch on `failed` alone", async () => {
    const results = [
      await runImageTier(null, FRAME),
      await runImageTier(returning(null), FRAME),
      await runImageTier(okAdapter(["sexual"], [2]), FRAME),
      await runImageTier(okAdapter(["nope"], [0.5]), FRAME),
      await runImageTier({ async infer() { throw new Error("bad weights"); } }, FRAME),
    ];
    for (const r of results) expect(r).toStrictEqual(FAIL_CLOSED);
    // ...and they are distinct objects, so a caller mutating one cannot corrupt another.
    expect(new Set(results).size).toBe(results.length);
  });
});

// --- runImageTier: the error + timeout lifecycle ---------------------------

describe("runImageTier — error and timeout both fail closed", () => {
  it("an async reject fails closed", async () => {
    expect(
      await runImageTier({ async infer() { throw new Error("OOM"); } }, FRAME)
    ).toStrictEqual(FAIL_CLOSED);
  });

  it("a SYNCHRONOUS throw from infer fails closed too", async () => {
    // infer() is called inside the try, as an argument to Promise.race, so a sync throw is caught.
    expect(
      await runImageTier({ infer() { throw new TypeError("corrupt frame"); } }, FRAME)
    ).toStrictEqual(FAIL_CLOSED);
  });

  it("a rejected promise returned without async fails closed", async () => {
    expect(
      await runImageTier({ infer: () => Promise.reject(new Error("wasm trap")) }, FRAME)
    ).toStrictEqual(FAIL_CLOSED);
  });

  it("a hung infer hits budgetMs and fails closed rather than leaking the frame", async () => {
    const t0 = Date.now();
    const r = await runImageTier({ infer: () => new Promise(() => {}) }, FRAME, { budgetMs: 20 });
    expect(r).toStrictEqual(FAIL_CLOSED);
    expect(Date.now() - t0).toBeLessThan(2000); // it resolved on the timer, not on the model
  });

  it("a fast adapter does NOT wait out a long budget — the timer is cleared", async () => {
    // If the pending timeout were awaited or left uncleared, a 5s budget would cost 5s per frame at
    // ~0.25 fps, i.e. the tier would fall permanently behind.
    const t0 = Date.now();
    const r = await runImageTier(okAdapter(["sexual"], [0.5]), FRAME, { budgetMs: 5000 });
    expect(r.sensitive).toBe(true);
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("budgetMs defaults to 300ms when the option bag is absent or empty", async () => {
    const t0 = Date.now();
    expect(await runImageTier({ infer: () => new Promise(() => {}) }, FRAME)).toStrictEqual(
      FAIL_CLOSED
    );
    const elapsed = Date.now() - t0;
    expect(elapsed).toBeGreaterThanOrEqual(250);
    expect(elapsed).toBeLessThan(2000);
  });

  it("an adapter that returns a __timeout marker itself is treated as a timeout", async () => {
    // The race sentinel is an in-band marker on an untrusted, app-supplied payload. Frozen because
    // it means an adapter can never legitimately return a `__timeout` field: it fails closed.
    expect(
      await runImageTier(returning({ __timeout: true, categories: [], scores: [] }), FRAME)
    ).toStrictEqual(FAIL_CLOSED);
  });

  it("a supplied clock is used for the deadline and never appears in the verdict", async () => {
    let calls = 0;
    const clock = () => {
      calls += 1;
      return 1000 + calls;
    };
    const r = await runImageTier(okAdapter(["sexual"], [0.5]), FRAME, { clock });
    expect(calls).toBeGreaterThan(0); // the clock was consulted
    expect(r).toStrictEqual({ sensitive: true, categories: ["sexual"], score: 0.5, spans: [] });
  });

  it("BUG-G-02-01: a throwing clock fails closed — the clock read is INSIDE the try", async () => {
    // The module doctrine is "on timeout OR error we FAIL CLOSED: ... a model that hangs can never
    // leak a frame". The clock is caller-supplied (privacy.js threads it through from PrivacyScreen
    // opts), so a bad clock is an adapter-side fault like any other -- but `started = now()` used to
    // sit ABOVE the try, so the throw escaped runImageTier entirely. The egress caller saw a
    // rejected promise where the contract promises a verdict object, and whether the frame was held
    // depended on the caller's own error handling rather than on the tier's fail-closed rule.
    const r = await runImageTier(okAdapter(["sexual"], [0.5]), FRAME, {
      clock: () => {
        throw new Error("clock backend unavailable");
      },
    });
    expect(r).toStrictEqual(FAIL_CLOSED);
  });

  it("BUG-G-02-01: the same throwing clock reaches egress as a verdict, not a rejection", async () => {
    // The end-to-end shape of the same defect: PrivacyScreen threads `clock` straight through, so
    // before the fix `screenAsync` rejected on a valid frame.
    const screen = new PrivacyScreen({
      image: okAdapter(["sexual"], [0.5]),
      clock: () => {
        throw new Error("clock backend unavailable");
      },
    });
    const v = await screen.screenAsync({ modality: "image", frame: FRAME });
    expect(v.action).toBe("drop"); // fail closed, with the default failAction
    expect(v.sensitive).toBe(true);
  });
});

// --- the fail-closed deadline itself ---------------------------------------

describe("PrivacyScreen — imageBudgetMs is validated like the bands are", () => {
  it("a non-finite or non-positive budget is a construction-time RangeError", () => {
    // Unvalidated, a NaN budget made setTimeout fire at 0ms, so the timeout won the race on EVERY
    // frame: every image verdict was fail-closed and, with the default failAction, every image was
    // dropped at egress -- silently, and indistinguishably from a genuinely hung model. The bands
    // next to it were already validated; this knob was not.
    for (const bad of [NaN, Infinity, -1, 0, "300", null]) {
      expect(() => new PrivacyScreen({ imageBudgetMs: bad }), String(bad)).toThrow(RangeError);
    }
    expect(new PrivacyScreen({ imageBudgetMs: 20 })._imageBudgetMs).toBe(20);
    expect(new PrivacyScreen()._imageBudgetMs).toBe(300);
  });
});
