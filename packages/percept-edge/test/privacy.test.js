import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { scanText } from "../src/privacy-text.js";
import {
  PrivacyScreen,
  makePrivacyVerdict,
  PRIVACY_ACTIONS,
  PRIVACY_SCREEN_VERSION,
} from "../src/privacy.js";
import { PrivacyImageAdapter, runImageTier, failClosedImage } from "../src/privacy-image.js";
import { PrivacyAudioHead, AUDIO_SCORE_CAP } from "../src/privacy-audio.js";

const HERE = dirname(fileURLToPath(import.meta.url));

// --- text tier -------------------------------------------------------------

describe("scanText — deterministic detectors", () => {
  it("catches a valid hyphenated SSN as pii-id and spans the digits", () => {
    const r = scanText("my ssn is 123-45-6789 ok");
    expect(r.sensitive).toBe(true);
    expect(r.categories).toEqual(["pii-id"]);
    const span = r.spans[0];
    expect("my ssn is 123-45-6789 ok".slice(span.start, span.end)).toBe("123-45-6789");
  });

  it("rejects a structurally invalid SSN (reserved area 000/666/900+)", () => {
    expect(scanText("000-45-6789").sensitive).toBe(false);
    expect(scanText("666-45-6789").sensitive).toBe(false);
    expect(scanText("900-45-6789").sensitive).toBe(false);
  });

  it("Luhn-gates payment card numbers (valid fires, off-by-one does not)", () => {
    expect(scanText("card 4111 1111 1111 1111").categories).toEqual(["payment"]);
    expect(scanText("ref 4111 1111 1111 1112").sensitive).toBe(false);
  });

  it("only fires CVV in context and redacts the digits only", () => {
    const r = scanText("the cvv is 123 today");
    expect(r.categories).toEqual(["payment"]);
    expect("the cvv is 123 today".slice(r.spans[0].start, r.spans[0].end)).toBe("123");
    expect(scanText("we are 123 strong").sensitive).toBe(false); // 123 with no CVV context
  });

  it("gates EIN and ABA routing on context + checksum", () => {
    expect(scanText("EIN 12-3456789 filed").categories).toEqual(["financial"]);
    expect(scanText("code 12-3456789").sensitive).toBe(false); // no tax context
    expect(scanText("routing 021000021").categories).toEqual(["financial"]); // valid ABA checksum
    expect(scanText("routing 021000022").sensitive).toBe(false); // checksum fails
  });

  it("catches self-identifying vendor credentials with no context needed", () => {
    expect(scanText("sk-abcdefghijklmnopqrstuvwxyz012345").categories).toEqual(["credential"]);
    expect(scanText("AKIAIOSFODNN7EXAMPLE").categories).toEqual(["credential"]);
    expect(scanText("ghp_abcdefghijklmnopqrstuvwxyz0123456789").categories).toEqual(["credential"]);
    expect(scanText("-----BEGIN RSA PRIVATE KEY-----").categories).toEqual(["credential"]);
  });

  it("gates a generic high-entropy secret on a secret-ish label", () => {
    expect(scanText("api_key = a1b2c3d4e5f6g7h8i9").categories).toEqual(["credential"]);
    expect(scanText("the room code a1b2c3d4e5f6g7h8i9").sensitive).toBe(false);
  });

  it("clean text is not sensitive (over-mask guard)", () => {
    for (const s of ["let's meet at 3pm", "call 555-1234", "order 1234567 shipped", "total was 42 dollars"]) {
      expect(scanText(s).sensitive).toBe(false);
    }
  });

  it("is deterministic — same string yields a byte-identical result", () => {
    const s = "ssn 123-45-6789 and card 4111 1111 1111 1111 and sk-abcdefghijklmnopqrstuvwxyz012345";
    expect(JSON.stringify(scanText(s))).toBe(JSON.stringify(scanText(s)));
  });

  it("categories are deduped + sorted for a multi-hit string", () => {
    const r = scanText("ssn 123-45-6789 card 4111 1111 1111 1111 key sk-abcdefghijklmnopqrstuvwxyz012345");
    expect(r.categories).toEqual(["credential", "payment", "pii-id"]); // sorted
    expect(r.score).toBeCloseTo(0.95, 10); // max weight
  });
});

// --- verdict builder -------------------------------------------------------

describe("makePrivacyVerdict — validation", () => {
  it("accepts a well-formed verdict", () => {
    const v = makePrivacyVerdict({ sensitive: true, categories: ["pii-id"], score: 0.9, spans: [], action: "redact" });
    expect(v).toEqual({ sensitive: true, categories: ["pii-id"], score: 0.9, spans: [], action: "redact" });
  });
  it("rejects an unknown category, out-of-range score, and bad action", () => {
    expect(() => makePrivacyVerdict({ sensitive: true, categories: ["nope"], score: 0.5, action: "drop" })).toThrow();
    expect(() => makePrivacyVerdict({ sensitive: true, score: 1.5, action: "drop" })).toThrow();
    expect(() => makePrivacyVerdict({ sensitive: true, score: 0.5, action: "nuke" })).toThrow();
  });
});

// --- orchestrator: sync text path -----------------------------------------

describe("PrivacyScreen.screen — synchronous text-only hot path", () => {
  it("screens text synchronously and redacts a detected secret", () => {
    const ps = new PrivacyScreen();
    const v = ps.screen({ modality: "text", text: "ssn 123-45-6789" });
    expect(v.action).toBe("redact");
    expect(v.sensitive).toBe(true);
  });
  it("allows clean text", () => {
    const ps = new PrivacyScreen();
    expect(ps.screen({ modality: "text", text: "hello there" }).action).toBe("allow");
  });
  it("throws if handed image/audio (async modalities)", () => {
    const ps = new PrivacyScreen();
    expect(() => ps.screen({ modality: "image", frame: {} })).toThrow();
    expect(() => ps.screen({ modality: "audio", pcm: new Float32Array(4) })).toThrow();
  });
  it("uses no clock in the text path (a throwing clock never runs)", () => {
    const ps = new PrivacyScreen({ clock: () => { throw new Error("clock touched"); } });
    expect(() => ps.screen({ modality: "text", text: "ssn 123-45-6789" })).not.toThrow();
  });
});

// --- orchestrator: async image path + fail-closed --------------------------

const okImageAdapter = (categories, scores, boxes) => ({
  async infer() { return { categories, scores, boxes }; },
});

describe("PrivacyScreen.screenAsync — image tier", () => {
  it("redacts when the model returns a box (surgical)", async () => {
    const ps = new PrivacyScreen({ image: okImageAdapter(["document-id"], [0.9], [{ start: { x: 0, y: 0 }, end: { x: 10, y: 10 } }]) });
    const v = await ps.screenAsync({ modality: "image", frame: {} });
    expect(v.action).toBe("redact");
    expect(v.spans).toHaveLength(1);
  });
  it("drops a confident HARD category with no box", async () => {
    const ps = new PrivacyScreen({ image: okImageAdapter(["credential"], [0.8]) });
    expect((await ps.screenAsync({ modality: "image", frame: {} })).action).toBe("drop");
  });
  it("confirms a low-confidence MED category with no box", async () => {
    const ps = new PrivacyScreen({ image: okImageAdapter(["document-id"], [0.42]) });
    expect((await ps.screenAsync({ modality: "image", frame: {} })).action).toBe("confirm");
  });
  it("allows a clean frame", async () => {
    const ps = new PrivacyScreen({ image: okImageAdapter([], []) });
    expect((await ps.screenAsync({ modality: "image", frame: {} })).action).toBe("allow");
  });
  it("FAILS CLOSED (drop) when the adapter throws", async () => {
    const ps = new PrivacyScreen({ image: { async infer() { throw new Error("bad weights"); } } });
    const v = await ps.screenAsync({ modality: "image", frame: {} });
    expect(v).toMatchObject({ sensitive: true, action: "drop", score: 1 });
  });
  it("FAILS CLOSED (drop) when the adapter exceeds the budget", async () => {
    const ps = new PrivacyScreen({ imageBudgetMs: 20, image: { async infer() { return new Promise(() => {}); } } });
    const v = await ps.screenAsync({ modality: "image", frame: {} });
    expect(v.action).toBe("drop");
    expect(v.sensitive).toBe(true);
  });
  it("FAILS CLOSED when a malformed result is returned (mismatched arrays / unknown label)", async () => {
    const bad1 = new PrivacyScreen({ image: okImageAdapter(["credential"], [0.5, 0.6]) });
    const bad2 = new PrivacyScreen({ image: okImageAdapter(["not-a-category"], [0.9]) });
    expect((await bad1.screenAsync({ modality: "image", frame: {} })).action).toBe("drop");
    expect((await bad2.screenAsync({ modality: "image", frame: {} })).action).toBe("drop");
  });
  it("FAILS CLOSED with no adapter wired at all", async () => {
    const ps = new PrivacyScreen(); // no image adapter
    expect((await ps.screenAsync({ modality: "image", frame: {} })).action).toBe("drop");
  });
  it("honors a configurable failAction", async () => {
    const ps = new PrivacyScreen({ failAction: "confirm", image: { async infer() { throw new Error("x"); } } });
    expect((await ps.screenAsync({ modality: "image", frame: {} })).action).toBe("confirm");
  });
});

// --- orchestrator: audio ceiling -------------------------------------------

describe("PrivacyScreen.screenAsync — audio tier is capped at confirm", () => {
  it("never escalates audio past confirm and asserts no category", async () => {
    // A very loud, mid-ZCR buffer — the strongest the coarse head can score.
    const n = 1600;
    const pcm = new Float32Array(n);
    for (let i = 0; i < n; i++) pcm[i] = 0.9 * Math.sin((2 * Math.PI * i) / 13.5);
    const ps = new PrivacyScreen();
    const v = await ps.screenAsync({ modality: "audio", pcm });
    expect(PRIVACY_ACTIONS.indexOf(v.action)).toBeLessThanOrEqual(PRIVACY_ACTIONS.indexOf("confirm"));
    expect(v.categories).toEqual([]);
  });
  it("scores silence as allow", async () => {
    const ps = new PrivacyScreen();
    expect((await ps.screenAsync({ modality: "audio", pcm: new Float32Array(1600) })).action).toBe("allow");
  });
  it("the audio head score is hard-capped", () => {
    const head = new PrivacyAudioHead();
    const n = 1600;
    const pcm = new Float32Array(n);
    for (let i = 0; i < n; i++) pcm[i] = 0.99 * Math.sin((2 * Math.PI * i) / 13.5);
    expect(head.screen(pcm).score).toBeLessThanOrEqual(AUDIO_SCORE_CAP);
  });
});

describe("PrivacyScreen.screenAsync — misc", () => {
  it("rejects an unknown modality", async () => {
    const ps = new PrivacyScreen();
    await expect(ps.screenAsync({ modality: "smell" })).rejects.toThrow();
  });
  it("load() is idempotent and awaits the image adapter", async () => {
    let loads = 0;
    const ps = new PrivacyScreen({ image: { async load() { loads++; }, async infer() { return { categories: [], scores: [] }; } } });
    await ps.load();
    await ps.load();
    expect(loads).toBe(1);
  });
});

// --- adapter slot + fail-closed helper -------------------------------------

describe("PrivacyImageAdapter / runImageTier", () => {
  it("the shipped adapter stub throws (documented slot, no model)", async () => {
    const a = new PrivacyImageAdapter();
    await expect(a.load()).rejects.toThrow();
    await expect(a.infer({})).rejects.toThrow();
  });
  it("failClosedImage is a conservative sensitive partial", () => {
    expect(failClosedImage()).toMatchObject({ sensitive: true, score: 1, failed: true });
  });
  it("runImageTier with no adapter fails closed", async () => {
    expect(await runImageTier(null, {})).toMatchObject({ failed: true, sensitive: true });
  });
});

// --- the on-device invariant: no network / no external imports -------------

describe("on-device invariant — the privacy sources import nothing external", () => {
  it("imports nothing but relative sibling modules — no bare npm dep, no dynamic import, no fetch", () => {
    for (const f of ["privacy.js", "privacy-text.js", "privacy-image.js", "privacy-audio.js", "privacy-constants.js"]) {
      const src = readFileSync(join(HERE, "..", "src", f), "utf8");
      // strip comments AND string literals so a doc/error mention of "onnxruntime" isn't read as usage —
      // the invariant is about what the code IMPORTS and CALLS, not what its help text says.
      const code = src
        .replace(/\/\/[^\n]*/g, "")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`/g, '""');
      expect(code).not.toMatch(/\bfetch\s*\(/); // no network egress from the on-device screen
      expect(code).not.toMatch(/\bimport\s*\(|\brequire\s*\(/); // no dynamic/CJS imports (could load a dep)
      // every static import resolves to a relative sibling module (./...) — no bare npm dependency
      for (const m of src.matchAll(/^\s*import[^;]*?from\s+["']([^"']+)["']/gm)) {
        expect(m[1].startsWith("./")).toBe(true);
      }
    }
  });
  it("exposes a capability version", () => {
    expect(PRIVACY_SCREEN_VERSION).toBe("0.7.0");
  });
});
