// src/privacy.js
//
// PrivacyScreen — the on-device, multi-modal privacy classifier that sits on the EGRESS boundary, sibling
// to Edge. Multi-modal in (text | image | audio), one verdict out, in the same allow/redact/drop/confirm
// vocabulary the app's gates already speak — so it drops into existing egress paths without changing them.
//
// CORE INVARIANT: on-device only. Nothing here (nor any tier) makes a network call, ever — you can't ship
// sensitive data to a remote LLM to ask whether it's safe to ship. The learned image model is a
// product-supplied artifact behind an adapter slot (src/privacy-image.js), so percept-edge itself stays
// zero-runtime-dependency.
//
// Determinism: the TEXT path is pure and clock-free (same string → same verdict, always). The async
// image/audio paths are deterministic in their classification too; the ONE relaxation is fail-closed on
// timeout — a hung model yields a conservative sensitive verdict rather than a leak. `clock` feeds only
// the image timeout deadline, never a decision.
//
// Wiring: PrivacyScreen is a SIBLING the app calls at egress, not an Edge semantic adapter. Edge's
// semantic slot is a sync per-frame salience fn on the ingress reflex; the async ~0.25 fps privacy CNN
// would defeat that reflex, and egress-safety ≠ ingress-salience. (An optional read-only bridge that
// surfaces the last image score into Tier0Signal.semantic is left for 0.7.x.)

import { scanText } from "./privacy-text.js";
import { runImageTier } from "./privacy-image.js";
import { PrivacyAudioHead } from "./privacy-audio.js";
import {
  PRIVACY_CATEGORIES,
  PRIVACY_ACTIONS,
  ACTION_RANK,
  HARD_CATEGORIES,
  MED_CATEGORIES,
} from "./privacy-constants.js";

export { PRIVACY_CATEGORIES, PRIVACY_ACTIONS } from "./privacy-constants.js";

/** Version of the PrivacyScreen capability. Distinct from the wire CONTRACT_VERSION (PrivacyVerdict is
 *  a package-local shape, not a Python↔JS message). */
export const PRIVACY_SCREEN_VERSION = "0.7.0";

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * Build + validate a PrivacyVerdict. Mirrors the make-and-validate style in contract.js. There is no
 * parse() twin — this shape never crosses the wire.
 * @returns {{sensitive:boolean,categories:string[],score:number,spans:Array,action:string}}
 */
export function makePrivacyVerdict({ sensitive, categories = [], score = 0, spans = [], action } = {}) {
  if (typeof sensitive !== "boolean") throw new TypeError("makePrivacyVerdict: sensitive must be a boolean");
  if (!Array.isArray(categories)) throw new TypeError("makePrivacyVerdict: categories must be an array");
  for (const c of categories) {
    if (!PRIVACY_CATEGORIES.has(c)) throw new TypeError(`makePrivacyVerdict: unknown category "${c}"`);
  }
  if (!isFiniteNumber(score) || score < 0 || score > 1) {
    throw new TypeError("makePrivacyVerdict: score must be a finite number in [0,1]");
  }
  if (!PRIVACY_ACTIONS.includes(action)) {
    throw new TypeError(`makePrivacyVerdict: action must be one of ${PRIVACY_ACTIONS.join("|")}`);
  }
  if (!Array.isArray(spans)) throw new TypeError("makePrivacyVerdict: spans must be an array");
  return { sensitive, categories: [...categories], score, spans: [...spans], action };
}

function severityOf(categories) {
  if (categories.some((c) => HARD_CATEGORIES.has(c))) return "hard";
  if (categories.some((c) => MED_CATEGORIES.has(c))) return "med";
  return "low";
}

/** Least-restrictive of two actions (used to cap a tier, e.g. audio never past `confirm`). */
function capAction(action, ceiling) {
  return ACTION_RANK[action] <= ACTION_RANK[ceiling] ? action : ceiling;
}

export class PrivacyScreen {
  /**
   * @param {object} [opts]
   * @param {number} [opts.threshold=0.35]  convenience knob = the `confirm` band (when we start acting)
   * @param {{confirm:number,drop:number}} [opts.bands]  full override of the score bands
   * @param {() => number} [opts.clock]  monotonic ms source; used ONLY for the image timeout deadline
   * @param {object} [opts.image]  product-supplied image adapter { async load(), async infer(frame) }
   * @param {object} [opts.audio]  audio head (defaults to the coarse PrivacyAudioHead; provisional in 0.7)
   * @param {number} [opts.imageBudgetMs=300]  fail-closed deadline for image inference
   * @param {"allow"|"redact"|"confirm"|"drop"} [opts.failAction="drop"]  action on a fail-closed tier
   */
  constructor({
    threshold = 0.35,
    bands = null,
    clock = null,
    image = null,
    audio = undefined,
    imageBudgetMs = 300,
    failAction = "drop",
  } = {}) {
    const confirm = bands?.confirm ?? threshold;
    const drop = bands?.drop ?? Math.max(0.6, confirm);
    if (!(confirm >= 0 && confirm <= 1) || !(drop >= confirm && drop <= 1)) {
      throw new RangeError("PrivacyScreen: require 0 <= confirm <= drop <= 1");
    }
    if (!PRIVACY_ACTIONS.includes(failAction)) {
      throw new TypeError(`PrivacyScreen: failAction must be one of ${PRIVACY_ACTIONS.join("|")}`);
    }
    this.bands = { confirm, drop };
    this._clock = clock;
    this._image = image;
    // Default to the coarse acoustic head so the audio socket is live behind the interface; it only ever
    // raises `confirm` (honestly bounded — see privacy-audio.js). Pass audio:null to disable it.
    this._audio = audio === undefined ? new PrivacyAudioHead() : audio;
    this._imageBudgetMs = imageBudgetMs;
    this._failAction = failAction;
    this._loaded = false;
  }

  /** Lazy model init (mirrors Edge.loadSkills / SkillBank.load). Idempotent; no-op without an image adapter. */
  async load() {
    if (this._loaded) return;
    if (this._image && typeof this._image.load === "function") await this._image.load();
    if (this._audio && typeof this._audio.load === "function") await this._audio.load();
    this._loaded = true;
  }

  /**
   * SYNCHRONOUS screen — text only (the sub-millisecond hot path on every message). Throws for image/audio
   * (their inference is async): use screenAsync().
   * @param {{modality:"text",text:string}} input
   * @returns {{sensitive:boolean,categories:string[],score:number,spans:Array,action:string}}
   */
  screen(input) {
    if (!input || input.modality !== "text") {
      throw new TypeError("PrivacyScreen.screen is text-only and synchronous — use screenAsync() for image/audio");
    }
    return this._verdict(scanText(input.text), "text");
  }

  /**
   * ASYNC screen — all modalities. Image/audio await their tier; image runs under a fail-closed timeout.
   * @param {{modality:"text"|"image"|"audio", text?:string, frame?:object, pcm?:Float32Array}} input
   * @returns {Promise<{sensitive:boolean,categories:string[],score:number,spans:Array,action:string}>}
   */
  async screenAsync(input) {
    if (!input || typeof input.modality !== "string") {
      throw new TypeError("PrivacyScreen.screenAsync: input must be { modality, ... }");
    }
    switch (input.modality) {
      case "text":
        return this._verdict(scanText(input.text), "text");
      case "image": {
        const part = await runImageTier(this._image, input.frame, {
          budgetMs: this._imageBudgetMs,
          clock: this._clock,
        });
        return this._verdict(part, "image");
      }
      case "audio": {
        if (!this._audio) {
          // No audio head configured → fail closed (never silently allow an unscreened modality).
          return this._verdict({ sensitive: true, categories: [], score: 1, spans: [], failed: true }, "audio");
        }
        return this._verdict(this._audio.screen(input.pcm), "audio");
      }
      default:
        throw new TypeError(`PrivacyScreen.screenAsync: unknown modality "${input.modality}"`);
    }
  }

  /** Map a tier's verdict-partial → a full, validated PrivacyVerdict. Pure + deterministic. */
  _verdict(part, modality) {
    const action = this._decide(part, modality);
    return makePrivacyVerdict({
      sensitive: action !== "allow",
      categories: part.categories || [],
      score: part.score || 0,
      spans: part.spans || [],
      action,
    });
  }

  /**
   * The action ladder — a pure function of (partial, modality, bands). Recall-biased; always escalates to
   * the most restrictive applicable rung, never de-escalates.
   *
   *   not sensitive                         → allow
   *   fail-closed (timeout/error/malformed) → failAction (default drop)
   *   we know WHERE it is (spans present)   → redact   (surgical: mask the span, keep the rest — no leak)
   *   confident, unlocalizable (no spans)   → drop     (can't mask it → drop the whole payload)
   *   HARD + past confirm band              → drop     (unrecoverable category — bias hard toward masking)
   *   past confirm band                     → confirm  (ask/hold)
   *   below the confirm band                → allow
   *   audio is capped at `confirm` (no on-device STT can justify a hard drop)
   */
  _decide(part, modality) {
    if (!part || !part.sensitive) return "allow";
    if (part.failed) return this._failAction;

    const { confirm, drop } = this.bands;
    const s = part.score || 0;
    const hasSpans = Array.isArray(part.spans) && part.spans.length > 0;
    const sev = severityOf(part.categories || []);

    let action;
    if (hasSpans) action = "redact";
    else if (s >= drop) action = "drop";
    else if (s >= confirm) action = sev === "hard" ? "drop" : "confirm";
    else action = "allow";

    if (modality === "audio") action = capAction(action, "confirm");
    return action;
  }
}
