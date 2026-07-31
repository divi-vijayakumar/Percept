// test/privacy-text.test.js
//
// The dedicated suite for the TEXT tier (src/privacy-text.js) and the shared vocabulary
// (src/privacy-constants.js). The aggregate privacy.test.js drives the orchestrator and samples the
// detectors through it; this file pins the detector table itself.
//
// Why a frozen vector table. The text tier is the only synchronous, model-free egress screen in the
// edge package, and it is the piece most likely to gain a second implementation in another runtime.
// No second implementation exists in this repository today, so "the two runtimes agree" cannot be
// asserted against anything here. What this file does instead is freeze the exact contract a port
// has to reproduce -- input string in, the whole verdict object out, byte offsets included -- so a
// divergence shows up as a failing vector rather than as PII leaving the device.
//
// Three properties a port must know, all frozen below:
//   * spans are UTF-16 code-unit offsets, not code points. A runtime that indexes by code point
//     slices a different substring out of any text containing an astral character.
//   * `score` is the MAX hit weight, never a sum. Adding detectors must not inflate it.
//   * the module-level regexes carry the /g flag and are shared across calls, so scanText has to be
//     idempotent: the same string must give the same verdict on the first and the hundredth call.

import { describe, it, expect } from "vitest";
import { scanText } from "../src/privacy-text.js";
import {
  PRIVACY_CATEGORIES,
  PRIVACY_ACTIONS,
  ACTION_RANK,
  HARD_CATEGORIES,
  MED_CATEGORIES,
} from "../src/privacy-constants.js";

// --- vector helpers --------------------------------------------------------

const CLEAN = { sensitive: false, categories: [], score: 0, spans: [] };
const at = (start, end, category) => ({ start, end, category });
const hit = (categories, score, spans) => ({ sensitive: true, categories, score, spans });

/**
 * The frozen table. Each row is [why it is here, input, the entire expected verdict].
 * Every expectation was read off the shipped implementation and is asserted whole -- not
 * `toMatchObject`, not "sensitive is true" -- because a partial assertion is what lets a port drift.
 */
const VECTORS = [
  // -- nothing to find -----------------------------------------------------
  ["the empty string is the identity case", "", CLEAN],
  ["ordinary prose stays clean", "let's meet at 3pm", CLEAN],

  // -- the context-gated bare 9-digit run ----------------------------------
  [
    "a bare 9-digit run fires only with an ssn keyword nearby, and scores below the hyphenated form",
    "ssn 123456789 on file",
    hit(["pii-id"], 0.9, [at(4, 13, "pii-id")]),
  ],
  [
    "the context keyword is matched case-insensitively",
    "SSN 123456789 on file",
    hit(["pii-id"], 0.9, [at(4, 13, "pii-id")]),
  ],
  [
    "the spelled-out keyword arms the same detector",
    "Social Security 123456789",
    hit(["pii-id"], 0.9, [at(16, 25, "pii-id")]),
  ],
  ["with no keyword anywhere the same 9 digits are not sensitive", "123456789", CLEAN],
  [
    "the leading word boundary is load-bearing: digits glued to the keyword do not match",
    "ssn123456789",
    CLEAN,
  ],
  ["a 10-digit run is not a 9-digit run, keyword or not", "ssn 1234567890", CLEAN],

  // -- the structural validity gate ----------------------------------------
  ["group 00 is reserved and never issued, so the hyphenated form is rejected", "123-00-6789", CLEAN],
  ["serial 0000 is reserved and never issued", "123-45-0000", CLEAN],
  ["group 00 is rejected on the bare form too", "ssn 123006789 on file", CLEAN],
  ["serial 0000 is rejected on the bare form too", "ssn 123450000 on file", CLEAN],
  ["area 000 is rejected on the bare form too", "ssn 000456789 on file", CLEAN],
  ["a structurally valid hyphenated identifier scores highest of the two forms", "ssn 123-45-6789", hit(["pii-id"], 0.95, [at(4, 15, "pii-id")])],
  [
    "two valid identifiers in one string yield two spans, so the shared /g regex does not stop at the first",
    "ssn 123-45-6789 and ssn 078-05-1120",
    hit(["pii-id"], 0.95, [at(4, 15, "pii-id"), at(24, 35, "pii-id")]),
  ],

  // -- payment numbers -----------------------------------------------------
  [
    "hyphen separators are absorbed before the checksum, and the span covers the separators too",
    "card 4111-1111-1111-1111",
    hit(["payment"], 0.9, [at(5, 24, "payment")]),
  ],

  // -- capture-group spans exclude the label -------------------------------
  [
    "the span starts at the digits, not at the word that armed the detector",
    "account 1234567890",
    hit(["financial"], 0.6, [at(8, 18, "financial")]),
  ],
  [
    "punctuation between the label and the digits is skipped and stays outside the span",
    "acct: 1234567890",
    hit(["financial"], 0.6, [at(6, 16, "financial")]),
  ],
  ["five digits is under the minimum run length", "account 12345", CLEAN],
  ["six digits is the minimum run length", "account 123456", hit(["financial"], 0.6, [at(8, 14, "financial")])],
  [
    "two labelled runs in one string yield two spans in reading order",
    "account 1234567890 and acct 9876543210",
    hit(["financial"], 0.6, [at(8, 18, "financial"), at(28, 38, "financial")]),
  ],

  // -- the self-identifying chat-platform credential -----------------------
  [
    "a bot credential is caught on its prefix alone, no surrounding context needed",
    "xoxb-1234567890-abcdefghijklmnop",
    hit(["credential"], 0.95, [at(0, 32, "credential")]),
  ],
  [
    "the user-token prefix is in the same class",
    "xoxp-0123456789-abcdefgh",
    hit(["credential"], 0.95, [at(0, 24, "credential")]),
  ],
  ["ten characters after the prefix is the minimum tail", "xoxb-1234567890", hit(["credential"], 0.95, [at(0, 15, "credential")])],
  ["nine characters after the prefix is one short", "xoxb-123456789", CLEAN],
  ["the prefix needs its type letter", "xox-1234567890-abcdefghij", CLEAN],

  // -- several detectors at once -------------------------------------------
  [
    "three detectors in one string: categories dedupe and sort, spans stay in offset order",
    "routing 021000021 acct 1234567890 ssn 123-45-6789",
    hit(
      ["financial", "pii-id"],
      0.95,
      [at(8, 17, "financial"), at(23, 33, "financial"), at(38, 49, "pii-id")],
    ),
  ],
  [
    "two credential detectors sharing a start offset sort by end offset, shortest first",
    "api_key = sk-abcdefghijklmnopqrstuvwxyz012345.tail",
    hit(["credential"], 0.95, [at(10, 45, "credential"), at(10, 50, "credential")]),
  ],
  [
    "overlapping detectors both report; the screen is recall-biased and does not resolve them",
    "ssn account 123456789",
    hit(["financial", "pii-id"], 0.9, [at(12, 21, "pii-id"), at(12, 21, "financial")]),
  ],

  // -- the offset convention a port has to match ---------------------------
  [
    "offsets are UTF-16 code units: one astral character shifts the span by two, not one",
    "\u{1F511} ssn 123-45-6789",
    hit(["pii-id"], 0.95, [at(7, 18, "pii-id")]),
  ],
];

// --- the vectors themselves ------------------------------------------------

describe("scanText -- frozen redaction vectors", () => {
  for (const [why, input, expected] of VECTORS) {
    it(why, () => {
      expect(scanText(input)).toEqual(expected);
    });
  }
});

describe("scanText -- what the frozen vectors imply as a whole", () => {
  it("every span slices a non-empty substring that is inside the input", () => {
    for (const [why, input, expected] of VECTORS) {
      for (const s of expected.spans) {
        expect(s.start, why).toBeGreaterThanOrEqual(0);
        expect(s.end, why).toBeLessThanOrEqual(input.length);
        expect(s.end, why).toBeGreaterThan(s.start);
        expect(input.slice(s.start, s.end).length, why).toBe(s.end - s.start);
      }
    }
  });

  it("the labelled-account spans carry only the digits, never the label", () => {
    for (const input of ["account 1234567890", "acct: 1234567890", "account 123456"]) {
      const r = scanText(input);
      const text = input.slice(r.spans[0].start, r.spans[0].end);
      expect(text).toMatch(/^\d+$/);
      expect(input.startsWith(text)).toBe(false); // the label is upstream of the span
    }
  });

  it("spans come back sorted by start, then by end", () => {
    for (const [why, input] of VECTORS) {
      const { spans } = scanText(input);
      for (let i = 1; i < spans.length; i++) {
        const prev = spans[i - 1];
        const cur = spans[i];
        expect(prev.start < cur.start || (prev.start === cur.start && prev.end <= cur.end), why).toBe(true);
      }
    }
  });

  it("categories are deduped, sorted, and always drawn from the declared taxonomy", () => {
    for (const [why, input] of VECTORS) {
      const { categories, spans } = scanText(input);
      expect(categories, why).toEqual([...new Set(categories)].sort());
      for (const c of categories) expect(PRIVACY_CATEGORIES.has(c), why).toBe(true);
      // every category on a span is announced in `categories`, and vice versa
      expect([...new Set(spans.map((s) => s.category))].sort(), why).toEqual(categories);
    }
  });

  it("the shape of a clean verdict is exactly the documented empty verdict", () => {
    for (const [why, input, expected] of VECTORS) {
      if (expected !== CLEAN) continue;
      const r = scanText(input);
      expect(r, why).toEqual({ sensitive: false, categories: [], score: 0, spans: [] });
      expect(Object.keys(r).sort(), why).toEqual(["categories", "score", "sensitive", "spans"]);
    }
  });

  it("sensitive is true exactly when there is at least one span, and score stays in [0,1]", () => {
    for (const [why, input] of VECTORS) {
      const r = scanText(input);
      expect(r.sensitive, why).toBe(r.spans.length > 0);
      expect(r.score, why).toBeGreaterThanOrEqual(0);
      expect(r.score, why).toBeLessThanOrEqual(1);
      expect(r.score > 0, why).toBe(r.sensitive);
    }
  });
});

// --- the payment-number length window --------------------------------------

/**
 * An independent Luhn oracle. Deliberately a second implementation: the point of the length-bound
 * vectors is to show that a 12- and a 20-digit run are rejected for their LENGTH, and that only
 * holds as a statement if the numbers themselves are known-good under the checksum.
 */
function luhnOracle(digits) {
  const d = [...digits].map(Number).reverse();
  let sum = 0;
  for (let i = 0; i < d.length; i++) {
    let x = d[i];
    if (i % 2 === 1) {
      x *= 2;
      if (x > 9) x -= 9;
    }
    sum += x;
  }
  return sum % 10 === 0;
}

const LENGTH_VECTORS = [
  ["411111111117", false],
  ["4111111111119", true],
  ["4111111111111111", true],
  ["4111111111111111110", true],
  ["41111111111111111115", false],
];

describe("scanText -- the payment detector's 13-to-19 digit window", () => {
  it("every length vector is checksum-valid, so only length can decide the outcome", () => {
    for (const [digits] of LENGTH_VECTORS) {
      expect(luhnOracle(digits), digits).toBe(true);
    }
  });

  for (const [digits, shouldFire] of LENGTH_VECTORS) {
    it(`${digits.length} digits ${shouldFire ? "fires" : "does not fire"}`, () => {
      const r = scanText(digits);
      if (shouldFire) {
        expect(r).toEqual(hit(["payment"], 0.9, [at(0, digits.length, "payment")]));
      } else {
        expect(r).toEqual(CLEAN);
      }
    });
  }

  it("a checksum-invalid number inside the window still does not fire", () => {
    // 4111111111111112 differs from a valid number in the last digit only.
    expect(luhnOracle("4111111111111112")).toBe(false);
    expect(scanText("4111111111111112")).toEqual(CLEAN);
  });
});

// --- score aggregation -----------------------------------------------------

describe("scanText -- score is the maximum hit weight", () => {
  const WEAK = "account 1234567890"; //  financial, 0.6
  const MID = "cvv 123"; //              payment,   0.85
  const STRONG = "ssn 123-45-6789"; //   pii-id,    0.95

  it("the parts have the weights this test reasons about", () => {
    expect(scanText(WEAK).score).toBe(0.6);
    expect(scanText(MID).score).toBe(0.85);
    expect(scanText(STRONG).score).toBe(0.95);
  });

  it("a composite scores the max of its parts, never the sum and never the last hit", () => {
    const combos = [
      [WEAK, MID],
      [WEAK, STRONG],
      [MID, STRONG],
      [WEAK, MID, STRONG],
    ];
    for (const parts of combos) {
      const joined = parts.join(" ");
      const expectedMax = Math.max(...parts.map((p) => scanText(p).score));
      const sum = parts.reduce((t, p) => t + scanText(p).score, 0);
      const r = scanText(joined);
      expect(r.score, joined).toBe(expectedMax);
      expect(r.score, joined).not.toBe(sum);
      expect(r.spans.length, joined).toBe(parts.length);
    }
  });

  it("the score does not depend on the order the detectors happen to hit in", () => {
    expect(scanText(`${WEAK} ${MID}`).score).toBe(scanText(`${MID} ${WEAK}`).score);
    expect(scanText(`${STRONG} ${WEAK}`).score).toBe(scanText(`${WEAK} ${STRONG}`).score);
  });
});

// --- the input contract ----------------------------------------------------

describe("scanText -- input contract", () => {
  it("rejects every non-string with the documented TypeError", () => {
    for (const bad of [null, undefined, 0, 123, true, {}, ["a"], new String("boxed"), Symbol("s"), () => {}]) {
      expect(() => scanText(bad)).toThrow(TypeError);
      expect(() => scanText(bad)).toThrow("scanText: expected a string");
    }
  });

  it("does not mutate or retain the input", () => {
    const s = "ssn 123-45-6789";
    scanText(s);
    expect(s).toBe("ssn 123-45-6789");
  });

  it("returns a fresh object each call, so a caller cannot poison the next verdict", () => {
    const a = scanText("ssn 123-45-6789");
    const b = scanText("ssn 123-45-6789");
    expect(a).not.toBe(b);
    expect(a.spans).not.toBe(b.spans);
    a.spans.push(at(0, 1, "payment"));
    a.categories.push("payment");
    expect(scanText("ssn 123-45-6789")).toEqual(hit(["pii-id"], 0.95, [at(4, 15, "pii-id")]));
  });
});

// --- idempotence: the /g regexes are module-level and shared ---------------

describe("scanText -- repeated calls do not leak regex state", () => {
  it("every vector gives the identical verdict on three consecutive calls", () => {
    for (const [why, input, expected] of VECTORS) {
      expect(scanText(input), why).toEqual(expected);
      expect(scanText(input), why).toEqual(expected);
      expect(scanText(input), why).toEqual(expected);
    }
  });

  it("the whole table replays identically forwards, then backwards", () => {
    const forward = VECTORS.map(([, input]) => JSON.stringify(scanText(input)));
    const backward = [...VECTORS].reverse().map(([, input]) => JSON.stringify(scanText(input))).reverse();
    expect(backward).toEqual(forward);
  });

  it("a non-matching call between two matching ones changes nothing", () => {
    const s = "routing 021000021 acct 1234567890 ssn 123-45-6789";
    const first = scanText(s);
    for (const filler of ["", "nothing here", "just some prose", "0", "----"]) scanText(filler);
    expect(scanText(s)).toEqual(first);
  });

  it("a string with many hits of one detector does not drop hits on a later call", () => {
    const many = "acct 111111 acct 222222 acct 333333 acct 444444";
    const once = scanText(many);
    expect(once.spans).toHaveLength(4);
    for (let i = 0; i < 5; i++) expect(scanText(many)).toEqual(once);
  });

  it("a throwing call leaves no residue behind it", () => {
    const s = "ssn 123-45-6789";
    const before = scanText(s);
    expect(() => scanText(42)).toThrow(TypeError);
    expect(scanText(s)).toEqual(before);
  });
});

// --- the shared vocabulary -------------------------------------------------

const EXPECTED_CATEGORIES = ["credential", "document-id", "financial", "payment", "pii-id", "sexual"];

describe("privacy-constants -- the vocabulary both tiers validate against", () => {
  it("the taxonomy is exactly six categories", () => {
    expect([...PRIVACY_CATEGORIES].sort()).toEqual(EXPECTED_CATEGORIES);
    expect(PRIVACY_CATEGORIES.size).toBe(6);
    for (const c of EXPECTED_CATEGORIES) expect(PRIVACY_CATEGORIES.has(c)).toBe(true);
    expect(PRIVACY_CATEGORIES.has("pii")).toBe(false); // near-miss spellings must not resolve
    expect(PRIVACY_CATEGORIES.has("PII-ID")).toBe(false); // membership is case-sensitive
  });

  it("the taxonomy is a read-only FACADE, not a Set — the three members it exposes", () => {
    // Deliberately not `instanceof Set`: a Set cannot be made immutable (see the freeze test
    // below), so the export is a frozen object carrying exactly `has`, `size` and the iterator.
    // Set-only methods are absent by design; spread first if you want them.
    expect(PRIVACY_CATEGORIES).not.toBeInstanceOf(Set);
    expect(typeof PRIVACY_CATEGORIES.has).toBe("function");
    expect(typeof PRIVACY_CATEGORIES[Symbol.iterator]).toBe("function");
    for (const absent of ["add", "delete", "clear", "forEach", "values", "keys", "entries"]) {
      expect(PRIVACY_CATEGORIES[absent], absent).toBeUndefined();
    }
    // The two consumption shapes the rest of the package and its tests actually use.
    expect(new Set(PRIVACY_CATEGORIES).size).toBe(6);
    expect([...PRIVACY_CATEGORIES].length).toBe(6);
    // Iteration is re-entrant: the facade hands out a FRESH iterator per call, so a second spread
    // is not an exhausted one.
    expect([...PRIVACY_CATEGORIES]).toEqual([...PRIVACY_CATEGORIES]);
  });

  it("the actions are exactly four, ordered least to most restrictive", () => {
    expect(PRIVACY_ACTIONS).toEqual(["allow", "redact", "confirm", "drop"]);
    expect(PRIVACY_ACTIONS[0]).toBe("allow"); // the only non-restrictive action
    expect(PRIVACY_ACTIONS[PRIVACY_ACTIONS.length - 1]).toBe("drop"); // the terminal one
    expect(new Set(PRIVACY_ACTIONS).size).toBe(PRIVACY_ACTIONS.length);
  });

  it("the rank map is frozen and is the index map, so escalation never de-escalates", () => {
    expect(Object.isFrozen(ACTION_RANK)).toBe(true);
    expect(ACTION_RANK).toEqual({ allow: 0, redact: 1, confirm: 2, drop: 3 });
    PRIVACY_ACTIONS.forEach((a, i) => expect(ACTION_RANK[a]).toBe(i));
    expect(Object.keys(ACTION_RANK)).toEqual(PRIVACY_ACTIONS);
    for (let i = 1; i < PRIVACY_ACTIONS.length; i++) {
      expect(ACTION_RANK[PRIVACY_ACTIONS[i]]).toBeGreaterThan(ACTION_RANK[PRIVACY_ACTIONS[i - 1]]);
    }
    expect(ACTION_RANK.nope).toBeUndefined();
  });

  it("a write to the frozen rank map is a no-op", () => {
    const original = ACTION_RANK.drop;
    try {
      ACTION_RANK.drop = 0; // silent in sloppy mode, throws in strict -- either is acceptable
    } catch {
      /* strict-mode TypeError is the stronger outcome */
    }
    expect(ACTION_RANK.drop).toBe(original);
    expect(ACTION_RANK.drop).toBe(3);
  });

  it("the severity classes partition the taxonomy: union is everything, intersection is empty", () => {
    const union = new Set([...HARD_CATEGORIES, ...MED_CATEGORIES]);
    expect([...union].sort()).toEqual([...PRIVACY_CATEGORIES].sort());
    expect(union.size).toBe(PRIVACY_CATEGORIES.size);
    const intersection = [...HARD_CATEGORIES].filter((c) => MED_CATEGORIES.has(c));
    expect(intersection).toEqual([]);
    for (const c of [...HARD_CATEGORIES, ...MED_CATEGORIES]) expect(PRIVACY_CATEGORIES.has(c)).toBe(true);
  });

  it("the classes are the frozen ones, not merely a partition of the right size", () => {
    expect([...HARD_CATEGORIES].sort()).toEqual(["credential", "payment", "pii-id"]);
    expect([...MED_CATEGORIES].sort()).toEqual(["document-id", "financial", "sexual"]);
  });

  it("every category the text tier can emit is classified, so nothing falls through severity", () => {
    const emitted = new Set();
    for (const [, input] of VECTORS) for (const c of scanText(input).categories) emitted.add(c);
    expect(emitted.size).toBeGreaterThan(0);
    for (const c of emitted) {
      expect(HARD_CATEGORIES.has(c) || MED_CATEGORIES.has(c), c).toBe(true);
    }
  });

  // BUG-G-01-01. The docstring said "Frozen Set for O(1) membership + immutability" over a plain
  // `new Set([...])`, which is neither: `Object.freeze` does not touch a Set's contents, so
  // add/delete/clear stay functional on a frozen one. Any importer could widen the taxonomy that
  // scanText, makePrivacyVerdict AND the image tier all validate against.
  it("PRIVACY_CATEGORIES is immutable, as its docstring states", () => {
    const before = new Set(PRIVACY_CATEGORIES);
    try {
      try {
        PRIVACY_CATEGORIES.add("not-a-real-category");
      } catch {
        /* a genuinely immutable set rejects the write here — the facade has no `add` at all */
      }
      expect([...PRIVACY_CATEGORIES].sort()).toEqual([...before].sort());
      expect(Object.isFrozen(PRIVACY_CATEGORIES)).toBe(true);
      // Freezing the FACADE is what makes this stick: reassigning `has` cannot smuggle a category
      // in either, which a frozen-Set-shaped fix would still have allowed.
      try {
        PRIVACY_CATEGORIES.has = () => true;
      } catch {
        /* strict-mode module scope: assigning to a frozen property throws */
      }
      expect(PRIVACY_CATEGORIES.has("not-a-real-category")).toBe(false);
    } finally {
      PRIVACY_CATEGORIES.delete?.("not-a-real-category"); // no-op: the facade exposes no mutator
    }
  });

  it("the taxonomy is intact after the immutability probe above", () => {
    expect([...PRIVACY_CATEGORIES].sort()).toEqual(EXPECTED_CATEGORIES);
  });
});
