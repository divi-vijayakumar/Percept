// src/privacy-text.js
//
// PrivacyScreen — the TEXT tier. Deterministic, sub-millisecond, NO model, NO clock, NO randomness:
// pure regex + checksums. The same string always yields the same hits, so it carries the strict
// same-input→same-verdict guarantee (the async model tiers only relax it for fail-closed timeouts).
//
// This generalizes the app's scattered v0 secret-detection (regex + Luhn) into one owned detector
// table. Each detector is { category, weight, run(text) -> Hit[] }; a Hit is
//   { category, start, end, weight }     // [start,end) are TEXT OFFSETS for surgical redaction
//
// Recall-biased by construction: any single detector firing marks the text sensitive, and the verdict
// score is the MAX hit weight (the strongest signal wins). Over-masking is the accepted cost — the eval
// harness tracks the per-category over-mask rate so the ceilings stay honest.

import { PRIVACY_CATEGORIES } from "./privacy-constants.js";

// --- checksums -------------------------------------------------------------

/** Luhn mod-10 over a digit string. Payment card numbers (PAN) satisfy this; most random runs don't. */
function luhnOk(digits) {
  let sum = 0;
  let dbl = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48; // '0' === 48
    if (dbl) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    dbl = !dbl;
  }
  return sum % 10 === 0;
}

/** ABA (US bank routing) checksum: weighted 3-7-1 over 9 digits, mod 10 == 0. */
function abaOk(digits) {
  if (digits.length !== 9) return false;
  const w = [3, 7, 1, 3, 7, 1, 3, 7, 1];
  let sum = 0;
  for (let i = 0; i < 9; i++) sum += (digits.charCodeAt(i) - 48) * w[i];
  return sum % 10 === 0;
}

/** SSN structural validity: reject reserved areas (000, 666, 900–999), group 00, serial 0000. */
function ssnOk(area, group, serial) {
  const a = Number(area);
  if (a === 0 || a === 666 || a >= 900) return false;
  if (Number(group) === 0) return false;
  if (Number(serial) === 0) return false;
  return true;
}

// --- context gate ----------------------------------------------------------
//
// A context-gated detector only fires when a nearby keyword is present. It cuts the false-positive rate
// of the weakest patterns (a bare 9-digit run is only an SSN when "ssn"/"social security" is nearby)
// without adding a model. The window is the whole input (short egress-candidate strings), lower-cased.

function hasContext(lowerText, words) {
  return words.some((w) => lowerText.includes(w));
}

// --- detectors -------------------------------------------------------------
//
// Order is irrelevant to the verdict (hits are aggregated, categories sorted) but kept stable for
// deterministic span ordering. Each `run` returns hits with absolute [start,end) offsets.

const RE_SSN_HYPHEN = /\b(\d{3})-(\d{2})-(\d{4})\b/g;
const RE_SSN_BARE = /\b(\d{3})(\d{2})(\d{4})\b/g; // context-gated (any bare 9-digit run)
const RE_PAN = /\b\d(?:[ -]?\d){12,18}\b/g; // 13–19 digits, optional single space/hyphen separators
const RE_CVV = /\b(?:cvv2?|cvc2?|security code)\b\D{0,8}(\d{3,4})\b/gid;
const RE_EIN = /\b(\d{2}-\d{7})\b/gd; // context-gated (employer/tax id)
const RE_ABA = /\b(\d{9})\b/g; // routing: checksum + context gated
const RE_ACCOUNT = /\b(?:account|acct|a\/c)\b\D{0,8}(\d{6,17})\b/gid;
// Vendor credential shapes (self-identifying prefixes; no context needed).
const RE_KEY_OPENAI = /\bsk-[A-Za-z0-9]{20,}\b/g;
const RE_KEY_AWS = /\bAKIA[0-9A-Z]{16}\b/g;
const RE_KEY_GITHUB = /\bgh[porsu]_[A-Za-z0-9]{36,}\b/g;
const RE_KEY_SLACK = /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g;
const RE_KEY_PEM = /-----BEGIN (?:[A-Z0-9 ]*)?PRIVATE KEY-----/g;
// Generic high-entropy secret, context-gated (only when the surrounding text calls it a secret).
const RE_SECRET_GENERIC = /\b(?:token|secret|api[_-]?key|apikey|bearer|password|passwd|pwd)\b\W{0,6}([A-Za-z0-9+/_=.-]{16,})/gid;

function reHits(text, re, category, weight) {
  const out = [];
  re.lastIndex = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    // If a capture group is present, redact only the sensitive span (the digits/secret), not the label.
    if (m.indices && m.indices[1]) {
      out.push({ category, start: m.indices[1][0], end: m.indices[1][1], weight });
    } else {
      out.push({ category, start: m.index, end: m.index + m[0].length, weight });
    }
    if (m.index === re.lastIndex) re.lastIndex++; // guard against zero-width loops
  }
  return out;
}

const DETECTORS = [
  // SSN — pii-id
  {
    run(text) {
      const out = [];
      RE_SSN_HYPHEN.lastIndex = 0;
      let m;
      while ((m = RE_SSN_HYPHEN.exec(text)) !== null) {
        if (ssnOk(m[1], m[2], m[3])) {
          out.push({ category: "pii-id", start: m.index, end: m.index + m[0].length, weight: 0.95 });
        }
      }
      const lower = text.toLowerCase();
      if (hasContext(lower, ["ssn", "social security"])) {
        RE_SSN_BARE.lastIndex = 0;
        while ((m = RE_SSN_BARE.exec(text)) !== null) {
          if (ssnOk(m[1], m[2], m[3])) {
            out.push({ category: "pii-id", start: m.index, end: m.index + m[0].length, weight: 0.9 });
          }
        }
      }
      return out;
    },
  },
  // PAN (payment card) — payment. Luhn-gated so random 16-digit runs don't fire.
  {
    run(text) {
      const out = [];
      RE_PAN.lastIndex = 0;
      let m;
      while ((m = RE_PAN.exec(text)) !== null) {
        const digits = m[0].replace(/[ -]/g, "");
        if (digits.length >= 13 && digits.length <= 19 && luhnOk(digits)) {
          out.push({ category: "payment", start: m.index, end: m.index + m[0].length, weight: 0.9 });
        }
      }
      return out;
    },
  },
  // CVV/CVC in context — payment (span covers the digits only).
  { run: (text) => reHits(text, RE_CVV, "payment", 0.85) },
  // EIN / tax id (NN-NNNNNNN), context-gated — financial.
  {
    run(text) {
      const lower = text.toLowerCase();
      if (!hasContext(lower, ["ein", "tax id", "taxid", "employer id", "tax identification"])) return [];
      return reHits(text, RE_EIN, "financial", 0.7);
    },
  },
  // ABA routing (9-digit, checksum + context) — financial.
  {
    run(text) {
      const lower = text.toLowerCase();
      if (!hasContext(lower, ["routing", "aba", "rtn"])) return [];
      const out = [];
      RE_ABA.lastIndex = 0;
      let m;
      while ((m = RE_ABA.exec(text)) !== null) {
        if (abaOk(m[1])) {
          out.push({ category: "financial", start: m.index, end: m.index + m[0].length, weight: 0.8 });
        }
      }
      return out;
    },
  },
  // Bank account number in context — financial.
  { run: (text) => reHits(text, RE_ACCOUNT, "financial", 0.6) },
  // Vendor credentials (self-identifying) — credential.
  { run: (text) => reHits(text, RE_KEY_OPENAI, "credential", 0.95) },
  { run: (text) => reHits(text, RE_KEY_AWS, "credential", 0.95) },
  { run: (text) => reHits(text, RE_KEY_GITHUB, "credential", 0.95) },
  { run: (text) => reHits(text, RE_KEY_SLACK, "credential", 0.95) },
  { run: (text) => reHits(text, RE_KEY_PEM, "credential", 0.98) },
  // Generic high-entropy secret, context-gated — credential.
  { run: (text) => reHits(text, RE_SECRET_GENERIC, "credential", 0.7) },
];

/**
 * Scan a text string for sensitive content. Deterministic and side-effect-free.
 *
 * @param {string} text
 * @returns {{ sensitive: boolean, categories: string[], score: number,
 *             spans: Array<{start:number,end:number,category:string}> }}
 *   `categories` is deduped + sorted (stable across runs); `score` is the max hit weight (recall bias);
 *   `spans` carry [start,end) text offsets + the category, for surgical redaction.
 */
export function scanText(text) {
  if (typeof text !== "string") {
    throw new TypeError("scanText: expected a string");
  }
  const hits = [];
  for (const det of DETECTORS) {
    for (const h of det.run(text)) hits.push(h);
  }
  if (hits.length === 0) {
    return { sensitive: false, categories: [], score: 0, spans: [] };
  }
  const categories = [...new Set(hits.map((h) => h.category))].sort();
  // guard: every category we emit is a declared privacy category (net catches a typo'd detector).
  for (const c of categories) {
    if (!PRIVACY_CATEGORIES.has(c)) throw new Error(`scanText: detector emitted unknown category "${c}"`);
  }
  const score = hits.reduce((mx, h) => (h.weight > mx ? h.weight : mx), 0);
  const spans = hits
    .map((h) => ({ start: h.start, end: h.end, category: h.category }))
    .sort((a, b) => a.start - b.start || a.end - b.end);
  return { sensitive: true, categories, score, spans };
}
