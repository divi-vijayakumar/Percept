// src/privacy-constants.js
//
// The PrivacyScreen vocabulary, shared by the orchestrator and every tier (kept in its own module so the
// tiers can validate categories without importing the orchestrator — no import cycle).
//
// These mirror the app's existing egress-gate vocabulary so PrivacyScreen drops into the same paths
// without changing their contracts.

// The taxonomy itself stays module-private. `Object.freeze` on a Set is a no-op for its CONTENTS —
// add/delete/clear are prototype methods and stay fully functional on a frozen Set — so exporting the
// Set directly would have made the "immutability" claim below unenforceable, and this taxonomy is
// what scanText, makePrivacyVerdict and the image tier all validate against: any importer widening it
// widens what three validators accept.
const _categories = new Set([
  "pii-id", //       SSN, tax ID, national ID
  "payment", //      card number, CVV, expiry
  "financial", //    bank account / routing
  "credential", //   API key, token, password
  "sexual", //       nudity / exposed
  "document-id", //  identity document, passport, document in view
]);

/**
 * The category taxonomy a verdict may cite. A frozen READ-ONLY facade over the Set: O(1) `has`,
 * a `size`, and iteration (so `[...PRIVACY_CATEGORIES]` and `new Set(PRIVACY_CATEGORIES)` both
 * work) — and no mutator, so the claim of immutability is now true rather than documented.
 *
 * Deliberately NOT a `Set` instance: `x instanceof Set` is false and the Set-only methods
 * (`add`/`delete`/`clear`, and also `forEach`/`values`/`keys`/`entries`) are absent. Spread or
 * `new Set(PRIVACY_CATEGORIES)` first if you need those.
 */
export const PRIVACY_CATEGORIES = Object.freeze({
  has: (c) => _categories.has(c),
  get size() {
    return _categories.size;
  },
  [Symbol.iterator]: () => _categories[Symbol.iterator](),
});

/** The egress actions, least → most restrictive. Order is load-bearing for escalation (never de-escalate). */
export const PRIVACY_ACTIONS = Object.freeze(["allow", "redact", "confirm", "drop"]);

/** Restrictiveness rank (index into PRIVACY_ACTIONS). Higher = more restrictive. */
export const ACTION_RANK = Object.freeze(
  PRIVACY_ACTIONS.reduce((m, a, i) => ((m[a] = i), m), {})
);

/**
 * Severity classes drive the action mapping. HARD = an unrecoverable leak (a secret you can't rotate
 * back), so it escalates fastest; MED still masks but tolerates a `confirm` at the margin.
 */
export const HARD_CATEGORIES = new Set(["credential", "payment", "pii-id"]);
export const MED_CATEGORIES = new Set(["financial", "document-id", "sexual"]);
