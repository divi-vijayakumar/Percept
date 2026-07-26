// src/privacy-constants.js
//
// The PrivacyScreen vocabulary, shared by the orchestrator and every tier (kept in its own module so the
// tiers can validate categories without importing the orchestrator — no import cycle).
//
// These mirror the app's existing egress-gate vocabulary so PrivacyScreen drops into the same paths
// without changing their contracts.

/** The category taxonomy a verdict may cite. Frozen Set for O(1) membership + immutability. */
export const PRIVACY_CATEGORIES = new Set([
  "pii-id", //       SSN, tax ID, national ID
  "payment", //      card number, CVV, expiry
  "financial", //    bank account / routing
  "credential", //   API key, token, password
  "sexual", //       nudity / exposed
  "document-id", //  identity document, passport, document in view
]);

/** The egress actions, least → most restrictive. Order is load-bearing for escalation (never de-escalate). */
export const PRIVACY_ACTIONS = ["allow", "redact", "confirm", "drop"];

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
