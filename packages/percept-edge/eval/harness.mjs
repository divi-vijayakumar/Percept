// eval/harness.mjs — the PrivacyScreen eval harness (importable by run.mjs and the vitest gate).
//
// Mirrors the parity/ harness shape: deterministic labeled fixtures → run through the real runtime →
// structured scorecard. The IMAGE tier is exercised via a GOLDEN MOCK adapter (fixed {categories,scores,
// boxes} per case) so wiring + action-mapping are tested with NO model asset — the app owns the real CNN.
//
// Metrics (the app's integration precondition — a regression-testable scorecard):
//   leak-rate   = 1 − recall     (a sensitive payload we FAILED to catch — the unrecoverable cost)
//   over-mask   = 1 − specificity (a clean payload we masked — the tolerable cost)
//   recall / precision, per category and overall
//   added-latency p50 / p95      (printed, not gated — nondeterministic wall-clock)

import { PrivacyScreen } from "../src/privacy.js";

const round = (x) => Math.round(x * 1e4) / 1e4;

/** Deterministic PCM synthesis for audio fixtures (no Math.random → stable). */
function synthPcm(spec = {}) {
  const n = spec.n ?? 1600;
  const out = new Float32Array(n);
  if (spec.kind === "tone") {
    const A = spec.amp ?? 0.5;
    const period = spec.period ?? 14; // ~ZCR 0.15 (voiced-articulation band center)
    for (let i = 0; i < n; i++) out[i] = A * Math.sin((2 * Math.PI * i) / period);
  }
  // "silence" (and default) → zeros
  return out;
}

/** A one-shot image adapter returning a fixture's golden result (or hanging, to exercise fail-closed). */
function mockImageAdapter(mock = {}) {
  return {
    async infer() {
      if (mock.__hang) return new Promise(() => {}); // never resolves → runImageTier times out → fail closed
      return { categories: mock.categories ?? [], scores: mock.scores ?? [], boxes: mock.boxes };
    },
  };
}

/** Run one labeled case through a fresh PrivacyScreen. Returns { verdict, ms }. */
export async function runCase(c) {
  const opts = { imageBudgetMs: 50 };
  if (c.input.modality === "image") opts.image = mockImageAdapter(c.input.imageMock);
  const screen = new PrivacyScreen(opts);

  let input = c.input;
  if (c.input.modality === "audio") input = { modality: "audio", pcm: synthPcm(c.input.pcmSpec) };

  const t0 = performance.now();
  const verdict =
    c.input.modality === "text"
      ? screen.screen({ modality: "text", text: c.input.text })
      : await screen.screenAsync(input);
  const ms = performance.now() - t0;
  return { verdict, ms };
}

/** Score a whole fixture set → the structured scorecard. */
export async function scoreCases(cases) {
  const rows = [];
  for (const c of cases) rows.push({ c, ...(await runCase(c)) });

  let tp = 0, fp = 0, fn = 0, tn = 0;
  const perCat = {};
  const latencies = [];
  const actionMismatches = [];

  for (const { c, verdict, ms } of rows) {
    latencies.push(ms);

    const expSens = !!c.expect.sensitive;
    const gotSens = verdict.sensitive;
    if (expSens && gotSens) tp++;
    else if (!expSens && gotSens) fp++;
    else if (expSens && !gotSens) fn++;
    else tn++;

    if (c.expect.action && c.expect.action !== verdict.action) {
      actionMismatches.push({ id: c.id, expected: c.expect.action, got: verdict.action });
    }

    const expCats = new Set(c.expect.categories ?? []);
    const gotCats = new Set(verdict.categories ?? []);
    for (const cat of new Set([...expCats, ...gotCats])) {
      perCat[cat] ??= { tp: 0, fp: 0, fn: 0 };
      const e = expCats.has(cat), g = gotCats.has(cat);
      if (e && g) perCat[cat].tp++;
      else if (!e && g) perCat[cat].fp++;
      else if (e && !g) perCat[cat].fn++;
    }
  }

  const recall = tp + fn ? tp / (tp + fn) : 1;
  const precision = tp + fp ? tp / (tp + fp) : 1;
  const perCategory = {};
  for (const [cat, m] of Object.entries(perCat)) {
    perCategory[cat] = {
      recall: round(m.tp + m.fn ? m.tp / (m.tp + m.fn) : 1),
      precision: round(m.tp + m.fp ? m.tp / (m.tp + m.fp) : 1),
      leakRate: round(m.tp + m.fn ? m.fn / (m.tp + m.fn) : 0),
      tp: m.tp, fp: m.fp, fn: m.fn,
    };
  }

  latencies.sort((a, b) => a - b);
  const pct = (p) => (latencies.length ? latencies[Math.min(latencies.length - 1, Math.floor(p * latencies.length))] : 0);

  return {
    n: cases.length,
    // quality metrics are deterministic → safe to regression-test
    quality: {
      recall: round(recall),
      precision: round(precision),
      leakRate: round(1 - recall),
      overMaskRate: round(fp + tn ? fp / (fp + tn) : 0),
      confusion: { tp, fp, fn, tn },
      perCategory,
      actionMismatches,
    },
    // latency is wall-clock → printed only, never gated
    latencyMs: { p50: round(pct(0.5)), p95: round(pct(0.95)) },
  };
}

/** Only the deterministic slice, for baseline comparison (latency excluded). */
export function qualityOf(scorecard) {
  return scorecard.quality;
}
