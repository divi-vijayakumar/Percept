// eval/run.mjs — human-readable PrivacyScreen scorecard + regression gate.
//
//   node eval/run.mjs            print the scorecard, compare to the checked-in baseline, exit non-zero on
//                                any quality regression (leak-rate up, action mismatch, category drift)
//   node eval/run.mjs --update   overwrite eval/scorecard.baseline.json with the current quality metrics
//
// Latency (p50/p95) is printed but never gates — it is wall-clock and machine-dependent.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { scoreCases, qualityOf } from "./harness.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const CASES = JSON.parse(readFileSync(join(HERE, "fixtures", "privacy.cases.json"), "utf8"));
const BASELINE_PATH = join(HERE, "scorecard.baseline.json");

// Recall is the metric that must not regress: a leaked secret is unrecoverable. Ceiling is generous for
// the coarse audio head (deferred to 0.7.x) but strict on text/image leak-rate.
const LEAK_RATE_CEILING = 0.0; // text + image fixtures must have ZERO leaks

const sc = await scoreCases(CASES.cases);
const q = qualityOf(sc);

const bar = "=".repeat(72);
console.log(bar);
console.log("  PRIVACYSCREEN EVAL SCORECARD   (percept-edge)");
console.log(bar);
console.log(`  cases: ${sc.n}`);
console.log(
  `  overall   recall ${q.recall.toFixed(3)}   precision ${q.precision.toFixed(3)}   ` +
    `leak-rate ${q.leakRate.toFixed(3)}   over-mask ${q.overMaskRate.toFixed(3)}`
);
console.log(`  confusion   tp=${q.confusion.tp}  fp=${q.confusion.fp}  fn=${q.confusion.fn}  tn=${q.confusion.tn}`);
console.log("-".repeat(72));
console.log("  per-category   recall  prec   leak   (tp/fp/fn)");
for (const [cat, m] of Object.entries(q.perCategory).sort()) {
  console.log(
    `   ${cat.padEnd(12)} ${m.recall.toFixed(3)}  ${m.precision.toFixed(3)}  ${m.leakRate.toFixed(3)}   ` +
      `(${m.tp}/${m.fp}/${m.fn})`
  );
}
console.log("-".repeat(72));
console.log(`  added-latency   p50 ${sc.latencyMs.p50.toFixed(2)} ms   p95 ${sc.latencyMs.p95.toFixed(2)} ms`);
console.log(bar);

if (process.argv.includes("--update")) {
  writeFileSync(BASELINE_PATH, JSON.stringify(q, null, 2) + "\n");
  console.log("  baseline updated →", BASELINE_PATH);
  process.exit(0);
}

// --- regression gate -------------------------------------------------------
let baseline = null;
try {
  baseline = JSON.parse(readFileSync(BASELINE_PATH, "utf8"));
} catch {
  console.error("  no baseline found — run `node eval/run.mjs --update` to create one");
  process.exit(2);
}

const problems = [];
if (q.leakRate > LEAK_RATE_CEILING) problems.push(`leak-rate ${q.leakRate} > ceiling ${LEAK_RATE_CEILING}`);
if (q.actionMismatches.length) problems.push(`action mismatches: ${JSON.stringify(q.actionMismatches)}`);
if (JSON.stringify(q) !== JSON.stringify(baseline)) {
  problems.push("quality metrics drifted from baseline (run --update if intended)");
}

if (problems.length) {
  console.error("  REGRESSION:");
  for (const p of problems) console.error("   - " + p);
  process.exit(1);
}
console.log("  OK — no regression vs baseline");
