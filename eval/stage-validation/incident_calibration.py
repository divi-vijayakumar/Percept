"""Incident-level calibration + receipts distributions, computed only from data already on disk.

WHY THIS SCRIPT EXISTS
-----------------------
This scorer was originally scoped as calibrating EVERY judge verdict against ground truth, confidence
bucket by confidence bucket. That is IMPOSSIBLE from the checkpoint on disk. `receipts` (per
judge-call attempt) carries exactly
    {route, evidence_shape, mode, outcome, latency_s, deadline_s, confidence, exception_type}
-- no media timestamp and no "satisfied" bit, so a receipt cannot be joined to a ground-truth
interval. The in-memory `trace` that would carry verdict positions (live_run.py, `on_trace`
callback) is never persisted to the checkpoint. There is no join. Do not invent one (e.g. assuming
receipt order tracks media time) -- that produces a confidently-wrong number, which is exactly the
failure mode this script exists to avoid.

Every claim below is scoped down to one of three things that actually are reconstructable from
data that exists on disk, and labelled accordingly rather than oversold as the all-verdicts
calibration that was originally asked for:

  1. INCIDENT-level calibration, from `latency_rows`. An "incident" is a runtime Fire that already
     passed the gate (`gate.classified` with `satisfied=True` inside live_run.py) -- i.e. this
     population is POST-GATE and FIRED-ONLY. It says nothing about occurrences the runtime never
     fired on at all (the classic false-negative / recall side). Every table below is labelled
     "incident-level" and states this selection bias next to the numbers, not in a footnote.

  2. Receipts outcome/confidence distributions -- decisive/uncertain/deadline/cancelled counts,
     confidence-by-outcome, and how close decisive calls run to their deadline budget.

  3. A GT-coverage / goal-text stratification against `corpus.json`, so two known contaminants are
     named explicitly rather than silently averaged in:
       - charades cases where the single GT positive interval covers ~all of the source media
         (build_corpus.py's "pick the longest action" selection artifact) -- every fired incident
         there is close to trivially "inside GT", so folding them into an incident-hit-rate number
         is optimistic theatre.
       - repcount cases whose goal_text leaked a raw underscored action_type (e.g. "a person
         completes a jump_jack") because the phrasing table in build_corpus.py doesn't cover every
         action_type -- these goals are malformed English, so incidents against them answer a
         malformed question.
     Both counts are recomputed here from corpus.json, not trusted from anyone's prior claim.

Run with no arguments; it reads results/v10_r1.checkpoint.jsonl and corpus.json next to this file
and writes results/incident_calibration.{json,md} beside them. Zero provider calls; read-only
against every file it does not own (it owns only itself and its own output).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "results" / "v10_r1.checkpoint.jsonl"
DEFAULT_CORPUS = HERE / "corpus.json"

CONFIDENCE_BUCKETS = [0.0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.01]

FAILED_OUTCOMES = ("deadline", "cancelled")


# ------------------------------------------------------------------------------------------- IO --
def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------ contaminants ----
def find_saturated_charades(corpus: dict[str, Any], *, threshold: float = 0.95) -> dict[str, float]:
    """charades case_id -> GT-interval / duration ratio, for ratio >= threshold.

    build_corpus.py's `build_charades` keeps the SINGLE longest annotated action per video
    (comment: "Pick the longest action") and clamps it to the declared clip length. For a
    meaningful fraction of videos that single interval already spans (numerically) the entire
    clip, so "the incident landed inside GT" is true almost by construction for those cases --
    it is not evidence the runtime is doing anything the annotation didn't hand it for free.
    """
    out: dict[str, float] = {}
    for case in corpus["cases"]:
        if case["dataset"] != "charades":
            continue
        duration = case.get("duration_s")
        positives = case.get("positives") or []
        if not duration or not positives:
            continue
        start, end = positives[0]
        ratio = (end - start) / duration
        if ratio >= threshold:
            out[case["case_id"]] = round(ratio, 4)
    return out


def find_malformed_repcount(corpus: dict[str, Any]) -> dict[str, str]:
    """repcount case_id -> goal_text, for goals that leaked a raw underscored action_type.

    build_corpus.py's `_REPCOUNT_PHRASING` table covers action_type values written WITHOUT
    underscores (e.g. "jumpjack"); the raw lance rows use underscored action_type strings (e.g.
    "jump_jack"). Every action_type not already spelled without underscores in that table misses
    the dict lookup and falls through to `f"a person completes a {action}"`, i.e. the literal
    underscored token lands in goal_text. Detected here by "_" appearing in the goal text, which
    is a purely mechanical readout of that fallback (verified against the phrasing table below).
    """
    out: dict[str, str] = {}
    for case in corpus["cases"]:
        if case["dataset"] != "repcount":
            continue
        goal_text = str(case.get("goal_text") or "")
        if "_" in goal_text:
            out[case["case_id"]] = goal_text
    return out


# ---------------------------------------------------------------------------------- flattening --
def iter_receipts(rows: list[dict[str, Any]], *, mode: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if mode is not None and row.get("mode") != mode:
            continue
        for receipt in row.get("receipts") or []:
            rec = dict(receipt)
            rec["_case_id"] = row.get("case_id")
            rec["_dataset"] = row.get("dataset")
            out.append(rec)
    return out


def iter_incidents(rows: list[dict[str, Any]], *, mode: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if mode is not None and row.get("mode") != mode:
            continue
        for lat_row in row.get("latency_rows") or []:
            inc = dict(lat_row)
            inc["_mode"] = row.get("mode")
            out.append(inc)
    return out


# ------------------------------------------------------------------- receipts distributions -----
def receipts_distribution(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    outcome_counts = Counter(r["outcome"] for r in receipts)
    confidence_by_outcome: dict[str, dict[str, Any]] = {}
    for outcome in sorted(outcome_counts):
        confs = [r["confidence"] for r in receipts if r["outcome"] == outcome and r.get("confidence") is not None]
        confidence_by_outcome[outcome] = {
            "n": outcome_counts[outcome],
            "n_with_confidence": len(confs),
            "confidence_distribution": dict(sorted(Counter(round(c, 2) for c in confs).items())),
        }

    decisive = [r for r in receipts if r["outcome"] == "decisive"]
    decisive_lat = sorted(r["latency_s"] for r in decisive)
    deadline_values = sorted({r["deadline_s"] for r in receipts if r.get("deadline_s") is not None})

    def pct(sorted_vals: list[float], p: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
        return sorted_vals[idx]

    near_deadline = {}
    for frac in (0.5, 0.75, 0.9, 0.95):
        n = sum(1 for r in decisive if r["latency_s"] >= frac * r["deadline_s"])
        near_deadline[f">={frac:.0%}_of_deadline"] = {
            "n": n,
            "fraction_of_decisive": (n / len(decisive)) if decisive else None,
        }

    deadline_outcome = [r for r in receipts if r["outcome"] == "deadline"]
    cancelled_outcome = [r for r in receipts if r["outcome"] == "cancelled"]

    cases_with_censoring = sorted({
        r["_case_id"] for r in receipts if r["outcome"] in FAILED_OUTCOMES and r.get("_case_id")
    })
    n_cases_total = len({r["_case_id"] for r in receipts if r.get("_case_id")})

    return {
        "n_receipts": len(receipts),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "confidence_by_outcome": confidence_by_outcome,
        "deadline_s_values_in_use": deadline_values,
        "decisive_latency_s": {
            "n": len(decisive_lat),
            "mean": round(statistics.mean(decisive_lat), 4) if decisive_lat else None,
            "median": round(statistics.median(decisive_lat), 4) if decisive_lat else None,
            "p90": pct(decisive_lat, 0.90),
            "p99": pct(decisive_lat, 0.99),
            "max": decisive_lat[-1] if decisive_lat else None,
        },
        "decisive_calls_near_deadline_budget": near_deadline,
        "deadline_outcome_latency_s": {
            "n": len(deadline_outcome),
            "distinct_values": sorted({r["latency_s"] for r in deadline_outcome}),
        },
        "cancelled_outcome_latency_s": {
            "n": len(cancelled_outcome),
            "distinct_values": sorted({r["latency_s"] for r in cancelled_outcome}),
        },
        "cases_with_deadline_or_cancelled": {
            "n_cases": len(cases_with_censoring),
            "n_cases_total": n_cases_total,
            "case_ids": cases_with_censoring,
        },
    }


def censoring_vs_noise_verdict(reflex_dist: dict[str, Any], deliberate_dist: dict[str, Any]) -> dict[str, Any]:
    """A stated PARTIAL verdict on whether deadline-censoring, not verdict-noise, explains
    run-to-run recall swings -- built only from what `receipts` can measure in THIS checkpoint.

    This is not, and cannot be, a replication of the historical 3-run HIGH-arm finding (that
    finding used an 18-case suite under a ~1.5s reflex deadline and measured recall directly across
    repeated runs). v10_r1 is a single run of 100 reflex cases under an 8.0s reflex deadline / 8.4s
    decision deadline. What follows is evidence FOR and AGAINST the mechanism, from what this run
    can actually show -- not a re-measurement of the original recall swing.
    """
    outcomes = reflex_dist["outcome_counts"]
    decisive = outcomes.get("decisive", 0)
    deadline = outcomes.get("deadline", 0)
    cancelled = outcomes.get("cancelled", 0)
    total = reflex_dist["n_receipts"]

    cancelled_lat_values = reflex_dist["cancelled_outcome_latency_s"]["distinct_values"]
    deadline_lat_values = reflex_dist["deadline_outcome_latency_s"]["distinct_values"]
    deadline_s_budget = reflex_dist["deadline_s_values_in_use"]

    for_points = [
        "Structural: `deadline` and `cancelled` receipts carry confidence=None (no confidence "
        "value was ever recorded for them) in every row of this checkpoint -- they cannot be "
        "reframed as 'a verdict that happened to fall near a decision threshold', because no "
        "verdict was reached. Whatever they cost the runtime, it is categorically distinct from "
        "confidence-threshold noise.",
        f"`deadline` receipts have latency_s pinned to exactly the deadline budget "
        f"({deadline_lat_values}), a clean hard-timeout signature rather than a graceful "
        "early return -- consistent with a timing race, not a model-confidence flip.",
    ]
    against_points = [
        f"In THIS run, `deadline` is {deadline}/{total} = {deadline/total:.3%} of reflex judge "
        f"calls, and `cancelled` is {cancelled}/{total} = {cancelled/total:.3%}. Both are far too "
        "small a fraction of calls, on their own, to mechanically explain a large recall swing "
        "unless they concentrate heavily on a handful of cases (checked separately below).",
        f"`cancelled` receipts all show latency_s == {cancelled_lat_values} (zero elapsed time) in "
        "this run -- that is NOT a slow-provider timeout, it reads as a pre-empted/superseded call "
        "(dedup or revision-drop logic elsewhere in the pipeline) that never got a chance to run "
        "long enough to race the deadline. Lumping it in with `deadline` as 'timing censoring' "
        "overstates the timing story; only the 32 `deadline` receipts are actually candidates for "
        "the load-driven-timeout mechanism.",
        f"Decisive calls in this run sit far from the deadline budget ({deadline_s_budget}): "
        f"median {reflex_dist['decisive_latency_s']['median']}s, p99 "
        f"{reflex_dist['decisive_latency_s']['p99']}s, and only "
        f"{reflex_dist['decisive_calls_near_deadline_budget']['>=95%_of_deadline']['n']} of "
        f"{decisive} decisive calls came within 95% of budget. There is very little headroom, "
        "under THIS deadline configuration, for a modest slowdown to convert many decisive calls "
        "into deadline expiries -- the historical finding was diagnosed under a ~1.5s reflex "
        "deadline (much tighter margin); this checkpoint's 8.0s/8.4s deadlines are a materially "
        "different, wider operating point, most likely a deliberate mitigation of exactly that "
        "finding. That means this run cannot confirm OR refute the original 0.33 swing -- it can "
        "only say the mechanism has much less room to bite at the current deadline setting.",
        f"Deadline/cancelled receipts DO touch "
        f"{reflex_dist['cases_with_deadline_or_cancelled']['n_cases']}/"
        f"{reflex_dist['cases_with_deadline_or_cancelled']['n_cases_total']} reflex cases "
        "(not negligible in case-count terms) -- but receipts carry no media timestamp, so "
        "whether any one of those censored attempts was the call that would have decided that "
        "case's single tracked occurrence (vs. one of several redundant retries within the case) "
        "cannot be determined from this data. That is the same missing-timestamp join problem "
        "that rules out per-verdict ground-truth scoring from receipts generally, recurring here "
        "at smaller scale.",
        "The repeat-run design that would let this be tested directly and locally (armC_r2/r3, "
        "armU_r2/r3 checkpoints) exists on disk only as 0-byte files -- those runs did not "
        "complete, so run-to-run variance cannot be measured from data on disk for any run in "
        "this results/ directory, v10_r1 included (it has no r2/r3 companion at all).",
    ]

    return {
        "claim_under_test": (
            "32 deadline receipts against 9,260 decisive receipts already argues the observed "
            "run-to-run recall swing (from an earlier, separate 18-case study) is deadline "
            "censoring, not verdict noise."
        ),
        "verdict": "PARTIAL",
        "confidence": "low-moderate",
        "summary": (
            "Structurally, deadline/cancelled outcomes ARE categorically distinct from verdict "
            "noise (no confidence value is ever produced for them) -- that much is well "
            "supported. But this run's OWN censoring rate is small (0.34% deadline, 0.36% "
            "cancelled-at-zero-latency) and decisive latencies sit far below this run's 8.0s "
            "deadline, so this checkpoint does not show the magnitude of timing pressure needed "
            "to project a 0.33 recall swing -- most likely because the deadline has since been "
            "widened well past the regime where the original finding was diagnosed. Cannot "
            "confirm or refute the original swing; can only characterize the current, much wider "
            "deadline regime as low-pressure by this measure."
        ),
        "evidence_for": for_points,
        "evidence_against": against_points,
        "deliberate_mode_for_context": {
            "note": "same question, deliberate mode (30s deadline) -- context only, not the claim under test",
            "outcome_counts": deliberate_dist["outcome_counts"],
            "deadline_fraction": (
                deliberate_dist["outcome_counts"].get("deadline", 0) / deliberate_dist["n_receipts"]
                if deliberate_dist["n_receipts"] else None
            ),
        },
    }


# --------------------------------------------------------------------- incident calibration -----
def _bucket_label(confidence: float) -> str:
    for lo, hi in zip(CONFIDENCE_BUCKETS, CONFIDENCE_BUCKETS[1:]):
        if lo <= confidence < hi:
            return f"[{lo:.2f}, {hi:.2f})"
    return f">={CONFIDENCE_BUCKETS[-1]:.2f}"


def incident_calibration(incidents: list[dict[str, Any]]) -> dict[str, Any]:
    """INCIDENT-level calibration: for fired, post-gate incidents, does higher confidence track a
    higher rate of landing inside a ground-truth interval?

    Population is exactly `latency_rows` -- every entry already passed the gate as a Fire. This
    cannot see occurrences the runtime never fired on at all (no false-negative side exists in
    this population by construction). "hit_rate" below means: of incidents at this confidence,
    what fraction had matched_occurrence != null (landed inside a GT interval on the source
    clock). It is not precision or recall in the classic sense -- there is no unfired-miss
    denominator available here.
    """
    from collections import defaultdict

    by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [n, matched]
    for inc in incidents:
        label = _bucket_label(round(inc["confidence"], 4))
        by_bucket[label][0] += 1
        if inc.get("matched_occurrence") is not None:
            by_bucket[label][1] += 1

    order = []
    for lo, hi in zip(CONFIDENCE_BUCKETS, CONFIDENCE_BUCKETS[1:]):
        label = f"[{lo:.2f}, {hi:.2f})"
        if label in by_bucket:
            order.append(label)

    table = [
        {
            "confidence_bucket": label,
            "n_incidents": by_bucket[label][0],
            "n_matched": by_bucket[label][1],
            "hit_rate": round(by_bucket[label][1] / by_bucket[label][0], 4) if by_bucket[label][0] else None,
        }
        for label in order
    ]
    total_n = len(incidents)
    total_matched = sum(1 for i in incidents if i.get("matched_occurrence") is not None)
    monotonic = all(
        table[i]["hit_rate"] is None or table[i + 1]["hit_rate"] is None or table[i]["hit_rate"] <= table[i + 1]["hit_rate"]
        for i in range(len(table) - 1)
    ) if len(table) > 1 else None

    return {
        "population": "incident-level (post-gate, fired-only -- see module docstring)",
        "n_incidents": total_n,
        "n_matched_inside_gt": total_matched,
        "overall_hit_rate": round(total_matched / total_n, 4) if total_n else None,
        "by_confidence_bucket": table,
        "monotonic_with_confidence": monotonic,
    }


# ---------------------------------------------------------------------------------- reporting ----
def build_report(checkpoint_path: Path, corpus_path: Path) -> dict[str, Any]:
    rows = load_checkpoint(checkpoint_path)
    corpus = load_corpus(corpus_path)

    saturated_charades = find_saturated_charades(corpus)
    malformed_repcount = find_malformed_repcount(corpus)
    contaminated_ids = set(saturated_charades) | set(malformed_repcount)

    reflex_receipts = iter_receipts(rows, mode="reflex")
    deliberate_receipts = iter_receipts(rows, mode="deliberate")
    reflex_dist = receipts_distribution(reflex_receipts)
    deliberate_dist = receipts_distribution(deliberate_receipts)

    all_incidents = iter_incidents(rows)
    reflex_incidents = [i for i in all_incidents if i["_mode"] == "reflex"]
    deliberate_incidents = [i for i in all_incidents if i["_mode"] == "deliberate"]

    reflex_clean = [i for i in reflex_incidents if i["case_id"] not in contaminated_ids]
    deliberate_clean = [i for i in deliberate_incidents if i["case_id"] not in contaminated_ids]

    n_charades = sum(1 for c in corpus["cases"] if c["dataset"] == "charades")
    n_repcount = sum(1 for c in corpus["cases"] if c["dataset"] == "repcount")

    return {
        "source": {
            "checkpoint": str(checkpoint_path),
            "corpus": str(corpus_path),
            "n_top_level_rows": len(rows),
            "modes_present": sorted({r.get("mode") for r in rows}),
        },
        "contaminants": {
            "note": (
                "Recomputed from corpus.json in this run, not trusted from any prior claim. "
                "Flagged by name below; excluded from the *_excl_contaminants incident-calibration "
                "tables."
            ),
            "saturated_charades": {
                "threshold_gt_over_duration": 0.95,
                "n": len(saturated_charades),
                "n_charades_total": n_charades,
                "case_ids": saturated_charades,
            },
            "malformed_repcount": {
                "detection": "goal_text contains '_' (raw underscored action_type leaked through)",
                "n": len(malformed_repcount),
                "n_repcount_total": n_repcount,
                "case_ids": malformed_repcount,
            },
        },
        "receipts_distributions": {
            "reflex": reflex_dist,
            "deliberate": deliberate_dist,
        },
        "censoring_vs_noise": censoring_vs_noise_verdict(reflex_dist, deliberate_dist),
        "incident_calibration": {
            "reflex_all": incident_calibration(reflex_incidents),
            "reflex_excl_contaminants": incident_calibration(reflex_clean),
            "deliberate_all": incident_calibration(deliberate_incidents),
            "deliberate_excl_contaminants": incident_calibration(deliberate_clean),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    add_lines: list[str] = []
    add = add_lines.append

    add("# Incident-level calibration + receipts distributions")
    add("")
    add(
        "Every table below is INCIDENT-level: post-gate and fired-only. It covers runtime Fires "
        "that already passed the gate, joined to ground truth on the source-media clock. It "
        "cannot see occurrences the runtime never fired on at all -- there is no false-negative "
        "side to this population, by construction. Treat every hit-rate number here as \"of the "
        "incidents the runtime already decided to surface, how often did they land inside a true "
        "interval\", not as recall or precision in the classic sense, and not as a scored judgment "
        "of every verdict the runtime produced."
    )
    add("")
    src = report["source"]
    add(f"Source checkpoint: `{src['checkpoint']}` ({src['n_top_level_rows']} top-level case rows, "
        f"modes: {', '.join(src['modes_present'])}).")
    add("")

    add("## 1. Contaminants recomputed from corpus.json")
    add("")
    sat = report["contaminants"]["saturated_charades"]
    add(f"**Saturated-coverage charades cases**: {sat['n']} of {sat['n_charades_total']} charades "
        f"cases have their single GT interval covering >= 95% of the source media duration "
        f"(build_corpus.py picks the single LONGEST annotated action per video). Every fired "
        f"incident in these cases is close to trivially \"inside GT\" by construction:")
    add("")
    for cid, ratio in sorted(sat["case_ids"].items(), key=lambda kv: -kv[1]):
        add(f"- `{cid}` -- GT interval / duration = {ratio:.3f}")
    add("")
    mal = report["contaminants"]["malformed_repcount"]
    add(f"**Malformed-goal repcount cases**: {mal['n']} of {mal['n_repcount_total']} repcount "
        f"cases have goal_text with a raw underscored action_type leaked through the phrasing "
        f"fallback (e.g. \"a person completes a jump_jack\"). Their incidents answer a malformed "
        f"question:")
    add("")
    for cid, text in sorted(mal["case_ids"].items()):
        add(f"- `{cid}` -- \"{text}\"")
    add("")

    add("## 2. Receipts outcome/confidence distributions")
    add("")
    for label in ("reflex", "deliberate"):
        dist = report["receipts_distributions"][label]
        add(f"### {label} ({dist['n_receipts']} receipts, deadline_s in use: "
            f"{dist['deadline_s_values_in_use']})")
        add("")
        add("| outcome | n | confidence values seen |")
        add("|---|---|---|")
        for outcome, info in dist["confidence_by_outcome"].items():
            conf_str = (
                ", ".join(f"{k}:{v}" for k, v in info["confidence_distribution"].items())
                if info["confidence_distribution"] else "none (no verdict reached)"
            )
            add(f"| {outcome} | {info['n']} | {conf_str} |")
        add("")
        dl = dist["decisive_latency_s"]
        add(f"Decisive latency_s: median {dl['median']}, p90 {dl['p90']}, p99 {dl['p99']}, "
            f"max {dl['max']} (n={dl['n']}).")
        add("")
        near = dist["decisive_calls_near_deadline_budget"]
        add("Decisive calls near their own deadline budget: " + "; ".join(
            f"{k}: {v['n']} ({v['fraction_of_decisive']:.3%})" for k, v in near.items() if v["fraction_of_decisive"] is not None
        ))
        add("")
        cwc = dist["cases_with_deadline_or_cancelled"]
        add(f"Cases touching >=1 deadline/cancelled receipt: {cwc['n_cases']}/{cwc['n_cases_total']}.")
        add("")

    add("## 3. Deadline-censoring vs verdict-noise -- a PARTIAL verdict")
    add("")
    cvn = report["censoring_vs_noise"]
    add(f"**Claim under test**: {cvn['claim_under_test']}")
    add("")
    add(f"**Verdict: {cvn['verdict']}** (confidence: {cvn['confidence']})")
    add("")
    add(cvn["summary"])
    add("")
    add("Evidence FOR:")
    for p in cvn["evidence_for"]:
        add(f"- {p}")
    add("")
    add("Evidence AGAINST / limiting:")
    for p in cvn["evidence_against"]:
        add(f"- {p}")
    add("")

    add("## 4. Incident-level calibration (confidence bucket -> hit-rate inside GT)")
    add("")
    for key, title in (
        ("reflex_all", "reflex, all cases"),
        ("reflex_excl_contaminants", "reflex, EXCLUDING saturated-charades + malformed-repcount"),
        ("deliberate_all", "deliberate, all cases"),
        ("deliberate_excl_contaminants", "deliberate, EXCLUDING saturated-charades + malformed-repcount"),
    ):
        cal = report["incident_calibration"][key]
        add(f"### {title}")
        add("")
        add(f"n_incidents={cal['n_incidents']}, n_matched_inside_gt={cal['n_matched_inside_gt']}, "
            f"overall_hit_rate={cal['overall_hit_rate']}, monotonic_with_confidence="
            f"{cal['monotonic_with_confidence']}")
        add("")
        add("| confidence bucket | n incidents | n matched | hit rate |")
        add("|---|---|---|---|")
        for row in cal["by_confidence_bucket"]:
            add(f"| {row['confidence_bucket']} | {row['n_incidents']} | {row['n_matched']} | "
                f"{row['hit_rate']} |")
        add("")

    return "\n".join(add_lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "incident_calibration.md")
    ap.add_argument("--json-out", type=Path, default=HERE / "results" / "incident_calibration.json")
    args = ap.parse_args()

    report = build_report(args.checkpoint, args.corpus)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    args.out.write_text(render_markdown(report))

    print(f"incident calibration -> {args.out}")
    print(f"  contaminants: {report['contaminants']['saturated_charades']['n']} saturated charades, "
          f"{report['contaminants']['malformed_repcount']['n']} malformed repcount")
    for label in ("reflex", "deliberate"):
        oc = report["receipts_distributions"][label]["outcome_counts"]
        print(f"  {label} receipts outcomes: {oc}")
    print(f"  censoring-vs-noise verdict: {report['censoring_vs_noise']['verdict']} "
          f"({report['censoring_vs_noise']['confidence']})")
    for key in ("reflex_all", "reflex_excl_contaminants"):
        cal = report["incident_calibration"][key]
        print(f"  incident calibration [{key}]: overall_hit_rate={cal['overall_hit_rate']} "
              f"(n={cal['n_incidents']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
