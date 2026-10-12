"""Scoring: one-to-one occurrence matching, the metrics, and misses attributed to a stage.

Matching is by the frames judged. Every decision and look carries ``frames``: the case times of
the frames actually shown to the judge (a detector-only decision: the frame it fired on). A
decision matches an occurrence of its condition when at least one of its frames lies in
``[start - tol, end + tol]`` and its last frame is no earlier than ``start - 0.5`` s (no credit
for deciding before the onset could be seen). The interval the frames span earns nothing by
itself: a 30-second chunk judged from 4 frames gets credit only if one of those frames lies in
the window. Decisions are taken in the order they were made; each takes the unmatched occurrence
it shows that started first. A decision that shows only already-matched occurrences is a
duplicate; any other unmatched decision is a false event. A record without frames is scored at
its interval's two ends, which are always judged frames.

Onset recall. Conditions say "<activity> begins". A matched occurrence counts toward onset
recall when its matching decision had a frame within ``[start - tol, start + tol]``: the judge
saw the beginning, not only the activity under way.

Coverage. Only cases an arm actually watched count: a case skipped (after a spend-cap abort, or
because it failed) adds nothing to any denominator, and a case cut short counts only the footage
watched and the occurrences that began in it. A condition the arm can't express is not scored
for it; comparisons between arms are then made on the case-conditions both score.

Metrics. Recall is matched occurrences over occurrences. Precision is matched decisions over
all decisions (duplicates count against it). False events per source-hour count every
unmatched, non-duplicate decision, absent-condition rows included, over the hours watched.
Latency is decision time minus onset, over matched occurrences only; misses are reported next to
it, never folded in. The recall ceiling is the share of occurrences a perfect judge could still
report given the arm's cooldown (Percept's ``rearm_s``): a repeat that begins and ends within
the cooldown of the previous report cannot be reported. Uncertainty: 95% intervals from a
case-level bootstrap (cases resampled with replacement, metrics recomputed from per-case
tallies, a fixed seed).

Judge failures. Every look is counted by its failure (``timeout``, ``malformed``, ``refused``,
``unavailable``, ...). ``failed_look_rate`` is failed looks over looks, both without
``budget`` (the harness spend cap refusing a request), which is reported separately. An
``unknown`` answer is an answer, not a failure: it counts as answered and not ``yes``, and its
share is reported as ``unknown_rate``.

Stages. A missed occurrence is attributed, from the looks the arm made at the condition that
showed a frame while the occurrence was on (``[start - tol, end + tol]``; a dropped look by the
clip it would have judged), to the first that applies: a look said yes (at ``MIN_CONFIDENCE`` or
more) and a matched report of the condition came less than the cooldown earlier
(``cooldown_absorbed``); a look said yes but a verification of it said no (``verify_rejected``);
a look said yes but no decision matched (``decide_suppressed``); looks answered, none yes
(``judge_said_no``); only failed looks (``judge_failed``); no answered look and the budget
dropped one, or one expired waiting for it (``budget_dropped``); nothing looked
(``trigger_never_fired``).

Funnel. Every annotated occurrence is followed through the stages a pipeline can lose it at,
each counted only if every earlier stage passed: a candidate was generated (a look showed a
frame in the window, a dropped look's clip overlapped it, or a detector gate opened in it), the
budget admitted a look, the judge evaluated one (answered, not failed), a detect answer was
positive (``yes`` at ``MIN_CONFIDENCE`` or more), it was verified (a verification said ``yes``;
only for arms that verify, and only events), and a decision delivered it (matched). A stage an
arm doesn't have passes through and is reported as not applicable.

Frames. ``intentional_sampling`` counts frames the configured sampling skipped (the harness's
``--fps`` sampling of the source, the Monitor's processing fps cap): expected, not a defect.
``unintended_loss`` counts frames lost because decoding, a queue or processing couldn't keep up
(decode gaps, frames replaced in the push handoff, frames pushed more than ``MAX_LAG_S`` behind
real time). They are reported separately; only unintended loss disqualifies a run.
"""

from __future__ import annotations

from collections import Counter

from percept.plans import score as _sdk
from percept.plans.score import (  # noqa: F401 - re-exported: one implementation, in the SDK
    BOOTSTRAP_N, FUNNEL, MIN_CONFIDENCE, PRECOG_S, STAGES, _ratio, attribute, frames_of, percentile, reachable,
    shows, survived)

MAX_FAILED_LOOK_RATE = 0.05  # more failed looks than this in any arm: the run is not publishable
CI_METRICS = ("recall", "onset_recall", "precision", "false_per_hour", "cost_per_true_event", "latency_p50_s")


def match(preds: list, occs: list, tol: float) -> dict:
    """``preds`` and ``occs`` for one condition on one case (:func:`percept.plans.score.match`)."""
    m = _sdk.match(preds, occs, tol)
    return {"matched": dict(m.matched), "onset": set(m.onset), "dups": list(m.dups), "false": list(m.false),
            "missed": list(m.missed)}


def from_tallies(tallies: list) -> dict:
    """The bootstrapped metrics from a list of per-case tallies (a case may repeat)."""
    m = _sdk.from_tallies(tallies)
    return {k: m[k] for k in CI_METRICS}


def bootstrap(tallies: list, seed: int = 0, n: int = BOOTSTRAP_N) -> dict:
    """95% case-level bootstrap intervals, ``{metric: [low, high]}`` (None when undefined)."""
    return _sdk.bootstrap(tallies, seed, n, metrics=CI_METRICS)


def looks_by_failure(results: list) -> dict:
    """Judge failures by type, over every look the arm made (watched or not)."""
    looks = [lk for r in results for lk in r.looks]
    by_type = Counter(lk.failed for lk in looks if lk.failed)
    refused = by_type.pop("budget", 0)
    judged = len(looks) - refused
    answered = [lk for lk in looks if not lk.failed]
    return {"looks_total": len(looks), "looks_refused_by_spend_cap": refused,
            "looks_failed_by_type": dict(sorted(by_type.items())),
            "failed_look_rate": _ratio(sum(by_type.values()), judged),
            "unknown_answers": sum(lk.answer == "unknown" for lk in answered),
            "unknown_rate": _ratio(sum(lk.answer == "unknown" for lk in answered), len(answered))}


def watched_ok(r) -> bool:
    return not (r.error or not r.digest or r.seconds <= 0)


def scored_pairs(results: list, cases: dict) -> set:
    """The (case id, condition id) pairs an arm is scored on."""
    return {(r.case_id, c.id) for r in results if watched_ok(r)
            for c in cases[r.case_id].conditions if c.id not in r.unsupported}


def score(results: list, cases: dict, tol: float, *, only: set | None = None, seed: int = 0) -> dict:
    """Metrics for one arm from its per-case ``ArmResult`` list; ``only`` restricts scoring to
    those (case id, condition id) pairs (and to the cases they name)."""
    occ_n = tp = onset_n = dup_n = false_n = pred_n = ceiling = 0
    latencies, stages, false_absent = [], Counter(), 0
    seconds = cost = 0.0
    unsupported, skipped, conditions, watched, tallies = [], [], 0, [], []
    sampling, loss = Counter(), Counter()
    funnel, outside = [0] * len(FUNNEL), 0
    judged = any(r.looks for r in results)
    verifies = any(lk.purpose == "verify" for r in results for lk in r.looks)
    drop_reasons = Counter()
    for r in results:
        if only is not None and not any(k[0] == r.case_id for k in only):
            continue
        cost += r.cost_usd
        sampling.update(r.intentional_sampling)
        loss.update(r.unintended_loss)
        if not watched_ok(r):
            skipped.append({"case_id": r.case_id, "reason": r.error or "not run: the spend cap was reached"})
            continue
        watched.append(r)
        drop_reasons.update(d[3] if len(d) > 3 else "budget" for d in r.budget_drops)
        case = cases[r.case_id]
        seconds += r.seconds
        tally = Counter(occ=0, tp=0, onset=0, preds=0, false=0, seconds=r.seconds, cost=r.cost_usd)
        tally_lat: list = []
        for cond in case.conditions:
            if cond.id in r.unsupported:
                unsupported.append(f"{r.case_id}/{cond.id}")
                continue
            if only is not None and (r.case_id, cond.id) not in only:
                continue
            conditions += 1
            occs = [o for o in case.occurrences_of(cond.label) if o.start < r.seconds]
            preds = [p for p in r.preds if p.condition_id == cond.id]
            looks = [lk for lk in r.looks if lk.condition_id == cond.id]
            drops = [d for d in r.budget_drops if d[0] == cond.id]
            fires = [f for f in r.fires if f[0] == cond.id]
            cs = _sdk.score_condition(occs, preds, looks, drops, fires, tol, cooldown=r.cooldown_s, judged=judged,
                                      verifies=verifies and cond.kind == "event")   # Percept verifies events only
            m = cs.match
            tally.update(occ=len(occs), tp=len(m.matched), onset=len(m.onset), preds=len(preds),
                         false=len(m.false))
            tally_lat += cs.latencies
            occ_n += len(occs)
            pred_n += len(preds)
            tp += len(m.matched)
            onset_n += len(m.onset)
            dup_n += len(m.dups)
            false_n += len(m.false)
            ceiling += cs.ceiling
            false_absent += 0 if occs else len(m.false)
            latencies += cs.latencies
            stages.update(cs.stages)
            funnel = [a + b for a, b in zip(funnel, cs.funnel)]
            outside += cs.outside
        tallies.append(dict(tally, latencies=tally_lat))
    hours = seconds / 3600.0
    return {
        "cases": len(watched), "cases_skipped": len(skipped), "conditions": conditions,
        "watched_s": round(seconds, 1), "source_hours": round(hours, 4),
        "occurrences": occ_n, "decisions": pred_n, "matched": tp, "missed": occ_n - tp,
        "onset_matched": onset_n,
        "duplicates": dup_n, "false_events": false_n, "false_on_absent_conditions": false_absent,
        "recall": tp / occ_n if occ_n else None,
        "onset_recall": onset_n / occ_n if occ_n else None,
        "recall_ceiling": ceiling / occ_n if occ_n else None,
        "precision": tp / pred_n if pred_n else None,
        "false_per_hour": false_n / hours if hours else None,
        "duplicates_per_occurrence": dup_n / tp if tp else None,
        "latency_p50_s": percentile(latencies, 0.5), "latency_p95_s": percentile(latencies, 0.95),
        "cost_usd": round(cost, 6), "cost_per_hour": cost / hours if hours else None,
        "cost_per_true_event": cost / tp if tp else None,
        "ci95": bootstrap(tallies, seed),
        "requests": sum(r.requests for r in results),
        "uncertain": sum(r.uncertain for r in results),
        "unverified": sum(r.unverified for r in results),
        "empty_clips": sum(r.empty_clips for r in results),
        "preds_without_yes_frames": sum(r.preds_without_yes_frames for r in results),
        "budget_drops": sum(r.budget_dropped for r in results),
        "drops_budget": drop_reasons.get("budget", 0), "drops_deadline": drop_reasons.get("deadline", 0),
        **looks_by_failure(results),
        "trigger_never_fired": stages.get("trigger_never_fired", 0),
        "verify_rejected": stages.get("verify_rejected", 0),
        "intentional_sampling": dict(sorted(sampling.items())),
        "unintended_loss": dict(sorted(loss.items())),
        "unintended_loss_total": sum(loss.values()),
        "misses_by_stage": {s: stages.get(s, 0) for s in STAGES},
        "funnel": dict(zip(FUNNEL, funnel)), "funnel_matched_outside": outside,
        "funnel_not_applicable": [s for s, on in (("admitted", judged), ("evaluated", judged),
                                                  ("verified", verifies)) if not on],
        "unsupported": unsupported, "skipped": skipped,
        "aborted": any(r.aborted for r in results),
    }
