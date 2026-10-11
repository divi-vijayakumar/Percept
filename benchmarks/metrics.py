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

import math
import random
from collections import Counter

PRECOG_S = 0.5
MIN_CONFIDENCE = 0.6        # a positive judgment, as Percept and the baselines apply it
MAX_FAILED_LOOK_RATE = 0.05  # more failed looks than this in any arm: the run is not publishable
BOOTSTRAP_N = 1000
FUNNEL = ("annotated", "candidate", "admitted", "evaluated", "positive", "verified", "delivered")
STAGES = ("trigger_never_fired", "budget_dropped", "judge_failed", "judge_said_no", "cooldown_absorbed",
          "verify_rejected", "decide_suppressed")
CI_METRICS = ("recall", "onset_recall", "precision", "false_per_hour", "cost_per_true_event", "latency_p50_s")


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 <= b1 and b0 <= a1


def frames_of(x) -> tuple:
    """The case times of the frames a decision or look showed the judge."""
    return tuple(x.frames) if x.frames else (x.t0, x.t1)


def shows(x, w0: float, w1: float) -> bool:
    """At least one judged frame lies in ``[w0, w1]``."""
    return any(w0 <= t <= w1 for t in frames_of(x))


def _yes(lk) -> bool:
    return not lk.failed and lk.answer == "yes" and lk.confidence >= MIN_CONFIDENCE


def match(preds: list, occs: list, tol: float) -> dict:
    """``preds`` and ``occs`` for one condition on one case."""
    matched: dict[int, object] = {}         # occurrence index -> the decision that matched it
    dups, false = [], []
    for p in sorted(preds, key=lambda p: (p.t_decided, p.t1)):
        last = max(frames_of(p))
        hits = [i for i, o in enumerate(occs) if shows(p, o.start - tol, o.end + tol) and last >= o.start - PRECOG_S]
        free = [i for i in hits if i not in matched]
        if free:
            matched[min(free, key=lambda i: occs[i].start)] = p
        elif hits:
            dups.append(p)
        else:
            false.append(p)
    onset = {i for i, p in matched.items() if shows(p, occs[i].start - tol, occs[i].start + tol)}
    return {"matched": matched, "onset": onset, "dups": dups, "false": false,
            "missed": [o for i, o in enumerate(occs) if i not in matched]}


def _dropped_near(drops: list, w0: float, w1: float) -> bool:
    return any(_overlaps(d[1], d[2], w0, w1) for d in drops)


def attribute(occ, looks: list, drops: list, tol: float, *, reports: tuple = (), cooldown: float = 0.0) -> str:
    """The stage a missed occurrence was lost at; ``reports`` are the matched decisions' times."""
    w0, w1 = occ.start - tol, occ.end + tol
    seen = [lk for lk in looks if shows(lk, w0, w1)]
    answered = [lk for lk in seen if not lk.failed]
    yes = [lk for lk in answered if _yes(lk)]
    if yes:
        first = min(max(frames_of(lk)) for lk in yes)
        if any(0 <= first - t < cooldown for t in reports):
            return "cooldown_absorbed"
        if any(lk.purpose == "verify" and lk.answer == "no" for lk in answered):
            return "verify_rejected"
        return "decide_suppressed"
    if answered:
        return "judge_said_no"
    if any(lk.failed and lk.failed != "budget" for lk in seen):
        return "judge_failed"
    if seen or _dropped_near(drops, w0, w1):
        return "budget_dropped"
    return "trigger_never_fired"


def survived(occ, looks: list, drops: list, fires: list, tol: float, *, judged: bool, verifies: bool) -> int:
    """How many funnel stages after ``annotated`` the occurrence passed, up to ``verified``
    (0-5); a stage the arm doesn't have passes through."""
    w0, w1 = occ.start - tol, occ.end + tol
    seen = [lk for lk in looks if shows(lk, w0, w1)]
    fired = any(w0 <= t <= w1 for _, t in fires)
    admitted = [lk for lk in seen if lk.failed != "budget"]     # a refused look never reached the judge
    answered = [lk for lk in seen if not lk.failed]
    positive = [lk for lk in answered if lk.purpose == "detect" and _yes(lk)]
    stages = (bool(seen) or fired or _dropped_near(drops, w0, w1),
              bool(admitted) if judged else True,
              bool(answered) if judged else True,
              bool(positive) if judged else fired,
              any(lk.purpose == "verify" and lk.answer == "yes" for lk in answered) if verifies else True)
    n = 0
    for ok in stages:
        if not ok:
            break
        n += 1
    return n


def reachable(occs: list, cooldown: float, tol: float) -> int:
    """How many occurrences a perfect judge could report, one report per cooldown."""
    n, ready = 0, -math.inf
    for o in sorted(occs, key=lambda o: o.start):
        at = max(o.start, ready)
        if at <= o.end + tol:
            n, ready = n + 1, at + cooldown
    return n


def percentile(values: list, q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def from_tallies(tallies: list) -> dict:
    """The bootstrapped metrics from a list of per-case tallies (a case may repeat)."""
    t = Counter()
    lat: list = []
    for c in tallies:
        t.update({k: v for k, v in c.items() if k != "latencies"})
        lat += c["latencies"]
    hours = t["seconds"] / 3600.0
    return {"recall": _ratio(t["tp"], t["occ"]), "onset_recall": _ratio(t["onset"], t["occ"]),
            "precision": _ratio(t["tp"], t["preds"]), "false_per_hour": _ratio(t["false"], hours),
            "cost_per_true_event": _ratio(t["cost"], t["tp"]), "latency_p50_s": percentile(lat, 0.5)}


def bootstrap(tallies: list, seed: int = 0, n: int = BOOTSTRAP_N) -> dict:
    """95% case-level bootstrap intervals, ``{metric: [low, high]}`` (None when undefined)."""
    rng = random.Random(seed)
    draws: dict = {k: [] for k in CI_METRICS}
    if tallies:
        for _ in range(n):
            m = from_tallies([tallies[rng.randrange(len(tallies))] for _ in tallies])
            for k in CI_METRICS:
                if m[k] is not None:
                    draws[k].append(m[k])
    return {k: [percentile(v, 0.025), percentile(v, 0.975)] if v else None for k, v in draws.items()}


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
        drop_reasons.update(d[3] if len(d) > 3 else "budget" for d in r.budget_drops)
        if not watched_ok(r):
            skipped.append({"case_id": r.case_id, "reason": r.error or "not run: the spend cap was reached"})
            continue
        watched.append(r)
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
            m = match(preds, occs, tol)
            lat = [p.t_decided - occs[i].start for i, p in m["matched"].items()]
            tally.update(occ=len(occs), tp=len(m["matched"]), onset=len(m["onset"]), preds=len(preds),
                         false=len(m["false"]))
            tally_lat += lat
            occ_n += len(occs)
            pred_n += len(preds)
            tp += len(m["matched"])
            onset_n += len(m["onset"])
            dup_n += len(m["dups"])
            false_n += len(m["false"])
            ceiling += reachable(occs, r.cooldown_s, tol)
            false_absent += 0 if occs else len(m["false"])
            latencies += lat
            looks = [lk for lk in r.looks if lk.condition_id == cond.id]
            drops = [d for d in r.budget_drops if d[0] == cond.id]
            reports = tuple(max(frames_of(p)) for p in m["matched"].values())
            stages.update(attribute(o, looks, drops, tol, reports=reports, cooldown=r.cooldown_s)
                          for o in m["missed"])
            fires = [f for f in r.fires if f[0] == cond.id]
            hit = set(m["matched"])
            for i, o in enumerate(occs):
                n = survived(o, looks, drops, fires, tol, judged=judged,
                             verifies=verifies and cond.kind == "event")     # Percept verifies events only
                for k in range(n + 1):
                    funnel[k] += 1
                if i in hit:
                    if n == len(FUNNEL) - 2:
                        funnel[-1] += 1
                    else:
                        outside += 1        # matched although an earlier stage shows no pass
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
