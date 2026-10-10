"""Scoring: one-to-one occurrence matching, the metrics, and misses attributed to a stage.

Matching. A decision's evidence interval ``[t0, t1]`` matches an occurrence of its condition
when it overlaps ``[start - tol, end + tol]`` and its evidence ends no earlier than
``start - 0.5`` s (no credit for deciding before the onset could be seen). Decisions are taken
in the order they were made; each takes the unmatched occurrence it overlaps that started first.
A decision that overlaps only already-matched occurrences is a duplicate; any other unmatched
decision is a false event.

Coverage. Only cases an arm actually watched count: a case skipped (after a spend-cap abort, or
because it failed) adds nothing to any denominator, and a case cut short counts only the footage
watched and the occurrences that began in it.

Metrics. Recall is matched occurrences over occurrences. Precision is matched decisions over
all decisions (duplicates count against it). False events per source-hour count every
unmatched, non-duplicate decision, absent-condition rows included, over the hours watched.
Latency is decision time minus onset, over matched occurrences only; misses are reported next to
it, never folded in. The recall ceiling is the share of occurrences a perfect judge could still
report given the arm's cooldown (Percept's ``rearm_s``): a repeat that begins and ends within
the cooldown of the previous report cannot be reported.

Stages. A missed occurrence is attributed, from the looks the arm made at the condition while
the occurrence was on (``[start - tol, end + tol]``), to the first that applies: a look said
yes and a matched report of the condition came less than the cooldown earlier
(``cooldown_absorbed``); a look said yes but a verification of it said no (``verify_rejected``);
a look said yes but no decision matched (``decide_suppressed``); looks answered, none yes
(``judge_said_no``); only failed looks (``judge_failed``); no answered look and the budget
dropped one (``budget_dropped``: approximate for Percept, whose trace counts drops per minute);
nothing looked (``trigger_never_fired``).
"""

from __future__ import annotations

import math
from collections import Counter

PRECOG_S = 0.5
STAGES = ("trigger_never_fired", "budget_dropped", "judge_failed", "judge_said_no", "cooldown_absorbed",
          "verify_rejected", "decide_suppressed")


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 <= b1 and b0 <= a1


def match(preds: list, occs: list, tol: float) -> dict:
    """``preds`` and ``occs`` for one condition on one case."""
    matched: dict[int, object] = {}         # occurrence index -> the decision that matched it
    dups, false = [], []
    for p in sorted(preds, key=lambda p: (p.t_decided, p.t1)):
        hits = [i for i, o in enumerate(occs)
                if _overlaps(p.t0, p.t1, o.start - tol, o.end + tol) and p.t1 >= o.start - PRECOG_S]
        free = [i for i in hits if i not in matched]
        if free:
            matched[min(free, key=lambda i: occs[i].start)] = p
        elif hits:
            dups.append(p)
        else:
            false.append(p)
    return {"matched": matched, "dups": dups, "false": false,
            "missed": [o for i, o in enumerate(occs) if i not in matched]}


def attribute(occ, looks: list, drops: list, tol: float, *, reports: tuple = (), cooldown: float = 0.0) -> str:
    """The stage a missed occurrence was lost at; ``reports`` are the matched decisions' times."""
    w0, w1 = occ.start - tol, occ.end + tol
    seen = [lk for lk in looks if _overlaps(lk.t0, lk.t1, w0, w1)]
    answered = [lk for lk in seen if not lk.failed]
    yes = [lk for lk in answered if lk.answer == "yes"]
    if yes:
        first = min(lk.t1 for lk in yes)
        if any(0 <= first - t < cooldown for t in reports):
            return "cooldown_absorbed"
        if any(lk.purpose == "verify" and lk.answer == "no" for lk in answered):
            return "verify_rejected"
        return "decide_suppressed"
    if answered:
        return "judge_said_no"
    if any(lk.failed and lk.failed != "budget" for lk in seen):
        return "judge_failed"
    if seen or any(_overlaps(t0, t1, w0, w1) for _, t0, t1 in drops):
        return "budget_dropped"
    return "trigger_never_fired"


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


def score(results: list, cases: dict, tol: float) -> dict:
    """Metrics for one arm from its per-case ``ArmResult`` list."""
    occ_n = tp = dup_n = false_n = pred_n = ceiling = 0
    latencies, stages, false_absent = [], Counter(), 0
    seconds = cost = 0.0
    unsupported, skipped, conditions, watched = [], [], 0, []
    for r in results:
        cost += r.cost_usd
        if r.error or not r.digest or r.seconds <= 0:
            skipped.append({"case_id": r.case_id, "reason": r.error or "not run: the spend cap was reached"})
            continue
        watched.append(r)
        case = cases[r.case_id]
        seconds += r.seconds
        for cond in case.conditions:
            if cond.id in r.unsupported:
                unsupported.append(f"{r.case_id}/{cond.id}")
                continue
            conditions += 1
            occs = [o for o in case.occurrences_of(cond.label) if o.start < r.seconds]
            preds = [p for p in r.preds if p.condition_id == cond.id]
            m = match(preds, occs, tol)
            occ_n += len(occs)
            pred_n += len(preds)
            tp += len(m["matched"])
            dup_n += len(m["dups"])
            false_n += len(m["false"])
            ceiling += reachable(occs, r.cooldown_s, tol)
            false_absent += 0 if occs else len(m["false"])
            latencies += [p.t_decided - occs[i].start for i, p in m["matched"].items()]
            looks = [lk for lk in r.looks if lk.condition_id == cond.id]
            drops = [d for d in r.budget_drops if d[0] == cond.id]
            reports = tuple(p.t1 for p in m["matched"].values())
            stages.update(attribute(o, looks, drops, tol, reports=reports, cooldown=r.cooldown_s)
                          for o in m["missed"])
    hours = seconds / 3600.0
    return {
        "cases": len(watched), "conditions": conditions, "source_hours": round(hours, 4),
        "occurrences": occ_n, "decisions": pred_n, "matched": tp, "missed": occ_n - tp,
        "duplicates": dup_n, "false_events": false_n, "false_on_absent_conditions": false_absent,
        "recall": tp / occ_n if occ_n else None,
        "recall_ceiling": ceiling / occ_n if occ_n else None,
        "precision": tp / pred_n if pred_n else None,
        "false_per_hour": false_n / hours if hours else None,
        "duplicates_per_occurrence": dup_n / tp if tp else None,
        "latency_p50_s": percentile(latencies, 0.5), "latency_p95_s": percentile(latencies, 0.95),
        "cost_usd": round(cost, 6), "cost_per_hour": cost / hours if hours else None,
        "cost_per_true_event": cost / tp if tp else None,
        "requests": sum(r.requests for r in results),
        "uncertain": sum(r.uncertain for r in results),
        "frames_capped": sum(r.frames_capped for r in results),
        "misses_by_stage": {s: stages.get(s, 0) for s in STAGES},
        "unsupported": unsupported, "skipped": skipped,
        "aborted": any(r.aborted for r in results),
    }
