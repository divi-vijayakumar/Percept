"""Scoring: one-to-one occurrence matching, the metrics, and misses attributed to a stage.

Experimental: names and outputs may change before they are stable.

:func:`evaluate` scores what a run decided on labelled clips (:mod:`percept.plans.labels`); the
functions it is built from (:func:`match`, :func:`attribute`, :func:`survived`,
:func:`reachable`, :func:`bootstrap`) take any records with the attributes of
:class:`Prediction`, :class:`Look`, :class:`Drop` and :class:`Fire`. It reports what happened;
it does not compare or pick between runs.

Matching is by the frames judged. A prediction matches an occurrence of its condition when at
least one of its ``frames`` (the clip times of the frames shown to the judge; without frames, the
interval's two ends) lies in ``[start - tol, end + tol]`` and its last frame is no earlier than
``start - PRECOG_S`` (no credit for deciding before the onset could be seen). Predictions are
taken in the order they were decided; each takes the unmatched occurrence it shows that started
first. One that shows only already-matched occurrences is a duplicate; any other unmatched
prediction is a false alert. A matched occurrence counts toward onset recall when its prediction
had a frame within ``[start - tol, start + tol]``.

Metrics. Recall: matched occurrences over occurrences. Precision: matched predictions over all
predictions (duplicates count against it). False alerts per hour of footage and per 100
ground-truth occurrences. Latency: decision time minus onset, over matches only. The recall
ceiling: the share of occurrences a perfect judge could report given the cooldown (a repeat that
begins and ends within the cooldown of the previous report cannot be reported). A clip cut short
counts only the footage watched and the occurrences that began in it. 95% intervals come from a
clip-level bootstrap with a fixed seed, so they are reproducible.

Stages. A missed occurrence is attributed, from the looks at its condition that showed a frame
while it was on (``[start - tol, end + tol]``; a dropped look by the clip it would have judged),
to the first that applies: a look said yes (at ``MIN_CONFIDENCE`` or more) and a matched report
came less than the cooldown earlier (``cooldown_absorbed``); a look said yes but a verification
said no (``verify_rejected``); a look said yes but no prediction matched (``decide_suppressed``);
looks answered, none yes (``judge_said_no``); only failed looks (``judge_failed``); no answered
look and one was dropped or refused by the budget (``budget_dropped``); nothing looked
(``trigger_never_fired``).

Funnel. Every occurrence is followed through the stages a pipeline can lose it at, each counted
only if every earlier stage passed: a candidate (a look showed a frame in the window, a dropped
look's clip overlapped it, or a detector fired in it), admitted by the budget, evaluated by the
judge (answered), positive (a detect ``yes`` at ``MIN_CONFIDENCE`` or more), verified (only when
the run verifies, events only), delivered (matched). A stage a run doesn't have passes through.

Cost. API dollars are passed in per clip; local compute is reported separately, in seconds,
never converted to dollars.

Pure: no I/O, no clock.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Optional, Sequence

from percept.plans.labels import LabelError, Labels

PRECOG_S = 0.5
MIN_CONFIDENCE = 0.6        # a positive judgment
TOLERANCE_S = 2.0
BOOTSTRAP_N = 1000
FUNNEL = ("annotated", "candidate", "admitted", "evaluated", "positive", "verified", "delivered")
STAGES = ("trigger_never_fired", "budget_dropped", "judge_failed", "judge_said_no", "cooldown_absorbed",
          "verify_rejected", "decide_suppressed")
CI_METRICS = ("recall", "onset_recall", "precision", "false_per_hour", "false_per_100_occurrences",
              "cost_per_true_event", "latency_p50_s")


@dataclass(frozen=True)
class Prediction:
    """A decision: ``label`` of its condition, the evidence interval ``t0``-``t1`` and ``frames``
    (clip seconds), and ``t_decided`` (clip seconds)."""

    label: str
    t0: float
    t1: float
    t_decided: float
    confidence: float = 1.0
    frames: tuple[float, ...] = ()


@dataclass(frozen=True)
class Look:
    """A judge request for a condition: ``answer`` (``yes``, ``no``, ``unknown``), ``failed`` (the
    failure when it got no answer; ``budget`` when the budget refused it), ``purpose``
    (``detect`` or ``verify``)."""

    label: str
    t0: float
    t1: float
    answer: str = ""
    failed: Optional[str] = None
    purpose: str = "detect"
    confidence: float = 0.0
    frames: tuple[float, ...] = ()


class Drop(NamedTuple):
    """A look dropped unjudged (budget or deadline), with the clip it would have judged."""

    label: str
    t0: float
    t1: float
    reason: str = "budget"


class Fire(NamedTuple):
    """A detector gate opened for a condition at ``t``."""

    label: str
    t: float


# --------------------------------------------------------------------------- primitives


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 <= b1 and b0 <= a1


def frames_of(x: Any) -> tuple:
    """The clip times of the frames a prediction or look showed the judge."""
    return tuple(x.frames) if x.frames else (x.t0, x.t1)


def shows(x: Any, w0: float, w1: float) -> bool:
    """At least one judged frame lies in ``[w0, w1]``."""
    return any(w0 <= t <= w1 for t in frames_of(x))


def _yes(lk: Any) -> bool:
    return not lk.failed and lk.answer == "yes" and lk.confidence >= MIN_CONFIDENCE


@dataclass(frozen=True)
class Match:
    """``matched``: occurrence index -> the prediction that matched it, in matching order;
    ``onset``: the matched indices whose prediction saw the onset; ``dups``, ``false``: the
    duplicate and false predictions; ``missed``: the unmatched occurrences."""

    matched: Mapping[int, Any]
    onset: frozenset
    dups: tuple
    false: tuple
    missed: tuple


def match(preds: Sequence, occs: Sequence, tol: float) -> Match:
    """Match one condition's predictions to its occurrences on one clip."""
    matched: dict[int, Any] = {}
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
    onset = frozenset(i for i, p in matched.items() if shows(p, occs[i].start - tol, occs[i].start + tol))
    return Match(matched, onset, tuple(dups), tuple(false), tuple(o for i, o in enumerate(occs) if i not in matched))


def _dropped_near(drops: Sequence, w0: float, w1: float) -> bool:
    return any(_overlaps(d[1], d[2], w0, w1) for d in drops)


def attribute(occ: Any, looks: Sequence, drops: Sequence, tol: float, *, reports: tuple = (),
              cooldown: float = 0.0) -> str:
    """The stage a missed occurrence was lost at; ``reports`` are the matched predictions' last
    frame times; ``drops`` are ``(label, t0, t1, ...)`` tuples."""
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


def survived(occ: Any, looks: Sequence, drops: Sequence, fires: Sequence, tol: float, *, judged: bool,
             verifies: bool) -> int:
    """How many funnel stages after ``annotated`` the occurrence passed, up to ``verified``
    (0-5); a stage the run doesn't have passes through. ``fires`` are ``(label, t)`` pairs."""
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


def reachable(occs: Sequence, cooldown: float, tol: float) -> int:
    """How many occurrences a perfect judge could report, one report per cooldown."""
    n, ready = 0, -math.inf
    for o in sorted(occs, key=lambda o: o.start):
        at = max(o.start, ready)
        if at <= o.end + tol:
            n, ready = n + 1, at + cooldown
    return n


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _ratio(a: float, b: float) -> Optional[float]:
    return a / b if b else None


@dataclass(frozen=True)
class ConditionScore:
    """One condition on one clip: its :class:`Match`, latencies, recall ceiling, misses by stage
    and funnel counts (``outside``: matched although an earlier stage shows no pass)."""

    match: Match
    latencies: tuple[float, ...]
    ceiling: int
    stages: Mapping[str, int]
    funnel: tuple[int, ...]
    outside: int


def score_condition(occs: Sequence, preds: Sequence, looks: Sequence, drops: Sequence, fires: Sequence,
                    tol: float, *, cooldown: float = 0.0, judged: bool = True, verifies: bool = False
                    ) -> ConditionScore:
    """Score one condition's records on one clip."""
    m = match(preds, occs, tol)
    reports = tuple(max(frames_of(p)) for p in m.matched.values())
    stages = Counter(attribute(o, looks, drops, tol, reports=reports, cooldown=cooldown) for o in m.missed)
    funnel, outside = [0] * len(FUNNEL), 0
    for i, o in enumerate(occs):
        n = survived(o, looks, drops, fires, tol, judged=judged, verifies=verifies)
        for k in range(n + 1):
            funnel[k] += 1
        if i in m.matched:
            if n == len(FUNNEL) - 2:
                funnel[-1] += 1
            else:
                outside += 1
    return ConditionScore(m, tuple(p.t_decided - occs[i].start for i, p in m.matched.items()),
                          reachable(occs, cooldown, tol), dict(stages), tuple(funnel), outside)


def from_tallies(tallies: Sequence[Mapping]) -> dict:
    """Metrics from per-clip tallies (``occ``, ``tp``, ``onset``, ``preds``, ``false``,
    ``seconds``, ``cost``, ``latencies``); a clip may repeat."""
    t: Counter = Counter()
    lat: list = []
    for c in tallies:
        t.update({k: v for k, v in c.items() if k != "latencies"})
        lat += c["latencies"]
    hours = t["seconds"] / 3600.0
    return {"recall": _ratio(t["tp"], t["occ"]), "onset_recall": _ratio(t["onset"], t["occ"]),
            "precision": _ratio(t["tp"], t["preds"]), "false_per_hour": _ratio(t["false"], hours),
            "false_per_100_occurrences": _ratio(100 * t["false"], t["occ"]),
            "cost_per_true_event": _ratio(t["cost"], t["tp"]), "latency_p50_s": percentile(lat, 0.5)}


def bootstrap(tallies: Sequence[Mapping], seed: int = 0, n: int = BOOTSTRAP_N,
              metrics: Sequence[str] = CI_METRICS) -> dict:
    """95% clip-level bootstrap intervals, ``{metric: [low, high]}`` (None when undefined).
    Deterministic for a given ``seed``."""
    rng = random.Random(seed)
    draws: dict = {k: [] for k in metrics}
    if tallies:
        for _ in range(n):
            m = from_tallies([tallies[rng.randrange(len(tallies))] for _ in tallies])
            for k in metrics:
                if m[k] is not None:
                    draws[k].append(m[k])
    return {k: [percentile(v, 0.025), percentile(v, 0.975)] if v else None for k, v in draws.items()}


# --------------------------------------------------------------------------- evaluate


@dataclass(frozen=True)
class ClipRun:
    """What a run did on one labelled clip: ``seconds`` of it watched, its predictions, looks,
    drops and detector fires, the API dollars it spent, the local compute seconds it used, and
    the cooldown one report imposes on the next of its condition."""

    file: str
    seconds: float
    predictions: Sequence[Prediction] = ()
    looks: Sequence[Look] = ()
    drops: Sequence[Drop] = ()
    fires: Sequence[Fire] = ()
    cost_usd: float = 0.0
    compute_s: float = 0.0
    cooldown_s: float = 0.0


@dataclass(frozen=True)
class Score:
    """The metrics of one run over labelled clips. ``tp``: matched occurrences; ``fp``: false
    alerts (unmatched predictions that are not duplicates); ``fn``: missed occurrences.
    ``misses_by_stage`` sums to ``fn``. Ratios are None when their denominator is zero."""

    clips: int
    hours: float
    occurrences: int
    predictions: int
    tp: int
    fp: int
    fn: int
    duplicates: int
    onset_matched: int
    recall: Optional[float]
    onset_recall: Optional[float]
    recall_ceiling: Optional[float]
    precision: Optional[float]
    false_per_hour: Optional[float]
    false_per_100_occurrences: Optional[float]
    latency_p50_s: Optional[float]
    latency_p95_s: Optional[float]
    misses_by_stage: Mapping[str, int]
    funnel: Mapping[str, int]
    funnel_matched_outside: int
    funnel_not_applicable: tuple[str, ...]
    cost_usd: float
    cost_per_true_event: Optional[float]
    compute_s: float
    compute_s_per_true_event: Optional[float]
    ci95: Mapping[str, Any] = field(default_factory=dict)


def evaluate(labels: Labels, runs: Sequence[ClipRun], *, tol: float = TOLERANCE_S, seed: int = 0,
             n: int = BOOTSTRAP_N) -> Score:
    """Score one run (a :class:`ClipRun` per watched clip) against ``labels``. Only the clips in
    ``runs`` count; a clip's occurrences that begin after ``seconds`` don't."""
    judged = any(r.looks for r in runs)
    verifies = any(lk.purpose == "verify" for r in runs for lk in r.looks)
    seen: set = set()
    occ_n = tp = onset_n = dup_n = false_n = pred_n = ceiling = outside = 0
    seconds = cost = compute = 0.0
    latencies: list = []
    stages: Counter = Counter()
    funnel = [0] * len(FUNNEL)
    tallies = []
    for i, r in enumerate(runs):
        try:
            clip = labels.clip(r.file)
        except KeyError:
            raise LabelError(f"runs[{i}].file", f"{r.file!r} is not in the labels") from None
        if clip.file in seen:
            raise LabelError(f"runs[{i}].file", f"{r.file!r} is scored twice")
        seen.add(clip.file)
        seconds += r.seconds
        cost += r.cost_usd
        compute += r.compute_s
        tally: Counter = Counter(occ=0, tp=0, onset=0, preds=0, false=0, seconds=r.seconds, cost=r.cost_usd)
        tally_lat: list = []
        for cond in labels.conditions:
            occs = [o for o in clip.occurrences_of(cond.label) if o.start < r.seconds]
            preds = [p for p in r.predictions if p.label == cond.label]
            cs = score_condition(occs, preds, [lk for lk in r.looks if lk.label == cond.label],
                                 [d for d in r.drops if d[0] == cond.label], [f for f in r.fires if f[0] == cond.label],
                                 tol, cooldown=r.cooldown_s, judged=judged,
                                 verifies=verifies and cond.kind == "event")
            m = cs.match
            tally.update(occ=len(occs), tp=len(m.matched), onset=len(m.onset), preds=len(preds), false=len(m.false))
            tally_lat += cs.latencies
            occ_n, pred_n, tp = occ_n + len(occs), pred_n + len(preds), tp + len(m.matched)
            onset_n, dup_n, false_n = onset_n + len(m.onset), dup_n + len(m.dups), false_n + len(m.false)
            ceiling += cs.ceiling
            outside += cs.outside
            latencies += cs.latencies
            stages.update(cs.stages)
            funnel = [a + b for a, b in zip(funnel, cs.funnel)]
        tallies.append(dict(tally, latencies=tally_lat))
    hours = seconds / 3600.0
    return Score(
        clips=len(runs), hours=hours, occurrences=occ_n, predictions=pred_n, tp=tp, fp=false_n, fn=occ_n - tp,
        duplicates=dup_n, onset_matched=onset_n, recall=_ratio(tp, occ_n), onset_recall=_ratio(onset_n, occ_n),
        recall_ceiling=_ratio(ceiling, occ_n), precision=_ratio(tp, pred_n), false_per_hour=_ratio(false_n, hours),
        false_per_100_occurrences=_ratio(100 * false_n, occ_n),
        latency_p50_s=percentile(latencies, 0.5), latency_p95_s=percentile(latencies, 0.95),
        misses_by_stage={s: stages.get(s, 0) for s in STAGES}, funnel=dict(zip(FUNNEL, funnel)),
        funnel_matched_outside=outside,
        funnel_not_applicable=tuple(s for s, on in (("admitted", judged), ("evaluated", judged),
                                                    ("verified", verifies)) if not on),
        cost_usd=cost, cost_per_true_event=_ratio(cost, tp), compute_s=compute,
        compute_s_per_true_event=_ratio(compute, tp), ci95=bootstrap(tallies, seed, n))
