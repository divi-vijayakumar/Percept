"""Per-STAGE recall / precision / economy / drop-rate for the Percept runtime.

Every number this repository publishes today is end-to-end.  When an e2e metric moves, nothing
says *which stage moved it*.  This module is the instrument that localises a regression: it drives
the REAL public path (``percept.run(...)`` with scripted, network-free judges) over the 100-case
validation corpus and reports the four metrics **at each stage boundary**.

The two-population rule
----------------------
Every stage reports recall TWICE:

``recall_stage_local``   versus what THAT STAGE actually received.
``recall_cumulative``    versus the ORIGINAL source ground truth.

Only the pair separates "this stage is lossy" from "this stage faithfully passed on an already-lossy
input".  A stage with ``recall_stage_local == 1.0`` and ``recall_cumulative == 0.4`` is innocent;
its input was already down to 0.4.  Reporting one without the other is what makes an e2e number
un-localisable, so this module refuses to emit one alone.

Metric definitions (fixed by internal-docs/PIPELINE_STAGE_VALIDATION_SPEC.md, not re-chosen here)
------------------------------------------------------------------------------------------------
recall      of the ground-truth-positive source TIME, the fraction the stage passed through.
precision   of what the stage passed through, the fraction overlapping ground-truth-positive time.
            Low precision at Stage 1 is NOT a defect -- admission headroom is intentional.
economy     ``units_passed / units_received``.  The codebase expresses the complement:
            ``LoopReport.reduction = 1 - vision_calls/frames`` (``percept/harness/tier0.py``),
            re-exported as ``fovea_call_reduction`` (``percept/slo.py``) with
            ``FOVEA_CALL_REDUCTION_FLOOR = 0.50``.
drop_rate   units lost WITHOUT a receipt, over units received.  A suppressed unit that carries a
            receipt is NOT a drop.  This distinction is the whole point of the metric: tier-0
            suppresses ~2/3 of all frames by design and that must never read as loss.

The measurement currency: unit-time tiling
------------------------------------------
Recall/precision are defined over *time*, but a pipeline moves *units*.  Each envelope at source
time ``t_i`` is treated as owning the half-open interval ``[t_i, t_{i+1})``; the last owns one median
sampling period.  The units therefore TILE the source clock exactly, so "the time a stage passed
through" is the union of the intervals owned by the units it passed.  No time is invented and none
is double-counted.  ``omnipro`` ground truth is zero-width instants, which have no time to recall;
those cases are widened by a DECLARED tolerance (``INSTANT_TOLERANCE_S``) and reported separately so
they never contaminate an interval-based bar.

Arms
----
``FILE``    ``run(core, source="<clip>.mp4")`` -- the real public file-replay front door.  This is
            the only arm that can measure Stage 0 honestly, because it is the only one where the
            runtime owns ingest.
``ORACLE``  ``run(core, source=[(t, gate_frame, payload), ...])`` replaying the FILE arm's own
            timestamps and BIT-IDENTICAL reproduced gate frames, with a scripted judge keyed on the
            payload hash that answers from ground truth.  This isolates Stages 2-4 plumbing from
            judge *quality*, which is explicitly out of scope.

The substitution is not assumed -- it is VERIFIED per case: the two arms must produce the identical
set of admitted timestamps (``stage1_substitution_ok``).  A case where they diverge is reported as
such rather than averaged in.

Salience reproduction
---------------------
The ``edge.decision`` trace carries the whole-frame ``mean`` statistic for every frame plus the
``escalate`` bit, but not the ``block_max`` value the shipped gate actually compares.  Gate frames
are therefore reproduced (``cv2`` grey + resize to 48x36, exactly ``percept/runtime/source.py``) and
scored with the WHEEL'S OWN ``percept.harness.tier0.salience_stats`` -- never a reimplementation.
Two identities are then asserted per case and recorded:

    trace.motion == salience_stats(...)["mean"]      (the trace's stat is the mean)
    escalate     == (block_max >= motion_gate_block_max)   (the shipped gate's actual rule)

Only cases where both hold contribute to the threshold sweep.

Run it::

    /private/tmp/percept-launch/oss-final-venv-083b972/bin/python eval/stage-validation/stage_metrics.py

Writes ``results/stage_metrics.json`` and ``results/stage_metrics.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"

# --------------------------------------------------------------------------------------------
# pinned constants -- every one is either cited from the runtime or declared here BEFORE the run
# --------------------------------------------------------------------------------------------

#: Zero-width ground-truth instants (omnipro `question_time_sec`) carry no time to recall. They are
#: widened by this DECLARED tolerance. Not fitted -- it matches the tolerance already declared for
#: the same annotation in eval/stage-validation/pins/PIN_PROPOSALS.md so the two agree.
INSTANT_TOLERANCE_S = 1.0

#: Tier-0 gate frame geometry -- percept/runtime/source.py:26 (GATE_W, GATE_H).
GATE_W, GATE_H = 48, 36

#: The economy floor the architecture already pins: percept/slo.py:243.
FOVEA_CALL_REDUCTION_FLOOR = 0.50

#: Deterministic payload seed, so the ORACLE arm's judge script reproduces across checkouts.
PAYLOAD_SEED = 20260726

STAGES = ("stage0_ingest", "stage1_admission", "stage2_perception", "stage3_judge", "stage4_cognition")
STAGE_TITLES = {
    "stage0_ingest": "Stage 0 - Ingest + normalize",
    "stage1_admission": "Stage 1 - Admission (VID tier-0 gate)",
    "stage2_perception": "Stage 2 - Local perception",
    "stage3_judge": "Stage 3 - Judge dispatch",
    "stage4_cognition": "Stage 4 - Cognition + disposition",
}

# --------------------------------------------------------------------------------------------
# runtime imports -- from the WHEEL UNDER TEST, never from a source tree on sys.path
# --------------------------------------------------------------------------------------------

import cv2  # noqa: E402

from percept import Goal, Percept  # noqa: E402
from percept._build import build_info  # noqa: E402
from percept.adapters import Verdict  # noqa: E402
from percept.config import load_settings  # noqa: E402
from percept.harness.tier0 import Frame, salience_stats  # noqa: E402
from percept.runtime import RuntimeStats, run  # noqa: E402
from percept.slo import FOVEA_CALL_REDUCTION_FLOOR as _SLO_FLOOR  # noqa: E402

try:
    from percept.eval.exact_ci import clopper_pearson_lower, required_positives
except Exception:  # pragma: no cover - the CI helper is optional to the measurement
    clopper_pearson_lower = None  # type: ignore[assignment]
    required_positives = None  # type: ignore[assignment]

assert _SLO_FLOOR == FOVEA_CALL_REDUCTION_FLOOR, "slo floor drifted from the value pinned here"

YES = Verdict(satisfied=True, confidence=0.95, evidence="scripted: ground truth positive")
NO = Verdict(satisfied=False, confidence=0.95, evidence="scripted: ground truth negative")


# --------------------------------------------------------------------------------------------
# interval algebra -- exact, on the source clock
# --------------------------------------------------------------------------------------------


def union(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/touching spans into a canonical disjoint ascending list."""
    xs = sorted((a, b) for a, b in spans if b > a)
    if not xs:
        return []
    out = [list(xs[0])]
    for a, b in xs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def measure(spans: Sequence[tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in spans))


def intersect(x: Sequence[tuple[float, float]], y: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intersection of two canonical disjoint span lists."""
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(x) and j < len(y):
        a = max(x[i][0], y[j][0])
        b = min(x[i][1], y[j][1])
        if b > a:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(spans: Iterable[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return union((max(a, lo), min(b, hi)) for a, b in spans)


# --------------------------------------------------------------------------------------------
# the scripted, network-free judge
# --------------------------------------------------------------------------------------------


class OracleVision:
    """A scripted judge keyed on the SHA-256 of the payload it is handed.

    Wiring, never intelligence: it looks a hash up in a dict built from ground truth.  Using ground
    truth as the judge is deliberate -- Stage 3/4 here measure the PLUMBING (does a correct answer
    reach an Incident?), and semantic-judge quality is explicitly out of scope.  It records every
    physical call so Stage 3's logical-vs-physical split is measured at the adapter boundary rather
    than inferred.
    """

    PARSES_CLAIM_CHANNEL = True

    def __init__(self, script: dict[str, Verdict] | None = None) -> None:
        self.script = dict(script or {})
        self.n_calls = 0
        self.unlisted = 0
        self.conditions: Counter = Counter()

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        self.n_calls += 1
        self.conditions[str(condition)[:80]] += 1
        raw = frame if isinstance(frame, bytes) else repr(frame).encode()
        verdict = self.script.get(hashlib.sha256(raw).hexdigest())
        if verdict is None:
            self.unlisted += 1
            return NO
        return verdict

    async def detect(self, frame, goals):  # pragma: no cover - the corpus arms no detector
        return []


# --------------------------------------------------------------------------------------------
# driving the real public path
# --------------------------------------------------------------------------------------------


@dataclass
class RunChannels:
    """Every observable channel of one ``percept.run`` call."""

    stats: RuntimeStats
    trace: list
    health: list
    incidents: list
    funnel: dict
    wall_s: float


def drive(core: Percept, source: Any, **kw) -> RunChannels:
    stats = RuntimeStats()
    trace: list = []
    health: list = []
    incidents: list = []

    async def go():
        async for inc in run(core, source=source, stats=stats, on_trace=trace.append,
                             on_health=health.append, sink_errors="ignore", **kw):
            incidents.append(inc)

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # the scripted-backend banner is not our signal
        asyncio.run(go())
    return RunChannels(stats, trace, health, incidents, dict(core.metrics.snapshot()), time.time() - t0)


def make_core(vision: Any, goal_text: str, settings: Any = None) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False, settings=settings)
    core.add_goal(Goal(id="g", condition=goal_text, say="condition met"))
    return core


def settings_for(stat: str, threshold: float) -> Any:
    """A Settings with the tier-0 salience statistic and ITS OWN cutoff replaced.

    block_max/p95 sit on a higher value scale than the whole-frame mean, so each stat carries its own
    field (``percept/config.py``).  Writing the wrong one is the classic way to produce a comparison
    that is really a threshold change, so the field is selected by stat here rather than by the
    caller.
    """
    cfg = load_settings()
    field_by_stat = {
        "mean": "motion_gate",
        "block_max": "motion_gate_block_max",
        "p95": "motion_gate_p95",
    }
    return dataclasses.replace(
        cfg, edge=dataclasses.replace(cfg.edge, salience_stat=stat, **{field_by_stat[stat]: threshold})
    )


# --------------------------------------------------------------------------------------------
# corpus + ground truth
# --------------------------------------------------------------------------------------------


def probe_media(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def corpus_path() -> Path:
    """Which corpus revision this process scores against.

    Defaults to the tracked supported revision, ``corpus.v1.2.json``. Historical revisions are
    intentionally not required by a fresh clone; a historical artifact that needs it must select
    its archived bytes explicitly through ``PERCEPT_SV_CORPUS``.  The selected revision is stamped
    into every artifact's ``meta.harness.corpus_version`` so no table can mix revisions unlabelled.
    """
    override = os.environ.get("PERCEPT_SV_CORPUS", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (HERE / p)
    return HERE / "corpus.v1.2.json"


def load_corpus() -> list[dict]:
    cases = json.loads(corpus_path().read_text())["cases"]
    for c in cases:
        c["video_path"] = str(REPO / c["video_path"]) if not os.path.isabs(c["video_path"]) else c["video_path"]
    return cases


def ground_truth(case_positives: Sequence[Sequence[float]], media_end: float
                 ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], str, int]:
    """Positive TIME spans and positive EVENT intervals on the source clock.

    These are deliberately two different objects and must not be conflated:

    ``spans``   the UNION, used for every time-based metric.  Overlapping or touching annotations
                describe the same second of source once, so unioning is the only way not to
                double-count time.
    ``events``  the INDIVIDUAL annotated intervals, used for event recall.  ``repcount`` annotates
                back-to-back repetitions whose bounds touch exactly (``[2.586,4.046]``,
                ``[4.046,5.172]``, ...); unioning collapses twelve repetitions into one span and
                would silently under-count events by an order of magnitude on the whole ``count``
                goal_kind.  Event recall therefore keeps them apart.

    ``omnipro`` positives are zero-width instants (``question_time_sec``): they measure edge TIMING,
    not interval recall, so they are widened by the declared ``INSTANT_TOLERANCE_S`` and tagged
    ``instant`` so every consumer can keep them out of an interval bar.
    """
    raw = [(float(a), float(b)) for a, b in case_positives]
    occurrences = len(raw)
    kind = "interval"
    if raw and all(b - a <= 0 for a, b in raw):
        raw = [(a - INSTANT_TOLERANCE_S, b + INSTANT_TOLERANCE_S) for a, b in raw]
        kind = "instant"
    events = [(max(a, 0.0), min(b, media_end)) for a, b in raw if min(b, media_end) > max(a, 0.0)]
    return clip(raw, 0.0, media_end), events, kind, occurrences


# --------------------------------------------------------------------------------------------
# unit tiling
# --------------------------------------------------------------------------------------------


def tile(ts: Sequence[float], media_end: float) -> list[tuple[float, float]]:
    """Each unit owns ``[t_i, t_{i+1})``; the last owns one median period. The units tile exactly."""
    if not ts:
        return []
    if len(ts) == 1:
        return [(ts[0], min(media_end, ts[0] + 0.25))]
    deltas = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    tail = statistics.median(deltas) if deltas else 0.25
    spans = [(ts[i], ts[i + 1]) for i in range(len(ts) - 1)]
    spans.append((ts[-1], min(media_end, ts[-1] + tail)))
    return spans


def spans_of(tiles: Sequence[tuple[float, float]], keep: Sequence[bool]) -> list[tuple[float, float]]:
    return union(t for t, k in zip(tiles, keep) if k)


# --------------------------------------------------------------------------------------------
# per-stage metric assembly
# --------------------------------------------------------------------------------------------


def touched(spans: Sequence[tuple[float, float]], gt: Sequence[tuple[float, float]]) -> int:
    """How many ground-truth intervals the given span set overlaps at all."""
    return sum(1 for g in gt if intersect(spans, [g]))


def stage_metrics(
    *,
    passed_time: Sequence[tuple[float, float]],
    received_time: Sequence[tuple[float, float]],
    gt: Sequence[tuple[float, float]],
    gt_events: Sequence[tuple[float, float]] | None = None,
    units_received: int,
    units_passed: int,
    units_dropped_no_receipt: int,
) -> dict[str, Any]:
    """The metrics at one boundary, with recall reported against BOTH populations.

    Recall is computed in TWO families, because on a sampled pipeline they answer different
    questions and only reporting both is honest:

    ``time``      the spec's literal definition -- the fraction of ground-truth-positive SOURCE TIME
                  the stage passed through.  Note this is structurally bounded above by the stage's
                  own economy: a gate that admits 18% of units cannot cover more than 18% of the
                  clock, so at Stage 1 this metric largely restates economy.
    ``interval``  the fraction of ground-truth OCCURRENCES the stage passed at least one unit
                  inside.  This is the metric that localises a regression, because it answers "did
                  the event get through at all" rather than "how densely was it sampled", and it is
                  the definition the pre-registered tier-0 numbers in ``percept/config.py`` use.
    """
    events = list(gt_events) if gt_events is not None else list(gt)
    gt_time = measure(gt)
    recv_pos = measure(intersect(received_time, gt))
    pass_pos = measure(intersect(passed_time, gt))
    pass_time = measure(passed_time)
    gt_n = len(events)
    gt_recv = touched(received_time, events)
    gt_pass = touched(passed_time, events)
    return {
        # vs what THIS stage received -- "is this stage lossy?"
        "recall_stage_local": (pass_pos / recv_pos) if recv_pos > 0 else None,
        # vs the ORIGINAL source ground truth -- "how much of the event survives to here?"
        "recall_cumulative": (pass_pos / gt_time) if gt_time > 0 else None,
        "recall_interval_stage_local": (gt_pass / gt_recv) if gt_recv else None,
        "recall_interval_cumulative": (gt_pass / gt_n) if gt_n else None,
        "precision": (pass_pos / pass_time) if pass_time > 0 else None,
        "economy": (units_passed / units_received) if units_received else None,
        "drop_rate": (units_dropped_no_receipt / units_received) if units_received else None,
        "units_received": units_received,
        "units_passed": units_passed,
        "positive_time_received_s": recv_pos,
        "positive_time_passed_s": pass_pos,
        "passed_time_s": pass_time,
        "gt_time_s": gt_time,
        "gt_intervals": gt_n,
        "gt_intervals_received": gt_recv,
        "gt_intervals_passed": gt_pass,
    }


# --------------------------------------------------------------------------------------------
# the per-case measurement
# --------------------------------------------------------------------------------------------


def reproduce_gates(path: str, step: int) -> list[Frame]:
    """Reproduce the runtime's own tier-0 gate frames, exactly as percept/runtime/source.py builds
    them: BGR -> COLOR_BGR2GRAY -> resize to (48, 36) -> Frame of int pixels."""
    cap = cv2.VideoCapture(path)
    gates: list[Frame] = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (GATE_W, GATE_H))
            gates.append(Frame(w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in g.flatten())))
        i += 1
    cap.release()
    return gates


def edge_rows(ch: RunChannels) -> list:
    return [e for e in ch.trace if e.stage == "edge.decision"]


def health_data(ch: RunChannels, kind: str) -> dict:
    for h in ch.health:
        if h.kind == kind:
            return dict(h.data or {})
    return {}


def measure_case(case: dict, *, shipped_stat: str, shipped_threshold: float) -> dict[str, Any]:
    """One corpus case, all five stages, both arms."""
    path = case["video_path"]
    src_fps, n_native = probe_media(path)
    out: dict[str, Any] = {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "goal_kind": case["goal_kind"],
        "native_fps": src_fps,
        "native_frames": n_native,
        "notes": [],
    }

    # ---- ARM FILE (shipped config): the real public file-replay front door ----------------
    ch = drive(make_core(OracleVision({}), case["goal_text"]), path)
    ed = edge_rows(ch)
    if not ed:
        out["notes"].append("no edge.decision trace -- case not measurable")
        out["measurable"] = False
        return out
    out["measurable"] = True
    ts = [float(e.t) for e in ed]
    esc = [bool(e.data.get("escalate") or e.data.get("coast_admit")) for e in ed]
    mean_vals = [float(e.data.get("motion", 0.0)) for e in ed]
    thresholds = {float(e.data.get("threshold", 0.0)) for e in ed}

    eff = health_data(ch, "effective_config").get("fps", {})
    ended = health_data(ch, "runtime_ended")
    cov = (ended.get("source_coverage") or {}).get("video", {})

    media_end = max(ts[-1], float(case.get("duration_s") or 0.0), (n_native / src_fps) if src_fps else 0.0)
    gt, gt_events, gt_kind, gt_occurrences = ground_truth(case["positives"], media_end)
    out.update(gt_kind=gt_kind, gt_occurrences=gt_occurrences, gt_events=len(gt_events),
               gt_time_s=measure(gt),
               media_end_s=media_end, effective_fps=eff.get("effective_fps"),
               requested_fps=eff.get("requested_fps"), fps_reason=eff.get("reason"),
               fps_overridden=eff.get("overridden"), gate_threshold=sorted(thresholds))

    tiles = tile(ts, media_end)
    received_all = union(tiles)
    admitted_time = spans_of(tiles, esc)

    # ---- STAGE 0 -- ingest ---------------------------------------------------------------
    # The runtime DELIBERATELY decimates: step = max(1, round(src_fps / effective_fps)). That is
    # designed subsampling announced through effective_config, not loss, so it is reported as
    # economy and never as drop. Drop is what the bounded capture queue discarded.
    step = max(1, round(n_native / len(ed))) if len(ed) else 1
    step_from_fps = max(1, round(src_fps / float(eff.get("effective_fps") or src_fps or 1.0))) if src_fps else 1
    if step != step_from_fps:
        out["notes"].append(f"decimation step {step} (observed) != {step_from_fps} (from effective_fps)")
    selected = math.ceil(n_native / step) if step else n_native
    emitted = int(ch.stats.frames)
    frames_dropped = int(ch.stats.frames_dropped)          # carries a receipt (counter + health)
    excluded = int(cov.get("excluded", 0))                  # carries a receipt (typed exclusion)
    missing = int(cov.get("missing_sequences", 0))
    unaccounted0 = max(0, selected - emitted - frames_dropped - excluded)
    s0 = stage_metrics(
        passed_time=received_all, received_time=[(0.0, media_end)], gt=gt, gt_events=gt_events,
        units_received=n_native, units_passed=emitted, units_dropped_no_receipt=unaccounted0,
    )
    s0.update(deliberate_decimation_step=step, units_selected_after_decimation=selected,
              frames_dropped_with_receipt=frames_dropped, envelopes_excluded_with_receipt=excluded,
              missing_sequences=missing, discontinuities=int(cov.get("discontinuities", 0)),
              source_observed=int(cov.get("observed", 0)))

    # ---- STAGE 1 -- admission -------------------------------------------------------------
    # Suppression is NOT a drop: every suppressed frame emits its own edge.decision record naming
    # the signal and the bar it failed to clear. That IS the receipt, so drop_rate is 0 here and the
    # loss shows up -- correctly -- as economy.
    n_adm = sum(esc)
    s1 = stage_metrics(
        passed_time=admitted_time, received_time=received_all, gt=gt, gt_events=gt_events,
        units_received=len(ed), units_passed=n_adm, units_dropped_no_receipt=0,
    )
    s1.update(reduction=1.0 - (n_adm / len(ed)) if ed else None,
              admission_by_cause=dict(ch.stats.admission_by_cause),
              escalations=int(ch.stats.escalations), signal_admits=int(ch.stats.signal_admits),
              coast_admits=int(ch.stats.coast_admits), fixed_admits=int(ch.stats.fixed_admits),
              suppressed_with_receipt=len(ed) - n_adm)

    # ---- salience reproduction (for the threshold sweep only) -----------------------------
    gates = reproduce_gates(path, step)
    n = min(len(gates), len(ed))
    bm: list[float] = []
    mn: list[float] = []
    prev = None
    for j in range(n):
        st = salience_stats(prev, gates[j])
        bm.append(st["block_max"])
        mn.append(st["mean"])
        prev = gates[j]
    mean_ok = sum(1 for j in range(n) if abs(mn[j] - mean_vals[j]) < 5e-5)
    gate_ok = sum(1 for j in range(n) if esc[j] == (bm[j] >= shipped_threshold))
    repro_ok = n == len(ed) and mean_ok == n and gate_ok == n
    out.update(salience_n=n, salience_mean_identity=mean_ok, salience_gate_identity=gate_ok,
               salience_reproduction_ok=repro_ok)
    if not repro_ok:
        out["notes"].append(
            f"salience reproduction imperfect (n={n}/{len(ed)} mean_id={mean_ok} gate_id={gate_ok});"
            " case excluded from the threshold sweep, runtime-measured numbers unaffected"
        )

    # ---- ARM ORACLE -- replay the SAME timestamps + gate frames, judge from ground truth ----
    sub_ok = None
    s2 = s3 = s4 = None
    oracle_info: dict[str, Any] = {}
    if n == len(ed):
        rnd = random.Random(PAYLOAD_SEED)
        payloads = [rnd.randbytes(256) for _ in range(n)]
        in_gt = [any(a <= ts[j] < b for a, b in gt) for j in range(n)]
        script = {hashlib.sha256(payloads[j]).hexdigest(): (YES if in_gt[j] else NO) for j in range(n)}
        vision = OracleVision(script)
        ch2 = drive(make_core(vision, case["goal_text"]),
                    [(ts[j], gates[j], payloads[j]) for j in range(n)])
        ed2 = edge_rows(ch2)
        a1 = {round(t, 6) for t, e in zip(ts, esc) if e}
        a2 = {round(float(e.t), 6) for e in ed2 if e.data.get("escalate") or e.data.get("coast_admit")}
        sub_ok = a1 == a2
        if not sub_ok:
            out["notes"].append(
                f"stage-1 substitution mismatch: FILE admitted {len(a1)}, ORACLE {len(a2)} "
                f"(jaccard {len(a1 & a2) / max(1, len(a1 | a2)):.4f}) -- stages 2-4 reported but flagged"
            )

        def stage_times(stage: str) -> tuple[list[float], list[tuple[float, float]]]:
            got = sorted({float(e.t) for e in ch2.trace if e.stage == stage})
            index = {round(t, 6): i for i, t in enumerate(ts)}
            keep = [False] * len(ts)
            for t in got:
                i = index.get(round(t, 6))
                if i is not None:
                    keep[i] = True
            return got, spans_of(tiles, keep)

        _, s2_time = stage_times("source.frame")
        _, s3_time = stage_times("judge.finished")
        gate_true = sorted({float(e.t) for e in ch2.trace
                            if e.stage == "gate.classified" and e.data.get("satisfied")})
        index = {round(t, 6): i for i, t in enumerate(ts)}
        keep4 = [False] * len(ts)
        for t in gate_true:
            i = index.get(round(t, 6))
            if i is not None:
                keep4[i] = True
        s4_time = spans_of(tiles, keep4)

        n_s2 = len({round(float(e.t), 6) for e in ch2.trace if e.stage == "source.frame"})
        n_started = sum(1 for e in ch2.trace if e.stage == "judge.started")
        n_finished = sum(1 for e in ch2.trace if e.stage == "judge.finished")
        n_degraded = sum(1 for e in ch2.trace if e.stage == "judge.degraded")
        n_gate = sum(1 for e in ch2.trace if e.stage == "gate.classified")
        n_true = len(gate_true)
        st2 = ch2.stats
        judge_receipted = (int(st2.judge_superseded) + int(st2.judge_revision_dropped)
                           + int(st2.judge_deadline_dropped) + int(st2.judge_deadline_expired)
                           + int(st2.judge_source_discontinuity_dropped))

        n_adm2 = int(st2.escalations)
        # Stage 2: with no EvidenceExtractor armed this boundary is an identity pass-through.
        s2 = stage_metrics(passed_time=s2_time, received_time=admitted_time, gt=gt, gt_events=gt_events,
                           units_received=n_adm2, units_passed=int(st2.perceive_dispatches),
                           units_dropped_no_receipt=max(0, n_adm2 - int(st2.perceive_dispatches)))
        s2.update(source_frame_units=n_s2, extractor_enriched=sum(
            1 for e in ch2.trace if e.stage == "extractor.enriched"),
            identity_passthrough=int(st2.perceive_dispatches) == n_adm2)

        s3 = stage_metrics(passed_time=s3_time, received_time=s2_time, gt=gt, gt_events=gt_events,
                           units_received=int(st2.perceive_dispatches), units_passed=n_finished,
                           units_dropped_no_receipt=max(
                               0, int(st2.perceive_dispatches) - n_finished - judge_receipted))
        s3.update(logical_calls=n_started, physical_attempts=vision.n_calls,
                  judge_calls_batches=int(st2.judge_calls), finished=n_finished,
                  degraded=n_degraded, unlisted_payloads=vision.unlisted,
                  dropped_with_receipt=judge_receipted,
                  terminal_outcomes={"finished": n_finished, "degraded": n_degraded,
                                     "superseded": int(st2.judge_superseded),
                                     "revision_dropped": int(st2.judge_revision_dropped),
                                     "deadline_dropped": int(st2.judge_deadline_dropped),
                                     "deadline_expired": int(st2.judge_deadline_expired),
                                     "discontinuity_dropped": int(st2.judge_source_discontinuity_dropped)})

        s4 = stage_metrics(passed_time=s4_time, received_time=s3_time, gt=gt, gt_events=gt_events,
                           units_received=n_finished, units_passed=n_true,
                           units_dropped_no_receipt=max(0, n_finished - n_gate))
        inc_ts = [float(getattr(i, "ts", 0.0)) for i in ch2.incidents]
        # Occurrence matching uses the INDIVIDUAL annotations, not the union: a rising-edge gate
        # cannot re-fire between two repetitions that touch, and collapsing them would hide that.
        matched = sum(1 for a, b in gt_events if any(a <= t < b for t in inc_ts))
        s4.update(gate_classified=n_gate, gate_true_units=n_true,
                  incidents=len(ch2.incidents), incident_ts=inc_ts,
                  gt_occurrences=len(gt_events), occurrences_matched=matched,
                  occurrence_recall=(matched / len(gt_events)) if gt_events else None,
                  contiguous_events_unresolvable=len(gt_events) - len(gt),
                  incidents_outside_gt=sum(1 for t in inc_ts if not any(a <= t < b for a, b in gt)),
                  asks=int(st2.asks), fires=int(st2.fires))
        oracle_info = {"funnel": ch2.funnel, "wall_s": ch2.wall_s,
                       "trace_stages": dict(Counter(e.stage for e in ch2.trace))}
    else:
        out["notes"].append("gate-frame reproduction count != envelope count -- stages 2-4 not measured")

    out.update(
        stage1_substitution_ok=sub_ok,
        stages={"stage0_ingest": s0, "stage1_admission": s1, "stage2_perception": s2,
                "stage3_judge": s3, "stage4_cognition": s4},
        file_arm={"funnel": ch.funnel, "wall_s": ch.wall_s,
                  "trace_stages": dict(Counter(e.stage for e in ch.trace))},
        oracle_arm=oracle_info,
        # retained for the threshold sweep / matched-economy comparison
        _sweep={"ts": ts, "mean": mn if repro_ok else mean_vals, "block_max": bm,
                "gt": gt, "gt_events": gt_events, "tiles": tiles, "ok": repro_ok},
    )
    return out


# --------------------------------------------------------------------------------------------
# Stage-1 statistic comparison: block_max@0.20 (shipped) vs mean (historical), at MATCHED economy
# --------------------------------------------------------------------------------------------


def admission_at(case_sweep: dict, stat: str, threshold: float) -> tuple[list[bool], int]:
    vals = case_sweep[stat]
    keep = [v >= threshold for v in vals]
    return keep, sum(keep)


def sweep_point(cases: list[dict], stat: str, threshold: float) -> dict[str, Any]:
    """Corpus-wide recall/precision/economy for one (stat, threshold), pooled over positive TIME."""
    pos_passed = pos_total = passed_total = 0.0
    units_p = units_r = 0
    gi = gip = 0
    zero_admission_cases = 0
    zero_admission_on_positive = 0
    per_kind: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for c in cases:
        sw = c.get("_sweep")
        if not sw or not sw["ok"]:
            continue
        keep, n_adm = admission_at(sw, stat, threshold)
        spans = spans_of(sw["tiles"], keep)
        gt = sw["gt"]                       # union -- for TIME
        events = sw.get("gt_events") or gt  # individual annotations -- for EVENT recall
        pp = measure(intersect(spans, gt))
        pos_passed += pp
        pos_total += measure(gt)
        passed_total += measure(spans)
        units_p += n_adm
        units_r += len(sw["tiles"])
        hit = touched(spans, events)
        gi += len(events)
        gip += hit
        if n_adm == 0:
            zero_admission_cases += 1
        if events and hit == 0:
            zero_admission_on_positive += 1
        per_kind[c["goal_kind"]].append((hit, len(events)))
    return {
        "stat": stat, "threshold": threshold,
        # interval recall -- the definition the pre-registered tier-0 numbers in percept/config.py use
        "recall": (gip / gi) if gi else None,
        "recall_time": (pos_passed / pos_total) if pos_total else None,
        "precision": (pos_passed / passed_total) if passed_total else None,
        "economy": (units_p / units_r) if units_r else None,
        "reduction": 1.0 - (units_p / units_r) if units_r else None,
        "zero_admission_cases": zero_admission_cases,
        "cases_missing_every_positive": zero_admission_on_positive,
        "gt_intervals": gi, "gt_intervals_hit": gip,
        "recall_by_goal_kind": {
            k: (sum(h for h, _ in v) / sum(n for _, n in v)) if sum(n for _, n in v) else None
            for k, v in sorted(per_kind.items())
        },
        "n_cases": sum(1 for c in cases if c.get("_sweep", {}).get("ok")),
    }


def match_economy(cases: list[dict], stat: str, target_economy: float) -> dict[str, Any]:
    """The threshold at which ``stat`` spends the SAME admission budget as the shipped default.

    Comparing two statistics at their own thresholds compares a threshold change and a statistic
    change at once; the conclusion can invert.  This binary-searches the threshold that equalises
    economy so the comparison isolates the statistic.
    """
    vals = sorted(v for c in cases if c.get("_sweep", {}).get("ok") for v in c["_sweep"][stat])
    if not vals:
        return {}
    lo, hi = vals[0], vals[-1]
    best = None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pt = sweep_point(cases, stat, mid)
        econ = pt["economy"] or 0.0
        if best is None or abs(econ - target_economy) < abs((best["economy"] or 0.0) - target_economy):
            best = pt
        if econ > target_economy:
            lo = mid
        else:
            hi = mid
    return best or {}


# --------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def pool(cases: list[dict], stage: str) -> dict[str, Any]:
    """Pool a stage over cases by TIME and UNITS -- never by averaging per-case ratios.

    A per-case mean weights a 15 s clip the same as a 360 s one and lets a case with 0.2 s of
    positive time swing the headline.  Pooling the numerators and denominators is the honest
    aggregate; the per-case mean is reported alongside so the spread is visible.
    """
    rows = [c["stages"][stage] for c in cases if c.get("stages", {}).get(stage)]
    if not rows:
        return {"n": 0}
    pp = sum(r["positive_time_passed_s"] for r in rows)
    pr = sum(r["positive_time_received_s"] for r in rows)
    gtt = sum(r["gt_time_s"] for r in rows)
    pt = sum(r["passed_time_s"] for r in rows)
    ur = sum(r["units_received"] for r in rows)
    up = sum(r["units_passed"] for r in rows)
    dr = sum((r["drop_rate"] or 0.0) * r["units_received"] for r in rows)
    gi = sum(r.get("gt_intervals", 0) for r in rows)
    gir = sum(r.get("gt_intervals_received", 0) for r in rows)
    gip = sum(r.get("gt_intervals_passed", 0) for r in rows)
    return {
        "n": len(rows),
        "recall_stage_local": (pp / pr) if pr > 0 else None,
        "recall_cumulative": (pp / gtt) if gtt > 0 else None,
        "recall_interval_stage_local": (gip / gir) if gir else None,
        "recall_interval_cumulative": (gip / gi) if gi else None,
        "precision": (pp / pt) if pt > 0 else None,
        "economy": (up / ur) if ur else None,
        "drop_rate": (dr / ur) if ur else None,
        "units_received": ur,
        "units_passed": up,
        "positive_time_passed_s": round(pp, 3),
        "positive_time_received_s": round(pr, 3),
        "gt_time_s": round(gtt, 3),
        "gt_intervals": gi,
        "gt_intervals_received": gir,
        "gt_intervals_passed": gip,
        "per_case_mean_recall_cumulative": _mean([r["recall_cumulative"] for r in rows]),
        "per_case_mean_recall_stage_local": _mean([r["recall_stage_local"] for r in rows]),
        "per_case_mean_recall_interval_cumulative": _mean([r.get("recall_interval_cumulative") for r in rows]),
    }


def breakdown(cases: list[dict], stage: str, key: str) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        groups[str(c.get(key))].append(c)
    return {k: pool(v, stage) for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------------------------
# long-horizon sources (multi-occurrence / re-arm)
# --------------------------------------------------------------------------------------------


def measure_long_source(src: dict) -> dict[str, Any]:
    path = src["video_path"]
    if not os.path.isabs(path):
        path = str(REPO / path)
    if not os.path.exists(path):
        alt = HERE / "long_sources" / os.path.basename(path)
        path = str(alt) if alt.exists() else path
    if not os.path.exists(path):
        return {"goal_key": src["goal_key"], "error": f"media missing: {path}"}

    src_fps, n_native = probe_media(path)
    goal_text = (src.get("watch") or {}).get("condition") or src["goal_key"].split("::")[-1].replace("_", " ")
    ch = drive(make_core(OracleVision({}), goal_text), path)
    ed = edge_rows(ch)
    if not ed:
        return {"goal_key": src["goal_key"], "error": "no edge.decision trace"}
    ts = [float(e.t) for e in ed]
    esc = [bool(e.data.get("escalate") or e.data.get("coast_admit")) for e in ed]
    media_end = max(ts[-1], float(src.get("total_duration_s") or 0.0))
    gt, gt_events, _kind, _occ = ground_truth(src["positives"], media_end)
    tiles = tile(ts, media_end)
    admitted = spans_of(tiles, esc)

    step = max(1, round(n_native / len(ed)))
    gates = reproduce_gates(path, step)
    n = min(len(gates), len(ed))
    rnd = random.Random(PAYLOAD_SEED)
    payloads = [rnd.randbytes(256) for _ in range(n)]
    in_gt = [any(a <= ts[j] < b for a, b in gt) for j in range(n)]
    script = {hashlib.sha256(payloads[j]).hexdigest(): (YES if in_gt[j] else NO) for j in range(n)}
    vision = OracleVision(script)
    ch2 = drive(make_core(vision, goal_text), [(ts[j], gates[j], payloads[j]) for j in range(n)])

    inc_ts = [float(getattr(i, "ts", 0.0)) for i in ch2.incidents]
    matched = [any(a <= t < b for t in inc_ts) for a, b in gt_events]
    # Re-arm: distinct occurrences must yield DISTINCT incidents. Duplicates inside one occurrence
    # mean the gate re-fired without the condition having gone false; zero means it never re-armed.
    per_occ = [sum(1 for t in inc_ts if a <= t < b) for a, b in gt_events]
    return {
        "goal_key": src["goal_key"],
        "media_s": round(media_end, 2),
        "expected_occurrences": src.get("expected_occurrences"),
        "gt_occurrences": len(gt_events),
        "gt_time_spans_after_union": len(gt),
        "gt_time_s": round(measure(gt), 2),
        "frames": len(ed),
        "admitted": sum(esc),
        "admission_recall_cumulative": (measure(intersect(admitted, gt)) / measure(gt)) if measure(gt) else None,
        "admission_event_recall": (touched(admitted, gt_events) / len(gt_events)) if gt_events else None,
        "admission_economy": sum(esc) / len(ed),
        "incidents": len(ch2.incidents),
        "occurrences_matched": sum(matched),
        "occurrence_recall": (sum(matched) / len(gt_events)) if gt_events else None,
        "incidents_per_occurrence": per_occ,
        "duplicate_fires_within_occurrence": sum(max(0, k - 1) for k in per_occ),
        "incidents_outside_gt": sum(1 for t in inc_ts if not any(a <= t < b for a, b in gt)),
        "rearmed_every_occurrence": all(k >= 1 for k in per_occ) if per_occ else None,
        "unique_incidents": len({getattr(i, "id", None) for i in ch2.incidents}),
    }


# --------------------------------------------------------------------------------------------
# reconciliation against the runtime's OWN ledgers
# --------------------------------------------------------------------------------------------


def reconcile(cases: list[dict]) -> dict[str, Any]:
    """Cross-check every stage boundary against the counters the runtime keeps for itself.

    A harness that only trusts its own arithmetic can be wrong in the same direction as the thing it
    measures.  Each check below compares a number this module derived from the trace stream against
    an INDEPENDENT counter the runtime maintains (``RuntimeStats``, the ``runtime_ended``
    ``source_coverage`` ledger, and the ``core.metrics`` funnel).  Disagreement is reported, never
    reconciled away.
    """
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, agree: int, total: int, detail: str) -> None:
        checks[name] = {"agree": agree, "total": total,
                        "ok": agree == total and total > 0, "detail": detail}

    n = a_seen = a_gated = a_judged = a_bal = a_esc = 0
    balance_examples: list[dict] = []
    for c in cases:
        fn = (c.get("file_arm") or {}).get("funnel") or {}
        st = c.get("stages") or {}
        s0, s1 = st.get("stage0_ingest"), st.get("stage1_admission")
        if not (fn and s0 and s1):
            continue
        n += 1
        # The funnel omits a counter entirely when it never incremented, so an absent key means
        # zero. Comparing the raw `None` against 0 would manufacture a disagreement on exactly the
        # cases that are most interesting (the ones that admitted nothing).
        g = lambda k: int(fn.get(k) or 0)  # noqa: E731
        if g("funnel.seen") == s0["units_passed"]:
            a_seen += 1
        if g("funnel.gated") == s1["units_passed"]:
            a_gated += 1
        if s1["units_passed"] == s1.get("escalations"):
            a_esc += 1
        judged = g("funnel.judged")
        if judged == s1["units_passed"]:
            a_judged += 1
        fired = g("funnel.fired")
        refused = g("funnel.refused")
        silent = g("gate.silent")
        if judged == fired + refused + silent:
            a_bal += 1
        elif len(balance_examples) < 5:
            balance_examples.append({"case_id": c["case_id"], "judged": judged, "fired": fired,
                                     "refused": refused, "silent": silent,
                                     "fired+refused+silent": fired + refused + silent})

    check("funnel.seen == envelopes emitted (Stage 0 out)", a_seen, n,
          "the funnel's ingress counter against the envelope count this harness derived from `edge.decision`")
    check("funnel.gated == units admitted (Stage 1 out)", a_gated, n,
          "the funnel's admission counter against the `escalate` flags in the trace")
    check("RuntimeStats.escalations == admitted units", a_esc, n,
          "the runtime's own admission counter against the trace-derived count")
    check("funnel.judged == units admitted", a_judged, n,
          "every admitted unit reached a judge call")
    check("funnel.judged == fired + refused + silent", a_bal, n,
          "the funnel's own stated balance invariant")
    return {"n_cases": n, "checks": checks, "balance_disagreements": balance_examples,
            "note": (
                "A failing check here is a BOOKKEEPING defect in the runtime's own mirror, not "
                "necessarily a data-path defect: the per-stage metrics in this report are derived "
                "from the trace stream and the `source_coverage` ledger, both of which reconcile."
            ),
            "workload_caveat": (
                "The balance invariant is exercised only over the dispositions this workload "
                "produces. The scripted judge answers with high confidence, so the gate resolves to "
                "fire/silent and almost never to ASK; a known double-increment on the ASK path is "
                "therefore NOT exercised here and this check should not be read as clearing it."
            )}


# --------------------------------------------------------------------------------------------
# audio (VAD / acoustic admission) -- Stage 0 + Stage 1 on the sound channel
# --------------------------------------------------------------------------------------------


def measure_audio(limit: int = 0) -> dict[str, Any]:
    """Stage 0 and Stage 1 on the AUDIO channel, over the audio ground-truth corpus.

    The corpus is built in-process (``build_audio_corpus.build()``) rather than read from disk --
    that is the suite's own design, so no stale file can drift from what was measured.  Cases are
    partitioned by ``gt_strength`` and the ``proxy`` cases (a Charades label that is *usually*
    audible but whose annotated interval is VISUAL) are reported but NEVER used as an audio bar.

    Two things are measured, and they are deliberately not merged:

    ``public_path``   what ``percept.run(source=..., audio_source=...)`` actually exposes.  The
                      audio-only entrypoint (``source=None``) refuses an uncompiled Watch, so the
                      fused path is the reachable one.
    ``component``     the shipped ``AcousticOnset`` gate driven at the shipped capture cadence.
                      This is a COMPONENT measurement, not the public path, and is labelled so.
    """
    sys.path.insert(0, str(HERE))
    out: dict[str, Any] = {"available": False}
    try:
        import build_audio_corpus as builder  # noqa: PLC0415
    except Exception as exc:
        out["error"] = f"audio corpus builder unavailable: {type(exc).__name__}: {exc}"
        return out
    try:
        corpus = builder.build()
    except Exception as exc:
        out["error"] = f"audio corpus build failed: {type(exc).__name__}: {exc}"
        return out

    cases = corpus["cases"]
    by_strength: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_strength[c["gt_strength"]].append(c)
    bar_cases = by_strength.get("audio_annotated", [])
    if limit:
        bar_cases = bar_cases[:limit]

    out.update(
        available=True,
        n_cases=len(cases),
        cases_by_gt_strength={k: len(v) for k, v in sorted(by_strength.items())},
        bar_eligible_strength="audio_annotated",
        excluded_from_bar={
            "proxy": len(by_strength.get("proxy", [])),
            "audio_annotated_point": len(by_strength.get("audio_annotated_point", [])),
        },
    )

    # ---- the audio-only public entrypoint: is it reachable with an ordinary Goal? -----------
    probe = bar_cases[0] if bar_cases else None
    if probe:
        path = str(REPO / probe["video_path"])
        try:
            drive(make_core(OracleVision({}), probe["goal_text"]), None, audio_source=path)
            out["audio_only_entrypoint"] = {"reachable": True, "error": None}
        except Exception as exc:
            out["audio_only_entrypoint"] = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- the fused public path over every bar-eligible case ---------------------------------
    chunks = wakes = observed = excluded = disc = missing = 0
    per_case: list[dict] = []
    for c in bar_cases:
        path = str(REPO / c["video_path"])
        if not os.path.exists(path):
            continue
        try:
            ch = drive(make_core(OracleVision({}), c["goal_text"]), path, audio_source=path)
        except Exception as exc:
            per_case.append({"case_id": c["case_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        acov = (health_data(ch, "runtime_ended").get("source_coverage") or {}).get("audio", {})
        chunks += int(ch.stats.audio_chunks)
        wakes += int(ch.stats.audio_wakes)
        observed += int(acov.get("observed", 0))
        excluded += int(acov.get("excluded", 0))
        disc += int(acov.get("discontinuities", 0))
        missing += int(acov.get("missing_sequences", 0))
        per_case.append({
            "case_id": c["case_id"], "audio_chunks": int(ch.stats.audio_chunks),
            "audio_wakes": int(ch.stats.audio_wakes),
            "audio_observed": int(acov.get("observed", 0)),
            "gt_intervals": len(c["positives"]),
            "admission_by_cause": dict(ch.stats.admission_by_cause),
        })
    out["public_path"] = {
        "n_cases": len([p for p in per_case if "error" not in p]),
        "stage0_audio_ingest": {
            "units_observed": observed, "units_excluded_with_receipt": excluded,
            "discontinuities": disc, "missing_sequences": missing,
            "drop_rate": (max(0, chunks - observed) / chunks) if chunks else None,
            "economy": (observed / chunks) if chunks else None,
        },
        "stage1_audio_admission": {
            "audio_chunks_received": chunks, "audio_wakes": wakes,
            "economy": (wakes / chunks) if chunks else None,
            "recall_stage_local": None, "recall_cumulative": None, "precision": None,
            "why_recall_is_none": (
                "The runtime emits NO per-audio-chunk admission event. `edge.decision` is the video "
                "gate only, and an acoustic wake surfaces solely as the aggregate counters "
                "`RuntimeStats.audio_wakes` / `admission_by_cause['audio']`. Without a time-resolved "
                "admission series there is nothing to intersect with audio ground truth, so audio "
                "Stage-1 recall and precision are NOT instrumentable through the public API."
            ),
        },
        "per_case": per_case,
    }

    # ---- component-level acoustic-onset recall (NOT the public path) ------------------------
    try:
        import sv_audio  # noqa: PLC0415

        comp = sv_audio.admission_recall(
            bar_cases, high=sv_audio.SHIPPED_ONSET_HIGH, low=sv_audio.SHIPPED_ONSET_LOW
        )
        comp.pop("per_case", None)
        comp["measurement_level"] = (
            "COMPONENT, not the public path: `percept.harness.detectors.AcousticOnset` driven "
            "directly at the shipped capture cadence, exactly as `_run_audio_only` feeds it."
        )
        comp["tolerance_s"] = sv_audio.ADMISSION_TOLERANCE_S
        comp["gt_strength"] = "audio_annotated"
        out["component_acoustic_onset"] = comp
    except Exception as exc:
        out["component_acoustic_onset"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------------------------


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_md(payload: dict) -> str:
    L: list[str] = []
    A = L.append
    meta = payload["meta"]
    A("# Per-stage recall, precision, economy and drop rate")
    A("")
    A(f"Runtime under test: `percept` wheel at `{meta['build_commit']}` "
      f"(`{meta['interpreter']}`).")
    A(f"Corpus: {meta['n_cases_measured']} of {meta['n_cases_total']} video cases; "
      f"{meta['gt_time_s']:.0f} s of ground-truth-positive time over "
      f"{meta['media_s']:.0f} s of media. Generated {meta['generated']}.")
    A("")
    A("No provider or network call was made. Judges are scripted and offline: Stage 1 runs against "
      "the real decoded media through the shipped gate, and Stages 3-4 are driven by a ground-truth "
      "oracle so they measure PLUMBING, not judge quality (explicitly out of scope).")
    A("")

    A("## Headline - where the pipeline loses")
    A("")
    A("Recall is reported twice at every stage. **stage-local** is versus what that stage actually "
      "received; **cumulative** is versus the original source ground truth. A stage whose "
      "stage-local recall is 1.000 lost nothing itself, however low its cumulative number is — that "
      "pair is the whole point, and it is what an end-to-end number cannot give you.")
    A("")
    A("| Stage | Recall (stage-local) | Recall (cumulative) | Precision | Economy | Drop rate | Units in -> out | n |")
    A("|---|---|---|---|---|---|---|---|")
    for st in STAGES:
        p = payload["overall"][st]
        if not p.get("n"):
            A(f"| {STAGE_TITLES[st]} | not instrumented | not instrumented | - | - | - | - | 0 |")
            continue
        A(f"| {STAGE_TITLES[st]} | {_f(p['recall_stage_local'])} | {_f(p['recall_cumulative'])} | "
          f"{_f(p['precision'])} | {_f(p['economy'])} | {_f(p['drop_rate'])} | "
          f"{p['units_received']:,} -> {p['units_passed']:,} | {p['n']} |")
    A("")
    A("Recall above is the spec's literal definition: the fraction of ground-truth-positive SOURCE "
      "TIME the stage passed through. On a sampled pipeline that quantity is bounded above by the "
      "stage's own economy — a gate admitting 18% of units cannot cover more than 18% of the clock — "
      "so at Stage 1 it largely restates economy and understates whether the event actually got "
      "through. The table below is the same two-population pair computed on OCCURRENCES: did at "
      "least one unit inside each ground-truth event survive this stage? This is the definition the "
      "pre-registered tier-0 numbers in `percept/config.py` use, and it is the one that localises a "
      "regression.")
    A("")
    A("| Stage | Event recall (stage-local) | Event recall (cumulative) | GT events in -> out |")
    A("|---|---|---|---|")
    for st in STAGES:
        p = payload["overall"][st]
        if not p.get("n"):
            A(f"| {STAGE_TITLES[st]} | not instrumented | not instrumented | - |")
            continue
        A(f"| {STAGE_TITLES[st]} | {_f(p.get('recall_interval_stage_local'))} | "
          f"{_f(p.get('recall_interval_cumulative'))} | "
          f"{p.get('gt_intervals_received', 0):,} -> {p.get('gt_intervals_passed', 0):,} |")
    A("")
    worst = payload["analysis"]["largest_loss_stage"]
    A(f"**The stage that loses the most is {STAGE_TITLES[worst['stage']]}**: stage-local time recall "
      f"{_f(worst['recall_stage_local'])} and stage-local event recall "
      f"{_f(worst.get('recall_interval_stage_local'))} — it discards "
      f"{worst['positive_time_lost_s']:.0f} s of the {worst['positive_time_received_s']:.0f} s of "
      f"ground-truth-positive time it was handed, and loses "
      f"{worst.get('gt_intervals_lost', 0)} ground-truth "
      f"{'event' if worst.get('gt_intervals_lost') == 1 else 'events'} entirely. "
      f"{payload['analysis']['loss_narrative']}")
    A("")

    A("## The funnel, source units to Incidents")
    A("")
    A("| Boundary | Units | Fraction of source frames |")
    A("|---|---|---|")
    fn = payload["funnel"]
    base = fn["native_frames_decoded"] or 1
    for label, key in (
        ("Native frames decoded from media", "native_frames_decoded"),
        ("Selected by deliberate decimation", "selected_after_decimation"),
        ("Envelopes emitted by ingest", "envelopes_emitted"),
        ("Admitted by tier-0 (Stage 1)", "admitted"),
        ("Dispatched to local perception (Stage 2)", "perceive_dispatches"),
        ("Judge calls started (Stage 3, logical)", "judge_logical_calls"),
        ("Judge calls completed (Stage 3)", "judge_finished"),
        ("Gate classifications (Stage 4)", "gate_classified"),
        ("Gate concluded condition true", "gate_true"),
        ("Incidents emitted", "incidents"),
    ):
        v = fn.get(key) or 0
        A(f"| {label} | {v:,} | {v / base:.4f} |")
    A("")
    A(f"Ground-truth occurrences across the corpus: {fn['gt_occurrences']:,}; "
      f"occurrences matched by at least one Incident: {fn['occurrences_matched']:,} "
      f"({fn['occurrence_recall']:.3f}).")
    A("")

    A("## Stage 1 - the statistic change: block_max@0.20 vs mean")
    A("")
    A("The default salience statistic changed to `block_max` @ 0.20 in this wheel "
      "(`percept/config.py`), so every prior Stage-1 number computed under `mean` @ 0.10 is stale. "
      "Both are regenerated here against the same corpus and the same reproduced gate frames, "
      "scored with the wheel's own `percept.harness.tier0.salience_stats`.")
    A("")
    A("Recall here is EVENT recall (fraction of ground-truth occurrences given at least one admitted "
      "frame) — the same definition as the pre-registered numbers this replaces. Time recall is "
      "carried alongside.")
    A("")
    A("| Configuration | Event recall | Time recall | Precision | Economy | Reduction | "
      "Cases admitting zero frames | Cases missing every positive |")
    A("|---|---|---|---|---|---|---|---|")
    for row in payload["stage1_statistic_comparison"]["points"]:
        A(f"| `{row['label']}` | {_f(row['recall'])} | {_f(row.get('recall_time'))} | "
          f"{_f(row['precision'])} | {_f(row['economy'])} | {_f(row['reduction'])} | "
          f"{row['zero_admission_cases']} | {row.get('cases_missing_every_positive', 0)} |")
    A("")
    cmp_ = payload["stage1_statistic_comparison"]
    A(f"**At matched economy** ({_f(cmp_['matched_economy_target'])}), which is the only fair "
      "comparison — comparing each statistic at its own threshold changes the statistic and the "
      "budget at once, and the conclusion can invert — "
      f"`block_max` event recall is {_f(cmp_['matched']['block_max_recall'])} against `mean`'s "
      f"{_f(cmp_['matched']['mean_recall'])}, a difference of "
      f"**{_f(cmp_['matched']['delta'], 3)}** at identical admission spend.")
    A("")
    A(cmp_["verdict"])
    A("")
    A("Per-`goal_kind` event recall at each configuration:")
    A("")
    kinds = sorted({k for row in cmp_["points"] for k in (row.get("recall_by_goal_kind") or {})})
    A("| Configuration | " + " | ".join(f"`{k}`" for k in kinds) + " |")
    A("|---" * (len(kinds) + 1) + "|")
    for row in cmp_["points"]:
        cells = " | ".join(_f((row.get("recall_by_goal_kind") or {}).get(k)) for k in kinds)
        A(f"| `{row['label']}` | {cells} |")
    A("")

    A("## Per-stage breakdown by goal_kind")
    A("")
    A("`state` has n=4 and cannot support a claim: it is reported, never gated.")
    A("")
    for st in STAGES:
        rows = payload["by_goal_kind"].get(st) or {}
        if not any(r.get("n") for r in rows.values()):
            continue
        A(f"### {STAGE_TITLES[st]}")
        A("")
        A("| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | "
          "Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for k, r in rows.items():
            if not r.get("n"):
                continue
            A(f"| {k} | {_f(r['recall_stage_local'])} | {_f(r['recall_cumulative'])} | "
              f"{_f(r.get('recall_interval_stage_local'))} | {_f(r.get('recall_interval_cumulative'))} | "
              f"{_f(r['precision'])} | {_f(r['economy'])} | {_f(r['drop_rate'])} | {r['n']} | "
              f"{r.get('gt_intervals', 0)} |")
        A("")

    A("## Per-stage breakdown by dataset")
    A("")
    for st in STAGES:
        rows = payload["by_dataset"].get(st) or {}
        if not any(r.get("n") for r in rows.values()):
            continue
        A(f"### {STAGE_TITLES[st]}")
        A("")
        A("| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | "
          "Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for k, r in rows.items():
            if not r.get("n"):
                continue
            A(f"| {k} | {_f(r['recall_stage_local'])} | {_f(r['recall_cumulative'])} | "
              f"{_f(r.get('recall_interval_stage_local'))} | {_f(r.get('recall_interval_cumulative'))} | "
              f"{_f(r['precision'])} | {_f(r['economy'])} | {_f(r['drop_rate'])} | {r['n']} | "
              f"{r.get('gt_intervals', 0)} |")
        A("")

    A("## Long-horizon sources - multi-occurrence and re-arm")
    A("")
    A("| Source | Media s | GT occurrences | Admission event recall | Admission time recall | Incidents | Occurrences matched | Duplicate fires | Re-armed on every occurrence |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in payload["long_sources"]:
        if r.get("error"):
            A(f"| {r.get('goal_key')} | - | - | - | - | - | - | - | {r['error']} |")
            continue
        A(f"| `{r['goal_key']}` | {r['media_s']:.0f} | {r['gt_occurrences']} | "
          f"{_f(r.get('admission_event_recall'))} | {_f(r['admission_recall_cumulative'])} | "
          f"{r['incidents']} | {r['occurrences_matched']}/{r['gt_occurrences']} | "
          f"{r['duplicate_fires_within_occurrence']} | {r['rearmed_every_occurrence']} |")
    A("")
    A("`Duplicate fires` counts Incidents beyond the first inside one ground-truth occurrence — a "
      "gate that re-fired without the condition having gone false. `Re-armed on every occurrence` is "
      "the multi-occurrence question these sources exist to answer: after firing and closing, does "
      "the gate come back for the next one?")
    A("")

    A("## Audio channel (VAD / acoustic admission)")
    A("")
    au = payload.get("audio") or {}
    if not au.get("available"):
        A(f"Not measured: {au.get('error', 'audio corpus unavailable')}")
        A("")
    else:
        A(f"Audio ground-truth corpus: {au['n_cases']} cases — "
          + ", ".join(f"{v} `{k}`" for k, v in sorted(au["cases_by_gt_strength"].items()))
          + ". Only `audio_annotated` (a human annotated a SOUND event with a start and an end) is "
            "used as a bar. `proxy` cases carry a Charades label that is usually audible but whose "
            "annotated interval is VISUAL, and are excluded from every audio number below.")
        A("")
        ent = au.get("audio_only_entrypoint") or {}
        if ent and not ent.get("reachable"):
            A(f"**The audio-only entrypoint is not reachable with an ordinary `Goal`.** "
              f"`run(source=None, audio_source=...)` raises `{ent.get('error')}`. Audio therefore "
              f"had to be measured on the FUSED path (`source=` and `audio_source=` both set).")
            A("")
        pp = au.get("public_path") or {}
        s0a = pp.get("stage0_audio_ingest") or {}
        s1a = pp.get("stage1_audio_admission") or {}
        A("| Audio boundary | Units received | Units passed | Economy | Drop rate | Recall | Precision |")
        A("|---|---|---|---|---|---|---|")
        A(f"| Stage 0 — audio ingest | {s0a.get('units_observed', 0):,} | "
          f"{s0a.get('units_observed', 0):,} | {_f(s0a.get('economy'))} | "
          f"{_f(s0a.get('drop_rate'))} | 1.000 | n/a |")
        A(f"| Stage 1 — acoustic admission | {s1a.get('audio_chunks_received', 0):,} | "
          f"{s1a.get('audio_wakes', 0):,} | {_f(s1a.get('economy'))} | 0.000 | "
          f"**not instrumentable** | **not instrumentable** |")
        A("")
        A(s1a.get("why_recall_is_none", ""))
        A("")
        comp = au.get("component_acoustic_onset") or {}
        if comp and not comp.get("error"):
            A(f"For a number that does exist, the shipped `AcousticOnset` gate was driven directly "
              f"at the shipped capture cadence over the {comp.get('positives', 0)} annotated sound "
              f"events in the {au['cases_by_gt_strength'].get('audio_annotated', 0)} "
              f"`audio_annotated` cases: **event recall {_f(comp.get('recall'))}**, economy "
              f"{_f(comp.get('economy'))} ({comp.get('admissions', 0):,} wakes over "
              f"{comp.get('chunks', 0):,} capture chunks), at a declared "
              f"{_f(comp.get('tolerance_s'))} s tolerance (one capture chunk — an onset is reported "
              f"at the chunk's END). This is a COMPONENT measurement, not the public path, and must "
              f"not be quoted as an end-to-end audio number.")
            A("")

    A("## Statistical power")
    A("")
    A(payload["analysis"]["power_note"])
    A("")

    A("## Boundaries that could NOT be instrumented")
    A("")
    A("An honest 'not instrumentable at this boundary' is a finding. These are the boundaries this "
      "harness could not measure through the public API, and why.")
    A("")
    for item in payload["not_instrumented"]:
        A(f"- **{item['boundary']}** — {item['why']}")
    A("")

    A("## Reconciliation against the runtime's own ledgers")
    A("")
    rec = payload.get("reconciliation") or {}
    A(f"Each per-stage number this report derives from the trace stream is cross-checked against an "
      f"INDEPENDENT counter the runtime keeps for itself, over {rec.get('n_cases', 0)} cases.")
    A("")
    A("| Check | Cases agreeing | Verdict |")
    A("|---|---|---|")
    for name, c in (rec.get("checks") or {}).items():
        A(f"| {name} | {c['agree']}/{c['total']} | {'RECONCILES' if c['ok'] else 'DISAGREES'} |")
    A("")
    ex = rec.get("balance_disagreements") or []
    if ex:
        e = ex[0]
        A(f"The funnel's own balance invariant does not hold. Example (`{e['case_id']}`): "
          f"`judged={e['judged']}` but `fired+refused+silent={e['fired+refused+silent']}` "
          f"(fired {e['fired']}, refused {e['refused']}, silent {e['silent']}). "
          f"{rec.get('note', '')}")
        A("")
    A(rec.get("workload_caveat", ""))
    A("")

    A("## Method and self-checks")
    A("")
    for item in payload["method_checks"]:
        A(f"- {item}")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def strip_sweep(cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        d = {k: v for k, v in c.items() if k != "_sweep"}
        st = d.get("stages") or {}
        s4 = st.get("stage4_cognition")
        if s4 and len(s4.get("incident_ts") or []) > 50:
            s4 = dict(s4)
            s4["incident_ts"] = s4["incident_ts"][:50] + ["...truncated"]
            st = dict(st)
            st["stage4_cognition"] = s4
            d["stages"] = st
        out.append(d)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N cases (smoke)")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--audio-limit", type=int, default=0)
    args = ap.parse_args(argv)

    commit = build_info().git_commit
    settings = load_settings()
    shipped_stat = settings.edge.salience_stat
    shipped_threshold = {
        "mean": settings.edge.motion_gate,
        "block_max": settings.edge.motion_gate_block_max,
        "p95": settings.edge.motion_gate_p95,
    }[shipped_stat]
    print(f"[stage_metrics] wheel {commit}", file=sys.stderr)
    print(f"[stage_metrics] shipped tier-0 gate: {shipped_stat} @ {shipped_threshold}", file=sys.stderr)

    corpus = load_corpus()
    if args.limit:
        corpus = corpus[: args.limit]

    measured: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(corpus, 1):
        try:
            m = measure_case(case, shipped_stat=shipped_stat, shipped_threshold=shipped_threshold)
        except Exception as exc:  # a case that cannot be measured is reported, never estimated
            m = {"case_id": case["case_id"], "dataset": case["dataset"],
                 "goal_kind": case["goal_kind"], "measurable": False,
                 "notes": [f"exception: {type(exc).__name__}: {exc}"]}
        measured.append(m)
        print(f"[{i}/{len(corpus)}] {case['case_id']} "
              f"{'ok' if m.get('measurable') else 'UNMEASURABLE'} "
              f"({time.time() - t0:.0f}s)", file=sys.stderr)

    ok = [m for m in measured if m.get("measurable") and m.get("stages")]

    overall = {st: pool(ok, st) for st in STAGES}
    by_kind = {st: breakdown(ok, st, "goal_kind") for st in STAGES}
    by_ds = {st: breakdown(ok, st, "dataset") for st in STAGES}

    # ---- Stage-1 statistic comparison, including at matched economy ----------------------
    shipped_pt = sweep_point(ok, "block_max", shipped_threshold)
    shipped_pt["label"] = f"block_max @ {shipped_threshold} (SHIPPED DEFAULT)"
    mean_pt = sweep_point(ok, "mean", 0.10)
    mean_pt["label"] = "mean @ 0.10 (historical default)"
    target = shipped_pt["economy"] or 0.0
    mean_matched = match_economy(ok, "mean", target)
    if mean_matched:
        mean_matched["label"] = f"mean @ {mean_matched['threshold']:.4f} (MATCHED ECONOMY)"
    delta = None
    if mean_matched and shipped_pt.get("recall") is not None and mean_matched.get("recall") is not None:
        delta = shipped_pt["recall"] - mean_matched["recall"]
    verdict = (
        "At equal admission spend the localized statistic is the better one: `block_max` keeps more "
        "ground-truth-positive time for the same number of judge calls. The `mean` statistic's "
        "failure mode is not that it is stingy — it is that a whole-frame average dilutes a small "
        "mover against a still background, so on some cases it admits nothing at all."
        if (delta or 0) > 0 else
        "At equal admission spend the two statistics are within noise on this corpus; the shipped "
        "default's advantage at its own threshold is a budget effect, not a statistic effect."
    )
    stage1_cmp = {
        "points": [shipped_pt, mean_pt] + ([mean_matched] if mean_matched else []),
        "matched_economy_target": target,
        "matched": {"block_max_recall": shipped_pt.get("recall"),
                    "mean_recall": (mean_matched or {}).get("recall"),
                    "mean_threshold": (mean_matched or {}).get("threshold"),
                    "mean_economy": (mean_matched or {}).get("economy"),
                    "delta": delta},
        "verdict": verdict,
        "n_cases_in_sweep": sum(1 for c in ok if c.get("_sweep", {}).get("ok")),
    }

    # ---- funnel --------------------------------------------------------------------------
    def _sum(stage: str, key: str) -> int:
        return int(sum((c["stages"].get(stage) or {}).get(key) or 0 for c in ok))

    funnel = {
        "native_frames_decoded": int(sum(c.get("native_frames") or 0 for c in ok)),
        "selected_after_decimation": _sum("stage0_ingest", "units_selected_after_decimation"),
        "envelopes_emitted": _sum("stage0_ingest", "units_passed"),
        "admitted": _sum("stage1_admission", "units_passed"),
        "perceive_dispatches": _sum("stage2_perception", "units_passed"),
        "judge_logical_calls": _sum("stage3_judge", "logical_calls"),
        "judge_physical_attempts": _sum("stage3_judge", "physical_attempts"),
        "judge_finished": _sum("stage3_judge", "units_passed"),
        "gate_classified": _sum("stage4_cognition", "gate_classified"),
        "gate_true": _sum("stage4_cognition", "units_passed"),
        "incidents": _sum("stage4_cognition", "incidents"),
        "gt_occurrences": int(sum(c.get("gt_occurrences") or 0 for c in ok)),
        "occurrences_matched": _sum("stage4_cognition", "occurrences_matched"),
    }
    funnel["occurrence_recall"] = (funnel["occurrences_matched"] / funnel["gt_occurrences"]
                                   if funnel["gt_occurrences"] else 0.0)

    # ---- which stage loses the most -------------------------------------------------------
    worst = None
    for st in STAGES:
        p = overall[st]
        if not p.get("n") or p.get("recall_stage_local") is None:
            continue
        lost = p["positive_time_received_s"] - p["positive_time_passed_s"]
        if worst is None or lost > worst["positive_time_lost_s"]:
            worst = {"stage": st, "recall_stage_local": p["recall_stage_local"],
                     "recall_interval_stage_local": p.get("recall_interval_stage_local"),
                     "positive_time_lost_s": lost,
                     "positive_time_received_s": p["positive_time_received_s"],
                     "gt_intervals_lost": p.get("gt_intervals_received", 0) - p.get("gt_intervals_passed", 0)}

    lossless = [STAGE_TITLES[st] for st in STAGES
                if overall[st].get("n") and (overall[st].get("recall_stage_local") or 0) >= 0.999
                and st != (worst or {}).get("stage")]
    loss_narrative = (
        ("Lossless on positive time at this boundary: " + "; ".join(lossless) + ".")
        if lossless else "No other boundary is lossless on positive time."
    )

    # ---- statistical power ----------------------------------------------------------------
    n_intervals = Counter()
    for c in ok:
        n_intervals[c["goal_kind"]] += c.get("gt_occurrences") or 0
    need = required_positives(0.95, 0) if required_positives else 59
    s1_kind = by_kind["stage1_admission"]
    bounds = []
    for k, r in sorted(s1_kind.items()):
        gi, gp = r.get("gt_intervals") or 0, r.get("gt_intervals_passed") or 0
        lo = clopper_pearson_lower(gp, gi) if (clopper_pearson_lower and gi) else None
        bounds.append({"goal_kind": k, "n": gi, "hits": gp,
                       "point_recall": (gp / gi) if gi else None, "cp_lower_95": lo,
                       "clears_0_95": bool(lo is not None and lo >= 0.95)})
    power = (
        "Per-cell ground-truth event counts at Stage 1: "
        + ", ".join(f"`{b['goal_kind']}` n={b['n']}" for b in bounds)
        + f". A 0.95 recall bar is NOT statistically ratifiable on this corpus at any configuration: "
        f"an exact one-sided Clopper-Pearson lower bound clearing 0.95 with ZERO misses needs "
        f"n={need} positives — even a flawless {n_intervals.get('duration', 0)}-event `duration` cell "
        f"bounds only at "
        f"{_f(clopper_pearson_lower(n_intervals.get('duration', 0), n_intervals.get('duration', 0)) if clopper_pearson_lower and n_intervals.get('duration') else None)}. "
        f"Measured exact lower bounds on Stage-1 event recall: "
        + "; ".join(f"`{b['goal_kind']}` {_f(b['point_recall'])} (95% lower {_f(b['cp_lower_95'])}, n={b['n']})"
                    for b in bounds)
        + f". `state` (n={n_intervals.get('state', 0)} events over "
        f"{sum(1 for c in ok if c['goal_kind'] == 'state')} cases) is reported and NEVER gated — it "
        f"cannot support a claim at any confidence."
    )

    # ---- long sources ---------------------------------------------------------------------
    long_rows: list[dict] = []
    if not args.skip_long:
        long_srcs = json.loads((HERE / "long_sources.json").read_text())["sources"]
        for s in long_srcs:
            try:
                long_rows.append(measure_long_source(s))
            except Exception as exc:
                long_rows.append({"goal_key": s.get("goal_key"), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[long] {s.get('goal_key')} done", file=sys.stderr)

    # ---- audio ------------------------------------------------------------------------------
    audio: dict[str, Any] = {"available": False, "error": "skipped by flag"}
    if not args.skip_audio:
        try:
            audio = measure_audio(limit=args.audio_limit)
        except Exception as exc:
            audio = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        print("[audio] done", file=sys.stderr)

    # ---- boundaries not instrumented ------------------------------------------------------
    audio_why = (
        "The runtime emits NO per-audio-chunk admission event. `edge.decision` is the VIDEO gate "
        "only; an acoustic wake surfaces solely as the aggregate counters `RuntimeStats.audio_wakes` "
        "and `admission_by_cause['audio']`. With no time-resolved admission series there is nothing "
        "to intersect with audio ground truth, so audio Stage-1 recall/precision cannot be computed "
        "from the public API. Separately, the audio-ONLY entrypoint (`run(source=None, "
        "audio_source=...)`) refuses an ordinary `Goal` — it requires a compiled Watch — so audio had "
        "to be measured on the fused path. A COMPONENT-level `AcousticOnset` recall is reported "
        "instead, clearly labelled as not-the-public-path."
    )
    not_instrumented = [
        {"boundary": "Audio / VAD admission recall and precision (Stage 1, acoustic)",
         "why": audio_why},
        {"boundary": "Stage 2 as a measuring point (local perception)",
         "why": "With no `EvidenceExtractor` armed, Stage 2 is an identity pass-through: it emits one "
                "`source.frame` per admitted unit and `perceive_dispatches == escalations` exactly. "
                "Its economy of 1.000 therefore cannot distinguish 'lossless' from 'absent'. The "
                "senses (`rhythm`, `identity`, `entity`, `sequencing`) are independent modules with no "
                "shared metrics contract, so there is no per-sense precision/recall to read."},
        {"boundary": "Physical provider attempts and typed terminal receipts (Stage 3)",
         "why": "`run()` exposes no route/receipt surface. `JudgeRouteReceipt` outcomes "
                "(`deadline`, `capacity`, `no_route`, `cancelled`) exist only on a `JudgePool` the "
                "caller constructs and holds. Physical attempts here are counted at the ADAPTER "
                "boundary (the scripted judge's own call counter), which equals logical calls because "
                "a direct adapter does not hedge. Route-level hedging is unmeasured by this harness."},
        {"boundary": "`metrics_collector=` on a video run",
         "why": "Accepted by `run()` and SILENTLY IGNORED on the video path: it is forwarded only to "
                "`_run_audio_only`, so the video `_JudgeScheduler` never receives it. Queue depth and "
                "oldest-pending-age for video are therefore unobservable through the documented "
                "mechanism. This harness reconciles against `RuntimeStats` and the `on_health` "
                "`runtime_ended.source_coverage` ledger instead."},
        {"boundary": "Units offered UPSTREAM of the decoder",
         "why": "Stage 0's denominator here is frames the decoder actually yielded. Frames lost inside "
                "the container/codec before `cap.read()` returns them are invisible to any counter in "
                "the runtime, so ingest recall is measured against what the decoder delivered, not "
                "against what the file nominally contains."},
        {"boundary": "Judge logical-vs-physical split under batching",
         "why": "`RuntimeStats.judge_calls` counts scheduler BATCH launches while `perceive_dispatches` "
                "counts per-job launches; they are different units and must not be divided by each "
                "other. Both are reported separately rather than combined into one economy figure."},
    ]

    method_checks = [
        f"Runtime under test verified at import: `build_info().git_commit == {commit}`.",
        "Zero provider or network calls: the only judge is a scripted in-process object keyed on a "
        "payload hash.",
        f"Stage-1 substitution VERIFIED per case, not assumed: the FILE arm and the ORACLE arm must "
        f"admit the identical set of timestamps. Held on "
        f"{sum(1 for c in ok if c.get('stage1_substitution_ok'))} of "
        f"{sum(1 for c in ok if c.get('stage1_substitution_ok') is not None)} cases where it was checkable.",
        f"Salience reproduction VERIFIED per case against the wheel's own `salience_stats`: "
        f"`trace.motion == mean` and `escalate == (block_max >= {shipped_threshold})` held exactly on "
        f"{sum(1 for c in ok if c.get('salience_reproduction_ok'))} of {len(ok)} cases; only those "
        f"contribute to the threshold sweep.",
        "Aggregates pool numerators and denominators over time and units. Per-case means are reported "
        "alongside so a pooled figure driven by one long clip is visible.",
        "Suppression is never counted as a drop: a tier-0 rejection emits its own `edge.decision` "
        "record naming the signal and the threshold it failed, which is the receipt.",
        f"Economy floor cross-checked against the value pinned in the runtime "
        f"(`percept/slo.py` `FOVEA_CALL_REDUCTION_FLOOR = {FOVEA_CALL_REDUCTION_FLOOR}`).",
        f"omnipro ground truth is zero-width instants; widened by a declared "
        f"{INSTANT_TOLERANCE_S} s tolerance and tagged `gt_kind=instant` so it never enters an "
        f"interval bar unannounced.",
    ]

    payload = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build_commit": commit,
            "interpreter": sys.executable,
            "shipped_salience_stat": shipped_stat,
            "shipped_threshold": shipped_threshold,
            "fovea_call_reduction_floor": FOVEA_CALL_REDUCTION_FLOOR,
            "instant_tolerance_s": INSTANT_TOLERANCE_S,
            "n_cases_total": len(corpus),
            "n_cases_measured": len(ok),
            "gt_time_s": sum(c.get("gt_time_s") or 0.0 for c in ok),
            "media_s": sum(c.get("media_end_s") or 0.0 for c in ok),
            "provider_calls": 0,
        },
        "metric_definitions": {
            "recall_stage_local": "positive source time passed by the stage / positive source time the stage received",
            "recall_cumulative": "positive source time passed by the stage / positive source time in the ORIGINAL source",
            "precision": "positive source time passed / total time passed",
            "economy": "units_passed / units_received",
            "drop_rate": "units lost WITHOUT a receipt / units received (deliberate suppression is not a drop)",
        },
        "overall": overall,
        "by_goal_kind": by_kind,
        "by_dataset": by_ds,
        "funnel": funnel,
        "stage1_statistic_comparison": stage1_cmp,
        "long_sources": long_rows,
        "audio": audio,
        "reconciliation": reconcile(ok),
        "analysis": {"largest_loss_stage": worst, "power_note": power,
                     "loss_narrative": loss_narrative,
                     "stage1_exact_bounds": bounds,
                     "required_positives_for_0_95_bar": need},
        "not_instrumented": not_instrumented,
        "method_checks": method_checks,
        "cases": strip_sweep(measured),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stage_metrics.json").write_text(json.dumps(payload, indent=1, default=str))
    (RESULTS / "stage_metrics.md").write_text(render_md(payload))
    print(f"[stage_metrics] wrote {RESULTS / 'stage_metrics.json'}", file=sys.stderr)
    print(f"[stage_metrics] wrote {RESULTS / 'stage_metrics.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
