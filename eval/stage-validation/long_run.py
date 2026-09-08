"""One long source, many concurrent watches — long-horizon qualification.

``live_run.py`` measures per-stage recall and end-to-end latency on ~100 SHORT clips, each with ONE
goal armed, each in its own process-lifetime ``Percept``.  Everything it measures is per-clip, and
every piece of state it touches is born and dies inside 30-300 seconds.  Five things the runtime
actually claims are therefore untested by it, and untestable by it in principle:

1. **Entity tracks** — ``percept/entity.py``.  ``max_entities=64`` is an eviction bound.  A clip that
   never mints 64 entities cannot tell you whether the bound is reached, whether eviction is
   drop-oldest-by-``last_seen`` as designed, or whether tracks fragment as the scene churns.
2. **Episode state** — ``percept/episodes.py``.  ``Settings.episodes.window_s`` defaults to 3600 s.
   No clip in the corpus is longer than 6 minutes, so no prior test has ever run PAST the window.
3. **The event ledger** — admitted + suppressed-with-receipt + dropped-with-receipt must account for
   every unit.  A prior suite found the primary ledger reconciles exactly while a ``funnel`` mirror
   did not; on 40 000 units the disagreement is measurable rather than anecdotal.
4. **Judge allocation** — with ONE goal there is no allocation question.  With many goals sharing
   one scheduler there is nothing BUT an allocation question.
5. **Bounded retention** — RSS over 80 minutes, not the peak of a 40-second clip.

So this file runs the opposite shape: one manifest-selected long source, every selected watch armed
simultaneously on ONE core, one paced real-time replay, one live Gemini judge behind the shipped
``JudgePool``.  It accepts both the retained L2 manifest and the relocated two-hour injected L3
manifest; the latter carries occurrence ground truth for recall/latency but explicitly unlabelled
background, so precision remains ineligible.

Declared deviations from ``live_run.py`` — each one exists because the long shape forces it
==========================================================================================

``--decision-deadline-s``
    The per-admission decision deadline in ``live_run`` is 1.9 s, sized for ONE goal.  A judgment
    batch judges its admitted goals SEQUENTIALLY (``percept/agent.py:4090`` — ``for goal in
    active_goals: ... await``), so an M-goal batch needs ~M x the provider latency.  Held at 1.9 s
    the batch is cancelled wholesale (``percept/runtime/worker.py:1120-1143``) and the run emits
    nothing at all — which is a finding, and is reported as one, but is not a measurement of
    anything downstream.  The deadline is therefore scaled to the armed-goal count and the scaling
    is printed with every number.

``--max-provider-calls``
    One bound in a durable global call-and-USD ledger enforced inside the provider adapter.  Every
    attempt is reserved on disk before transport; exhausted attempts fail explicitly without a
    synthetic verdict, and therefore cannot be silently absorbed into quality metrics.

entity observations
    The runtime feeds ``EntityGraph`` from ``vision.detect`` (``percept/agent.py:4084``), and the
    live vision backend's ``detect`` is a per-frame VLM object-detection call — thousands of extra
    provider calls over 80 minutes, which the budget does not buy.  This harness instead feeds the
    graph from a LOCAL motion-blob detector at frame rate, through the same public call site the
    shipped ``percept/rhythm_hero.py:67`` uses (``core.graph.ingest(obs, frame=frame, t=t)``).  The
    ASSOCIATION logic under test — IoU + label continuity, predictive re-link, eviction — is the
    shipped one, unmodified.  What is harness-supplied is the observation stream, and that is stated
    on every entity number in the report.

episode probes
    The armed watches are event-chain watches (``rearm='each_event'``), and
    ``percept/agent.py:2299`` returns ``None`` from ``_document_episode_fold`` for exactly that
    shape, so a watch goal NEVER opens an episode.  That is finding #2 and is reported first.  To
    also measure the episode machinery itself across the 3600 s window boundary, the harness drives
    the public counting API (``Percept.count_event``, ``percept/agent.py:1639``) against probe logs
    keyed off unregistered goal ids — real ``EpisodeLog`` objects, real ``Settings.episodes`` bounds,
    zero interaction with the armed goals' gates.

Everything else is held to ``live_run``: same pacing discipline (lossless bounded queue, one clock
crossing at the replay anchor), same ``JudgePool`` shape, same reflex latency mode and 1.5 s provider
deadline, same tier-0 operating point, same ``block_max`` salience.

Run it::

    set -a; . .env; set +a
    PERCEPT_VISION_BACKEND=gemini PERCEPT_VISION_MODEL=gemini-3.1-flash-lite \\
      <venv>/bin/python eval/stage-validation/long_run.py

Writes ``results/long_run.{json,md}`` and an append-only ``results/long_run.checkpoint.jsonl`` that
carries the whole timeline sample-by-sample, so a host restart costs the tail of the replay rather
than the run.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import copy
import json
import math
import os
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))

import live_run as LR  # noqa: E402  -- pacing, watch-plan shape and constants are REUSED
import qualification_contract as QC  # noqa: E402

from percept import Percept  # noqa: E402
from percept._build import build_info  # noqa: E402
from percept.adapters import Box, Observation  # noqa: E402
from percept.backends.judge_pool import (  # noqa: E402
    LATENCY_MODE_CONTEXT_KEY,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.config import load_settings  # noqa: E402
from percept.episodes import EpisodeLog  # noqa: E402
from percept.incidents import incident_judgment  # noqa: E402
from percept.resolver import ConfigResolver  # noqa: E402
from percept.runtime import RuntimeStats, TemporalImageVision  # noqa: E402
from percept.runtime.source import open_source  # noqa: E402
from percept.runtime.worker import run as worker_run  # noqa: E402
from percept.sensorplan import sensor_plan_for  # noqa: E402
from percept.source_profiles_default import PHYSICAL_SCENE  # noqa: E402
from percept.watchplan_apply import apply_watch_plan  # noqa: E402

MANIFEST = REPO / "eval" / "stage-validation" / "long_sources" / "long_run_all.manifest.json"

#: Blob extraction geometry.  Deliberately coarse: the point is a stable, cheap, frame-rate
#: observation stream for the entity graph, not a detector benchmark.
BLOB_W, BLOB_H = 160, 120
BLOB_DIFF_THRESHOLD = 18
BLOB_MIN_AREA = 24  # px of BLOB_W*BLOB_H = 19 200
BLOB_MAX_PER_FRAME = 8


# =================================================================================================
# provider wrapper -- physical-call accounting plus a HARD spend cap
# =================================================================================================


class BudgetedVision:
    """Record the adapter whose transport is guarded by the durable global USD/call ledger.

    ``GeminiVision`` owns the only safe reservation seam: it calls ``begin_attempt`` immediately
    before transport and settles/fails that reservation on every terminal path.  This wrapper is
    therefore observational only.  In particular, it must never replace an exhausted request with
    a synthetic verdict, because doing so hides the budget terminal outcome inside quality metrics.
    """

    def __init__(
        self,
        inner: Any,
        *,
        request_budget: LR.GlobalProviderBudget,
        on_cap: Any = None,
    ) -> None:
        self.inner = inner
        self.request_budget = request_budget
        #: Fired ONCE, the instant the cap first bites, so the pre-cap window (in which the judge was
        #: a real provider) can be reported separately from the post-cap window (in which it was not).
        self.on_cap = on_cap
        self.capped_at_wall: float | None = None
        self.SUPPORTS_NATIVE_VIDEO_TRANSPORT = bool(
            getattr(inner, "SUPPORTS_NATIVE_VIDEO_TRANSPORT", False)
        )
        self.PARSES_CLAIM_CHANNEL = bool(getattr(inner, "PARSES_CLAIM_CHANNEL", False))
        self.budget_capped = 0
        self.errors = 0
        self.cancelled = 0
        self.latencies_completed: list[float] = []
        self.latencies_censored: list[float] = []
        self.reflex_stamped = 0
        self.evidence_nonempty = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_rows = 0
        self.input_bytes = 0
        self.error_types: Counter = Counter()
        #: (wall_since_start, cumulative_calls) so spend can be plotted against the horizon.
        self.call_marks: list[tuple[float, int]] = []
        self._t0 = time.monotonic()

    @property
    def calls(self) -> int:
        """Physical attempts durably admitted by the request ledger (not judge invocations)."""

        return int(self.request_budget.snapshot()["physical_attempts"])

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any):
        context = dict(kwargs.get("context") or {})
        if context.get(LATENCY_MODE_CONTEXT_KEY) == "reflex":
            self.reflex_stamped += 1
        payload = frame[1] if isinstance(frame, tuple) and len(frame) == 2 else frame
        if isinstance(payload, (bytes, bytearray)):
            self.input_bytes += len(payload)
        for f in kwargs.get("frames") or []:
            fp = f[1] if isinstance(f, tuple) and len(f) == 2 else f
            if isinstance(fp, (bytes, bytearray)):
                self.input_bytes += len(fp)
        t0 = time.monotonic()
        try:
            verdict = await self.inner.judge(condition, frame, **kwargs)
        except LR.ProviderBudgetExceeded:
            self.budget_capped += 1
            if self.capped_at_wall is None:
                self.capped_at_wall = time.monotonic()
                if self.on_cap is not None:
                    try:
                        self.on_cap()
                    except Exception:
                        pass
            self.latencies_censored.append(time.monotonic() - t0)
            raise
        except asyncio.CancelledError:
            # CancelledError is a BaseException: an `except Exception` here would report a run that
            # burned thousands of calls as having made none.
            self.cancelled += 1
            self.latencies_censored.append(time.monotonic() - t0)
            raise
        except Exception as exc:
            self.errors += 1
            self.error_types[type(exc).__name__] += 1
            self.latencies_censored.append(time.monotonic() - t0)
            raise
        self.latencies_completed.append(time.monotonic() - t0)
        usage = dict(getattr(verdict, "usage", None) or {})
        if usage:
            self.usage_rows += 1
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
        if str(getattr(verdict, "evidence", "") or "").strip():
            self.evidence_nonempty += 1
        if self.calls and self.calls % 50 == 0:
            self.call_marks.append((round(time.monotonic() - self._t0, 1), self.calls))
        return verdict


# =================================================================================================
# the local observation stream that makes the entity graph measurable
# =================================================================================================


class MotionBlobDetector:
    """Frame-differencing blob detector — a cheap, deterministic observation stream.

    It is NOT a semantic detector and nothing in this report claims it is.  Its job is to hand the
    shipped :class:`~percept.entity.EntityGraph` a realistic, noisy, frame-rate stream of moving
    boxes over 80 minutes so that association, re-link, fragmentation and the ``max_entities``
    eviction bound are exercised at all.  One label for every blob, so matching is pure IoU
    continuity (``percept.identity.ContinuityIdentity``) with no label-driven shortcut.
    """

    LABEL = "motion_blob"

    def __init__(self) -> None:
        import cv2  # noqa: F401  -- the [runtime] extra; the source decoder already requires it

        self._cv2 = cv2
        self._prev = None
        self.frames_seen = 0
        self.frames_with_blobs = 0
        self.observations = 0

    def _gray(self, payload: Any):
        import numpy as np

        cv2 = self._cv2
        data = payload[1] if isinstance(payload, tuple) and len(payload) == 2 else payload
        if isinstance(data, (bytes, bytearray)):
            buf = np.frombuffer(bytes(data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return cv2.resize(img, (BLOB_W, BLOB_H))
        arr = np.asarray(data)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim != 2:
            return None
        return cv2.resize(arr.astype("uint8"), (BLOB_W, BLOB_H))

    def observe(self, payload: Any) -> list[Observation]:
        cv2 = self._cv2
        import numpy as np

        gray = self._gray(payload)
        self.frames_seen += 1
        if gray is None:
            return []
        prev, self._prev = self._prev, gray
        if prev is None:
            return []
        diff = cv2.absdiff(gray, prev)
        _, mask = cv2.threshold(diff, BLOB_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < BLOB_MIN_AREA:
                continue
            blobs.append((area, x, y, w, h))
        blobs.sort(reverse=True)
        out = [
            Observation(
                label=self.LABEL,
                box=Box(x / BLOB_W, y / BLOB_H, w / BLOB_W, h / BLOB_H),
                confidence=min(1.0, area / 600.0),
                attrs={},
            )
            for area, x, y, w, h in blobs[:BLOB_MAX_PER_FRAME]
        ]
        if out:
            self.frames_with_blobs += 1
        self.observations += len(out)
        return out


@dataclass
class EntityTelemetry:
    """Track lifecycle, reconstructed from what the graph RETURNS per ingest.

    ``EntityGraph.ingest`` returns the entities the observations were associated to, in observation
    order, so every ingest tells us exactly which ids were touched.  From the id timeline:

    * **creation** — an id seen for the first time.
    * **re-identification** — an id touched again after a gap longer than ``coast_s`` (the graph's own
      predictive-coast horizon, 1.0 s by default): normal continuity could not have carried it, so the
      re-link came from IoU against a stale box or from the predictive path.
    * **fragmentation** — a NEW id minted in a frame where the number of observations did not grow,
      i.e. the same physical thing acquired a second identity.
    * **eviction** — reported by the graph itself through ``on_evict``.
    * **resurrection-after-eviction** — an evicted id touched again.  The graph mints monotonically,
      so this must never happen; it is checked rather than assumed.
    """

    first_seen: dict[str, float] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)
    touches: Counter = field(default_factory=Counter)
    creations: int = 0
    reids: int = 0
    fragmentations: int = 0
    evictions: int = 0
    evicted_ids: set = field(default_factory=set)
    resurrections: int = 0
    label_switches: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    gap_s: list[float] = field(default_factory=list)
    ingests: int = 0
    observations: int = 0
    #: (media_t, live_entities, cumulative_mints) sampled every ingest-batch boundary.
    size_marks: list[tuple[float, int, int]] = field(default_factory=list)
    _prev_obs: int = 0

    def record(self, t: float, entities: list, n_obs: int, coast_s: float) -> None:
        self.ingests += 1
        self.observations += n_obs
        minted_here = 0
        for e in entities:
            eid = e.id
            if eid not in self.first_seen:
                self.first_seen[eid] = t
                self.creations += 1
                minted_here += 1
                if eid in self.evicted_ids:
                    self.resurrections += 1
            else:
                gap = t - self.last_seen[eid]
                if gap > coast_s:
                    self.reids += 1
                    self.gap_s.append(gap)
            prior = self.labels.get(eid)
            if prior is not None and prior != e.label:
                self.label_switches += 1
            self.labels[eid] = e.label
            self.last_seen[eid] = t
            self.touches[eid] += 1
        if minted_here and n_obs <= self._prev_obs:
            self.fragmentations += minted_here
        self._prev_obs = n_obs

    def on_evict(self, entity_id: str) -> None:
        self.evictions += 1
        self.evicted_ids.add(entity_id)


# =================================================================================================
# pacing (structurally live_run.paced_source, plus the per-frame entity feed)
# =================================================================================================


@dataclass
class Channels:
    frames_yielded: int = 0
    anchor_media: float | None = None
    anchor_wall: float | None = None
    last_media: float | None = None
    pace_lag_s: list[float] = field(default_factory=list)
    #: (wall_since_anchor, media_since_anchor) every 200 frames — the replay's own realtime proof.
    pace_marks: list[tuple[float, float]] = field(default_factory=list)


async def paced_source(
    path: str,
    *,
    fps: float,
    ch: Channels,
    dropped: list[int],
    detector: MotionBlobDetector | None,
    graph: Any,
    telemetry: EntityTelemetry | None,
    coast_s: float,
    max_wall_s: float = 0.0,
):
    """Real-time replay with a LOSSLESS bounded queue, plus the harness's entity feed.

    Identical in structure to ``live_run.paced_source`` (same queue policy, same media->monotonic
    anchor, same per-frame lag record).  The one addition is the entity feed: every yielded frame is
    run through the local blob detector and ingested into the graph on the SOURCE clock, which is the
    call shape ``percept/rhythm_hero.py:67`` uses.
    """
    started_wall: float | None = None
    first_media: float | None = None
    async for envelope in open_source(
        path, fps=fps, maxsize=8, queue_policy="block", timebase="media", dropped=dropped
    ):
        if envelope.source_time is None:
            raise ValueError("paced replay requires decoder source timestamps")
        media_t = float(envelope.source_time)
        if first_media is None:
            first_media = media_t
            started_wall = time.monotonic()
            ch.anchor_media, ch.anchor_wall = media_t, started_wall
        assert started_wall is not None
        if max_wall_s and (time.monotonic() - started_wall) > max_wall_s:
            # A clean end-of-source, so the runtime drains and emits `runtime_ended` normally rather
            # than being torn down mid-judgment (which would corrupt the ledger reconciliation).
            break
        target = started_wall + (media_t - first_media)
        remaining = target - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        else:
            ch.pace_lag_s.append(-remaining)
        ch.frames_yielded += 1
        ch.last_media = media_t
        if ch.frames_yielded % 200 == 0:
            ch.pace_marks.append(
                (round(time.monotonic() - started_wall, 2), round(media_t - first_media, 2))
            )
        if detector is not None:
            try:
                obs = detector.observe(envelope.payload)
                if obs:
                    entities = graph.ingest(obs, frame=None, source="camera", t=media_t)
                    if telemetry is not None:
                        telemetry.record(media_t, entities, len(obs), coast_s)
                        if ch.frames_yielded % 400 == 0:
                            telemetry.size_marks.append(
                                (round(media_t, 1), len(graph), telemetry.creations)
                            )
            except Exception as exc:  # a detector fault must never take the measurement down
                if telemetry is not None and telemetry.ingests == 0:
                    print(f"[long_run] entity feed failed: {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
        yield envelope


# =================================================================================================
# helpers
# =================================================================================================


def rss_bytes() -> int | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip()) * 1024
    except Exception:
        return None


def goal_id_for(index: int, text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")[:40]
    return f"g{index:03d}-{slug}" if slug else f"g{index:03d}"


def normalize_long_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize both retained L2 and relocated injected-L3 manifests for one runner.

    The injected fixture deliberately calls its arm list ``watches`` and its source tiling
    ``segments``; the older runner expected ``goals`` and ``members``.  Keeping the translation at
    one fail-loud boundary prevents a second, subtly different long-horizon runner from growing
    beside this one.
    """
    kind = payload.get("kind")
    if kind == "stage-validation-single-long-source":
        out = copy.deepcopy(payload)
        out.setdefault("source_manifest_kind", kind)
        # The retained L2 source does not claim exhaustively labelled background.
        out.setdefault("precision_eligible", False)
        out.setdefault(
            "precision_ineligible_reason",
            "source does not declare exhaustive goal-specific negative/background labels",
        )
        return out
    if kind != "longhorizon-injected-source":
        raise ValueError(f"unsupported long-run manifest kind {kind!r}")

    out = copy.deepcopy(payload)
    goals: list[dict[str, Any]] = []
    for index, watch in enumerate(payload.get("watches") or []):
        datasets = [str(x) for x in (watch.get("datasets") or [])]
        goals.append(
            {
                "case_ids": list(watch.get("injection_ids") or []),
                "case_id": f"injected-watch::{index}",
                "dataset": ",".join(datasets) if datasets else "unknown",
                "goal_kind": watch.get("goal_kind"),
                "goal_text": watch.get("goal_text"),
                "gt_strength": watch.get("gt_strength"),
                "n_occurrences": int(watch.get("n_occurrences") or 0),
                "positives": copy.deepcopy(watch.get("positives") or []),
            }
        )
    clock = copy.deepcopy(payload.get("clock") or {})
    clock["measured_container_duration_s"] = payload.get("container_duration_s")
    clock["drift_s"] = payload.get("drift_s")
    segments = copy.deepcopy(payload.get("segments") or [])
    unlabelled = [s for s in segments if s.get("gt") == "none"]

    out.update(
        source_manifest_kind=kind,
        goals=goals,
        n_goals=len(goals),
        members=segments,
        n_members=len(segments),
        clock=clock,
        precision_eligible=not unlabelled,
        precision_ineligible_reason=(
            None
            if not unlabelled
            else f"{len(unlabelled)} background segment(s) are explicitly unlabelled, not negative"
        ),
    )
    return out


def long_manifest_issues(manifest: dict[str, Any]) -> list[str]:
    """Return schema/clock/label defects that make a long run unscoreable."""
    issues: list[str] = []
    video = manifest.get("video_path")
    if not video:
        issues.append("video_path is missing")
    else:
        path = Path(str(video))
        resolved = path if path.is_absolute() else REPO / path
        if not resolved.is_file():
            issues.append(f"video_path does not exist: {resolved}")

    duration = (manifest.get("clock") or {}).get("measured_container_duration_s")
    try:
        duration_f = float(duration)
    except (TypeError, ValueError):
        duration_f = 0.0
    if not math.isfinite(duration_f) or duration_f <= 0:
        issues.append(f"measured container duration must be finite and positive, got {duration!r}")

    goals = manifest.get("goals") or []
    if not goals:
        issues.append("manifest contains no goals/watches")
    for index, goal in enumerate(goals):
        prefix = str(goal.get("case_id") or f"goal[{index}]")
        if not str(goal.get("goal_text") or "").strip():
            issues.append(f"{prefix}: goal_text is missing")
        if goal.get("goal_kind") not in {"state", "transition", "duration", "count"}:
            issues.append(f"{prefix}: unsupported goal_kind {goal.get('goal_kind')!r}")
        positives = goal.get("positives") or []
        if int(goal.get("n_occurrences") or 0) != len(positives):
            issues.append(
                f"{prefix}: n_occurrences={goal.get('n_occurrences')} but positives={len(positives)}"
            )
        for occurrence_index, interval in enumerate(positives):
            try:
                lo, hi = float(interval[0]), float(interval[1])
            except (TypeError, ValueError, IndexError):
                issues.append(f"{prefix}: occurrence[{occurrence_index}] is not a numeric interval")
                continue
            ordered = lo < hi if goal.get("gt_strength") == "annotated_interval" else lo <= hi
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo and ordered and hi <= duration_f):
                issues.append(
                    f"{prefix}: occurrence[{occurrence_index}]={interval!r} outside [0,{duration_f}]"
                )
    return issues


async def preflight_long_manifest(
    manifest: dict[str, Any], *, require_conductor: bool
) -> list[Any | None]:
    """Validate source/labels and, when requested, every authored mechanic before provider setup."""
    issues = long_manifest_issues(manifest)
    if issues:
        raise ValueError("; ".join(issues[:20]))
    goals = list(manifest["goals"])
    if not require_conductor:
        return [None] * len(goals)
    if not LR.ACTION_CONDUCTOR_PIN:
        raise ValueError("--expect-conductor requires PERCEPT_SV_CONDUCTOR=<frozen artifact>")
    cases = [
        {
            "case_id": goal.get("case_id") or goal_id_for(i, str(goal["goal_text"])),
            "goal_text": goal["goal_text"],
            "goal_kind": goal["goal_kind"],
        }
        for i, goal in enumerate(goals)
    ]
    return list(await LR.preflight_authored_plans(cases))


def long_armability_issues(
    goals: Sequence[dict[str, Any]],
    authored: Sequence[Any | None],
    *,
    settings: Any = None,
) -> list[str]:
    """Dry-arm the exact multi-goal shape on an offline core before provider construction."""

    if len(goals) != len(authored):
        return [f"authored plan count {len(authored)} != selected goal count {len(goals)}"]
    resolved_settings = settings or load_settings()
    core = Percept.create(
        vision="scripted",
        specialist="scripted",
        conductor="scripted-conductor",
        stt="fake",
        tts="fake",
        settings=resolved_settings,
        discover_plugins=False,
        mode="offline",
    )
    issues: list[str] = []
    for index, (goal, primitive) in enumerate(zip(goals, authored)):
        goal_id = goal_id_for(index, str(goal["goal_text"]))
        receipt = apply_watch_plan(
            core,
            LR.watch_plan_for(goal_id, str(goal["goal_text"]), primitive),
            resolver=ConfigResolver(),
            source_profile=PHYSICAL_SCENE,
        )
        if receipt.status == "applied" and receipt.watch_ids == (goal_id,):
            continue
        dropped = [
            f"{item.code} at {item.path}: {item.message}" for item in receipt.dropped_fields
        ]
        issues.append(
            f"{goal.get('case_id') or goal_id}: status={receipt.status}; "
            f"watch_ids={list(receipt.watch_ids)!r}; "
            f"warnings={[item.code for item in receipt.warnings]!r}; dropped_fields={dropped!r}"
        )
    return issues


def provider_budget_policy_from_args(args: Any) -> Any:
    """Require an explicit paid allocation and prove it fits the user's absolute $100 cap."""

    required = {
        "--max-provider-usd": args.max_provider_usd,
        "--max-provider-calls": args.max_provider_calls,
        "--conservative-usd-per-call": args.conservative_usd_per_call,
        "--global-prior-committed-usd": args.global_prior_committed_usd,
        "--global-prior-commitment-source": args.global_prior_commitment_source,
        "--global-prior-commitment-basis": args.global_prior_commitment_basis,
    }
    missing = [flag for flag, value in required.items() if value is None or value == ""]
    if missing:
        raise LR.ProviderBudgetInputError(
            "paid long run requires explicit " + ", ".join(missing)
        )
    if (args.price_in is None) != (args.price_out is None):
        raise LR.ProviderBudgetInputError("--price-in and --price-out must be supplied together")
    if args.settle_from_official_token_prices and (
        args.price_in is None or not args.price_source.strip() or not args.price_version.strip()
    ):
        raise LR.ProviderBudgetInputError(
            "--settle-from-official-token-prices requires --price-in, --price-out, "
            "--price-source, and --price-version"
        )
    policy = LR.PaidRunPolicy(
        max_usd=args.max_provider_usd,
        max_calls=args.max_provider_calls,
        conservative_usd_per_call=args.conservative_usd_per_call,
        request_timeout_s=args.provider_request_timeout_s,
    )
    prior = args.global_prior_committed_usd
    if not isinstance(prior, (int, float)) or isinstance(prior, bool):
        raise LR.ProviderBudgetInputError(
            "--global-prior-committed-usd must be finite and non-negative"
        )
    prior = float(prior)
    if not math.isfinite(prior) or prior < 0:
        raise LR.ProviderBudgetInputError(
            "--global-prior-committed-usd must be finite and non-negative"
        )
    if prior + float(policy.max_usd) > LR.ABSOLUTE_PROVIDER_RUN_CAP_USD + 1e-12:
        raise LR.ProviderBudgetInputError(
            f"prior commitment (${prior:.6f}) plus long-run allocation "
            f"(${policy.max_usd:.6f}) exceeds the "
            f"${LR.ABSOLUTE_PROVIDER_RUN_CAP_USD:.2f} total user cap"
        )
    return policy


def dist(xs: Sequence[float]) -> dict[str, Any]:
    return LR.dist(xs)


def score_long_quality(
    armed: Sequence[dict[str, Any]],
    incidents: Sequence[dict[str, Any]],
    *,
    anchor_media: float | None,
    precision_eligible: bool,
    precision_ineligible_reason: str | None,
    delivery_enabled: bool = False,
) -> dict[str, Any]:
    """Score labeled long-horizon occurrences at SDK and webhook-receiver boundaries.

    The earliest Incident for one ``(goal_id, occurrence)`` owns its latency sample; later repeats
    remain emissions and therefore affect precision when the source has exhaustive negatives. The
    receiver view independently takes the earliest successfully delivered emission. A zero-width
    legacy point annotation is named unscorable rather than silently given an invented tolerance.
    """
    by_goal = {str(goal["goal_id"]): goal for goal in armed}
    emitted_by_goal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for incident in incidents:
        emitted_by_goal[str(incident.get("goal_id"))].append(incident)

    occurrence_rows: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    inside_emissions: set[int] = set()
    inside_receiver_emissions: set[int] = set()
    for goal_id, goal in by_goal.items():
        goal_incidents = emitted_by_goal.get(goal_id, [])
        for occurrence_index, interval in enumerate(goal.get("positives") or []):
            lo, hi = float(interval[0]), float(interval[1])
            if hi <= lo:
                unscorable.append(
                    {
                        "goal_id": goal_id,
                        "occurrence_index": occurrence_index,
                        "interval": [lo, hi],
                        "reason": "point annotation has no declared matching tolerance",
                    }
                )
                continue
            hits = [item for item in goal_incidents if lo <= float(item.get("ts") or 0.0) < hi]
            for item in hits:
                inside_emissions.add(id(item))
            receiver_hits = [
                item
                for item in hits
                if (item.get("webhook") or {}).get("receiver_arrival_since_anchor_s") is not None
            ]
            for item in receiver_hits:
                inside_receiver_emissions.add(id(item))
            earliest = min(
                hits,
                key=lambda item: (
                    float(item["observed_since_anchor_s"])
                    if item.get("observed_since_anchor_s") is not None
                    else math.inf
                ),
                default=None,
            )
            earliest_receiver = min(
                receiver_hits,
                key=lambda item: float(
                    (item.get("webhook") or {})["receiver_arrival_since_anchor_s"]
                ),
                default=None,
            )
            latency = None
            latency_valid = False
            if earliest is not None and anchor_media is not None:
                observed = earliest.get("observed_since_anchor_s")
                if observed is not None:
                    latency = float(observed) - (lo - float(anchor_media))
                    latency_valid = math.isfinite(latency) and latency >= 0.0
            occurrence_rows.append(
                {
                    "goal_id": goal_id,
                    "goal_kind": goal.get("goal_kind"),
                    "dataset": goal.get("dataset"),
                    "occurrence_index": occurrence_index,
                    "interval": [lo, hi],
                    "matched": earliest is not None,
                    "first_incident_id": None if earliest is None else earliest.get("id"),
                    "emissions_in_occurrence": len(hits),
                    "onset_to_incident_s": round(latency, 6) if latency_valid else None,
                    "latency_invalid_reason": (
                        None
                        if earliest is None or latency_valid
                        else "negative/non-finite source-to-monotonic clock crossing"
                    ),
                    "receiver_matched": earliest_receiver is not None,
                    "first_receiver_incident_id": (
                        None if earliest_receiver is None else earliest_receiver.get("id")
                    ),
                    "receiver_emissions_in_occurrence": len(receiver_hits),
                    "onset_to_webhook_arrival_s": (
                        None
                        if earliest_receiver is None or anchor_media is None
                        else round(
                            float(
                                (earliest_receiver.get("webhook") or {})[
                                    "receiver_arrival_since_anchor_s"
                                ]
                            )
                            - (lo - float(anchor_media)),
                            6,
                        )
                    ),
                }
            )

    scoreable = len(occurrence_rows)
    matched = sum(1 for row in occurrence_rows if row["matched"])
    valid_latencies = [
        float(row["onset_to_incident_s"])
        for row in occurrence_rows
        if row.get("onset_to_incident_s") is not None
    ]
    emissions = [item for values in emitted_by_goal.values() for item in values]
    inside = sum(1 for item in emissions if id(item) in inside_emissions)
    receiver_emissions = [
        item
        for item in emissions
        if (item.get("webhook") or {}).get("receiver_arrival_since_anchor_s") is not None
    ]
    receiver_inside = sum(
        1 for item in receiver_emissions if id(item) in inside_receiver_emissions
    )

    def summary(
        rows: Sequence[dict[str, Any]],
        *,
        matched_field: str = "matched",
        latency_field: str = "onset_to_incident_s",
        latency_name: str = "onset_to_incident_s",
    ) -> dict[str, Any]:
        n = len(rows)
        hit = sum(1 for row in rows if row[matched_field])
        latencies = [
            float(row[latency_field])
            for row in rows
            if row.get(latency_field) is not None and float(row[latency_field]) >= 0.0
        ]
        return {
            "gt_occurrences": n,
            "occurrences_matched": hit,
            "occurrences_missed": n - hit,
            "occurrence_recall": (hit / n) if n else None,
            latency_name: dist(latencies),
            "timely_recall": {
                "2s": (sum(x <= 2.0 for x in latencies) / n) if n else None,
                "5s": (sum(x <= 5.0 for x in latencies) / n) if n else None,
            },
        }

    pooled = summary(occurrence_rows)
    pooled.update(
        incidents=len(emissions),
        incidents_inside_labeled_positive=inside,
        precision_per_emission=(inside / len(emissions))
        if precision_eligible and emissions
        else None,
        precision_eligible=precision_eligible,
        precision_ineligible_reason=None
        if precision_eligible
        else precision_ineligible_reason,
        latency_valid_occurrences=len(valid_latencies),
    )
    receiver = summary(
        occurrence_rows,
        matched_field="receiver_matched",
        latency_field="onset_to_webhook_arrival_s",
        latency_name="onset_to_webhook_arrival_s",
    )
    receiver.update(
        enabled=delivery_enabled,
        incidents=len(receiver_emissions),
        incidents_inside_labeled_positive=receiver_inside,
        precision_per_emission=(receiver_inside / len(receiver_emissions))
        if precision_eligible and receiver_emissions
        else None,
        precision_eligible=precision_eligible,
        precision_ineligible_reason=None
        if precision_eligible
        else precision_ineligible_reason,
    )
    kinds = sorted({str(row.get("goal_kind")) for row in occurrence_rows})
    return {
        "schema": "percept.longhorizon-quality/1",
        "matching": "same goal_id; Incident media timestamp inside [onset,end); earliest hit owns latency",
        "pooled": pooled,
        "receiver": receiver,
        "by_goal_kind": {
            kind: summary([row for row in occurrence_rows if str(row.get("goal_kind")) == kind])
            for kind in kinds
        },
        "unscorable_occurrences": unscorable,
        "occurrences": occurrence_rows,
    }


# =================================================================================================
# the run
# =================================================================================================


async def run_long(args: argparse.Namespace) -> dict[str, Any]:
    manifest = getattr(args, "_prepared_manifest", None)
    if manifest is None:
        manifest = normalize_long_manifest(json.loads(Path(args.manifest).read_text()))
    source_path = str(REPO / manifest["video_path"])
    goals_in = manifest["goals"][: args.max_goals] if args.max_goals else manifest["goals"]
    authored_in = getattr(args, "_prepared_authored", None)
    if authored_in is None:
        authored_in = await preflight_long_manifest(
            {**manifest, "goals": goals_in}, require_conductor=args.expect_conductor
        )
    provider_budget = getattr(args, "_provider_budget", None)
    if not isinstance(provider_budget, LR.GlobalProviderBudget):
        raise LR.ProviderBudgetInputError(
            "durable provider budget was not opened before long-run provider setup"
        )
    webhook_probe = getattr(args, "_webhook_probe", None)
    member_offsets = [m["offset_s"] for m in manifest["members"]]

    settings = load_settings()
    from percept.backends.gemini import GeminiVision

    # Same process-local pin live_run.py uses, so both v1.0 runs can be read at the same vision-token
    # budget. Unset = the provider default, which is what every earlier long run measured.
    media_resolution = LR.install_media_resolution_override(LR.ACTION_MEDIA_RESOLUTION)
    if media_resolution:
        print(f"[long_run] media_resolution pinned to {media_resolution}", file=sys.stderr,
              flush=True)

    inner = GeminiVision(settings, request_budget=provider_budget)
    model = getattr(inner, "_model", None) or getattr(inner, "model", None)
    provider_budget.assert_provider(backend="gemini", model=str(model))
    cap_snapshot: dict[str, Any] = {}
    provider = BudgetedVision(
        inner, request_budget=provider_budget,
        on_cap=lambda: cap_snapshot.update(snapshot_sample(), kind="cap_reached"),
    )
    temporal = TemporalImageVision(provider, max_frames=min(LR.ACTION_TEMPORAL_FRAMES, 5))

    receipt_outcomes: Counter = Counter()
    receipt_latency: list[float] = []

    def on_receipt(r: Any) -> None:
        receipt_outcomes[r.outcome] += 1
        if r.route is not None and r.outcome in ("decisive", "uncertain"):
            receipt_latency.append(float(r.latency_s))

    # ---- judge concurrency K ---------------------------------------------------------------
    # K is the batch fan-out bound. It has to be declared in THREE places that all floor each
    # other (percept/agent.py:effective_judge_concurrency): the caller's ceiling
    # (``judge_concurrency``), the pool's own ``max_parallel``, and every route's declared
    # ``max_inflight``. Setting only one leaves the batch pinned at the smallest of them, which is
    # how the previous run ended up sequential at K=1 while believing otherwise.
    #
    # K also SETS THE BACKSTOP. ``_WorkerJudgeScheduler._batch_deadline_s`` multiplies the
    # per-decision deadline by ``ceil(N / K)`` waves, so at K >= N the multiplier collapses to 1 and
    # the batch backstop IS the shipped per-decision number, unscaled. Raising K is therefore the
    # only honest way to shrink that bound -- the alternative (hand-scaling the deadline) inflates a
    # budget to paper over a fan-out that never happened.
    K = max(1, int(args.judge_concurrency))

    pool = JudgePool(
        [
            JudgeRoute(
                "timestamped-images-primary",
                temporal,
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=LR.ACTION_PROVIDER_DEADLINE_S,
                    modes=frozenset({"reflex"}),
                    max_inflight=K,
                ),
            )
        ],
        mode="reflex",
        deadline_s=LR.ACTION_PROVIDER_DEADLINE_S,
        max_parallel=K,
        on_receipt=on_receipt,
        terminal_outcomes=(),  # a measurement run COUNTS a deadline, it does not raise on one
    )

    core = Percept.create(
        vision=pool, specialist="scripted", conductor="scripted-conductor",
        stt="fake", tts="fake", settings=settings, discover_plugins=False,
        judge_concurrency=K,
    )

    # ---- arm every distinct goal on the SAME core ------------------------------------------
    armed: list[dict[str, Any]] = []
    arm_failures: list[dict[str, Any]] = []
    for i, g in enumerate(goals_in):
        gid = goal_id_for(i, g["goal_text"])
        receipt = apply_watch_plan(
            core, LR.watch_plan_for(gid, g["goal_text"], authored_in[i]),
            resolver=ConfigResolver(), source_profile=PHYSICAL_SCENE,
        )
        if receipt.status != "applied" or receipt.watch_ids != (gid,):
            arm_failures.append({"goal_id": gid, "goal_text": g["goal_text"],
                                 "status": receipt.status,
                                 "warnings": [w.code for w in receipt.warnings]})
            continue
        armed.append({"goal_id": gid, "goal_text": g["goal_text"], "goal_kind": g["goal_kind"],
                      "dataset": g["dataset"], "n_occurrences": g["n_occurrences"],
                      "positives": g["positives"]})
    if args.expect_conductor and arm_failures:
        raise SystemExit(
            f"mechanics-qualified arm refused {len(arm_failures)} selected goal(s) before provider "
            f"calls; first={arm_failures[0]!r}"
        )
    if not armed:
        raise SystemExit("no goals armed -- refusing to run")

    goal_ids = [a["goal_id"] for a in armed]
    goal_text_by_id = {a["goal_id"]: a["goal_text"] for a in armed}

    fps = max(
        LR.capture_fps(sensor_plan_for(core.goals.get(a["goal_id"])).cadence) for a in armed
    )
    decision_deadline = (
        args.decision_deadline_s
        if args.decision_deadline_s > 0
        else max(LR.ACTION_DECISION_DEADLINE_S, len(armed) * LR.ACTION_DECISION_DEADLINE_S)
    )

    # ---- entity feed -------------------------------------------------------------------------
    telemetry = EntityTelemetry()
    prior_evict = getattr(core.graph, "on_evict", None)

    def chained_evict(entity_id: str) -> None:
        telemetry.on_evict(entity_id)
        if prior_evict is not None:
            prior_evict(entity_id)

    core.graph.on_evict = chained_evict
    detector = MotionBlobDetector() if args.entity_feed else None
    coast_s = float(getattr(core.graph, "_coast_s", 1.0))

    # ---- episode probes ------------------------------------------------------------------------
    # Real EpisodeLog objects with the shipped Settings bounds, driven through the public
    # `Percept.count_event` API on ids that are NOT registered goals -- so the armed watches' gates
    # are untouched and the measurement is of the episode machinery alone.
    episode_drops: Counter = Counter()

    def on_drop(ep: Any, reason: str) -> None:
        episode_drops[reason] += 1

    probe_ids: dict[str, str] = {}
    if args.episode_probe:
        for a in armed:
            pid = f"probe::{a['goal_id']}"
            probe_ids[a["goal_id"]] = pid
            core.episodes[pid] = EpisodeLog(
                pid,
                window_s=settings.episodes.window_s,
                max_episodes=settings.episodes.max_episodes,
                on_drop=on_drop,
            )

    def member_index(media_t: float) -> int:
        return max(0, bisect.bisect_right(member_offsets, media_t) - 1)

    # ---- channels ------------------------------------------------------------------------------
    stats = RuntimeStats()
    ch = Channels()
    dropped = [0]
    t_start = time.monotonic()

    edge_ts: list[float] = []
    edge_escalated: list[float] = []
    edge_goal_admits: Counter = Counter()
    stage_counts: Counter = Counter()
    judge_started: Counter = Counter()
    judge_finished: Counter = Counter()
    # PRE-CAP mirrors. Past the spend cap dispatches fail before provider transport and produce no
    # fabricated verdict. Keep the real-provider window separate from those terminal dispatches.
    judge_started_pre: Counter = Counter()
    gate_actions: dict[str, Counter] = defaultdict(Counter)
    gate_actions_pre: dict[str, Counter] = defaultdict(Counter)
    gate_satisfied: Counter = Counter()
    episode_new: Counter = Counter()
    episode_cont: Counter = Counter()
    episode_errors: list[str] = []
    health_kinds: Counter = Counter()
    health_samples: dict[str, list[dict]] = defaultdict(list)
    health_timeline: list[tuple[float, str]] = []
    incidents: list[dict] = []
    asks: list[dict] = []
    trace_errors: list[str] = []

    def on_trace(ev: Any) -> None:
        stage = ev.stage
        stage_counts[stage] += 1
        data = ev.data
        t = ev.t
        # ``goal_id`` is a TOP-LEVEL TraceEvent field (percept/agent.py:2192), not a member of
        # ``data`` — reading it out of ``data`` silently attributes every per-goal event to "?" and
        # reports 73 armed goals as never judged.
        ev_goal = getattr(ev, "goal_id", None) or data.get("goal_id")
        if stage == "edge.decision" and t is not None:
            edge_ts.append(float(t))
            if data.get("escalate") or data.get("coast_admit"):
                edge_escalated.append(float(t))
            for gid in data.get("goal_ids") or ():
                edge_goal_admits[gid] += 1
        elif stage == "judge.started":
            judge_started[ev_goal or "?"] += 1
            if provider.capped_at_wall is None:
                judge_started_pre[ev_goal or "?"] += 1
        elif stage == "judge.finished":
            judge_finished[ev_goal or "?"] += 1
        elif stage == "gate.classified":
            gid = ev_goal or "?"
            gate_actions[gid][str(data.get("action"))] += 1
            if provider.capped_at_wall is None:
                gate_actions_pre[gid][str(data.get("action"))] += 1
            if data.get("satisfied"):
                gate_satisfied[gid] += 1
            if args.episode_probe and t is not None:
                pid = probe_ids.get(gid)
                if pid is not None:
                    try:
                        _ep, is_new, _d = core.count_event_decision(
                            pid,
                            {"goal": gid, "satisfied": bool(data.get("satisfied"))},
                            t=float(t),
                            subject_key=f"{gid}::m{member_index(float(t))}",
                        )
                        (episode_new if is_new else episode_cont)[gid] += 1
                    except Exception as exc:
                        if len(episode_errors) < 5:
                            episode_errors.append(f"{type(exc).__name__}: {exc}")

    def on_health(ev: Any) -> None:
        health_kinds[ev.kind] += 1
        health_timeline.append((round(time.monotonic() - t_start, 1), ev.kind))
        if len(health_samples[ev.kind]) < 5:
            health_samples[ev.kind].append({"ts": ev.ts, "data": dict(ev.data or {})})

    def on_ask(fire: Any) -> None:
        asks.append({"wall": round(time.monotonic() - t_start, 2),
                     "goal_id": getattr(fire, "goal_id", None)})

    # ---- the sampler: the timeline this run exists to produce -----------------------------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS / f"{args.out_stem}.checkpoint.jsonl"
    ck = ckpt.open("a")
    samples: list[dict] = []
    stop_flag = {"stop": False}

    def snapshot_sample() -> dict[str, Any]:
        now = time.monotonic()
        media_t = None
        if ch.anchor_wall is not None and ch.anchor_media is not None:
            media_t = round(ch.anchor_media + (now - ch.anchor_wall), 2)
        fn = dict(core.metrics.snapshot())
        probe_logs = [core.episodes[p] for p in probe_ids.values() if p in core.episodes]
        return {
            "kind": "sample",
            "wall_s": round(now - t_start, 1),
            "media_s": media_t,
            "rss_bytes": rss_bytes(),
            "frames_yielded": ch.frames_yielded,
            "decoder_dropped": dropped[0],
            "entities_live": len(core.graph),
            "entity_mints": telemetry.creations,
            "entity_evictions": telemetry.evictions,
            "entity_reids": telemetry.reids,
            "entity_fragmentations": telemetry.fragmentations,
            "entity_ingests": telemetry.ingests,
            "episode_logs": len(core.episodes),
            "episode_live_total": sum(len(lg._episodes) for lg in probe_logs),
            "episode_live_max": max((len(lg._episodes) for lg in probe_logs), default=0),
            "episode_drops": dict(episode_drops),
            "provider_calls": provider.calls,
            "provider_capped": provider.budget_capped,
            "provider_cancelled": provider.cancelled,
            "input_tokens": provider.input_tokens,
            "output_tokens": provider.output_tokens,
            "judge_started": sum(judge_started.values()),
            "goals_judged_distinct": len(judge_started),
            "escalations": int(stats.escalations),
            "judge_superseded": int(stats.judge_superseded),
            "judge_deadline_expired": int(stats.judge_deadline_expired),
            "judge_deadline_dropped": int(stats.judge_deadline_dropped),
            "incidents": len(incidents),
            "asks": int(stats.asks),
            "fires": int(stats.fires),
            "funnel_judged": int(fn.get("funnel.judged") or 0),
            "funnel_fired": int(fn.get("funnel.fired") or 0),
            "funnel_refused": int(fn.get("funnel.refused") or 0),
            "gate_ask": int(fn.get("gate.ask") or 0),
            "health_kinds": dict(health_kinds),
        }

    async def sampler() -> None:
        while not stop_flag["stop"]:
            await asyncio.sleep(args.sample_s)
            try:
                s = snapshot_sample()
            except Exception as exc:
                s = {"kind": "sample_error", "wall_s": round(time.monotonic() - t_start, 1),
                     "error": f"{type(exc).__name__}: {exc}"}
            samples.append(s)
            ck.write(json.dumps(s, default=str) + "\n")
            ck.flush()
            os.fsync(ck.fileno())
            if len(samples) % 10 == 0:
                print(f"[long_run] t={s.get('wall_s')}s media={s.get('media_s')}s "
                      f"calls={s.get('provider_calls')} inc={s.get('incidents')} "
                      f"ent={s.get('entities_live')}/{s.get('entity_mints')} "
                      f"rss={(s.get('rss_bytes') or 0) / 1e6:.0f}MB", file=sys.stderr, flush=True)

    ck.write(json.dumps({
        "kind": "run_header",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": manifest["video_path"],
        "source_duration_s": manifest["clock"]["measured_container_duration_s"],
        "n_goals_armed": len(armed),
        "capture_fps": fps,
        "decision_deadline_s": decision_deadline,
        "provider_deadline_s": LR.ACTION_PROVIDER_DEADLINE_S,
        "max_provider_calls": args.max_provider_calls,
        "model": model,
        "provider_budget_checkpoint": str(provider_budget.checkpoint_path),
        "provider_budget_global_allocation": provider_budget.provenance()["global_allocation"],
    }, default=str) + "\n")
    ck.flush()

    # ---- PROVE the backstop is unscaled, do not assume it ------------------------------------
    # Ask the shipped runtime what it will actually apply to THIS batch of goal ids, rather than
    # asserting that K >= N collapsed the wave multiplier. Reported in meta either way.
    # ``_decision_waves`` IS the multiplier both scheduler implementations
    # (``_LegacyJudgeScheduler._batch_deadline_s`` and ``_JudgeScheduler._batch_deadline_s``) apply
    # to the per-decision budget, so reading it directly is class-agnostic and exact.
    resolved_batch_deadline_s = None
    effective_K = None
    decision_waves = None
    try:
        from percept.runtime.worker import _decision_waves

        effective_K = int(core.effective_judge_concurrency(
            [core.goals.get(g) for g in goal_ids]
        ))
        decision_waves = int(_decision_waves(core, len(goal_ids)))
        resolved_batch_deadline_s = decision_deadline * decision_waves
    except Exception as exc:  # noqa: BLE001 - an instrument must never break the run
        resolved_batch_deadline_s = f"probe failed: {type(exc).__name__}: {exc}"

    print(f"[long_run] source={manifest['video_path']} "
          f"{(manifest['clock']['measured_container_duration_s'] or 0) / 60:.1f} min | "
          f"goals={len(armed)} | fps={fps} | decision_deadline={decision_deadline:.1f}s | "
          f"K={K} (effective {effective_K}) | waves={decision_waves} | "
          f"batch_deadline={resolved_batch_deadline_s} | "
          f"cap={args.max_provider_calls} calls", file=sys.stderr, flush=True)

    sampler_task = asyncio.create_task(sampler())
    error = None
    try:
        runtime = worker_run(
            core,
            source=paced_source(source_path, fps=fps, ch=ch, dropped=dropped,
                                detector=detector, graph=core.graph, telemetry=telemetry,
                                coast_s=coast_s, max_wall_s=args.max_wall_s),
            fps=fps,
            stats=stats,
            gate_source="camera",
            judge_dispatch="async",
            judge_stall_after_s=decision_deadline,
            judge_decision_deadline_s=decision_deadline,
            timebase="media",
            on_trace=on_trace,
            on_health=on_health,
            on_ask=on_ask,
            on_incident=None if webhook_probe is None else webhook_probe.sink,
            sink_errors="ignore" if webhook_probe is None else "raise",
        )
        try:
            async for inc in runtime:
                observed = time.monotonic()
                judgment = incident_judgment(inc)
                timing = getattr(judgment, "timing", None)
                # The contract field is ``concern_id`` (percept/contracts/__init__.py:1394), NOT
                # ``goal_id``. Reading the wrong name yields None for every incident and silently
                # empties the per-goal incident column, which is what the first full run reported.
                gid = getattr(inc, "concern_id", None) or getattr(inc, "goal_id", None)
                incidents.append({
                    "id": getattr(inc, "id", None),
                    "goal_id": gid,
                    "ts": float(getattr(inc, "ts", 0.0) or 0.0),
                    "wall": round(observed - t_start, 2),
                    "observed_since_anchor_s": (
                        None if ch.anchor_wall is None else observed - ch.anchor_wall
                    ),
                    "confidence": float(
                        getattr(getattr(inc, "verdict", None), "confidence", 0.0) or 0.0),
                    # NOT an action-onset latency, and no longer named as one.
                    #
                    # `inc.ts` is the MEDIA timestamp of the frame this Incident was judged from —
                    # the moment the pipeline ingested its evidence, not the moment the action
                    # began. This leg is therefore `judged frame's media instant -> Incident
                    # observed`: it measures what the pipeline added AFTER the evidence existed,
                    # which is a real and useful number, but it is blind to how long the action had
                    # already been running before a frame of it was admitted.
                    #
                    # It is also PER EMISSION and UNMATCHED. Qualification quality below separately
                    # matches this emission to manifest occurrences and deduplicates to the earliest
                    # Incident/receiver arrival; this raw leg must never be quoted in its place.
                    "frame_media_to_incident_s": (
                        None if timing is None or ch.anchor_wall is None else round(
                            observed - (ch.anchor_wall
                                        + (float(getattr(inc, "ts", 0.0) or 0.0) - ch.anchor_media)),
                            3)
                    ),
                })
                if args.max_wall_s and (observed - t_start) > args.max_wall_s:
                    break
        finally:
            await runtime.aclose()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[long_run] RUN ERROR {error}", file=sys.stderr, flush=True)
    finally:
        stop_flag["stop"] = True
        sampler_task.cancel()
        try:
            await sampler_task
        except (asyncio.CancelledError, Exception):
            pass

    delivery_missing: list[str] = []
    delivery_attempts = 0
    for incident in incidents:
        observation = (
            None if webhook_probe is None else webhook_probe.lookup(str(incident.get("id")))
        )
        if observation is None:
            incident["webhook"] = None
            if webhook_probe is not None:
                delivery_missing.append(str(incident.get("id")))
            continue
        delivery_attempts += int(observation.attempts)
        incident["webhook"] = {
            "receiver_arrival_since_anchor_s": (
                None
                if ch.anchor_wall is None
                else observation.arrival_at - ch.anchor_wall
            ),
            "receiver_ack_since_anchor_s": (
                None
                if ch.anchor_wall is None or observation.ack_at is None
                else observation.ack_at - ch.anchor_wall
            ),
            "attempts": observation.attempts,
        }
    if delivery_missing and error is None:
        error = (
            f"receiver did not observe {len(delivery_missing)} emitted incident(s); "
            f"first={delivery_missing[0]}"
        )

    final = snapshot_sample()
    final["kind"] = "final_sample"
    samples.append(final)
    ck.write(json.dumps(final, default=str) + "\n")
    ck.flush()
    ck.close()

    wall = time.monotonic() - t_start
    payload = analyse(
        args=args, manifest=manifest, armed=armed, goal_text_by_id=goal_text_by_id,
        arm_failures=arm_failures, core=core, stats=stats, provider=provider,
        telemetry=telemetry, samples=samples, incidents=incidents, asks=asks,
        edge_ts=edge_ts, edge_escalated=edge_escalated, edge_goal_admits=edge_goal_admits,
        stage_counts=stage_counts, judge_started=judge_started, judge_finished=judge_finished,
        cap_snapshot=cap_snapshot, gate_actions=gate_actions,
        judge_started_pre=judge_started_pre, gate_actions_pre=gate_actions_pre,
        gate_satisfied=gate_satisfied, episode_new=episode_new, episode_cont=episode_cont,
        episode_drops=episode_drops, episode_errors=episode_errors, probe_ids=probe_ids,
        health_kinds=health_kinds, health_samples=health_samples,
        health_timeline=health_timeline, receipt_outcomes=receipt_outcomes,
        receipt_latency=receipt_latency, ch=ch, dropped=dropped, fps=fps,
        decision_deadline=decision_deadline, wall=wall, model=model, error=error,
        judge_concurrency=K, effective_K=effective_K, decision_waves=decision_waves,
        media_resolution=media_resolution,
        resolved_batch_deadline_s=resolved_batch_deadline_s,
        detector=detector, trace_errors=trace_errors,
        delivery={
            "enabled": webhook_probe is not None,
            "incidents": len(incidents),
            "receiver_arrivals": len(incidents) - len(delivery_missing)
            if webhook_probe is not None
            else 0,
            "receiver_acks": sum(
                1 for item in incidents
                if (item.get("webhook") or {}).get("receiver_ack_since_anchor_s") is not None
            ),
            "attempts": delivery_attempts,
            "missing_event_ids": delivery_missing,
        },
    )
    return payload


# =================================================================================================
# analysis
# =================================================================================================


def analyse(**k: Any) -> dict[str, Any]:
    args = k["args"]
    manifest = k["manifest"]
    armed = k["armed"]
    core = k["core"]
    stats = k["stats"]
    provider = k["provider"]
    tel: EntityTelemetry = k["telemetry"]
    samples = k["samples"]
    incidents = k["incidents"]
    judge_started: Counter = k["judge_started"]
    gate_actions: dict[str, Counter] = k["gate_actions"]
    edge_goal_admits: Counter = k["edge_goal_admits"]
    probe_ids: dict[str, str] = k["probe_ids"]
    fn = dict(core.metrics.snapshot())

    quality = score_long_quality(
        armed,
        incidents,
        anchor_media=k["ch"].anchor_media,
        precision_eligible=bool(manifest.get("precision_eligible")),
        precision_ineligible_reason=manifest.get("precision_ineligible_reason"),
        delivery_enabled=bool(k["delivery"]["enabled"]),
    )
    quality["launch_recall_gate"] = {
        "target_recall": 0.99,
        "confidence": 0.95,
        "minimum_independent_occurrences_with_zero_misses": 299,
        "nominal_gt_occurrences": quality["pooled"]["gt_occurrences"],
        "eligible": False,
        "status": "not_eligible",
        "reason": (
            "injected occurrences share source clips/watches and are not declared independent; "
            "the nominal total is also below the 299 zero-miss minimum"
        ),
    }

    def g(key: str) -> int:
        return int(fn.get(key) or 0)

    # ---- 1. ENTITY TRACKS ---------------------------------------------------------------------
    lifetimes = [
        tel.last_seen[e] - tel.first_seen[e] for e in tel.first_seen if e in tel.last_seen
    ]
    entity = {
        "observation_source": ("harness motion-blob detector at frame rate "
                               "(percept/rhythm_hero.py:67 call shape)"
                               if k["detector"] is not None else "none (feed disabled)"),
        "max_entities_setting": int(core.settings.entity.max_entities),
        "iou_match_setting": float(core.settings.entity.iou_match),
        "coast_s": float(getattr(core.graph, "_coast_s", 1.0)),
        "ingest_batches": tel.ingests,
        "observations_ingested": tel.observations,
        "frames_with_blobs": None if k["detector"] is None else k["detector"].frames_with_blobs,
        "frames_seen_by_detector": None if k["detector"] is None else k["detector"].frames_seen,
        "tracks_created": tel.creations,
        "tracks_live_at_end": len(core.graph),
        "evictions": tel.evictions,
        "reidentifications_after_coast": tel.reids,
        "reid_gap_s": dist(tel.gap_s),
        "fragmentations": tel.fragmentations,
        "label_switches": tel.label_switches,
        "resurrections_after_eviction": tel.resurrections,
        "track_lifetime_s": dist(lifetimes),
        "track_touches": dist([float(v) for v in tel.touches.values()]),
        "saturated_max_entities": bool(tel.evictions > 0),
        "first_saturation_media_s": None,
        "live_curve": [
            {"media_s": s.get("media_s"), "live": s.get("entities_live"),
             "mints": s.get("entity_mints"), "evictions": s.get("entity_evictions")}
            for s in samples if s.get("kind") == "sample"
        ][:400],
        "size_marks": tel.size_marks[:400],
    }
    for s in samples:
        if s.get("kind") == "sample" and (s.get("entity_evictions") or 0) > 0:
            entity["first_saturation_media_s"] = s.get("media_s")
            break

    # ---- 2. EPISODE STATE ----------------------------------------------------------------------
    probe_logs = {gid: core.episodes[p] for gid, p in probe_ids.items() if p in core.episodes}
    live_counts = [len(lg._episodes) for lg in probe_logs.values()]
    minted = [lg._counter for lg in probe_logs.values()]
    runtime_logs = [key for key in core.episodes if not str(key).startswith("probe::")]
    episodes = {
        "window_s_setting": float(core.settings.episodes.window_s),
        "max_episodes_setting": int(core.settings.episodes.max_episodes),
        "source_duration_s": manifest["clock"]["measured_container_duration_s"],
        "run_exceeds_window": (manifest["clock"]["measured_container_duration_s"] or 0)
        > float(core.settings.episodes.window_s),
        "runtime_owned_episode_logs": len(runtime_logs),
        "runtime_owned_episode_log_ids": runtime_logs[:20],
        "probe_logs": len(probe_logs),
        "probe_observations": sum(k["episode_new"].values()) + sum(k["episode_cont"].values()),
        "probe_episodes_opened": sum(k["episode_new"].values()),
        "probe_continuations": sum(k["episode_cont"].values()),
        "probe_episodes_minted_total": sum(minted),
        "probe_episodes_live_total": sum(live_counts),
        "probe_episodes_live_max": max(live_counts, default=0),
        "probe_episodes_live_dist": dist([float(x) for x in live_counts]),
        "bound_drops": dict(k["episode_drops"]),
        "probe_errors": k["episode_errors"],
        "retained_evidence_bytes": sum(
            lg.retained_evidence_nbytes() for lg in probe_logs.values()
        ),
        "live_curve": [
            {"media_s": s.get("media_s"), "live_total": s.get("episode_live_total"),
             "live_max": s.get("episode_live_max"), "drops": s.get("episode_drops")}
            for s in samples if s.get("kind") == "sample"
        ][:400],
        "metrics": {key: fn[key] for key in sorted(fn) if key.startswith("episode.")},
    }

    # ---- 3. EVENT LOG / LEDGER -----------------------------------------------------------------
    n_edge = k["stage_counts"].get("edge.decision", 0)
    n_adm = int(stats.escalations)
    n_disp = int(stats.perceive_dispatches)
    scheduler_receipted = (int(stats.judge_superseded) + int(stats.judge_revision_dropped)
                           + int(stats.judge_deadline_dropped)
                           + int(stats.judge_source_discontinuity_dropped))
    n_started = k["stage_counts"].get("judge.started", 0)
    n_finished = k["stage_counts"].get("judge.finished", 0)
    n_degraded = k["stage_counts"].get("judge.degraded", 0)
    n_gate = k["stage_counts"].get("gate.classified", 0)
    # UNIT DISCIPLINE. Two different units live in this ledger and conflating them invents a
    # phantom imbalance: an ADMISSION is one frame, and ``stats.judge_calls`` counts admissions
    # LAUNCHED (worker.py:910).  A DISPATCH (``perceive_dispatches`` / ``judge.started``) is one
    # frame x one goal, so with M goals armed a single launched admission produces M of them.
    # The scheduler invariant documented at ``percept/runtime/worker.py:264`` is stated in the
    # ADMISSION unit and is checked here in that unit only.
    n_batches = int(stats.judge_calls)
    admission_residual = n_adm - n_batches - scheduler_receipted
    ledger = {
        "frames_yielded": k["ch"].frames_yielded,
        "frames_emitted_by_runtime": int(stats.frames),
        "frames_dropped_with_receipt": int(stats.frames_dropped),
        "decoder_dropped_no_receipt": k["dropped"][0],
        "edge_decisions": n_edge,
        "edge_decisions_minus_frames": n_edge - int(stats.frames),
        "admitted": n_adm,
        "admission_by_cause": dict(stats.admission_by_cause),
        "signal_admits": int(stats.signal_admits),
        "coast_admits": int(stats.coast_admits),
        "fixed_admits": int(stats.fixed_admits),
        "suppressed_with_receipt": n_edge - n_adm,
        "perceive_dispatches": n_disp,
        "judge_superseded": int(stats.judge_superseded),
        "judge_revision_dropped": int(stats.judge_revision_dropped),
        "judge_deadline_dropped": int(stats.judge_deadline_dropped),
        "judge_deadline_expired": int(stats.judge_deadline_expired),
        "judge_source_discontinuity_dropped": int(stats.judge_source_discontinuity_dropped),
        "scheduler_receipted_total": scheduler_receipted,
        "admissions_launched_as_batches": n_batches,
        "admission_residual": admission_residual,
        "admission_balances": admission_residual == 0,
        "goals_armed": len(armed),
        "perceive_dispatches_goal_units": n_disp,
        "dispatches_per_batch": round(n_disp / n_batches, 3) if n_batches else None,
        "judge_started_trace": n_started,
        "judge_finished_trace": n_finished,
        "judge_degraded_trace": n_degraded,
        "gate_classified_trace": n_gate,
        "started_minus_finished_minus_degraded": n_started - n_finished - n_degraded,
        "finished_minus_gate_classified": n_finished - n_gate,
        "stage_counts": dict(k["stage_counts"]),
        "funnel_mirror": {
            "funnel.judged": g("funnel.judged"),
            "funnel.fired": g("funnel.fired"),
            "funnel.refused": g("funnel.refused"),
            "funnel.gated": g("funnel.gated"),
            "funnel.dropped": g("funnel.dropped"),
            "gate.ask": g("gate.ask"),
            "judged_equals_judge_started": g("funnel.judged") == n_started,
            "judged_minus_judge_started": g("funnel.judged") - n_started,
            "refused_equals_ask": g("funnel.refused") == g("gate.ask"),
            "refused_minus_ask": g("funnel.refused") - g("gate.ask"),
            "refused_is_double_ask": g("gate.ask") > 0
            and g("funnel.refused") == 2 * g("gate.ask"),
        },
        "token_cross_check": {
            "funnel_input_tokens": g("judge.tokens.input"),
            "funnel_output_tokens": g("judge.tokens.output"),
            "adapter_input_tokens": provider.input_tokens,
            "adapter_output_tokens": provider.output_tokens,
            "agree": g("judge.tokens.input") == provider.input_tokens
            and g("judge.tokens.output") == provider.output_tokens,
        },
    }

    # ---- 4. JUDGE ALLOCATION -------------------------------------------------------------------
    per_goal = []
    for a in armed:
        gid = a["goal_id"]
        acts = k["gate_actions"].get(gid, Counter())
        acts_pre = k["gate_actions_pre"].get(gid, Counter())
        per_goal.append({
            "judged_pre_cap": k["judge_started_pre"].get(gid, 0),
            "fire_pre_cap": acts_pre.get("fire", 0),
            "ask_pre_cap": acts_pre.get("ask", 0),
            "silent_pre_cap": acts_pre.get("silent", 0),
            "goal_id": gid,
            "goal_text": a["goal_text"],
            "goal_kind": a["goal_kind"],
            "arm_index": len(per_goal),
            "gt_occurrences": a["n_occurrences"],
            "admitted": edge_goal_admits.get(gid, 0),
            "judged": judge_started.get(gid, 0),
            "judge_finished": k["judge_finished"].get(gid, 0),
            "gate_classified": sum(acts.values()),
            "satisfied": k["gate_satisfied"].get(gid, 0),
            "fire": acts.get("fire", 0),
            "ask": acts.get("ask", 0),
            "silent": acts.get("silent", 0),
            "incidents": sum(1 for i in incidents if i.get("goal_id") == gid),
            "episodes_opened": k["episode_new"].get(gid, 0),
            "episode_continuations": k["episode_cont"].get(gid, 0),
        })
    judged_counts = [p["judged"] for p in per_goal]
    pre_counts = [p["judged_pre_cap"] for p in per_goal]
    total_judged = sum(judged_counts)
    n_goals = max(1, len(per_goal))
    fair_share = total_judged / n_goals
    ordered = sorted(judged_counts)
    cum = 0.0
    gini_num = 0.0
    for i, v in enumerate(ordered, start=1):
        cum += v
        gini_num += (2 * i - n_goals - 1) * v
    gini = (gini_num / (n_goals * sum(ordered))) if sum(ordered) else 0.0
    allocation = {
        "n_goals_armed": len(armed),
        "goals_ever_judged": sum(1 for v in judged_counts if v > 0),
        "goals_never_judged": sum(1 for v in judged_counts if v == 0),
        "judged_total": total_judged,
        "judged_per_goal": dist([float(v) for v in judged_counts]),
        "fair_share_per_goal": round(fair_share, 2),
        "gini_of_judged": round(gini, 4),
        "starvation_ratio_min_over_max": (
            (min(judged_counts) / max(judged_counts)) if judged_counts and max(judged_counts)
            else None
        ),
        "by_arm_position": [
            {"goal_id": p["goal_id"], "arm_index": p["arm_index"], "judged": p["judged"],
             "gate_classified": p["gate_classified"]}
            for p in per_goal
        ],
        "gate_actions_total": {
            "fire": sum(p["fire"] for p in per_goal),
            "ask": sum(p["ask"] for p in per_goal),
            "silent": sum(p["silent"] for p in per_goal),
        },
        # The window in which a provider attempt could still be durably admitted. Dispatch counts
        # above may include post-cap terminal failures; these lines do not.
        "pre_cap": {
            "judged_total": sum(pre_counts),
            "goals_ever_judged": sum(1 for v in pre_counts if v > 0),
            "goals_never_judged": sum(1 for v in pre_counts if v == 0),
            "judged_per_goal": dist([float(v) for v in pre_counts]),
            "min_over_max": ((min(pre_counts) / max(pre_counts))
                             if pre_counts and max(pre_counts) else None),
            "gate_actions": {
                "fire": sum(p["fire_pre_cap"] for p in per_goal),
                "ask": sum(p["ask_pre_cap"] for p in per_goal),
                "silent": sum(p["silent_pre_cap"] for p in per_goal),
            },
        },
        "stats_asks": int(stats.asks),
        "stats_fires": int(stats.fires),
        "on_ask_callbacks": len(k["asks"]),
        "spend_cap_snapshot": k["cap_snapshot"] or None,
        "health": {
            "judge_scheduler_backpressure": k["health_kinds"].get(
                "judge_scheduler_backpressure", 0),
            "judge_superseded": k["health_kinds"].get("judge_superseded", 0),
            "judge_deadline_expired": k["health_kinds"].get("judge_deadline_expired", 0),
            "judge_revision_dropped": k["health_kinds"].get("judge_revision_dropped", 0),
            "all_kinds": dict(k["health_kinds"]),
            "samples": {kk: v for kk, v in k["health_samples"].items()},
        },
        "per_goal": per_goal,
    }

    # ---- 5. MEMORY -----------------------------------------------------------------------------
    rss = [(s.get("wall_s"), s.get("rss_bytes")) for s in samples
           if s.get("kind") in ("sample", "final_sample") and s.get("rss_bytes")]
    rss_vals = [v for _t, v in rss]
    first_third = rss_vals[: max(1, len(rss_vals) // 3)]
    last_third = rss_vals[-max(1, len(rss_vals) // 3):]
    slope = None
    if len(rss) >= 4:
        xs = [float(t) for t, _v in rss]
        ys = [float(v) for _t, v in rss]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else None
    memory = {
        "samples": len(rss),
        "rss_curve": [{"wall_s": t, "rss_mb": round(v / 1e6, 1)} for t, v in rss][:600],
        "rss_start_mb": round(rss_vals[0] / 1e6, 1) if rss_vals else None,
        "rss_end_mb": round(rss_vals[-1] / 1e6, 1) if rss_vals else None,
        "rss_peak_mb": round(max(rss_vals) / 1e6, 1) if rss_vals else None,
        "rss_mean_first_third_mb": round(statistics.fmean(first_third) / 1e6, 1)
        if first_third else None,
        "rss_mean_last_third_mb": round(statistics.fmean(last_third) / 1e6, 1)
        if last_third else None,
        "growth_last_minus_first_third_mb": (
            round((statistics.fmean(last_third) - statistics.fmean(first_third)) / 1e6, 1)
            if first_third and last_third else None
        ),
        "slope_mb_per_hour": round(slope * 3600 / 1e6, 2) if slope is not None else None,
        "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        "frame_window_setting": int(core.settings.events.window),
    }

    # ---- provider accounting -------------------------------------------------------------------
    price_in = args.price_in
    price_out = args.price_out
    cost = None
    if price_in is not None and price_out is not None:
        cost = round(provider.input_tokens / 1e6 * price_in
                     + provider.output_tokens / 1e6 * price_out, 6)
    source_duration_s = float(manifest["clock"]["measured_container_duration_s"] or 0.0)
    observed_source_s = 0.0
    if k["ch"].anchor_media is not None and k["ch"].last_media is not None:
        # Include the final sampled frame interval, bounded by the declared container duration.
        observed_source_s = min(
            source_duration_s,
            max(0.0, k["ch"].last_media - k["ch"].anchor_media) + (1.0 / float(k["fps"])),
        )
    source_stream_hours = observed_source_s / 3600.0
    scheduled_watch_hours = source_stream_hours * len(armed)
    cost_per_source_hour = (
        cost / source_stream_hours if cost is not None and source_stream_hours > 0 else None
    )
    cost_per_scheduled_watch_hour = (
        cost / scheduled_watch_hours if cost is not None and scheduled_watch_hours > 0 else None
    )
    accounting = {
        "model": k["model"],
        "physical_calls": provider.calls,
        "budget_capped_calls": provider.budget_capped,
        "cancelled": provider.cancelled,
        "errors": provider.errors,
        "error_types": dict(provider.error_types),
        "usage_rows": provider.usage_rows,
        "input_tokens": provider.input_tokens,
        "output_tokens": provider.output_tokens,
        "input_bytes": provider.input_bytes,
        "reflex_context_stamped": provider.reflex_stamped,
        "verdicts_with_evidence_text": provider.evidence_nonempty,
        "provider_latency_completed": dist(provider.latencies_completed),
        "provider_latency_censored": dist(provider.latencies_censored),
        "route_latency_completed": dist(k["receipt_latency"]),
        "terminal_outcomes": dict(sorted(k["receipt_outcomes"].items())),
        "price_usd_per_million": {"input": price_in, "output": price_out,
                                  "source": args.price_source},
        "estimated_cost_usd": cost,
        "source_stream_hours": source_stream_hours,
        "scheduled_watch_hours": scheduled_watch_hours,
        "estimated_provider_cost_usd_per_source_hour": cost_per_source_hour,
        "estimated_provider_cost_usd_per_scheduled_watch_hour": cost_per_scheduled_watch_hour,
        "provider_cost_estimate_is_lower_bound": provider.usage_rows != provider.calls,
        "launch_cost_gate": {
            "ceiling_usd_per_source_hour": LR.ABSOLUTE_COST_CEILING_USD_PER_SOURCE_HOUR,
            "eligible": False,
            "status": "not_eligible",
            "observed_provider_estimate_usd_per_source_hour": cost_per_source_hour,
            "reason": (
                "declared-price x returned-token usage excludes calls without usage rows, "
                "runtime/hosting/egress costs, and the provider invoice; an all-in gate needs "
                "billed provider spend plus infrastructure metering"
            ),
        },
        "cost_caveat": ("Tokens are measured only on calls that RETURNED a usage row. A call the "
                        "pool cancelled at the 1.5 s deadline may still have been billed by the "
                        "provider; the fraction of such calls is reported so the figure can be "
                        "read as a LOWER bound."),
        "completed_fraction": (provider.usage_rows / provider.calls) if provider.calls else None,
        "provider_budget": args._provider_budget.provenance(),
    }

    bi = build_info()
    return {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "package_version": bi.package_version,
            "build_commit": bi.git_commit,
            "contract_version": bi.contract_version,
            "interpreter": sys.executable,
            "source": manifest["video_path"],
            "source_manifest_kind": manifest.get("source_manifest_kind"),
            "source_duration_s": manifest["clock"]["measured_container_duration_s"],
            "source_members": manifest["n_members"],
            "source_profile": manifest["profile"],
            "clock_drift_s": manifest["clock"]["drift_s"],
            "precision_eligible": bool(manifest.get("precision_eligible")),
            "precision_ineligible_reason": manifest.get("precision_ineligible_reason"),
            "conductor_pin": LR.ACTION_CONDUCTOR_PIN or None,
            "webhook_probe": (
                "loopback real HTTP receiver; monotonic arrival + ACK timestamps"
                if k["delivery"]["enabled"]
                else "disabled"
            ),
            "qualification_contract": getattr(args, "_qualification_contract_summary", None),
            "gt_occurrences": sum(a["n_occurrences"] for a in armed),
            "n_goals_armed": len(armed),
            "arm_failures": k["arm_failures"],
            "capture_fps": k["fps"],
            "provider_deadline_s": LR.ACTION_PROVIDER_DEADLINE_S,
            "decision_deadline_s": k["decision_deadline"],
            "decision_deadline_note": (
                "per-decision budget as PASSED to the scheduler. Whether the runtime scales it for "
                "the batch is reported separately as resolved_batch_deadline_s, read back from "
                "percept.runtime.worker._WorkerJudgeScheduler._batch_deadline_s rather than assumed."
            ),
            "media_resolution": k["media_resolution"] or "provider default (HIGH)",
            "judge_concurrency_requested": k["judge_concurrency"],
            "judge_concurrency_effective": k["effective_K"],
            "decision_waves": k["decision_waves"],
            "resolved_batch_deadline_s": k["resolved_batch_deadline_s"],
            "batch_deadline_unscaled": (
                isinstance(k["resolved_batch_deadline_s"], (int, float))
                and abs(float(k["resolved_batch_deadline_s"]) - float(k["decision_deadline"])) < 1e-6
            ),
            "temporal_frames": LR.ACTION_TEMPORAL_FRAMES,
            "wall_s": round(k["wall"], 1),
            "realtime_ratio": (round(k["wall"] / manifest["clock"]
                                     ["measured_container_duration_s"], 3)
                               if manifest["clock"]["measured_container_duration_s"] else None),
            "error": k["error"],
        },
        "replay": {
            "frames_yielded": k["ch"].frames_yielded,
            "decoder_dropped": k["dropped"][0],
            "pace_lag_s": dist(k["ch"].pace_lag_s),
            "pace_marks": k["ch"].pace_marks[:400],
            "media_covered_s": observed_source_s,
        },
        "entity": entity,
        "episodes": episodes,
        "ledger": ledger,
        "allocation": allocation,
        "memory": memory,
        "accounting": accounting,
        "quality": quality,
        "delivery": k["delivery"],
        "incidents": {
            "n": len(incidents),
            "distinct_goals": len({i.get("goal_id") for i in incidents}),
            "frame_media_to_incident_s": dist([i["frame_media_to_incident_s"] for i in incidents
                                               if i.get("frame_media_to_incident_s") is not None]),
            "frame_media_to_incident_s_unit": "incident_emission",
            "frame_media_to_incident_s_note": (
                "judged frame's media instant -> Incident observed. NOT the action-onset latency "
                "reported in quality: this raw diagnostic is per emission and unmatched. Use "
                "quality.pooled.onset_to_incident_s for occurrence-deduplicated latency."
            ),
            "rows": incidents[:500],
        },
        "health_timeline": k["health_timeline"][:2000],
        "samples": samples,
        "limits": [
            "Entity observations come from a harness-local motion-blob detector, not a semantic "
            "detector. Track counts are counts of MOVING REGIONS, not of people or objects. The "
            "association/eviction logic measured is the shipped EntityGraph, unmodified.",
            "Episode numbers come from probe logs driven through the public count_event API on "
            "unregistered goal ids. They measure EpisodeLog under the shipped Settings bounds; they "
            "do NOT show the armed watches opening episodes, because watch goals never do (see the "
            "episode section).",
            "The decision deadline is scaled to the goal count and is therefore NOT the shipped "
            "operating point. Every allocation number is a number for THAT deadline.",
            "Cost is measured tokens x a declared price, and only calls that returned a usage row "
            "carry tokens. Cancelled calls may still have been billed: the figure is a lower bound.",
            "One source, one provider, one model, one day, one machine.",
        ],
    }


# =================================================================================================
# report
# =================================================================================================


def _mb(x: Any) -> str:
    return "n/a" if x is None else f"{x}"


def render_md(p: dict) -> str:
    L: list[str] = []
    A = L.append
    m = p["meta"]
    A("# Long-horizon run: one source, many concurrent watches")
    A("")
    A(f"Runtime under test: `percept` {m['package_version']} built from `{m['build_commit']}` "
      f"(contract {m['contract_version']}), interpreter `{m['interpreter']}`.")
    A(f"Source `{m['source']}` — {(m['source_duration_s'] or 0) / 60:.1f} min joined from "
      f"{m['source_members']} corpus videos, container-vs-sum drift {m['clock_drift_s']} s. "
      f"**{m['n_goals_armed']} distinct goals armed simultaneously** on one core, "
      f"{m['gt_occurrences']} ground-truth occurrences. Generated {m['generated']}.")
    A("")
    A(f"Replay wall {m['wall_s']:.0f} s (realtime ratio {m['realtime_ratio']}), capture "
      f"{m['capture_fps']} fps, provider deadline {m['provider_deadline_s']} s, decision deadline "
      f"**{m['decision_deadline_s']:.1f} s** — {m['decision_deadline_note']}.")
    if m.get("error"):
        A("")
        A(f"**The run raised:** `{m['error']}` — everything below is what was collected up to that "
          f"point and is reported, not discarded.")
    A("")
    rep = p["replay"]
    A(f"Replay integrity: {rep['frames_yielded']:,} frames yielded, {rep['decoder_dropped']} "
      f"dropped by the decoder without a receipt, per-frame pacing lag "
      f"p50 {rep['pace_lag_s']['p50']} s / p95 {rep['pace_lag_s']['p95']} s / "
      f"max {rep['pace_lag_s']['max']} s.")
    A("")

    q = p["quality"]["pooled"]
    qr = p["quality"]["receiver"]
    A("## Qualification quality")
    A("")
    A(f"Occurrence recall: **{q['occurrences_matched']}/{q['gt_occurrences']}** "
      f"({q['occurrence_recall']}); misses {q['occurrences_missed']}. Timely recall: "
      f"2 s `{q['timely_recall']['2s']}`, 5 s `{q['timely_recall']['5s']}`.")
    A(f"Launch recall gate: **{p['quality']['launch_recall_gate']['status']}** — "
      f"{p['quality']['launch_recall_gate']['reason']}.")
    A(f"Occurrence-deduplicated onset → SDK Incident latency: `{q['onset_to_incident_s']}`.")
    if qr["enabled"]:
        A(f"Webhook-receiver recall: **{qr['occurrences_matched']}/{qr['gt_occurrences']}** "
          f"({qr['occurrence_recall']}); misses {qr['occurrences_missed']}. Timely recall: "
          f"2 s `{qr['timely_recall']['2s']}`, 5 s `{qr['timely_recall']['5s']}`.")
        A(f"Occurrence-deduplicated onset → webhook receiver latency: "
          f"`{qr['onset_to_webhook_arrival_s']}`. Delivery: `{p['delivery']}`.")
    else:
        A("Webhook-receiver quality: **NOT MEASURED** — delivery probe disabled.")
    if q["precision_eligible"]:
        A(f"Per-emission precision: `{q['precision_per_emission']}` "
          f"({q['incidents_inside_labeled_positive']}/{q['incidents']}).")
    else:
        A(f"Per-emission precision: **NOT ELIGIBLE** — {q['precision_ineligible_reason']}.")
    A("")

    # --- entity ---
    e = p["entity"]
    A("## 1. Entity tracks")
    A("")
    A(f"Observation stream: {e['observation_source']}.")
    A("")
    A("| Quantity | Value |")
    A("|---|---|")
    for label, key in (
        ("`max_entities` setting", "max_entities_setting"),
        ("`iou_match` setting", "iou_match_setting"),
        ("predictive coast horizon `coast_s`", "coast_s"),
        ("ingest batches", "ingest_batches"),
        ("observations ingested", "observations_ingested"),
        ("tracks created (total mints)", "tracks_created"),
        ("tracks live at end", "tracks_live_at_end"),
        ("evictions (`on_evict`)", "evictions"),
        ("re-identifications after the coast horizon", "reidentifications_after_coast"),
        ("fragmentations (new id, no new observation)", "fragmentations"),
        ("label switches", "label_switches"),
        ("resurrections of an evicted id", "resurrections_after_eviction"),
        ("first eviction at media", "first_saturation_media_s"),
    ):
        A(f"| {label} | {e.get(key)} |")
    A("")
    A(f"Track lifetime (s): {e['track_lifetime_s']}")
    A(f"Touches per track: {e['track_touches']}")
    A(f"Re-id gap (s): {e['reid_gap_s']}")
    A("")

    # --- episodes ---
    ep = p["episodes"]
    A("## 2. Episode state")
    A("")
    A(f"`Settings.episodes.window_s` = {ep['window_s_setting']} s, `max_episodes` = "
      f"{ep['max_episodes_setting']}. Source is {ep['source_duration_s']} s, so the run "
      f"{'EXCEEDS' if ep['run_exceeds_window'] else 'does not exceed'} the window.")
    A("")
    A(f"**Episode logs the RUNTIME itself created for the {p['meta']['n_goals_armed']} armed "
      f"watches: {ep['runtime_owned_episode_logs']}.**")
    A("")
    A("| Probe-log quantity | Value |")
    A("|---|---|")
    for label, key in (
        ("probe logs", "probe_logs"),
        ("observations folded", "probe_observations"),
        ("episodes opened (`is_new`)", "probe_episodes_opened"),
        ("continuations", "probe_continuations"),
        ("episodes minted, all logs", "probe_episodes_minted_total"),
        ("episodes LIVE at end, all logs", "probe_episodes_live_total"),
        ("largest single log at end", "probe_episodes_live_max"),
        ("bound-driven drops", "bound_drops"),
        ("retained evidence bytes", "retained_evidence_bytes"),
    ):
        A(f"| {label} | {ep.get(key)} |")
    A("")
    if ep.get("metrics"):
        A("Runtime episode metrics: `" + json.dumps(ep["metrics"]) + "`")
        A("")

    # --- ledger ---
    lg = p["ledger"]
    A("## 3. Event log — completeness and ordering")
    A("")
    A("| Ledger line | Units |")
    A("|---|---|")
    for label, key in (
        ("frames yielded by the paced source", "frames_yielded"),
        ("frames emitted by the runtime", "frames_emitted_by_runtime"),
        ("frames dropped WITH a receipt", "frames_dropped_with_receipt"),
        ("decoder drops WITHOUT a receipt", "decoder_dropped_no_receipt"),
        ("`edge.decision` receipts", "edge_decisions"),
        ("admitted (escalations)", "admitted"),
        ("suppressed WITH a receipt", "suppressed_with_receipt"),
        ("perceive dispatches", "perceive_dispatches"),
        ("superseded (latest-wins)", "judge_superseded"),
        ("dropped: pending deadline", "judge_deadline_dropped"),
        ("dropped: revision moved", "judge_revision_dropped"),
        ("expired: active decision deadline", "judge_deadline_expired"),
        ("scheduler receipts total", "scheduler_receipted_total"),
        ("admissions LAUNCHED as batches", "admissions_launched_as_batches"),
        ("**admission residual** (admitted - launched - receipted)", "admission_residual"),
        ("dispatch units (frame x goal)", "perceive_dispatches_goal_units"),
        ("dispatches per launched batch", "dispatches_per_batch"),
        ("`judge.started` trace units", "judge_started_trace"),
        ("`judge.finished` trace units", "judge_finished_trace"),
        ("`judge.degraded` trace units", "judge_degraded_trace"),
        ("started - finished - degraded", "started_minus_finished_minus_degraded"),
        ("`gate.classified` trace units", "gate_classified_trace"),
        ("finished - gate.classified", "finished_minus_gate_classified"),
    ):
        A(f"| {label} | {lg.get(key)} |")
    A("")
    fm = lg["funnel_mirror"]
    A(f"Primary ledger balances (ADMISSION unit): **{lg['admission_balances']}** "
      f"(admitted - launched - receipted = {lg['admission_residual']}).")
    A(f"Funnel mirror: `funnel.judged`={fm['funnel.judged']} vs `judge.started` trace "
      f"={lg['judge_started_trace']} (delta {fm['judged_minus_judge_started']}); "
      f"`funnel.refused`={fm['funnel.refused']} vs `gate.ask`={fm['gate.ask']} "
      f"(delta {fm['refused_minus_ask']}, double-count "
      f"{fm['refused_is_double_ask']}).")
    A(f"Token cross-check adapter vs funnel: agree = {lg['token_cross_check']['agree']} "
      f"({lg['token_cross_check']})")
    A("")

    # --- allocation ---
    al = p["allocation"]
    A("## 4. Judge allocation across concurrent goals")
    A("")
    A(f"{al['n_goals_armed']} goals armed; {al['goals_ever_judged']} were judged at least once, "
      f"{al['goals_never_judged']} never were. {al['judged_total']:,} judge dispatches, fair share "
      f"{al['fair_share_per_goal']} per goal, Gini of the per-goal judged count "
      f"**{al['gini_of_judged']}** (0 = perfectly even).")
    A("")
    A(f"Per-goal judged: {al['judged_per_goal']}")
    A(f"Gate actions: {al['gate_actions_total']}; RuntimeStats asks={al['stats_asks']} "
      f"fires={al['stats_fires']}.")
    A(f"Scheduler health: {al['health']['all_kinds']}")
    A("")
    if al.get("spend_cap_snapshot"):
        cs = al["spend_cap_snapshot"]
        A(f"**Spend cap bit at wall {cs.get('wall_s')} s / media {cs.get('media_s')} s** — up to "
          f"that instant the judge was a live provider ({cs.get('judge_started')} dispatches over "
          f"{cs.get('goals_judged_distinct')} goals, {cs.get('incidents')} incidents); after it, "
          f"new attempts fail at the durable ledger before transport and produce no synthetic "
          f"verdict. Allocation numbers spanning both regimes are marked.")
        A("")
        A(f"PRE-CAP ONLY (the live-provider window): {al['pre_cap']}")
        A("")
    A("| # | goal | admitted | judged | finished | fire | ask | silent | incidents |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in al["per_goal"][:80]:
        A(f"| {row['arm_index']} | {row['goal_text'][:44]} | {row['admitted']} | {row['judged']} "
          f"| {row['judge_finished']} | {row['fire']} | {row['ask']} | {row['silent']} "
          f"| {row['incidents']} |")
    A("")

    # --- memory ---
    mem = p["memory"]
    A("## 5. Memory over the horizon")
    A("")
    A(f"RSS start {_mb(mem['rss_start_mb'])} MB -> end {_mb(mem['rss_end_mb'])} MB, peak "
      f"{_mb(mem['rss_peak_mb'])} MB over {mem['samples']} samples. First-third mean "
      f"{_mb(mem['rss_mean_first_third_mb'])} MB, last-third mean "
      f"{_mb(mem['rss_mean_last_third_mb'])} MB "
      f"(growth {_mb(mem['growth_last_minus_first_third_mb'])} MB); least-squares slope "
      f"{_mb(mem['slope_mb_per_hour'])} MB/h.")
    A("")
    A("| wall s | RSS MB |")
    A("|---|---|")
    curve = mem["rss_curve"]
    stride = max(1, len(curve) // 40)
    for row in curve[::stride]:
        A(f"| {row['wall_s']} | {row['rss_mb']} |")
    A("")

    # --- accounting ---
    ac = p["accounting"]
    A("## Provider accounting and spend")
    A("")
    A(f"Model `{ac['model']}`, {ac['physical_calls']:,} physical calls "
      f"plus {ac['budget_capped_calls']} pre-transport budget refusals, {ac['cancelled']:,} "
      f"cancelled at the deadline, {ac['errors']} errors {ac['error_types']}.")
    A(f"Usage rows {ac['usage_rows']:,} ({(ac['completed_fraction'] or 0) * 100:.1f}% of calls); "
      f"input {ac['input_tokens']:,} tok, output {ac['output_tokens']:,} tok; "
      f"**estimated spend ${ac['estimated_cost_usd']}** at "
      f"{ac['price_usd_per_million']}.")
    A(f"Observed {ac['source_stream_hours']:.4f} source-stream hours / "
      f"{ac['scheduled_watch_hours']:.4f} scheduled watch-hours; estimated provider cost "
      f"`${ac['estimated_provider_cost_usd_per_source_hour']}` per source-hour and "
      f"`${ac['estimated_provider_cost_usd_per_scheduled_watch_hour']}` per watch-hour.")
    A("")
    A(f"_{ac['cost_caveat']}_")
    A(f"Launch cost gate: **{ac['launch_cost_gate']['status']}** against the all-in "
      f"${LR.ABSOLUTE_COST_CEILING_USD_PER_SOURCE_HOUR:.2f}/source-hour ceiling — "
      f"{ac['launch_cost_gate']['reason']}.")
    A("")
    A(f"Terminal route outcomes: {ac['terminal_outcomes']}")
    A(f"Provider latency, completed: {ac['provider_latency_completed']}")
    A(f"Provider latency, censored (cancelled/errored — a LOWER bound): "
      f"{ac['provider_latency_censored']}")
    A("")
    A("## Limits")
    A("")
    for line in p["limits"]:
        A(f"- {line}")
    A("")
    return "\n".join(L)


# =================================================================================================
# cli
# =================================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Long-horizon qualification: one source, many watches")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--expect-conductor", action="store_true",
                    help="require complete, goal-kind-correct frozen mechanics before provider setup")
    ap.add_argument("--qualification", action="store_true",
                    help="launch-claim run: require an approved --qualification-contract before "
                         "provider setup")
    ap.add_argument("--qualification-contract", default="",
                    help="frozen launch-bar JSON; its exact sha256 is stamped into the artifact")
    ap.add_argument("--preflight-only", action="store_true",
                    help="validate source, occurrence bounds, and optional conductor pin; no provider")
    ap.add_argument("--webhook-probe", action=argparse.BooleanOptionalAction, default=True,
                    help="deliver through a real local HTTP receiver and record arrival/ACK "
                         "timestamps (default on)")
    ap.add_argument("--max-goals", type=int, default=0, help="0 = every distinct goal")
    ap.add_argument("--decision-deadline-s", type=float, default=0.0,
                    help="0 = scale to the armed-goal count (see the module docstring)")
    ap.add_argument("--judge-concurrency", type=int, default=1,
                    help="K: judge batch fan-out. Floors/is floored by the pool's max_parallel and "
                         "every route's max_inflight, all three set together. K >= armed-goal count "
                         "collapses the _batch_deadline_s wave multiplier to 1. Default 1 = the "
                         "previous sequential behaviour, unchanged.")
    ap.add_argument("--max-provider-usd", type=float, default=None,
                    help="run-local USD allocation; required for paid runs")
    ap.add_argument("--max-provider-calls", type=int, default=None,
                    help="global physical-call cap; required for paid runs")
    ap.add_argument("--conservative-usd-per-call", type=float, default=None,
                    help="amount durably reserved before every provider transport")
    ap.add_argument("--global-prior-committed-usd", type=float, default=None,
                    help="provider dollars already committed outside this run")
    ap.add_argument("--global-prior-commitment-source", default="",
                    help="artifact/path identifying the imported prior commitment")
    ap.add_argument("--global-prior-commitment-basis", default="", choices=(
                        "official-token-settled", "conservative-reservations",
                        "mixed-token-settled-and-conservative"),
                    help="explicit accounting basis of the imported prior commitment")
    ap.add_argument("--provider-request-timeout-s", type=float, default=120.0)
    ap.add_argument("--max-wall-s", type=float, default=0.0)
    ap.add_argument("--sample-s", type=float, default=15.0)
    ap.add_argument("--entity-feed", action="store_true", default=True)
    ap.add_argument("--no-entity-feed", dest="entity_feed", action="store_false")
    ap.add_argument("--episode-probe", action="store_true", default=True)
    ap.add_argument("--no-episode-probe", dest="episode_probe", action="store_false")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--price-source", default="")
    ap.add_argument("--price-version", default="")
    ap.add_argument("--settle-from-official-token-prices", action="store_true",
                    help="settle completed calls from complete token counts and the pinned official "
                         "price; all incomplete/error/cancelled calls retain their reservation")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="resume the durable provider ledger (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--out-stem", default="long_run")
    args = ap.parse_args(argv)

    if args.qualification and not args.qualification_contract:
        raise SystemExit(
            "[long_run] PREFLIGHT FAILED before provider setup: --qualification requires an "
            "approved --qualification-contract"
        )
    args._qualification_contract_summary = None
    if args.qualification_contract:
        try:
            _contract, args._qualification_contract_summary = QC.load(
                args.qualification_contract, require_approved=args.qualification
            )
        except Exception as exc:
            raise SystemExit(
                f"[long_run] PREFLIGHT FAILED before provider setup: {type(exc).__name__}: {exc}"
            ) from exc

    if args.preflight_only:
        manifest = normalize_long_manifest(json.loads(Path(args.manifest).read_text()))
        goals = manifest["goals"][: args.max_goals] if args.max_goals else manifest["goals"]
        try:
            authored = asyncio.run(
                preflight_long_manifest(
                    {**manifest, "goals": goals}, require_conductor=args.expect_conductor
                )
            )
        except Exception as exc:
            raise SystemExit(
                f"[long_run] PREFLIGHT FAILED before provider setup: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        print(json.dumps({
            "status": "ok",
            "manifest_kind": manifest.get("source_manifest_kind"),
            "video_path": manifest["video_path"],
            "source_duration_s": manifest["clock"]["measured_container_duration_s"],
            "n_goals": len(goals),
            "n_occurrences": sum(int(g.get("n_occurrences") or 0) for g in goals),
            "precision_eligible": bool(manifest.get("precision_eligible")),
            "precision_ineligible_reason": manifest.get("precision_ineligible_reason"),
            "conductor_verified": args.expect_conductor,
            "authored_types": dict(Counter(
                getattr(item, "behavior_type", "none") for item in authored
            )),
        }, indent=2))
        return 0

    # Every schema, source, authored-mechanic and exact multi-goal arm check completes on an
    # offline core before a credentialed adapter can be constructed.
    try:
        manifest = normalize_long_manifest(json.loads(Path(args.manifest).read_text()))
        goals = manifest["goals"][: args.max_goals] if args.max_goals else manifest["goals"]
        authored = asyncio.run(
            preflight_long_manifest(
                {**manifest, "goals": goals}, require_conductor=args.expect_conductor
            )
        )
        arm_issues = long_armability_issues(goals, authored)
        if arm_issues:
            raise ValueError("; ".join(arm_issues[:20]))
        paid_policy = provider_budget_policy_from_args(args)
    except Exception as exc:
        raise SystemExit(
            f"[long_run] PREFLIGHT FAILED before provider setup: {type(exc).__name__}: {exc}"
        ) from exc

    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS / f"{args.out_stem}.checkpoint.jsonl"
    budget_ckpt = RESULTS / f"{args.out_stem}.provider-budget.json"
    settings = load_settings()
    pricing = LR.PricingStamp(
        backend="gemini",
        model=str(settings.models.vision_model),
        pricing_version=(
            args.price_version
            or ("unpriced-conservative" if args.price_in is None else "declared-unsettled")
        ),
        pricing_source=(
            args.price_source
            or ("declared on command line" if args.price_in is not None
                else "unpriced; full conservative reservation retained")
        ),
        input_usd_per_million_tokens=float(args.price_in or 0.0),
        output_usd_per_million_tokens=float(args.price_out or 0.0),
    )
    try:
        args._provider_budget = LR.GlobalProviderBudget.open(
            policy=paid_policy,
            pricing=pricing,
            checkpoint_path=budget_ckpt,
            global_prior_committed_usd=float(args.global_prior_committed_usd),
            global_prior_commitment_source=args.global_prior_commitment_source,
            global_prior_commitment_basis=args.global_prior_commitment_basis,
            settle_from_official_token_prices=args.settle_from_official_token_prices,
            resume=args.resume,
            case_checkpoint_exists=ckpt.exists(),
        )
    except (LR.ProviderBudgetInputError, LR.ProviderBudgetExceeded, OSError) as exc:
        raise SystemExit(
            f"[long_run] PREFLIGHT FAILED before provider setup: {type(exc).__name__}: {exc}"
        ) from exc
    args._prepared_manifest = manifest
    args._prepared_authored = authored
    allocation = args._provider_budget.provenance()["global_allocation"]
    print(
        "[long_run] provider budget: "
        f"${allocation['prior_committed_usd']:.2f} prior + "
        f"${allocation['live_run_local_max_usd']:.2f} long-run allocation <= "
        f"${LR.ABSOLUTE_PROVIDER_RUN_CAP_USD:.2f} total; "
        f"{paid_policy.max_calls} physical calls max",
        file=sys.stderr,
    )

    webhook_probe = None
    if args.webhook_probe:
        from percept.harness.webhook_probe import LoopbackWebhookProbe

        webhook_probe = LoopbackWebhookProbe(sink_timeout_s=5.0).start()
        args._webhook_probe = webhook_probe
        print(f"[long_run] webhook receiver probe: {webhook_probe.url}", file=sys.stderr)
    try:
        payload = asyncio.run(run_long(args))
    finally:
        if webhook_probe is not None:
            webhook_probe.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.out_stem}.json").write_text(json.dumps(payload, indent=1, default=str))
    (RESULTS / f"{args.out_stem}.md").write_text(render_md(payload))
    print(f"[long_run] wrote {RESULTS / (args.out_stem + '.json')}", file=sys.stderr)
    print(f"[long_run] wrote {RESULTS / (args.out_stem + '.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
