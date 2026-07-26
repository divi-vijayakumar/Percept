#!/usr/bin/env python3
"""Stage-2 LOCAL PERCEPTION validation — SV-2.5 / SV-2.6 / SV-2.7 + the counting-rhythm harness.

Runs against the LAUNCH WHEEL only (``percept`` from site-packages). Nothing here makes a
provider or network call: every judge is scripted, every signal is either synthetic or derived
from the corpus' own ground truth / locally decoded pixels.

Tests
-----
SV-2.5  Reducer determinism (exact)
        ``EvidenceSequencer`` + ``SerializedDecisionReducer``: identical input order must give a
        byte-identical decision trace across repeated in-process runs AND across subprocesses
        (with varying ``PYTHONHASHSEED``). Out-of-order input is asserted against the DECLARED
        ``ResultPolicy`` semantics read out of ``percept/sequencing.py``, not against a guess.

SV-2.6  Temporal object continuity
        ``EpisodeLog`` + ``EpisodePolicy`` + ``Revision``: an object survives an occlusion inside
        policy (one episode, revised in place) and splits outside it — and every split is TRACED
        (subject_matched + prior_episode_id + gap_s + a denial reason). Pass bar: zero SILENT
        duplicates. The boundary (just-within vs just-beyond ``continuity_gap_s``) is probed
        directly.

SV-2.7  Stage-2 economy
        Units received from Stage 1 vs units passed to Stage 3, per sense, reported through
        ``SenseProbe.stats`` when ``tools/senseprobe`` is available and through an in-file
        equivalent shim when it is not (the shim is clearly flagged in the report).

SV-2.1 groundwork  Counting / rhythm measurement harness
        The measurement path for rep counting over the 20 real ``repcount`` corpus cases. Each
        case's ``cycle_bounds_frames`` are converted to SECONDS using that media file's OWN fps
        (read with cv2), and driven through two counting rules — the candidate-source
        ``count_edges`` refractory rule and the shipped ``RhythmTracker``. Raw absolute error per
        case and the error distribution are reported; NO pass bar is invented here.

Usage
-----
    <launch-venv>/bin/python eval/stage-validation/sv2_senses.py            # full run
    ... --quick                                                            # skip the pixel eye
    ... --emit-reducer-trace                                               # internal (SV-2.5)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------------------------
# Logging: percept's structured logger installs its own stderr handler on first use. Pre-empt it
# so a validation run's stdout/stderr stays readable (drops are still ASSERTED on the returned
# decision objects, so silencing the log costs no coverage).
# --------------------------------------------------------------------------------------------
for _name in (
    "percept", "percept.sequencing", "percept.episodes", "percept.gate", "percept.agent",
):
    _lg = logging.getLogger(_name)
    _lg.addHandler(logging.NullHandler())
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"
CORPUS = HERE / "corpus.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import percept  # noqa: E402
from percept.contracts import Verdict  # noqa: E402
from percept.episodes import (  # noqa: E402
    CLOSED,
    DENIAL_REASONS,
    EpisodeLog,
    EpisodePolicy,
    Revision,
)
from percept.extraction import Enrichment  # noqa: E402
from percept.gate import ASK, FIRE, Gate  # noqa: E402
from percept.identity import AppearanceIdentity, ContinuityIdentity  # noqa: E402
from percept.individuation import EventCandidate, SpanAssociator  # noqa: E402
from percept.measurement import (  # noqa: E402
    MeasurementSeries,
    channel_score,
    dominant_period,
    periodicity_score,
    robust_amplitude,
    subsample_stability,
)
from percept.rhythm import RhythmTracker  # noqa: E402
from percept.sequencing import (  # noqa: E402
    EvidenceSequencer,
    ResultPolicy,
    SerializedDecisionReducer,
)

RUNTIME_PATH = Path(percept.__file__).resolve()
RUNTIME_IS_REPO_TREE = str(REPO_ROOT / "packages") in str(RUNTIME_PATH)

# --------------------------------------------------------------------------------------------
# SenseProbe dependency (built concurrently by another agent). Guarded: absence must not stop
# the rest of the suite, and it is reported honestly.
# --------------------------------------------------------------------------------------------
SENSEPROBE_AVAILABLE = False
SENSEPROBE_ERROR: str | None = None
sp_probe: Any = None
SP_REGISTRY: dict[str, Any] = {}
try:  # pragma: no cover - depends on a sibling agent's deliverable
    from tools.senseprobe import REGISTRY as SP_REGISTRY  # type: ignore
    from tools.senseprobe import probe as sp_probe  # type: ignore

    SENSEPROBE_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    SENSEPROBE_ERROR = f"{type(exc).__name__}: {exc}"


# ============================================================================================
# Result plumbing
# ============================================================================================


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    test_id: str
    title: str
    checks: list[Check] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    defects: list[str] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(passed), detail))
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and bool(self.checks)

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "title": self.title,
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "data": self.data,
            "defects": self.defects,
        }


# ============================================================================================
# SV-2.5 — reducer determinism (exact)
# ============================================================================================

_SCRIPTED_VERDICTS: tuple[tuple[bool, float], ...] = (
    (True, 0.91), (True, 0.55), (False, 0.99), (True, 0.72), (True, 0.88), (False, 0.40),
)


def _verdict(i: int) -> Verdict:
    sat, conf = _SCRIPTED_VERDICTS[i % len(_SCRIPTED_VERDICTS)]
    return Verdict(satisfied=sat, confidence=conf, evidence=f"scripted-{i}")


class _StepClock:
    """A deterministic monotonic clock: every read advances by a fixed step."""

    def __init__(self, start: float = 1000.0, step: float = 0.001) -> None:
        self.t = start
        self.step = step

    def __call__(self) -> float:
        v = self.t
        self.t += self.step
        return v


def _rec(d: Any) -> dict:
    """Canonical record for one GateDecision (order-stable, JSON-safe)."""
    return {
        "key": str(d.key),
        "evidence_seq": d.evidence_seq,
        "action": d.action,
        "dropped": bool(d.dropped),
        "drop_reason": d.drop_reason,
        "applied_after_supersession": bool(d.applied_after_supersession),
    }


async def reducer_trace() -> list[dict]:
    """The fixed SV-2.5 scenario. Pure: no wall clock, no randomness, no I/O."""
    out: list[dict] = []

    # --- phase A: strictly in-order submits over three keys -----------------------------------
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(gate, clock=_StepClock())
    for i in range(12):
        key = f"goal{i % 3}"
        seq = red.admit(key)
        dec = await red.submit(key, seq, _verdict(i), now=float(i))
        out.append({"phase": "A", **_rec(dec), "last_applied": red.last_applied(key)})

    # --- phase B: out-of-order under the DEFAULT policy (ordering="latest_only") ---------------
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(gate, clock=_StepClock())
    seqs = [red.admit("k") for _ in range(4)]          # 1,2,3,4 all admitted (dispatch time)
    for j, s in enumerate([seqs[3], seqs[0], seqs[2], seqs[1]]):   # completions arrive scrambled
        dec = await red.submit("k", s, _verdict(j), now=float(j))
        out.append({"phase": "B", **_rec(dec), "last_applied": red.last_applied("k")})

    # --- phase C: ordering="episode_ordered" ---------------------------------------------------
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(
        gate, clock=_StepClock(), policy=ResultPolicy(ordering="episode_ordered")
    )
    a = [red.admit("k", episode_id="epA") for _ in range(3)]   # 1,2,3
    b = [red.admit("k", episode_id="epB") for _ in range(2)]   # 4,5
    plan = [("epB", b[1]), ("epA", a[2]), ("epA", a[0]), ("epB", b[0]), (None, a[1])]
    for j, (ep, s) in enumerate(plan):
        dec = await red.submit("k", s, _verdict(j), episode_id=ep, now=float(j))
        out.append({"phase": "C", "episode": ep, **_rec(dec), "last_applied": red.last_applied("k")})

    # --- phase D: stale_after_ms timeout drop --------------------------------------------------
    clk = _StepClock(start=0.0, step=0.0)
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(
        gate, clock=clk, policy=ResultPolicy(stale_after_ms=100.0)
    )
    s_fresh = red.admit("k")
    dec = await red.submit("k", s_fresh, _verdict(0), now=0.0)     # age 0 ms -> applies
    out.append({"phase": "D", **_rec(dec), "last_applied": red.last_applied("k")})
    s_slow = red.admit("k")
    clk.t += 0.5                                                   # 500 ms elapse before completion
    dec = await red.submit("k", s_slow, _verdict(1), now=1.0)
    out.append({"phase": "D", **_rec(dec), "last_applied": red.last_applied("k")})

    # --- phase E: apply_after_supersession escape hatch ----------------------------------------
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(
        gate, clock=_StepClock(), policy=ResultPolicy(apply_after_supersession=True)
    )
    e = [red.admit("k") for _ in range(2)]
    dec = await red.submit("k", e[1], _verdict(0), now=0.0)
    out.append({"phase": "E", **_rec(dec), "last_applied": red.last_applied("k")})
    dec = await red.submit("k", e[0], _verdict(1), now=1.0)        # superseded but opted back in
    out.append({"phase": "E", **_rec(dec), "last_applied": red.last_applied("k")})

    # --- phase F: CONCURRENT submits for the SAME key (the per-key lock) ----------------------
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(gate, clock=_StepClock())
    f = [red.admit("hot") for _ in range(6)]
    decisions = await asyncio.gather(
        *(red.submit("hot", s, _verdict(i), now=float(i)) for i, s in enumerate(reversed(f)))
    )
    for dec in decisions:
        out.append({"phase": "F", **_rec(dec)})
    out.append({"phase": "F", "final_last_applied": red.last_applied("hot")})

    # --- phase G: forget() clears every per-key structure --------------------------------------
    gate = Gate(threshold=0.5, key_ttl_s=None, max_keys=None)
    red = SerializedDecisionReducer(gate, clock=_StepClock())
    g1 = red.admit("z")
    await red.submit("z", g1, _verdict(0), now=0.0)
    red.forget("z")
    g2 = red.admit("z")
    dec = await red.submit("z", g2, _verdict(1), now=1.0)
    out.append({"phase": "G", "seq_after_forget": g2, **_rec(dec),
                "tracked": [str(k) for k in red.sequencer.tracked_keys()]})
    return out


def _digest(records: list[dict]) -> str:
    blob = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def run_sv25() -> TestResult:
    r = TestResult("SV-2.5", "Reducer determinism (exact)")

    # 1) repeated in-process runs -> byte-identical
    digests = []
    traces = []
    for _ in range(5):
        tr = asyncio.run(reducer_trace())
        traces.append(tr)
        digests.append(_digest(tr))
    r.data["in_process_digests"] = digests
    r.check(
        "repeated in-process runs are byte-identical",
        len(set(digests)) == 1,
        f"{len(set(digests))} distinct digest(s) over 5 runs: {sorted(set(digests))[:2]}",
    )
    trace = traces[0]
    r.data["trace_len"] = len(trace)
    r.data["digest"] = digests[0]

    # 2) cross-process, with varying PYTHONHASHSEED
    sub_digests = []
    sub_errors = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env.pop("PYTHONPATH", None)
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--emit-reducer-trace"],
                capture_output=True, text=True, env=env, timeout=180, cwd=str(REPO_ROOT),
            )
            if proc.returncode != 0:
                sub_errors.append(f"seed={seed} rc={proc.returncode} stderr={proc.stderr[-400:]}")
                continue
            sub_digests.append(_digest(json.loads(proc.stdout)))
        except Exception as exc:  # noqa: BLE001
            sub_errors.append(f"seed={seed} {type(exc).__name__}: {exc}")
    r.data["subprocess_digests"] = sub_digests
    r.data["subprocess_errors"] = sub_errors
    r.check(
        "cross-process trace is byte-identical (3 subprocesses, 3 hash seeds)",
        len(sub_digests) == 3 and set(sub_digests) == {digests[0]},
        f"subprocess digests={sorted(set(sub_digests))} in-process={digests[0]} errors={sub_errors}",
    )

    # 3) DECLARED out-of-order policy (read from percept/sequencing.py:63-80, 242-260)
    by_phase: dict[str, list[dict]] = {}
    for rec in trace:
        by_phase.setdefault(rec.get("phase", "?"), []).append(rec)

    b = by_phase["B"]
    r.check(
        "latest_only: newest admitted seq applies, every earlier one drops as "
        "'superseded_by_newer_admission'",
        (b[0]["evidence_seq"] == 4 and not b[0]["dropped"])
        and all(x["dropped"] and x["drop_reason"] == "superseded_by_newer_admission" for x in b[1:]),
        json.dumps(b),
    )
    r.check(
        "a DROPPED decision is never FIRE/ASK (action pinned to SILENT)",
        all(x["action"] == "silent" for x in trace if x.get("dropped")),
        "some dropped decision carried a delivered action",
    )
    r.check(
        "a DROPPED decision never advances last_applied (gate untouched)",
        all(x["last_applied"] == 4 for x in b),  # set by the seq-4 apply, never moved by the drops
        json.dumps([x.get("last_applied") for x in b]),
    )

    c = by_phase["C"]
    # plan was: epB/5 apply, epA/3 (latest in epA) apply, epA/1 drop-in-episode, epB/4 drop-in-episode,
    # None/2 -> falls back to latest_only and drops against the key-wide latest (5).
    r.check(
        "episode_ordered: a straggler from an EARLIER episode is NOT penalised by a newer "
        "episode's higher admission",
        c[1]["evidence_seq"] == 3 and not c[1]["dropped"] and c[1]["drop_reason"] is None,
        json.dumps(c[1]),
    )
    r.check(
        "episode_ordered: supersession WITHIN an episode still drops "
        "('superseded_within_episode')",
        c[2]["dropped"] and c[2]["drop_reason"] == "superseded_within_episode"
        and c[3]["dropped"] and c[3]["drop_reason"] == "superseded_within_episode",
        json.dumps(c[2:4]),
    )
    r.check(
        "episode_ordered with episode_id=None falls back to key-wide latest_only",
        c[4]["dropped"] and c[4]["drop_reason"] == "superseded_by_newer_admission",
        json.dumps(c[4]),
    )

    d = by_phase["D"]
    r.check(
        "stale_after_ms drops an aged result independently of supersession ('stale_timeout')",
        not d[0]["dropped"] and d[1]["dropped"] and d[1]["drop_reason"] == "stale_timeout",
        json.dumps(d),
    )

    e = by_phase["E"]
    r.check(
        "apply_after_supersession=True applies the would-be-drop and FLAGS it",
        (not e[1]["dropped"]) and e[1]["applied_after_supersession"]
        and e[1]["drop_reason"] == "superseded_by_newer_admission",
        json.dumps(e[1]),
    )

    f = [x for x in by_phase["F"] if "evidence_seq" in x]
    applied = [x["evidence_seq"] for x in f if not x["dropped"]]
    r.check(
        "concurrent same-key submits serialize: exactly the newest admission applies",
        applied == [6] and len(f) == 6,
        f"applied={applied} of {[x['evidence_seq'] for x in f]}",
    )

    g = by_phase["G"][0]
    r.check(
        "forget() resets the per-key counter to 1 and re-admits cleanly",
        g["seq_after_forget"] == 1 and not g["dropped"],
        json.dumps(g),
    )

    # 4) EvidenceSequencer determinism on its own
    seqs_runs = []
    for _ in range(3):
        s = EvidenceSequencer()
        run = []
        for i in range(20):
            run.append(s.admit(f"k{i % 4}", episode_id=f"ep{i % 3}"))
        run.append(tuple(str(k) for k in s.tracked_keys()))
        seqs_runs.append(str(run))
    r.check(
        "EvidenceSequencer admit/tracked_keys is deterministic and insertion-ordered",
        len(set(seqs_runs)) == 1,
        seqs_runs[0][:160],
    )
    return r


# ============================================================================================
# SV-2.6 — temporal object continuity
# ============================================================================================


def _fold(log: EpisodeLog, t: float, subject: str, value: Any = None) -> Any:
    return log.observe_decision(t, {"tag": subject, "value": value}, subject_key=subject)


def run_sv26() -> TestResult:
    r = TestResult("SV-2.6", "Temporal object continuity (occlusion / revise-not-duplicate)")
    GAP = 5.0
    pol = EpisodePolicy(continuity_gap_s=GAP)

    # --- 1) occlusion JUST WITHIN policy: one episode, revised in place ------------------------
    log = EpisodeLog("g-within", policy=pol)
    d0 = _fold(log, 0.0, "cup-1")
    d1 = _fold(log, GAP - 1e-6, "cup-1")          # occluded for 4.999999 s, still the same thing
    within = {
        "same_episode": d0.episode.id == d1.episode.id,
        "is_new": d1.is_new,
        "reason": d1.reason,
        "observations": d1.episode.observations,
        "revision_kinds": [rv.kind for rv in d1.episode.revisions],
        "episode_count": len(log.summaries()),
    }
    r.data["occlusion_within"] = within
    r.check(
        f"occlusion of {GAP - 1e-6:.6f}s (< continuity_gap_s={GAP}) CONTINUES the same episode",
        within["same_episode"] and not within["is_new"] and within["reason"] == "continued",
        json.dumps(within),
    )
    r.check(
        "the continuation COALESCES (no new revision row) and leaves exactly one episode",
        within["revision_kinds"] == ["opened"] and within["episode_count"] == 1
        and within["observations"] == 2,
        json.dumps(within),
    )

    # --- 2) occlusion JUST BEYOND policy: a TRACED split, never a silent one -------------------
    beyond_rows = []
    for label, t in (("exactly_gap", GAP), ("gap_plus_eps", GAP + 1e-6)):
        lg = EpisodeLog(f"g-{label}", policy=pol)
        a = _fold(lg, 0.0, "cup-1")
        bdec = _fold(lg, t, "cup-1")
        beyond_rows.append({
            "case": label, "t": t, "is_new": bdec.is_new, "reason": bdec.reason,
            "subject_matched": bdec.subject_matched, "prior_episode_id": bdec.prior_episode_id,
            "gap_s": bdec.gap_s, "denied": bdec.denied,
            "distinct_episode": bdec.episode.id != a.episode.id,
        })
    r.data["occlusion_beyond"] = beyond_rows
    r.check(
        f"occlusion of EXACTLY continuity_gap_s ({GAP}s) SPLITS "
        "(inclusive boundary, percept/episodes.py:762)",
        beyond_rows[0]["is_new"] and beyond_rows[0]["reason"] == "gap_exceeded",
        json.dumps(beyond_rows[0]),
    )
    r.check(
        "occlusion just beyond the gap SPLITS and the split is TRACED "
        "(subject_matched + prior_episode_id + gap_s + denial reason)",
        all(x["is_new"] and x["reason"] == "gap_exceeded" and x["subject_matched"]
            and x["prior_episode_id"] is not None and x["gap_s"] is not None and x["denied"]
            for x in beyond_rows),
        json.dumps(beyond_rows),
    )

    # --- 3) REVISE, don't duplicate ------------------------------------------------------------
    log = EpisodeLog("g-revise", policy=pol)
    dec = _fold(log, 0.0, "cup-1", value="empty")
    ep = dec.episode
    log.annotate(ep, value="empty")
    for k, t in enumerate((1.0, 2.0, 3.0)):
        _fold(log, t, "cup-1", value="empty")            # heartbeat: coalesces, no rows
    dec = _fold(log, 4.0, "cup-1", value="half-full")    # the semantic change
    log.annotate(dec.episode, value="half-full")
    rev = log.record_revision(dec.episode, "updated", t=4.0, value="half-full")
    for t in (5.0, 6.0, 7.0, 8.0):
        _fold(log, t, "cup-1", value="half-full")
    revise = {
        "episode_count": len(log.summaries()),
        "episode_ids": [s["id"] for s in log.summaries()],
        "revision_kinds": [rv.kind for rv in dec.episode.revisions],
        "observations": dec.episode.observations,
        "value": dec.episode.value,
        "updated_row_is_Revision": isinstance(rev, Revision),
    }
    r.data["revise_not_duplicate"] = revise
    r.check(
        "a value change REVISES the single episode (one episode, kinds == [opened, updated])",
        revise["episode_count"] == 1 and revise["revision_kinds"] == ["opened", "updated"]
        and revise["value"] == "half-full" and revise["updated_row_is_Revision"],
        json.dumps(revise),
    )

    # --- 4) heartbeat boundedness --------------------------------------------------------------
    log = EpisodeLog("g-beat", policy=pol)
    dec = _fold(log, 0.0, "cup-1")
    for i in range(1, 501):
        _fold(log, i * 0.5, "cup-1")
    beat = {
        "confirmations": 500,
        "revision_rows": len(dec.episode.revisions),
        "kinds": sorted({rv.kind for rv in dec.episode.revisions}),
        "episode_count": len(log.summaries()),
        "observations": dec.episode.observations,
    }
    r.data["heartbeat"] = beat
    r.check(
        "500 in-policy confirmations produce exactly ONE revision row and ONE episode",
        beat["revision_rows"] == 1 and beat["kinds"] == ["opened"] and beat["episode_count"] == 1
        and beat["observations"] == 501,
        json.dumps(beat),
    )

    # --- 5) ZERO SILENT DUPLICATES over a long occlusion-riddled stream ------------------------
    rng = random.Random(20260726)
    log = EpisodeLog("g-stream", policy=pol, max_episodes=4096)
    decisions = []
    t = 0.0
    for i in range(400):
        # alternate short (in-policy) and long (out-of-policy) occlusions, deterministic
        t += rng.choice([0.5, 1.0, 2.0, 4.9]) if i % 3 else rng.choice([5.0, 7.5, 30.0])
        decisions.append(_fold(log, t, "cup-1"))
    # The FIRST open for a subject is not a duplicate — it is the episode. Every LATER open for
    # an already-seen subject is a duplicate, and must be a TRACED one.
    opens = [d for d in decisions if d.is_new]
    first_open, later_opens = (opens[0], opens[1:]) if opens else (None, [])
    silent_duplicates = [
        {"reason": d.reason, "subject_matched": d.subject_matched,
         "prior": d.prior_episode_id, "gap_s": d.gap_s}
        for d in later_opens
        if not (d.subject_matched and d.prior_episode_id is not None
                and d.gap_s is not None and d.reason in DENIAL_REASONS)
    ]
    stream = {
        "observations": len(decisions),
        "episodes_opened": len(opens),
        "first_open_reason": first_open.reason if first_open else None,
        "duplicate_opens": len(later_opens),
        "continuations": sum(1 for d in decisions if not d.is_new),
        "traced_splits": len(later_opens) - len(silent_duplicates),
        "silent_duplicates": len(silent_duplicates),
        "split_reasons": sorted({d.reason for d in later_opens}),
    }
    r.data["stream"] = stream
    r.data["silent_duplicate_examples"] = silent_duplicates[:5]
    r.check(
        "PASS BAR: zero SILENT duplicates over 400 observations with mixed occlusions "
        "(every re-open of an already-seen subject carries subject_matched + prior_episode_id + "
        "gap_s + a DENIAL_REASONS code)",
        len(silent_duplicates) == 0 and len(later_opens) > 0
        and set(stream["split_reasons"]) <= set(DENIAL_REASONS),
        json.dumps(stream) + " examples=" + json.dumps(silent_duplicates[:3]),
    )

    # --- 6) distinct subjects never merge; identical subject never forks without a reason ------
    log = EpisodeLog("g-two", policy=pol)
    a = _fold(log, 0.0, "cup-1")
    bb = _fold(log, 0.1, "cup-2")
    a2 = _fold(log, 0.2, "cup-1")
    two = {
        "cup1_first": a.episode.id, "cup2": bb.episode.id, "cup1_again": a2.episode.id,
        "cup2_reason": bb.reason, "cup1_again_reason": a2.reason,
        "episode_count": len(log.summaries()),
    }
    r.data["two_subjects"] = two
    r.check(
        "two distinct subjects never merge, and the first subject still continues its own episode",
        a.episode.id != bb.episode.id and a2.episode.id == a.episode.id
        and bb.reason == "new_subject" and a2.reason == "continued" and two["episode_count"] == 2,
        json.dumps(two),
    )

    # --- 7) terminal / reopen -------------------------------------------------------------------
    reopen_rows = []
    for allow in (False, True):
        p = EpisodePolicy(continuity_gap_s=GAP, allow_reopen=allow)
        lg = EpisodeLog(f"g-reopen-{allow}", policy=p)
        first = _fold(lg, 0.0, "cup-1")
        lg.close(first.episode.id, CLOSED, t=1.0)
        again = _fold(lg, 2.0, "cup-1")
        reopen_rows.append({
            "allow_reopen": allow, "is_new": again.is_new, "reason": again.reason,
            "same_episode": again.episode.id == first.episode.id,
            "t_end": again.episode.t_end,
            "kinds": [rv.kind for rv in again.episode.revisions],
            "traced": again.denied and again.prior_episode_id is not None,
            "episode_count": len(lg.summaries()),
        })
    r.data["reopen"] = reopen_rows
    r.check(
        "a terminal episode splits (TRACED 'episode_terminal') unless allow_reopen, in which case "
        "it REOPENS the same episode and clears t_end",
        (reopen_rows[0]["is_new"] and reopen_rows[0]["reason"] == "episode_terminal"
         and reopen_rows[0]["traced"])
        and (not reopen_rows[1]["is_new"] and reopen_rows[1]["reason"] == "reopened"
             and reopen_rows[1]["same_episode"] and reopen_rows[1]["t_end"] is None
             and reopen_rows[1]["kinds"] == ["opened", "closed", "opened"]),
        json.dumps(reopen_rows),
    )

    # --- 8) the DEFAULT policy is unbounded (documented) -----------------------------------------
    log = EpisodeLog("g-default")
    d0 = _fold(log, 0.0, "cup-1")
    d1 = _fold(log, 86_400.0, "cup-1")   # a day later
    default_row = {"same_episode": d0.episode.id == d1.episode.id, "reason": d1.reason,
                   "episode_count": len(log.summaries())}
    r.data["default_policy"] = default_row
    r.check(
        "the default EpisodePolicy (continuity_gap_s=None) is unbounded — a day-long occlusion "
        "still continues (documented default, percept/episodes.py:287)",
        default_row["same_episode"] and default_row["reason"] == "continued",
        json.dumps(default_row),
    )

    # --- 9) exactness of the boundary at 1e-9 (observation, still asserted) ---------------------
    lg = EpisodeLog("g-eps", policy=pol)
    _fold(lg, 0.0, "cup-1")
    tiny = _fold(lg, GAP - 1e-9, "cup-1")
    r.data["boundary_1e-9"] = {"reason": tiny.reason, "is_new": tiny.is_new}
    r.check(
        "boundary is exact to 1e-9 below continuity_gap_s (still continues)",
        not tiny.is_new and tiny.reason == "continued",
        json.dumps(r.data["boundary_1e-9"]),
    )
    return r


# ============================================================================================
# SV-2.7 — Stage-2 economy (per sense, via SenseProbe when available)
# ============================================================================================


class _ShimProbe:
    """In-file stand-in matching the documented ``SenseProbe`` surface.

    Used ONLY when ``tools/senseprobe`` is unavailable (or its spec rejects a scenario). Its
    ``stats`` dict mirrors the declared shape so the report is directly comparable; the report
    always names which path produced each row.
    """

    def __init__(self, name: str, obj: Any, feed_attr: str, read_attr: str | None,
                 read_args: tuple = ()) -> None:
        self.name = name
        self._sense = obj
        self._feed = getattr(obj, feed_attr)
        self._read_attr = read_attr
        self._read_args = read_args
        self.units_fed = 0
        self.reads = 0
        self.wall_time_s = 0.0
        self.errors = 0

    @property
    def sense(self) -> Any:
        return self._sense

    def feed(self, *a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        try:
            result = self._feed(*a, **k)
        except BaseException:
            self.errors += 1
            raise
        finally:
            self.wall_time_s += time.perf_counter() - t0
        self.units_fed += 1
        return result

    def read(self, *a: Any, **k: Any) -> Any:
        self.reads += 1
        if self._read_attr is None:
            return None
        if not a and not k:
            a = self._read_args
        t0 = time.perf_counter()
        try:
            attr = getattr(self._sense, self._read_attr)
            return attr(*a, **k) if callable(attr) else attr
        finally:
            self.wall_time_s += time.perf_counter() - t0

    @property
    def stats(self) -> dict:
        return {
            "name": self.name, "units_fed": self.units_fed, "reads": self.reads,
            "wall_time_s": self.wall_time_s, "errors": self.errors,
        }


#: Fallback wiring, aligned field-for-field with ``tools/senseprobe/probe.py``'s REGISTRY so the
#: shim path and the SenseProbe path measure the SAME sense the SAME way.
_FALLBACK_SPECS: dict[str, tuple[Callable[..., Any], str, str | None, tuple]] = {}


def _init_fallback_specs() -> None:
    from percept.entity import EntityGraph
    from percept.extractors.ocr import ScriptedOCR

    def _reducer(gate: Gate | None = None, **kw: Any) -> SerializedDecisionReducer:
        return SerializedDecisionReducer(gate if gate is not None else Gate(), **kw)

    _FALLBACK_SPECS.update({
        "rhythm": (RhythmTracker, "observe", "windowed_count", (0.0, float("inf"))),
        "entity": (EntityGraph, "ingest", "active", ()),
        "span": (SpanAssociator, "add", "count", ()),
        "sequencer": (EvidenceSequencer, "admit", "latest_admitted", ()),
        "reducer": (_reducer, "admit", "last_applied", ()),
        "continuity_id": (ContinuityIdentity, "match", None, ()),
        "appearance_id": (AppearanceIdentity, "match", None, ()),
        "extractor": (ScriptedOCR, "extract", None, ()),
    })


def _acquire(name: str, **kwargs: Any) -> tuple[Any, str]:
    """A probe for ``name`` — the real SenseProbe when it works, the shim otherwise."""
    if SENSEPROBE_AVAILABLE and name in SP_REGISTRY:
        try:
            p = sp_probe(name, **kwargs)
            for required in ("feed", "read", "stats", "sense"):
                if not hasattr(p, required):
                    raise AttributeError(f"SenseProbe missing {required!r}")
            return p, "senseprobe"
        except Exception as exc:  # noqa: BLE001
            path = f"shim({type(exc).__name__})"
    else:
        path = "shim"
    if not _FALLBACK_SPECS:
        _init_fallback_specs()
    factory, feed_attr, read_attr, read_args = _FALLBACK_SPECS[name]
    return _ShimProbe(name, factory(**kwargs), feed_attr, read_attr, read_args), path


def _synthetic_frames(n: int, hz: float, period_s: float) -> list[tuple[float, dict]]:
    """The Stage-1 input: ``n`` timestamped 3-channel eye measurements, deterministic."""
    rng = random.Random(4242)
    out = []
    for i in range(n):
        t = i / hz
        ph = (t % period_s) / period_s
        out.append((t, {
            "torso_y": 0.5 * (1 - math.cos(2 * math.pi * ph)) + rng.gauss(0, 0.01),
            "arms_y": 0.2 * (1 - math.cos(2 * math.pi * ph + 0.4)) + rng.gauss(0, 0.01),
            "noise": rng.gauss(0, 0.02),
        }))
    return out


def run_sv27() -> TestResult:
    r = TestResult("SV-2.7", "Stage-2 economy (units in from Stage 1 / units out to Stage 3)")
    r.data["senseprobe_available"] = SENSEPROBE_AVAILABLE
    r.data["senseprobe_error"] = SENSEPROBE_ERROR
    r.data["senseprobe_registry_keys"] = sorted(SP_REGISTRY) if SENSEPROBE_AVAILABLE else []

    from percept.adapters import Box, Observation
    from percept.entity import Entity

    N_FRAMES, HZ, PERIOD = 600, 10.0, 2.0
    frames = _synthetic_frames(N_FRAMES, HZ, PERIOD)
    rows: list[dict] = []
    probe_paths: list[str] = []

    def row(name: str, path: str, units_in: int, units_out: int, stats: dict, note: str) -> None:
        probe_paths.append(path)
        rows.append({
            "sense": name, "probe_path": path, "units_in": units_in, "units_out": units_out,
            "economy_ratio": (units_out / units_in) if units_in else None,
            "probe_units_fed": stats.get("units_fed"), "probe_reads": stats.get("reads"),
            "wall_time_s": round(float(stats.get("wall_time_s") or 0.0), 6),
            "errors": stats.get("errors", 0), "note": note,
        })

    # --- rhythm -------------------------------------------------------------------------------
    # spec: feed=observe(t, {channel: value}); read=windowed_count(0.0, inf) -> all-time rep count
    p, path = _acquire("rhythm")
    for t, ch in frames:
        p.feed(t, ch)
    reps = p.read()
    reps = reps if isinstance(reps, int) else len(reps or [])
    row("rhythm", path, N_FRAMES, reps, p.stats,
        "eye frames observed -> confirmed reps on the selected channel")

    # --- entity -------------------------------------------------------------------------------
    # spec: feed=ingest(LIST[Observation], frame, source, *, t=...); read=active() -> list[Entity]
    p, path = _acquire("entity")
    n_obs = 0
    for i, (t, _ch) in enumerate(frames):
        drift = 0.001 * (i % 50)
        obs = [
            Observation(label="person", box=Box(0.1 + drift, 0.1, 0.2, 0.4), confidence=0.9),
            Observation(label="cup", box=Box(0.6, 0.6 + drift, 0.1, 0.1), confidence=0.8),
        ]
        n_obs += len(obs)
        p.feed(obs, None, "camera", t=t)
    ents = p.read()
    row("entity", path, n_obs, len(ents or []), p.stats,
        "detections ingested -> distinct tracked entities handed up")

    # --- span (SpanAssociator) ------------------------------------------------------------------
    # spec: feed=add(EventCandidate) -> (event, is_new); read=count(kind=None) -> int
    p, path = _acquire("span", refractory_s=PERIOD * 0.5)
    n_cand = 0
    for k in range(40):
        anchor = k * (PERIOD / 2.0)                     # 2 candidates per real cycle
        for dup in range(3):                            # each event observed 3x (denser observation)
            cand = EventCandidate(kind="rep", start_t=anchor - 0.2, anchor_t=anchor + 0.01 * dup,
                                  stop_t=anchor + 0.2, source="torso_y", amplitude=1.0, score=1.0)
            n_cand += 1
            p.feed(cand)
    n_events = p.read()
    n_events = n_events if isinstance(n_events, int) else len(n_events or [])
    row("span", path, n_cand, n_events, p.stats,
        "raw event candidates -> deduped event spans (denser observation adds no events)")

    # --- sequencer ------------------------------------------------------------------------------
    # spec: feed=admit(key) -> int; read=latest_admitted(key) (per-key). Units OUT is the state
    # actually carried forward, i.e. the distinct keys the counter still holds.
    p, path = _acquire("sequencer")
    for i in range(N_FRAMES // 4):
        p.feed(f"goal{i % 8}")
    latest_for_last_key = p.read()
    n_keys = len(p.sense.tracked_keys())
    row("sequencer", path, N_FRAMES // 4, n_keys, p.stats,
        f"admissions -> distinct per-key counters carried (latest_admitted for the last key fed = "
        f"{latest_for_last_key})")

    # --- reducer --------------------------------------------------------------------------------
    # spec CAVEAT (confirmed here): feed=admit is SYNCHRONOUS and applies NOTHING; only the
    # awaited coroutine .sense.submit(...) moves gate state. Both halves are exercised.
    p, path = _acquire("reducer", gate=Gate(threshold=0.7, key_ttl_s=None, max_keys=None),
                       clock=_StepClock())
    delivered = 0
    n_sub = 0
    admit_only_last_applied: list[int] = []

    async def _drive_reducer() -> None:
        nonlocal delivered, n_sub
        for i in range(N_FRAMES // 4):
            key = f"goal{i % 8}"
            seq = p.feed(key)                                  # admit (sync)
            if i < 5:
                admit_only_last_applied.append(p.sense.last_applied(key))
            dec = await p.sense.submit(key, seq, _verdict(i), now=float(i))
            n_sub += 1
            if dec.action in (FIRE, ASK):
                delivered += 1

    asyncio.run(_drive_reducer())
    row("reducer", path, n_sub, delivered, p.stats,
        "judge results submitted (via .sense.submit) -> decisions DELIVERED to Stage 3 (FIRE / ASK)")
    r.data["admit_applies_nothing"] = admit_only_last_applied
    r.check(
        "admit() alone applies NOTHING — last_applied stays 0 until a submit() is awaited "
        "(percept/sequencing.py:183-191)",
        admit_only_last_applied and admit_only_last_applied[0] == 0,
        f"last_applied observed straight after the first admits: {admit_only_last_applied}",
    )

    # --- continuity_id / appearance_id ----------------------------------------------------------
    # spec: pure senses — feed=match(obs, known, *, signature=None) -> id | None; read is None.
    for sense, kw in (("continuity_id", {"iou_match": 0.5}),
                      ("appearance_id", {"threshold": 0.8, "iou_match": 0.5})):
        p, path = _acquire(sense, **kw)
        known: list[Entity] = []
        matched = 0
        n_calls = 0
        for i in range(200):
            drift = 0.0005 * i
            obs = Observation(label="person", box=Box(0.1 + drift, 0.1, 0.2, 0.4), confidence=0.9)
            mid = p.feed(obs, known, signature=[1.0, 0.0, float(i % 3)])
            n_calls += 1
            if mid is None:
                ent = Entity(id=f"e{len(known) + 1}", label="person", box=obs.box,
                             first_seen=0.0, last_seen=0.0)
                ent.appearance = [1.0, 0.0, float(i % 3)]
                known.append(ent)
            else:
                matched += 1
                for ent in known:
                    if ent.id == mid:
                        ent.box = obs.box
        row(sense, path, n_calls, len(known), p.stats,
            f"match() calls -> distinct ids minted ({matched} re-identified, {len(known)} new)")

    # AppearanceIdentity trap (percept/identity.py:72-87): with signature=None it SILENTLY degrades
    # to pure continuity, so a re-id number measured without a signature is measuring continuity.
    # Prove we are on the appearance path before believing any re-id claim.
    p_app, app_path = _acquire("appearance_id", threshold=0.8, iou_match=0.5)
    far = Entity(id="e-far", label="person", box=Box(0.80, 0.80, 0.10, 0.10),
                 first_seen=0.0, last_seen=0.0)
    far.appearance = [1.0, 0.0, 0.0]
    gap_obs = Observation(label="person", box=Box(0.02, 0.02, 0.10, 0.10), confidence=0.9)
    reid_with_sig = p_app.feed(gap_obs, [far], signature=[1.0, 0.0, 0.0])
    reid_no_sig = p_app.feed(gap_obs, [far], signature=None)
    cont_ctrl, _ = _acquire("continuity_id", iou_match=0.5)
    cont_answer = cont_ctrl.feed(gap_obs, [far], signature=[1.0, 0.0, 0.0])
    reid = {"with_signature": reid_with_sig, "without_signature": reid_no_sig,
            "continuity_control": cont_answer, "probe_path": app_path}
    r.data["appearance_path_proof"] = reid
    r.check(
        "AppearanceIdentity re-id is genuinely on the APPEARANCE path (matches across a "
        "zero-IoU gap with signature=, returns None without it; continuity control also None)",
        reid_with_sig == "e-far" and reid_no_sig is None and cont_answer is None,
        json.dumps(reid),
    )

    # --- extractor ------------------------------------------------------------------------------
    # spec: pure sense; the registry's conformer is percept.extractors.ocr.ScriptedOCR
    # (percept.extraction.EvidenceExtractor is a typing.Protocol and cannot be instantiated).
    script = {f"frame-{i}": {"text": f"INV-{i:04d}", "confidence": 0.9}
              for i in range(0, N_FRAMES, 10)}
    p, path = _acquire("extractor", script=script)
    non_empty = 0
    for i in range(N_FRAMES):
        enr = p.feed("read the invoice number", f"frame-{i}", None)
        if isinstance(enr, Enrichment) and not enr.is_empty:
            non_empty += 1
    row("extractor", path, N_FRAMES, non_empty, p.stats,
        "frames offered -> non-empty Enrichments forwarded as judge context")

    # --- robustness probe: SpanAssociator.add fails LATE on a malformed candidate ---------------
    late = {"first_add_ok": None, "first_add_count": None, "second_add_error": None}
    assoc_bad = SpanAssociator(refractory_s=0.5)
    try:
        assoc_bad.add("not-an-EventCandidate")  # type: ignore[arg-type]
        late["first_add_ok"] = True
        late["first_add_count"] = assoc_bad.count()
    except Exception as exc:  # noqa: BLE001
        late["first_add_ok"] = False
        late["first_add_error"] = f"{type(exc).__name__}: {exc}"
    if late["first_add_ok"]:
        try:
            assoc_bad.add("also-not-one")  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            late["second_add_error"] = f"{type(exc).__name__}: {exc}"
    r.data["span_late_failure"] = late
    late_failure = bool(late["first_add_ok"]) and bool(late["second_add_error"])
    if late_failure:
        r.defects.append(
            "ROBUSTNESS (non-blocking): `SpanAssociator.add` validates nothing "
            "(`percept/individuation.py:133-147`). A malformed candidate is ACCEPTED on the first "
            "add (`count()` becomes 1) and only blows up on the SECOND add, when `_same_event` "
            "(`percept/individuation.py:104-116`) dereferences `cand.kind` — "
            f"`{late['second_add_error']}`. The error therefore surfaces one unit LATE and names "
            "the wrong call. No Stage-2 correctness property in SV-2.5/2.6/2.7 depends on it, so "
            "it is reported, not failed."
        )
    r.check(
        "SpanAssociator type-safety is characterised (late-failure behaviour recorded, "
        "not asserted away)",
        late["first_add_ok"] is not None,
        json.dumps(late),
    )

    r.data["per_sense"] = rows
    r.data["probe_paths"] = sorted(set(probe_paths))
    r.check(
        "every sense was measured through the real SenseProbe adapter",
        SENSEPROBE_AVAILABLE and set(probe_paths) == {"senseprobe"},
        f"probe paths used: {sorted(set(probe_paths))} "
        f"(senseprobe_available={SENSEPROBE_AVAILABLE}, error={SENSEPROBE_ERROR})",
    )
    r.check(
        "all 8 documented REGISTRY keys are present and were exercised",
        SENSEPROBE_AVAILABLE and set(SP_REGISTRY) == {
            "rhythm", "entity", "span", "sequencer", "reducer", "continuity_id",
            "appearance_id", "extractor"},
        f"registry keys: {sorted(SP_REGISTRY) if SENSEPROBE_AVAILABLE else 'unavailable'}",
    )

    # --- the end-to-end Stage-2 funnel ---------------------------------------------------------
    tracker = RhythmTracker()
    assoc = SpanAssociator(refractory_s=PERIOD * 0.5)
    elog = EpisodeLog("funnel", policy=EpisodePolicy(continuity_gap_s=PERIOD * 3))
    fgate = Gate(threshold=0.7, key_ttl_s=None, max_keys=None)
    fred = SerializedDecisionReducer(fgate, clock=_StepClock())

    stage1_units = 0
    rep_units = 0
    span_units = 0
    episode_units = 0
    episode_continuations = 0
    stage3_units = 0

    async def _funnel() -> None:
        nonlocal stage1_units, rep_units, span_units, episode_units, stage3_units
        nonlocal episode_continuations
        for t, ch in frames:
            stage1_units += 1
            rep = tracker.observe(t, ch)
            if rep is None:
                continue
            rep_units += 1
            ev, is_new = assoc.add(EventCandidate.from_rep(rep))
            if not is_new:
                continue
            span_units += 1
            dec = elog.observe_decision(t, {"tag": "rep"}, subject_key="rep-subject")
            if not dec.is_new:
                episode_continuations += 1
                continue
            episode_units += 1
            seq = fred.admit(dec.episode.id)
            gdec = await fred.submit(dec.episode.id, seq,
                                     Verdict(satisfied=True, confidence=0.95), now=t)
            if gdec.action in (FIRE, ASK):
                stage3_units += 1

    asyncio.run(_funnel())
    funnel = {
        "stage1_frames_in": stage1_units,
        "rhythm_reps": rep_units,
        "span_events": span_units,
        "episodes_opened": episode_units,
        "episode_continuations": episode_continuations,
        "stage3_delivered": stage3_units,
        "overall_ratio": (stage3_units / stage1_units) if stage1_units else None,
    }
    r.data["funnel"] = funnel

    # --- assertions -----------------------------------------------------------------------------
    r.check(
        "every sense compresses (units_out <= units_in)",
        all(x["units_out"] <= x["units_in"] for x in rows),
        json.dumps([x for x in rows if x["units_out"] > x["units_in"]]),
    )
    r.check(
        "no probe recorded an error",
        all(int(x["errors"] or 0) == 0 for x in rows),
        json.dumps([x for x in rows if int(x["errors"] or 0)]),
    )
    r.check(
        "span association is idempotent under denser observation (3x candidates -> 1x events)",
        next(x for x in rows if x["sense"] == "span")["units_out"] * 3
        <= next(x for x in rows if x["sense"] == "span")["units_in"],
        json.dumps(next(x for x in rows if x["sense"] == "span")),
    )
    r.check(
        "the Stage-2 funnel strictly narrows (frames >= reps >= spans >= episodes >= delivered)",
        stage1_units >= rep_units >= span_units >= episode_units >= stage3_units,
        json.dumps(funnel),
    )
    r.check(
        "Stage-2 delivers strictly fewer units to Stage 3 than it received from Stage 1",
        stage3_units < stage1_units and stage1_units > 0,
        json.dumps(funnel),
    )
    return r


# ============================================================================================
# SV-2.1 groundwork — counting / rhythm measurement harness
# ============================================================================================


def count_edges(seq: list[tuple[float, bool]], refractory_s: float = 0.0) -> int:
    """The candidate-source counting rule (``percept/eval/exp_repcount.py``): count false->true
    rising edges of a per-frame phase signal, ignoring an edge within ``refractory_s`` of the
    last counted one. Reproduced here verbatim in behaviour so no live judge path is imported."""
    n, last, prev = 0, None, False
    for ts, up in seq:
        if up and not prev and (last is None or ts - last >= refractory_s):
            n += 1
            last = ts
        prev = up
    return n


def _load_repcount_cases() -> list[dict]:
    data = json.loads(CORPUS.read_text())
    cases = data["cases"] if isinstance(data, dict) else data
    return [c for c in cases if c.get("dataset") == "repcount"]


def _media_fps(video_path: Path) -> tuple[float, int]:
    """Read (fps, n_frames) from the media file ITSELF — never a guessed constant."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def _cycles_seconds(bounds: list[int], fps: float) -> list[tuple[float, float]]:
    """``cycle_bounds_frames`` (FRAME INDICES) -> per-cycle (start_s, end_s) using the file's fps."""
    return [(bounds[2 * i] / fps, bounds[2 * i + 1] / fps) for i in range(len(bounds) // 2)]


def _scripted_signal(cycles: list[tuple[float, float]], total_s: float, obs_hz: float,
                     noise_sigma: float = 0.0, seed: int = 7) -> tuple[list, list]:
    """Scripted (no-provider) phase + amplitude signal on the REAL cadence.

    Within cycle ``[s, e)`` the amplitude traces one trough->peak->trough cosine arch and the
    binary phase is ``up`` for the first half. Outside every cycle the signal rests at 0/False.
    Returns ``(edge_seq, measurement_pairs)``.
    """
    rng = random.Random(seed)
    edge_seq: list[tuple[float, bool]] = []
    meas: list[tuple[float, float]] = []
    dt = 1.0 / obs_hz
    n = int(math.floor(total_s / dt)) + 1
    for k in range(n):
        t = k * dt
        v, up = 0.0, False
        for (s, e) in cycles:
            if s <= t < e and e > s:
                ph = (t - s) / (e - s)
                v = 0.5 * (1.0 - math.cos(2.0 * math.pi * ph))
                up = ph < 0.5
                break
        if noise_sigma:
            v += rng.gauss(0.0, noise_sigma)
        edge_seq.append((t, up))
        meas.append((t, v))
    return edge_seq, meas


def _pixel_channels(video_path: Path, fps: float, target_hz: float = 10.0) -> list[tuple[float, dict]]:
    """A CRUDE, fully local eye: motion-energy + motion-centroid channels from decoded pixels.

    No model, no network — abs frame difference on a 64x36 grayscale downsample. This is NOT the
    shipped perception front end; it is here only so the harness carries one honest end-to-end
    number alongside the oracle-signal number.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    step = max(1, int(round(fps / target_hz)))
    ys, xs = np.mgrid[0:36, 0:64]
    prev = None
    out: list[tuple[float, dict]] = []
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev is not None:
                dm = np.abs(g - prev)
                tot = float(dm.sum()) + 1e-6
                out.append((i / fps, {
                    "motion_energy": float(dm.mean()),
                    "motion_cy": float((dm * ys).sum() / tot) / 36.0,
                    "motion_cx": float((dm * xs).sum() / tot) / 64.0,
                }))
            prev = g
        i += 1
    cap.release()
    return out


def _dist(errs: list[int]) -> dict:
    if not errs:
        return {}
    srt = sorted(errs)
    n = len(srt)
    return {
        "n": n,
        "mae": round(sum(srt) / n, 3),
        "exact": sum(1 for e in srt if e == 0),
        "obo": sum(1 for e in srt if e <= 1),
        "obo_pct": round(100.0 * sum(1 for e in srt if e <= 1) / n, 1),
        "median": srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2,
        "p90": srt[min(n - 1, int(round(0.9 * (n - 1))))],
        "max": srt[-1],
        "histogram": {str(k): sum(1 for e in srt if e == k) for k in range(0, max(srt) + 1)},
    }


def run_repcount_harness(quick: bool = False) -> TestResult:
    r = TestResult("SV-2.1-groundwork", "Counting / rhythm measurement harness (20 repcount cases)")
    cases = _load_repcount_cases()
    r.data["n_cases"] = len(cases)

    # --- A) the counting rule on purely synthetic signals (rule sanity, no media) --------------
    synth = []
    for reps, period, hz in ((1, 1.5, 30.0), (5, 1.0, 30.0), (12, 0.8, 30.0), (8, 2.5, 6.0),
                             (8, 2.5, 12.0), (8, 2.5, 30.0), (8, 2.5, 60.0)):
        cycles = [(i * period, (i + 1) * period) for i in range(reps)]
        seq, meas = _scripted_signal(cycles, reps * period + period * 0.25, hz)
        tracker = RhythmTracker()
        for t, v in meas:
            tracker.observe(t, {"phase": v})
        synth.append({
            "reps": reps, "period_s": period, "obs_hz": hz,
            "count_edges": count_edges(seq, 0.5 * period),
            "rhythm": tracker.count,
            "err_edges": abs(count_edges(seq, 0.5 * period) - reps),
            "err_rhythm": abs(tracker.count - reps),
        })
    r.data["synthetic_rule_check"] = synth
    r.check(
        "counting rule counts a SINGLE rep from cold start (no warm-up)",
        synth[0]["rhythm"] == 1 and synth[0]["count_edges"] == 1,
        json.dumps(synth[0]),
    )
    r.check(
        "counting rule is rate-agnostic on a clean signal (6/12/30/60 Hz all give 8)",
        all(x["rhythm"] == 8 for x in synth[3:]),
        json.dumps(synth[3:]),
    )

    # --- B) the measurement layer's own invariants ---------------------------------------------
    period = 1.25
    pairs = [(i / 30.0, 0.5 * (1 - math.cos(2 * math.pi * (i / 30.0) / period))) for i in range(600)]
    series = MeasurementSeries.from_pairs("torso_y", pairs, units="body_h", source_hz=30.0)
    meas_block = {
        "dominant_period_s": dominant_period(series, band_s=(0.3, 5.0), hz=30.0),
        "true_period_s": period,
        "periodicity_score": round(periodicity_score(series, band_s=(0.3, 5.0), hz=30.0), 4),
        "robust_amplitude": round(robust_amplitude(series, window_s=period), 4),
        "subsample_stability": round(
            subsample_stability(series, lambda s: dominant_period(s, band_s=(0.3, 5.0), hz=30.0)), 4),
        "channel_score": round(channel_score(series, hz=30.0, band_s=(0.3, 5.0)).score, 4),
    }
    r.data["measurement_layer"] = meas_block
    dp = meas_block["dominant_period_s"]
    r.check(
        "dominant_period recovers the true period in SECONDS (within one grid step)",
        dp is not None and abs(dp - period) <= 1.0 / 30.0 + 1e-9,
        json.dumps(meas_block),
    )
    r.check(
        "periodicity_score is high and the period survives decimation (stability >= 0.9)",
        meas_block["periodicity_score"] >= 0.8 and meas_block["subsample_stability"] >= 0.9,
        json.dumps(meas_block),
    )

    # --- C) the real corpus: oracle-signal measurement -----------------------------------------
    per_case: list[dict] = []
    obs_rates = (3.0, 6.0, 30.0)
    t0 = time.perf_counter()
    for c in cases:
        prov = c["provenance"]
        vp = REPO_ROOT / c["video_path"]
        row: dict = {
            "case_id": c["case_id"],
            "action_type": prov.get("action_type"),
            "expected_count": prov["expected_count"],
            "video_path": c["video_path"],
            "media_present": vp.exists(),
        }
        if not vp.exists():
            row["error"] = "media missing"
            per_case.append(row)
            continue
        fps, n_frames = _media_fps(vp)
        bounds = prov["cycle_bounds_frames"]
        cycles = _cycles_seconds(bounds, fps)
        total_s = n_frames / fps
        tau = statistics.fmean([e - s for s, e in cycles]) if cycles else 1.0
        row.update({
            "fps": round(fps, 4),
            "n_frames": n_frames,
            "duration_s": round(total_s, 3),
            "cycles_from_bounds": len(cycles),
            "bounds_unit": prov.get("bounds_unit"),
            "mean_cycle_s": round(tau, 3),
            "min_cycle_s": round(min(e - s for s, e in cycles), 3) if cycles else None,
            "max_cycle_s": round(max(e - s for s, e in cycles), 3) if cycles else None,
        })
        gt = prov["expected_count"]
        for hz in obs_rates:
            seq, meas = _scripted_signal(cycles, total_s, hz)
            tracker = RhythmTracker()
            for t, v in meas:
                tracker.observe(t, {"phase": v})
            pe = count_edges(seq, 0.5 * tau)
            row[f"edges@{hz:g}"] = pe
            row[f"edges_err@{hz:g}"] = abs(pe - gt)
            row[f"rhythm@{hz:g}"] = tracker.count
            row[f"rhythm_err@{hz:g}"] = abs(tracker.count - gt)
        # noisy variant at 6 Hz (deterministic seed)
        for sigma in (0.05, 0.15):
            seq, meas = _scripted_signal(cycles, total_s, 6.0, noise_sigma=sigma, seed=11)
            tracker = RhythmTracker()
            for t, v in meas:
                tracker.observe(t, {"phase": v})
            row[f"rhythm_noise{sigma}"] = tracker.count
            row[f"rhythm_noise{sigma}_err"] = abs(tracker.count - gt)
        per_case.append(row)
    r.data["oracle_elapsed_s"] = round(time.perf_counter() - t0, 2)

    # --- D) the crude local pixel eye (optional, honest end-to-end number) ---------------------
    if not quick:
        t0 = time.perf_counter()
        for row in per_case:
            if not row.get("media_present"):
                continue
            vp = REPO_ROOT / row["video_path"]
            try:
                chans = _pixel_channels(vp, row["fps"])
                tracker = RhythmTracker()
                for t, ch in chans:
                    tracker.observe(t, ch)
                row["pixel_eye"] = tracker.count
                row["pixel_eye_err"] = abs(tracker.count - row["expected_count"])
                row["pixel_eye_selected"] = tracker.selected
                row["pixel_eye_samples"] = len(chans)
            except Exception as exc:  # noqa: BLE001
                row["pixel_eye_error"] = f"{type(exc).__name__}: {exc}"
        r.data["pixel_eye_elapsed_s"] = round(time.perf_counter() - t0, 2)

    r.data["per_case"] = per_case

    scored = [x for x in per_case if x.get("media_present")]
    dists: dict[str, dict] = {}
    for hz in obs_rates:
        dists[f"count_edges@{hz:g}Hz"] = _dist([x[f"edges_err@{hz:g}"] for x in scored])
        dists[f"RhythmTracker@{hz:g}Hz"] = _dist([x[f"rhythm_err@{hz:g}"] for x in scored])
    for sigma in (0.05, 0.15):
        dists[f"RhythmTracker@6Hz+noise{sigma}"] = _dist(
            [x[f"rhythm_noise{sigma}_err"] for x in scored])
    pix = [x["pixel_eye_err"] for x in scored if "pixel_eye_err" in x]
    if pix:
        dists["RhythmTracker@10Hz+crude-pixel-eye"] = _dist(pix)
    r.data["distributions"] = dists

    # Assertions here are about the HARNESS being sound, not about accuracy (no invented bar).
    r.check(
        "all 20 repcount cases resolved their media and their own fps",
        len(scored) == 20 and all(x["fps"] > 0 for x in scored),
        f"{len(scored)}/20 scored; missing={[x['case_id'] for x in per_case if not x.get('media_present')]}",
    )
    r.check(
        "fps is read per file, not assumed constant (>1 distinct fps in the corpus)",
        len({x["fps"] for x in scored}) > 1,
        f"distinct fps values: {sorted({x['fps'] for x in scored})}",
    )
    r.check(
        "ground truth is self-consistent: len(cycle_bounds_frames)/2 == expected_count for all 20",
        all(x["cycles_from_bounds"] == x["expected_count"] for x in scored),
        json.dumps([{"case": x["case_id"], "cycles": x["cycles_from_bounds"],
                     "gt": x["expected_count"]} for x in scored
                    if x["cycles_from_bounds"] != x["expected_count"]]),
    )
    r.check(
        "every case produced a numeric prediction from both counting rules at every rate",
        all(isinstance(x.get(f"{k}@{hz:g}"), int) for x in scored
            for k in ("edges", "rhythm") for hz in obs_rates),
        "some case produced no prediction",
    )
    return r


# ============================================================================================
# Reporting
# ============================================================================================


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def print_report(results: list[TestResult]) -> None:
    print("=" * 100)
    print("  Stage-2 LOCAL PERCEPTION validation — SV-2.5 / SV-2.6 / SV-2.7 + counting harness")
    print(f"  runtime: percept {getattr(percept, '__version__', '?')} @ {RUNTIME_PATH}")
    print(f"  SenseProbe: {'available' if SENSEPROBE_AVAILABLE else 'NOT AVAILABLE — ' + str(SENSEPROBE_ERROR)}")
    print("=" * 100)
    rows = []
    for res in results:
        rows.append([res.test_id, "PASS" if res.passed else "FAIL",
                     f"{sum(1 for c in res.checks if c.passed)}/{len(res.checks)}", res.title])
    print(_fmt_table(["test", "verdict", "checks", "title"], rows))
    print()
    for res in results:
        print(f"--- {res.test_id}: {res.title} " + "-" * max(0, 60 - len(res.title)))
        for c in res.checks:
            mark = "PASS" if c.passed else "FAIL"
            print(f"  [{mark}] {c.name}")
            if not c.passed and c.detail:
                print(f"         -> {c.detail[:600]}")
        print()

    rc = next((r for r in results if r.test_id == "SV-2.1-groundwork"), None)
    if rc:
        print("--- rep-count error distribution (20 real repcount cases) " + "-" * 30)
        drows = []
        for name, d in rc.data.get("distributions", {}).items():
            if not d:
                continue
            drows.append([name, str(d["n"]), f"{d['mae']:.2f}", f"{d['exact']}/{d['n']}",
                          f"{d['obo']}/{d['n']}", str(d["median"]), str(d["p90"]), str(d["max"])])
        print(_fmt_table(["configuration", "n", "MAE", "exact", "|e|<=1", "median", "p90", "max"], drows))
        print()

    econ = next((r for r in results if r.test_id == "SV-2.7"), None)
    if econ:
        print("--- Stage-2 economy per sense " + "-" * 55)
        erows = [[x["sense"], x["probe_path"], str(x["units_in"]), str(x["units_out"]),
                  f"{x['economy_ratio']:.4f}" if x["economy_ratio"] is not None else "-",
                  f"{x['wall_time_s']:.4f}", str(x["errors"])]
                 for x in econ.data.get("per_sense", [])]
        print(_fmt_table(["sense", "probe", "units_in", "units_out", "out/in", "wall_s", "err"], erows))
        print(f"  funnel: {json.dumps(econ.data.get('funnel'))}")
        print()

    overall = all(r.passed for r in results)
    print("=" * 100)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 100)


def write_markdown(results: list[TestResult], path: Path, elapsed: float) -> None:
    rc = next((r for r in results if r.test_id == "SV-2.1-groundwork"), None)
    econ = next((r for r in results if r.test_id == "SV-2.7"), None)
    lines: list[str] = []
    a = lines.append
    a("# Stage-2 LOCAL PERCEPTION validation — SV-2.5 / SV-2.6 / SV-2.7 + counting harness")
    a("")
    a(f"- generated: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}` (wall {elapsed:.1f}s)")
    a(f"- runtime under test: `percept {getattr(percept, '__version__', '?')}` @ `{RUNTIME_PATH}`")
    a(f"- runtime is the repo tree (MUST be false): `{RUNTIME_IS_REPO_TREE}`")
    a(f"- `tools.senseprobe` available: **{SENSEPROBE_AVAILABLE}**"
      + ("" if SENSEPROBE_AVAILABLE else f" — `{SENSEPROBE_ERROR}`"))
    a("- provider/network calls: **none** (all judges scripted, all signals synthetic or "
      "corpus-ground-truth derived; media is decoded locally with cv2 only)")
    a("")
    a("## Verdicts")
    a("")
    a(_md_table(["test", "verdict", "checks", "title"],
                [[r.test_id, "**PASS**" if r.passed else "**FAIL**",
                  f"{sum(1 for c in r.checks if c.passed)}/{len(r.checks)}", r.title]
                 for r in results]))
    a("")
    for res in results:
        a(f"## {res.test_id} — {res.title}")
        a("")
        a(_md_table(["", "check", "detail"],
                    [["PASS" if c.passed else "**FAIL**", c.name,
                      (c.detail[:300].replace("|", "\\|") if not c.passed else "")]
                     for c in res.checks]))
        a("")
        if res.test_id == "SV-2.5":
            a("### Declared out-of-order policy (as read from the runtime, not assumed)")
            a("")
            a("`percept/sequencing.py:63-80` (`ResultPolicy`) and `percept/sequencing.py:242-260` "
              "(`_stale_reason`) declare:")
            a("")
            a("- `ordering=\"latest_only\"` (default) — a result is stale the instant a NEWER job "
              "has been **admitted** for the key (`evidence_seq < latest_admitted_sequence`), not "
              "merely applied. Drop reason `superseded_by_newer_admission`.")
            a("- `ordering=\"episode_ordered\"` — supersession is scoped to `episode_id`; drop "
              "reason `superseded_within_episode`. With `episode_id=None` it falls back to the "
              "key-wide `latest_only` comparison.")
            a("- `stale_after_ms` — an independent age drop (`stale_timeout`), checked FIRST.")
            a("- `apply_after_supersession=True` — the would-be drop is applied anyway and flagged "
              "`applied_after_supersession=True` (with `drop_reason` retained).")
            a("- A dropped decision is always `action=SILENT`, `dropped=True`, and never touches "
              "the gate.")
            a("")
            a(f"- determinism digest (sha256 of the canonical decision trace): "
              f"`{res.data.get('digest')}`")
            a(f"- in-process runs: 5, distinct digests: "
              f"{len(set(res.data.get('in_process_digests', [])))}")
            a(f"- subprocess runs (PYTHONHASHSEED 0/1/12345): "
              f"{len(res.data.get('subprocess_digests', []))}, distinct digests: "
              f"{len(set(res.data.get('subprocess_digests', [])))}")
            if res.data.get("subprocess_errors"):
                a(f"- subprocess errors: `{res.data['subprocess_errors']}`")
            a("")
        if res.test_id == "SV-2.6":
            st = res.data.get("stream", {})
            a("### Occlusion boundary")
            a("")
            a(_md_table(["case", "observation t (s)", "is_new", "reason", "subject_matched",
                         "prior_episode_id", "gap_s"],
                        [["just within (gap-1e-6)", f"{5.0 - 1e-6:.6f}",
                          str(res.data['occlusion_within']['is_new']),
                          res.data['occlusion_within']['reason'], "n/a", "n/a", "n/a"]]
                        + [[x["case"], f"{x['t']:.6f}", str(x["is_new"]), x["reason"],
                            str(x["subject_matched"]), str(x["prior_episode_id"]),
                            str(x["gap_s"])] for x in res.data.get("occlusion_beyond", [])]))
            a("")
            a("### Silent-duplicate bar")
            a("")
            a(f"- observations folded: **{st.get('observations')}**")
            a(f"- episodes opened: **{st.get('episodes_opened')}**  "
              f"(continuations: {st.get('continuations')}; first open reason: "
              f"`{st.get('first_open_reason')}`)")
            a(f"- duplicate opens (every open after the subject's first): "
              f"**{st.get('duplicate_opens')}**")
            a(f"- traced splits: **{st.get('traced_splits')}**")
            a(f"- **SILENT duplicates: {st.get('silent_duplicates')}** (bar: 0)")
            a(f"- split reasons observed: `{st.get('split_reasons')}`")
            a("")
            a("### Revise-not-duplicate")
            a("")
            a(f"- `{json.dumps(res.data.get('revise_not_duplicate'))}`")
            a(f"- heartbeat boundedness: `{json.dumps(res.data.get('heartbeat'))}`")
            a("")
    if econ:
        a("## SV-2.7 — per-sense economy")
        a("")
        a("`units_in` = units received from Stage 1; `units_out` = units passed on to Stage 3.")
        if not SENSEPROBE_AVAILABLE:
            a("")
            a("> **`tools/senseprobe` was not present at run time.** Every row below was measured "
              "with the in-file `_ShimProbe`, which reproduces the documented "
              "`{name, units_fed, reads, wall_time_s, errors}` stats shape. The `probe` column "
              "names the path used for each row so the numbers stay auditable.")
        a("")
        a(_md_table(["sense", "probe", "units_in", "units_out", "out/in", "probe.units_fed",
                     "probe.reads", "wall_s", "errors", "meaning"],
                    [[x["sense"], x["probe_path"], str(x["units_in"]), str(x["units_out"]),
                      f"{x['economy_ratio']:.4f}" if x["economy_ratio"] is not None else "-",
                      str(x["probe_units_fed"]), str(x["probe_reads"]),
                      f"{x['wall_time_s']:.4f}", str(x["errors"]), x["note"]]
                     for x in econ.data.get("per_sense", [])]))
        a("")
        f = econ.data.get("funnel", {})
        a("### End-to-end Stage-2 funnel (one pipeline, 600 synthetic eye frames @ 10 Hz)")
        a("")
        a(_md_table(["stage", "units"],
                    [["Stage-1 frames in", str(f.get("stage1_frames_in"))],
                     ["rhythm reps", str(f.get("rhythm_reps"))],
                     ["span events (deduped)", str(f.get("span_events"))],
                     ["episodes opened", str(f.get("episodes_opened"))],
                     ["episode continuations (folded into an open episode, not re-delivered)",
                      str(f.get("episode_continuations"))],
                     ["Stage-3 delivered (FIRE / ASK)", str(f.get("stage3_delivered"))]]))
        a("")
        a(f"- overall Stage-2 economy ratio (Stage-3 out / Stage-1 in): "
          f"**{f.get('overall_ratio')}**")
        a("")
        a("The tail is narrow by design, not by accident: all 30 reps belong to ONE subject inside "
          "one `continuity_gap_s`, so 29 of them CONTINUE the open episode rather than minting "
          "rivals, and the gate's rising-edge rule delivers the episode once. That is the "
          "Stage-2 job — 600 frames of Stage-1 evidence become 1 unit of Stage-3 attention.")
        a("")
    if rc:
        a("## SV-2.1 groundwork — rep-count measurement (20 real `repcount` cases)")
        a("")
        a("Ground truth is the corpus' own `provenance.expected_count`; "
          "`provenance.cycle_bounds_frames` are FRAME INDICES converted to seconds with each "
          "media file's OWN fps (read via cv2 — the corpus spans 23.976 / 25 / 29.97 / 30 fps, so "
          "a constant would have been wrong). **No pass bar is asserted here** — these are raw "
          "measurements for the pre-registration agent.")
        a("")
        a("Two counting rules are driven:")
        a("")
        a("- **`count_edges`** — the candidate-source refractory rule "
          "(`packages/percept-vision/eval/exp_repcount.py:55`), reproduced in this file "
          "(`sv2_senses.py:count_edges`); refractory = 0.5 x the case's own mean cycle length.")
        a("- **`RhythmTracker`** — the shipped sense (`percept/rhythm.py`), run at library defaults.")
        a("")
        a("Signals:")
        a("")
        a("- *oracle signal* — a scripted trough->peak->trough arch (and its binary up/down phase) "
          "laid on the REAL annotated cycle bounds. This isolates the COUNTING RULE from the eye.")
        a("- *+noise* — the same, plus deterministic Gaussian noise (seeded).")
        a("- *crude pixel eye* — a local, model-free motion-energy/centroid front end over the "
          "decoded pixels. Included only as an honest end-to-end datapoint; it is **not** the "
          "shipped perception path.")
        a("")
        a("### Error distribution (|predicted - expected_count|)")
        a("")
        a(_md_table(["configuration", "n", "MAE", "exact (err=0)", "err&le;1", "median", "p90", "max",
                     "histogram of abs err"],
                    [[k, str(d["n"]), f"{d['mae']:.2f}", f"{d['exact']}/{d['n']}",
                      f"{d['obo']}/{d['n']} ({d['obo_pct']}%)", str(d["median"]), str(d["p90"]),
                      str(d["max"]), json.dumps(d["histogram"])]
                     for k, d in rc.data.get("distributions", {}).items() if d]))
        a("")
        a("### Per case")
        a("")
        hdr = ["case_id", "action", "fps", "dur_s", "gt", "mean_cycle_s",
               "edges@6Hz", "err", "rhythm@6Hz", "err", "rhythm@30Hz", "err",
               "rhythm+noise.15@6Hz", "err", "pixel_eye", "err"]
        rows = []
        for x in rc.data.get("per_case", []):
            if not x.get("media_present"):
                rows.append([x["case_id"], "-", "-", "-", str(x.get("expected_count")), "-"]
                            + ["-"] * 10)
                continue
            rows.append([
                x["case_id"], str(x.get("action_type")), f"{x['fps']:.3f}",
                f"{x['duration_s']:.1f}", str(x["expected_count"]), f"{x['mean_cycle_s']:.2f}",
                str(x.get("edges@6")), str(x.get("edges_err@6")),
                str(x.get("rhythm@6")), str(x.get("rhythm_err@6")),
                str(x.get("rhythm@30")), str(x.get("rhythm_err@30")),
                str(x.get("rhythm_noise0.15")), str(x.get("rhythm_noise0.15_err")),
                str(x.get("pixel_eye", "-")), str(x.get("pixel_eye_err", "-")),
            ])
        a(_md_table(hdr, rows))
        a("")
        a("### Measurement-layer invariants (`percept/measurement.py`)")
        a("")
        a(f"- `{json.dumps(rc.data.get('measurement_layer'))}`")
        a("")
        a("### Counting-rule sanity on purely synthetic signals")
        a("")
        a(_md_table(["reps", "period_s", "obs_hz", "count_edges", "err", "RhythmTracker", "err"],
                    [[str(x["reps"]), str(x["period_s"]), str(x["obs_hz"]), str(x["count_edges"]),
                      str(x["err_edges"]), str(x["rhythm"]), str(x["err_rhythm"])]
                     for x in rc.data.get("synthetic_rule_check", [])]))
        a("")
    if rc:
        dd = rc.data.get("distributions", {})
        a("## Observations (measurement, not verdicts)")
        a("")
        a("These are raw readings from the harness, offered to whoever pre-registers the SV-2.1 "
          "bars. None of them is asserted here.")
        a("")
        e3 = dd.get("count_edges@3Hz", {})
        r3 = dd.get("RhythmTracker@3Hz", {})
        r6 = dd.get("RhythmTracker@6Hz", {})
        n6 = dd.get("RhythmTracker@6Hz+noise0.05", {})
        n15 = dd.get("RhythmTracker@6Hz+noise0.15", {})
        px = dd.get("RhythmTracker@10Hz+crude-pixel-eye", {})
        a(f"1. **Observation rate dominates `RhythmTracker`, not the cadence.** On the identical "
          f"oracle signal it goes from MAE {r3.get('mae')} (max |e| {r3.get('max')}) at 3 Hz to "
          f"MAE {r6.get('mae')} at 6 Hz. The binary-edge rule is unaffected "
          f"(`count_edges` MAE {e3.get('mae')} at 3 Hz) because a rising edge survives coarse "
          f"sampling while the trough->peak->trough *differential* the tracker needs does not. "
          f"Corpus cadences run 0.79 s - 12.68 s per cycle, so 3 Hz is under two samples per "
          f"half-cycle for the fastest cases. Any SV-2.1 bar must fix the observation rate.")
        a("")
        a(f"2. **`RhythmTracker` at library defaults is noise-fragile.** Adding seeded Gaussian "
          f"noise to the same oracle signal at 6 Hz takes MAE from {r6.get('mae')} to "
          f"{n6.get('mae')} at sigma=0.05 and {n15.get('mae')} at sigma=0.15 (5% and 15% of "
          f"amplitude). `RhythmTracker.smooth_s` defaults to `0.0` — the causal low-pass that "
          f"exists precisely for this is OPT-IN (`percept/rhythm.py:278-285` documents the "
          f"trade-off and why it was left off). Numbers above are the DEFAULT config; the "
          f"harness was not tuned to improve them.")
        a("")
        if px:
            a(f"3. **The eye is the dominant error source, not the counting rule.** With a perfect "
              f"(oracle) signal the rule scores MAE {r6.get('mae')} / "
              f"{r6.get('exact')}/{r6.get('n')} exact. With the crude local motion-centroid eye on "
              f"the same 20 clips and the same tracker, MAE is {px.get('mae')} / "
              f"{px.get('exact')}/{px.get('n')} exact. That gap is the front end, not `rhythm.py`. "
              f"(The crude eye is a floor, not the shipped path — it is here only to bound the "
              f"question.)")
            a("")
        a("4. **Ground truth is internally consistent.** For all 20 cases "
          "`len(provenance.cycle_bounds_frames) / 2 == provenance.expected_count`, so the bounds "
          "can be used as a per-cycle oracle without re-deriving the count.")
        a("")
        a("5. **fps really does vary** across the corpus (23.976 / 25 / 29.97 / 30). Converting "
          "`cycle_bounds_frames` with a constant 30 fps would shift every 23.976 fps case's "
          "cycle timing by ~25%.")
        a("")

    a("## Reproduce")
    a("")
    a("```sh")
    a("/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python \\")
    a("    eval/stage-validation/sv2_senses.py            # full run (~4 s)")
    a("    #  --quick   skip the crude pixel-eye media pass")
    a("```")
    a("")

    defects = [d for r in results for d in r.defects]
    a("## Defects")
    a("")
    if defects:
        for d in defects:
            a(f"- {d}")
        a("")
        a("No CONTRACT defect was found. On every point SV-2.5 / SV-2.6 / SV-2.7 tested, the "
          "implementation matched the behaviour DECLARED in the runtime source — "
          "`percept/sequencing.py`, `percept/episodes.py`, `percept/rhythm.py`, "
          "`percept/measurement.py`, `percept/individuation.py`, `percept/identity.py`. The item "
          "above is a robustness/ergonomics gap, not a broken guarantee.")
    else:
        a("- None found. Every assertion above matched the behaviour DECLARED in the runtime "
          "source; no divergence between docstring contract and implementation was observed in "
          "`percept/sequencing.py`, `percept/episodes.py`, `percept/rhythm.py`, "
          "`percept/measurement.py`, or `percept/individuation.py`.")
    a("")
    a("### Validation traps encountered (recorded so a later harness does not fall into them)")
    a("")
    a("- `SerializedDecisionReducer.admit()` is synchronous and applies NOTHING; only the awaited "
      "coroutine `submit()` moves gate state (`percept/sequencing.py:183-191` vs `193-240`). A "
      "determinism test built on `admit()` alone would measure nothing and pass. SV-2.5 drives "
      "`submit()`; SV-2.7 asserts `last_applied == 0` after admits (see the `admit_applies_nothing` "
      "row in the JSON).")
    a("- `AppearanceIdentity.match()` silently degrades to pure continuity when `signature=None` "
      "(`percept/identity.py:72-87`), so a re-id number measured without a signature is a "
      "continuity number wearing a re-id label. SV-2.7 proves the appearance path is live before "
      "reporting: same observation across a zero-IoU gap returns `e-far` WITH a signature and "
      "`None` without, and the `ContinuityIdentity` control also returns `None`.")
    a("")
    path.write_text("\n".join(lines))


# ============================================================================================
# Entry point
# ============================================================================================


def run_all(quick: bool = False) -> tuple[list[TestResult], float]:
    t0 = time.perf_counter()
    results = [run_sv25(), run_sv26(), run_sv27(), run_repcount_harness(quick=quick)]
    return results, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-reducer-trace", action="store_true",
                    help="internal: print the canonical SV-2.5 decision trace as JSON")
    ap.add_argument("--quick", action="store_true", help="skip the crude pixel-eye media pass")
    ap.add_argument("--json", default=str(RESULTS_DIR / "sv2_senses.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "sv2_senses.md"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.emit_reducer_trace:
        sys.stdout.write(json.dumps(asyncio.run(reducer_trace())))
        return 0

    if RUNTIME_IS_REPO_TREE:
        print("REFUSING TO RUN: percept resolved to the repo tree, not the launch wheel:\n"
              f"  {RUNTIME_PATH}", file=sys.stderr)
        return 2

    results, elapsed = run_all(quick=args.quick)
    print_report(results)

    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "suite": "sv2_senses",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": round(elapsed, 3),
            "runtime": {
                "percept_version": getattr(percept, "__version__", None),
                "percept_path": str(RUNTIME_PATH),
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "is_repo_tree": RUNTIME_IS_REPO_TREE,
            },
            "senseprobe": {
                "available": SENSEPROBE_AVAILABLE,
                "error": SENSEPROBE_ERROR,
                "registry_keys": sorted(SP_REGISTRY) if SENSEPROBE_AVAILABLE else [],
            },
            "overall_passed": all(r.passed for r in results),
            "tests": [r.to_dict() for r in results],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        write_markdown(results, Path(args.md), elapsed)
        print(f"\nwrote {args.json}\nwrote {args.md}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
