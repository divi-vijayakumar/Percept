"""Stage-4 COGNITION + DISPOSITION validation — SV-4.1 / 4.5 / 4.6 / 4.7 / 4.8.

The exactness tests: every bar here is a hard equality or a zero-tolerance invariant, so none of
them needs a pre-registered threshold.

    SV-4.1  Gate decision + state  — rising edge fires exactly once, refractory honoured, re-arm
                                     after occurrence close, repeated occurrences yield DISTINCT
                                     Incidents.
    SV-4.5  Cache correctness      — hit / miss / invalidation, and the load-bearing one: a stale
                                     cache entry must NEVER satisfy a CHANGED Watch config.
    SV-4.6  Media buffers          — bounded retention, pre-admission window retained per policy,
                                     no unbounded growth over a long run (measured RSS + tracemalloc).
    SV-4.7  Evidence integrity     — ZERO TOLERANCE. Every Incident's evidence must resolve to REAL
                                     retained media whose content hash matches what was fed in. No
                                     synthesized, substituted or re-encoded-beyond-policy evidence.
    SV-4.8  Drop-rate reconciliation — units in == units accounted, at every stage boundary.

Everything is driven through the real public runtime path (``percept.run`` with scripted,
network-free judges), so genuine Incident construction is exercised — not unit-level mocks. The
only exceptions are explicitly labelled ``LAYER PROBE`` checks, which call the same live objects
the agent uses (``core.cache.layers[0]``, ``percept.incidents._encode_frame_bundle``) with the
exact key/argument shapes ``percept/agent.py`` builds, in order to reach a state the outer loop
cannot be steered into deterministically.

Run standalone::

    /private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv4_cognition.py

or import ``run_all()`` for the machine-readable result dict.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import random
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import percept
from percept import EvidenceStore, Goal, Percept
from percept.adapters import Box, Verdict
from percept.cache import CacheKey
from percept.cache.framehash import dhash, hamming
from percept.config import load_settings
from percept.evidence import EvidenceCorrupt, RetentionPolicy
from percept.gate import GATE_REASONS, REASON_ACTION
from percept.goals import GoalOp
from percept.harness.tier0 import gray_frame
from percept.incidents import (
    _encode_frame_bundle,
    incident_revision,
    incident_supersedes,
)
from percept.runtime import RuntimeStats, run

RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSON_PATH = RESULTS_DIR / "sv4_cognition.json"

# The tier-0 motion signal. Alternating STILL/MOVED escalates every tick after the prime frame.
GW, GH = 8, 8
STILL = gray_frame(GW, GH, 0)
MOVED = gray_frame(GW, GH, 255)


# ---------------------------------------------------------------------------------------------
# result model
# ---------------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "data": _jsonable(self.data)}


@dataclass
class Defect:
    id: str
    severity: str  # "blocker" | "high" | "medium" | "low"
    where: str  # file:line
    summary: str
    repro: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TestResult:
    id: str
    title: str
    checks: list[Check] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    defects: list[Defect] = field(default_factory=list)
    error: str | None = None

    def check(self, name: str, passed: bool, detail: str = "", **data: Any) -> Check:
        c = Check(name=name, passed=bool(passed), detail=detail, data=data)
        self.checks.append(c)
        return c

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        if not self.checks:
            return "ERROR"
        return "PASS" if all(c.passed for c in self.checks) else "FAIL"

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "error": self.error,
            "checks": [c.to_json() for c in self.checks],
            "measurements": _jsonable(self.measurements),
            "defects": [d.to_json() for d in self.defects],
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(obj)).hexdigest(), "len": len(obj)}
    return repr(obj)


# ---------------------------------------------------------------------------------------------
# frame + judge harness (deterministic, offline, no provider calls)
# ---------------------------------------------------------------------------------------------


def frame_bytes(frame: Any) -> bytes:
    """Canonical byte view of a frame, for content hashing."""
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return bytes(frame)
    tobytes = getattr(frame, "tobytes", None)
    if callable(tobytes):  # numpy-like
        try:
            return bytes(tobytes())
        except Exception:  # noqa: BLE001
            pass
    return repr(frame).encode("utf-8")


def sha(frame: Any) -> str:
    return hashlib.sha256(frame_bytes(frame)).hexdigest()


def payload(i: int, n: int = 256) -> bytes:
    """A deterministic, structurally-distinct judge frame.

    Pseudo-random bytes so the perception cache's dHash of two different payloads is far apart
    (measured min pairwise Hamming >= 20 for the indices used here, vs a tau_bits of 6) — a cache
    HIT in these tests therefore only ever comes from a genuinely repeated frame, never from an
    accidental hash collision.
    """
    return random.Random(1_000_000 + i).randbytes(n)


YES = Verdict(satisfied=True, confidence=0.95, evidence="scripted: condition holds")
NO = Verdict(satisfied=False, confidence=0.95, evidence="scripted: condition does not hold")
UNSURE = Verdict(satisfied=True, confidence=0.20, evidence="scripted: cannot tell")
UNLISTED = Verdict(satisfied=False, confidence=0.95, evidence="scripted: unlisted frame")


class ScriptedVision:
    """A scripted, network-free vision backend that RECORDS exactly what it was shown.

    Keyed on the content hash of the frame it is handed, so the recording is the ground truth for
    "what the judge actually saw" that SV-4.7 compares Incident evidence against. Wiring, never
    intelligence: it looks a hash up in a dict.
    """

    PARSES_CLAIM_CHANNEL = True

    def __init__(self, script: dict[str, Verdict] | None = None, *, delay_s: float = 0.0) -> None:
        self.script = dict(script or {})
        self.delay_s = delay_s
        self.calls: list[dict] = []
        self.seen_hashes: set[str] = set()
        self.n_calls = 0

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        h = sha(frame)
        self.n_calls += 1
        self.seen_hashes.add(h)
        if len(self.calls) < 20_000:  # bounded recorder: never the thing that grows in SV-4.6
            self.calls.append({"condition": condition, "frame_sha": h, "roi": repr(roi)})
        return self.script.get(h, UNLISTED)

    async def detect(self, frame: Any, goals: list) -> list:
        return []


def make_core(
    vision: ScriptedVision,
    *,
    goal: Goal,
    settings: Any = None,
    on_trace: Callable | None = None,
) -> Percept:
    core = Percept.create(
        vision=vision, discover_plugins=False, settings=settings, on_trace=on_trace
    )
    core.add_goal(goal)
    return core


@dataclass
class RunOutput:
    incidents: list
    stats: RuntimeStats
    health: list
    trace: list
    asks: list
    metrics: dict


def drive(core: Percept, source: Iterable, *, keep_incidents: bool = True, **kw) -> RunOutput:
    """Run the REAL public runtime path over ``source`` and collect every observable channel."""
    health: list = []
    trace: list = []
    asks: list = []
    stats: RuntimeStats = kw.pop("stats", None) or RuntimeStats()
    incidents: list = []
    n_incidents = [0]

    def _trace(ev):
        trace.append(ev)

    async def _main():
        async for inc in run(
            core,
            source=source,
            stats=stats,
            on_health=health.append,
            on_trace=_trace,
            on_ask=asks.append,
            **kw,
        ):
            n_incidents[0] += 1
            if keep_incidents:
                incidents.append(inc)

    asyncio.run(_main())
    return RunOutput(
        incidents=incidents,
        stats=stats,
        health=health,
        trace=trace,
        asks=asks,
        metrics=dict(core.metrics.snapshot()),
    )


def settings_with(**overrides) -> Any:
    """A Settings built off the ambient default with nested sub-settings replaced."""
    cfg = load_settings()
    for section, changes in overrides.items():
        cfg = dataclasses.replace(
            cfg, **{section: dataclasses.replace(getattr(cfg, section), **changes)}
        )
    return cfg


def alternating(seq: list[tuple[float, bytes]]) -> list[tuple[float, Any, bytes]]:
    """Attach an alternating tier-0 motion signal so every tick after the first escalates."""
    return [(t, (MOVED if i % 2 else STILL), p) for i, (t, p) in enumerate(seq)]


# ---------------------------------------------------------------------------------------------
# SV-4.1 — gate decision + state
# ---------------------------------------------------------------------------------------------


def sv4_1_gate_decision_and_state() -> TestResult:
    r = TestResult("SV-4.1", "Gate decision + state (rising edge / refractory / re-arm / distinctness)")

    # ---- scenario A: baseline rising edge, no refractory -------------------------------------
    # ticks 0..13, even = confident-false, odd = confident-true. Every odd tick is a rising edge
    # (the preceding even tick re-armed it), so the baseline gate must fire on all 7.
    tags = ["no" if i % 2 == 0 else "yes" for i in range(14)]
    frames = [payload(i) for i in range(14)]
    script = {sha(frames[i]): (YES if tags[i] == "yes" else NO) for i in range(14)}
    src = alternating([(float(i), frames[i]) for i in range(14)])

    v = ScriptedVision(script)
    core = make_core(v, goal=Goal(id="zone", condition="a person is in the zone", say="Alert"))
    out_a = drive(core, src)

    fire_ts = [i.ts for i in out_a.incidents]
    r.check(
        "A/rising-edge fires exactly once per occurrence",
        fire_ts == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        f"fired at {fire_ts}; expected one fire per false->true edge",
        fire_ts=fire_ts,
    )

    # exactly-once: a continuing true state must NOT re-fire. Prove it with a held-true run.
    held_tags = ["no"] + ["yes"] * 7
    held_frames = [payload(100 + i) for i in range(8)]
    held_script = {
        sha(held_frames[i]): (YES if held_tags[i] == "yes" else NO) for i in range(8)
    }
    v_h = ScriptedVision(held_script)
    core_h = make_core(v_h, goal=Goal(id="zone", condition="c", say="Alert"))
    out_h = drive(core_h, alternating([(float(i), held_frames[i]) for i in range(8)]))
    held_reasons = [
        e.data.get("gate_reason")
        for e in out_h.trace
        if e.stage == "gate.classified"
    ]
    r.check(
        "A/held-true occurrence fires exactly ONCE",
        len(out_h.incidents) == 1 and held_reasons.count("already_fired") == 6,
        f"{len(out_h.incidents)} incident(s); gate reasons {held_reasons}",
        n_incidents=len(out_h.incidents),
        reasons=held_reasons,
    )

    # ---- distinctness of repeated occurrences -------------------------------------------------
    ids = [i.id for i in out_a.incidents]
    jids = [i.judgment_event_id for i in out_a.incidents]
    seqs = [i.evidence_seq for i in out_a.incidents]
    newest_evidence = [
        sha(i.snippet.frames[-1]) if i.snippet and i.snippet.frames else None
        for i in out_a.incidents
    ]
    r.check(
        "A/repeated occurrences produce DISTINCT Incidents",
        len(set(ids)) == len(ids)
        and len(set(jids)) == len(jids)
        and len(set(seqs)) == len(seqs)
        and len(set(newest_evidence)) == len(newest_evidence)
        and None not in newest_evidence,
        f"{len(ids)} incidents; {len(set(ids))} distinct ids, {len(set(jids))} distinct "
        f"judgment_event_ids, {len(set(seqs))} distinct evidence_seqs, "
        f"{len(set(newest_evidence))} distinct newest-evidence hashes",
        ids=ids,
        judgment_event_ids=jids,
        evidence_seqs=seqs,
    )
    r.check(
        "A/no Incident is a field-for-field duplicate of another",
        len({_incident_identity(i) for i in out_a.incidents}) == len(out_a.incidents),
        "every delivered Incident differs in id/ts/verdict/evidence-link",
    )

    # ---- scenario B: refractory honoured -------------------------------------------------------
    v_b = ScriptedVision(script)
    core_b = make_core(
        v_b,
        goal=Goal(id="zone", condition="a person is in the zone", say="Alert"),
        settings=settings_with(gate={"refractory_s": 10.0}),
    )
    out_b = drive(core_b, src)
    b_ts = [i.ts for i in out_b.incidents]
    suppressions = [h for h in out_b.health if h.kind == "gate_suppressed"]
    supp_reasons = [(h.data or {}).get("reason") for h in suppressions]
    supp_at = [(h.data or {}).get("at") for h in suppressions]
    # t=1 fires; t=3,5,7,9 land inside the 10 s window and are suppressed; t=11 is exactly 10 s
    # after the last real fire (>= refractory) so it fires; t=13 is 2 s later -> suppressed.
    r.check(
        "B/refractory suppresses re-fires inside the window",
        b_ts == [1.0, 11.0],
        f"fired at {b_ts}; expected [1.0, 11.0] under refractory_s=10.0",
        fire_ts=b_ts,
    )
    r.check(
        "B/every suppressed fire carries a typed 'refractory' receipt",
        len(suppressions) == 5 and set(supp_reasons) == {"refractory"} and supp_at == [3.0, 5.0, 7.0, 9.0, 13.0],
        f"{len(suppressions)} gate_suppressed events at {supp_at} reasons={set(supp_reasons)}",
        suppressed_at=supp_at,
    )
    r.check(
        "B/re-arm after the refractory window elapses",
        11.0 in b_ts,
        "a genuine later occurrence fires once refractory_s has elapsed since the last real fire",
    )

    # ---- gate reason vocabulary is closed and action-consistent --------------------------------
    all_reasons = [
        e.data.get("gate_reason")
        for out in (out_a, out_b, out_h)
        for e in out.trace
        if e.stage == "gate.classified"
    ]
    all_actions = [
        e.data.get("action")
        for out in (out_a, out_b, out_h)
        for e in out.trace
        if e.stage == "gate.classified"
    ]
    unknown = sorted({x for x in all_reasons if x not in GATE_REASONS})
    mismatched = [
        (rr, aa) for rr, aa in zip(all_reasons, all_actions) if REASON_ACTION.get(rr) != aa
    ]
    r.check(
        "gate reasons stay inside the closed GATE_REASONS vocabulary",
        not unknown,
        f"unknown reasons: {unknown}" if unknown else f"{len(all_reasons)} decisions, all typed",
        observed=sorted(set(all_reasons)),
    )
    r.check(
        "every gate reason implies exactly its documented action",
        not mismatched,
        f"mismatches: {mismatched[:5]}" if mismatched else "reason -> action agreement is total",
    )

    r.measurements = {
        "baseline_fire_ts": fire_ts,
        "refractory_fire_ts": b_ts,
        "refractory_suppressed_at": supp_at,
        "held_true_reasons": held_reasons,
        "observed_gate_reasons": sorted(set(all_reasons)),
    }
    return r


def _incident_identity(inc: Any) -> tuple:
    return (
        inc.id,
        inc.ts,
        inc.concern_id,
        inc.evidence_seq,
        inc.judgment_event_id,
        inc.verdict.satisfied,
        round(float(inc.verdict.confidence), 6),
    )


# ---------------------------------------------------------------------------------------------
# SV-4.5 — cache correctness
# ---------------------------------------------------------------------------------------------


def _offset_ramp(rows_a: int = 8, rows_b: int = 8, width: int = 16, base: int = 0, offset: int = 120) -> bytes:
    """A frame whose top and bottom halves have IDENTICAL horizontal gradient structure but
    DIFFERENT absolute content (a constant luminance offset).

    dHash is brightness-invariant by construction (it compares adjacent pixels), so cropping to
    the top half and cropping to the bottom half produce the SAME hash while the two regions are
    genuinely different scenes. This is the adversarial construction SV-4.5 needs: a Watch config
    change (a moved ROI) that the cache key cannot see.
    """
    out = bytearray()
    for _ in range(rows_a):
        out += bytes((base + (x * 7) % 100) for x in range(width))
    for _ in range(rows_b):
        out += bytes((base + offset + (x * 7) % 100) for x in range(width))
    return bytes(out)


def sv4_5_cache_correctness() -> TestResult:
    r = TestResult("SV-4.5", "Cache correctness (hit / miss / invalidation; zero stale satisfactions)")
    stale_satisfactions: list[dict] = []

    cache_cfg = settings_with(cache={"enabled": True, "audit_rate": 0.0, "ttl_s": 10_000.0})

    # ---- A. HIT / MISS through the real runtime path -------------------------------------------
    p_rep = payload(1)
    p_new = payload(2)
    script = {sha(p_rep): NO, sha(p_new): YES}
    src = alternating(
        [(0.0, p_rep), (1.0, p_rep), (2.0, p_rep), (3.0, p_rep)]
    )
    v = ScriptedVision(script)
    core = make_core(v, goal=Goal(id="zone", condition="cond A", say="Alert"), settings=cache_cfg)
    out = drive(core, src)
    r.check(
        "A/an unchanged frame is served from cache (provider called once)",
        v.n_calls == 1 and out.metrics.get("cache.hit", 0) == out.stats.escalations - 1,
        f"escalations={out.stats.escalations} provider_calls={v.n_calls} "
        f"cache.hit={out.metrics.get('cache.hit', 0)}",
        escalations=out.stats.escalations,
        provider_calls=v.n_calls,
        cache_hits=out.metrics.get("cache.hit", 0),
    )

    # distinct frames must MISS (no accidental dHash collision in the fixture set)
    ps = [payload(10 + i) for i in range(6)]
    hs = [dhash(p) for p in ps]
    min_ham = min(hamming(a, b) for i, a in enumerate(hs) for b in hs[i + 1 :])
    script2 = {sha(p): NO for p in ps}
    v2 = ScriptedVision(script2)
    core2 = make_core(v2, goal=Goal(id="zone", condition="cond A", say="Alert"), settings=cache_cfg)
    out2 = drive(core2, alternating([(float(i), ps[i]) for i in range(6)]))
    r.check(
        "A/structurally distinct frames MISS (one provider call each)",
        v2.n_calls == out2.stats.escalations and out2.metrics.get("cache.hit", 0) == 0,
        f"escalations={out2.stats.escalations} provider_calls={v2.n_calls} "
        f"min pairwise dHash Hamming={min_ham} (tau_bits={cache_cfg.cache.tau_bits})",
        min_pairwise_hamming=min_ham,
    )

    # ---- B. invalidation ------------------------------------------------------------------------
    layer = core.cache.layers[0]
    before = len(layer)
    core.invalidate_cache("zone")  # goal-change invalidation (1st in the documented order)
    r.check(
        "B/goal-change invalidation drops the goal's entries",
        before > 0 and len(layer) == 0,
        f"entries before={before} after={len(layer)}",
    )

    # The "same goal id, new condition" staleness vector: verify the runtime CLOSES it at the front
    # door rather than relying on the cache to notice.
    v3 = ScriptedVision({sha(p_rep): NO})
    core3 = make_core(v3, goal=Goal(id="zone", condition="cond A", say="Alert"), settings=cache_cfg)
    drive(core3, alternating([(0.0, p_rep), (1.0, p_rep)]))
    layer3 = core3.cache.layers[0]
    entries_armed = len(layer3)

    same_id_refused = False
    try:
        core3.add_goal(Goal(id="zone", condition="cond B — a DIFFERENT question", say="Alert"))
    except ValueError:
        same_id_refused = True
    condition_update_refused = False
    try:
        core3.apply_goal_ops([GoalOp(op="update", goal_id="zone", fields={"condition": "cond B"})])
    except ValueError:
        condition_update_refused = True
    r.check(
        "B/the runtime FORBIDS re-arming a goal id under a new condition",
        same_id_refused and condition_update_refused and entries_armed > 0,
        f"add_goal(same id) refused={same_id_refused}; "
        f"GoalOp(update, condition=...) refused={condition_update_refused} — the "
        f"'stale entry under a changed condition' vector is closed at the front door, not left to "
        f"the cache",
    )
    # ...and any accepted goal mutation still invalidates the goal's cache.
    core3.apply_goal_ops([GoalOp(op="update", goal_id="zone", fields={"say": "New alert text"})])
    r.check(
        "B/an accepted goal mutation invalidates that goal's cache",
        len(layer3) == 0,
        f"entries before mutation={entries_armed} after={len(layer3)}",
    )

    # TTL: a fake clock proves the hard ceiling on the live layer of a live core.
    tick = [0.0]
    v4 = ScriptedVision({sha(p_rep): NO})
    core4 = Percept.create(
        vision=v4,
        discover_plugins=False,
        settings=settings_with(cache={"enabled": True, "audit_rate": 0.0, "ttl_s": 5.0}),
        clock=lambda: tick[0],
    )
    core4.add_goal(Goal(id="zone", condition="cond A", say="Alert"))
    layer4 = core4.cache.layers[0]
    key4 = CacheKey(goal_id="zone", frame=p_rep, roi=None, entity_id=None, condition="cond A")
    layer4.store(key4, NO, 0.95)
    hit_fresh = layer4.lookup(key4).status
    tick[0] = 6.0
    hit_stale = layer4.lookup(key4).status
    r.check(
        "B/TTL is a hard ceiling (past it, always a miss)",
        hit_fresh == "HIT" and hit_stale == "DEFER",
        f"at age 0s -> {hit_fresh}; at age 6s (ttl 5s) -> {hit_stale}",
    )

    # assertion-override invalidation
    v5 = ScriptedVision({sha(p_rep): NO})
    core5 = make_core(v5, goal=Goal(id="zone", condition="cond A", say="Alert"), settings=cache_cfg)
    layer5 = core5.cache.layers[0]
    layer5.store(
        CacheKey(goal_id="zone", frame=p_rep, roi=None, entity_id=None, condition="cond A"),
        NO,
        0.95,
    )
    n_before = len(layer5)
    core5.invalidate_cache_entity(None)
    r.check(
        "B/assertion-override invalidation drops entries resting on the entity",
        n_before > 0 and len(layer5) == 0,
        f"entries before={n_before} after={len(layer5)}",
    )

    # ---- C. THE BAR: zero stale satisfactions of a CHANGED Watch config -------------------------
    # LAYER PROBE. The live FrameHashLayer of a live core, keyed exactly as percept/agent.py:5934
    # keys it: CacheKey(goal_id, frame, roi, entity_id, condition).
    v6 = ScriptedVision({})
    core6 = make_core(v6, goal=Goal(id="zone", condition="cond A", say="Alert"), settings=cache_cfg)
    layer6 = core6.cache.layers[0]

    collide_frame = _offset_ramp()
    roi_top = Box(0.0, 0.0, 1.0, 0.5)
    roi_bottom = Box(0.0, 0.5, 1.0, 0.5)
    roi_hash_top = dhash(collide_frame, roi_top)
    roi_hash_bottom = dhash(collide_frame, roi_bottom)

    mutations = [
        (
            "condition changed",
            CacheKey("zone", p_rep, None, None, "cond A"),
            CacheKey("zone", p_rep, None, None, "cond B"),
        ),
        (
            "goal id changed",
            CacheKey("zone", p_rep, None, None, "cond A"),
            CacheKey("other-goal", p_rep, None, None, "cond A"),
        ),
        (
            "entity scope changed",
            CacheKey("zone", p_rep, None, None, "cond A"),
            CacheKey("zone", p_rep, None, "e2", "cond A"),
        ),
        (
            "ROI moved to a structurally different region",
            CacheKey("zone", payload(3), Box(0.0, 0.0, 0.5, 0.5), None, "cond A"),
            CacheKey("zone", payload(3), Box(0.5, 0.5, 0.5, 0.5), None, "cond A"),
        ),
        (
            "ROI moved to a dHash-colliding region (different content, same gradient structure)",
            CacheKey("zone", collide_frame, roi_top, None, "cond A"),
            CacheKey("zone", collide_frame, roi_bottom, None, "cond A"),
        ),
    ]

    mutation_results = []
    for name, armed_key, changed_key in mutations:
        layer6.invalidate(None)
        layer6.store(armed_key, YES, 0.95)
        control = layer6.lookup(armed_key)
        probe = layer6.lookup(changed_key)
        served_stale = probe.status == "HIT"
        mutation_results.append(
            {
                "mutation": name,
                "unchanged_config_status": control.status,
                "changed_config_status": probe.status,
                "stale_satisfaction": served_stale,
                "served_verdict_satisfied": None if probe.verdict is None else probe.verdict.satisfied,
            }
        )
        if served_stale:
            stale_satisfactions.append(mutation_results[-1])

    r.check(
        "C/ZERO stale satisfactions of a changed Watch config",
        not stale_satisfactions,
        (
            "STALE SATISFACTIONS: "
            + "; ".join(m["mutation"] for m in stale_satisfactions)
            if stale_satisfactions
            else f"all {len(mutations)} config mutations produced a clean miss"
        ),
        mutations=mutation_results,
    )

    if stale_satisfactions:
        r.defects.append(
            Defect(
                id="SV-4.5-D1",
                severity="high",
                where=(
                    "percept/cache/framehash.py:195 (slot = (goal_id, entity_id, condition)) "
                    "and percept/cache/framehash.py:213 (roi folded into the HASH, not the KEY)"
                ),
                summary=(
                    "FrameHashLayer's entry slot is (goal_id, entity_id, condition). The ROI is "
                    "folded into the dHash rather than the key, and dHash is brightness-invariant. "
                    "Two DIFFERENT regions of the same frame with identical gradient structure "
                    "hash identically, so moving a Watch's ROI (goal.state['region_hint'] -> "
                    "PerceptionSpec roi) is served the OLD region's verdict: a stale entry "
                    "satisfying a CHANGED Watch config."
                ),
                repro=(
                    "frame = _offset_ramp()  # top half = ramp, bottom half = ramp + 120\n"
                    "layer.store(CacheKey('zone', frame, Box(0,0,1,0.5), None, 'cond A'), YES, .95)\n"
                    "layer.lookup(CacheKey('zone', frame, Box(0,0.5,1,0.5), None, 'cond A')).status\n"
                    f"# -> 'HIT'  (dhash(top)={roi_hash_top}, dhash(bottom)={roi_hash_bottom}, "
                    f"Hamming={hamming(roi_hash_top, roi_hash_bottom)} <= tau_bits="
                    f"{cache_cfg.cache.tau_bits})"
                ),
            )
        )

    r.measurements = {
        "provider_calls_repeated_frame": v.n_calls,
        "cache_hits_repeated_frame": out.metrics.get("cache.hit", 0),
        "min_pairwise_dhash_hamming_distinct_frames": min_ham,
        "tau_bits": cache_cfg.cache.tau_bits,
        "config_mutations": mutation_results,
        "stale_satisfactions": len(stale_satisfactions),
        "roi_collision": {
            "dhash_top": roi_hash_top,
            "dhash_bottom": roi_hash_bottom,
            "hamming": hamming(roi_hash_top, roi_hash_bottom),
        },
    }
    return r


# ---------------------------------------------------------------------------------------------
# SV-4.6 — media buffers
# ---------------------------------------------------------------------------------------------


def _memory_leg(
    *,
    label: str,
    n_frames: int,
    frame_bytes_n: int,
    snippet_size: int,
    snippet_max_bytes: int,
    sample_every: int = 50,
    yes_every: int = 200,
) -> dict:
    """One long synthetic run through percept.run, instrumented for memory.

    The HARNESS allocates O(1) per frame — the frame carries its own index in its first four
    bytes so the judge needs no growing index map, health events are counted rather than
    retained, and incidents are consumed and dropped. Without that discipline the curve measures
    the test's accumulation instead of the runtime's (an earlier revision of this leg reported
    +941 bytes/frame that were entirely the harness's own retained health events and hash map).
    """

    def frame_for(i: int) -> bytes:
        return i.to_bytes(4, "big") + random.Random(5_000_000 + i).randbytes(frame_bytes_n - 4)

    class LongScript(ScriptedVision):
        """Decodes the frame's self-describing index; retains nothing per call."""

        def __init__(self) -> None:
            super().__init__({})

        async def judge(self, condition, frame, **kw):
            self.n_calls += 1
            idx = int.from_bytes(frame_bytes(frame)[:4], "big")
            return YES if (idx % yes_every) == 1 else NO

    samples: list[tuple[int, int, int]] = []  # (frame_no, tracemalloc_current, rss_bytes)
    frames_seen = [0]

    def source():
        for i in range(n_frames):
            yield (float(i) * 0.25, (MOVED if i % 2 else STILL), frame_for(i))

    core = make_core(LongScript(), goal=Goal(id="zone", condition="c", say="Alert"))

    def on_frame(t, f):
        frames_seen[0] += 1
        n = frames_seen[0]
        if n % sample_every == 0:
            cur, _peak = tracemalloc.get_traced_memory()
            samples.append((n, cur, _rss_bytes()))

    stats = RuntimeStats()
    health_counts: dict[str, int] = {}
    max_held = {"frames": 0, "bytes": 0}

    def on_health(ev) -> None:
        health_counts[ev.kind] = health_counts.get(ev.kind, 0) + 1
        if ev.kind == "preroll_capped":
            d = ev.data or {}
            max_held["frames"] = max(max_held["frames"], int(d.get("held_frames", 0)))
            max_held["bytes"] = max(max_held["bytes"], int(d.get("held_bytes", 0)))

    n_incidents = [0]
    snippet_lengths: list[int] = []
    snippet_byte_sizes: list[int] = []
    preroll_present = [0]
    preroll_absent = [0]

    async def _main():
        async for inc in run(
            core,
            source=source(),
            stats=stats,
            on_health=on_health,
            on_frame=on_frame,
            snippet_size=snippet_size,
            snippet_max_bytes=snippet_max_bytes,
            snippet_pre_s=5.0,
            fps=100.0,
        ):
            n_incidents[0] += 1  # consume and DROP
            if inc.snippet is not None and inc.snippet.frames:
                snippet_lengths.append(len(inc.snippet.frames))
                snippet_byte_sizes.append(sum(len(frame_bytes(f)) for f in inc.snippet.frames))
                if len(inc.snippet.frames) > 1 and inc.snippet.t_start < inc.ts:
                    preroll_present[0] += 1
                else:
                    preroll_absent[0] += 1

    tracemalloc.start()
    t0 = time.monotonic()
    asyncio.run(_main())
    elapsed = time.monotonic() - t0
    cur_end, peak_end = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    steady = [s for s in samples if s[0] >= n_frames // 2]
    xs = [float(s[0]) for s in steady]
    ys = [float(s[1]) for s in steady]
    slope, r2 = _linfit(xs, ys)
    mean_steady = sum(ys) / len(ys) if ys else 0.0
    q2, q4 = ys[: len(ys) // 2], ys[len(ys) // 2 :]
    drift_pct = (
        100.0 * (sum(q4) / len(q4) - sum(q2) / len(q2)) / (sum(q2) / len(q2))
        if q2 and q4 and sum(q2)
        else 0.0
    )

    return {
        "label": label,
        "n_frames": n_frames,
        "frame_bytes": frame_bytes_n,
        "snippet_size": snippet_size,
        "snippet_max_bytes": snippet_max_bytes,
        "elapsed_s": round(elapsed, 2),
        "incidents": n_incidents[0],
        "n_samples": len(samples),
        "n_steady_samples": len(steady),
        "mean_steady_heap_bytes": mean_steady,
        "drift_pct_second_half": drift_pct,
        "slope_bytes_per_frame": slope,
        "linfit_r2": r2,
        "projected_growth_bytes_over_run": slope * n_frames,
        "heap_end_bytes": cur_end,
        "heap_peak_bytes": peak_end,
        "rss_maxrss_bytes": _rss_bytes(),
        "preroll_frames_dropped": stats.preroll_frames_dropped,
        "preroll_capped_events": health_counts.get("preroll_capped", 0),
        "max_held_frames": max_held["frames"],
        "max_held_bytes": max_held["bytes"],
        "max_snippet_frames": max(snippet_lengths) if snippet_lengths else 0,
        "max_snippet_bytes": max(snippet_byte_sizes) if snippet_byte_sizes else 0,
        "incidents_with_preroll": preroll_present[0],
        "incidents_without_preroll": preroll_absent[0],
        "curve": [{"frame": n, "heap_bytes": cur, "rss_bytes": rss} for (n, cur, rss) in samples],
    }


def sv4_6_media_buffers(n_frames: int = 3000, frame_bytes_n: int = 16_384) -> TestResult:
    r = TestResult("SV-4.6", "Media buffers (bounded retention, pre-admission window, no unbounded growth)")

    # Leg 1: the BYTE budget binds (48 x 16 KiB = 768 KiB of media against a 250 KiB ceiling).
    byte_bound = _memory_leg(
        label="byte-bounded",
        n_frames=n_frames,
        frame_bytes_n=frame_bytes_n,
        snippet_size=48,
        snippet_max_bytes=256_000,
    )
    # Leg 2: the byte bound DISABLED (snippet_max_bytes=0) so only the COUNT bound holds the ring.
    count_only = _memory_leg(
        label="count-only",
        n_frames=n_frames,
        frame_bytes_n=frame_bytes_n,
        snippet_size=48,
        snippet_max_bytes=0,
    )
    legs = {"byte_bounded": byte_bound, "count_only": count_only}

    for key, leg in legs.items():
        r.check(
            f"{key}/the memory curve was actually sampled (guards against a vacuously flat result)",
            leg["n_steady_samples"] >= 20 and leg["mean_steady_heap_bytes"] > 0.0,
            f"{leg['n_samples']} samples, {leg['n_steady_samples']} in the steady-state half; "
            f"mean steady-state heap {leg['mean_steady_heap_bytes'] / 1e6:.2f} MB",
        )
        r.check(
            f"{key}/steady-state Python heap is FLAT (< 5% drift across the second half)",
            leg["n_steady_samples"] >= 20
            and leg["mean_steady_heap_bytes"] > 0.0
            and abs(leg["drift_pct_second_half"]) < 5.0,
            f"drift {leg['drift_pct_second_half']:+.2f}%; slope "
            f"{leg['slope_bytes_per_frame']:+.1f} bytes/frame (projected "
            f"{leg['projected_growth_bytes_over_run'] / 1e6:+.2f} MB over {leg['n_frames']} "
            f"frames); mean steady-state heap {leg['mean_steady_heap_bytes'] / 1e6:.2f} MB; "
            f"peak {leg['heap_peak_bytes'] / 1e6:.2f} MB",
        )

    # -- bounded retention, announced -------------------------------------------------------------
    r.check(
        "byte_bounded/the pre-roll ring is bounded by BOTH the count and the byte budget",
        byte_bound["max_held_frames"] <= byte_bound["snippet_size"]
        and byte_bound["max_held_bytes"] <= byte_bound["snippet_max_bytes"] + frame_bytes_n,
        f"max held {byte_bound['max_held_frames']} frames / {byte_bound['max_held_bytes']} bytes "
        f"(bounds: {byte_bound['snippet_size']} frames, {byte_bound['snippet_max_bytes']} bytes "
        f"+ one newest frame, which is never evicted)",
    )
    r.check(
        "byte_bounded/every byte-budget eviction is announced (no silent drop)",
        byte_bound["preroll_frames_dropped"] > 0
        and byte_bound["preroll_capped_events"] == byte_bound["preroll_frames_dropped"],
        f"stats.preroll_frames_dropped={byte_bound['preroll_frames_dropped']}, "
        f"preroll_capped health events={byte_bound['preroll_capped_events']}",
    )
    r.check(
        "count_only/the COUNT bound alone still caps retention",
        count_only["max_snippet_frames"] <= count_only["snippet_size"]
        and count_only["preroll_capped_events"] == 0
        and count_only["max_snippet_bytes"] <= count_only["snippet_size"] * frame_bytes_n,
        f"max snippet {count_only['max_snippet_frames']} frames / "
        f"{count_only['max_snippet_bytes']} bytes with the byte bound disabled "
        f"(ceiling {count_only['snippet_size']} x {frame_bytes_n} = "
        f"{count_only['snippet_size'] * frame_bytes_n} bytes)",
    )
    for key, leg in legs.items():
        r.check(
            f"{key}/no snippet exceeds the retention policy",
            leg["max_snippet_frames"] <= leg["snippet_size"]
            and leg["max_snippet_bytes"]
            <= (leg["snippet_max_bytes"] or leg["snippet_size"] * frame_bytes_n) + frame_bytes_n,
            f"max snippet {leg['max_snippet_frames']} frames / {leg['max_snippet_bytes']} bytes",
        )
        r.check(
            f"{key}/the PRE-ADMISSION window is retained (every incident carries lead-up frames)",
            leg["incidents_without_preroll"] == 0
            and leg["incidents_with_preroll"] == leg["incidents"]
            and leg["incidents"] > 0,
            f"{leg['incidents_with_preroll']}/{leg['incidents']} incidents carry pre-fire frames "
            f"({leg['incidents_without_preroll']} without)",
        )

    r.measurements = {"legs": legs}
    return r


def _rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports kilobytes.
    return int(ru) if sys.platform == "darwin" else int(ru) * 1024


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy) / (sxx * syy) if syy else 0.0
    return slope, r2


# ---------------------------------------------------------------------------------------------
# SV-4.7 — EVIDENCE HALLUCINATION (zero tolerance)
# ---------------------------------------------------------------------------------------------


def _audit_snippet_media(
    incidents: list,
    *,
    fed: list[tuple[float, bytes]],
    judge_seen: set[str],
    label: str,
) -> dict:
    """Compare every Incident's evidence media against what was actually fed to the runtime."""
    fed_seq = [sha(p) for (_t, p) in fed]
    fed_ts = [t for (t, _p) in fed]
    fed_hashes = set(fed_seq)
    report = {
        "label": label,
        "incidents": len(incidents),
        "evidence_frames": 0,
        "frames_not_in_fed_stream": [],
        "not_a_contiguous_window_of_the_fed_stream": [],
        "snippets_after_fire": [],
        "newest_frame_never_judged": [],
        "empty_snippets": [],
    }
    for inc in incidents:
        snip = inc.snippet
        if snip is None or not snip.frames:
            report["empty_snippets"].append(inc.id)
            continue
        hashes = [sha(f) for f in snip.frames]
        report["evidence_frames"] += len(hashes)
        for h in hashes:
            if h not in fed_hashes:
                report["frames_not_in_fed_stream"].append({"incident": inc.id, "sha256": h})
        # The snippet must be an ORDERED, CONTIGUOUS window of the frames actually fed, ending at
        # or before the fire. Alignment (not a hash-set membership test) is what rules out
        # re-ordering, interleaving and duplication of real media.
        n = len(hashes)
        aligned = [
            k
            for k in range(len(fed_seq) - n + 1)
            if fed_seq[k : k + n] == hashes and fed_ts[k + n - 1] <= inc.ts + 1e-9
        ]
        if not aligned:
            report["not_a_contiguous_window_of_the_fed_stream"].append(
                {"incident": inc.id, "ts": inc.ts, "n_frames": n}
            )
        if snip.t_end > inc.ts + 1e-9:
            report["snippets_after_fire"].append(
                {"incident": inc.id, "t_end": snip.t_end, "incident_ts": inc.ts}
            )
        if hashes[-1] not in judge_seen:
            report["newest_frame_never_judged"].append(
                {"incident": inc.id, "ts": inc.ts, "sha256": hashes[-1]}
            )
    report["clean"] = not (
        report["frames_not_in_fed_stream"]
        or report["not_a_contiguous_window_of_the_fed_stream"]
        or report["snippets_after_fire"]
        or report["empty_snippets"]
    )
    return report


def sv4_7_evidence_integrity() -> TestResult:
    r = TestResult("SV-4.7", "EVIDENCE INTEGRITY — zero tolerance for synthesized/substituted evidence")

    # ---- A. baseline: multi-occurrence run, exact content-hash audit ---------------------------
    n = 16
    frames = [payload(200 + i) for i in range(n)]
    script = {sha(frames[i]): (YES if i % 2 else NO) for i in range(n)}
    fed = [(float(i), frames[i]) for i in range(n)]
    v = ScriptedVision(script)
    core = make_core(v, goal=Goal(id="zone", condition="c", say="Alert"))
    out_a = drive(core, alternating(fed))
    audit_a = _audit_snippet_media(out_a.incidents, fed=fed, judge_seen=v.seen_hashes, label="baseline")

    r.check(
        "A/every evidence frame resolves to REAL fed media (content-hash exact)",
        not audit_a["frames_not_in_fed_stream"] and audit_a["evidence_frames"] > 0,
        f"{audit_a['evidence_frames']} evidence frames across {audit_a['incidents']} incidents; "
        f"{len(audit_a['frames_not_in_fed_stream'])} unmatched",
        **{k: audit_a[k] for k in ("evidence_frames", "incidents")},
    )
    r.check(
        "A/evidence is a contiguous ordered window of the fed stream, ending at/before the fire",
        not audit_a["not_a_contiguous_window_of_the_fed_stream"]
        and not audit_a["snippets_after_fire"],
        "every snippet aligns exactly to a contiguous run of the frames actually fed — no "
        "re-ordering, interleaving, duplication or post-dating",
    )
    r.check(
        "A/no Incident is delivered with empty evidence",
        not audit_a["empty_snippets"],
        f"{len(audit_a['empty_snippets'])} incident(s) without a snippet",
    )
    r.check(
        "A/the frame the judge actually saw is the newest evidence frame",
        not audit_a["newest_frame_never_judged"],
        f"{len(audit_a['newest_frame_never_judged'])} incident(s) whose newest evidence frame the "
        f"judge never saw",
        offenders=audit_a["newest_frame_never_judged"],
    )

    # ---- A'. negative control: the detector is NOT vacuous ---------------------------------------
    forged = list(out_a.incidents)
    fake_fed = [(t, p) for (t, p) in fed if sha(p) != sha(frames[1])]
    audit_neg = _audit_snippet_media(
        forged, fed=fake_fed, judge_seen=v.seen_hashes, label="negative-control"
    )
    r.check(
        "A'/NEGATIVE CONTROL: the audit detects evidence absent from the fed stream",
        len(audit_neg["frames_not_in_fed_stream"]) > 0,
        f"withholding one fed frame from the ground truth produced "
        f"{len(audit_neg['frames_not_in_fed_stream'])} unmatched evidence frames — the detector fires",
    )

    # ---- B. adversarial: an occurrence spanning an ADMISSION GAP ---------------------------------
    # A long motionless stretch means the tier-0 gate escalates nothing; the occurrence's evidence
    # must still be real fed media, and the un-admitted frames must appear as honest pre-roll.
    gap_frames = [payload(300 + i) for i in range(14)]
    gap_script = {sha(gap_frames[i]): (YES if i in (1, 13) else NO) for i in range(14)}
    gate_signal = [STILL, MOVED] + [MOVED] * 10 + [STILL, MOVED]
    gap_src = [(float(i), gate_signal[i], gap_frames[i]) for i in range(14)]
    gap_fed = [(float(i), gap_frames[i]) for i in range(14)]
    v_b = ScriptedVision(gap_script)
    core_b = make_core(v_b, goal=Goal(id="zone", condition="c", say="Alert"))
    out_b = drive(core_b, gap_src)
    audit_b = _audit_snippet_media(
        out_b.incidents, fed=gap_fed, judge_seen=v_b.seen_hashes, label="admission-gap"
    )
    n_admitted = out_b.stats.escalations
    r.check(
        "B/evidence across an ADMISSION GAP is real fed media",
        audit_b["clean"] and audit_b["evidence_frames"] > 0,
        f"{out_b.stats.frames} frames observed, {n_admitted} admitted (gap of "
        f"{out_b.stats.frames - n_admitted} un-admitted ticks); "
        f"{audit_b['evidence_frames']} evidence frames, {len(audit_b['frames_not_in_fed_stream'])} unmatched",
        frames=out_b.stats.frames,
        admitted=n_admitted,
    )
    gap_covered = any(
        inc.snippet is not None
        and inc.snippet.frames
        and len(inc.snippet.frames) > (n_admitted // max(1, len(out_b.incidents)))
        for inc in out_b.incidents
    )
    r.check(
        "B/un-admitted frames are retained as pre-roll, not fabricated",
        gap_covered,
        "the incident's pre-roll spans more frames than were ever admitted — the retained window "
        "covers the gap with real captured media",
    )

    # ---- C. adversarial: evidence AFTER A CACHE HIT ----------------------------------------------
    cache_cfg = settings_with(cache={"enabled": True, "audit_rate": 0.0, "ttl_s": 10_000.0})
    # p_judged and p_near differ in CONTENT (a constant luminance offset) but have an IDENTICAL
    # dHash, because dHash compares adjacent pixels and is therefore brightness-invariant. p_near
    # is inside the layer's near-match radius, so it is served p_judged's verdict.
    p_judged = bytes(((x * 7) % 100) for x in range(256))
    p_near = bytes((v + 120) for v in p_judged)
    p_quiet = payload(402)
    near_hamming = hamming(dhash(p_judged), dhash(p_near))
    c_script = {sha(p_judged): YES, sha(p_quiet): NO}
    # t=0 prime; t=1 p_judged judged YES -> FIRE; t=2 does not escalate (cache slot untouched) and
    # the edge is re-armed through the documented Gate.rearm seam (the post-correction path);
    # t=3 p_near -> served from CACHE -> FIRE whose verdict the provider never computed, and whose
    # newest evidence frame the judge never saw.
    c_src = [
        (0.0, STILL, p_quiet),
        (1.0, MOVED, p_judged),
        (2.0, MOVED, p_quiet),
        (3.0, STILL, p_near),
    ]
    c_fed = [(t, f) for (t, _g, f) in c_src]
    v_c = ScriptedVision(c_script)
    core_c = make_core(
        v_c, goal=Goal(id="zone", condition="c", say="Alert"), settings=cache_cfg
    )

    def _rearm(t, _f):
        if t == 2.0:
            core_c.gate.rearm(("zone", None))

    out_c = drive(core_c, c_src, on_frame=_rearm)
    audit_c = _audit_snippet_media(
        out_c.incidents, fed=c_fed, judge_seen=v_c.seen_hashes, label="cache-hit"
    )
    cache_hits = out_c.metrics.get("cache.hit", 0)
    r.check(
        "C/a cache-served FIRE was actually produced (the adversarial state was reached)",
        cache_hits > 0 and len(out_c.incidents) >= 2 and v_c.n_calls < out_c.stats.escalations,
        f"escalations={out_c.stats.escalations} provider_calls={v_c.n_calls} "
        f"cache.hit={cache_hits} incidents={len(out_c.incidents)}; the served frame differs in "
        f"content from the judged one but has dHash Hamming distance {near_hamming} "
        f"(tau_bits={cache_cfg.cache.tau_bits})",
        provider_calls=v_c.n_calls,
        cache_hits=cache_hits,
        near_match_hamming=near_hamming,
    )
    r.check(
        "C/evidence media after a cache hit is REAL captured media (no substitution)",
        audit_c["clean"],
        f"{audit_c['evidence_frames']} evidence frames, "
        f"{len(audit_c['frames_not_in_fed_stream'])} not in the fed stream, "
        f"{len(audit_c['not_a_contiguous_window_of_the_fed_stream'])} misaligned",
    )
    never_judged = audit_c["newest_frame_never_judged"]
    r.check(
        "C/a cache-served Incident's newest evidence frame IS a frame the judge saw",
        not never_judged,
        f"{len(never_judged)} incident(s) whose newest evidence frame the judge never saw — the "
        f"cache serves on a dHash NEAR-match (Hamming <= {cache_cfg.cache.tau_bits}), so the "
        f"verdict was computed on different pixels than the ones delivered as its evidence",
        offenders=never_judged,
    )
    cached_incident = out_c.incidents[-1] if len(out_c.incidents) >= 2 else None
    provenance = getattr(cached_incident, "provenance", None) if cached_incident else None
    r.check(
        "C/a cache-served Incident is distinguishable from a freshly-judged one",
        provenance not in (None, "observed"),
        f"the Incident delivered off a cache HIT carries provenance={provenance!r} — identical to "
        f"a freshly-judged fire; nothing on the wire Incident marks the verdict as re-served",
        provenance=provenance,
        verdict_evidence=None if cached_incident is None else cached_incident.verdict.evidence,
    )

    # ---- D. revised / superseded incidents --------------------------------------------------------
    all_incidents = out_a.incidents + out_b.incidents + out_c.incidents
    revisions = {incident_revision(i) for i in all_incidents} - {None}
    supersedes = [incident_supersedes(i) for i in all_incidents]
    dangling = [s for s in supersedes if s is not None and s not in revisions]
    r.check(
        "D/no Incident supersedes a revision that was never delivered",
        not dangling,
        f"{len(dangling)} dangling supersession link(s); revisions seen: {sorted(revisions)}",
        dangling=dangling,
    )
    # Injectivity is a PER-RUN property: the handle is derived from (goal_id, entity_id,
    # evidence_seq, occurrence_id), all of which are scoped to one runtime.
    injective = []
    for name, o in (("baseline", out_a), ("admission-gap", out_b), ("cache-hit", out_c)):
        jj = [i.judgment_event_id for i in o.incidents]
        injective.append(
            {
                "run": name,
                "incidents": len(o.incidents),
                "links": len([x for x in jj if x is not None]),
                "distinct": len({x for x in jj if x is not None}),
            }
        )
    r.check(
        "D/evidence LINKS are injective within a run (one judgment record per Incident)",
        all(x["links"] == x["incidents"] == x["distinct"] for x in injective),
        "; ".join(
            f"{x['run']}: {x['distinct']}/{x['links']} distinct over {x['incidents']} incidents"
            for x in injective
        ),
        per_run=injective,
    )

    # ---- E. the EvidenceStore resolver is genuinely tamper-evident --------------------------------
    store = EvidenceStore(policy=RetentionPolicy())
    first = out_a.incidents[0]
    art = store.put(
        frame_bytes(first.snippet.frames[-1]),
        kind="frame",
        media_type="application/octet-stream",
        capture_start_ts=first.snippet.t_start,
        capture_end_ts=first.snippet.t_end,
        retention_class="fire",
    )
    resolved = store.resolve(art.ref)
    round_trip_ok = (
        resolved.sha256 == sha(first.snippet.frames[-1])
        and store.open_bytes(art.ref) == frame_bytes(first.snippet.frames[-1])
    )
    store._entries[art.ref].data = b"tampered"  # noqa: SLF001 — deliberate tamper for the check
    tamper_detected = False
    try:
        store.open_bytes(art.ref)
    except EvidenceCorrupt:
        tamper_detected = True
    r.check(
        "E/evidence resolves by content hash and tampering is detected",
        round_trip_ok and tamper_detected,
        f"round-trip hash match={round_trip_ok}; EvidenceCorrupt raised on tamper={tamper_detected}",
    )

    # ---- F. the CLIP encoding path (snippet_kind='clip') ------------------------------------------
    # LAYER PROBE on the exact encoder the runtime calls (percept/incidents.py:203 ->
    # _encode_frame_bundle), plus an end-to-end run through percept.run.
    clip_frames = [payload(500 + i) for i in range(6)]
    clip_script = {sha(clip_frames[i]): (YES if i == 3 else NO) for i in range(6)}
    clip_fed = [(float(i), clip_frames[i]) for i in range(6)]
    v_f = ScriptedVision(clip_script)
    core_f = make_core(v_f, goal=Goal(id="zone", condition="c", say="Alert"))
    out_f = drive(core_f, alternating(clip_fed), snippet_kind="clip")

    clip_report = {
        "incidents": len(out_f.incidents),
        "entries": [],
        "base64_data": 0,
        "repr_only": 0,
        "byte_recoverable": 0,
    }
    for inc in out_f.incidents:
        bundle = json.loads(inc.snippet.media)
        for entry in bundle["frames"]:
            if "data" in entry:
                clip_report["base64_data"] += 1
                clip_report["byte_recoverable"] += 1
            else:
                clip_report["repr_only"] += 1
                rep = entry.get("repr", "")
                recoverable = False
                try:
                    import ast  # noqa: PLC0415

                    recoverable = isinstance(ast.literal_eval(rep), (bytes, bytearray))
                except Exception:  # noqa: BLE001
                    recoverable = False
                clip_report["byte_recoverable"] += int(recoverable)
                clip_report["entries"].append(
                    {
                        "keys": sorted(entry.keys()),
                        "truncated": "..." in rep,
                        "byte_recoverable": recoverable,
                    }
                )

    # The realistic frame type: a decoded ndarray-like frame, which is what the cv2 runtime source
    # (percept/runtime/source.py:219, ``cap.read()``) delivers.
    ndarray_probe = _ndarray_clip_probe()
    clip_frames_total = clip_report["base64_data"] + clip_report["repr_only"]

    r.check(
        "F1/clip-kind evidence for a bytes frame is byte-recoverable",
        clip_frames_total > 0 and clip_report["byte_recoverable"] == clip_frames_total,
        f"{clip_report['byte_recoverable']}/{clip_frames_total} clip entries recover the exact "
        f"media bytes ({clip_report['base64_data']} as base64 'data', "
        f"{clip_report['repr_only']} as a Python 'repr' string — recoverable, but NOT the "
        f"documented base64 media field a sink reads)",
        bytes_frames=clip_report,
    )
    r.check(
        "F2/clip-kind evidence for a DECODED frame is not silently truncated",
        not ndarray_probe.get("truncated", False),
        (
            f"decoded ndarray frame -> keys={ndarray_probe.get('keys')}, "
            f"TRUNCATED={ndarray_probe.get('truncated')} "
            f"({ndarray_probe.get('source_bytes')} pixel bytes replaced by a "
            f"{ndarray_probe.get('repr_len')}-char string, unrecoverable)"
        ),
        ndarray_frame=ndarray_probe,
    )

    if ndarray_probe["truncated"]:
        r.defects.append(
            Defect(
                id="SV-4.7-D1",
                severity="blocker",
                where="percept/incidents.py:165-179 (_encode_frame_bundle), reached from percept/incidents.py:203 (snippet_from_window kind='clip')",
                summary=(
                    "With snippet_kind='clip', any frame that is not a literal ``(mime, bytes)`` "
                    "2-tuple falls into the else branch and is serialized as ``{'repr': "
                    "repr(frame)}``. A decoded frame from the shipped camera source is a numpy "
                    "ndarray, whose repr is TRUNCATED with '...' above 1000 elements. The delivered "
                    "IncidentSnippet therefore advertises media_type "
                    "'application/x-percept-frame-bundle+json' while carrying a lossy, "
                    "irreversible ASCII summary in place of the pixels the judge saw. It is silent: "
                    "no health event, no error, no counter. This is SUBSTITUTED evidence."
                ),
                repro=(
                    "import numpy as np, json\n"
                    "from percept.incidents import _encode_frame_bundle\n"
                    "a = (np.arange(64*64*3, dtype=np.uint8) % 251).reshape(64, 64, 3)  # 12288 px bytes\n"
                    "mime, media = _encode_frame_bundle([a], t_start=0.0, t_end=1.0)\n"
                    "e = json.loads(media)['frames'][0]\n"
                    "assert list(e) == ['repr'] and '...' in e['repr']\n"
                    f"# repr length {ndarray_probe['repr_len']} chars for "
                    f"{ndarray_probe['source_bytes']} bytes of pixels — unrecoverable.\n"
                    "# End-to-end: percept.run(core, source=[(t, gate_frame, ndarray)], "
                    "snippet_kind='clip')\n"
                    "#             -> incident.snippet.media carries the same truncated reprs."
                ),
            )
        )

    if never_judged:
        r.defects.append(
            Defect(
                id="SV-4.7-D3",
                severity="high",
                where="percept/cache/framehash.py:218-232 (near-match serve) reached from percept/agent.py:5937 / percept/agent.py:2799",
                summary=(
                    "With the perception cache enabled, a FIRE can be delivered whose newest "
                    "evidence frame the judge NEVER saw. The layer serves on a dHash NEAR-match "
                    "(Hamming <= tau_bits, default 6) and dHash is brightness-invariant, so a frame "
                    "with different content but the same gradient structure is served the earlier "
                    "frame's verdict. The Incident then pairs a verdict computed on frame X with "
                    "evidence media ending at frame Y != X. The media itself is real captured "
                    "media (nothing is synthesized), but the verdict-to-evidence binding is broken "
                    "and nothing on the wire says so. Scope: PERCEPT_CACHE_ENABLED is OFF by "
                    "default; audit_rate (default 0.05) re-perceives only 5% of hits."
                ),
                repro=(
                    "p_judged = bytes(((x*7) % 100) for x in range(256))\n"
                    "p_near   = bytes(v + 120 for v in p_judged)   # different pixels\n"
                    f"hamming(dhash(p_judged), dhash(p_near)) == {near_hamming}  "
                    f"<= tau_bits={cache_cfg.cache.tau_bits}\n"
                    "cache enabled, audit_rate=0.0; judge p_judged at t=1 -> FIRE; re-arm at t=2; "
                    "feed p_near at t=3 -> cache HIT -> FIRE.\n"
                    f"# incident.snippet.frames[-1] == p_near, which is NOT in the set of frames "
                    f"the judge was shown ({len(never_judged)} such incident(s))."
                ),
            )
        )

    if provenance in (None, "observed"):
        r.defects.append(
            Defect(
                id="SV-4.7-D2",
                severity="medium",
                where="percept/incidents.py:99 (provenance read off the fire) / percept/cache/framehash.py:226-232 (CacheResult provenance='cached')",
                summary=(
                    "A FIRE whose verdict was served from the perception cache is delivered as an "
                    "Incident with provenance='observed' — byte-identical to a freshly-judged fire. "
                    "The cache layer's own CacheResult(provenance='cached') never reaches the wire "
                    "Incident. The evidence MEDIA is real, but a consumer cannot tell that the "
                    "verdict was computed on an earlier (only dHash-near, not necessarily "
                    "identical) frame."
                ),
                repro=(
                    "cache enabled (audit_rate=0.0); judge frame pA at t=1 -> FIRE (provider call); "
                    "gate re-armed via the documented Gate.rearm seam at t=2; pA again at t=3 -> "
                    "cache HIT -> FIRE. incident.provenance == 'observed', "
                    "incident.verdict.evidence == the t=1 verdict's text. cache.hit="
                    f"{cache_hits}, provider_calls={v_c.n_calls} over "
                    f"{out_c.stats.escalations} escalations."
                ),
            )
        )

    r.measurements = {
        "baseline": audit_a,
        "admission_gap": audit_b,
        "cache_hit": audit_c,
        "cache_hit_provenance": provenance,
        "clip_bytes_frames": clip_report,
        "clip_ndarray_probe": ndarray_probe,
        "evidence_store_round_trip": round_trip_ok,
        "evidence_store_tamper_detected": tamper_detected,
        "judgment_link_injectivity": injective,
        "cache_near_match_hamming": near_hamming,
    }
    return r


def _ndarray_clip_probe() -> dict:
    """Encode a realistic decoded frame through the runtime's own clip encoder."""
    try:
        import numpy as np  # noqa: PLC0415 — optional; the shipped camera source requires it
    except ImportError:  # pragma: no cover
        return {"skipped": "numpy not installed", "truncated": False, "repr_only": 0}
    arr = (np.arange(64 * 64 * 3, dtype=np.uint8) % 251).reshape(64, 64, 3)
    mime, media = _encode_frame_bundle([arr], t_start=0.0, t_end=1.0)
    entry = json.loads(media)["frames"][0]
    rep = entry.get("repr", "")
    return {
        "mime": mime,
        "keys": sorted(entry.keys()),
        "repr_only": int("repr" in entry),
        "truncated": "..." in rep,
        "repr_len": len(rep),
        "source_bytes": int(arr.nbytes),
        "media_bytes": len(media),
    }


# ---------------------------------------------------------------------------------------------
# SV-4.8 — drop-rate reconciliation
# ---------------------------------------------------------------------------------------------


def _ledger(out: RunOutput, *, provider_calls: int) -> dict:
    s = out.stats
    m = out.metrics
    gate_rows = [e for e in out.trace if e.stage == "gate.classified"]
    fire_rows = [e for e in gate_rows if e.data.get("action") == "fire"]
    ask_rows = [e for e in gate_rows if e.data.get("action") == "ask"]
    silent = [e for e in gate_rows if e.data.get("action") == "silent"]
    silent_with_reason = [e for e in silent if e.data.get("gate_reason") in GATE_REASONS]
    drop_stats = {
        "judge_superseded": s.judge_superseded,
        "judge_revision_dropped": s.judge_revision_dropped,
        "judge_deadline_dropped": s.judge_deadline_dropped,
        "judge_deadline_expired": s.judge_deadline_expired,
        "judge_source_discontinuity_dropped": s.judge_source_discontinuity_dropped,
    }
    total_judge_drops = sum(drop_stats.values())
    drop_health = [
        h for h in out.health if h.kind in ("judge_superseded",) or h.kind.endswith("_dropped")
    ]
    return {
        "source": {
            "observed": s.source_observed,
            "accepted": s.frames,
            "excluded": s.source_excluded,
            "queue_dropped": s.frames_dropped,
            "balances": s.source_observed == s.frames + s.source_excluded,
        },
        "admission": {
            "frames_in": s.frames,
            "admitted": s.escalations,
            "not_admitted": s.frames - s.escalations,
            "signal_admits": s.signal_admits,
            "fixed_admits": s.fixed_admits,
            "coast_admits": s.coast_admits,
            "audio_wakes": s.audio_wakes,
            "partition_balances": s.escalations
            == s.signal_admits + s.fixed_admits + s.coast_admits + s.audio_wakes,
            "funnel_seen": m.get("funnel.seen", 0),
            "funnel_gated": m.get("funnel.gated", 0),
            "funnel_agrees": m.get("funnel.seen", 0) == s.frames
            and m.get("funnel.gated", 0) == s.escalations,
        },
        "judge_dispatch": {
            "admitted": s.escalations,
            "judge_calls": s.judge_calls,
            **drop_stats,
            "total_drops": total_judge_drops,
            "unaccounted": s.escalations - (s.judge_calls + total_judge_drops),
            "balances": s.escalations == s.judge_calls + total_judge_drops,
            "drop_receipts": len(drop_health),
            "every_drop_has_a_receipt": len(drop_health) == total_judge_drops,
            "funnel_dropped": m.get("funnel.dropped", 0),
            "funnel_dropped_agrees": m.get("funnel.dropped", 0) == total_judge_drops,
        },
        "judge_execution": {
            "judge_calls": s.judge_calls,
            "provider_calls": provider_calls,
            "cache_hits": m.get("cache.hit", 0),
            "funnel_judged": m.get("funnel.judged", 0),
            "balances": s.judge_calls == provider_calls + m.get("cache.hit", 0),
        },
        "gate": {
            # The PRIMARY, per-unit ledger: one gate.classified receipt per judged unit.
            "judged": m.get("funnel.judged", 0),
            "decisions": len(gate_rows),
            "decision_fire": len(fire_rows),
            "decision_ask": len(ask_rows),
            "decision_silent": len(silent),
            "silent_with_typed_reason": len(silent_with_reason),
            "silent_reasons": _counts(e.data.get("gate_reason") for e in silent),
            "unaccounted": m.get("funnel.judged", 0) - len(gate_rows),
            "balances": (
                m.get("funnel.judged", 0) == len(gate_rows)
                and len(gate_rows) == len(fire_rows) + len(ask_rows) + len(silent)
            ),
            "every_silent_has_a_reason": len(silent) == len(silent_with_reason),
            # The MIRROR ledger (the funnel counters a dashboard reads).
            "funnel_fired": m.get("funnel.fired", 0),
            "funnel_refused": m.get("funnel.refused", 0),
            "delivered_asks": s.asks,
            "funnel_fired_agrees": m.get("funnel.fired", 0) == len(fire_rows),
            "funnel_refused_agrees": m.get("funnel.refused", 0) == len(ask_rows),
            "funnel_mirror_balances": m.get("funnel.judged", 0)
            == m.get("funnel.fired", 0) + m.get("funnel.refused", 0) + len(silent),
        },
        "delivery": {
            "fires": s.fires,
            "asks": s.asks,
            "incidents": s.incidents,
            "yielded": len(out.incidents),
            "sink_errors": s.sink_errors,
            "balances": s.fires == s.incidents + s.sink_errors and len(out.incidents) == s.incidents,
        },
    }


def _counts(it) -> dict:
    out: dict = {}
    for x in it:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def sv4_8_drop_reconciliation() -> TestResult:
    r = TestResult("SV-4.8", "Drop-rate reconciliation (units in == units accounted, every boundary)")

    scenarios: dict[str, dict] = {}

    # -- 1. clean run, including an ASK (low-confidence refusal) ---------------------------------
    n = 14
    frames = [payload(600 + i) for i in range(n)]
    script = {}
    for i in range(n):
        script[sha(frames[i])] = UNSURE if i == 5 else (YES if i % 2 else NO)
    v1 = ScriptedVision(script)
    core1 = make_core(v1, goal=Goal(id="zone", condition="c", say="Alert"))
    out1 = drive(core1, alternating([(float(i), frames[i]) for i in range(n)]))
    scenarios["clean"] = _ledger(out1, provider_calls=v1.n_calls)

    # -- 2. overload: a slow judge under a fast source forces supersession drops -------------------
    m = 40
    of = [payload(700 + i) for i in range(m)]
    oscript = {sha(of[i]): (YES if i % 2 else NO) for i in range(m)}
    v2 = ScriptedVision(oscript, delay_s=0.02)
    core2 = make_core(v2, goal=Goal(id="zone", condition="c", say="Alert"))

    async def fast_source():
        for i in range(m):
            yield (float(i) * 0.25, (MOVED if i % 2 else STILL), of[i])
            await asyncio.sleep(0)

    out2 = drive(core2, fast_source(), fps=100.0)
    scenarios["overload"] = _ledger(out2, provider_calls=v2.n_calls)

    # -- 3. cache-enabled run: judge_calls splits into provider calls + cache hits -----------------
    p = payload(800)
    cscript = {sha(p): NO}
    v3 = ScriptedVision(cscript)
    core3 = make_core(
        v3,
        goal=Goal(id="zone", condition="c", say="Alert"),
        settings=settings_with(cache={"enabled": True, "audit_rate": 0.0, "ttl_s": 10_000.0}),
    )
    out3 = drive(core3, alternating([(float(i), p) for i in range(8)]))
    scenarios["cache"] = _ledger(out3, provider_calls=v3.n_calls)

    # -- assertions across every scenario ------------------------------------------------------------
    def all_true(path: list[str]) -> tuple[bool, str]:
        bad = []
        for name, led in scenarios.items():
            node = led
            for k in path[:-1]:
                node = node[k]
            if not node[path[-1]]:
                bad.append(f"{name}:{'.'.join(path)}")
        return (not bad), ("; ".join(bad) if bad else "balances in every scenario")

    for label, path in [
        ("source boundary: observed == accepted + excluded", ["source", "balances"]),
        (
            "admission boundary: escalations partition into signal/fixed/coast/audio",
            ["admission", "partition_balances"],
        ),
        ("admission boundary: funnel.seen/gated agree with RuntimeStats", ["admission", "funnel_agrees"]),
        (
            "judge-dispatch boundary: admitted == judge_calls + all named drops (zero unaccounted)",
            ["judge_dispatch", "balances"],
        ),
        ("every dropped unit carries a health receipt", ["judge_dispatch", "every_drop_has_a_receipt"]),
        (
            "judge-execution boundary: judge_calls == provider calls + cache hits",
            ["judge_execution", "balances"],
        ),
        (
            "gate boundary: judged == one typed decision receipt per unit (fire + ask + silent)",
            ["gate", "balances"],
        ),
        ("every SILENT decision carries a typed reason", ["gate", "every_silent_has_a_reason"]),
        ("delivery boundary: fires == incidents + sink errors == yielded", ["delivery", "balances"]),
        (
            "funnel.dropped mirrors the RuntimeStats drop counters",
            ["judge_dispatch", "funnel_dropped_agrees"],
        ),
        ("funnel.refused mirrors the gate's ASK decisions", ["gate", "funnel_refused_agrees"]),
        (
            "funnel mirror balances: judged == funnel.fired + funnel.refused + silent",
            ["gate", "funnel_mirror_balances"],
        ),
    ]:
        ok, detail = all_true(path)
        r.check(label, ok, detail)

    unaccounted = {k: v["judge_dispatch"]["unaccounted"] for k, v in scenarios.items()}
    gate_unaccounted = {k: v["gate"]["unaccounted"] for k, v in scenarios.items()}
    r.check(
        "ZERO unaccounted units at the judge-dispatch boundary",
        all(x == 0 for x in unaccounted.values()),
        f"unaccounted per scenario: {unaccounted}",
        unaccounted=unaccounted,
    )
    r.check(
        "ZERO unaccounted units at the gate boundary",
        all(x == 0 for x in gate_unaccounted.values()),
        f"unaccounted per scenario: {gate_unaccounted}",
        unaccounted=gate_unaccounted,
    )

    clean_gate = scenarios["clean"]["gate"]
    if not clean_gate["funnel_refused_agrees"]:
        r.defects.append(
            Defect(
                id="SV-4.8-D2",
                severity="high",
                where="percept/agent.py:3777 and percept/agent.py:3828 (both bump Funnel.refused for the SAME judged unit)",
                summary=(
                    "One ASK is counted twice in funnel.refused. agent.py:3777 bumps it when "
                    "``classify`` returns 'unknown'; agent.py:3828 bumps it again when the executor "
                    "emits the resulting non-FIRE event for the same unit. The funnel identity "
                    "``judged == fired + refused + silent`` therefore cannot hold on any run "
                    "containing an ASK — the funnel over-reports refusals by exactly the number of "
                    "delivered ASKs, which is also the refusal rate a deployment tunes fresh_min / "
                    "the confidence threshold against."
                ),
                repro=(
                    "One scripted low-confidence verdict (satisfied=True, confidence=0.20) in a "
                    "13-unit run:\n"
                    f"  gate.classified ASK decisions = {clean_gate['decision_ask']}\n"
                    f"  stats.asks (delivered)        = {clean_gate['delivered_asks']}\n"
                    f"  funnel.refused                = {clean_gate['funnel_refused']}   <- 2x\n"
                    f"  funnel.judged={clean_gate['judged']} vs funnel.fired="
                    f"{clean_gate['funnel_fired']} + funnel.refused="
                    f"{clean_gate['funnel_refused']} + silent={clean_gate['decision_silent']} = "
                    f"{clean_gate['funnel_fired'] + clean_gate['funnel_refused'] + clean_gate['decision_silent']}"
                ),
            )
        )

    overload = scenarios["overload"]["judge_dispatch"]
    if not overload["funnel_dropped_agrees"]:
        r.defects.append(
            Defect(
                id="SV-4.8-D1",
                severity="high",
                where="percept/runtime/worker.py:1417 (class _JudgeScheduler — no `funnel` reference anywhere in 1417-2239), vs percept/runtime/worker.py:793 in _LegacyJudgeScheduler",
                summary=(
                    "The visual judge scheduler used on the DEFAULT judge_dispatch='async' path "
                    "(_ProviderScheduler = _JudgeScheduler, instantiated at "
                    "percept/runtime/worker.py:3616) bumps RuntimeStats and appends a trace tuple "
                    "for every drop, but never calls Funnel.dropped(). Its predecessor "
                    "_LegacyJudgeScheduler does (worker.py:793, :838, :1172, :1196, :1230, :1303). "
                    "Result: funnel.dropped reads 0 while the run genuinely dropped "
                    f"{overload['total_drops']} admitted units. Two ledgers that must agree do not, "
                    "and the one that goes quiet is the funnel — the single definition percept "
                    "doctor / percept eval / the visualizer derive their numbers from."
                ),
                repro=(
                    "core = Percept.create(vision=<judge with a 20 ms delay>, discover_plugins=False)\n"
                    "async for _ in percept.run(core, source=<40 frames, alternating motion>, "
                    "stats=stats, fps=100.0): pass\n"
                    f"# stats.judge_superseded == {overload['judge_superseded']}\n"
                    f"# core.metrics.get('funnel.dropped') == {overload['funnel_dropped']}  "
                    "<- should equal the drop total\n"
                    f"# on_health DID receive {overload['drop_receipts']} judge_superseded events, "
                    "so the drop is not fully silent — only the funnel counter is."
                ),
            )
        )

    r.measurements = {"ledgers": scenarios}
    return r


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

TESTS: list[tuple[str, Callable[[], TestResult]]] = [
    ("SV-4.1", sv4_1_gate_decision_and_state),
    ("SV-4.5", sv4_5_cache_correctness),
    ("SV-4.6", sv4_6_media_buffers),
    ("SV-4.7", sv4_7_evidence_integrity),
    ("SV-4.8", sv4_8_drop_reconciliation),
]


def runtime_identity() -> dict:
    pkg = Path(percept.__file__).resolve().parent
    return {
        "percept_version": percept.__version__,
        "percept_path": str(pkg),
        "is_installed_wheel": "site-packages" in pkg.parts,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
    }


def assert_runtime_under_test() -> None:
    """Refuse to validate anything but an INSTALLED percept wheel.

    A repo checkout on ``sys.path`` would silently validate source that is not what ships. This
    is a hard stop, not a warning — a green result against the wrong tree is worse than no result.
    """
    ident = runtime_identity()
    if not ident["is_installed_wheel"]:
        raise SystemExit(
            "REFUSING TO RUN: percept resolved to a source tree, not an installed wheel:\n"
            f"  {ident['percept_path']}\n"
            "Run with the launch wheel's interpreter, e.g.\n"
            "  /private/tmp/percept-launch/oss-final-venv-bba572a/bin/python "
            "eval/stage-validation/sv4_cognition.py"
        )


def run_all(only: list[str] | None = None) -> dict:
    results: list[TestResult] = []
    for tid, fn in TESTS:
        if only and tid not in only:
            continue
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 — a crashed test is a reported ERROR, never a silent skip
            import traceback

            res = TestResult(tid, tid)
            res.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            results.append(res)
    blockers = [d.to_json() for r_ in results for d in r_.defects if d.severity == "blocker"]
    return {
        "suite": "SV-4 COGNITION + DISPOSITION",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime": runtime_identity(),
        "results": [r_.to_json() for r_ in results],
        "summary": {
            r_.id: r_.status for r_ in results
        },
        "release_blockers": blockers,
    }


def _print_table(report: dict) -> None:
    print()
    print("=" * 92)
    print("  SV-4 COGNITION + DISPOSITION — validation results")
    print(f"  runtime: percept {report['runtime']['percept_version']} @ {report['runtime']['percept_path']}")
    print("=" * 92)
    for res in report["results"]:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR "}[res["status"]]
        n_pass = sum(1 for c in res["checks"] if c["passed"])
        print(f"\n[{mark}] {res['id']}  {res['title']}")
        print(f"        {n_pass}/{len(res['checks'])} checks passed")
        for c in res["checks"]:
            flag = "  ok  " if c["passed"] else " FAIL "
            print(f"   [{flag}] {c['name']}")
            if c["detail"]:
                print(f"            {c['detail']}")
        if res["error"]:
            print(f"   ERROR: {res['error'].splitlines()[0]}")
        for d in res["defects"]:
            print(f"   >> DEFECT {d['id']} [{d['severity']}] {d['where']}")
    print()
    print("-" * 92)
    for tid, status in report["summary"].items():
        print(f"  {tid:<8} {status}")
    if report["release_blockers"]:
        print()
        print("  !! RELEASE BLOCKERS !!")
        for d in report["release_blockers"]:
            print(f"     {d['id']}: {d['summary'][:150]}")
    print("-" * 92)
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SV-4 cognition + disposition validation")
    ap.add_argument("--only", nargs="*", help="run only these test ids (e.g. SV-4.7)")
    ap.add_argument("--json", default=str(JSON_PATH), help="machine-readable output path")
    args = ap.parse_args(argv)

    assert_runtime_under_test()
    report = run_all(only=args.only)
    _print_table(report)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False))
    print(f"  wrote {out}")

    failed = [k for k, v in report["summary"].items() if v != "PASS"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
