#!/usr/bin/env python3
"""Percept demo harness — run the REAL runtime behind the two demo pages.

This is an INTERNAL product-test harness (dogfood the wheel before a client wheel ships), not a
shipped API. It wraps the frozen ``watch()``/``run()`` seam — the exact path ``percept.watch_cli
--trace`` uses — so the browser pages render REAL WatchEvents and a REAL ``.ptrace``, never mocked data.

    python docs/demo/serve.py                 # http://localhost:8788  (keyless: bundled fixture, real runtime)
    PERCEPT_VISION_MODEL=... OPENROUTER_API_KEY=...  python docs/demo/serve.py   # real judge over uploads

Endpoints
    GET  /                         -> demo index (links to the demo pages)
    GET  /demo-01-video-watch.html -> the MP4 upload demo (real client)
    GET  /demo-02-screen-watch.html-> the screen-share demo (real client)
    GET  /demo-04-stream-uploaded-video.html
                                   -> the live-stream demo (watch the run happen, then replay it)
    POST /run?goal=..&backend=..&fps=..&source_name=..   body = raw video bytes
                                   -> runs the REAL runtime over the upload, returns the real trace JSON
    POST /run?mode=fixture         -> runs the REAL runtime over the bundled fixture (keyless, no spend)
    POST /run/stream?<same params> -> the SAME run, streamed as NDJSON while it happens: one line per
                                   trace record and per delivered Incident, then a final {"type":"done"}
                                   line carrying the exact payload POST /run would have returned
    POST /run/stream?source_url=rtsp://localhost:8554/traffic&max_s=90
                                   -> [demo 04 source 1] watch a LIVE feed in place, no upload
    POST /run/stream?restream=1    body = raw video bytes
                                   -> [demo 04 source 2] republish the upload to RTSP with ffmpeg -re
                                   and watch THAT, so a file behaves exactly like the live feed
    POST /stop/<run_id>            -> end a live run early (it still writes its .ptrace)
    GET  /source-check?url=…       -> can the runtime open this live URI right now? {ok, reason}
    GET  /trace/<run_id>.ptrace    -> download the real .ptrace written for that run
    GET  /frame/<run_id>/<seq>.jpg -> the JPEG for that source.frame record — re-seeked from the
                                   upload on a file run, kept from the runtime's own on_frame tee on
                                   a live run (404 + reason on a fixture run)

No judge key present -> uploads are refused with a clear message and the fixture path is offered, so the
harness is honest about when it is and is not exercising a real model.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PERCEPT_DEMO_PORT", "8788"))
RUNS: dict[str, dict] = {}  # run_id -> {"trace": <path>, "json": <dict>}

# --- make percept-vision importable whether or not it is pip-installed -------------------------
_PKG = HERE.parent.parent / "packages" / "percept-vision"
if _PKG.exists() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _load_dotenv() -> None:
    """Best-effort: pull only API KEYS from a repo .env (same rule as percept.watch_cli) — never the
    PERCEPT_* config, so a stale PERCEPT_VISION_MODEL cannot hijack the UI-chosen provider. Values are
    loaded into this process only; they are never printed or sent to the browser."""
    for env in (Path.cwd() / ".env", _PKG / ".env", _PKG.parent.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = (s.strip() for s in line.split("=", 1))
                    if k.endswith("_API_KEY") or k.endswith("_KEY"):
                        os.environ.setdefault(k, v)


def _have_key() -> str | None:
    # names this repo's .env actually uses: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, GOOGLE_AISTUDIO_API_KEY
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(k):
            return k
    return None


_DEFAULT_MODEL = {  # same per-provider defaults percept.watch_cli forces (vision seam only — kept
    # for the fixture/reporting paths that only ever needed the vision model; see _SEAM_MODEL_MAP /
    # _seam_plan() below for the FULL per-seam routing a real run now uses).
    "openrouter": "qwen/qwen3-vl-32b-instruct",
    "gemini": "gemini-3.1-flash-lite",
    "claude": "claude-sonnet-5",
}

# [DEMO4 Finding 2] the UI's single "judge backend" choice used to route ONLY the vision seam
# (PERCEPT_VISION_MODEL + Percept.configure(vision=backend)); the conductor/specialist seams
# (percept/config.py ModelSettings.conductor_model/specialist_model) silently kept their
# claude-haiku-4-5 field defaults regardless of the chosen provider, which is a lie the Resolved
# configuration panel exposed the moment someone read it closely. This is the per-seam model map,
# read against percept/registry.py's actual registrations (checked, not guessed):
#   vision:      fake, fixture, claude, gemini, gemini-live, openrouter
#   specialist:  fake, claude, gemini            <- NO openrouter adapter
#   conductor:   fake, claude, gemini, scripted-conductor   <- NO openrouter adapter
# So "openrouter" (the vision judge these demos default to) has no real conductor/specialist
# backend at all; see _seam_plan() for the honest, visible fallback.
_SEAM_MODEL_MAP: dict[str, dict[str, str]] = {
    "claude": {
        "vision": "claude-sonnet-5", "conductor": "claude-haiku-4-5", "specialist": "claude-haiku-4-5",
    },
    "gemini": {
        # conductor/specialist reuse GEMINI_DEFAULT (percept/backends/gemini.py) — the same cheap/
        # fast flash model gemini.py itself falls back to when a Settings tree carries no Gemini
        # model id for that seam.
        "vision": "gemini-3.1-flash-lite", "conductor": "gemini-3.5-flash", "specialist": "gemini-3.5-flash",
    },
    "openrouter": {
        "vision": "qwen/qwen3-vl-32b-instruct",  # conductor/specialist: no adapter — see _seam_plan()
    },
}
_SEAM_FALLBACK_PROVIDER = "claude"  # the always-shipped real text backend a vision-only provider falls back to


def _seam_plan(backend: str) -> dict[str, dict]:
    """Route ``backend`` (the UI's single judge-backend choice) to ALL THREE model-bearing seams —
    vision, conductor, specialist — honestly.

    A provider with no registered conductor/specialist adapter (``percept/registry.py``; today only
    "openrouter" ships vision-only) does NOT silently inherit whatever ``ModelSettings()`` defaults
    to (that was Finding 2's bug). Instead it falls back to :data:`_SEAM_FALLBACK_PROVIDER`
    ("claude") — but ONLY when that fallback's own credential is actually present in this process;
    forcing a live Anthropic client into existence with no ANTHROPIC_API_KEY would turn a working
    openrouter-only run into a crash, which is a worse dishonesty than a visible fallback. When even
    the fallback's credential is missing, this degrades one step further to the SCRIPTED double for
    that seam only (vision — the seam that matters for these demos — is never touched by this
    degradation path; the UI/goal authoring still runs, `add_goal`-style, on a scripted conductor).

    Returns ``{seam: {"backend", "model", "fallback_from", "reason"}}`` — ``fallback_from``/
    ``reason`` are ``None`` on a direct, adapter-backed route; both are always human-readable
    strings on a fallback, never silently blank, so the UI can render exactly what happened.
    """
    from percept.config import missing_credential

    claude_has_key = missing_credential("claude") is None
    plan: dict[str, dict] = {}
    for seam in ("vision", "conductor", "specialist"):
        direct = _SEAM_MODEL_MAP.get(backend, {}).get(seam)
        if direct is not None:
            plan[seam] = {"backend": backend, "model": direct, "fallback_from": None, "reason": None}
            continue
        if claude_has_key:
            plan[seam] = {
                "backend": _SEAM_FALLBACK_PROVIDER,
                "model": _SEAM_MODEL_MAP[_SEAM_FALLBACK_PROVIDER][seam],
                "fallback_from": backend,
                "reason": f"no {seam} adapter registered for backend={backend!r} (percept.registry) "
                          f"— falling back to {_SEAM_FALLBACK_PROVIDER!r}.",
            }
        else:
            scripted = "scripted-conductor" if seam == "conductor" else "scripted"
            plan[seam] = {
                "backend": scripted,
                "model": None,
                "fallback_from": backend,
                "reason": f"no {seam} adapter for backend={backend!r}, and no "
                          f"{_SEAM_FALLBACK_PROVIDER!r} credential to fall back to either — "
                          f"degrading to the SCRIPTED {seam} so this run does not crash "
                          f"(vision stays real; only {seam} is scripted).",
            }
    return plan

# [DEMO3] ADR-008 configuration contract: demo-01 (physical video) and demo-02 (screen watch) each
# name their SourceProfile EXPLICITLY through Percept.configure()/ConfigResolver — see
# _resolve_watch() below. This replaces the old PERCEPT_SALIENCE_STAT / PERCEPT_BENCH_BLOCKMAX_THRESHOLD
# / PERCEPT_GATE_REFRACTORY_S / PERCEPT_EGO_COMPENSATE env-var plumbing that used to live here: ZERO
# of those env vars are exported by this harness now. The client HTML pages pass ?profile=physical_scene
# or ?profile=screen_document on every /run call so the server always knows which explicit profile to
# resolve — see the "Resolved configuration" panel on both pages for the per-field provenance this
# produces.
_PROFILE_NAMES = ("physical_scene", "screen_document")


def _profile_for(name: str):
    """The real, shipped SourceProfile object for ``name`` (never a string re-interpreted downstream)
    — falls back to physical_scene, the SDK's calibrated default, for an unknown/missing name."""
    from percept.source_profiles_default import PHYSICAL_SCENE, SCREEN_DOCUMENT

    return {"physical_scene": PHYSICAL_SCENE, "screen_document": SCREEN_DOCUMENT}.get(name, PHYSICAL_SCENE)


def _execution_policy_for(name: "str | None"):
    """[DEMO4 Finding 3c] resolve the advanced panel's ``execution_policy`` select ("balanced" /
    "burst" / "low_cost" / "privacy_edge") to the real ``percept.execution_policy.ExecutionPolicy``
    object — or ``None`` for "balanced"/unset, which is a documented no-op (see
    ``percept/execution_policies_default.py``: ``BALANCED_V1`` contributes nothing to the precedence
    chain, so omitting it entirely is byte-for-byte identical). An unrecognized name degrades to
    ``None`` rather than raising — a stray/typo'd query param must not 500 the demo."""
    if not name or name == "balanced":
        return None
    from percept.execution_policy import get_execution_policy

    try:
        return get_execution_policy(f"{name}@1")
    except Exception:
        return None


def _resolve_watch(profile_name: str, goal_text: str, *, source_key: str, settings=None,
                    execution_policy=None, explicit: "dict | None" = None):
    """Resolve the explicit SourceProfile through the REAL percept.resolver.ConfigResolver — the exact
    seam ``Percept.configure()``/``apply_watch_plan()`` use, so the founder sees the SAME provenance the
    runtime armed with, not a hand-summarized approximation.

    ``settings`` — an EXPLICIT ``percept.config.Settings`` object; defaults to the 15s re-fire
    refractory-only Settings this harness has always built when the caller passes none (block_max@0.20
    + the coast floor + content_key identity all come from the profile itself). [DEMO4] a real run now
    passes a Settings tree whose ``models`` field carries the per-seam routing plan (see
    ``_seam_plan()``), so THIS SAME resolve — not a second copy — is what the "Resolved configuration"
    panel's per-field provenance reflects.

    ``explicit`` — [DEMO4 Finding 3b] ``WatchPlan.explicit`` overrides, e.g.
    ``{"Goal.state.identity_policy": "content_key"}`` — the advanced panel's identity-policy /
    episode-tracking authoring, applied through the SAME ``WatchPlan``/``apply_watch_plan`` seam
    ``test_c2f_source_profile_defaults.py`` proves an explicit concern-level choice wins over a
    resolved SourceProfile's own default.

    Returns ``(resolved: ResolvedWatch, profile: SourceProfile, settings: Settings, plan: WatchPlan)``
    — the SAME ``plan`` object a real run then hands to ``apply_watch_plan`` (one source of truth, not
    a second hand-built copy).
    """
    from percept.config import GateSettings, Settings
    from percept.contracts.watchplan import WatchPlan
    from percept.resolver import ConfigResolver

    profile = _profile_for(profile_name)
    if settings is None:
        settings = Settings(gate=GateSettings(refractory_s=15.0))
    plan = WatchPlan(
        condition=goal_text or "percept demo — deployment/profile resolution only",
        source=source_key, explicit=explicit or {},
    )
    resolved = ConfigResolver().resolve(
        plan, source_profile=profile, source_profile_explicit=True, settings=settings,
        execution_policy=execution_policy,
    )
    return resolved, profile, settings, plan


def _json_safe(value):
    """Recursively coerce a value into something ``json.dumps`` accepts — SourceProfile snapshot
    fields carry real ``frozenset``/``set``/``tuple`` values (e.g. ``SourceProfile.modalities``) that
    are not JSON types by themselves. Sets become sorted lists (deterministic wire output); anything
    else json.dumps still can't handle falls back to ``str()`` rather than crashing the /run response."""
    if isinstance(value, (frozenset, set)):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolved_explain_json(resolved) -> "dict | None":
    """Structured per-field provenance ({path, value, owner_layer, source_id, reason, validated_by})
    PLUS the human ``.explain()`` text — the founder sees both the table and the "for a human at 2am"
    rendering ResolvedWatch itself ships."""
    if resolved is None:
        return None
    fields = [
        dict(path=p, **{k: _json_safe(v) for k, v in resolved.provenance[p].to_dict().items()})
        for p in sorted(resolved.provenance)
    ]
    return {
        "text": resolved.explain(),
        "fields": fields,
        "plan_hash": resolved.plan_hash,
        "source_profile": _json_safe(dict(resolved.source_profile)) if resolved.source_profile is not None else None,
    }


def _receipt_json(receipt) -> dict:
    """The WatchReceipt summary — or, for the fixture path (goals come pre-armed from the bundled
    fixture script, never through apply_watch_plan — rearming them would risk changing what the
    fixture replays, which this card forbids), an honest 'legacy' stand-in that says so."""
    if receipt is None:
        return {
            "status": "legacy", "watch_ids": [], "plan_revision": None,
            "accepted_fields": [], "dropped_fields": [], "coerced_fields": [], "warnings": [],
            "summary": "legacy fixture path — goals came pre-armed from the bundled fixture, never "
                       "through apply_watch_plan (rearming them could change what the fixture replays).",
        }
    d = receipt.to_dict()
    d["summary"] = (f"{d['status']} · {len(d['watch_ids'])} goal(s) armed · "
                     f"{len(d['warnings'])} warning(s) · {len(d['dropped_fields'])} dropped · "
                     f"{len(d['coerced_fields'])} coerced")
    return d


def _apply_good_defaults() -> None:
    """[DEMO3]: the salience/motion-gate/refractory/ego-compensate knobs this used to export as
    PERCEPT_* env vars now ride the explicit SourceProfile + Settings objects _resolve_watch() builds
    per-run — ZERO of those env vars are set here anymore. Only PERCEPT_NO_BANNER remains (output
    formatting, unrelated to admission tuning)."""
    os.environ.setdefault("PERCEPT_NO_BANNER", "1")


_load_dotenv()          # load keys from .env before anything reads them
_apply_good_defaults()  # PERCEPT_NO_BANNER only — see docstring


def _recorder(core, goals, *, source: str, driver: str, resolved=None, on_record=None):
    """Same recorder the real CLI builds for --trace: full retention + the live Gate's knobs.

    ``resolved`` — the run's ``ResolvedWatch`` (or ``None``), forwarded straight to
    ``RunTraceManifest.create`` so the manifest's own ``source_profile_id``/``profiles``/
    ``component_snapshot_hash``/``optimization_lock_hash`` fields are filled from the SAME resolution
    the founder sees in the "Resolved configuration" panel — one source of truth, not a second copy.

    ``on_record`` — [DEMO 04] ``TraceRecorder``'s own per-record sink, used by ``POST /run/stream`` to
    tee each record onto the wire AS the runtime emits it. It is the recorder's shipped hook, not a
    demo-side fork of the ingest path: the ``.ptrace`` this run writes is byte-for-byte what it would
    have been without a listener, and TraceRecorder.record() already swallows sink exceptions so a
    disconnected browser can never break capture.
    """
    from percept.replay import gate_config_of
    from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

    manifest = RunTraceManifest.create(
        source=source, driver=driver, goals=goals,
        policy=TracePolicy(level="full", max_duration_s=None),
        gate_config=gate_config_of(core.gate),
        resolved=resolved,
    )
    return TraceRecorder(manifest, on_record=on_record)


def _record_sink(sink):
    """Adapt a ``/run/stream`` line-sink into a :class:`TraceRecorder` ``on_record`` callback."""
    if sink is None:
        return None

    def _cb(rec) -> None:
        sink({"type": "record", "record": _json_safe(rec.to_dict())})

    return _cb


async def _run_fixture(goal_text: str, profile_name: str, trace_path: Path, sink=None) -> dict:
    """Keyless: replay the bundled fixture through the REAL watch() seam. No spend, no pretence.

    [DEMO3]: goals still come straight off the bundled fixture (``offline_goals`` — unchanged, legacy)
    because the fixture's scripted verdicts are keyed to those EXACT goal/frame tokens; routing them
    through ``apply_watch_plan`` first would risk re-minting a different goal id or state and silently
    changing what gets replayed, which this card forbids. The ``_resolve_watch`` call below is
    side-effect-free reporting only — it never touches ``core``/``goals`` — so the founder still sees
    which profile a LIVE run of this same page would resolve to, without altering the fixture at all
    (the raw ``watch()`` seam the fixture runs over never consults tier-0 salience/gate config anyway).
    """
    from percept.offline import load_fixture, offline_core, offline_frames, offline_goals
    from percept.watch import watch

    fixture = load_fixture(None)
    core = offline_core(fixture)
    goals = offline_goals(fixture)
    condition = goals[0].condition if goals else goal_text
    resolved, _profile, _settings, _plan = _resolve_watch(profile_name, condition, source_key="fixture")
    rec = _recorder(core, goals, source="fixture", driver="watch", resolved=resolved,
                    on_record=_record_sink(sink))
    core.on_trace = rec.tracer()
    events = []
    async for ev in watch(offline_frames(fixture), goals, percept=core, source="fixture"):
        item = {"t": getattr(ev, "t", None), "goal_id": getattr(ev, "goal_id", None),
                "action": getattr(ev, "action", None), "text": getattr(ev, "text", None),
                # WatchEvent (the raw watch() contract) carries no revision_id/supersedes —
                # those are Incident-only fields the live percept.run() path fills below. Explicit
                # None here (not an absent key) so the client can tell "no correction" from
                # "field unsupported on this path".
                "revision_id": None, "supersedes": None}
        events.append(item)
        if sink is not None:
            sink({"type": "incident", "incident": _json_safe(item)})
    rec.close()
    rec.write_json(str(trace_path))
    return {"mode": "fixture", "events": events, "resolved": resolved, "receipt": None, "note":
            "Bundled fixture through the real runtime — scripted verdicts, proves WIRING not "
            "intelligence. (the fixture replay never consults tier-0 salience/gate config, so the "
            "resolved configuration below shows what a LIVE run of this page would use, not what "
            "shaped this replay.)"}


async def _run_real(video_path, goal_text: str, backend: str, fps: float,
                    profile_name: str, trace_path: Path, advanced: "dict | None" = None,
                    sink=None, live_frames: "_LiveFrames | None" = None,
                    max_s: "float | None" = None, run_id: "str | None" = None) -> dict:
    """Real judge over the source — the exact path percept.watch_cli --trace runs.

    [DEMO3]: the core is built via ``Percept.configure()`` — the ADR-008 configuration boundary — with
    an EXPLICIT ``SourceProfile`` (physical_scene / screen_document, see ``_resolve_watch``), and the
    goal is armed via ``apply_watch_plan`` (the WatchPlan drop-in for ``core.add_goal``), not
    hand-built. Zero PERCEPT_SALIENCE_STAT / PERCEPT_MOTION_GATE_* / PERCEPT_GATE_REFRACTORY_S /
    PERCEPT_EGO_COMPENSATE env vars anywhere in this path.

    ``advanced`` — [DEMO4] the advanced-settings panel's raw string params (``execution_policy``,
    ``identity_policy``, ``episode_tracking``, ``vision_model``/``conductor_model``/``specialist_model``
    overrides), forwarded verbatim from ``/run``'s query string. ``None``/omitted keys are each
    individually optional — an empty panel reproduces today's behavior byte-for-byte.

    ``video_path`` — a local file path OR a live URI (``rtsp://``, ``http://``, …). Both are handed
    to the SAME ``percept.run(core, source=…)`` call; ``percept/runtime/source.py`` opens an RTSP URI
    over FFMPEG/TCP itself. See the "[DEMO 04] Live sources" block above for why both demo sources
    are routed through RTSP.

    ``sink`` — [DEMO 04] a one-argument callable that puts a JSON-safe dict on ``POST /run/stream``'s
    wire. When present this run also emits ``armed`` / ``record`` / ``incident`` lines AS they happen.
    ``None`` (the ``POST /run`` path) makes every branch below a no-op.

    ``live_frames`` / ``max_s`` / ``run_id`` — [DEMO 04] the live-source trio: the ``on_frame`` tee
    that feeds the viewport and filmstrip, the wall-clock ceiling that ends an endless stream, and
    the id ``POST /stop/<run_id>`` cancels. All ``None`` on the plain file path.
    """
    advanced = advanced or {}
    import logging
    logging.disable(logging.WARNING)
    import percept
    from percept import Percept
    from percept.config import GateSettings, ModelSettings, Settings
    from percept.watchplan_apply import apply_watch_plan

    # [DEMO4 Finding 2] route the chosen backend to ALL THREE model-bearing seams, honestly — never
    # vision-only. advanced[...] per-seam text-input overrides (Finding 3c) win over the map when set.
    seams = _seam_plan(backend)
    for seam, override_key in (("vision", "vision_model"), ("conductor", "conductor_model"),
                               ("specialist", "specialist_model")):
        override = (advanced.get(override_key) or "").strip()
        if override:
            seams[seam] = {**seams[seam], "model": override, "override": True}
    _ms_default = ModelSettings()
    os.environ["PERCEPT_VISION_MODEL"] = seams["vision"]["model"] or ""
    model_settings = ModelSettings(
        vision=seams["vision"]["backend"], vision_model=seams["vision"]["model"] or _ms_default.vision_model,
        conductor=seams["conductor"]["backend"],
        conductor_model=seams["conductor"]["model"] or _ms_default.conductor_model,
        specialist=seams["specialist"]["backend"],
        specialist_model=seams["specialist"]["model"] or _ms_default.specialist_model,
    )
    exec_policy = _execution_policy_for(advanced.get("execution_policy"))
    # [DEMO4 Finding 3b] the advanced panel's identity-policy / episode-tracking authoring: an explicit
    # Goal.state.identity_policy ALWAYS wins over the resolved SourceProfile's own default (the same
    # precedence test_c2f_source_profile_defaults.py proves) — only authored when episode tracking is
    # ON and a real policy (not "none") is picked, so leaving the panel untouched stays byte-for-byte
    # "let the profile's own default_identity_policy govern" (already true for screen_document today).
    identity_policy = (advanced.get("identity_policy") or "none").strip() or "none"
    episode_tracking = bool(advanced.get("episode_tracking"))
    explicit: dict = {}
    if episode_tracking and identity_policy != "none":
        explicit["Goal.state.identity_policy"] = identity_policy
    # [DEMO4 audio addendum] "Audio (fused)" toggle: authors Goal.state.perception.modality="fused"
    # so the goal judges over BOTH vision and the AudioWindow (percept/behavior.py Modality). With a
    # video-container upload and no explicit audio_source, percept.runtime.worker.run auto-derives
    # audio_source from the SAME upload (ffmpeg demux, [AUD1]) — no extra wiring needed here. Off by
    # default (an untouched panel stays byte-for-byte vision-only).
    #
    # [DEMO 04] A LIVE source has no demuxable container sitting on disk, so that auto-derivation
    # has nothing to derive from. Rather than author modality="fused" and then starve the judge of
    # the audio half it was promised, a live run is vision-only and SAYS SO in the response.
    source_uri = str(video_path)
    is_live = _is_live_uri(source_uri)
    audio_fused = bool(advanced.get("audio_fused")) and not is_live
    audio_note = ("audio (fused) is off for a live source — there is no container to demux; "
                  "this run is vision-only.") if (is_live and advanced.get("audio_fused")) else None
    if audio_fused:
        explicit["Goal.state.perception.modality"] = "fused"
        explicit["Goal.state.perception.audio_s"] = 4.0

    # [DEMO 04] GOAL TEMPORALITY — the single most consequential thing a caller can get wrong here.
    #
    # PerceptionSpec.temporal defaults to "single": ONE focus-cropped still frame per judge call.
    # A *stative* condition ("a person is visible", "no helmet") is decidable from one frame. A
    # *dynamic* one is not, at all: "a vehicle turns right", "a vehicle stops in the intersection",
    # "someone sits down" are all statements about CHANGE, and a still photograph of a car in an
    # intersection is identical whether that car is parked or doing 40. The judge is not wrong to
    # answer "no" — it was handed evidence that cannot contain the answer.
    #
    # percept.profile_resolution names this exact axis (GOAL_TEMPORALITIES: "stative"/"dynamic",
    # _resolve_goal_type -> plan_mode "single"/"window"), so "dynamic" here authors the ordered
    # multi-frame window the SDK already resolves for that case.
    #
    # It is DECLARED by the caller, never guessed from the condition text — inferring goal semantics
    # from wording is the authoring boundary this harness deliberately does not cross.
    temporality = (advanced.get("goal_temporality") or "stative").strip() or "stative"
    temporal_note = None
    if temporality == "dynamic":
        try:
            window_s = float(advanced.get("window_s") or 4.0)
        except (TypeError, ValueError):
            window_s = 4.0
        window_s = max(1.0, min(30.0, window_s))          # SDK clamps at _MAX_WINDOW_S=30 anyway
        n_frames = max(2, min(8, int(round(window_s * fps))))
        explicit["Goal.state.perception.temporal"] = "window"
        explicit["Goal.state.perception.window_s"] = window_s
        explicit["Goal.state.perception.n_frames"] = n_frames
        explicit["Goal.state.perception.sampling"] = "even"   # span the window, not the last N frames
        temporal_note = (f"dynamic goal — judging an ordered {window_s:.0f}s window of {n_frames} "
                         f"frames, not one still frame.")
        if fps * window_s < 2:
            temporal_note += (" WARNING: at this fps the window holds barely two frames; raise fps "
                              "to see motion.")

    resolved, profile, settings, plan = _resolve_watch(
        profile_name, goal_text, source_key="camera",
        settings=Settings(gate=GateSettings(refractory_s=15.0), models=model_settings),
        execution_policy=exec_policy, explicit=explicit,
    )
    # discover_plugins left at its default (True) — byte-for-byte the old Percept.create(vision=backend)
    # call's behavior; a live run must still see installed backend plugins the same way it always did.
    core = Percept.configure(
        vision=backend, mode="live",
        source_profile=profile, source_profile_explicit=True, settings=settings,
        execution_policy=exec_policy,
    )
    receipt = apply_watch_plan(
        core, plan, source_profile=profile, source_profile_explicit=True, settings=settings,
        execution_policy=exec_policy,
    )
    goals = [g for g in (core.goals.get(gid) for gid in receipt.watch_ids) if g is not None]
    # [DEMO 04] The recorder's per-record sink does double duty on a live run: it tees the record
    # onto the wire AND, for a source.frame, keeps that frame's JPEG under the record's seq — a
    # live source leaves no file behind to re-extract thumbnails from afterwards.
    _tee = _record_sink(sink)

    def _on_record(r) -> None:
        if live_frames is not None and r.stage == "source.frame":
            live_frames.save_for_record(r)
        if _tee is not None:
            _tee(r)

    rec = _recorder(core, goals, source=("live" if is_live else "upload"), driver="run",
                    resolved=resolved, on_record=_on_record)
    if sink is not None:
        # Everything the client needs to render the run's header BEFORE the first frame lands:
        # which seams were routed, which goal ids were armed, and the manifest the recorder built.
        sink({"type": "armed", "backend": backend, "fps": fps, "seams": _json_safe(seams),
              "goal_ids": list(receipt.watch_ids), "manifest": _json_safe(rec.manifest.to_dict()),
              "audio_fused": audio_fused, "live": is_live, "source_uri": source_uri,
              "max_s": max_s, "note": " ".join(x for x in (temporal_note, audio_note) if x) or None,
              "temporality": temporality,
              "perception": {k.rsplit(".", 1)[-1]: v for k, v in explicit.items()
                             if k.startswith("Goal.state.perception.")}})
    events = []
    from percept.runtime.worker import RuntimeStats

    run_stats = RuntimeStats()
    # File uploads: inline judge dispatch (a faster-than-realtime file run would supersede the
    # interesting middle under async latest-wins), real-world onset thresholds (a phone recording of
    # a loud appliance peaks near RMS 0.08 — the 0.3 default never fires), and the onset settle so
    # the judged clip contains the sound rather than the silence before it.
    #
    # [DEMO 04] A LIVE source inverts that trade: frames keep arriving whether or not the judge has
    # answered, so an inline dispatch would stall capture and the run would fall behind real time.
    # "async" is latest-wins — the correct policy when you cannot pause the world — and it is what
    # a deployed camera watch uses. Reconnects are on because a restream/publisher can blip.
    live_kwargs = ({"judge_dispatch": "async", "source_reconnects": 2,
                    "source_reconnect_backoff_s": 1.0}
                   if is_live else {"judge_dispatch": "inline"})
    stream = percept.run(core, source=source_uri, fps=fps,
                         audio_wake={"high": 0.02, "low": 0.01, "refractory_s": 5.0},
                         audio_settle_s=2.0,
                         on_frame=(live_frames.on_frame if live_frames is not None else None),
                         on_trace=rec.tracer(), on_health=rec.health_sink(), stats=run_stats,
                         **live_kwargs)

    async def _drain() -> None:
        async for inc in stream:
            _fold_incident(inc, events, sink)

    # Whatever ends this run — EOF, the ceiling, a stop, or the source dropping out — the trace is
    # written from what was captured up to that point and the reason is named. A live source that
    # goes away is a normal outcome to REPORT, not a 500 that throws away the run.
    stop_reason = "source ended"
    try:
        if max_s is None and run_id is None:
            await _drain()
        else:
            # A live feed has no EOF, so the drain runs as a task under a watchdog: the wall-clock
            # ceiling OR POST /stop/<run_id> cancels it. Cancellation is the intended exit here.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(max_s or _LIVE_MAX_S_CEILING)
            task = asyncio.ensure_future(_drain())
            while not task.done():
                if run_id and STOPPED.get(run_id):
                    stop_reason = "stopped from the page"
                    task.cancel()
                    break
                if loop.time() >= deadline:
                    stop_reason = f"reached the {float(max_s or 0):.0f}s ceiling"
                    task.cancel()
                    break
                await asyncio.sleep(0.2)
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception as exc:  # noqa: BLE001 — reported as the run's ending, not raised
        stop_reason = f"source stopped delivering frames ({type(exc).__name__}: {exc})"
        if sink is not None:
            sink({"type": "note", "text": stop_reason})
    finally:
        try:
            await stream.aclose()
        except Exception:  # noqa: BLE001
            pass

    rec.close()
    rec.write_json(str(trace_path))
    return {"mode": "real", "backend": backend, "events": events, "resolved": resolved, "receipt": receipt,
            "seams": seams, "video_path": source_uri, "live": is_live, "stop_reason": stop_reason,
            "note": " ".join(x for x in (temporal_note, audio_note) if x) or None,
            "temporality": temporality,
            "audio": {"enabled": audio_fused, "chunks": run_stats.audio_chunks, "wakes": run_stats.audio_wakes}}


def _fold_incident(inc, events: list, sink) -> None:
    """One delivered ``Incident`` -> the wire dict both ``/run`` and ``/run/stream`` speak."""
    item = {
        "t": None,
        "goal_id": getattr(inc, "concern_id", getattr(inc, "goal_id", None)),
        "action": getattr(getattr(inc, "verdict", None), "satisfied", None) and "fire" or "fire",
        "text": getattr(inc, "message", None) or getattr(inc, "say", None),
        "confidence": getattr(getattr(inc, "verdict", None), "confidence", None),
        "entity_id": getattr(inc, "entity_id", None),
        # D11 — the same correlation handle the .ptrace's judgment.event record carries
        # (dispatch_seq there, evidence_seq here — one JudgmentEvent, two labels on the SAME
        # number): lets the client resolve this delivered Incident back to the trace record for
        # the evidence that produced it. None on a fire from a path that built no JudgmentEvent.
        "evidence_seq": getattr(inc, "evidence_seq", None),
        # Episode-owned provisional/superseding link (Incident 1.8 wire fields, Y-epic): a fire that
        # CORRECTS an earlier provisional fire on the same episode names it in ``supersedes``; an
        # original/provisional fire carries ``supersedes=None``. See percept/incidents.py.
        "revision_id": getattr(inc, "revision_id", None),
        "supersedes": getattr(inc, "supersedes", None),
    }
    events.append(item)
    if sink is not None:
        sink({"type": "incident", "incident": _json_safe(item)})


# =================================================================================================
# [DEMO 04] Live sources. Demo 04 feeds the runtime from TWO source classes that reach it through
# the SAME code path — ``percept.run(core, source=<uri>)``:
#
#   (1) a real live feed, restreamed to a local RTSP server:
#         mediamtx                                            # terminal 1 (or: docker run --rm -it \
#                                                             #   -p 8554:8554 bluenviron/mediamtx)
#         ffmpeg -i "$(yt-dlp -g '<youtube live url>')" \      # terminal 2
#                -c copy -f rtsp rtsp://localhost:8554/traffic
#       -> the page watches rtsp://localhost:8554/traffic
#
#   (2) an uploaded video RESTREAMED through that same RTSP server by this harness (see
#       ``_restream_publisher``), so a file behaves exactly like the live feed: real-time paced,
#       no seeking, no rewind, no look-ahead.
#
# The runtime does not know the difference — ``percept/runtime/source.py`` already opens an
# ``rtsp://`` URI over FFMPEG/TCP. That is the whole point of routing both through RTSP: source (2)
# is a rehearsal for source (1), on the same seam.
# =================================================================================================
_LIVE_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "rtmp://")
#: Wall-clock ceiling on a live run, so a stream with no end still terminates and writes its trace.
_LIVE_MAX_S_DEFAULT = 90.0
_LIVE_MAX_S_CEILING = 900.0
#: Publish target for a restreamed upload. Override with PERCEPT_DEMO_RTSP_BASE.
_RTSP_BASE = os.environ.get("PERCEPT_DEMO_RTSP_BASE", "rtsp://127.0.0.1:8554")
#: run_id -> True once a client asked to stop it (POST /stop/<run_id>).
STOPPED: dict[str, bool] = {}


def _is_live_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.lower().startswith(_LIVE_SCHEMES)


def _probe_rtsp(uri: str, timeout_s: float = 6.0) -> "tuple[bool, str]":
    """Can we actually open ``uri`` right now? Returns ``(ok, reason)``.

    Used both by ``GET /source-check`` (so the page can say "mediamtx isn't running" BEFORE you
    arm and spend tokens) and by ``_restream_publisher`` to wait for its own publish to go live.
    """
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return False, f"opencv not installed in this environment ({e}) — pip install opencv-python-headless"
    deadline = time.monotonic() + timeout_s
    last = "could not open the source"
    while time.monotonic() < deadline:
        cap = None
        try:
            cap = (cv2.VideoCapture(uri, cv2.CAP_FFMPEG) if uri.lower().startswith("rtsp")
                   else cv2.VideoCapture(uri))
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return True, "reachable"
                last = "opened but delivered no frame (publisher not sending yet?)"
            else:
                last = "connection refused / no such path (is mediamtx running on 8554?)"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        finally:
            if cap is not None:
                cap.release()
        time.sleep(0.4)
    return False, last


class _RestreamPublisher:
    """ffmpeg process that republishes a local file to the demo's RTSP server, in real time.

    ``-re`` is the entire point: it paces the file at its native frame rate, so the runtime sees a
    LIVE feed it cannot outrun, exactly like source (1). ``-c copy`` is tried first (near-zero CPU,
    works for the H.264 MP4s people actually upload); a source ffmpeg refuses to copy falls back to
    a fast libx264 transcode rather than failing the run.

    ``-stream_loop -1`` makes the file ENDLESS, which is not a convenience — it is the property that
    makes this a live source. A one-shot ``-re`` publish is a race the harness loses: arming a watch
    (configure -> plan -> registry discovery) takes seconds, and a short clip can be over before the
    runtime opens the socket. A looping publisher behaves like the traffic camera it stands in for:
    it is always mid-stream, you join wherever it happens to be, and the run ends on the wall-clock
    ceiling or on POST /stop — never on the file running out.
    """

    def __init__(self, video_path: Path, path_name: str) -> None:
        self.video_path = video_path
        self.url = f"{_RTSP_BASE.rstrip('/')}/{path_name}"
        self.proc = None
        self.mode = None
        self._err: "list[str]" = []

    def _spawn(self, codec_args: list) -> "subprocess.Popen":
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-stream_loop", "-1", "-re", "-i", str(self.video_path),
               "-an", *codec_args, "-f", "rtsp", "-rtsp_transport", "tcp", self.url]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, errors="replace")
        # Drain stderr on a thread. Two reasons, both load-bearing: ffmpeg BLOCKS once a 64K pipe
        # fills, and its stderr is the only place that says WHY a publish failed. Guessing at the
        # reason ("is the server up?") when ffmpeg already printed "path is not configured" is
        # exactly the kind of unhelpful error this harness is supposed to not produce.
        self._err = []

        def _drain(stream, buf):
            try:
                for line in stream:
                    buf.append(line.rstrip())
                    del buf[:-20]
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=_drain, args=(proc.stderr, self._err), daemon=True).start()
        return proc

    def _ffmpeg_says(self) -> str:
        msg = " / ".join(x for x in self._err if x)
        return msg[:400]

    def start(self) -> "tuple[bool, str]":
        """Publish, then wait until the stream is actually readable. Returns ``(ok, reason)``."""
        if shutil.which("ffmpeg") is None:
            return False, "ffmpeg is not on PATH — install it to restream an upload."
        attempts = []
        for mode, args in (("copy", ["-c:v", "copy"]),
                           ("transcode", ["-c:v", "libx264", "-preset", "veryfast",
                                          "-tune", "zerolatency", "-pix_fmt", "yuv420p"])):
            self.proc = self._spawn(args)
            self.mode = mode
            # Poll the publisher AND the stream: if ffmpeg dies, stop waiting immediately and keep
            # what it said, rather than burning the whole probe timeout on a corpse.
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    break
                ok, _ = _probe_rtsp(self.url, timeout_s=1.0)
                if ok:
                    return True, f"publishing with -c:v {mode}"
            attempts.append(f"-c:v {mode}: {self._ffmpeg_says() or 'timed out with no output'}")
            self.stop()
        hint = ""
        joined = " ".join(attempts).lower()
        if "not configured" in joined or "400 bad request" in joined:
            hint = (" — the RTSP server is up but refuses this path. mediamtx only accepts paths its "
                    "config allows: run it with a config containing `paths:\\n  all_others:` "
                    "(docs/demo/mediamtx.yml ships one: `mediamtx docs/demo/mediamtx.yml`).")
        elif "connection refused" in joined or "immediate exit" in joined:
            hint = (" — nothing is listening on that address. Start the RTSP server first: "
                    "`mediamtx docs/demo/mediamtx.yml` (or `docker run --rm -it -p 8554:8554 "
                    "bluenviron/mediamtx`).")
        return False, f"could not publish to {self.url}{hint} [ffmpeg said] " + " | ".join(attempts)

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try: p.kill()
            except Exception: pass


class _LiveFrames:
    """Tee of the runtime's OWN frames, for the live viewport and the filmstrip.

    ``percept.run``'s documented ``on_frame(t, judge_frame)`` hook hands over exactly what the judge
    will see — ``("image/jpeg", <bytes>)`` straight off ``percept/runtime/source.py``. There is no
    second decode of the source anywhere in this class: the preview IS the judged frame.

    Two consumers, one buffer:
      * the wire — a downscaled JPEG, throttled to :data:`PREVIEW_FPS`, so a 1080p feed does not
        push megabytes per second at the browser;
      * the filmstrip — when a ``source.frame`` record lands, the frame with the SAME ``source_ts``
        is written to this run's cache dir as ``<seq>.jpg``, which is what ``GET /frame/...`` serves.
        A live source has no file left to seek afterwards, so this is the only chance to keep it.
    """

    PREVIEW_FPS = 2.0
    PREVIEW_W = 480
    KEEP = 12  # recent frames held for source.frame correlation

    def __init__(self, run_id: str, sink=None) -> None:
        self.run_id = run_id
        self.sink = sink
        self.recent: "list[tuple[float, bytes]]" = []
        self.n = 0
        self._last_emit = -1e9

    def _shrink(self, jpeg: bytes) -> bytes:
        try:
            import cv2
            import numpy as np
        except Exception:  # noqa: BLE001
            return jpeg
        try:
            bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[1] <= self.PREVIEW_W:
                return jpeg
            h = max(1, int(bgr.shape[0] * self.PREVIEW_W / bgr.shape[1]))
            ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (self.PREVIEW_W, h)),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            return buf.tobytes() if ok else jpeg
        except Exception:  # noqa: BLE001
            return jpeg

    def on_frame(self, t, judge_frame) -> None:
        """``percept.run(on_frame=...)`` sink. Never raises into the capture loop."""
        try:
            jpeg = judge_frame[1] if isinstance(judge_frame, tuple) and len(judge_frame) == 2 else None
            if not isinstance(jpeg, (bytes, bytearray)):
                return
            small = self._shrink(bytes(jpeg))
            self.recent.append((float(t), small))
            if len(self.recent) > self.KEEP:
                del self.recent[: len(self.recent) - self.KEEP]
            if self.sink is not None and (t - self._last_emit) >= (1.0 / self.PREVIEW_FPS):
                self._last_emit = float(t)
                self.n += 1
                self.sink({"type": "frame", "n": self.n, "t": float(t),
                           "jpeg_b64": base64.b64encode(small).decode()})
        except Exception:  # noqa: BLE001 — a preview must never break perception
            pass

    def save_for_record(self, rec) -> None:
        """Persist the frame this ``source.frame`` record is about, under its recorder seq."""
        try:
            ts = rec.source_ts
            if ts is None or not self.recent:
                return
            _, jpeg = min(self.recent, key=lambda p: abs(p[0] - ts))
            (_frame_cache_dir(self.run_id) / f"{int(rec.seq)}.jpg").write_bytes(jpeg)
        except Exception:  # noqa: BLE001
            pass


def _frame_cache_dir(run_id: str) -> Path:
    """[DEMO4 Finding 4] the per-run cache dir GET /frame/<run_id>/<seq>.jpg serves from — created
    lazily so a fixture run (which never extracts anything) doesn't litter tempdir."""
    d = Path(tempfile.gettempdir()) / "percept-demo-frames" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_frame_thumbnails(video_path: Path, records: list, run_id: str) -> None:
    """[DEMO4 Finding 4] best-effort: pull one JPEG thumbnail per ``source.frame`` trace record by
    seeking the SAME uploaded video at that record's ``source_ts`` — the runtime's own media
    timebase (``percept/runtime/source.py``: ``t = i / src_fps``) — and cache them in this run's
    temp dir so ``GET /frame/<run_id>/<seq>.jpg`` can serve them AFTER the upload itself is deleted
    (``_do_run`` unlinks it once the run finishes). Must run BEFORE that unlink.

    Never raises: a missing thumbnail is a UI nicety gap (the filmstrip shows a placeholder / the
    endpoint 404s with a clear reason), never a reason to fail an otherwise-successful /run.
    """
    try:
        import cv2
    except Exception:
        return
    frame_recs = [r for r in records
                  if r.get("stage") == "source.frame" and isinstance(r.get("source_ts"), (int, float))]
    if not frame_recs:
        return
    try:
        cap = cv2.VideoCapture(str(video_path))
    except Exception:
        return
    if not cap.isOpened():
        cap.release()
        return
    out_dir = _frame_cache_dir(run_id)
    try:
        for r in frame_recs:
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(r["source_ts"])) * 1000.0)
                ok, bgr = cap.read()
                if not ok:
                    continue
                ok, buf = cv2.imencode(".jpg", bgr)
                if not ok:
                    continue
                (out_dir / f"{int(r['seq'])}.jpg").write_bytes(buf.tobytes())
            except Exception:
                continue  # one bad frame must not abort the rest of the filmstrip
    finally:
        cap.release()


#: [DEMO4 Finding 3c] the advanced-panel query params a caller may set on /run — every one is
#: individually optional so an empty panel is byte-for-byte today's behavior.
_ADVANCED_PARAM_KEYS = (
    "execution_policy", "identity_policy", "episode_tracking", "audio_fused",
    "vision_model", "conductor_model", "specialist_model",
    # [DEMO 04] goal temporality — see _run_real. Declared by the caller, never inferred from the
    # condition text: reading the goal string to guess what it "means" is the authoring boundary
    # this harness does not cross.
    "goal_temporality", "window_s",
)
_ADVANCED_BOOL_KEYS = ("episode_tracking", "audio_fused")


def _advanced_from_params(params: dict) -> dict:
    out = {k: (params.get(k, [""])[0] or "").strip() for k in _ADVANCED_PARAM_KEYS}
    for k in _ADVANCED_BOOL_KEYS:
        out[k] = out[k] in ("1", "true", "on", "yes")
    # audio (fused) defaults ON for an upload run when the parameter is ABSENT entirely — a page
    # loaded before this panel existed sends no audio_fused key at all, and treating that as "off"
    # silently starves the fused judge of the upload's own soundtrack. An explicit "0" still wins.
    if "audio_fused" not in params:
        out["audio_fused"] = True
    return out


def _do_run(params: dict, body: bytes, sink=None) -> dict:
    """Blocking entry: run one real run to completion, return {run_id, manifest, summary, records, events}.

    ``sink`` — [DEMO 04] optional one-argument callable fed a JSON-safe dict per live event
    (``start`` / ``armed`` / ``record`` / ``incident`` / ``thumbnails``). ``POST /run`` passes
    ``None``, so its behaviour is unchanged; ``POST /run/stream`` passes a queue-put.
    """
    run_id = f"run_{int(time.time()*1000)}"
    trace_path = Path(tempfile.gettempdir()) / f"percept-demo-{run_id}.ptrace"
    # Register the run BEFORE it starts, not after it finishes. GET /frame/<run_id>/<seq>.jpg
    # rejects an unknown run_id, and a live run serves frames WHILE it is still going — registering
    # at the end made every in-flight thumbnail 404 with "unknown run" until the run completed.
    # `mode` is filled in below once the run is done; GET /trace guards on the file existing.
    RUNS[run_id] = {"trace": str(trace_path), "mode": None}
    goal = (params.get("goal", ["the target condition is met"])[0]).strip() or "the target condition is met"
    mode = params.get("mode", [""])[0]
    profile_name = (params.get("profile", ["physical_scene"])[0]).strip() or "physical_scene"
    advanced = _advanced_from_params(params)

    # [DEMO 04] Which of the three source classes is this?
    #   source_url=<rtsp://…>  -> a LIVE feed, watched in place (demo 04 source 1)
    #   restream=1 + body      -> an upload RE-PUBLISHED to RTSP and watched live (demo 04 source 2)
    #   body                   -> the upload watched as a file, faster-than-real-time (demo 01)
    source_url = (params.get("source_url", [""])[0] or "").strip()
    restream = (params.get("restream", [""])[0] or "").strip() in ("1", "true", "on", "yes")
    try:
        max_s = min(float(params.get("max_s", [_LIVE_MAX_S_DEFAULT])[0] or _LIVE_MAX_S_DEFAULT),
                    _LIVE_MAX_S_CEILING)
    except ValueError:
        max_s = _LIVE_MAX_S_DEFAULT

    if mode == "fixture" or (not body and not source_url):
        if sink is not None:
            sink({"type": "start", "run_id": run_id, "goal": goal, "profile": profile_name,
                  "mode": "fixture"})
        result = asyncio.run(_run_fixture(goal, profile_name, trace_path, sink=sink))
    else:
        key = _have_key()
        if not key:
            return {"error": "no_key",
                    "message": "No judge API key is set (OPENROUTER_API_KEY / GEMINI_API_KEY / "
                               "ANTHROPIC_API_KEY). Set one to run a real judge over your video, or use "
                               "the keyless fixture button (real runtime, scripted verdicts)."}
        backend = (params.get("backend", ["openrouter"])[0]).strip() or "openrouter"
        fps = float((params.get("fps", ["1.0"])[0]) or 1.0)
        vid = None            # the temp upload, if there is one
        publisher = None      # the ffmpeg restream, if there is one
        live_frames = None
        try:
            if source_url:
                if not _is_live_uri(source_url):
                    return {"error": "bad_source",
                            "message": f"{source_url!r} is not a live URI — expected one of "
                                       + ", ".join(_LIVE_SCHEMES)}
                uri, source_kind = source_url, "live_url"
            else:
                suffix = Path(params.get("source_name", ["upload.mp4"])[0]).suffix or ".mp4"
                vid = Path(tempfile.gettempdir()) / f"percept-demo-{run_id}{suffix}"
                vid.write_bytes(body)
                if restream:
                    publisher = _RestreamPublisher(vid, f"percept-{run_id}")
                    if sink is not None:
                        sink({"type": "start", "run_id": run_id, "goal": goal, "profile": profile_name,
                              "mode": "real", "source_kind": "restream", "bytes": len(body),
                              "source_uri": publisher.url, "max_s": max_s,
                              "note": "republishing your upload to RTSP in real time…"})
                    ok, reason = publisher.start()
                    if not ok:
                        return {"error": "restream_failed", "message": reason}
                    uri, source_kind = publisher.url, "restream"
                    if sink is not None:
                        sink({"type": "note", "text": f"restream live at {publisher.url} ({reason})"})
                else:
                    uri, source_kind = str(vid), "file"

            live = source_kind != "file"
            if live:
                live_frames = _LiveFrames(run_id, sink=sink)
                STOPPED.pop(run_id, None)
            if sink is not None and source_kind != "restream":
                sink({"type": "start", "run_id": run_id, "goal": goal, "profile": profile_name,
                      "mode": "real", "source_kind": source_kind, "source_uri": uri,
                      "bytes": len(body), "max_s": max_s if live else None})

            result = asyncio.run(_run_real(
                uri, goal, backend, fps, profile_name, trace_path, advanced, sink=sink,
                live_frames=live_frames,
                max_s=(max_s if live else None), run_id=(run_id if live else None)))
            result["source_kind"] = source_kind

            # [DEMO4 Finding 4] A FILE run extracts thumbnails by re-seeking the upload — which must
            # happen before the finally-block unlink. A LIVE run has no seekable file at all: its
            # frames were already saved by _LiveFrames.save_for_record as they arrived.
            try:
                if not live and vid is not None:
                    _extract_frame_thumbnails(
                        vid, json.loads(trace_path.read_text()).get("records", []), run_id)
                # The live client renders placeholder cells while the run is in flight; this is the
                # line that tells it GET /frame/<run_id>/<seq>.jpg will now resolve.
                if sink is not None:
                    sink({"type": "thumbnails", "run_id": run_id})
            except Exception:
                pass
        finally:
            if publisher is not None:
                publisher.stop()
            STOPPED.pop(run_id, None)
            if vid is not None:
                try: vid.unlink()
                except OSError: pass

    trace_json = json.loads(trace_path.read_text())
    RUNS[run_id] = {"trace": str(trace_path), "mode": result.get("mode")}
    return {"run_id": run_id, "goal": goal, "profile": profile_name,
            # manifest already carries build/provenance (build_commit/build_timestamp) PLUS — now that
            # _recorder is handed the run's ResolvedWatch — source_profile_id/profiles/
            # component_snapshot_hash/optimization_lock_hash: surfaced as-is, one source of truth.
            "manifest": trace_json.get("manifest"), "summary": trace_json.get("summary"),
            "records": trace_json.get("records", []), "events": result.get("events", []),
            "mode": result.get("mode"), "note": result.get("note"), "backend": result.get("backend"),
            "resolved_explain": _resolved_explain_json(result.get("resolved")),
            "receipt": _receipt_json(result.get("receipt")),
            # [DEMO4 Finding 2] per-seam routing plan (vision/conductor/specialist -> {backend, model,
            # fallback_from, reason}) — None on the fixture path (no live backend is ever chosen there).
            "seams": result.get("seams"),
            # [DEMO4] the advanced panel's own selections, echoed back verbatim — the UI reads THIS
            # (not its own form state) to prove the Resolved configuration panel reflects what the
            # server actually armed, not merely what the client asked for.
            "advanced": advanced,
            # [DEMO4 audio addendum] RuntimeStats.audio_chunks/audio_wakes when the Audio (fused)
            # toggle was on; None on the fixture path (no percept.run() stats object there).
            "audio": result.get("audio"),
            # [DEMO 04] which source class actually fed this run ("file" / "restream" / "live_url"),
            # the URI it was opened on, and WHY a live run ended (EOF vs ceiling vs stopped).
            "source_kind": result.get("source_kind"), "source_uri": result.get("video_path"),
            "live": bool(result.get("live")), "stop_reason": result.get("stop_reason")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, INDEX, "text/html; charset=utf-8")
        if path.startswith("/trace/") and path.endswith(".ptrace"):
            run_id = path[len("/trace/"):-len(".ptrace")]
            run = RUNS.get(run_id)
            if not run:
                return self._send(404, json.dumps({"error": "unknown run"}))
            p = Path(run["trace"])
            if not p.is_file():
                # The run is registered but still going — the recorder writes the artifact on close.
                return self._send(404, json.dumps({
                    "error": "trace_not_written_yet",
                    "message": "this run is still in flight; the .ptrace is written when it ends."}))
            return self._send(200, p.read_bytes(), "application/json")
        # [DEMO4 Finding 4] GET /frame/<run_id>/<seq>.jpg — a cached thumbnail extracted from the
        # uploaded video at that source.frame record's timestamp (see _extract_frame_thumbnails).
        # A fixture run never had a video file to extract from — that fact is the 404 reason, not
        # a bug to paper over with a blank image.
        if path.startswith("/frame/"):
            rest = path[len("/frame/"):]
            if "/" in rest and rest.endswith(".jpg"):
                run_id, seq_part = rest.rsplit("/", 1)
                seq_part = seq_part[:-len(".jpg")]
                run = RUNS.get(run_id)
                if not run:
                    return self._send(404, json.dumps({"error": "unknown_run", "message": "unknown run_id"}))
                if not seq_part.isdigit():
                    return self._send(400, json.dumps({"error": "bad_seq", "message": "seq must be an integer"}))
                f = _frame_cache_dir(run_id) / f"{seq_part}.jpg"
                if f.is_file():
                    return self._send(200, f.read_bytes(), "image/jpeg")
                if run.get("mode") == "fixture":
                    reason = ("fixture run — synthetic frames have no source video to extract a "
                              "thumbnail from; the trace's own source.frame data is the evidence here.")
                else:
                    reason = ("no thumbnail cached for this frame — either this seq is not a "
                              "source.frame record, or extraction failed for that frame (see server logs).")
                return self._send(404, json.dumps({"error": "no_thumbnail", "message": reason}))
            return self._send(404, json.dumps({"error": "not found"}))
        # [DEMO 04] GET /source-check?url=rtsp://… — can the runtime open this source RIGHT NOW?
        # Answered before arming so "nothing happened" is never a mystery: the page can say
        # "mediamtx isn't running on 8554" instead of starting a run that reads zero frames.
        if path == "/source-check":
            url = (parse_qs(urlparse(self.path).query).get("url", [""])[0] or "").strip()
            if not url:
                return self._send(400, json.dumps({"error": "no_url", "message": "pass ?url="}))
            if not _is_live_uri(url):
                return self._send(400, json.dumps({
                    "error": "bad_source",
                    "message": f"{url!r} is not a live URI — expected one of " + ", ".join(_LIVE_SCHEMES)}))
            ok, reason = _probe_rtsp(url, timeout_s=5.0)
            return self._send(200, json.dumps({"ok": ok, "url": url, "reason": reason,
                                               "ffmpeg": shutil.which("ffmpeg") is not None}))
        # static files from docs/demo/
        fname = path.lstrip("/")
        f = HERE / fname
        if f.suffix in (".html", ".md", ".css", ".js") and f.is_file() and HERE in f.resolve().parents:
            ctype = {"html": "text/html; charset=utf-8", "md": "text/markdown; charset=utf-8",
                     "css": "text/css", "js": "text/javascript"}[f.suffix.lstrip(".")]
            return self._send(200, f.read_bytes(), ctype)
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        # [DEMO 04] POST /stop/<run_id> — end a live run early. A stream with no EOF must be
        # stoppable from the page; the run still closes cleanly and writes its .ptrace.
        if path.startswith("/stop/"):
            run_id = path[len("/stop/"):]
            STOPPED[run_id] = True
            return self._send(200, json.dumps({"ok": True, "run_id": run_id}))
        if path not in ("/run", "/run/stream"):
            return self._send(404, json.dumps({"error": "not found"}))
        params = parse_qs(urlparse(self.path).query)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        if path == "/run/stream":
            return self._stream_run(params, body)
        try:
            result = _do_run(params, body)
            code = 400 if result.get("error") else 200
            return self._send(code, json.dumps(result))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": "run_failed", "message": str(e)}))

    # -- POST /run/stream ---------------------------------------------------------------------
    #: How long the drain loop waits on a silent queue before writing a keep-alive line. A real
    #: judge call routinely takes tens of seconds; without this the client cannot tell "still
    #: thinking" from "harness died".
    _STREAM_PING_S = 10.0

    def _stream_run(self, params: dict, body: bytes):
        """[DEMO 04] The SAME run as ``POST /run``, streamed as NDJSON while it happens.

        The run itself is blocking and synchronous (``_do_run`` → ``asyncio.run``), so it goes on a
        worker thread and publishes to a queue that THIS thread drains onto the socket. One JSON
        object per line:

            {"type":"start"}      run_id + goal, before the first frame is read
            {"type":"armed"}      seams / goal ids / manifest, once the plan is applied
            {"type":"record"}     one per trace record, as the recorder ingests it
            {"type":"incident"}   one per Incident the runtime actually delivered
            {"type":"thumbnails"} GET /frame/<run_id>/<seq>.jpg now resolves
            {"type":"ping"}       keep-alive during a long judge call
            {"type":"done"}       the exact payload POST /run would have returned

        No ``Content-Length``: the response is HTTP/1.0-framed, so end-of-stream is the connection
        close after ``done``. A client that disconnects early breaks the write and ends the drain;
        the run finishes anyway and its ``.ptrace`` is still fetchable by ``run_id``.
        """
        q: "queue.Queue" = queue.Queue()
        done = object()
        holder: dict = {}

        def worker():
            try:
                holder["result"] = _do_run(params, body, sink=q.put)
            except Exception as e:  # noqa: BLE001 — reported to the client, not swallowed
                traceback.print_exc()
                holder["result"] = {"error": "run_failed", "message": str(e)}
            finally:
                q.put(done)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")   # no proxy buffering, if one is ever in front
        self.end_headers()
        threading.Thread(target=worker, name="percept-demo-run", daemon=True).start()

        def write(obj) -> None:
            self.wfile.write((json.dumps(obj, default=str) + "\n").encode())
            self.wfile.flush()

        try:
            while True:
                try:
                    item = q.get(timeout=self._STREAM_PING_S)
                except queue.Empty:
                    write({"type": "ping"})
                    continue
                if item is done:
                    break
                write(item)
            write({"type": "done",
                   "result": holder.get("result") or {"error": "run_failed",
                                                      "message": "the run produced no result"}})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-run; the run itself is unaffected


INDEX = """<!doctype html><meta charset=utf-8><title>Percept demos</title>
<style>body{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 20px;color:#0a0a0a}
h1{font-family:Arial Narrow,sans-serif;letter-spacing:.02em}code{background:#f4f4f0;padding:2px 5px;font-size:13px}
a{display:block;border:1px solid #0a0a0a;padding:14px 16px;margin:12px 0;color:#0a0a0a;text-decoration:none}a:hover{background:#faf7ee}
.k{font:11px ui-monospace,monospace;color:#8a8a8a;margin-top:24px}</style>
<h1>Percept demo harness</h1>
<p>Real <code>watch()</code> runtime behind every page. Uploads run a real judge when an API key is set;
otherwise the keyless fixture runs the real runtime with scripted verdicts.</p>
<a href="/demo-01-video-watch.html">Demo 01 · MP4 video watch →</a>
<a href="/demo-02-screen-watch.html">Demo 02 · Screen-share watch →</a>
<a href="/demo-04-stream-uploaded-video.html">Demo 04 · Live stream + replay →</a>
<p class=k>judge key: %s</p>""" % ("SET — real judge available" if _have_key() else
                                   "UNSET — fixture only (set OPENROUTER_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY)")


def main():
    try:
        import percept  # noqa: F401
    except Exception as e:
        print(f"  ! cannot import percept-vision ({e}). Run from the repo, or pip install the package.")
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    key = _have_key()
    print("─" * 68)
    print(f"  PERCEPT demo harness  ·  http://localhost:{PORT}")
    print(f"  judge key: {key or 'UNSET (keyless fixture only — real runtime, scripted verdicts)'}")
    print(f"  real judge over uploads: {'ENABLED' if key else 'set an API key to enable'}")
    print("─" * 68)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
