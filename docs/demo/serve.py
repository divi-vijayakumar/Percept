#!/usr/bin/env python3
"""Percept demo harness — run the REAL runtime behind the two demo pages.

This is an INTERNAL product-test harness (dogfood the wheel before a client wheel ships), not a
shipped API. It wraps the frozen ``watch()``/``run()`` seam — the exact path ``percept.watch_cli
--trace`` uses — so the browser pages render REAL WatchEvents and a REAL ``.ptrace``, never mocked data.

    python docs/demo/serve.py                 # http://localhost:8788  (keyless: bundled fixture, real runtime)
    PERCEPT_VISION_MODEL=... OPENROUTER_API_KEY=...  python docs/demo/serve.py   # real judge over uploads

Endpoints
    GET  /                         -> demo index (links to both pages)
    GET  /demo-01-video-watch.html -> the MP4 upload demo (real client)
    GET  /demo-02-screen-watch.html-> the screen-share demo (real client)
    POST /run?goal=..&backend=..&fps=..&source_name=..   body = raw video bytes
                                   -> runs the REAL runtime over the upload, returns the real trace JSON
    POST /run?mode=fixture         -> runs the REAL runtime over the bundled fixture (keyless, no spend)
    GET  /trace/<run_id>.ptrace    -> download the real .ptrace written for that run
    GET  /frame/<run_id>/<seq>.jpg -> a cached thumbnail extracted from the uploaded video at that
                                   source.frame record's timestamp (404 + reason on a fixture run)

No judge key present -> uploads are refused with a clear message and the fixture path is offered, so the
harness is honest about when it is and is not exercising a real model.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
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


def _recorder(core, goals, *, source: str, driver: str, resolved=None):
    """Same recorder the real CLI builds for --trace: full retention + the live Gate's knobs.

    ``resolved`` — the run's ``ResolvedWatch`` (or ``None``), forwarded straight to
    ``RunTraceManifest.create`` so the manifest's own ``source_profile_id``/``profiles``/
    ``component_snapshot_hash``/``optimization_lock_hash`` fields are filled from the SAME resolution
    the founder sees in the "Resolved configuration" panel — one source of truth, not a second copy.
    """
    from percept.replay import gate_config_of
    from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

    manifest = RunTraceManifest.create(
        source=source, driver=driver, goals=goals,
        policy=TracePolicy(level="full", max_duration_s=None),
        gate_config=gate_config_of(core.gate),
        resolved=resolved,
    )
    return TraceRecorder(manifest)


async def _run_fixture(goal_text: str, profile_name: str, trace_path: Path) -> dict:
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
    rec = _recorder(core, goals, source="fixture", driver="watch", resolved=resolved)
    core.on_trace = rec.tracer()
    events = []
    async for ev in watch(offline_frames(fixture), goals, percept=core, source="fixture"):
        events.append({"t": getattr(ev, "t", None), "goal_id": getattr(ev, "goal_id", None),
                       "action": getattr(ev, "action", None), "text": getattr(ev, "text", None),
                       # WatchEvent (the raw watch() contract) carries no revision_id/supersedes —
                       # those are Incident-only fields the live percept.run() path fills below. Explicit
                       # None here (not an absent key) so the client can tell "no correction" from
                       # "field unsupported on this path".
                       "revision_id": None, "supersedes": None})
    rec.close()
    rec.write_json(str(trace_path))
    return {"mode": "fixture", "events": events, "resolved": resolved, "receipt": None, "note":
            "Bundled fixture through the real runtime — scripted verdicts, proves WIRING not "
            "intelligence. (the fixture replay never consults tier-0 salience/gate config, so the "
            "resolved configuration below shows what a LIVE run of this page would use, not what "
            "shaped this replay.)"}


async def _run_real(video_path: Path, goal_text: str, backend: str, fps: float,
                    profile_name: str, trace_path: Path, advanced: "dict | None" = None) -> dict:
    """Real judge over the uploaded video — the exact path percept.watch_cli --trace runs.

    [DEMO3]: the core is built via ``Percept.configure()`` — the ADR-008 configuration boundary — with
    an EXPLICIT ``SourceProfile`` (physical_scene / screen_document, see ``_resolve_watch``), and the
    goal is armed via ``apply_watch_plan`` (the WatchPlan drop-in for ``core.add_goal``), not
    hand-built. Zero PERCEPT_SALIENCE_STAT / PERCEPT_MOTION_GATE_* / PERCEPT_GATE_REFRACTORY_S /
    PERCEPT_EGO_COMPENSATE env vars anywhere in this path.

    ``advanced`` — [DEMO4] the advanced-settings panel's raw string params (``execution_policy``,
    ``identity_policy``, ``episode_tracking``, ``vision_model``/``conductor_model``/``specialist_model``
    overrides), forwarded verbatim from ``/run``'s query string. ``None``/omitted keys are each
    individually optional — an empty panel reproduces today's behavior byte-for-byte.
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
    rec = _recorder(core, goals, source="upload", driver="run", resolved=resolved)
    events = []
    stream = percept.run(core, source=str(video_path), fps=fps,
                         on_trace=rec.tracer(), on_health=rec.health_sink())
    async for inc in stream:
        events.append({
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
        })
    rec.close()
    rec.write_json(str(trace_path))
    return {"mode": "real", "backend": backend, "events": events, "resolved": resolved, "receipt": receipt,
            "seams": seams, "video_path": str(video_path)}


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
    "execution_policy", "identity_policy", "episode_tracking",
    "vision_model", "conductor_model", "specialist_model",
)


def _advanced_from_params(params: dict) -> dict:
    out = {k: (params.get(k, [""])[0] or "").strip() for k in _ADVANCED_PARAM_KEYS}
    out["episode_tracking"] = out["episode_tracking"] in ("1", "true", "on", "yes")
    return out


def _do_run(params: dict, body: bytes) -> dict:
    """Blocking entry: run one real run to completion, return {run_id, manifest, summary, records, events}."""
    run_id = f"run_{int(time.time()*1000)}"
    trace_path = Path(tempfile.gettempdir()) / f"percept-demo-{run_id}.ptrace"
    goal = (params.get("goal", ["the target condition is met"])[0]).strip() or "the target condition is met"
    mode = params.get("mode", [""])[0]
    profile_name = (params.get("profile", ["physical_scene"])[0]).strip() or "physical_scene"
    advanced = _advanced_from_params(params)

    if mode == "fixture" or not body:
        result = asyncio.run(_run_fixture(goal, profile_name, trace_path))
    else:
        key = _have_key()
        if not key:
            return {"error": "no_key",
                    "message": "No judge API key is set (OPENROUTER_API_KEY / GEMINI_API_KEY / "
                               "ANTHROPIC_API_KEY). Set one to run a real judge over your video, or use "
                               "the keyless fixture button (real runtime, scripted verdicts)."}
        backend = (params.get("backend", ["openrouter"])[0]).strip() or "openrouter"
        fps = float((params.get("fps", ["1.0"])[0]) or 1.0)
        suffix = Path(params.get("source_name", ["upload.mp4"])[0]).suffix or ".mp4"
        vid = Path(tempfile.gettempdir()) / f"percept-demo-{run_id}{suffix}"
        vid.write_bytes(body)
        try:
            result = asyncio.run(_run_real(vid, goal, backend, fps, profile_name, trace_path, advanced))
            # [DEMO4 Finding 4] extract thumbnails while the upload still exists — MUST run before
            # the finally-block unlink below.
            try:
                _extract_frame_thumbnails(vid, json.loads(trace_path.read_text()).get("records", []), run_id)
            except Exception:
                pass
        finally:
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
            "seams": result.get("seams")}


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
            return self._send(200, Path(run["trace"]).read_bytes(), "application/json")
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
        if path != "/run":
            return self._send(404, json.dumps({"error": "not found"}))
        params = parse_qs(urlparse(self.path).query)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            result = _do_run(params, body)
            code = 400 if result.get("error") else 200
            return self._send(code, json.dumps(result))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": "run_failed", "message": str(e)}))


INDEX = """<!doctype html><meta charset=utf-8><title>Percept demos</title>
<style>body{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 20px;color:#0a0a0a}
h1{font-family:Arial Narrow,sans-serif;letter-spacing:.02em}code{background:#f4f4f0;padding:2px 5px;font-size:13px}
a{display:block;border:1px solid #0a0a0a;padding:14px 16px;margin:12px 0;color:#0a0a0a;text-decoration:none}a:hover{background:#faf7ee}
.k{font:11px ui-monospace,monospace;color:#8a8a8a;margin-top:24px}</style>
<h1>Percept demo harness</h1>
<p>Real <code>watch()</code> runtime behind both pages. Uploads run a real judge when an API key is set;
otherwise the keyless fixture runs the real runtime with scripted verdicts.</p>
<a href="/demo-01-video-watch.html">Demo 01 · MP4 video watch →</a>
<a href="/demo-02-screen-watch.html">Demo 02 · Screen-share watch →</a>
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
