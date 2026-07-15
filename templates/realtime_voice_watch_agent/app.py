"""Realtime voice watch agent — the composition root. Public Percept seams ONLY.

    browser (prebuilt WebRTC client)                     server (this file)
      mic  ── AEC native ──►  SmallWebRTC P2P ──►  Gemini Live (S2S: hears+reasons+speaks)
      speaker  ◄──────────────────────────────────    │ watch/stop_watching/whats_armed tools
      camera ─────────────►  video frames ────────►   │ percept_state (grounding, PULL-only)
                                                    percept.start(core, source=video tee)
                                                    incidents → AttentionIntent → Floor → speech

Everything Percept-facing goes through the public runner (``percept.start``) + ``RuntimeSession``
seams; the floor/renderer/telemetry are application policy and live beside this file, not in the SDK.
Heavy voice deps (pipecat) are imported inside :func:`main`'s path so :func:`build_agent` — the real
wiring — is importable keyless (the offline E2E test drives it with a fake engine).

Run:  pip install -e .  &&  python app.py   (see README.md)
Keys: GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_AISTUDIO_API_KEY — from the shell env, or from the
nearest ``.env`` walking up from the working directory (a repo-root ``.env`` Just Works).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import percept


def _load_dotenv() -> None:
    """Nearest ``.env`` walking up from cwd; real environment always wins (never overridden)."""
    for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        cand = parent / ".env"
        if cand.exists():
            for line in cand.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
            return

from engagement import Engagement, EngagementConfig
from fire_renderer import FireRenderer
from floor import Floor
from percept_bridge import PerceptBridge
from telemetry import VoiceTelemetry

SYSTEM = (
    "You are Percept, a warm, low-key ambient assistant. You can SEE the user's camera live, so answer "
    "questions about what's visible directly and briefly. If the user asks you to watch for / alert them "
    "when something happens, call the `watch` tool with their request verbatim. Keep spoken replies to "
    "one or two sentences."
)


@dataclass
class Agent:
    """Everything :func:`build_agent` wired for one connection (the offline E2E drives this too)."""
    core: Any
    engine: Any
    session: Any
    floor: Floor
    bridge: PerceptBridge
    renderer: FireRenderer
    telemetry: VoiceTelemetry
    stats: Any = field(default_factory=percept.RuntimeStats)

    async def stop(self) -> None:
        self.bridge.detach()
        await self.session.stop()


async def build_agent(core: Any, engine: Any, *, source: Any, goals: tuple = (),
                      webhook_sink: Any = None, policy: Any = None,
                      clock: Any = time.monotonic, **start_kwargs: Any) -> Agent:
    """The REAL wiring, engine-agnostic: floor + renderer + telemetry + bridge over ``percept.start``."""
    stats = percept.RuntimeStats()
    telemetry = VoiceTelemetry(clock=clock)
    renderer = FireRenderer(engine, webhook_sink=webhook_sink, telemetry=telemetry, clock=clock)
    floor = Floor(engine, renderer, clock=clock, telemetry=telemetry)

    # Engine output edges: wire_speaking (floor) + the honest A2V close (telemetry) + fire_playing
    # clear (renderer). Registration is ADDITIVE on the engine — order-independent subscriber lists.
    def _output_started() -> Any:
        telemetry.on_output_started()
        return floor.set_wire_speaking(True)      # awaited by the engine's fan-out

    engine.on_output_started(_output_started)
    engine.on_output_stopped(lambda: floor.set_wire_speaking(False))
    renderer.bind_engine_edges()                  # output-stopped also clears fire_playing
    # The engine's synthesized turn-end (transcript-derived) also flushes queue-behind-turn fires;
    # the local energy VAD (media.AudioSplitter → floor.set_user_speaking) stays the barge-in source.
    engine.on_user_speaking(floor.set_user_speaking)

    # The watch: the PUBLIC runner. Incidents route through the bridge (late-bound: the bridge needs
    # the session this call creates). No private seams — frames fan out via session.subscribe().
    holder: dict = {}

    async def _on_incident(incident: Any) -> None:
        await holder["bridge"].on_incident(incident)

    session = await percept.start(core, source=source, on_incident=_on_incident,
                                  stats=stats, **start_kwargs)
    bridge = PerceptBridge(core, session, floor, renderer, policy=policy, telemetry=telemetry)
    holder["bridge"] = bridge
    for goal in goals:                            # config/app goals: structured → STRUCTURED_TRUSTED
        await bridge.arm(goal)
    bridge.attach(engine)                         # tools + transcript tee + guard, before engine.start()
    return Agent(core=core, engine=engine, session=session, floor=floor, bridge=bridge,
                 renderer=renderer, telemetry=telemetry, stats=stats)


def load_config(path: str | Path = "config.yaml") -> dict:
    """Optional YAML config beside this file (see config.example.yaml); absent → all defaults."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    if not p.exists():
        return {}
    import yaml

    return yaml.safe_load(p.read_text()) or {}


def _config_goals(cfg: dict) -> tuple:
    out = []
    for g in cfg.get("goals", []) or []:
        goal = percept.Goal(id=g["id"], condition=g["condition"], say=g.get("say", ""))
        if g.get("delivery"):
            goal.state["delivery"] = dict(g["delivery"])
        if g.get("cadence"):
            goal.state["cadence"] = dict(g["cadence"])   # e.g. gate_on: fixed for static conditions
        out.append(goal)
    return tuple(out)


def main() -> None:
    # Heavy voice deps enter HERE, not at module import (keyless tests import build_agent above).
    from engine_adapter import PipecatVoiceEngine
    import media

    _load_dotenv()
    cfg = load_config(os.environ.get("PERCEPT_TEMPLATE_CONFIG", "config.yaml"))
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
           or os.getenv("GOOGLE_AISTUDIO_API_KEY") or "")
    if not key:
        print("[app] no GEMINI_API_KEY — set one in the shell or a .env up-tree "
              "(the wire and the watch both use Gemini)", file=sys.stderr)
    else:
        # The Gemini SDKs read GOOGLE_API_KEY/GEMINI_API_KEY; normalize so an
        # AISTUDIO-named key in .env reaches them without a rename.
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("GOOGLE_API_KEY", key)
    engagement = Engagement(EngagementConfig(mode=cfg.get("engagement", "always_open")))
    webhook_url = cfg.get("webhook_url")

    async def on_connection(connection: Any) -> None:
        """A browser joined: one core + one engine + one agent for this call."""
        core = percept.Percept.create(vision=cfg.get("vision", "gemini"),
                                      conductor=cfg.get("conductor", "gemini"),
                                      discover_plugins=False)
        tee = media.VideoTee()
        splitter = media.AudioSplitter()
        engine = PipecatVoiceEngine(connection, gemini_api_key=key, model=cfg.get("model"),
                                    voice=cfg.get("voice", "Aoede"), system=cfg.get("system", SYSTEM),
                                    on_video_frame=tee.on_video_frame,
                                    on_audio_frame=splitter.on_audio_frame)
        webhook_sink = None
        if webhook_url:
            from percept.runtime.sinks import webhook
            webhook_sink = webhook(webhook_url)
        agent = await build_agent(core, engine, source=tee.source(),
                                  goals=_config_goals(cfg), webhook_sink=webhook_sink)
        # Three-way audio split (media.py topology): VAD → floor (always live); world audio →
        # core.push_audio, suppressed ONLY during local playout; the engine branch is untouched.
        splitter.bind(on_user_speaking=agent.floor.set_user_speaking,
                      push_world_audio=lambda chunk, t: core.push_audio(chunk, t=t),
                      suppressed=lambda: agent.floor.wire_speaking or agent.renderer.fire_playing)
        await engine.start()
        print(f"[app] call up (engagement={engagement.config.mode}); "
              f"goals: {[g['id'] for g in agent.session.list_goals()] or 'none — say “watch …”'}")

        async def heartbeat() -> None:   # the scheduler proof: capture never blocks on a judge
            while True:
                await asyncio.sleep(3.0)
                s = agent.stats
                print(f"[watch] frames={s.frames} escalations={s.escalations} judge_calls={s.judge_calls} "
                      f"superseded={getattr(s, 'judge_superseded', 0)} fires={s.fires} "
                      f"| voice={agent.telemetry.snapshot().get('fires.delivered.log', 0)} logs")

        hb = asyncio.ensure_future(heartbeat())

        @connection.event_handler("closed")
        async def _on_closed(_conn: Any) -> None:  # noqa: ANN001
            hb.cancel()
            await tee.close()
            await agent.stop()
            await engine.stop()
            print("[app] call ended, watch stopped")

    media.serve(on_connection, host=cfg.get("host", "127.0.0.1"), port=int(cfg.get("port", 8930)))


if __name__ == "__main__":
    main()
