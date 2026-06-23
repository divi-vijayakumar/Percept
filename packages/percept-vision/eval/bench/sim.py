"""Synthetic E2E proactivity simulator.

Drives the REAL cognition pipeline (FakeVision -> three-state gate -> executor -> fire) with scripted
ground-truth, so CueSpeak (interjection quality) and TimeSpeak (timeliness) get exact, deterministic
numbers without any video or VLM. "We can simulate e2e" because the gate + executor are the real code;
only PERCEPTION is scripted — which is exactly the layer a synthetic stream is allowed to stand in for.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision

try:
    from percept.config import load_settings
except Exception:  # pragma: no cover
    load_settings = None


@dataclass
class Episode:
    onset: int
    duration: int


def gate_threshold() -> float:
    """The confidence the gate needs to FIRE (vs ASK). Read from settings; fall back to 0.5."""
    if load_settings is not None:
        try:
            return float(load_settings().gate.threshold)
        except Exception:
            pass
    return 0.5


def token(t: int) -> str:
    return f"f{int(t)}"


def build_stream(episodes, T, *, peak_conf=0.95, ramp=0, noise_spikes=()):
    """Build a scripted stream. The cue is True during each episode; confidence ramps from 0.2 -> peak
    over ``ramp`` frames at onset (ambiguous -> clear). ``noise_spikes`` = [(t, conf)] spurious cue-True
    frames outside episodes. Returns ``(script, frames)`` for the agent + ``raw_fires``.
    """
    verdict_at: dict[int, Verdict] = {}
    for ep in episodes:
        for k in range(ep.duration):
            t = ep.onset + k
            conf = peak_conf if (not ramp or k >= ramp) else 0.2 + (peak_conf - 0.2) * (k + 1) / ramp
            verdict_at[t] = Verdict(satisfied=True, confidence=round(conf, 4), evidence=f"cue@{t}")
    for (t, conf) in noise_spikes:
        verdict_at[int(t)] = Verdict(satisfied=True, confidence=conf, evidence=f"noise@{t}")
    script, frames = {}, []
    for t in range(T):
        v = verdict_at.get(t, Verdict(satisfied=False, confidence=0.9, evidence="clear"))
        script[token(t)] = v
        frames.append((token(t), float(t)))
    return script, frames


def run_core(script, frames, *, condition="the cue is happening now", say="heads up"):
    """Drive the REAL agent through the stream; return fires as ``[(action, t)]``, action in fire|ask."""
    agent = Percept.create(vision=FakeVision(script=script), discover_plugins=False)
    agent.add_goal(Goal(id="cue", condition=condition, say=say))
    out: list[tuple[str, float]] = []

    async def go():
        for tok, t in frames:
            for ev in await agent.perceive_judged(tok, t=t):
                out.append((ev.action, t))

    asyncio.run(go())
    return out


def raw_fires(script, frames):
    """The naive VLM-in-a-loop baseline: fire on EVERY frame the VLM calls satisfied — no three-state
    gate, no rising edge, no refuse-to-guess. The control the cognition layer is measured against."""
    return [("fire", t) for (tok, t) in frames if script[tok].satisfied]


def conf_at(script, t) -> float:
    return script[token(t)].confidence
