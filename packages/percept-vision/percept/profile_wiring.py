"""Resolve a :class:`~percept.watchprofile.WatchProfile` into the runtime's REAL knobs.

``percept/watchprofile.py`` shipped the SHAPE of ``Q(p | environment, goal, state)`` but wired
nothing: "no line in ``agent.py`` / ``runtime/worker.py`` reads a WatchProfile yet". This module is that
wiring — the ONE seam that turns a ``WatchProfile`` into the concrete objects the rest of the codebase
already reads:

- ``salience_stat`` / ``stat_thresholds`` -> :class:`~percept.config.EdgeSettings`
  (``salience_stat``, ``motion_gate``/``motion_gate_block_max``/``motion_gate_p95``) — the exact fields
  :func:`percept.harness.tier0.compile_watchspec` and ``runtime/worker.py`` already read off
  ``Percept.settings.edge``.
- ``gate.require_false_first`` / ``gate.refractory_s`` -> :class:`~percept.config.GateSettings` — the
  exact fields :meth:`percept.agent.Percept.create` already reads to build the :class:`~percept.gate.Gate`.
- ``cadence`` -> a real :class:`~percept.sampling.SensePolicy` (via
  :meth:`~percept.watchprofile.CadenceProfile.to_sense_policy`), applied at CORE CREATION time (an
  immutable field on the built :class:`~percept.agent.Percept`, never mutated mid-run).
- ``plan_mode`` -> threaded through unchanged (a plain string) onto the core, where it becomes LIVE:
  :func:`percept.agent._planned_perception` floors every concern's ``PerceptionSpec.temporal`` with it on
  the judging path, so ``plan_mode="window"`` assembles an ordered frame window and calls
  ``judge(frames=[...])``. ``"sprite"`` is deprecated there and shimmed to ``"window"``.

Every value this module produces is either an attribute READ off the ``WatchProfile`` or a field carried
over from the base :class:`~percept.config.Settings` — there is no knob literal in this file. That is the
architectural rule this module enforces: *the runtime reads its behavior from a profile; the harness (or, for
now, this resolver) writes profiles. No knob is hardcoded in firing logic* (see
``tests/test_no_hardcoded_knobs.py``).

``profile=None`` never reaches this module — :func:`percept.watch._settings_from_profile` and
``watch()`` keep that case on the legacy path (``load_settings()``, env-var driven), byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import Settings, load_settings
from .sampling import SensePolicy
from .watchprofile import WatchProfile


@dataclass(frozen=True)
class ResolvedProfile:
    """The concrete runtime knobs a :class:`WatchProfile` resolves to — everything
    :meth:`percept.agent.Percept.create` needs to apply the profile at construction time."""

    settings: Settings
    sense_policy: SensePolicy
    plan_mode: str


def resolve_watch_profile(profile: WatchProfile, *, base: Settings | None = None) -> ResolvedProfile:
    """Resolve ``profile`` into a :class:`ResolvedProfile` — real ``Settings``, a real ``SensePolicy``,
    and the plan mode string.

    ``base`` seeds every field a ``WatchProfile`` does NOT own (cache, models, routing, identity,
    episodes, events, consent, and the gate/edge fields the profile itself doesn't carry) — it defaults
    to :func:`~percept.config.load_settings` (today's env-driven defaults), so a profile only overrides
    the specific knobs empirical tuning found; it never silently resets an operator's env-configured
    values it has no opinion on.
    """
    base_settings = base if base is not None else load_settings()

    edge = replace(
        base_settings.edge,
        salience_stat=profile.salience_stat,
        motion_gate=profile.stat_thresholds.get("mean", base_settings.edge.motion_gate),
        motion_gate_block_max=profile.stat_thresholds.get(
            "block_max", base_settings.edge.motion_gate_block_max
        ),
        motion_gate_p95=profile.stat_thresholds.get("p95", base_settings.edge.motion_gate_p95),
    )
    gate = replace(
        base_settings.gate,
        require_false_first=profile.gate.require_false_first,
        refractory_s=profile.gate.refractory_s,
    )
    settings = replace(base_settings, edge=edge, gate=gate)

    return ResolvedProfile(
        settings=settings,
        sense_policy=profile.cadence.to_sense_policy(),
        plan_mode=profile.plan_mode,
    )


__all__ = ["ResolvedProfile", "resolve_watch_profile"]
