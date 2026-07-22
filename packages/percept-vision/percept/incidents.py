"""Incident producer — turn a fire (a :class:`FireEvent`) into a durable :class:`Incident` record.

The `Incident`/`IncidentSnippet` DTOs live in `percept.contracts` but had no producer; this is it. The
runtime worker calls :func:`make_incident` on every fire and hands the result to its `on_incident` sink.
`fire` is duck-typed (``.goal_id`` / ``.text`` / ``.verdict``) so this module never imports the agent
(no cycle). Pure stdlib.
"""
from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from percept.contracts import AttentionIntent, Consequence, GoalOrigin, Incident, IncidentSnippet, Verdict

from .concern import consequence_of


def make_incident(
    fire: Any,
    *,
    ts: float,
    snippet: IncidentSnippet | None = None,
    id: str | None = None,
    entity_id: str | None = None,
    watchplan_id: str | None = None,
    shadow: Any = None,
) -> Incident:
    """Map a fire to an :class:`Incident`. ``id`` defaults to a deterministic, sortable handle.

    The fire's ``verdict`` carries the VLM's evidence; ``text`` (the goal's ``say``) becomes the
    human-facing ``message``. ``label`` stays ``None`` until a human / the calibration loop classifies it.

    ``entity_id`` (contract 1.3) carries the gated subject's correlation handle THROUGH the
    emission boundary. The gate key is already ``(goal_id, entity_id)``, so identity was computed
    upstream and then dropped here; a consumer that cannot tell two subjects apart reads their
    interleaved fires as out-of-order divergence. Taken from the fire's own ``entity_id`` (duck-typed,
    like every other field this module reads) unless the caller overrides it explicitly. ``None`` when
    the fire was not entity-scoped — the back-compatible default.
    """
    verdict = getattr(fire, "verdict", None) or Verdict(satisfied=True, confidence=0.0, evidence="")
    goal_id = getattr(fire, "goal_id", "")
    if entity_id is None:
        entity_id = getattr(fire, "entity_id", None)
    # The fire's JudgmentEvent (when the fire carried one) is the source of truth for evidence_seq — the
    # per-key ADR-001 §8 dispatch sequence. A fire from a path that takes no judge-funnel stamps
    # (episode/rhythm/boundary/transition/absence) carries no ``.judgment``, so this stays None rather
    # than a fabricated number.
    judgment = getattr(fire, "judgment", None)
    evidence_seq = getattr(judgment, "evidence_seq", None) if judgment is not None else None
    # The episode-owned provisional/superseding link, read once off the fire and used for BOTH the
    # 1.8 wire fields below and the runtime-local mirror further down.
    revision_id = getattr(fire, "revision_id", None)
    supersedes = getattr(fire, "supersedes", None)
    incident = Incident(
        id=id or f"inc-{goal_id}-{ts:.3f}-{uuid.uuid4().hex[:8]}",
        concern_id=goal_id,
        verdict=verdict,
        ts=float(ts),
        snippet=snippet,
        message=getattr(fire, "text", "") or "",
        entity_id=str(entity_id) if entity_id is not None else None,
        reason=getattr(fire, "reason", None),
        watchplan_id=watchplan_id if watchplan_id is not None else getattr(fire, "watchplan_id", None),
        roi=getattr(fire, "roi", None),
        modality=getattr(fire, "modality", None),
        provenance=getattr(fire, "provenance", None),
        evidence_seq=evidence_seq,
        revision_id=revision_id,
        supersedes=supersedes,
    )
    # ALSO mirror the link as RUNTIME-LOCAL annotations (private instance attrs, exactly like
    # Incident's bound label state) — kept for the existing ``SupersessionSink`` delivery-boundary
    # resolution (fold revision -> prior INCIDENT id) and any in-process consumer already reading
    # :func:`incident_revision` / :func:`incident_supersedes`. Redundant with the 1.8 wire fields
    # above by design: this mirror predates them and nothing here changes its meaning.
    incident._revision_id = revision_id  # type: ignore[attr-defined]
    incident._supersedes = supersedes  # type: ignore[attr-defined]
    # The SHADOW advisory (resolved subject + proposed EpisodeFoldResult), attached as a
    # RUNTIME-LOCAL annotation exactly like the revision link above: the WIRE Incident stays frozen (no
    # schema bump, no webhook-payload change), and a consumer that ignores it sees byte-for-byte
    # today's Incident. It GOVERNS NOTHING — it records what the fold WOULD have proposed. Taken from
    # the ``shadow`` kwarg, else the fire's own ``shadow`` (duck-typed) if it carries one. Stored as
    # its JSON-safe advisory dict (``ShadowObservation.to_advisory()``) or a raw dict; read via
    # :func:`incident_shadow`.
    shadow_val = shadow if shadow is not None else getattr(fire, "shadow", None)
    if shadow_val is not None and hasattr(shadow_val, "to_advisory"):
        shadow_val = shadow_val.to_advisory()
    incident._shadow = shadow_val  # type: ignore[attr-defined]
    return incident


def incident_revision(incident: Any) -> str | None:
    """This incident's stable episode-revision id (``<episode_id>#r<n>``), or ``None``.

    The runtime-local annotation :func:`make_incident` copies off the fire. ``None`` for an
    incident that carried no episode revision (an unbound/ambiguous or older fire)."""
    return getattr(incident, "_revision_id", None)


def incident_supersedes(incident: Any) -> str | None:
    """The PRIOR revision id this incident supersedes (a replace/retract correction), or ``None`` for
    an original ("provisional") incident. Runtime-local annotation."""
    return getattr(incident, "_supersedes", None)


def incident_shadow(incident: Any) -> dict | None:
    """The SHADOW advisory (``{"subject": ..., "fold": ...}``) attached to this incident, or
    ``None``. Runtime-local annotation — ADVISORY ONLY, it governs nothing. Reconstruct the typed
    :class:`~percept.behavior.ShadowObservation` with ``ShadowObservation.from_advisory(...)``."""
    return getattr(incident, "_shadow", None)


def _encode_frame_bundle(frames: list, *, t_start: float, t_end: float) -> tuple[str, bytes]:
    bundle = {"version": 1, "t_start": t_start, "t_end": t_end, "frames": []}
    for frame in frames:
        if (
            isinstance(frame, tuple)
            and len(frame) == 2
            and isinstance(frame[0], str)
            and isinstance(frame[1], (bytes, bytearray))
        ):
            bundle["frames"].append(
                {"mime": frame[0], "data": base64.b64encode(bytes(frame[1])).decode("ascii")}
            )
        else:
            bundle["frames"].append({"repr": repr(frame)})
    return "application/x-percept-frame-bundle+json", json.dumps(bundle).encode("utf-8")


def snippet_from_window(window: Any, *, kind: str = "frame", pre_s: float | None = None) -> IncidentSnippet | None:
    """Assemble an :class:`IncidentSnippet` from a rolling :class:`~percept.events.FrameWindow`.

    Grabs the buffered ``(t, frame)`` pairs as the pre-roll (the lead-up to the fire). ``pre_s`` bounds it
    by time (last ``pre_s`` seconds); ``None`` takes the whole window. ``kind="frame"`` keeps the decoded
    frames in ``snippet.frames`` (no encoding). ``kind="clip"`` stores a portable frame-bundle in
    ``snippet.media`` so webhook/storage sinks have one bytes payload without requiring ffmpeg.
    Returns ``None`` for an empty window.
    """
    frames = window.frames() if hasattr(window, "frames") else list(window)
    if not frames:
        return None
    t_end = frames[-1][0]
    if pre_s is not None:
        cutoff = t_end - pre_s
        frames = [(t, f) for (t, f) in frames if t >= cutoff]
    if not frames:
        return None
    payload_frames = [f for (_t, f) in frames]
    t_start, t_end = frames[0][0], frames[-1][0]
    if kind == "clip":
        mime, media = _encode_frame_bundle(payload_frames, t_start=t_start, t_end=t_end)
        return IncidentSnippet(kind="clip", media=media, mime=mime, t_start=t_start, t_end=t_end)
    return IncidentSnippet(kind=kind, frames=payload_frames, t_start=t_start, t_end=t_end)


# ---------------------------------------------------------------------------
# Attention policy — turn a fire + its goal into the EFFECTIVE :class:`AttentionIntent`.
#
# This lives here (not in the frozen contracts membrane) because it is *behaviour*, not a wire shape:
# it consumes an :class:`Incident` + a :class:`Goal`'s durable ``origin``/consequence and returns the
# clamped intent. Keeping the policy beside the incident producer keeps the contract package pure and
# puts the requested->effective decision in one deterministic, testable place (spec §2.2).
# ---------------------------------------------------------------------------

_Preempt = Literal["interrupt", "queue", "drop"]


@dataclass(frozen=True)
class RequestedDelivery:
    """The AUTHORED delivery request, parsed out of the legacy ``goal.state['delivery']`` dict.

    This is what the goal's author *asked for* — distinct from the effective intent, which is what the
    policy *grants* after clamping. Defaults reproduce the legacy ``_speak_fire`` semantics: an absent
    ``preempt`` means "hold behind the current turn" (``"queue"``), never a silent interrupt."""

    preempt: _Preempt = "queue"
    ttl_s: float | None = None
    verbatim: bool = False
    preferred_channel: str | None = None


def parse_requested_delivery(delivery: Any) -> RequestedDelivery:
    """Interpret the legacy ``delivery`` dict (``goal.state['delivery']``) into a :class:`RequestedDelivery`.

    The SINGLE interpreter of that dict — matches the semantics ``RuntimeSession._speak_fire`` read
    inline: ``preempt`` is a bare string (only ``"interrupt"`` cut into a turn; anything else, including
    a missing key, held behind it), ``ttl_s`` a float-or-None drop-if-stale bound, ``verbatim`` a bool.
    ``preferred_channel`` is the additive, optional app hint (absent in the legacy dict -> ``None``).

    Unknown/garbage ``preempt`` values collapse to the safe ``"queue"`` (never an accidental interrupt);
    a non-dict / ``None`` ``delivery`` yields all defaults."""
    d = delivery if isinstance(delivery, dict) else {}
    raw = d.get("preempt")
    preempt: _Preempt = raw if raw in ("interrupt", "queue", "drop") else "queue"
    ttl = d.get("ttl_s")
    ttl_s = None if ttl is None else float(ttl)
    channel = d.get("preferred_channel")
    return RequestedDelivery(
        preempt=preempt,
        ttl_s=ttl_s,
        verbatim=bool(d.get("verbatim", False)),
        preferred_channel=None if channel is None else str(channel),
    )


@dataclass(frozen=True)
class AttentionPolicy:
    """Deterministic requested->effective attention clamp (spec §2.2). No I/O, no clock, no RNG.

    The one rule that matters: a requested ``preempt="interrupt"`` is CLAMPED to ``"queue"`` unless the
    request carries earned authority. Authority comes ONLY from:

    - a TRUSTED :class:`GoalOrigin` — ``STRUCTURED_TRUSTED`` (armed in code) or ``SYSTEM_TEMPLATE`` (a
      trusted first-party template). A ``LANGUAGE_AUTHORED`` goal (the untrusted conductor) never earns
      it by asking; a ``RESTORED`` goal is neutral;
    - an allowlisted consequence CLASS — the goal's ``miss_cost`` is in ``allow_interrupt_consequences``
      (e.g. a deployment that lets ``"severe"``-miss watches interrupt regardless of who authored them);
    - an explicit per-goal allowlist entry — ``goal_id`` in ``allow_interrupt_goal_ids``.

    Both allowlists default EMPTY, so the out-of-the-box policy grants interrupt on trusted origin alone.
    ``"queue"``/``"drop"`` requests pass through untouched — the policy only ever *reduces* authority, it
    never elevates a request the author did not make."""

    allow_interrupt_consequences: frozenset[str] = field(default_factory=frozenset)
    allow_interrupt_goal_ids: frozenset[str] = field(default_factory=frozenset)

    def _may_interrupt(self, origin: GoalOrigin, consequence: Consequence | None, incident: Any) -> bool:
        if origin in (GoalOrigin.STRUCTURED_TRUSTED, GoalOrigin.SYSTEM_TEMPLATE):
            return True
        if consequence is not None and getattr(consequence, "miss_cost", None) in self.allow_interrupt_consequences:
            return True
        gid = getattr(incident, "concern_id", None)
        return gid is not None and gid in self.allow_interrupt_goal_ids

    def clamp(
        self,
        *,
        requested: RequestedDelivery,
        origin: GoalOrigin,
        consequence: Consequence | None,
        incident: Any,
    ) -> AttentionIntent:
        """Fold a :class:`RequestedDelivery` + provenance + consequence + the fired incident into the
        EFFECTIVE :class:`AttentionIntent`. The only field the policy may change is ``preempt`` (an
        unauthorized ``"interrupt"`` becomes ``"queue"``); ttl/verbatim/channel pass through verbatim."""
        preempt = requested.preempt
        if preempt == "interrupt" and not self._may_interrupt(origin, consequence, incident):
            preempt = "queue"
        return AttentionIntent(
            message=getattr(incident, "message", "") or "",
            goal_id=getattr(incident, "concern_id", "") or "",
            incident_id=getattr(incident, "id", "") or "",
            ts=getattr(incident, "ts", None),
            preempt=preempt,
            ttl_s=requested.ttl_s,
            verbatim=requested.verbatim,
            preferred_channel=requested.preferred_channel,
        )


#: The out-of-the-box policy: interrupt-authority from a trusted origin only (empty allowlists).
DEFAULT_ATTENTION_POLICY = AttentionPolicy()


def attention_intent_for(incident: Any, goal: Any, policy: AttentionPolicy | None = None) -> AttentionIntent:
    """Map a fired ``incident`` + its ``goal`` to the EFFECTIVE :class:`AttentionIntent` (§2.2).

    The application seam calls this instead of reading ``goal.state['delivery']`` itself — so the
    requested->effective clamp (a conductor cannot self-grant an interrupt) is enforced in ONE place.
    ``goal`` supplies the durable ``origin`` (interrupt-trust) and consequence (derived via
    :func:`~percept.concern.consequence_of`); ``incident`` supplies id/ts/message/goal_id. ``policy``
    defaults to :data:`DEFAULT_ATTENTION_POLICY` (trusted-origin-only interrupt authority)."""
    pol = policy if policy is not None else DEFAULT_ATTENTION_POLICY
    requested = parse_requested_delivery((getattr(goal, "state", None) or {}).get("delivery"))
    origin = getattr(goal, "origin", GoalOrigin.LANGUAGE_AUTHORED)
    consequence = consequence_of(goal) if goal is not None else None
    return pol.clamp(requested=requested, origin=origin, consequence=consequence, incident=incident)
