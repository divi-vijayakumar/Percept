"""Episode / entity-memory judge mode — the deferred arm.

An ablation study proved the **eye** (localized salience) is the biggest recall lever and the
**ordered window** (temporal input, oldest->newest) the next one. The remaining, still-unproven lever
is **episode memory**: instead of re-deciding a persistent/temporal goal from a single frame (or even
a fixed window), thread a per-goal RUNNING ANSWER across successive judged frames so the verdict
integrates evidence accumulated over the goal's whole lifetime, not just the last N frames.

The mechanism is already partly wired: every production ``Vision`` backend implements a process-QA
step (``context={"process_qa": True, "running_answer": <prior>}`` -> ``Verdict.running_answer``, see
``backends/openrouter.py:_judge_process_qa`` / ``backends/gemini.py:_judge_process_qa``) and
``agent.py`` already threads it for goals that opt into ``state["process_qa"]``. This module
PRODUCTIZES that mechanism as a standalone, named judge mode:

  * ``EpisodeMemoryState`` / ``EpisodeMemoryStore`` — the per-goal running state, decoupled from the
    agent's ``Executor``/``ConcernContext`` so this mode is usable on its own (eval harness ablation,
    a future WatchProfile-driven runtime path) without touching the executor.
  * ``judge_episode_memory`` — one judge step: thread the prior answer IN, fold the model's rewritten
    answer OUT, and — the productization over the raw process-QA plumbing — return a ``Verdict`` whose
    ``satisfied`` reflects INTEGRATED evidence (accumulated coverage + substance + confidence, via the
    existing :func:`~percept.concern.resolution_ready` bar) rather than one frame's read alone.
  * ``episode_memory_enabled`` — reads an opt-in ``goal.state["judge_mode"]`` flag, the seam a future
    ``WatchProfile`` can set to select this mode per (environment, goal-type, state).

The OTHER half of the memory — and the half that IS live
--------------------------------------------------------
The running answer above carries what the MODEL said. What the SYSTEM already established lives in
the concern's :class:`~percept.episodes.EpisodeLog`, and until the producer below existed nothing
ever put it into a model call: a concern could hold an open episode for hours while the judge
re-answered "is it still there?" every tick from a single frame, with no memory of ever having seen
the thing. ``episode_judge_context`` / ``with_open_episode`` serialize that open episode under the
reserved :data:`~percept.judge_context.JUDGE_OPEN_EPISODE_KEY`, and ``agent.py``'s judge dispatch
site calls them on every judged tick. A goal with no open episode gets no key and is byte-for-byte
unchanged, so this is inert for every concern that does not own episodes.

Because that injection makes the verdict depend on state no frame hash can see, this module also
mints the token the perception cache KEYS that dependency on —
``episode_context_and_fingerprint`` — rather than letting the presence of a context switch the cache
off. See the section header above :func:`episode_context_and_fingerprint` for the split it rests on
and the measurement that forced it.

The judge-MODE half above remains off by default: selecting it per (environment, goal-type, state)
from a ``WatchProfile`` and registering it with the swappable judge seam are still follow-up work.
``gate.py`` is untouched either way — nothing in this module decides anything, it only tells the
judge what is already known.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .adapters import Goal, Verdict
from .concern import resolution_ready
from .episodes import (
    JUDGE_CONTEXT_MAX_REVISIONS,
    JUDGE_CONTEXT_MAX_VALUE_CHARS,
    Episode,
    EpisodeLog,
)
from .judge_context import JUDGE_OPEN_EPISODE_KEY, ReservedContextKeyError

# The plan_mode/config value a WatchProfile will set to select this judge mode (future). A goal
# that never sets ``judge_mode`` is untouched by this module — no behavior changes by default.
JUDGE_MODE_EPISODE_MEMORY = "episode_memory"

# Bounded per-goal history — mirrors the existing "a 24/7 concern must not leak memory" discipline
# (agent.py's ``ctx.process_log`` cap).
_HISTORY_CAP = 256
# Bounded number of concurrently-tracked goals in a store (same discipline, applied per-store).
_STORE_CAP = 256


def episode_memory_enabled(goal: Goal) -> bool:
    """Has this goal opted into the episode-memory judge mode?

    Reads ``goal.state["judge_mode"]`` — the flag a future ``WatchProfile`` resolves per
    (environment, goal-type, state). Read-only: this does not change what ``agent.py`` does with
    the goal today, it only exposes the seam so a WatchProfile-driven caller can select this mode."""
    return goal.state.get("judge_mode") == JUDGE_MODE_EPISODE_MEMORY


@dataclass
class EpisodeMemoryState:
    """A single goal's running answer, threaded across successive judged frames (the episode memory).

    ``frame_count`` only increments on EVIDENCE-BEARING frames (the model returned a non-empty rewritten
    answer) — it is the coverage count :func:`~percept.concern.resolution_ready` reads, mirroring
    ``len(ConcernContext.process_log)`` in the existing process-QA path."""

    running_answer: str = ""
    history: list[str] = field(default_factory=list)
    frame_count: int = 0

    def record(self, answer: str) -> None:
        """Fold a new evidence-bearing answer into the running state. Empty/blank answers are a no-op
        (a parse miss on the backend side preserves the prior answer — never regress the accumulation)."""
        answer = (answer or "").strip()
        if not answer:
            return
        self.running_answer = answer
        self.history.append(answer)
        overflow = len(self.history) - _HISTORY_CAP
        if overflow > 0:
            del self.history[:overflow]
        self.frame_count += 1

    def reset(self) -> None:
        """End the episode: wipe the running answer and coverage (a new episode starts clean)."""
        self.running_answer = ""
        self.history = []
        self.frame_count = 0


class EpisodeMemoryStore:
    """Bounded per-goal store of :class:`EpisodeMemoryState`. Decoupled from the agent's ``Executor``
    so this mode threads state on its own — usable directly (eval harness, tests) without a running
    :class:`~percept.agent.Percept` instance."""

    def __init__(self, *, max_goals: int = _STORE_CAP) -> None:
        self._states: dict[str, EpisodeMemoryState] = {}
        self._max_goals = max_goals

    def get(self, goal_id: str) -> EpisodeMemoryState:
        """Fetch (creating if absent) the running state for ``goal_id``. Evicts the oldest-inserted
        goal once the bound is hit — the same "24/7 agent can't leak memory" discipline as elsewhere."""
        state = self._states.get(goal_id)
        if state is None:
            if len(self._states) >= self._max_goals:
                self._states.pop(next(iter(self._states)))
            state = self._states[goal_id] = EpisodeMemoryState()
        return state

    def reset(self, goal_id: str) -> None:
        """Drop a goal's running state (episode boundary / goal completion)."""
        self._states.pop(goal_id, None)

    def __len__(self) -> int:
        return len(self._states)


# ==================================================================================================
# THE OPEN EPISODE, AS THE JUDGE SEES IT
# ==================================================================================================
#
# The running answer above threads what the MODEL said. This half threads what the SYSTEM already
# established: when a concern has an open episode, the judge should be told about it instead of
# re-deciding "is the bag still there?" from a single frame with no memory of ever having seen it.
#
# The serialized form is deliberately a fixed-size TEXT, not the episode object:
#
#   * pixel-free — built from :meth:`percept.episodes.Episode.to_judge_dict`, which drops the
#     ``exemplar``/``representative`` payloads outright;
#   * constant-size — an episode confirmed 400 times serializes to the same length as one confirmed
#     twice (confirmations coalesce into scalars, and the timeline contributes at most
#     ``JUDGE_CONTEXT_MAX_REVISIONS`` rows), and a concern holding many open episodes contributes ONE
#     of them plus a count, never a list;
#   * text — every shipped judge adapter already surfaces an unrecognized context key to the model as
#     a structured fact, so a string reaches the prompt on all of them with no per-backend prompt
#     work. The same property is what makes the static-context key work.

#: Hard ceiling on the whole serialized episode, applied last, after every per-field bound. A
#: backstop: with the field bounds in place the text lands well under this, so reaching it means a
#: caller-owned value was pathological, and clipping is the right answer.
EPISODE_CONTEXT_MAX_CHARS = 1024

#: What the text says when the timeline had to be truncated / other episodes were left out. Named so
#: the omission is legible to the model rather than looking like the whole story.
_TRUNCATION_MARK = "…"


def open_episode_of(log: "EpisodeLog | None") -> "Episode | None":
    """The open episode a judge call should be told about, or ``None``.

    Thin, total wrapper over :meth:`percept.episodes.EpisodeLog.current_open` that also accepts
    ``None`` (a goal with no episode log at all — the common case), so a call site can ask without
    first proving the concern is episode-owning."""
    if log is None:
        return None
    return log.current_open()


def _fmt_t(t: Any) -> str:
    """Stream time, to milliseconds. Fixed width for a given magnitude, so it cannot drive growth."""
    try:
        return f"{float(t):.3f}"
    except (TypeError, ValueError):
        return "unknown"


def episode_context_text(
    episode: "Episode",
    *,
    other_open: int = 0,
    max_revisions: int = JUDGE_CONTEXT_MAX_REVISIONS,
    max_value_chars: int = JUDGE_CONTEXT_MAX_VALUE_CHARS,
    max_chars: int = EPISODE_CONTEXT_MAX_CHARS,
) -> str:
    """Serialize ONE open episode into the bounded, pixel-free text a judge call carries.

    Reads only :meth:`percept.episodes.Episode.to_judge_dict`, so the payload-stripping and the
    per-field bounds live in one place and this function cannot reintroduce pixels by reaching for a
    field directly. ``other_open`` is how many further open episodes this concern holds and were
    deliberately NOT serialized — reported as a count so the judge knows the view is partial while
    the size stays constant.

    No ``;`` is emitted: adapters join context entries with ``"; "``, and a semicolon inside the
    value would read as the start of a different fact.
    """
    return _text_from_judge_dict(
        episode.to_judge_dict(max_revisions=max_revisions, max_value_chars=max_value_chars),
        other_open=other_open,
        max_chars=max_chars,
    )


def _text_from_judge_dict(d: dict, *, other_open: int, max_chars: int) -> str:
    """The rendering half of :func:`episode_context_text`, off an ALREADY-TAKEN judge projection.

    Split out so the text and its cache fingerprint can be derived from ONE reading of the episode
    (see :func:`episode_context_and_fingerprint`) instead of two — two readings could straddle a
    mutation and disagree, which is precisely the failure a fingerprint exists to prevent.
    """
    subject = d["subject_key"] or "unidentified subject"
    parts = [
        f"you are continuing an already-established episode ({d['status']}) about {subject}",
        f"first seen at t={_fmt_t(d['started_at'])}",
        f"last seen at t={_fmt_t(d['last_seen'])}",
        # JUDGED observations, never the raw fold count: a verdict re-served from the cache advances
        # ``observations`` without spending any evidence, and quoting that number back to the model
        # would let cheap serves inflate the continuity prior that makes the next serve likelier.
        f"confirmed across {d['judged_observations']} observations",
    ]
    if d["last_confirmed"] is not None:
        parts.append(f"last confirmed at t={_fmt_t(d['last_confirmed'])}")
    if d["value"] is not None:
        parts.append(f"established finding: {d['value']}")
    if d["incident_id"] is not None:
        parts.append(f"already reported as incident {d['incident_id']}")
    if d["revisions"]:
        rows = ", ".join(
            f"{r['kind']}@t={_fmt_t(r['t'])}"
            + (f" ({r['value']})" if r["value"] is not None else "")
            for r in d["revisions"]
        )
        omitted = d["revisions_omitted"]
        prefix = f"{_TRUNCATION_MARK} " if omitted else ""
        label = (
            f"most recent {len(d['revisions'])} of {len(d['revisions']) + omitted} changes"
            if omitted else "changes"
        )
        parts.append(f"{label}: {prefix}{rows}")
    if other_open > 0:
        parts.append(f"{other_open} other open episode(s) not shown")
    text = ", ".join(parts).replace(";", ",")
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)] + _TRUNCATION_MARK
    return text


# ==================================================================================================
# THE CACHE FINGERPRINT — keying on context state instead of bypassing on its presence
# ==================================================================================================
#
# Injecting the open episode above made the judge's answer depend on state the ROI frame hash cannot
# see, and the first remedy was to switch the verdict cache OFF whenever a context was present. That
# is measurably the wrong trade: on 180s of real footage a single continuous occurrence produced 187
# judge calls and zero cache hits — the better episodes worked, the more completely the cache was
# disabled. The codebase had already solved this exact class of problem once, for ``condition``: a
# re-armed condition is also state the frame hash does not cover, and the answer there was to put it
# in the ENTRY SLOT, not to stop caching (``cache/framehash.py``). This does the same, one level up.
#
# It rests on splitting the serialized episode by ONE question: can this field change the answer?
#
#   answer-affecting            answer-neutral (moves every tick, says nothing new)
#   ----------------            ------------------------------------------------------
#   id, status                  started_at, last_seen, last_confirmed
#   subject_key                 observations / judged_observations
#   value, incident_id          each revision's timestamp
#   retained (kind, value)
#   revisions_omitted
#   count of other open
#
# The split is not a guess about model behaviour, it is a property of the log: a ``confirmed``
# heartbeat COALESCES IN PLACE. It is deliberately absent from
# :data:`percept.episodes.STATE_CHANGING_REVISION_KINDS`, so it appends no timeline row and moves
# only ``last_seen`` / ``last_confirmed`` / the observation counts — exactly the right-hand column. An
# unbroken true period therefore holds the fingerprint at ONE value while the frame hash keeps doing
# the job it is good at, and any genuine change (a reopen, an ``updated`` value, an annotation, a
# close, another episode opening alongside) lands in the left-hand column and is a clean miss.
#
# What is deliberately NOT keyed on: a revision COUNT as a version. ``record_revision`` drops the
# oldest row past ``max_revisions``, so ``len(revisions)`` saturates and is non-monotone; and
# ``EpisodeLog.annotate`` writes ``value`` / ``incident_id`` without touching ``revisions`` at all.
# A count would therefore both fail to change when the state changed and change when it had not.
# The retained ``(kind, value)`` pairs plus the single-writer fields are what actually pin the state.

#: Bumped if the projection below ever changes meaning, so entries minted under an older definition
#: can never be served under a newer one. It is the first field hashed.
_FINGERPRINT_VERSION = "ep-fp/1"

#: The :meth:`percept.episodes.Episode.to_judge_dict` fields the fingerprint COVERS.
FINGERPRINTED_EPISODE_FIELDS = frozenset(
    {"id", "goal_id", "status", "subject_key", "value", "incident_id", "revisions",
     "revisions_omitted"}
)

#: The ``to_judge_dict`` fields deliberately LEFT OUT — clock readings and counts that a coalescing
#: ``confirmed`` heartbeat advances without any state change. Leaving them out is what makes a
#: continuous occurrence cacheable; a test asserts these two sets exactly cover the projection, so a
#: field added to it has to be classified here rather than silently escaping the key.
ANSWER_NEUTRAL_EPISODE_FIELDS = frozenset(
    {"started_at", "last_seen", "last_confirmed", "observations", "judged_observations"}
)

#: The same split inside one retained revision row.
FINGERPRINTED_REVISION_FIELDS = frozenset({"kind", "value"})
ANSWER_NEUTRAL_REVISION_FIELDS = frozenset({"t"})


def _framed(fields: "list[str]") -> str:
    """Concatenate ``fields`` so the result determines them UNIQUELY.

    Each field is emitted length-prefixed (``"<len>:<text>"``), which makes the encoding injective:
    the concatenation can be parsed back field-by-field, so two different field lists can never
    produce the same string. That is what carries the fingerprint's correctness argument — equal
    digests mean equal inputs, not merely inputs that happened to render alike under some separator a
    caller-owned value could contain.
    """
    return "".join(f"{len(f)}:{f}" for f in fields)


def _opt(value: Any) -> str:
    """Presence-tag an optional field so absence has exactly ONE spelling.

    ``None`` becomes ``"0"``; anything present becomes ``"1"`` + its text. Without the tag a missing
    ``subject_key`` and the empty string would encode identically, and the whole point of the framing
    above is that nothing encodes identically to anything else.
    """
    return "0" if value is None else "1" + (value if isinstance(value, str) else str(value))


def _fingerprint_fields(d: dict, *, other_open: int) -> "list[str]":
    """The answer-affecting projection of one judge-dict, as a list of fields to frame and hash.

    Reads exactly :data:`FINGERPRINTED_EPISODE_FIELDS` (and, per retained row,
    :data:`FINGERPRINTED_REVISION_FIELDS`) — nothing from the answer-neutral set.
    """
    fields = [
        _opt(d["id"]),
        _opt(d["goal_id"]),
        _opt(d["status"]),
        _opt(d["subject_key"]),
        _opt(d["value"]),
        _opt(d["incident_id"]),
        str(int(other_open)),
        # How many rows the timeline had to drop. Not used as a version (see the note above) — it is
        # in here because the TEXT says "most recent 5 of 9 changes", so it is part of what the model
        # is told. It can only move when a state-changing row is appended, never on a heartbeat.
        str(int(d["revisions_omitted"])),
        str(len(d["revisions"])),
    ]
    for r in d["revisions"]:
        fields.append(_opt(r["kind"]))
        fields.append(_opt(r["value"]))
    return fields


def episode_context_and_fingerprint(
    log: "EpisodeLog | None",
    *,
    static_context: "str | None" = None,
    **kwargs: Any,
) -> "tuple[str | None, str | None]":
    """``(open-episode text, cache fingerprint)`` for this log — from ONE reading of the episode.

    The pairing is the correctness argument, not a convenience: both halves are derived from a single
    :meth:`~percept.episodes.Episode.to_judge_dict` snapshot and a single ``open_episodes()`` count,
    with nothing between them that could mutate the log. So a fingerprint can never describe a
    different episode state than the text it accompanies, and "same key implies same answer-affecting
    context" holds by construction rather than by two call sites agreeing to stay in step.

    The text is ``None`` when nothing is open (a call site can use it directly as "is there anything
    to inject?"). The fingerprint is ``None`` when there is no answer-affecting context AT ALL — no
    open episode and no bound static context — which keeps the cache key byte-for-byte the one every
    pre-existing caller builds.

    ``static_context`` is the goal's bound static-context text, if it has one. It is per-goal and
    stable, so it never churns; it is in here so the fingerprint is a COMPLETE projection of the
    answer-affecting context rather than only of the episode. Note that the dispatch site still
    bypasses the cache outright when a static context is bound, so today this term is what makes the
    projection total, not what makes the key correct.
    """
    episode = open_episode_of(log)
    if episode is None and static_context is None:
        return None, None
    fields = [_FINGERPRINT_VERSION]
    text: "str | None" = None
    if episode is None:
        fields.append(_opt(None))  # no open episode — distinct from any episode's projection
    else:
        max_chars = kwargs.pop("max_chars", EPISODE_CONTEXT_MAX_CHARS)
        d = episode.to_judge_dict(**kwargs)
        other_open = max(0, len(log.open_episodes()) - 1) if log is not None else 0
        text = _text_from_judge_dict(d, other_open=other_open, max_chars=max_chars)
        fields.append(_opt("episode"))
        fields.extend(_fingerprint_fields(d, other_open=other_open))
    fields.append(_opt(static_context))
    digest = hashlib.blake2s(_framed(fields).encode("utf-8"), digest_size=16)
    return text, digest.hexdigest()


def episode_judge_context(log: "EpisodeLog | None", **kwargs: Any) -> "str | None":
    """The judge-context text for whatever this log currently holds open, or ``None``.

    ``None`` — not an empty string — when the concern has no open episode, so a call site can use it
    directly as "is there anything to inject?" and a goal with nothing open is byte-for-byte
    unchanged. A caller that also needs the cache fingerprint must call
    :func:`episode_context_and_fingerprint` instead of this plus a second reading of the log."""
    return episode_context_and_fingerprint(log, **kwargs)[0]


def with_open_episode(context: "dict | None", text: str) -> dict:
    """Return a COPY of ``context`` carrying ``text`` under :data:`JUDGE_OPEN_EPISODE_KEY`.

    Never mutates the caller's dict, and refuses to inject over a caller-supplied value for the
    reserved key (:class:`~percept.judge_context.ReservedContextKeyError`) — the same contract the
    static-context key already has, for the same reason: two different meanings arriving under one
    key is a bug that must be loud, not a silent overwrite."""
    if context is not None and JUDGE_OPEN_EPISODE_KEY in context:
        raise ReservedContextKeyError(
            f"judge context already carries the reserved key {JUDGE_OPEN_EPISODE_KEY!r}; "
            "the concern's open episode cannot be injected over it"
        )
    merged = dict(context) if context else {}
    merged[JUDGE_OPEN_EPISODE_KEY] = text
    return merged


async def judge_episode_memory(
    vision: Any,
    goal: Goal,
    frame: Any,
    *,
    store: EpisodeMemoryStore,
    frames: list | None = None,
    roi: Any = None,
    exemplar: Any = None,
    corrections: dict | None = None,
    episode_log: "EpisodeLog | None" = None,
) -> Verdict:
    """One episode-memory judge step: the next precision/recall lever after the ordered window.

    Threads ``store``'s prior running answer for ``goal.id`` INTO the judge call (the same
    ``context={"process_qa": True, "running_answer": ...}`` seam ``agent.py`` already uses for opted-in
    process-QA goals — see ``backends/openrouter.py:_judge_process_qa``), folds the model's rewritten
    answer back OUT into ``store``, and returns a ``Verdict`` whose ``satisfied`` integrates evidence
    ACCUMULATED OVER TIME (accumulated coverage + substance + confidence, via
    :func:`~percept.concern.resolution_ready`) rather than reflecting only this one frame's read — this
    is the productization step over the raw per-frame ``running_answer`` plumbing.

    A parse miss (the backend returns no new ``running_answer``) is a no-op on the state — the prior
    answer (and its accumulated coverage) is preserved, never regressed.

    ``episode_log`` is optional and, when given, adds the concern's currently open episode under
    :data:`~percept.judge_context.JUDGE_OPEN_EPISODE_KEY` — so this path threads BOTH halves of the
    memory: what the model said last (the running answer) and what the system already established
    (the episode). Omitted, or with nothing open, the call is byte-for-byte the prior one."""
    state = store.get(goal.id)
    prior = state.running_answer
    context: dict = {"process_qa": True, "running_answer": prior}
    episode_text = episode_judge_context(episode_log)
    if episode_text is not None:
        context = with_open_episode(context, episode_text)
    verdict = await vision.judge(
        goal.condition, frame,
        roi=roi, exemplar=exemplar, corrections=corrections, frames=frames,
        context=context,
    )
    state.record(verdict.running_answer)
    answer = state.running_answer
    integrated = resolution_ready(answer, state.frame_count, verdict.confidence)
    return Verdict(
        satisfied=integrated,
        confidence=verdict.confidence,
        evidence=answer or verdict.evidence,
        running_answer=answer,
    )


__all__ = [
    "JUDGE_MODE_EPISODE_MEMORY",
    "EPISODE_CONTEXT_MAX_CHARS",
    "FINGERPRINTED_EPISODE_FIELDS",
    "ANSWER_NEUTRAL_EPISODE_FIELDS",
    "FINGERPRINTED_REVISION_FIELDS",
    "ANSWER_NEUTRAL_REVISION_FIELDS",
    "episode_memory_enabled",
    "EpisodeMemoryState",
    "EpisodeMemoryStore",
    "judge_episode_memory",
    "open_episode_of",
    "episode_context_text",
    "episode_context_and_fingerprint",
    "episode_judge_context",
    "with_open_episode",
]
