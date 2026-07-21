"""Episode counting — the cup problem (spec §5.5) and the THREE SEPARATED DECISIONS (card Y1).

Counting "cups of coffee" is **episode counting**: not sip-counting (over-counts) and not
IoU-track-counting (mis-counts across gaps). Per counting goal we keep a bounded **episode log**
(the counting window — "today's cups", a handful); the count is the number of distinct episodes.

Card Y1 — subject / continuation / mutation are THREE decisions, not one
------------------------------------------------------------------------
The original design collapsed them into one judge call ("which episode does this continue?"). That
is wrong, and wrong in the expensive direction: it fixes duplicate events by MANUFACTURING FALSE
CONTINUITY. PO-4471 is the same **subject** every time it appears; it is not necessarily the same
**episode**. A PO opened, closed and reopened. A person who leaves and re-enters. A door that opens
twice. The same claim slot established in two separate calls. The same invoice returning after the
continuity window as a new workflow occurrence. All of those are one subject and two episodes.

So the three decisions are named and evaluated separately here:

1. **SUBJECT** — *what is this observation about?* Exact subject equality (a judge-matched episode
   id today; a judge-EMITTED ``subject_key`` once cards Y5/Z3 land — pass ``subject_key=`` and no
   judge call is made). Subject equality is **NECESSARY EVIDENCE** for continuation, never its
   definition.
2. **CONTINUATION** — *does this observation extend the existing episode?* The conjunction below,
   evaluated in :meth:`EpisodeLog.observe_decision`::

       continues = (same_subject
                    and episode.status == "open"          # or reopen is allowed
                    and observation.ts - episode.last_seen <= continuity_gap_s
                    and mutation_is_compatible)

3. **MUTATION** — *how does this finding modify the episode?* Card Y2 owns the mutation CHANNEL;
   Y1 lands only its seam: the :data:`MutationCheck` predicate (``(episode, exemplar) -> bool``),
   defaulting to "compatible", which participates in the conjunction as its last conjunct. Y2
   replaces the boolean with a richer mutation without touching the other two decisions.

Every denial is TYPED (:data:`CONTINUATION_REASONS`) and OBSERVABLE (a structured log line plus the
optional ``on_continuation`` hook) — a same-subject observation that opened a NEW episode is exactly
the thing a client debugging percept at 2am needs to see, and it must never be silent.

Continuation semantics are **per concern**, never one global constant: a document-presence episode
(the document is there until it is gone — minutes), a door-open interval (seconds) and a
conversational claim (a whole call) do not agree on any of ``continuity_gap_s``, ``allow_reopen`` or
``terminal_statuses``. That is :class:`EpisodePolicy`, resolved per goal by
:func:`episode_policy_of`.

The judge is injected (an LLM in real use; a deterministic stub in L1) so the dedup *logic* is
testable without a model — and so we never assert backend intelligence with a fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from .observability import get_logger, log

# judge(summaries, exemplar) -> the id of the episode this observation shares a SUBJECT with, or
# None for a new subject. NOTE (Y1): the judge answers decision (1) ONLY. Its answer is evidence for
# continuation, not continuation itself — the returned episode still has to pass the continuation
# conjunction. The name is historical; ``subject_key=`` is the forward-looking seam (Y5/Z3).
EpisodeJudge = Callable[[list[dict], Any], "str | None"]

# The Y2 seam: does this finding MODIFY the episode in a way compatible with continuing it? A
# retraction, a contradicting value or a status flip is incompatible -> a new episode, not a
# silently mutated old one. Default (no check supplied) is "compatible", which is byte-for-byte the
# pre-Y1 behavior.
MutationCheck = Callable[["Episode", Any], bool]

#: Default status an episode is opened in. Anything in a policy's ``terminal_statuses`` is CLOSED.
OPEN = "open"
CLOSED = "closed"

ContinuationReason = Literal[
    # -- continued (is_new=False)
    "continued",              # same subject, open, inside the gap, mutation compatible
    "reopened",               # same subject, TERMINAL but the policy allows reopen -> same episode
    # -- a NEW episode was opened (is_new=True)
    "new_subject",            # no existing episode shares this subject — nothing to continue
    "gap_exceeded",           # same subject, but idle longer than the concern's continuity_gap_s
    "episode_terminal",       # same subject, but the episode is finished and reopen is not allowed
    "mutation_incompatible",  # same subject and live, but the finding contradicts the episode (Y2)
]

#: The closed vocabulary of WHY an observation did or did not continue an episode. Stable surface:
#: a code is never renamed or repurposed; a new path adds a NEW code.
CONTINUATION_REASONS: tuple[ContinuationReason, ...] = (
    "continued",
    "reopened",
    "new_subject",
    "gap_exceeded",
    "episode_terminal",
    "mutation_incompatible",
)

#: The reasons that mean "a same-subject episode existed and we refused to continue it". These are
#: the interesting ones: each is a place the collapsed design would have manufactured false
#: continuity. They are logged individually so a client can see WHICH rule split the episode.
DENIAL_REASONS: tuple[ContinuationReason, ...] = (
    "gap_exceeded",
    "episode_terminal",
    "mutation_incompatible",
)


@dataclass(frozen=True)
class EpisodePolicy:
    """CONCERN-LEVEL continuation semantics. There is no correct global constant for these.

    * ``continuity_gap_s`` — the maximum idle time (observation ts minus ``episode.last_seen``) that
      still counts as the SAME episode. ``None`` = unbounded (the pre-Y1 behavior; kept as the
      default so an un-authored concern is byte-for-byte unchanged).
    * ``allow_reopen`` — may a TERMINAL episode be reopened by a later same-subject observation
      (a PO reopened) instead of starting a fresh occurrence? Default no.
    * ``terminal_statuses`` — the statuses that mean "this episode is finished". A same-subject
      observation against a terminal episode opens a new one (or reopens, per ``allow_reopen``).
    """

    continuity_gap_s: float | None = None
    allow_reopen: bool = False
    terminal_statuses: frozenset[str] = frozenset({CLOSED})

    def is_terminal(self, status: str) -> bool:
        return status in self.terminal_statuses


#: Shipped starting points, NOT law — three domains whose continuation semantics genuinely differ.
#: Named so a concern can author ``goal.state["episode"] = {"policy": "door_interval"}`` and get a
#: defensible default instead of the unbounded one. Any field may still be overridden alongside it.
EPISODE_POLICIES: dict[str, EpisodePolicy] = {
    # A document is present until it is gone; a short occlusion is the SAME presence episode, and a
    # return after minutes is a new workflow occurrence.
    "document_presence": EpisodePolicy(continuity_gap_s=120.0, allow_reopen=False),
    # A door open/close interval: seconds. A second opening is emphatically a second episode.
    "door_interval": EpisodePolicy(continuity_gap_s=5.0, allow_reopen=False),
    # A conversational claim slot lives as long as the call and may be revisited within it.
    "conversational_claim": EpisodePolicy(continuity_gap_s=1800.0, allow_reopen=True),
    # Explicit "never split on time" — the pre-Y1 behavior, chosen rather than inherited.
    "unbounded": EpisodePolicy(),
}

#: Only these keys are honoured from an authored ``goal.state["episode"]`` block.
_POLICY_KEYS = ("continuity_gap_s", "allow_reopen", "terminal_statuses")
_MAX_TERMINAL_STATUSES = 16
_MAX_STATUS_LEN = 40


def episode_policy_of(goal: Any, default: "EpisodePolicy | None" = None) -> EpisodePolicy:
    """Resolve the CONCERN-LEVEL :class:`EpisodePolicy` for a goal from ``goal.state['episode']``.

    Reads the same validated ``goal.state`` channel every other authored knob uses
    (``percept.behavior.coerce_authored_episode`` is the writer). A goal that authored nothing gets
    ``default`` (or the unbounded legacy policy), so existing concerns are byte-for-byte unchanged.
    """
    base = default if default is not None else EpisodePolicy()
    state = getattr(goal, "state", None)
    spec = state.get("episode") if isinstance(state, dict) else None
    if not isinstance(spec, dict):
        return base
    named = spec.get("policy")
    if isinstance(named, str) and named in EPISODE_POLICIES:
        base = EPISODE_POLICIES[named]
    fields: dict = {}
    gap = spec.get("continuity_gap_s")
    if isinstance(gap, (int, float)) and not isinstance(gap, bool) and gap > 0:
        fields["continuity_gap_s"] = float(gap)
    if isinstance(spec.get("allow_reopen"), bool):
        fields["allow_reopen"] = spec["allow_reopen"]
    terminal = spec.get("terminal_statuses")
    if isinstance(terminal, (list, tuple, set, frozenset)):
        statuses = {s for s in terminal if isinstance(s, str) and 0 < len(s) <= _MAX_STATUS_LEN}
        if statuses:
            fields["terminal_statuses"] = frozenset(sorted(statuses)[:_MAX_TERMINAL_STATUSES])
    if not fields:
        return base
    return EpisodePolicy(**{k: fields.get(k, getattr(base, k)) for k in _POLICY_KEYS})


@dataclass
class Episode:
    id: str
    goal_id: str
    started_at: float
    last_seen: float
    exemplar: Any = None
    # Y1: ``exemplar`` is the FIRST observation that opened the episode; ``representative`` is the
    # MOST RECENT one folded into it. Before Y1 the two were set once and never diverged, which made
    # ``representative`` a dead field. It is now updated on every continuation, so a consumer asking
    # "what does this episode look like NOW" gets the current evidence, not the opening frame.
    representative: Any = None
    # Y1 decision (2): an episode has a LIFECYCLE. Only an ``open`` episode can be continued (unless
    # the concern's policy allows reopening a terminal one).
    status: str = OPEN
    # Y1 decision (1): the subject this episode is about. ``None`` today for judge-matched episodes;
    # populated when the caller passes a ``subject_key`` (the Y5/Z3 judge-emitted key seam).
    subject_key: str | None = None
    # How many observations folded into this episode (1 = just the opening one). Pure observability.
    observations: int = 1


@dataclass(frozen=True)
class ContinuationDecision:
    """One typed episode decision: the episode, whether it is NEW, and WHY (card Y1).

    ``subject_matched`` answers decision (1) on its own — it is true whenever an existing episode
    shared this observation's subject, INCLUDING the cases where we then refused to continue it.
    That split is the whole point of the card: ``subject_matched and is_new`` is precisely the
    situation the collapsed design got wrong.
    """

    episode: Episode
    is_new: bool
    reason: ContinuationReason
    subject_matched: bool = False
    prior_episode_id: str | None = None  # the same-subject episode that was NOT continued
    gap_s: float | None = None           # observed idle gap against that episode

    @property
    def denied(self) -> bool:
        """True iff a same-subject episode existed and a continuation was refused."""
        return self.reason in DENIAL_REASONS


class EpisodeLog:
    """Bounded per-goal episode log. ``observe`` folds a candidate event into new/continuation.

    Boundedness (ADR-001 §9, card A9 subset): ``window_s`` bounds the log in TIME (episodes idle
    longer than the counting window fall out — "today's cups", not "all-time cups") and
    ``max_episodes`` bounds it in COUNT (a hard ceiling, drop-oldest, the backstop when no window is
    set or the window alone isn't enough). Every bound-driven drop is TRACED — never silent —
    through ``on_drop(episode, reason)`` (``reason`` is ``"window_expired"`` or ``"max_episodes"``);
    the default logs a structured line so a drop is observable even when the caller wires nothing.

    Card Y1: ``policy`` carries the CONCERN-LEVEL continuation semantics (see :class:`EpisodePolicy`)
    and ``on_continuation(decision)`` observes every typed :class:`ContinuationDecision`. A denied
    continuation is additionally logged, so "why did this subject open a second episode?" is
    answerable from a log file alone.
    """

    def __init__(
        self, goal_id: str, *, window_s: float | None = None, max_episodes: int = 64,
        on_drop: "Callable[[Episode, str], None] | None" = None,
        policy: "EpisodePolicy | None" = None,
        on_continuation: "Callable[[ContinuationDecision], None] | None" = None,
    ) -> None:
        self.goal_id = goal_id
        self.window_s = window_s
        self.max_episodes = max_episodes
        self.on_drop = on_drop
        self.policy = policy if policy is not None else EpisodePolicy()
        self.on_continuation = on_continuation
        self._episodes: list[Episode] = []
        self._counter = 0

    def _dropped(self, episode: "Episode", reason: str) -> None:
        """ADR-001 §9: a bound-driven drop must never be silent."""
        if self.on_drop is not None:
            self.on_drop(episode, reason)
        else:
            log(
                get_logger("percept.episodes"), "episode.dropped",
                goal_id=self.goal_id, episode_id=episode.id, reason=reason,
            )

    def _prune(self, now: float) -> None:
        """Drop episodes outside the counting window (keeps the set bounded — the endurance story)."""
        if self.window_s is None:
            return
        keep: list[Episode] = []
        for e in self._episodes:
            if now - e.last_seen <= self.window_s:
                keep.append(e)
            else:
                self._dropped(e, "window_expired")
        self._episodes = keep

    def summaries(self) -> list[dict]:
        """Compact episode descriptors handed to the judge (no pixels on the hot path)."""
        return [
            {
                "id": e.id, "started_at": e.started_at, "last_seen": e.last_seen,
                "status": e.status, "subject_key": e.subject_key,
            }
            for e in self._episodes
        ]

    # -- decision (1): SUBJECT -------------------------------------------------------------------
    def _subject_match(
        self, exemplar: Any, judge: "EpisodeJudge | None", subject_key: "str | None"
    ) -> "Episode | None":
        """Which existing episode shares this observation's SUBJECT? Evidence only — the caller
        still has to run the continuation conjunction on whatever comes back.

        ``subject_key`` (the Y5/Z3 seam) wins when supplied: exact subject-key equality, most recent
        first, and NO judge call is made. Otherwise the injected judge answers, as it always did.
        """
        if not self._episodes:
            return None
        if subject_key is not None:
            for e in reversed(self._episodes):
                if e.subject_key == subject_key:
                    return e
            return None
        if judge is None:
            return None
        matched_id = judge(self.summaries(), exemplar)
        if matched_id is None:
            return None
        for e in self._episodes:
            if e.id == matched_id:
                return e
        return None

    # -- decision (2): CONTINUATION (with decision (3) as its last conjunct) ----------------------
    @staticmethod
    def continuation_reason(
        episode: Episode, t: float, policy: EpisodePolicy, *, mutation_compatible: bool = True
    ) -> ContinuationReason:
        """Evaluate the continuation conjunction against a SAME-SUBJECT episode.

        This is the card-Y1 shape, in order::

            continues = (same_subject                                   # the caller established this
                         and episode.status == "open"                   # or allow_reopen
                         and observation.ts - episode.last_seen <= continuity_gap_s
                         and mutation_is_compatible)

        Returns ``"continued"``/``"reopened"`` when it holds, else the FIRST conjunct that failed.
        The gap is checked before the mutation so the cheap, always-available evidence decides first
        and the (future, Y2) mutation channel is only consulted about a still-live episode.
        """
        if policy.is_terminal(episode.status) and not policy.allow_reopen:
            return "episode_terminal"
        if policy.continuity_gap_s is not None and (t - episode.last_seen) > policy.continuity_gap_s:
            return "gap_exceeded"
        if not mutation_compatible:
            return "mutation_incompatible"
        return "reopened" if policy.is_terminal(episode.status) else "continued"

    def _open(
        self, t: float, exemplar: Any, subject_key: "str | None",
        reason: ContinuationReason, prior: "Episode | None", gap_s: "float | None",
    ) -> ContinuationDecision:
        self._counter += 1
        episode = Episode(
            id=f"{self.goal_id}:ep{self._counter}",
            goal_id=self.goal_id,
            started_at=t,
            last_seen=t,
            exemplar=exemplar,
            representative=exemplar,
            status=OPEN,
            subject_key=subject_key,
        )
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            oldest = self._episodes.pop(0)  # oldest falls out of the bounded log
            self._dropped(oldest, "max_episodes")
        return ContinuationDecision(
            episode=episode, is_new=True, reason=reason,
            subject_matched=prior is not None,
            prior_episode_id=prior.id if prior is not None else None,
            gap_s=gap_s,
        )

    def _note(self, decision: ContinuationDecision) -> None:
        """Y1: a refused continuation must never be silent — it is the one event that distinguishes
        "two real occurrences" from "the SDK lost the thread". Logged unconditionally (so a bare
        ``EpisodeLog`` is still debuggable) and handed to ``on_continuation`` when wired."""
        if decision.denied:
            log(
                get_logger("percept.episodes"), "episode.continuation_denied",
                goal_id=self.goal_id, reason=decision.reason,
                prior_episode_id=decision.prior_episode_id,
                episode_id=decision.episode.id, gap_s=decision.gap_s,
            )
        if self.on_continuation is not None:
            self.on_continuation(decision)

    def observe_decision(
        self,
        t: float,
        exemplar: Any,
        judge: "EpisodeJudge | None" = None,
        *,
        subject_key: "str | None" = None,
        mutation: "MutationCheck | None" = None,
        policy: "EpisodePolicy | None" = None,
    ) -> ContinuationDecision:
        """Fold a candidate event at ``t`` into the log and return the TYPED decision (card Y1).

        The three decisions run in order and are individually visible on the result: subject
        (:attr:`ContinuationDecision.subject_matched`), continuation
        (:attr:`ContinuationDecision.is_new` + :attr:`~ContinuationDecision.reason`), and mutation
        (the ``mutation`` predicate, whose refusal surfaces as ``"mutation_incompatible"``).
        """
        pol = policy if policy is not None else self.policy
        self._prune(t)
        prior = self._subject_match(exemplar, judge, subject_key)
        if prior is None:
            decision = self._open(t, exemplar, subject_key, "new_subject", None, None)
            self._note(decision)
            return decision
        gap_s = t - prior.last_seen
        compatible = True if mutation is None else bool(mutation(prior, exemplar))
        reason = self.continuation_reason(prior, t, pol, mutation_compatible=compatible)
        if reason in ("continued", "reopened"):
            prior.last_seen = t              # continuation — extend, don't count again
            prior.representative = exemplar  # ...and the representative is now LIVE (was dead)
            prior.observations += 1
            if reason == "reopened":
                prior.status = OPEN
            decision = ContinuationDecision(
                episode=prior, is_new=False, reason=reason, subject_matched=True,
                prior_episode_id=prior.id, gap_s=gap_s,
            )
            self._note(decision)
            return decision
        # Same subject, refused continuation -> a genuinely NEW occurrence of the SAME subject.
        decision = self._open(t, exemplar, subject_key or prior.subject_key, reason, prior, gap_s)
        self._note(decision)
        return decision

    def observe(
        self,
        t: float,
        exemplar: Any,
        judge: "EpisodeJudge | None" = None,
        *,
        subject_key: "str | None" = None,
        mutation: "MutationCheck | None" = None,
        policy: "EpisodePolicy | None" = None,
    ) -> tuple[Episode, bool]:
        """Fold a candidate event at time ``t`` into the log. Returns ``(episode, is_new)``.

        The historical entry point, unchanged in shape: callers wanting the typed WHY (and the
        subject/continuation split) call :meth:`observe_decision` instead.
        """
        decision = self.observe_decision(
            t, exemplar, judge, subject_key=subject_key, mutation=mutation, policy=policy
        )
        return decision.episode, decision.is_new

    def close(self, episode_id: str, status: str = CLOSED) -> "Episode | None":
        """Mark an episode TERMINAL (decision (2)'s lifecycle input). A later same-subject
        observation then opens a new occurrence — or reopens this one, if the concern's policy says
        a reopen is the same episode. Returns the episode, or ``None`` if it is not in the log."""
        for e in self._episodes:
            if e.id == episode_id:
                e.status = status
                return e
        return None

    def get(self, episode_id: str) -> "Episode | None":
        for e in self._episodes:
            if e.id == episode_id:
                return e
        return None

    def count(self, now: float | None = None) -> int:
        """Distinct episodes (optionally pruning to the window at ``now`` first)."""
        if now is not None:
            self._prune(now)
        return len(self._episodes)


__all__ = [
    "Episode",
    "EpisodeJudge",
    "EpisodeLog",
    "EpisodePolicy",
    "EPISODE_POLICIES",
    "episode_policy_of",
    "MutationCheck",
    "ContinuationDecision",
    "ContinuationReason",
    "CONTINUATION_REASONS",
    "DENIAL_REASONS",
    "OPEN",
    "CLOSED",
]
