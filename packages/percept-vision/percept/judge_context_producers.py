"""The judge-context key registry: what may be sent, and how each thing is SAID to the model.

WHY THIS EXISTS. ``Vision.judge(condition, frame, context={...})`` carries the structured facts a
frame cannot show — the open episode, a specialist's extractions, a deployment's static rubric,
whether this is a claim-slot goal. Until this module there was no single statement of what those keys
are. Each backend carried a hand-copied DENY-list and the three had already drifted apart:
``openrouter`` excluded ``process_qa``, ``anthropic`` and ``gemini`` did not. Only the ``_percept_``
PREFIX was enforced structurally, so **a new public key rendered into the prompt by default** — and
so did an arbitrary key from a caller.

Worse than the leak was HOW it rendered. Everything without a special case fell through one generic
path and was announced to the model as::

    A specialist model has reported these structured facts (trust them over your own read): …

The concern's own episode history is not a specialist's report; it is what this system already
established. A deployment's rubric is a policy document. Neither should be framed as a third party's
authoritative claim, and both were.

So a key is declared here with its TREATMENT — the shape it takes in the request — rather than being
filtered by absence from a list somewhere else.

WHAT IS DECLARED, AND WHAT IS NOT. The eight PUBLIC keys are enumerated. The ``_percept_`` keys are
admitted by prefix rather than listed: that prefix is already enforced structurally by every backend,
a new one is always transport metadata rather than prompt content, and enumerating them would mean
importing ``percept.backends.judge_pool`` from a top-level module — a cycle. The prefix rule is the
stronger statement anyway: it holds for keys that do not exist yet.

WHERE THIS IS ENFORCED. At route SELECTION, never inside a backend's ``judge()``.
``percept.conformance`` deliberately passes an unregistered probe key straight into an adapter to
prove the seam accepts a caller's context; adapter-level enforcement would fail the published
conformance suite. A judge is a capability, and the place to refuse a capability is where the route
is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .extraction import CONTEXT_KEY as EXTRACTOR_CONTEXT_KEY
from .judge_context import JUDGE_OPEN_EPISODE_KEY, JUDGE_STATIC_CONTEXT_KEY

#: Keys with this prefix are RUNTIME TRANSPORT — route metadata, temporal spans, clip geometry. They
#: are stripped by every backend before rendering and never become prompt text. Admitted by rule so
#: that a key added to that private namespace needs no change here.
PRIVATE_KEY_PREFIX = "_percept_"

#: How a key is SAID to the model. The treatment is the point of this registry: a key's meaning
#: determines its shape in the request, and the generic fallback was wrong for most of them.
#:
#: * ``own_block`` — the backend strips the key and emits a purpose-built instruction for it. The
#:   two that already did this (``claim_mode``, ``event_chain``) are the model the others follow.
#: * ``early_return`` — the key selects a DIFFERENT request entirely; the ordinary judge prompt is
#:   never built.
#: * ``episode`` — what this system already established about the occurrence in progress. Its own
#:   voice, not a third party's.
#: * ``policy`` — a deployment-bound rubric. A standing instruction, not an observation, and
#:   potentially large; it must not be inlined into a sentence of facts.
#: * ``specialist_facts`` — a specialist model's structured report. The ONLY treatment for which
#:   "trust them over your own read" is the honest framing.
#: * ``transport`` — never rendered.
Treatment = Literal[
    "own_block", "early_return", "episode", "policy", "specialist_facts", "transport"
]


@dataclass(frozen=True)
class ContextKeySpec:
    """One declared judge-context key.

    ``answer_affecting`` records whether the key's value can change the verdict. It is the input to
    the cache question: a key that cannot change the answer need not move the cache key, and a key
    that can must either be covered by a fingerprint or force a bypass. ``fingerprinted`` says which
    of those two a covered key gets — ``True`` means some producer folds it into a cache fingerprint,
    ``False`` means its presence bypasses the cache outright.
    """

    key: str
    treatment: Treatment
    answer_affecting: bool
    fingerprinted: bool
    note: str


#: The eight PUBLIC judge-context keys. Anything else without the private prefix is unregistered and
#: is refused at selection.
PUBLIC_CONTEXT_KEYS: tuple[ContextKeySpec, ...] = (
    ContextKeySpec(
        key=JUDGE_OPEN_EPISODE_KEY, treatment="episode",
        answer_affecting=True, fingerprinted=True,
        note="the occurrence this concern already has open. Folded into the cache fingerprint by "
             "percept.episode_memory rather than bypassing the cache, so a continuing occurrence "
             "stays cacheable while a real change to it misses.",
    ),
    ContextKeySpec(
        key=JUDGE_STATIC_CONTEXT_KEY, treatment="policy",
        answer_affecting=True, fingerprinted=False,
        note="a deployment-bound rubric, verified by hash at resolve time. Answer-affecting, and its "
             "writer also forces a cache bypass, so it is not keyed. It is a standing instruction "
             "and can be large -- it must never be inlined into a facts sentence.",
    ),
    ContextKeySpec(
        key=EXTRACTOR_CONTEXT_KEY, treatment="specialist_facts",
        answer_affecting=True, fingerprinted=False,
        note="a specialist model's structured report. The one key for which 'trust these over your "
             "own read' is honest framing -- that phrasing exists for this and was being applied to "
             "everything.",
    ),
    ContextKeySpec(
        key="process_qa", treatment="early_return",
        answer_affecting=True, fingerprinted=False,
        note="selects the accumulating-answer request instead of the ordinary judge prompt.",
    ),
    ContextKeySpec(
        key="running_answer", treatment="early_return",
        answer_affecting=True, fingerprinted=False,
        note="the prior answer threaded into the process-QA request. Read only inside that branch; "
             "it was in NO backend's deny-list, so on the ordinary path it rendered as a fact.",
    ),
    ContextKeySpec(
        key="observe_step", treatment="early_return",
        answer_affecting=True, fingerprinted=False,
        note="selects the observe-step request. Bypasses the cache: its answer accumulates per call.",
    ),
    ContextKeySpec(
        key="claim_mode", treatment="own_block",
        answer_affecting=True, fingerprinted=False,
        note="the reply must carry a correction channel (operation/value). Stripped and given its "
             "own contract block by all three backends -- the model this registry generalises.",
    ),
    ContextKeySpec(
        key="event_chain", treatment="own_block",
        answer_affecting=True, fingerprinted=False,
        note="the reply must carry occurrence JSON. Stripped and given its own block, and it also "
             "drives the evidence shape at route selection.",
    ),
)

_BY_KEY: dict[str, ContextKeySpec] = {spec.key: spec for spec in PUBLIC_CONTEXT_KEYS}

#: Every PUBLIC key the runtime may emit. The default a route accepts (see
#: ``JudgeCapability.context_keys``) is this set: it admits everything the runtime actually sends
#: while refusing anything unregistered, which is the leak that mattered. Narrowing a route below
#: this is an opt-in a deployment chooses, not a migration every caller is forced through.
RESERVED_CONTEXT_KEYS: frozenset[str] = frozenset(_BY_KEY)

#: The keys a backend must NOT render as prompt facts, because each has its own treatment. Derived,
#: so the three backends stop hand-copying divergent deny-lists.
NON_RENDERABLE_KEYS: frozenset[str] = frozenset(
    spec.key for spec in PUBLIC_CONTEXT_KEYS if spec.treatment != "specialist_facts"
)


def spec_for(key: str) -> "ContextKeySpec | None":
    """The declaration for ``key``, or ``None`` if it is unregistered."""
    return _BY_KEY.get(key)


def is_registered(key: str) -> bool:
    """Whether ``key`` is one the runtime declares — a public key, or private transport by prefix."""
    return key in _BY_KEY or key.startswith(PRIVATE_KEY_PREFIX)


def unregistered(context: "dict | None") -> frozenset[str]:
    """The keys in ``context`` that nothing declares.

    Empty means the whole context is accounted for. A non-empty result at route selection is a
    refusal: the request carries state no judge has agreed to receive, and the alternative is that it
    reaches the model framed as a fact somebody's prompt will act on.
    """
    return frozenset(k for k in (context or {}) if not is_registered(k))
