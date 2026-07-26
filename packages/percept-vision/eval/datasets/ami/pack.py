"""``meeting_observation@1`` — a SELF-CONTAINED :class:`~percept.domain_pack.DomainPack` scaffold for
recorded-meeting evaluation.

This module deliberately does NOT register itself. It is dev tooling, it lives outside the wheel, and
must not mutate the process-wide default registries as an import side effect. It exports
:data:`MEETING_OBSERVATION_V1` and a :func:`register_meeting_pack` helper instead; see the package
``README.md`` for explicit registration.

Three templates, one per condition the AMI transform emits, all built from the EXISTING episode
("count") sense — no new temporal shape is invented here:

- ``decision_made``  — DECISIONS-linked act: "a decision was made".
- ``action_assigned`` — ACTIONS-linked act:  "an action item was assigned".
- ``problem_raised`` — PROBLEMS-linked act:  "a problem was raised".

RECORDED-MEETING EVALUATION ONLY. The pack's ``notes`` and its ``sense:conversation`` recommendation
carry the shipped ``conversation`` SourceProfile's two measured EXPERIMENTAL limits verbatim (no
diarization behind ``identity_strategy="speaker_label"``; goals judge serially at ~24s/turn so a live
conversation outruns the watch). This pack makes NO live-conversation claim.
"""

from __future__ import annotations

from percept.direct_judge import DirectRouting
from percept.domain_pack import (
    ComponentRecommendation,
    DomainPack,
    GoalTemplate,
    register_domain_pack,
)

__all__ = [
    "MEETING_OBSERVATION_V1",
    "CONVERSATION_EXPERIMENTAL_LIMITS",
    "register_meeting_pack",
]

#: The shipped ``conversation`` profile's two measured limits, quoted so this pack and the profile
#: cannot drift apart. Carried into the pack ``notes`` AND the ``sense:conversation`` recommendation.
CONVERSATION_EXPERIMENTAL_LIMITS = (
    "speaker identity is OFF-CONTRACT: identity_strategy='speaker_label' has no diarization behind "
    "it in this build, so two speakers can collapse to one entity; and goals judge SERIALLY at "
    "~24s/turn measured, so a live conversation outruns the watch. RECORDED-MEETING EVALUATION "
    "ONLY — this pack makes no live-conversation claim."
)

# Episode ("count") sense: each distinct decision/action/problem is counted as its own episode
# (percept.behavior.program_of: type == "count" -> sense="episode"), which matches the transform's
# one-episode-per-summary-linked-act shape. Fire-worthy events, so miss_cost is "mild"; a spurious
# note in a recorded eval is cheap, so false_fire_cost is "trivial".
_DECISION_MADE = GoalTemplate(
    name="decision_made",
    condition="a decision was made",
    target=None,
    behavior={"type": "count"},
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    source="microphone",
    notes=(
        "fires once per distinct decision (a DECISIONS-linked dialogue act in the AMI abstractive "
        "summary), not once per utterance."
    ),
)

_ACTION_ASSIGNED = GoalTemplate(
    name="action_assigned",
    condition="an action item was assigned",
    target=None,
    behavior={"type": "count"},
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    source="microphone",
    notes=(
        "fires once per distinct action item (an ACTIONS-linked dialogue act in the AMI abstractive "
        "summary)."
    ),
)

_PROBLEM_RAISED = GoalTemplate(
    name="problem_raised",
    condition="a problem was raised",
    target=None,
    behavior={"type": "count"},
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    source="microphone",
    notes=(
        "fires once per distinct problem (a PROBLEMS-linked dialogue act in the AMI abstractive "
        "summary)."
    ),
)

_TEMPLATES = {t.name: t for t in (_DECISION_MADE, _ACTION_ASSIGNED, _PROBLEM_RAISED)}

_COMPONENTS = (
    ComponentRecommendation(
        ref="sense:conversation",
        strength="preferred",
        reason=(
            "a recorded meeting is a conversation source, not a camera or a screen. NOTE the "
            "shipped conversation SourceProfile is EXPERIMENTAL: " + CONVERSATION_EXPERIMENTAL_LIMITS
        ),
    ),
    ComponentRecommendation(
        ref="source_profile:conversation",
        strength="preferred",
        reason=(
            "transform + score against the conversation source class (ADR-005). Experimental — see "
            "the reason above and this pack's notes."
        ),
    ),
)

# Reserved namespace, no rule implementation ships here (same forward-declared seam facility uses):
# conditions authored under this prefix could later be served by a DirectJudge rule.
_DIRECT_JUDGE_ROUTES = (DirectRouting(condition_namespace="meeting_observation."),)

# Fixture names the AMI transform produces one tree per (eval/datasets/ami/testdata/fixtures). Plain
# data — not required to resolve to packaged clips (this pack is qualified against the AMI transform's
# output, never shipped clips).
_FIXTURES = (
    "meeting_observation/decision_made",
    "meeting_observation/action_assigned",
    "meeting_observation/problem_raised",
)

_QUALITY_BARS = {
    "decision_made.recall_min": 0.60,
    "action_assigned.recall_min": 0.55,
    "problem_raised.recall_min": 0.55,
}

MEETING_OBSERVATION_V1 = DomainPack(
    id="meeting_observation",
    version="1",
    templates=_TEMPLATES,
    components=_COMPONENTS,
    policy_clamps={"false_negative_cost_min": "trivial", "retention_max_days": 30.0},
    direct_judge_routes=_DIRECT_JUDGE_ROUTES,
    fixtures=_FIXTURES,
    quality_bars=_QUALITY_BARS,
    notes=(
        "Recorded-meeting observation over the AMI Meeting Corpus: decision made, action item "
        "assigned, problem raised — each mapped from the abstractive summary's extractive links "
        "(eval.datasets.ami.transform), built entirely from the existing episode ('count') sense. "
        "SOURCE CLASS: conversation, which is shipped EXPERIMENTAL — " + CONVERSATION_EXPERIMENTAL_LIMITS
    ),
)


def register_meeting_pack(register=register_domain_pack, *, overwrite: bool = False) -> None:
    """The DOCUMENTED one-line wiring, as a callable (not run at import).

    A human/Fable applies it explicitly — either by calling this, or by adding
    ``register_domain_pack(MEETING_OBSERVATION_V1)`` to ``percept/packs_default.py`` if the meeting
    pack graduates into the shipped set. Kept out of import side effects so this dev-tooling module
    never mutates the shared process registry used by other pack modules.
    """
    register(MEETING_OBSERVATION_V1, overwrite=overwrite)
