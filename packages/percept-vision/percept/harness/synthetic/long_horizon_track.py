"""Deterministic, manifest-driven session continuity track.

This track exercises the state machines behind programmable Watches on scripted multimodal-stream
events. It is deliberately not a perception-quality benchmark: every verdict, identity, and speaker
role is supplied by a synthetic fixture. Successful execution means the downstream episode, claim,
and incident-action state machines preserve those labelled facts inside one process session.

Unsupported requirements are first-class results. In particular, this track must not turn
session-only episode state into a restart-durability claim, or scripted speaker labels into a
diarization-accuracy claim. Because those required launch capabilities remain unsupported, this
session-semantics track is not itself a release-readiness pass.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ...adapters import Goal, Verdict
from ...behavior import BehaviorExecutor, BehaviorProgram
from ...claims import fold_verdicts
from ...contracts import ConcernAuthority, SubjectRef
from ...episodes import EpisodeLog, EpisodePolicy
from ...gate import Gate
from ..scripted import ScriptedJudge
from .corpora import BUILDERS

__all__ = [
    "LongHorizonCaseResult",
    "LongHorizonTrackReport",
    "ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES",
    "REQUIRED_SEMANTIC_CAPABILITY_CONTRACT",
    "REQUIRED_SEMANTIC_CAPABILITIES",
    "SemanticCapabilityBinding",
    "load_long_horizon_manifest",
    "run_long_horizon_track",
]

_MANIFEST = "long_horizon_manifest.json"
_MONEY = re.compile(r"\$\d+(?:\.\d+)?[KMB]?", re.IGNORECASE)


@dataclass(frozen=True)
class SemanticCapabilityBinding:
    """The only handler, corpus, and support status allowed for one required capability."""

    handler: str | None
    corpus: str | None
    expected_status: str


_EPISODE_BOUNDARIES = SemanticCapabilityBinding(
    "episode_boundaries", "episode_counting", "implemented"
)
_TIMESTAMPED_RECURRENCE = SemanticCapabilityBinding(
    "timestamped_session_recurrence", "longhorizon", "implemented"
)
_SUBJECT_ASSOCIATION = SemanticCapabilityBinding(
    "subject_association", "subject_identity", "implemented"
)
_CLAIM_REDUCER = SemanticCapabilityBinding(
    "claim_reducer_semantics", "mutation", "implemented"
)
_INCIDENT_ACTION_SUPPRESSION = SemanticCapabilityBinding(
    "episode_owned_incident_action_suppression", "idempotency", "implemented"
)
_BOUNDED_EPISODE_LOG = SemanticCapabilityBinding(
    "bounded_episode_log_state", "longhorizon", "implemented"
)
_UNSUPPORTED = SemanticCapabilityBinding(None, None, "unsupported")

# This immutable, code-owned mapping is intentionally independent of manifest cases. It prevents a
# data-only edit from preserving capability names while swapping in a weaker handler/corpus or
# reclassifying an implemented requirement as unsupported (or vice versa).
REQUIRED_SEMANTIC_CAPABILITY_CONTRACT: Mapping[str, SemanticCapabilityBinding] = MappingProxyType(
    {
        "within_session_episode_boundaries": _EPISODE_BOUNDARIES,
        "episode_counting": _EPISODE_BOUNDARIES,
        "timestamped_within_session_episode_recurrence": _TIMESTAMPED_RECURRENCE,
        "subject_content_association": _SUBJECT_ASSOCIATION,
        "identity_abstention": _SUBJECT_ASSOCIATION,
        "claim_reducer_value_history": _CLAIM_REDUCER,
        "claim_reducer_authority_policy": _CLAIM_REDUCER,
        "claim_reducer_revision_chain": _CLAIM_REDUCER,
        "episode_owned_incident_action_suppression": _INCIDENT_ACTION_SUPPRESSION,
        "bounded_episode_log_state": _BOUNDED_EPISODE_LOG,
        "observable_episode_log_eviction": _BOUNDED_EPISODE_LOG,
        "restart_continuity": _UNSUPPORTED,
        "person_reidentification_accuracy": _UNSUPPORTED,
        "speaker_diarization_accuracy": _UNSUPPORTED,
    }
)
REQUIRED_SEMANTIC_CAPABILITIES = frozenset(REQUIRED_SEMANTIC_CAPABILITY_CONTRACT)
ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES = frozenset(
    {
        "restart_continuity",
        "person_reidentification_accuracy",
        "speaker_diarization_accuracy",
    }
)


@dataclass(frozen=True)
class LongHorizonCaseResult:
    """One manifest entry's result."""

    id: str
    requirements: tuple[str, ...]
    status: str
    fixture_cases: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class LongHorizonTrackReport:
    """Complete state-machine track result, including declared unsupported requirements."""

    track: str
    synthetic: bool
    session_scope: bool
    quality_claim: bool
    results: tuple[LongHorizonCaseResult, ...]

    @property
    def structurally_complete(self) -> bool:
        """Whether results cover every required capability exactly once."""
        requirements = [
            requirement for result in self.results for requirement in result.requirements
        ]
        return (
            len(requirements) == len(REQUIRED_SEMANTIC_CAPABILITIES)
            and set(requirements) == REQUIRED_SEMANTIC_CAPABILITIES
        )

    @property
    def execution_succeeded(self) -> bool:
        """Whether every implemented case ran successfully and unsupported scope stayed explicit."""
        return self.structurally_complete and all(
            result.status in ("pass", "unsupported") for result in self.results
        )

    @property
    def release_ready(self) -> bool:
        """Whether every required launch capability is supported and its case passed."""
        return (
            self.execution_succeeded
            and not ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES
            and not self.unsupported_requirements
            and all(result.status == "pass" for result in self.results)
        )

    @property
    def unsupported_requirements(self) -> tuple[str, ...]:
        return tuple(
            requirement
            for result in self.results
            if result.status == "unsupported"
            for requirement in result.requirements
        )


class _TrackFailure(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _TrackFailure(message)


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate scope, independent capability coverage, and executable case declarations."""
    manifest = dict(manifest)
    _require(manifest.get("schema_version") == "1", "unsupported long-horizon manifest schema")
    _require(manifest.get("synthetic") is True, "track must be labelled synthetic")
    _require(manifest.get("quality_claim") is False, "synthetic track must not claim quality")
    _require(manifest.get("session_scope") is True, "track must declare its session-only scope")
    declared = manifest.get("required_semantic_capabilities")
    _require(
        isinstance(declared, list)
        and all(isinstance(item, str) and item for item in declared)
        and len(declared) == len(set(declared)),
        "manifest required_semantic_capabilities must be unique non-empty strings",
    )
    _require(
        set(declared) == REQUIRED_SEMANTIC_CAPABILITIES,
        "manifest required_semantic_capabilities must exactly match the code-owned contract",
    )
    entries = manifest.get("cases")
    _require(isinstance(entries, list) and entries, "manifest cases must be a non-empty list")
    ids: set[str] = set()
    owners: dict[str, str] = {}
    for entry in entries:
        _require(isinstance(entry, dict), "manifest cases must contain objects")
        entry_id = entry.get("id")
        _require(isinstance(entry_id, str) and entry_id, "manifest entry needs an id")
        _require(entry_id not in ids, f"duplicate manifest entry {entry_id!r}")
        ids.add(entry_id)
        requirements = entry.get("requirements")
        _require(
            isinstance(requirements, list)
            and requirements
            and all(isinstance(item, str) and item for item in requirements),
            f"{entry_id}: requirements must be non-empty strings",
        )
        _require(
            len(requirements) == len(set(requirements)),
            f"{entry_id}: requirements must not contain duplicates",
        )
        expected_status = entry.get("expected_status")
        _require(
            expected_status in ("implemented", "unsupported"),
            f"{entry_id}: expected_status must be explicit: implemented or unsupported",
        )
        for requirement in requirements:
            _require(
                requirement in REQUIRED_SEMANTIC_CAPABILITIES,
                f"{entry_id}: undeclared semantic capability {requirement!r}",
            )
            _require(
                requirement not in owners,
                f"{entry_id}: semantic capability {requirement!r} is already owned by "
                f"{owners.get(requirement)!r}",
            )
            owners[requirement] = entry_id
            binding = REQUIRED_SEMANTIC_CAPABILITY_CONTRACT[requirement]
            _require(
                expected_status == binding.expected_status,
                f"{entry_id}: {requirement!r} status must be {binding.expected_status!r}",
            )
            _require(
                entry.get("handler") == binding.handler,
                f"{entry_id}: {requirement!r} must bind handler {binding.handler!r}",
            )
            _require(
                entry.get("corpus") == binding.corpus,
                f"{entry_id}: {requirement!r} must bind corpus {binding.corpus!r}",
            )
        if expected_status == "unsupported":
            _require(
                isinstance(entry.get("reason"), str) and entry["reason"],
                f"{entry_id}: unsupported entry needs a reason",
            )
            _require("handler" not in entry, f"{entry_id}: unsupported entry must not run a handler")
            _require("corpus" not in entry, f"{entry_id}: unsupported entry must not select a corpus")
            _require(
                "case_selector" not in entry and "case_ids" not in entry,
                f"{entry_id}: unsupported entry must not select fixture cases",
            )
            continue
        _require(entry.get("handler") in _HANDLERS, f"{entry_id}: unknown handler")
        _require(entry.get("corpus") in BUILDERS, f"{entry_id}: unknown corpus")
        all_cases = entry.get("case_selector") == "all"
        explicit_cases = entry.get("case_ids")
        _require(
            all_cases != bool(explicit_cases),
            f"{entry_id}: choose exactly one of case_selector='all' or non-empty case_ids",
        )
    missing = REQUIRED_SEMANTIC_CAPABILITIES.difference(owners)
    _require(
        not missing,
        f"manifest cases do not own required semantic capabilities {sorted(missing)!r}",
    )
    unsupported = {
        requirement
        for entry in entries
        if entry["expected_status"] == "unsupported"
        for requirement in entry["requirements"]
    }
    _require(
        unsupported == ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES,
        "manifest unsupported capabilities must exactly match the explicitly allowed set",
    )
    return manifest


def load_long_horizon_manifest() -> dict[str, Any]:
    """Load and structurally validate the packaged release-track manifest."""
    manifest = json.loads(files(__package__).joinpath(_MANIFEST).read_text(encoding="utf-8"))
    return _validate_manifest(manifest)


def _fixture_pairs(entry: Mapping[str, Any]) -> list[tuple[dict, dict]]:
    stream, labels = BUILDERS[str(entry["corpus"])]()
    labels_by_id = {case["id"]: case for case in labels["cases"]}
    stream_by_id = {case["id"]: case for case in stream["cases"]}
    if entry.get("case_selector") == "all":
        selected = [case["id"] for case in stream["cases"]]
    else:
        selected = list(entry["case_ids"])
    missing = [case_id for case_id in selected if case_id not in stream_by_id or case_id not in labels_by_id]
    _require(not missing, f"{entry['id']}: missing fixture cases {missing}")
    return [(stream_by_id[case_id], labels_by_id[case_id]) for case_id in selected]


async def _scripted(steps: list[dict]) -> list[Verdict]:
    judge = ScriptedJudge.from_steps(steps)
    return [await judge.judge("scripted condition", step["frame"]) for step in steps]


def _verdicts(steps: list[dict]) -> list[Verdict]:
    return asyncio.run(_scripted(steps))


def _episode_boundaries(entry: Mapping[str, Any]) -> tuple[str, ...]:
    fixture_ids: list[str] = []
    for stream_case, label in _fixture_pairs(entry):
        fixture_ids.append(stream_case["id"])
        expect = label["expect"]
        log = EpisodeLog(
            "long-horizon-count",
            policy=EpisodePolicy(continuity_gap_s=float(expect["boundary_gap_s"])),
        )
        for step, verdict in zip(stream_case["steps"], _verdicts(stream_case["steps"])):
            if verdict.satisfied:
                log.observe_decision(step["t"], step["frame"], subject_key=verdict.subject_key)
        summaries = log.summaries()
        actual = [[episode["started_at"], episode["last_seen"]] for episode in summaries]
        _require(log.count() == expect["count"], f"{stream_case['id']}: episode count mismatch")
        _require(actual == expect["boundaries"], f"{stream_case['id']}: boundary mismatch {actual!r}")
    return tuple(fixture_ids)


def _timestamped_session_recurrence(entry: Mapping[str, Any]) -> tuple[str, ...]:
    fixture_ids: list[str] = []
    for stream_case, label in _fixture_pairs(entry):
        fixture_ids.append(stream_case["id"])
        expect = label["expect"]
        timestamps = [float(step["t"]) for step in stream_case["steps"]]
        _require(
            timestamps == sorted(timestamps) and len(timestamps) == len(set(timestamps)),
            f"{stream_case['id']}: timestamps must be strictly increasing",
        )
        _require(
            timestamps[-1] - timestamps[0] == float(expect["duration_s"]),
            f"{stream_case['id']}: timestamped session duration mismatch",
        )
        log = EpisodeLog(
            "long-horizon-session",
            policy=EpisodePolicy(continuity_gap_s=float(expect["continuity_gap_s"])),
        )
        for step, verdict in zip(stream_case["steps"], _verdicts(stream_case["steps"])):
            if verdict.satisfied:
                log.observe_decision(step["t"], step["frame"], subject_key=verdict.subject_key)
        summaries = log.summaries()
        actual = [[episode["started_at"], episode["last_seen"]] for episode in summaries]
        _require(log.count() == expect["episode_count"], f"{stream_case['id']}: episode count mismatch")
        _require(actual == expect["boundaries"], f"{stream_case['id']}: recurrence boundaries mismatch")
        _require(
            {episode["subject_key"] for episode in summaries} == {expect["subject"]},
            f"{stream_case['id']}: scripted subject key changed within one EpisodeLog session",
        )
    return tuple(fixture_ids)


def _subject_ref(verdict: Verdict) -> SubjectRef:
    if verdict.binding == "resolved":
        return SubjectRef(
            "content", "synthetic", verdict.subject_key, "bound", "durable", "scripted"
        )
    return SubjectRef("content", "synthetic", None, "unknown", "session", "scripted")


def _subject_association(entry: Mapping[str, Any]) -> tuple[str, ...]:
    fixture_ids: list[str] = []
    for stream_case, label in _fixture_pairs(entry):
        fixture_ids.append(stream_case["id"])
        expect = label["expect"]
        log = EpisodeLog("long-horizon-subject")
        decisions = []
        for step, verdict in zip(stream_case["steps"], _verdicts(stream_case["steps"])):
            decisions.append(
                log.observe_decision(
                    step["t"],
                    Verdict(verdict.satisfied, verdict.confidence, subject=_subject_ref(verdict)),
                )
            )
        _require(
            log.count() == expect["distinct_subject_count"],
            f"{stream_case['id']}: distinct subject count mismatch",
        )
        for subject in expect["subjects"]:
            indexed = [decisions[index].episode for index in subject["step_indices"]]
            _require(
                len({episode.id for episode in indexed}) == 1,
                f"{stream_case['id']}: one labelled subject split across episodes",
            )
            if subject["binding"] == "resolved":
                _require(
                    all(episode.subject_key == subject["key"] for episode in indexed),
                    f"{stream_case['id']}: resolved content key mismatch",
                )
            else:
                _require(
                    all(episode.subject_key.startswith("UN_") for episode in indexed),
                    f"{stream_case['id']}: unknown subject did not remain provisional",
                )
                _require(
                    all(verdict.binding == "unknown" for verdict in _verdicts(stream_case["steps"])),
                    f"{stream_case['id']}: abstention was converted to a resolved identity",
                )
    return tuple(fixture_ids)


def _claim_specs(stream_case: Mapping[str, Any], expect: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for step, act in zip(stream_case["steps"], expect["acts"]):
        match = _MONEY.search(step["text"])
        specs.append(
            {
                "satisfied": True,
                "confidence": 0.9,
                "operation": "mention" if act == "negation" else act,
                "value": match.group(0) if match else None,
                "asserted_by": step["speaker"],
            }
        )
    return specs


async def _scripted_specs(specs: list[dict[str, Any]]) -> list[Verdict]:
    judge = ScriptedJudge(specs)
    return [await judge.judge("scripted claim", str(index)) for index in range(len(specs))]


def _claim_reducer_semantics(entry: Mapping[str, Any]) -> tuple[str, ...]:
    fixture_ids: list[str] = []
    for stream_case, label in _fixture_pairs(entry):
        fixture_ids.append(stream_case["id"])
        expect = label["expect"]
        verdicts = asyncio.run(_scripted_specs(_claim_specs(stream_case, expect)))
        authority = None
        if expect.get("authority_required"):
            authority = ConcernAuthority(frozenset({str(expect["authority_required"])}))
        state, outcomes = fold_verdicts(
            verdicts,
            claim_id=str(expect.get("claim_id", entry.get("claim_id", "claim"))),
            authority=authority,
            trace=False,
        )
        _require(
            [outcome.value_after for outcome in outcomes] == expect["value_after"],
            f"{stream_case['id']}: governed value history mismatch",
        )
        _require(state.value == expect["final_value"], f"{stream_case['id']}: final claim mismatch")
        blocked = list(expect.get("blocked_by_authority") or [False] * len(outcomes))
        _require(
            [outcome.downgraded for outcome in outcomes] == blocked,
            f"{stream_case['id']}: authority downgrade mismatch",
        )
        chain = expect.get("revision_chain", [])
        _require(
            {revision["turn_index"] for revision in chain}
            == {index for index, outcome in enumerate(outcomes) if outcome.applied},
            f"{stream_case['id']}: revision chain does not cover every applied claim change",
        )
        for revision in chain:
            outcome = outcomes[revision["turn_index"]]
            _require(
                outcome.revision_id == revision["revision_id"],
                f"{stream_case['id']}: revision id mismatch at turn {revision['turn_index']}",
            )
            _require(
                outcome.supersedes == revision["supersedes"],
                f"{stream_case['id']}: supersession mismatch at turn {revision['turn_index']}",
            )
    return tuple(fixture_ids)


def _goal() -> Goal:
    return Goal(id="long-horizon", condition="scripted condition", say="fired", state={"type": "watch"})


def _episode_owned_incident_action_suppression(
    entry: Mapping[str, Any],
) -> tuple[str, ...]:
    fixture_ids: list[str] = []
    for stream_case, label in _fixture_pairs(entry):
        fixture_ids.append(stream_case["id"])
        expect = label["expect"]
        gate = Gate(max_keys=128, key_ttl_s=expect.get("key_ttl_s"))
        executor = BehaviorExecutor(gate)
        log = EpisodeLog(
            "long-horizon",
            policy=EpisodePolicy(continuity_gap_s=max(step["t"] for step in stream_case["steps"]) + 1.0),
        )
        goal = _goal()
        program = BehaviorProgram(sense="judge")
        actions: list[str] = []
        for step, verdict in zip(stream_case["steps"], _verdicts(stream_case["steps"])):
            decision = log.observe_decision(
                step["t"], step["frame"], subject_key=verdict.subject_key
            )
            fired = executor.fire_judged(
                goal,
                Verdict(verdict.satisfied, verdict.confidence),
                (goal.id, "scene"),
                program,
                t=step["t"],
                fold=executor.fold_episode(decision, mutation="confirmed"),
            )
            actions.append(fired.action if fired is not None else "silent")
        _require(actions == expect["actions"], f"{stream_case['id']}: action trace mismatch {actions!r}")
        _require(
            sum(action == "fire" for action in actions) == expect["incident_count"],
            f"{stream_case['id']}: generated incident/action count mismatch",
        )
    return tuple(fixture_ids)


def _bounded_episode_log_state(entry: Mapping[str, Any]) -> tuple[str, ...]:
    fixture_ids: list[str] = []
    for stream_case, label in _fixture_pairs(entry):
        fixture_ids.append(stream_case["id"])
        expect = label["expect"]
        drops: list[tuple[str, str]] = []
        log = EpisodeLog(
            "long-horizon-bounded",
            max_episodes=int(expect["max_episodes"]),
            on_drop=lambda episode, reason: drops.append((episode.id, reason)),
        )
        opened = 0
        for step, verdict in zip(stream_case["steps"], _verdicts(stream_case["steps"])):
            decision = log.observe_decision(
                step["t"], step["frame"], subject_key=verdict.subject_key
            )
            opened += int(decision.is_new)
            _require(
                log.count() <= expect["max_episodes"],
                f"{stream_case['id']}: episode memory exceeded its count bound",
            )
        _require(opened == expect["episode_count"], f"{stream_case['id']}: opened episode mismatch")
        _require(
            log.count() == expect["retained_episode_count"],
            f"{stream_case['id']}: retained episode count mismatch",
        )
        _require(
            len(drops) == expect["dropped_episode_count"],
            f"{stream_case['id']}: observable drop count mismatch",
        )
        _require(
            all(reason == expect["drop_reason"] for _, reason in drops),
            f"{stream_case['id']}: unexpected eviction reason",
        )
        _require(
            [episode["subject_key"] for episode in log.summaries()] == expect["retained_subjects"],
            f"{stream_case['id']}: wrong episodes retained after eviction",
        )
    return tuple(fixture_ids)


_HANDLERS: dict[str, Callable[[Mapping[str, Any]], tuple[str, ...]]] = {
    "episode_boundaries": _episode_boundaries,
    "timestamped_session_recurrence": _timestamped_session_recurrence,
    "subject_association": _subject_association,
    "claim_reducer_semantics": _claim_reducer_semantics,
    "episode_owned_incident_action_suppression": _episode_owned_incident_action_suppression,
    "bounded_episode_log_state": _bounded_episode_log_state,
}


def run_long_horizon_track(
    manifest: Mapping[str, Any] | None = None,
) -> LongHorizonTrackReport:
    """Run every manifest entry, returning failures and declared unsupported requirements."""
    doc = _validate_manifest(manifest) if manifest is not None else load_long_horizon_manifest()
    results: list[LongHorizonCaseResult] = []
    for entry in doc["cases"]:
        requirements = tuple(entry["requirements"])
        if entry.get("expected_status") == "unsupported":
            results.append(
                LongHorizonCaseResult(
                    id=entry["id"],
                    requirements=requirements,
                    status="unsupported",
                    detail=entry["reason"],
                )
            )
            continue
        try:
            fixture_cases = _HANDLERS[entry["handler"]](entry)
        except Exception as exc:  # noqa: BLE001 - a release gate reports every failure, then fails closed
            results.append(
                LongHorizonCaseResult(
                    id=entry["id"],
                    requirements=requirements,
                    status="fail",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            results.append(
                LongHorizonCaseResult(
                    id=entry["id"],
                    requirements=requirements,
                    status="pass",
                    fixture_cases=fixture_cases,
                )
            )
    return LongHorizonTrackReport(
        track=str(doc["track"]),
        synthetic=bool(doc["synthetic"]),
        session_scope=bool(doc["session_scope"]),
        quality_claim=bool(doc["quality_claim"]),
        results=tuple(results),
    )
