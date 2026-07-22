"""The profile-derivation job: ``(labeled fixtures, pre-registered bars) -> profile``.

THE BRIDGE the MVP thesis rests on. A new environment is supposed to yield a **profile**, not a code
change. This module is that claim made executable: point it at a labeled fixture directory and an
operator-declared LADDER of candidate configurations, and it scores every rung, stamps the run's
provenance, and — only if bars were supplied and met — emits a provenance-stamped, content-addressed
:class:`~percept.profile_versioning.VersionedProfile` alongside a scorecard a human can read.

What this module refuses to do
------------------------------

**It does not invent an answer.** The bars, the held-out split, the ladder and its ORDER are all
INPUTS. With no bars file the job reports the whole ladder and declines to declare a winner
(:data:`DerivationVerdict.NOT_ASSESSED`); with bars that nothing meets it says FAIL and still emits no
profile. There is no default threshold anywhere in this file, and no code path that picks a rung on a
metric the operator did not declare a bar on. Tuning a threshold until a fixture goes green is the one
failure this job exists to prevent, so the job owns none of the thresholds.

**It does not reimplement anything.** Matching is
:func:`percept.harness.golden_eval.score` (the deterministic §5a matcher). The funnel/cost numbers are
:class:`percept.tracing.TraceSummary`. Paired deltas, their bootstrap CIs, the scope table (ADR-006),
the held-out split loader, the bars document and the bar evaluator all belong to
:mod:`percept.eval.differential` — a ladder is that same paired discipline applied over N rungs
instead of an A/B pair, so every rung r is scored as the *candidate* arm of a paired comparison
against rung 0. Serialization, content hashing, provenance and the registry diff all belong to
:mod:`percept.profile_versioning`.

Two axes, because ADR-005 says one is not enough
------------------------------------------------

:class:`~percept.watchprofile.WatchProfile` is CAMERA-SHAPED — its ``environment`` axis is a camera
mount and its resolution chain is goal-temporality / camera-mount / start-state. Two of the client
probes that motivated this card were not cameras at all (a screen recording; a conversation). So a
rung here is a PAIR: a :class:`~percept.sourceprofile.SourceProfile` (what KIND of world is on the
other end of the stream) times a :class:`~percept.watchprofile.WatchProfile` (how to tune the eye
*within* that world). The derived artifact names both. Per ADR-005's Consequences section, overloading
``WatchProfile.environment`` with a source-class value would silently break the registry's wildcard
fallback chain — so :func:`assert_source_class_not_overloaded` rejects a ladder that tries it, loudly,
before anything is scored.

Pre-registration
----------------

Everything that must be fixed BEFORE the numbers exist — the ladder (every rung's profile body and its
declared order), the held-out split, the bars, and the fixture set's own content hash — goes into a
:class:`DerivationPreRegistration` whose ``registration_hash`` covers all of it. Pass the hash back in
with ``expected_registration_hash`` and a configuration that moved after the bars were declared raises
:class:`~percept.eval.differential.ConfigDriftError` and scores NOTHING. The bars hash and the fixture
set hash are stamped into the emitted profile, so a later reader can prove the bars were not moved
after seeing the numbers.

Ladder file format (JSON, or YAML if PyYAML is installed)::

    {
      "source_profile": "physical_scene",
      "declared_at": "2026-07-20",
      "rungs": [
        {"name": "baseline", "description": "shipped defaults",
         "profile": { ... percept.profile_versioning.profile_to_dict body ... }},
        {"name": "block-max", "description": "+ localized salience", "profile": { ... }}
      ]
    }

``rungs[0]`` is the ladder's BASELINE: every delta is measured against it. Each rung reads its
predictions from ``<fixtures>/<fixture_id>/<fixture_dir>/`` (``fixture_dir`` defaults to the rung
name), exactly the arm layout ``percept eval`` already uses.

CLI::

    percept-derive-profile --fixtures ./fixtures --ladder ladder.json --bars bars.json \\
                           --out-profile derived.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..profile_versioning import (
    VersionedProfile,
    content_hash as profile_content_hash,
    profile_from_dict,
    profile_to_dict,
    stamp,
)
from ..sourceprofile import (
    DEFAULT_SOURCE_REGISTRY,
    SourceProfile,
    SourceProfileRegistry,
    get_source_profile,
)
from ..tracing import content_hash
from ..watchprofile import WatchProfile
from . import SPLIT_NAMES, SplitError
from .differential import (
    BarResult,
    BarsDocument,
    ConfigDriftError,
    Delta,
    EvalInputError,
    Fixture,
    HeldOutSplit,
    PairedFixtureResult,
    REPORTED_DELTAS,
    _load_structured,
    _score_arm,
    evaluate_bars,
    load_bars,
    load_fixtures,
    load_held_out_split,
    paired_delta,
)

__all__ = [
    "DerivationInputError",
    "LadderRung",
    "Ladder",
    "assert_source_class_not_overloaded",
    "fixture_set_hash",
    "bars_hash",
    "DerivationPreRegistration",
    "RungResult",
    "DerivationVerdict",
    "DerivationReport",
    "DerivedProfile",
    "load_ladder",
    "derive_profile",
    "render_scorecard",
    "build_parser",
    "main",
]

#: Schema id stamped into the emitted artifacts, so a consumer can tell what it is reading.
REPORT_SCHEMA = "percept.eval.derivation/1"
DERIVED_PROFILE_SCHEMA = "percept.eval.derivation.profile/1"


class DerivationInputError(EvalInputError):
    """A ladder file, fixture set or bars document is missing, malformed or self-contradictory."""


# ---------------------------------------------------------------------------
# 1. The ladder — the operator's candidate configurations, in the operator's order
# ---------------------------------------------------------------------------


def assert_source_class_not_overloaded(
    profile: WatchProfile, *, registry: SourceProfileRegistry = DEFAULT_SOURCE_REGISTRY
) -> None:
    """Refuse a WatchProfile whose ``environment`` is really a SOURCE CLASS (ADR-005).

    ``environment`` is a camera mount. Writing ``environment="screen_document"`` there looks like it
    works and then silently breaks the registry's wildcard fallback chain
    (``(env, goal, "*") -> (env, "*", "*") -> ("*", "*", "*")``), because nothing else in the codebase
    will ever key off that value. A source class belongs on the SourceProfile axis, which is why a
    rung carries both. Checked before scoring so the error arrives before any number does.
    """
    from .. import source_profiles_default  # noqa: F401, PLC0415 — import seeds the registry

    if profile.environment in registry:
        raise DerivationInputError(
            f"WatchProfile {profile.name!r} sets environment={profile.environment!r}, which is a "
            f"registered SOURCE PROFILE name, not a camera mount. ADR-005: the WatchProfile "
            f"ontology is camera-shaped and must not be overloaded with source-class values — that "
            f"breaks the registry's wildcard fallback chain. Name the source on the rung's "
            f"'source_profile' field instead and leave 'environment' a camera mount."
        )


@dataclass(frozen=True)
class LadderRung:
    """One candidate configuration on the ladder: a source class x an eye tuning, plus where its
    predictions live.

    - ``name`` — the rung's label; also the default fixture sub-directory.
    - ``watch_profile`` — the candidate :class:`~percept.watchprofile.WatchProfile`.
    - ``source_profile`` — the :class:`~percept.sourceprofile.SourceProfile` this rung ran against.
    - ``fixture_dir`` — arm directory under each fixture (defaults to ``name``).
    - ``description`` — the operator's one line about what this rung changes.
    """

    name: str
    watch_profile: WatchProfile
    source_profile: SourceProfile
    fixture_dir: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DerivationInputError(f"ladder rung name must be a non-empty string, got {self.name!r}")
        if not self.fixture_dir:
            object.__setattr__(self, "fixture_dir", self.name)
        assert_source_class_not_overloaded(self.watch_profile)

    @property
    def profile_hash(self) -> str:
        """The rung's WatchProfile content address — its identity, order-independent."""
        return profile_content_hash(self.watch_profile)

    def identity_payload(self) -> dict[str, Any]:
        """Everything about this rung that MUST NOT change after the bars were declared."""
        return {
            "name": self.name,
            "fixture_dir": self.fixture_dir,
            "watch_profile": profile_to_dict(self.watch_profile),
            "watch_profile_hash": self.profile_hash,
            "source_profile": self.source_profile.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class Ladder:
    """The ordered candidate set. ``rungs[0]`` is the BASELINE every delta is measured against.

    The ORDER is the operator's, and it is load-bearing: when more than one rung meets every supplied
    bar, the winner is the EARLIEST such rung in this declared order (see
    :meth:`DerivationReport.winner`). That rule is stated up front and hashed into the
    pre-registration precisely so it cannot be chosen after the numbers are in.
    """

    rungs: tuple[LadderRung, ...]
    source: str = ""
    declared_at: str = ""

    def __post_init__(self) -> None:
        if len(self.rungs) < 2:
            raise DerivationInputError(
                "a ladder needs at least 2 rungs (a baseline and at least one candidate); a "
                f"single configuration is not an ablation, got {len(self.rungs)}"
            )
        names = [r.name for r in self.rungs]
        if len(set(names)) != len(names):
            raise DerivationInputError(f"ladder rung names must be unique, got {names}")
        dirs = [r.fixture_dir for r in self.rungs]
        if len(set(dirs)) != len(dirs):
            raise DerivationInputError(
                f"ladder rung fixture_dirs must be unique (two rungs would read the same "
                f"predictions and their delta would be a tautology), got {dirs}"
            )
        source_names = {r.source_profile.name for r in self.rungs}
        if len(source_names) != 1:
            raise DerivationInputError(
                f"every rung of one ladder must name the SAME source profile — a ladder tunes the "
                f"eye WITHIN one kind of world; comparing across source classes compares worlds, "
                f"not knobs. Got {sorted(source_names)}."
            )

    @property
    def baseline(self) -> LadderRung:
        return self.rungs[0]

    @property
    def source_profile(self) -> SourceProfile:
        return self.rungs[0].source_profile

    def identity_payload(self) -> dict[str, Any]:
        return {
            "winner_rule": (
                "the earliest rung, in the operator's declared order, that meets EVERY supplied bar"
            ),
            "rungs": [r.identity_payload() for r in self.rungs],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "declared_at": self.declared_at,
            "source_profile": self.source_profile.to_dict(),
            **self.identity_payload(),
        }


def load_ladder(
    path: str | Path, *, registry: SourceProfileRegistry | None = None
) -> Ladder:
    """Load a ladder document (see the module docstring for the format)."""
    data = _load_structured(path)
    raw_rungs = data.get("rungs")
    if not isinstance(raw_rungs, list) or not raw_rungs:
        raise DerivationInputError(f"{path}: ladder has no non-empty 'rungs' list")
    default_source = data.get("source_profile")

    rungs: list[LadderRung] = []
    for index, raw in enumerate(raw_rungs):
        if not isinstance(raw, Mapping):
            raise DerivationInputError(f"{path}: rungs[{index}] is not a mapping")
        source_name = raw.get("source_profile", default_source)
        if not source_name:
            raise DerivationInputError(
                f"{path}: rungs[{index}] names no source profile and the document declares no "
                f"top-level 'source_profile'. A derivation must say what KIND of world it ran "
                f"against (ADR-005) — it is never assumed to be a camera."
            )
        source = (
            registry.resolve(str(source_name)) if registry is not None
            else get_source_profile(str(source_name))
        )
        body = raw.get("profile")
        if not isinstance(body, Mapping):
            raise DerivationInputError(
                f"{path}: rungs[{index}] has no 'profile' body (a serialized WatchProfile — see "
                f"percept.profile_versioning.profile_to_dict)"
            )
        rungs.append(
            LadderRung(
                name=str(raw.get("name") or body.get("name") or f"rung{index}"),
                watch_profile=profile_from_dict(body),
                source_profile=source,
                fixture_dir=str(raw.get("fixture_dir") or ""),
                description=str(raw.get("description") or ""),
            )
        )
    return Ladder(
        rungs=tuple(rungs),
        source=str(path),
        declared_at=str(data.get("declared_at") or ""),
    )


# ---------------------------------------------------------------------------
# 2. Content addresses for the two things a reader must be able to prove did not move
# ---------------------------------------------------------------------------


def fixture_set_hash(fixtures: Sequence[Fixture]) -> str:
    """``sha256`` over the LABELS themselves — ids, goal shapes and every episode's window.

    Not over file paths or mtimes: two checkouts of the same labeled corpus must hash identically,
    and a single relabeled episode must change the hash.
    """
    payload = [
        {
            "fixture_id": f.fixture_id,
            "goal_shape": f.goal_shape,
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "event_start_ts": e.ground_truth.event_start_ts,
                    "decision_eligible_ts": e.ground_truth.decision_eligible_ts,
                    "episode_end_ts": e.episode_end_ts,
                    "allowed_lateness_s": e.allowed_lateness_s,
                    "annotation_version": e.ground_truth.annotation_version,
                    "annotator_or_fixture_id": e.ground_truth.annotator_or_fixture_id,
                }
                for e in sorted(f.episodes, key=lambda e: e.episode_id)
            ],
        }
        for f in sorted(fixtures, key=lambda f: f.fixture_id)
    ]
    return content_hash(payload)


def bars_hash(bars: BarsDocument) -> str:
    """``sha256`` over the declared bars — the number a later reader checks to prove the bars they
    are looking at are the bars that were declared. An undeclared document still hashes (to the hash
    of an empty bar list), so the field is never blank and never ambiguous."""
    return content_hash([b.to_dict() for b in bars.bars])


# ---------------------------------------------------------------------------
# 3. Pre-registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivationPreRegistration:
    """Everything fixed BEFORE the numbers existed, content-addressed into one hash."""

    ladder: Ladder
    split: HeldOutSplit
    bars: BarsDocument
    fixture_set_hash: str
    split_name: str = "test"
    declared_at: str = ""
    expected_registration_hash: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "ladder": self.ladder.identity_payload(),
            "split": self.split.to_dict(),
            "bars": [b.to_dict() for b in self.bars.bars],
            "fixture_set_hash": self.fixture_set_hash,
        }

    @property
    def registration_hash(self) -> str:
        return content_hash(self.payload())

    @property
    def bars_hash(self) -> str:
        return bars_hash(self.bars)

    @property
    def seed(self) -> int:
        """Deterministic bootstrap seed — derived from the registration, never from wall time."""
        return int(self.registration_hash[:12], 16)

    @property
    def eval_run_id(self) -> str:
        """A stable id for THIS derivation: the same inputs always name the same run."""
        return f"derive-{self.registration_hash[:12]}"

    def assert_no_drift(self) -> None:
        if self.expected_registration_hash is None:
            return
        actual = self.registration_hash
        if actual != self.expected_registration_hash:
            raise ConfigDriftError(
                "REFUSING TO DERIVE: the configuration changed after the bars were declared.\n"
                f"  pre-registered registration_hash: {self.expected_registration_hash}\n"
                f"  configuration presented now:      {actual}\n"
                "Every rung's profile body, the ladder ORDER, the held-out split, the bars and the "
                "fixture set's labels are covered by this hash. Re-declare the bars against the new "
                "configuration and re-run; do not adjust the configuration until it passes."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_at": self.declared_at,
            "registration_hash": self.registration_hash,
            "expected_registration_hash": self.expected_registration_hash,
            "eval_run_id": self.eval_run_id,
            "bootstrap_seed": self.seed,
            "bars_hash": self.bars_hash,
            "bars_source": self.bars.source,
            "bars_declared_by": self.bars.declared_by,
            "ladder_source": self.ladder.source,
            **self.payload(),
        }


# ---------------------------------------------------------------------------
# 4. Per-rung results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RungResult:
    """One rung, fully scored — including the losers. A scored loser is data; a void is not.

    ``pairs`` holds this rung as the *candidate* arm of a paired comparison against the ladder's
    baseline rung, so ``deltas`` are the same paired deltas with their bootstrap CIs, computed by
    the same function ``percept eval`` uses.
    """

    rung: LadderRung
    pairs: tuple[PairedFixtureResult, ...]
    deltas: tuple[Delta, ...]
    bar_results: tuple[BarResult, ...]
    is_baseline: bool = False

    @property
    def scored_pairs(self) -> tuple[PairedFixtureResult, ...]:
        return tuple(p for p in self.pairs if p.scored)

    def mean(self, metric: str) -> float | None:
        from .differential import _fixture_metric, _mean  # noqa: PLC0415 — one metric authority

        vals = [v for v in (_fixture_metric(p.candidate, metric) for p in self.scored_pairs)
                if v is not None]
        return _mean(vals)

    def delta(self, metric: str) -> Delta | None:
        return next((d for d in self.deltas if d.metric == metric), None)

    @property
    def bars_met(self) -> bool | None:
        """``True`` only if EVERY supplied bar was measurable and passed. ``None`` when no bar was
        supplied or a bar could not be measured — a partial pass is never a pass."""
        if not self.bar_results:
            return None
        if any(r.passed is None for r in self.bar_results):
            return None
        return all(r.passed for r in self.bar_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung.to_dict(),
            "is_baseline": self.is_baseline,
            "scored_fixtures": [p.fixture.fixture_id for p in self.scored_pairs],
            "means": {m: self.mean(m) for m in ("recall", "precision", "false_fires", "cost_usd")},
            "fixtures": [p.to_dict() for p in self.pairs],
            "deltas_vs_baseline": [d.to_dict() for d in self.deltas],
            "bars": [r.to_dict() for r in self.bar_results],
            "bars_met": self.bars_met,
        }


class DerivationVerdict(str, Enum):
    """PASS names a winner. FAIL means bars were supplied and NO rung met them. NOT_ASSESSED means
    the job declined to declare anything — and always says why."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ASSESSED = "NOT_ASSESSED"


# ---------------------------------------------------------------------------
# 5. The derived artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedProfile:
    """The output of a successful derivation: BOTH axes, provenance-stamped and content-addressed.

    ``watch_profile`` is a :class:`~percept.profile_versioning.VersionedProfile` — so it
    round-trips, verifies its own content hash on load, and can be diffed against the shipped
    registry with :func:`percept.profile_versioning.diff_registries` (and reversed with
    :meth:`~percept.profile_versioning.RegistryDiff.inverse`). ``source_profile`` names the KIND of
    world it was derived on, because :class:`~percept.watchprofile.WatchProfile` cannot (ADR-005).
    """

    watch_profile: VersionedProfile
    source_profile: SourceProfile
    registration_hash: str
    fixture_set_hash: str
    bars_hash: str
    split: HeldOutSplit
    ladder_rung_names: tuple[str, ...]
    winning_rung: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DERIVED_PROFILE_SCHEMA,
            "watch_profile": self.watch_profile.to_dict(),
            "source_profile": self.source_profile.to_dict(),
            "derivation": {
                "eval_run_id": self.watch_profile.provenance.eval_run_id,
                "harness_version": self.watch_profile.provenance.harness_version,
                "created_at": self.watch_profile.provenance.created_at,
                "registration_hash": self.registration_hash,
                "fixture_set_hash": self.fixture_set_hash,
                "bars_hash": self.bars_hash,
                "split": self.split.to_dict(),
                "ladder": list(self.ladder_rung_names),
                "winning_rung": self.winning_rung,
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


@dataclass(frozen=True)
class DerivationReport:
    """The whole answer: the pre-registration, every rung (winners AND losers), and the verdict."""

    registration: DerivationPreRegistration
    rungs: tuple[RungResult, ...]
    verdict: DerivationVerdict
    verdict_reason: str
    winner: RungResult | None = None
    derived: DerivedProfile | None = None
    scope_summary: tuple[Any, ...] = ()

    def rung(self, name: str) -> RungResult | None:
        return next((r for r in self.rungs if r.rung.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "pre_registration": self.registration.to_dict(),
            "scope": [s.to_dict() for s in self.scope_summary],
            "ladder": [r.to_dict() for r in self.rungs],
            "verdict": self.verdict.value,
            "verdict_reason": self.verdict_reason,
            "winner": None if self.winner is None else self.winner.rung.name,
            "derived_profile": None if self.derived is None else self.derived.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# ---------------------------------------------------------------------------
# 6. The job
# ---------------------------------------------------------------------------


def _harness_version() -> str:
    from .. import __version__  # noqa: PLC0415 — read lazily, no import cycle

    return str(__version__)


def derive_profile(
    *,
    fixtures_root: str | Path,
    ladder: Ladder,
    bars: BarsDocument,
    split_name: str = "test",
    expected_registration_hash: str | None = None,
    declared_at: str = "",
    eval_run_id: str | None = None,
    created_at: str | None = None,
) -> DerivationReport:
    """Score the whole ladder and, ONLY if bars were supplied and met, emit a derived profile.

    Order is the discipline: fixtures and split are loaded, the pre-registration is assembled and
    drift-checked, and only then is anything scored — so a drifted configuration raises instead of
    producing a number someone could quote.
    """
    root = Path(fixtures_root)
    fixtures = load_fixtures(root)
    split = load_held_out_split(root, fixtures, split_name=split_name)

    registration = DerivationPreRegistration(
        ladder=ladder,
        split=split,
        bars=bars,
        fixture_set_hash=fixture_set_hash(fixtures),
        split_name=split_name,
        declared_at=declared_at or ladder.declared_at,
        expected_registration_hash=expected_registration_hash,
    )
    registration.assert_no_drift()

    held = set(split.held_out)
    # One scoring pass per (fixture, rung), through `_score_arm` — which is
    # golden_eval.score() plus tracing.load_trace(), i.e. the same code doctor and eval run.
    outcomes = {
        rung.name: {f.fixture_id: _score_arm(f, rung) for f in fixtures} for rung in ladder.rungs
    }

    baseline = ladder.baseline
    rung_results: list[RungResult] = []
    for rung in ladder.rungs:
        pairs = tuple(
            PairedFixtureResult(
                fixture=f,
                baseline=outcomes[baseline.name][f.fixture_id],
                candidate=outcomes[rung.name][f.fixture_id],
                in_held_out=f.fixture_id in held,
            )
            for f in fixtures
        )
        scored = tuple(p for p in pairs if p.scored)
        deltas = tuple(paired_delta(scored, m, seed=registration.seed) for m in REPORTED_DELTAS)
        bar_results = (
            evaluate_bars(bars.bars, deltas, scored)
            if (bars.declared and split.declared and scored)
            else ()
        )
        rung_results.append(
            RungResult(
                rung=rung,
                pairs=pairs,
                deltas=deltas,
                bar_results=bar_results,
                is_baseline=rung.name == baseline.name,
            )
        )

    winner, verdict, reason = _select_winner(registration, tuple(rung_results))

    derived = None
    if winner is not None:
        derived = _stamp_winner(
            registration, winner, eval_run_id=eval_run_id, created_at=created_at,
            fixtures_root=str(root),
        )

    seen: dict[str, Any] = {}
    for f in fixtures:
        seen.setdefault(f.scope.goal_shape, f.scope)

    return DerivationReport(
        registration=registration,
        rungs=tuple(rung_results),
        verdict=verdict,
        verdict_reason=reason,
        winner=winner,
        derived=derived,
        scope_summary=tuple(seen[k] for k in sorted(seen)),
    )


def _select_winner(
    registration: DerivationPreRegistration, rungs: Sequence[RungResult]
) -> tuple[RungResult | None, DerivationVerdict, str]:
    """Declare a winner, or refuse and say exactly why. NEVER invents a bar or a tie-break metric.

    The only selection rule is the one hashed into the pre-registration: the EARLIEST rung, in the
    operator's declared ladder order, that meets EVERY supplied bar. When several rungs pass, the
    ones that lost the tie are still reported — they are data, and the operator can re-order the
    ladder and re-derive if their preference differs.
    """
    if not registration.bars.declared:
        return (
            None,
            DerivationVerdict.NOT_ASSESSED,
            "NO BARS SUPPLIED. The full ladder is reported — every rung, including the losers — and "
            "no winner is declared. percept does not invent a pass/fail threshold, and it will not "
            "pick a 'best' rung on a metric nobody declared. Supply --bars to get a derived profile.",
        )
    if not registration.split.declared:
        return (
            None,
            DerivationVerdict.NOT_ASSESSED,
            "Bars were supplied but the held-out split is UNDECLARED (no splits.json under the "
            "fixtures root). Deriving a profile against a split chosen after seeing the ladder is "
            "tune-to-green; declare train/dev/test and re-run.",
        )
    scorable = [r for r in rungs if r.scored_pairs]
    if not scorable:
        return (
            None,
            DerivationVerdict.NOT_ASSESSED,
            "Bars were supplied but no fixture in the held-out split has a scorable goal shape (see "
            "the scope table). Nothing was scored, so nothing won.",
        )
    passing = [r for r in rungs if r.bars_met is True]
    if not passing:
        unassessed = [r.rung.name for r in rungs if r.bars_met is None and r.bar_results]
        detail = (
            f" ({len(unassessed)} rung(s) had at least one unmeasurable bar: "
            f"{', '.join(unassessed)} — a partial pass is not a pass)"
            if unassessed else ""
        )
        return (
            None,
            DerivationVerdict.FAIL,
            f"NO RUNG met every supplied bar{detail}. No profile is emitted. The bars are the "
            f"operator's and are not moved to produce a winner — the ladder below is the honest "
            f"result, and the next move is a better candidate, not a lower bar.",
        )
    winner = passing[0]
    others = [r.rung.name for r in passing[1:]]
    also = (
        f" {len(others)} later rung(s) also met the bars ({', '.join(others)}); they are reported "
        f"but did not win, because the winner rule is declared-order-first and was hashed into the "
        f"pre-registration before scoring."
        if others else ""
    )
    return (
        winner,
        DerivationVerdict.PASS,
        f"Rung {winner.rung.name!r} is the earliest rung in the operator's declared order that met "
        f"every supplied bar on the held-out split ({len(winner.scored_pairs)} fixtures).{also}",
    )


def _stamp_winner(
    registration: DerivationPreRegistration,
    winner: RungResult,
    *,
    eval_run_id: str | None,
    created_at: str | None,
    fixtures_root: str,
) -> DerivedProfile:
    """Wrap the winning rung's WatchProfile in a provenance stamp."""
    recall = winner.mean("recall")
    precision = winner.mean("precision")
    notes = (
        f"Derived by percept profile derivation over {len(winner.scored_pairs)} held-out "
        f"fixture(s) from the ladder {', '.join(r.name for r in registration.ladder.rungs)}. "
        f"Held-out mean recall={recall if recall is None else round(recall, 4)}, "
        f"precision={precision if precision is None else round(precision, 4)}. "
        f"All {len(winner.bar_results)} operator-supplied bar(s) met. "
        f"registration_hash={registration.registration_hash} "
        f"fixture_set_hash={registration.fixture_set_hash} bars_hash={registration.bars_hash}"
    )
    versioned = stamp(
        winner.rung.watch_profile,
        eval_run_id=eval_run_id or registration.eval_run_id,
        harness_version=_harness_version(),
        created_at=created_at,
        source=f"profile-derivation:{fixtures_root}",
        notes=notes,
    )
    return DerivedProfile(
        watch_profile=versioned,
        source_profile=winner.rung.source_profile,
        registration_hash=registration.registration_hash,
        fixture_set_hash=registration.fixture_set_hash,
        bars_hash=registration.bars_hash,
        split=registration.split,
        ladder_rung_names=tuple(r.name for r in registration.ladder.rungs),
        winning_rung=winner.rung.name,
    )


# ---------------------------------------------------------------------------
# 7. The scorecard
# ---------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return "—" if value is None else format(value, spec)


def render_scorecard(report: DerivationReport) -> str:
    """The human-readable scorecard: the whole ladder, the provenance stamp, and the refusal (or the
    derived profile) in plain words. Written for a client debugging percept at 2am."""
    reg = report.registration
    lines: list[str] = []
    add = lines.append

    add("percept — profile derivation scorecard")
    add("=" * 78)
    add("")
    add("SOURCE CLASS (ADR-005: what KIND of world; a WatchProfile cannot say this)")
    sp = reg.ladder.source_profile
    add(f"  source_profile     : {sp.name}  [{sp.stability}]")
    add(f"  modalities         : {', '.join(sorted(sp.modalities))}   temporal_unit: {sp.temporal_unit}")
    add(f"  admission          : {sp.admission_note()}")
    add("")
    add("PRE-REGISTERED BEFORE ANY SCORING")
    add(f"  registration_hash  : {reg.registration_hash}")
    add(f"  eval_run_id        : {reg.eval_run_id}")
    add(f"  fixture_set_hash   : {reg.fixture_set_hash}")
    add(f"  bars_hash          : {reg.bars_hash}")
    add(f"  bars               : {reg.bars.source}"
        f"{' (declared by ' + reg.bars.declared_by + ')' if reg.bars.declared_by else ''}")
    add(f"  ladder             : {reg.ladder.source or '(in-process)'}")
    add(f"  declared_at        : {reg.declared_at or '—'}")
    add(f"  split source       : {reg.split.source}")
    add(f"  held-out ({reg.split.split_name}) : {', '.join(reg.split.held_out) or '—'}")
    add(f"  exploratory        : {', '.join(reg.split.exploratory) or '—'} "
        f"(reported, never scored against bars)")
    add(f"  bootstrap seed     : {reg.seed} (from registration_hash; not wall time)")
    add(f"  winner rule        : {reg.ladder.identity_payload()['winner_rule']}")
    add("")
    add("SCOPE — what this matcher can honestly validate (ADR-006)")
    for s in report.scope_summary:
        add(f"  {s.goal_shape:<26} {s.verdict.value}")
        add(f"      {s.reason}")
    add("")
    add("THE LADDER — every candidate, including the losers (a scored loser is data)")
    add(f"  {'rung':<20}{'profile hash':<15}{'recall':<10}{'prec':<10}{'ff':<8}"
        f"{'cost usd':<12}{'Δrecall':<12}bars")
    for r in report.rungs:
        d = r.delta("recall")
        bars_state = (
            "—" if r.bars_met is None and not r.bar_results
            else ("MET" if r.bars_met else ("unassessed" if r.bars_met is None else "not met"))
        )
        mark = "*" if report.winner is not None and r is report.winner else " "
        add(f" {mark}{r.rung.name:<20}{r.rung.profile_hash[:12]:<15}"
            f"{_fmt(r.mean('recall')):<10}{_fmt(r.mean('precision')):<10}"
            f"{_fmt(r.mean('false_fires')):<8}{_fmt(r.mean('cost_usd')):<12}"
            f"{('baseline' if r.is_baseline else _fmt(None if d is None else d.mean_delta)):<12}"
            f"{bars_state}")
        if r.rung.description:
            add(f"      {r.rung.description}")
        for br in r.bar_results:
            state = "—" if br.passed is None else ("PASS" if br.passed else "FAIL")
            add(f"      bar {br.bar.metric:<18}{br.bar.direction} {br.bar.value:<8} "
                f"observed={_fmt(br.observed):<10}{state}"
                f"{('  ' + br.note) if br.note else ''}")
    add("")
    add("PAIRED DELTAS vs the ladder baseline "
        f"({reg.ladder.baseline.name!r}; 95% percentile bootstrap)")
    for r in report.rungs:
        if r.is_baseline:
            continue
        add(f"  {r.rung.name}")
        for d in r.deltas:
            ci = "—" if d.ci_low is None else f"[{_fmt(d.ci_low)}, {_fmt(d.ci_high)}]"
            add(f"    {d.metric:<14} n={d.n_pairs:<4} mean Δ={_fmt(d.mean_delta):<12} CI={ci:<26}"
                f"{'' if d.ci_excludes_zero is None else ('excludes 0' if d.ci_excludes_zero else 'includes 0')}")
    add("")
    add(f"VERDICT: {report.verdict.value}")
    add(f"  {report.verdict_reason}")
    add("")
    if report.derived is None:
        add("DERIVED PROFILE: none emitted.")
        add("  A profile is emitted only when bars were supplied AND a rung met them. This is the "
            "refusal, not a failure of the job.")
    else:
        vp = report.derived.watch_profile
        add("DERIVED PROFILE (provenance-stamped, content-addressed)")
        add(f"  name           : {vp.profile.name}")
        add(f"  content_hash   : {vp.content_hash}")
        add(f"  source_profile : {report.derived.source_profile.name}")
        add(f"  eval_run_id    : {vp.provenance.eval_run_id}")
        add(f"  harness        : {vp.provenance.harness_version}   created_at: "
            f"{vp.provenance.created_at}")
        add("  Diff it against the shipped registry with "
            "percept.profile_versioning.diff_registries (and .inverse() to revert).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="percept derive-profile",
        description=(
            "Derive a provenance-stamped WatchProfile from labeled fixtures and a pre-registered "
            "ladder of candidate configurations. Without --bars it reports the ladder and refuses "
            "to declare a winner."
        ),
    )
    p.add_argument("--fixtures", required=True, help="fixtures root directory")
    p.add_argument("--ladder", required=True, help="ladder document (.json/.yaml)")
    p.add_argument("--bars", default=None,
                   help="OPERATOR-SUPPLIED bars file; omit to report the ladder only")
    p.add_argument("--split", default="test", choices=list(SPLIT_NAMES),
                   help="which split is held out (default: test)")
    p.add_argument("--expect-registration-hash", default=None,
                   help="refuse to derive unless the configuration hashes to this")
    p.add_argument("--json", dest="json_out", default=None,
                   help="write the full scorecard as JSON (the whole ladder, losers included)")
    p.add_argument("--out-profile", default=None,
                   help="write the derived profile here (only written when one was derived)")
    p.add_argument("--run-id", default=None, help="override the derived eval_run_id")
    p.add_argument("--declared-at", default="", help="when the bars/split/ladder were declared")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``percept-derive-profile``. 0 = derived or reported, 1 = no rung met the bars, 2 = refused."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = derive_profile(
            fixtures_root=args.fixtures,
            ladder=load_ladder(args.ladder),
            bars=load_bars(args.bars),
            split_name=args.split,
            expected_registration_hash=args.expect_registration_hash,
            declared_at=args.declared_at,
            eval_run_id=args.run_id,
        )
    except (EvalInputError, SplitError) as exc:
        print(f"percept derive-profile: REFUSED — {exc}", file=sys.stderr)
        return 2
    print(render_scorecard(report))
    if args.json_out:
        Path(args.json_out).write_text(report.to_json())
    if args.out_profile:
        if report.derived is None:
            print(
                "percept derive-profile: no profile written to "
                f"{args.out_profile} — no winner was declared (see VERDICT above).",
                file=sys.stderr,
            )
        else:
            Path(args.out_profile).write_text(report.derived.to_json())
    return 1 if report.verdict is DerivationVerdict.FAIL else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
