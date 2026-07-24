"""The optimizer (ADR-008) — a typed SearchSpace over declared-safe tunables, scored through the
SAME ladder discipline :mod:`percept.eval.derivation` already runs, emitting an
:class:`~percept.optimization_lock.OptimizationLock` instead of a bare
:class:`~percept.profile_versioning.VersionedProfile`.

What this module is, precisely
-------------------------------

:mod:`percept.eval.derivation` already answers "which hand-authored WatchProfile wins?" — an
operator declares a :class:`~percept.eval.derivation.Ladder` of full candidate profiles, in a
pre-registered order, and the ladder scores every rung against pre-registered bars. This module
answers a NARROWER, MECHANICAL question on top of that: "within one operator-declared candidate
SHAPE, which value of a few SAFE numeric knobs scores best?" It never invents the shape (which
salience statistic is active, which source class, which plan mode) — that stays an operator-declared
:class:`OptimizerTemplate`, structurally identical to a :class:`~percept.eval.derivation.LadderRung`.
It only explores the SAFE, BOUNDED, NUMERIC dimensions a :class:`SearchSpace` declares on top of that
shape, then hands the resulting candidates to :func:`~percept.eval.derivation.derive_profile`
UNCHANGED — every pre-registration/held-out/bars discipline in that module applies to an
optimizer-generated ladder exactly as it applies to a hand-authored one, because it IS the same
:class:`~percept.eval.derivation.Ladder` object underneath.

Guardrails, hard and independent of the data
---------------------------------------------

``percept.tunables.REGISTRY`` already marks most of the codebase's risky/structural fields
``safe_to_optimize: false``. This module does not trust that alone: :data:`GUARDRAIL_FORBIDDEN_PREFIXES`
is a SEPARATE, hard-coded deny-list checked BEFORE the registry is even consulted, so a future data
change that flipped some field's flag in ``tunables.json`` could not silently hand the optimizer a
path it must never touch. Five things a scored recall number must never be able to move, whatever the
registry says:

- **gate vocabulary** — ``Settings.gate.*`` / ``WatchProfile.gate.*``; ``percept.gate.Gate`` is the
  sole firing authority (ADR-008 Invariants' "One Gate").
- **goal meaning / condition / consequence text** — the entire ``Goal.state.*`` namespace (condition
  predicates, ``fire_when``, ``miss_cost``/``false_fire_cost``, ``protected_floor``, rearm policy):
  WatchPlan's own "explicit user intent" ownership row. An optimizer that could rewrite what a goal
  MEANS would be authoring the goal, not tuning it.
- **envelope / consent** — ``Settings.consent.*`` and the scripted-brain safety guard (ADR-007
  actuation permission); a recall number must never be able to widen what the SDK is allowed to DO.
- **revision checks** — no ``tunables.json`` path represents Y-epic supersession/coalescing logic
  today (nothing to flip); the episode/identity intent surface those checks depend on is already
  inside the blanket ``Goal.state.`` block above.
- **retention / egress** — how long evidence is kept (``Settings.episodes.window_s``, the same field
  :class:`~percept.domain_pack.DomainPack`'s ``retention_max_days`` clamp polices) and memory/byte
  ceilings on what the process holds (``Settings.events.preroll_max_bytes``); a coverage-floor
  declaration (``SourceProfile.parameters.coverage_floor``) that changes when a run reads as
  "degraded" rather than quiet.

Materialization
----------------

A :class:`SearchSpace` point is a plain ``{tunables-path: value}`` mapping — the SAME shape an
:class:`~percept.optimization_lock.OptimizationLock` eventually carries. :func:`apply_search_point`
turns one onto an :class:`OptimizerTemplate`'s base :class:`~percept.watchprofile.WatchProfile` using
the SAME path -> field translation :mod:`percept.resolver` uses at resolve time
(``Settings.edge.motion_gate*`` -> ``stat_thresholds``, ``WatchProfile.cadence.*`` -> the cadence
sub-object) — so a candidate this module scores is byte-identical in shape to what a real
``ConfigResolver.resolve()`` would produce for the same values.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import catalog
from .eval import SPLIT_NAMES, SplitError
from .eval.derivation import (
    DerivationVerdict,
    Ladder,
    LadderRung,
    DerivationReport,
    derive_profile,
    render_scorecard,
)
from .eval.differential import BarsDocument, EvalInputError, _load_structured, load_bars
from .optimization_lock import OptimizationLock
from .sourceprofile import SourceProfile, get_source_profile
from .tracing import content_hash
from .tunables import REGISTRY, TunableSpec
from .watchprofile import WatchProfile

__all__ = [
    "SearchSpaceError",
    "GUARDRAIL_FORBIDDEN_PREFIXES",
    "SearchDimension",
    "SearchSpace",
    "OptimizerTemplate",
    "apply_search_point",
    "expand_templates",
    "OPTIMIZATION_REPORT_SCHEMA",
    "OptimizationReport",
    "optimize",
    "render_optimization_scorecard",
    "load_optimizer_document",
    "build_parser",
    "main",
]


class SearchSpaceError(ValueError):
    """A :class:`SearchSpace`/:class:`SearchDimension`/:class:`OptimizerTemplate` failed a
    shape/safety check, or named a path the optimizer must never touch."""


#: path -> TunableSpec, built once from the tunables registry — the single source of truth this
#: module reads and never duplicates (mirrors percept.resolver._BY_PATH exactly).
_BY_PATH: dict[str, TunableSpec] = {spec.path: spec for spec in REGISTRY}

#: Hard-coded, data-independent deny-list — see the module docstring's "Guardrails" section for the
#: rationale behind each entry. Checked BEFORE the registry's own safe_to_optimize flag, so flipping
#: that flag in tunables.json cannot bypass this.
GUARDRAIL_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "Settings.gate.",
    "WatchProfile.gate.",
    "Goal.state.",
    "Settings.consent.",
    "Percept.create.allow_scripted_brain",
    "Settings.episodes.window_s",
    "Settings.events.preroll_max_bytes",
    "SourceProfile.parameters.coverage_floor",
)


def _assert_path_optimizable(path: str) -> None:
    """Raise :class:`SearchSpaceError` unless ``path`` is a known, declared-safe, non-guardrailed
    tunables path. The guardrail check runs FIRST and is independent of the registry (module
    docstring); the registry check runs second and IS the registry (``safe_to_optimize``)."""
    for prefix in GUARDRAIL_FORBIDDEN_PREFIXES:
        if path == prefix or path.startswith(prefix):
            raise SearchSpaceError(
                f"{path!r} is guardrailed off the optimizer's reach (matches forbidden prefix "
                f"{prefix!r} — gate vocabulary, goal meaning/condition/consequence text, "
                "envelope/consent, revision checks and retention/egress paths are never "
                "optimizable). This check runs BEFORE consulting percept.tunables.REGISTRY, so "
                "flipping safe_to_optimize in tunables.json data cannot bypass it."
            )
    spec = _BY_PATH.get(path)
    if spec is None:
        raise SearchSpaceError(
            f"{path!r} is not a known percept.tunables.REGISTRY path — a SearchSpace may only "
            "cover declared tunables"
        )
    if not spec.safe_to_optimize:
        raise SearchSpaceError(
            f"{path!r} has safe_to_optimize=False in percept.tunables.REGISTRY "
            f"(owner={spec.owner!r}: {spec.note}) — a SearchSpace may only cover declared-safe "
            "paths (ADR-008 Invariants: 'the optimizer tunes only declared-safe paths... a lock "
            "covering any other field is invalid to construct')"
        )


# ---------------------------------------------------------------------------
# 1. SearchSpace — typed, safe-only, grid + random
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchDimension:
    """One tunables path the optimizer may vary, plus either a discrete ``choices`` set (grid- and
    random-searchable) or a continuous ``[low, high]`` range (random-searchable only — a grid needs
    an enumerable set, see :meth:`SearchSpace.grid`)."""

    path: str
    choices: tuple[Any, ...] | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise SearchSpaceError(f"SearchDimension.path must be a non-empty string, got {self.path!r}")
        _assert_path_optimizable(self.path)
        has_choices = self.choices is not None
        has_range = self.low is not None or self.high is not None
        if has_choices and has_range:
            raise SearchSpaceError(f"{self.path}: give EITHER choices= OR low=/high=, not both")
        if not has_choices and not has_range:
            raise SearchSpaceError(f"{self.path}: needs either choices=(...) or low=/high=")
        if has_choices:
            choices = tuple(self.choices)  # type: ignore[arg-type]
            if not choices:
                raise SearchSpaceError(f"{self.path}: choices must be a non-empty sequence")
            object.__setattr__(self, "choices", choices)
        else:
            if self.low is None or self.high is None:
                raise SearchSpaceError(f"{self.path}: a range needs BOTH low= and high=")
            if self.low > self.high:
                raise SearchSpaceError(f"{self.path}: low > high ({self.low} > {self.high})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "choices": list(self.choices) if self.choices is not None else None,
            "low": self.low,
            "high": self.high,
        }


@dataclass(frozen=True)
class SearchSpace:
    """An operator-declared, typed set of :class:`SearchDimension`, each covering ONLY a
    ``tunables.json`` path with ``safe_to_optimize: true`` (checked at construction — a space that
    names an unsafe or guardrailed path raises immediately, never at scoring time)."""

    dimensions: tuple[SearchDimension, ...]

    def __post_init__(self) -> None:
        dims = tuple(self.dimensions)
        if not dims:
            raise SearchSpaceError("a SearchSpace needs at least one dimension")
        paths = [d.path for d in dims]
        if len(set(paths)) != len(paths):
            raise SearchSpaceError(f"SearchSpace dimensions must have unique paths, got {paths}")
        object.__setattr__(self, "dimensions", dims)

    def grid(self) -> tuple[Mapping[str, Any], ...]:
        """Cross product of every dimension's discrete ``choices``. Raises if any dimension declared
        a continuous range instead (grid search needs an enumerable set)."""
        for d in self.dimensions:
            if d.choices is None:
                raise SearchSpaceError(
                    f"{d.path}: grid search needs discrete choices (this dimension declared a "
                    "low/high range instead) — use .sample('random', ...) or declare explicit choices"
                )
        combos = itertools.product(*(d.choices for d in self.dimensions))
        return tuple(
            MappingProxyType({d.path: v for d, v in zip(self.dimensions, combo)}) for combo in combos
        )

    def random(self, n: int, *, seed: int) -> tuple[Mapping[str, Any], ...]:
        """``n`` deterministic random draws (choices sampled uniformly; ranges sampled uniformly
        continuous) — deterministic in ``seed``, never wall-clock/OS entropy, so a re-run with the
        same seed reproduces the same points."""
        if n < 1:
            raise SearchSpaceError(f"random search needs n>=1, got {n}")
        rng = random.Random(seed)
        points: list[Mapping[str, Any]] = []
        for _ in range(n):
            point: dict[str, Any] = {}
            for d in self.dimensions:
                if d.choices is not None:
                    point[d.path] = rng.choice(d.choices)
                else:
                    point[d.path] = rng.uniform(d.low, d.high)  # type: ignore[arg-type]
            points.append(MappingProxyType(point))
        return tuple(points)

    def sample(self, strategy: str = "grid", *, n: int = 8, seed: int = 0) -> tuple[Mapping[str, Any], ...]:
        if strategy == "grid":
            return self.grid()
        if strategy == "random":
            return self.random(n, seed=seed)
        raise SearchSpaceError(f"unknown search strategy {strategy!r}; expected 'grid' or 'random'")

    def to_dict(self) -> dict[str, Any]:
        return {"dimensions": [d.to_dict() for d in self.dimensions]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchSpace":
        dims = []
        for raw in data.get("dimensions", []):
            choices = raw.get("choices")
            dims.append(SearchDimension(
                path=raw["path"],
                choices=tuple(choices) if choices is not None else None,
                low=raw.get("low"),
                high=raw.get("high"),
            ))
        return cls(dimensions=tuple(dims))


# ---------------------------------------------------------------------------
# 2. Materialization — a SearchSpace point onto a WatchProfile candidate
# ---------------------------------------------------------------------------

#: Mirrors percept.resolver._SALIENCE_THRESHOLD_PATH exactly: which stat_thresholds key a
#: Settings.edge.motion_gate* path rides on.
_STAT_THRESHOLD_KEY: dict[str, str] = {
    "Settings.edge.motion_gate": "mean",
    "Settings.edge.motion_gate_block_max": "block_max",
    "Settings.edge.motion_gate_p95": "p95",
}

#: Mirrors WatchProfile.cadence.* field names.
_CADENCE_FIELD: dict[str, str] = {
    "WatchProfile.cadence.base_fps": "base_fps",
    "WatchProfile.cadence.max_fps": "max_fps",
    "WatchProfile.cadence.escalate_threshold": "escalate_threshold",
    "WatchProfile.cadence.hold_s": "hold_s",
}


def apply_search_point(base: WatchProfile, point: Mapping[str, Any]) -> WatchProfile:
    """Materialize one SearchSpace grid/random point onto ``base``, returning a new
    :class:`~percept.watchprofile.WatchProfile` candidate. Raises :class:`SearchSpaceError` for a
    path this module has no materialization mapping for (only the two families above are wired
    today — adding a new safe path here is the same one-line change as adding it to
    ``percept.tunables.REGISTRY``, never a silent no-op)."""
    stat_thresholds = dict(base.stat_thresholds)
    cadence_updates: dict[str, Any] = {}
    unresolved: list[str] = []
    for path, value in point.items():
        if path in _STAT_THRESHOLD_KEY:
            stat_thresholds[_STAT_THRESHOLD_KEY[path]] = float(value)
        elif path in _CADENCE_FIELD:
            cadence_updates[_CADENCE_FIELD[path]] = value
        else:
            unresolved.append(path)
    if unresolved:
        raise SearchSpaceError(
            f"optimizer has no materialization mapping for {unresolved} onto a WatchProfile "
            "candidate (only Settings.edge.motion_gate*/WatchProfile.cadence.* are wired today)"
        )
    result = base
    if stat_thresholds != dict(base.stat_thresholds):
        result = replace(result, stat_thresholds=stat_thresholds)
    if cadence_updates:
        result = replace(result, cadence=replace(result.cadence, **cadence_updates))
    return result


# ---------------------------------------------------------------------------
# 3. Templates — the operator-declared STRUCTURAL axis the SearchSpace tunes within
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerTemplate:
    """One operator-declared structural candidate shape — e.g. "mean stat" vs. "block_max stat" —
    identical in role to a hand-authored :class:`~percept.eval.derivation.LadderRung`: WHICH shape is
    on the ladder is always the operator's choice, never the optimizer's. The :class:`SearchSpace`
    only explores the SAFE numeric fields layered on top of this template's ``base_profile``."""

    name: str
    base_profile: WatchProfile
    source_profile: SourceProfile
    fixture_dir: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SearchSpaceError(f"OptimizerTemplate.name must be a non-empty string, got {self.name!r}")
        if not self.fixture_dir:
            object.__setattr__(self, "fixture_dir", self.name)


def _point_suffix(point: Mapping[str, Any]) -> str:
    return "_".join(str(v) for _, v in sorted(point.items()))


def expand_templates(
    templates: Sequence[OptimizerTemplate],
    space: SearchSpace,
    *,
    strategy: str = "grid",
    n_random: int = 8,
    seed: int = 0,
) -> tuple[tuple[LadderRung, Mapping[str, Any]], ...]:
    """Materialize every (template, search-point) combination into a scoreable
    :class:`~percept.eval.derivation.LadderRung`, paired with the exact point that produced it (so
    the winner's point can become an :class:`~percept.optimization_lock.OptimizationLock`'s
    ``values``). A single-point space keeps a template's own name (readability); multiple points
    disambiguate with an ``@value`` suffix."""
    if not templates:
        raise SearchSpaceError("optimize() needs at least one OptimizerTemplate")
    points = space.sample(strategy, n=n_random, seed=seed)
    single_point = len(points) == 1
    out: list[tuple[LadderRung, Mapping[str, Any]]] = []
    for template in templates:
        for point in points:
            candidate = apply_search_point(template.base_profile, point)
            name = template.name if single_point else f"{template.name}@{_point_suffix(point)}"
            rung = LadderRung(
                name=name,
                watch_profile=candidate,
                source_profile=template.source_profile,
                fixture_dir=template.fixture_dir,
                description=template.description,
            )
            out.append((rung, point))
    return tuple(out)


# ---------------------------------------------------------------------------
# 4. The optimizer job — ladder scoring + OptimizationLock emission
# ---------------------------------------------------------------------------

OPTIMIZATION_REPORT_SCHEMA = "percept.optimizer.report/1"


@dataclass(frozen=True)
class OptimizationReport:
    """The whole answer: the underlying :class:`~percept.eval.derivation.DerivationReport`
    (every candidate, winners AND losers — nothing here reimplements that reporting), the
    :class:`SearchSpace` that generated the ladder, and the emitted
    :class:`~percept.optimization_lock.OptimizationLock` (``None`` exactly when
    ``derivation.derived`` is ``None`` — the same "no bars met, no artifact" refusal
    :mod:`percept.eval.derivation` already enforces, inherited rather than re-decided here)."""

    derivation: DerivationReport
    search_space: SearchSpace
    templates: tuple[OptimizerTemplate, ...]
    lock: OptimizationLock | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPTIMIZATION_REPORT_SCHEMA,
            "derivation": self.derivation.to_dict(),
            "search_space": self.search_space.to_dict(),
            "lock": None if self.lock is None else self.lock.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def optimize(
    *,
    fixtures_root: str | Path,
    templates: Sequence[OptimizerTemplate],
    search_space: SearchSpace,
    bars: BarsDocument,
    base_plan_hash: str = "",
    strategy: str = "grid",
    n_random: int = 8,
    seed: int = 0,
    split_name: str = "test",
    expected_registration_hash: str | None = None,
    declared_at: str = "",
    eval_run_id: str | None = None,
    created_at: str | None = None,
) -> OptimizationReport:
    """Expand ``templates`` x ``search_space`` into a :class:`~percept.eval.derivation.Ladder`,
    score it through :func:`percept.eval.derivation.derive_profile` UNCHANGED, and — only when a
    winner was declared — wrap its selected search-point values in a validated
    :class:`~percept.optimization_lock.OptimizationLock`. Deterministic and scripted/offline: no
    network I/O, no model calls, no wall-clock reads (``created_at`` is the caller's, exactly as
    :func:`percept.profile_versioning.stamp` requires)."""
    expanded = expand_templates(templates, search_space, strategy=strategy, n_random=n_random, seed=seed)
    rungs = tuple(r for r, _ in expanded)
    point_by_name = {r.name: p for r, p in expanded}
    ladder = Ladder(rungs=rungs, source=f"percept.optimizer:{strategy}", declared_at=declared_at)

    report = derive_profile(
        fixtures_root=fixtures_root,
        ladder=ladder,
        bars=bars,
        split_name=split_name,
        expected_registration_hash=expected_registration_hash,
        declared_at=declared_at,
        eval_run_id=eval_run_id,
        created_at=created_at,
    )

    lock: OptimizationLock | None = None
    if report.derived is not None and report.winner is not None:
        winner = report.winner
        selected_values = dict(point_by_name.get(winner.rung.name, {}))
        held_out_metrics = {
            metric: winner.mean(metric) for metric in ("recall", "precision", "false_fires", "cost_usd")
        }
        catalog_component_hash = catalog.snapshot().component_hash
        provisional = OptimizationLock(
            id=report.derived.watch_profile.provenance.eval_run_id,
            values=selected_values,
            base_plan_hash=base_plan_hash,
            catalog_component_hash=catalog_component_hash,
            fixture_hash=report.registration.fixture_set_hash,
            held_out_metrics=held_out_metrics,
            created_at=created_at or "",
        )
        # Stamp the lock's OWN content_hash as its locked_hash: this run VALIDATED exactly these
        # values, so the claim starts true (percept.resolver detects drift from here forward — see
        # ADR-008 Corrections(d) and the invalidation gate in percept.resolver.ConfigResolver.resolve).
        lock = replace(provisional, locked_hash=provisional.content_hash)

    return OptimizationReport(
        derivation=report, search_space=search_space, templates=tuple(templates), lock=lock,
    )


# ---------------------------------------------------------------------------
# 5. Scorecard
# ---------------------------------------------------------------------------


def render_optimization_scorecard(report: OptimizationReport) -> str:
    """The ladder scorecard :mod:`percept.eval.derivation` already renders, plus the lock this
    optimizer run produced (or the refusal, when none was)."""
    lines = [render_scorecard(report.derivation), "", "OPTIMIZATION LOCK"]
    if report.lock is None:
        lines.append("  none emitted — no winner was declared (see VERDICT above).")
    else:
        lock = report.lock
        lines.append(f"  id                     : {lock.id}")
        lines.append(f"  content_hash           : {lock.content_hash}")
        lines.append(f"  locked_hash            : {lock.locked_hash}")
        lines.append(f"  base_plan_hash         : {lock.base_plan_hash or '—'}")
        lines.append(f"  catalog_component_hash : {lock.catalog_component_hash or '—'}")
        lines.append(f"  fixture_hash           : {lock.fixture_hash}")
        lines.append(f"  selected values        : {dict(lock.values)}")
        lines.append(f"  held-out metrics       : {dict(lock.held_out_metrics)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. CLI — ``percept optimize`` / ``percept-optimize``
# ---------------------------------------------------------------------------


def _load_templates(data: Mapping[str, Any]) -> tuple[OptimizerTemplate, ...]:
    from .profile_versioning import profile_from_dict  # noqa: PLC0415 - avoid an import cycle at module load

    default_source = data.get("source_profile")
    raw_templates = data.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise SearchSpaceError("optimizer document has no non-empty 'templates' list")
    out: list[OptimizerTemplate] = []
    for index, raw in enumerate(raw_templates):
        if not isinstance(raw, Mapping):
            raise SearchSpaceError(f"templates[{index}] is not a mapping")
        source_name = raw.get("source_profile", default_source)
        if not source_name:
            raise SearchSpaceError(
                f"templates[{index}] names no source_profile and the document declares no "
                "top-level 'source_profile'"
            )
        body = raw.get("profile")
        if not isinstance(body, Mapping):
            raise SearchSpaceError(f"templates[{index}] has no 'profile' body")
        out.append(OptimizerTemplate(
            name=str(raw.get("name") or body.get("name") or f"template{index}"),
            base_profile=profile_from_dict(body),
            source_profile=get_source_profile(str(source_name)),
            fixture_dir=str(raw.get("fixture_dir") or ""),
            description=str(raw.get("description") or ""),
        ))
    return tuple(out)


def load_optimizer_document(
    path: str | Path,
) -> tuple[tuple[OptimizerTemplate, ...], SearchSpace, str, str]:
    """Load an optimizer document (JSON, or YAML if PyYAML is installed): ``templates`` (a
    :class:`OptimizerTemplate` list, same shape as a ladder file's ``rungs``) plus ``search_space``
    (a :class:`SearchSpace`), and the document's own ``strategy``/``declared_at`` defaults."""
    data = _load_structured(path)
    templates = _load_templates(data)
    space = SearchSpace.from_dict(data.get("search_space") or {})
    strategy = str(data.get("strategy") or "grid")
    declared_at = str(data.get("declared_at") or "")
    return templates, space, strategy, declared_at


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="percept optimize",
        description=(
            "Run a SearchSpace over declared-safe tunables through the same pre-registered ladder "
            "discipline percept-derive-profile uses, and emit a validated OptimizationLock for the "
            "winning candidate. Fixtures + plan + space in, lock artifact out."
        ),
    )
    p.add_argument("--fixtures", required=True, help="fixtures root directory")
    p.add_argument("--space", required=True, help="optimizer document (.json/.yaml): templates + search_space")
    p.add_argument("--bars", default=None, help="OPERATOR-SUPPLIED bars file; omit to report only")
    p.add_argument("--plan", default=None, help="a WatchPlan document; its content hash becomes base_plan_hash")
    p.add_argument("--split", default="test", choices=list(SPLIT_NAMES))
    p.add_argument("--strategy", default=None, choices=("grid", "random"), help="overrides the document's own 'strategy'")
    p.add_argument("--n-random", type=int, default=8, dest="n_random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expect-registration-hash", default=None)
    p.add_argument("--json", dest="json_out", default=None, help="write the full report as JSON")
    p.add_argument("--out-lock", default=None, help="write the OptimizationLock here (only when one was produced)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--declared-at", default="")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``percept-optimize``. 0 = a lock was written or the space was reported, 1 = no candidate met
    the bars, 2 = refused (malformed input, guardrail/unsafe path, config drift)."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        templates, space, doc_strategy, doc_declared_at = load_optimizer_document(args.space)
        bars = load_bars(args.bars)
        base_plan_hash = ""
        if args.plan:
            from .contracts.watchplan import WatchPlan  # noqa: PLC0415 - CLI-only, avoid an import cycle

            plan_data = dict(_load_structured(args.plan))
            # from_dict (not WatchPlan(**plan_data)) so a persisted plan's flat judge_context_* wire
            # keys round-trip back into an ArtifactBinding rather than TypeError-ing as unknown kwargs.
            plan = WatchPlan.from_dict(plan_data)
            base_plan_hash = content_hash(plan.to_dict())
        report = optimize(
            fixtures_root=args.fixtures,
            templates=templates,
            search_space=space,
            bars=bars,
            base_plan_hash=base_plan_hash,
            strategy=args.strategy or doc_strategy,
            n_random=args.n_random,
            seed=args.seed,
            split_name=args.split,
            expected_registration_hash=args.expect_registration_hash,
            declared_at=args.declared_at or doc_declared_at,
            eval_run_id=args.run_id,
        )
    except (EvalInputError, SplitError, SearchSpaceError) as exc:
        print(f"percept optimize: REFUSED — {exc}", file=sys.stderr)
        return 2
    print(render_optimization_scorecard(report))
    if args.json_out:
        Path(args.json_out).write_text(report.to_json())
    if args.out_lock:
        if report.lock is None:
            print(
                f"percept optimize: no lock written to {args.out_lock} — no winner was declared "
                "(see VERDICT above).",
                file=sys.stderr,
            )
        else:
            Path(args.out_lock).write_text(json.dumps(report.lock.to_dict(), indent=2, sort_keys=True))
    return 1 if report.derivation.verdict is DerivationVerdict.FAIL else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
