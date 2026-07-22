"""DomainPack (ADR-008) — a named, versioned bundle of domain vocabulary/templates/recommendations.

``docs/adr/ADR-008-configuration-ownership-resolution-provenance.md`` names :class:`DomainPack` as
"domain vocabulary, phrase templates, per-domain recommended WatchProfile/SourceProfile pairings...
new — not yet in code" and assigns it a precedence slot between ``SourceProfile`` and
``ExecutionPolicy``. This module ships the real, typed object: :mod:`percept.resolver` previously
carried a MINIMAL placeholder (``id``/``version``/``values`` only, "the precedence chain is real and
testable now; their full contracts... are later cards' scope, deliberately not built here") — this
module now OWNS that class (mirroring :class:`percept.execution_policy.ExecutionPolicy`'s own
graduation out of the same placeholder set); :mod:`percept.resolver` imports it from here.

A :class:`DomainPack` carries SEVEN things, kept strictly separate because they enforce differently:

- **``values``** — the SAME tunables-registry-path-keyed mapping the placeholder always had, gated
  exactly as before by :func:`percept.resolver._check_ownership` at resolve time (no tunable is
  owned by ``"DomainPack"`` today; a non-empty ``values`` is legitimate only for a still-
  ``"unclassified"`` path — this is unchanged, not a regression this module introduces).
- **``templates``** — named, versioned, PARAMETERIZED :class:`GoalTemplate` objects. Authored with
  a CLOSED schema: every ``behavior``/``episode`` sub-dict is validated at construction (with its
  declared parameter DEFAULTS substituted in) by running it through the SAME
  ``coerce_authored_behavior``/``coerce_authored_episode`` firewall :mod:`percept.watchplan_apply`
  runs at apply time — but here a dropped/changed key RAISES :class:`DomainPackError` instead of
  being silently reported, because a template is AUTHORED ahead of time, not submitted live by an
  untrusted caller: an authoring bug should fail the pack's own construction, never ship a template
  that silently produces a different goal than its author wrote.
- **``components``** — :class:`ComponentRecommendation` entries: ADVISORY component refs (never a
  tunables-path write, never gated the way ``values`` is). Distinct in kind from...
- **``policy_clamps``** — a closed two-key vocabulary (``false_negative_cost_min``,
  ``retention_max_days``) enforced like :mod:`percept.envelope`'s own OUT verdicts: a violation is a
  RESOLVE-TIME error (raised by :meth:`percept.resolver.ConfigResolver.resolve`, before anything is
  produced), never a silent clamp/average. A clamp is a FLOOR/CEILING the pack's author asserts is
  load-bearing for this domain (e.g. "an incident report in this domain must never be authored
  ``miss_cost='trivial'``") — merging it away would be exactly the silent-degrade failure mode
  ADR-007's envelope verdicts exist to close for actuation, applied here to configuration.
- **``consequence_defaults``** is deliberately NOT a separate field: a template's own ``consequence``
  IS the domain's default for that goal shape (see :class:`GoalTemplate`), composed the same way any
  :class:`~percept.contracts.WatchPlan.consequence` is — no second mechanism duplicating it.
- **``direct_judge_routes``** — real :class:`percept.direct_judge.DirectRouting` DECLARATIONS (data
  only): which condition-namespace prefixes this pack's templates could, in principle, be served by
  a :class:`~percept.direct_judge.DirectJudge` rule. No rule implementation ships here — routing a
  namespace to an actual cheap answer is a later card's concern; this only reserves the vocabulary
  using the REAL public type, never a reinvented shape.
- **``fixtures`` / ``quality_bars`` / ``search_space``** — plain data declarations. ``fixtures`` names
  the evaluation clips/traces this pack expects to be qualified against; ``quality_bars`` is a
  ``{"<template_name>.<metric>": floor}`` mapping; ``search_space`` is a
  ``{tunables-path: {"min": ..., "max": ...}}``-shaped declaration an OPTIMIZER card consumes later
  (never executed here — this module performs no search, no I/O).

**Templates never auto-arm** (ADR-008 Invariants): :func:`watch_template` / :meth:`GoalTemplate.render`
only ever return a portable :class:`~percept.contracts.WatchPlan`. Arming it — registering a live goal
on a running :class:`~percept.agent.Percept` core — is always the caller's own
:func:`percept.watchplan_apply.apply_watch_plan` call, never something this module does for them.

Entry-point discovery mirrors :func:`percept.sourceprofile.discover_source_profile_entrypoints`
exactly: :func:`discover_domain_pack_entrypoints` loads third-party packs advertised under the
``percept.packs`` entry-point group, each entry point a callable taking the register function.
:func:`get_domain_pack` only seeds the IN-CODE default (:mod:`percept.packs_default`) automatically
(the same "shipped names always resolve" guarantee :func:`percept.execution_policy.get_execution_policy`
gives); pip-installed packs are opt-in via a caller invoking :func:`discover_domain_pack_entrypoints`
first, same as source profiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Literal, Mapping

from .behavior import coerce_authored_behavior, coerce_authored_episode
from .contracts.watchplan import WatchPlan
from .direct_judge import DirectRouting


class DomainPackError(ValueError):
    """A :class:`DomainPack` (or a :class:`GoalTemplate`/:class:`ComponentRecommendation` inside
    one) failed a shape/consistency/authoring check, or a resolve-time policy clamp was violated."""


class DomainPackNotFoundError(KeyError):
    """No :class:`DomainPack` matches a registry lookup, or a named template isn't in one."""


#: The two strengths a :class:`ComponentRecommendation` may carry. ``"required"`` is part of the
#: type but unused by the shipped facility-observation pack (its recommendations are all
#: ``"preferred"`` — "recommended, not required" is the shipped posture until identity/tracking
#: calibration lands for a domain that needs a harder floor).
ComponentStrength = Literal["preferred", "required"]
_STRENGTHS: tuple[str, ...] = ("preferred", "required")

#: Cost severity order, mirrors ``percept.behavior._VALID_MISS`` / ``percept.contracts.Cost``'s own
#: closed vocabulary (``trivial < mild < severe``) — the ONLY ordering a ``false_negative_cost_min``
#: clamp needs, not reimplemented anywhere else.
_MISS_COST_ORDER: dict[str, int] = {"trivial": 0, "mild": 1, "severe": 2}

#: The closed policy-clamp vocabulary. Unknown clamp keys are authoring errors (closed schema, same
#: posture as every other authored surface in this module) — never silently ignored.
_VALID_POLICY_CLAMPS: tuple[str, ...] = ("false_negative_cost_min", "retention_max_days")

#: A bare ``"{name}"`` template token is a TYPED parameter reference (the substituted value keeps its
#: original type — a number stays a number); anything else runs through ``str.format`` for mixed
#: text+token strings (``"the door at {zone} is open"``).
_PARAM_REF_RE = re.compile(r"^\{(\w+)\}$")

#: ``behavior`` raw-shape top-level (non-nested) keys — mirrors ``percept.watchplan_apply``'s own
#: ``_BEHAVIOR_LEAF_KEYS`` exactly (the same firewall input shape).
_BEHAVIOR_TOP_KEYS: tuple[str, ...] = ("type", "rearm", "cadence_s", "deadline_after_s", "from", "to")


def _freeze_values(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise DomainPackError(f"values must be a mapping, got {type(values).__name__}")
    for key in values:
        if not isinstance(key, str) or not key.strip():
            raise DomainPackError(f"values keys must be non-empty strings, got {key!r}")
    return MappingProxyType(dict(values))


def _substitute(value: Any, params: Mapping[str, Any]) -> Any:
    """Fill ``{name}`` tokens in ``value`` from ``params``. A BARE ``"{name}"`` string returns the
    typed parameter value verbatim (so a numeric ``deadline_after_s`` parameter substitutes as a
    float, not a stringified one); any other string runs through ``str.format(**params)`` for
    mixed text+token authoring. Non-string values pass through unchanged. Raises
    :class:`DomainPackError` (never silently drops) on a reference to an undeclared parameter."""
    if not isinstance(value, str):
        return value
    bare = _PARAM_REF_RE.match(value)
    if bare:
        name = bare.group(1)
        if name not in params:
            raise DomainPackError(f"template references undeclared parameter {name!r}")
        return params[name]
    try:
        return value.format(**params)
    except (KeyError, IndexError) as exc:
        raise DomainPackError(
            f"template string {value!r} references an unknown parameter: {exc}"
        ) from exc


def _substitute_behavior(behavior: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Substitute ``{param}`` tokens into a RAW ``behavior`` dict's known leaf values. An
    UNRECOGNIZED top-level key is passed through UNCHANGED (never silently dropped here) so
    :func:`_behavior_authoring_diff`'s own "unrecognized key" check — run on this function's
    output — still sees it and raises; dropping it at substitution time would hide the authoring
    bug instead of catching it."""
    out: dict[str, Any] = {}
    for key, value in behavior.items():
        if key in _BEHAVIOR_TOP_KEYS:
            out[key] = _substitute(value, params)
        elif key == "accumulate" and isinstance(value, Mapping):
            out["accumulate"] = {k: _substitute(v, params) for k, v in value.items()}
        elif key == "consequence" and isinstance(value, Mapping):
            # cost labels are closed-vocabulary tokens, never parameterized text — passed through.
            out["consequence"] = dict(value)
        else:
            out[key] = value  # unrecognized key: passed through so the diff check below catches it
    return out


def _behavior_authoring_diff(behavior: Mapping[str, Any]) -> list[str]:
    """Run :func:`~percept.behavior.coerce_authored_behavior` over ``behavior`` and return every
    leaf path it dropped or changed — the SAME diff basis
    :func:`percept.watchplan_apply._behavior_leaf_checks` uses, reused here for a RAISE instead of a
    reported diagnostic (see module docstring: authored-ahead-of-time vs. submitted-live)."""
    result = coerce_authored_behavior(dict(behavior))
    bad: list[str] = []
    for key in _BEHAVIOR_TOP_KEYS:
        if key in behavior and (key not in result or result[key] != behavior[key]):
            bad.append(key)
    accumulate = behavior.get("accumulate")
    if isinstance(accumulate, Mapping):
        for key in ("fire_when", "window_s"):
            if key in accumulate and (key not in result or result[key] != accumulate[key]):
                bad.append(f"accumulate.{key}")
    consequence = behavior.get("consequence")
    if isinstance(consequence, Mapping):
        for key in ("miss_cost", "false_fire_cost", "protected_floor"):
            if key in consequence and (key not in result or result[key] != consequence[key]):
                bad.append(f"consequence.{key}")
    known_top = set(_BEHAVIOR_TOP_KEYS) | {"accumulate", "consequence"}
    bad.extend(f"{k} (unrecognized key)" for k in behavior if k not in known_top)
    return bad


def _behavior_to_explicit(behavior: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a rendered ``behavior`` raw-shape dict into ``WatchPlan.explicit`` dotted paths —
    the inverse of :func:`percept.runtime_compiler._goal_state_from_resolved`'s own flattening, so a
    template's behavior lands on the ``Goal.state.*`` tunables paths that dotted vocabulary owns
    (owner ``"WatchPlan"`` for every one of them per :data:`percept.tunables.REGISTRY`)."""
    out: dict[str, Any] = {}
    for key in _BEHAVIOR_TOP_KEYS:
        if key in behavior:
            out[f"Goal.state.{key}"] = behavior[key]
    accumulate = behavior.get("accumulate")
    if isinstance(accumulate, Mapping):
        for key in ("fire_when", "window_s"):
            if key in accumulate:
                out[f"Goal.state.{key}"] = accumulate[key]
    return out


@dataclass(frozen=True)
class ComponentRecommendation:
    """One ADVISORY component reference a :class:`DomainPack` surfaces — DISTINCT from
    ``policy_clamps``: a recommendation never blocks resolution (not ownership-gated, not a
    resolve-time error), it is purely descriptive. ``ref`` is a :mod:`percept.catalog` component
    reference (``"role:name"``, e.g. ``"detector:motion"``, matching
    :func:`percept.cli._cmd_components_describe`'s own ``ref`` grammar) or a bare name when
    unambiguous. ``strength`` is ``"preferred"`` (the shipped default posture) or ``"required"``
    (reserved — no shipped pack uses it yet; see module docstring)."""

    ref: str
    strength: ComponentStrength = "preferred"
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise DomainPackError(f"ComponentRecommendation.ref must be a non-empty string, got {self.ref!r}")
        if self.strength not in _STRENGTHS:
            raise DomainPackError(
                f"ComponentRecommendation.strength must be one of {_STRENGTHS}, got {self.strength!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "strength": self.strength, "reason": self.reason}


@dataclass(frozen=True)
class GoalTemplate:
    """One named, versioned, PARAMETERIZED goal shape a :class:`DomainPack` ships.

    - ``name`` — the template's key within its pack (matches the pack's ``templates`` dict key).
    - ``condition`` — plain-language condition text, may contain ``{param}`` tokens.
    - ``parameters`` — ``{name: default_value}``; every token used anywhere in this template MUST
      be declared here (an undeclared reference raises at render time, never silently drops).
    - ``sense`` / ``target`` / ``detectors`` / ``extractors`` / ``source`` — mirror
      :class:`~percept.contracts.WatchPlan`'s own same-named fields; ``target`` may also contain
      tokens.
    - ``behavior`` — the RAW ``goal.state["behavior"]``-shape dict (``type``/``rearm``/``cadence_s``/
      ``deadline_after_s``/``from``/``to``, plus nested ``accumulate``/``consequence`` sub-dicts) —
      the SAME input shape :func:`~percept.behavior.coerce_authored_behavior` expects. Validated at
      construction (with parameter DEFAULTS substituted in) — see module docstring.
    - ``episode`` — an optional RAW ``goal.state["episode"]``-shape dict, validated the same way via
      :func:`~percept.behavior.coerce_authored_episode`.
    - ``consequence`` — an optional ``{miss_cost, false_fire_cost, protected_floor}`` mapping; this
      IS the domain's consequence default for this goal shape (composed straight onto the rendered
      :class:`~percept.contracts.WatchPlan.consequence` field, never through the ownership-gated
      ``explicit`` channel).
    """

    name: str
    condition: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    sense: str | None = None
    target: str | None = None
    detectors: tuple[str, ...] = ()
    extractors: tuple[str, ...] = ()
    behavior: Mapping[str, Any] = field(default_factory=dict)
    episode: Mapping[str, Any] | None = None
    consequence: Mapping[str, Any] | None = None
    source: str = "camera"
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DomainPackError(f"GoalTemplate.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise DomainPackError(f"GoalTemplate.condition must be a non-empty string, got {self.condition!r}")
        if not isinstance(self.parameters, Mapping):
            raise DomainPackError("GoalTemplate.parameters must be a mapping")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "detectors", tuple(self.detectors))
        object.__setattr__(self, "extractors", tuple(self.extractors))

        if not isinstance(self.behavior, Mapping):
            raise DomainPackError("GoalTemplate.behavior must be a mapping")
        behavior = dict(self.behavior)
        # Validate the CLOSED schema using declared parameter DEFAULTS substituted in — an authoring
        # bug (unknown key, malformed numeric, unsafe fire_when) fails pack construction, never ships.
        rendered_default = _substitute_behavior(behavior, self.parameters) if behavior else {}
        bad = _behavior_authoring_diff(rendered_default) if rendered_default else []
        if bad:
            raise DomainPackError(
                f"GoalTemplate {self.name!r}: behavior failed closed-schema validation for "
                f"{bad} (unknown/malformed keys are authoring errors, never silently coerced)"
            )
        object.__setattr__(self, "behavior", MappingProxyType(behavior))

        if self.episode is not None:
            if not isinstance(self.episode, Mapping):
                raise DomainPackError("GoalTemplate.episode must be a mapping or None")
            episode = dict(self.episode)
            result = coerce_authored_episode(episode)
            after = result.get("episode", {})
            bad_ep = [k for k in episode if k not in after or after[k] != episode[k]]
            if bad_ep:
                raise DomainPackError(
                    f"GoalTemplate {self.name!r}: episode failed closed-schema validation for {bad_ep}"
                )
            object.__setattr__(self, "episode", MappingProxyType(episode))

        if self.consequence is not None:
            if not isinstance(self.consequence, Mapping):
                raise DomainPackError("GoalTemplate.consequence must be a mapping or None")
            cons_check = _behavior_authoring_diff({"consequence": dict(self.consequence)})
            if cons_check:
                raise DomainPackError(
                    f"GoalTemplate {self.name!r}: consequence failed closed-schema validation for "
                    f"{cons_check}"
                )
            object.__setattr__(self, "consequence", MappingProxyType(dict(self.consequence)))

    def render(self, **overrides: Any) -> WatchPlan:
        """Fill this template's declared parameters with ``overrides`` and return a portable
        :class:`~percept.contracts.WatchPlan`. NEVER arms anything (ADR-008 Invariants: "templates
        never auto-arm") — composes with :mod:`percept.resolver`; the caller's own
        :func:`percept.watchplan_apply.apply_watch_plan` call is what arms it.

        An override naming a parameter this template did not declare raises
        :class:`DomainPackError` (closed parameter surface — the same "unknown keys are authoring
        errors" posture the pack's own schema validation takes), never a silent no-op."""
        unknown = set(overrides) - set(self.parameters)
        if unknown:
            raise DomainPackError(
                f"GoalTemplate {self.name!r} has no parameter(s) {sorted(unknown)}; "
                f"declared parameters: {sorted(self.parameters)}"
            )
        params = dict(self.parameters)
        params.update(overrides)

        behavior = _substitute_behavior(self.behavior, params) if self.behavior else {}
        if behavior:
            bad = _behavior_authoring_diff(behavior)
            if bad:
                raise DomainPackError(
                    f"GoalTemplate {self.name!r}: override(s) {sorted(overrides)} produced an "
                    f"invalid behavior for {bad}"
                )
        explicit = _behavior_to_explicit(behavior) if behavior else {}
        if self.episode:
            for key, value in self.episode.items():
                explicit[f"Goal.state.episode.{key}"] = value

        condition = _substitute(self.condition, params)
        target = _substitute(self.target, params) if self.target else None

        return WatchPlan(
            condition=condition,
            sense=self.sense,
            target=target,
            detectors=self.detectors,
            extractors=self.extractors,
            consequence=dict(self.consequence) if self.consequence else None,
            explicit=explicit,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "parameters": dict(self.parameters),
            "sense": self.sense,
            "target": self.target,
            "detectors": list(self.detectors),
            "extractors": list(self.extractors),
            "behavior": dict(self.behavior),
            "episode": dict(self.episode) if self.episode is not None else None,
            "consequence": dict(self.consequence) if self.consequence is not None else None,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DomainPack:
    """A named, versioned bundle of domain vocabulary/templates/recommendations (ADR-008).

    - ``id`` / ``version`` — together name the pack (:attr:`qualified_name` renders
      ``f"{id}@{version}"``, e.g. ``"facility_observation@1"``).
    - ``values`` — tunables-registry-path-keyed mapping, gated exactly as the earlier placeholder
      was (see module docstring) — unchanged behavior, this module does not loosen it.
    - ``templates`` — ``{name: GoalTemplate}``.
    - ``components`` — advisory :class:`ComponentRecommendation` entries.
    - ``policy_clamps`` — closed two-key vocabulary, enforced at resolve time (see module docstring
      and :meth:`percept.resolver.ConfigResolver.resolve`).
    - ``direct_judge_routes`` — real :class:`~percept.direct_judge.DirectRouting` declarations.
    - ``fixtures`` / ``quality_bars`` / ``search_space`` — plain data declarations for later cards.
    """

    id: str
    version: str = "v1"
    values: Mapping[str, Any] = field(default_factory=dict)
    templates: Mapping[str, GoalTemplate] = field(default_factory=dict)
    components: tuple[ComponentRecommendation, ...] = ()
    policy_clamps: Mapping[str, Any] = field(default_factory=dict)
    direct_judge_routes: tuple[DirectRouting, ...] = ()
    fixtures: tuple[str, ...] = ()
    quality_bars: Mapping[str, float] = field(default_factory=dict)
    search_space: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise DomainPackError(f"DomainPack.id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.version, str) or not self.version.strip():
            raise DomainPackError(f"DomainPack.version must be a non-empty string, got {self.version!r}")
        object.__setattr__(self, "values", _freeze_values(self.values))

        if not isinstance(self.templates, Mapping):
            raise DomainPackError("DomainPack.templates must be a mapping")
        templates: dict[str, GoalTemplate] = dict(self.templates)
        for name, tmpl in templates.items():
            if not isinstance(tmpl, GoalTemplate):
                raise DomainPackError(f"DomainPack.templates[{name!r}] must be a GoalTemplate")
            if tmpl.name != name:
                raise DomainPackError(
                    f"DomainPack.templates key {name!r} must match GoalTemplate.name {tmpl.name!r}"
                )
        object.__setattr__(self, "templates", MappingProxyType(templates))

        components = tuple(self.components)
        for c in components:
            if not isinstance(c, ComponentRecommendation):
                raise DomainPackError("DomainPack.components must contain only ComponentRecommendation entries")
        object.__setattr__(self, "components", components)

        if not isinstance(self.policy_clamps, Mapping):
            raise DomainPackError("DomainPack.policy_clamps must be a mapping")
        clamps = dict(self.policy_clamps)
        for key in clamps:
            if key not in _VALID_POLICY_CLAMPS:
                raise DomainPackError(
                    f"DomainPack.policy_clamps key {key!r} is not one of {_VALID_POLICY_CLAMPS} "
                    "(unknown clamp keys are authoring errors, never silently ignored)"
                )
        if "false_negative_cost_min" in clamps and clamps["false_negative_cost_min"] not in _MISS_COST_ORDER:
            raise DomainPackError(
                "DomainPack.policy_clamps['false_negative_cost_min'] must be one of "
                f"{sorted(_MISS_COST_ORDER)}, got {clamps['false_negative_cost_min']!r}"
            )
        if "retention_max_days" in clamps:
            v = clamps["retention_max_days"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise DomainPackError(
                    f"DomainPack.policy_clamps['retention_max_days'] must be a positive number, got {v!r}"
                )
        object.__setattr__(self, "policy_clamps", MappingProxyType(clamps))

        routes = tuple(self.direct_judge_routes)
        for r in routes:
            if not isinstance(r, DirectRouting):
                raise DomainPackError(
                    "DomainPack.direct_judge_routes must contain only percept.direct_judge.DirectRouting entries"
                )
        object.__setattr__(self, "direct_judge_routes", routes)

        object.__setattr__(self, "fixtures", tuple(self.fixtures))
        object.__setattr__(self, "quality_bars", MappingProxyType(dict(self.quality_bars)))
        if not isinstance(self.search_space, Mapping):
            raise DomainPackError("DomainPack.search_space must be a mapping")
        object.__setattr__(self, "search_space", MappingProxyType(dict(self.search_space)))

    @property
    def qualified_name(self) -> str:
        """``"{id}@{version}"`` — the registry key (``"facility_observation@1"``)."""
        return f"{self.id}@{self.version}"

    def template(self, name: str) -> GoalTemplate:
        """Look up one template by name, or raise :class:`DomainPackNotFoundError` listing what IS
        declared."""
        try:
            return self.templates[name]
        except KeyError:
            raise DomainPackNotFoundError(
                f"pack {self.qualified_name!r} has no template {name!r}; declared: "
                f"{sorted(self.templates)}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "qualified_name": self.qualified_name,
            "values": dict(self.values),
            "templates": {name: t.to_dict() for name, t in self.templates.items()},
            "components": [c.to_dict() for c in self.components],
            "policy_clamps": dict(self.policy_clamps),
            "direct_judge_routes": [
                {"condition_namespace": r.condition_namespace, "role": r.role}
                for r in self.direct_judge_routes
            ],
            "fixtures": list(self.fixtures),
            "quality_bars": dict(self.quality_bars),
            "search_space": dict(self.search_space),
            "notes": self.notes,
        }


@dataclass
class DomainPackRegistry:
    """An OPEN, ``qualified_name``-keyed table of :class:`DomainPack` — same shape and same
    fail-loud-on-silent-replace rule as :class:`percept.execution_policy.ExecutionPolicyRegistry`."""

    _packs: dict[str, DomainPack] = field(default_factory=dict)

    def register(self, pack: DomainPack, *, overwrite: bool = False) -> None:
        if not isinstance(pack, DomainPack):
            raise DomainPackError(f"expected a DomainPack, got {type(pack).__name__}")
        existing = self._packs.get(pack.qualified_name)
        if existing is not None and existing != pack and not overwrite:
            raise DomainPackError(
                f"a domain pack named {pack.qualified_name!r} is already registered; pass "
                "overwrite=True to replace it"
            )
        self._packs[pack.qualified_name] = pack

    def unregister(self, qualified_name: str) -> None:
        self._packs.pop(qualified_name, None)

    def get(self, qualified_name: str) -> DomainPack | None:
        return self._packs.get(qualified_name)

    def resolve(self, qualified_name: str) -> DomainPack:
        pack = self._packs.get(qualified_name)
        if pack is None:
            raise DomainPackNotFoundError(
                f"no domain pack named {qualified_name!r}; registered: {sorted(self._packs)}"
            )
        return pack

    def names(self) -> list[str]:
        return sorted(self._packs)

    def all(self) -> list[DomainPack]:
        return list(self._packs.values())

    def __contains__(self, qualified_name: object) -> bool:
        return qualified_name in self._packs

    def __len__(self) -> int:
        return len(self._packs)

    def __iter__(self) -> Iterator[DomainPack]:
        return iter(self._packs.values())


#: The process-wide registry. ``percept.packs_default`` seeds the shipped packs into it at import;
#: third parties add to it with :func:`register_domain_pack` or the ``percept.packs`` entry-point
#: group (:func:`discover_domain_pack_entrypoints`).
DEFAULT_DOMAIN_PACK_REGISTRY = DomainPackRegistry()


def register_domain_pack(pack: DomainPack, *, overwrite: bool = False) -> None:
    """Add ``pack`` to :data:`DEFAULT_DOMAIN_PACK_REGISTRY` — the third-party extension point."""
    DEFAULT_DOMAIN_PACK_REGISTRY.register(pack, overwrite=overwrite)


def get_domain_pack(qualified_name: str) -> DomainPack:
    """Resolve ``qualified_name`` (e.g. ``"facility_observation@1"``) against
    :data:`DEFAULT_DOMAIN_PACK_REGISTRY`, seeding the shipped packs first so the shipped names
    always resolve, whatever import order the caller used. Pip-installed packs are NOT auto-loaded
    here — call :func:`discover_domain_pack_entrypoints` first (same opt-in shape as
    :func:`percept.sourceprofile.discover_source_profile_entrypoints`)."""
    from . import packs_default  # noqa: F401, PLC0415 — import seeds the registry

    return DEFAULT_DOMAIN_PACK_REGISTRY.resolve(qualified_name)


def discover_domain_pack_entrypoints() -> None:
    """Load third-party domain packs advertised under the ``percept.packs`` entry-point group. Each
    entry point is called with :func:`register_domain_pack`. One bad plugin never breaks the others
    (the same contract :func:`percept.registry.discover_entrypoints` /
    :func:`percept.sourceprofile.discover_source_profile_entrypoints` follow)."""
    from importlib import metadata as importlib_metadata  # noqa: PLC0415

    try:
        eps = importlib_metadata.entry_points(group="percept.packs")
    except TypeError:  # pragma: no cover — older importlib.metadata signature
        eps = importlib_metadata.entry_points().get("percept.packs", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()(register_domain_pack)
        except Exception:  # noqa: BLE001 — a bad plugin must not break the core
            continue


def watch_template(ref: str, parameters: Mapping[str, Any] | None = None) -> WatchPlan:
    """Render one named template into a portable :class:`~percept.contracts.WatchPlan`.

    ``ref`` is ``"{pack_qualified_name}/{template_name}"`` (e.g.
    ``"facility_observation@1/door_left_open"``). NEVER arms anything — see module docstring and
    :meth:`GoalTemplate.render`. Composes with :mod:`percept.resolver`: pass the returned
    ``WatchPlan`` to :func:`percept.watchplan_apply.apply_watch_plan` (optionally alongside a
    ``domain_pack=`` so the pack's clamps/values are enforced too) to actually arm it."""
    if "/" not in ref:
        raise DomainPackError(
            f"watch_template(ref=...) must be '<pack_qualified_name>/<template_name>', got {ref!r}"
        )
    pack_name, _, template_name = ref.partition("/")
    pack = get_domain_pack(pack_name)
    template = pack.template(template_name)
    return template.render(**dict(parameters or {}))


__all__ = [
    "DomainPackError",
    "DomainPackNotFoundError",
    "ComponentStrength",
    "ComponentRecommendation",
    "GoalTemplate",
    "DomainPack",
    "DomainPackRegistry",
    "DEFAULT_DOMAIN_PACK_REGISTRY",
    "register_domain_pack",
    "get_domain_pack",
    "discover_domain_pack_entrypoints",
    "watch_template",
]
