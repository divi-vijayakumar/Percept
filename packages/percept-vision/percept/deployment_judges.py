"""Deployment-owned judge catalog — who is AVAILABLE to serve a ``WatchPlan.judge`` request.

Per ADR-009 (Operational judge request) a ``WatchPlan.judge`` value is a *request*: the user
states which specialist judge they want, but the user does not get to decide whether that judge is
reachable in this deployment. Availability is a **deployment** concern (ADR-008's ``DeploymentConfig``
ownership row: "which providers are AVAILABLE and how to reach them"). This module is the FIRST
materialized piece of that object — previously ADR-008 recorded ``DeploymentConfig`` as "not yet in
code"; the judge catalog is the concrete slice that now exists.

The contract is **fail-closed**: a judge request is honored only if its ref appears in the
deployment's catalog. An empty catalog (the default when no catalog is configured) therefore means
*every* ``WatchPlan.judge`` request fails closed downstream — that is intentional. A deployment must
**explicitly authorize** a judge before the resolver will arm it; silence is a refusal, never an
implicit "sure, use the generalist instead."

This module is deliberately standalone: stdlib only, no I/O at import, and no dependency on
``percept.contracts`` or the runtime. Constructing a catalog validates shape and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


class DeploymentJudgesError(Exception):
    """A judge catalog could not be loaded: malformed JSON, missing/extra keys, or wrong types.

    Raised loudly and never partially: a catalog either loads whole or not at all, because a
    silently half-loaded catalog would authorize (or fail to authorize) judges by accident.
    """


@dataclass(frozen=True)
class JudgeDeploymentSpec:
    """How a single deployment reaches one judge.

    * ``factory_ref`` — the registry name of the **vision-seam factory** that implements this judge
      (the ``name`` half of ``percept.registry.register("vision", name, factory)``). It is a name
      that resolves to a factory, never a model id (invariant: model id != adapter identity).
    * ``provider_ref`` — the provider this judge runs against.
    * ``model_ref`` — the model id carried to that provider's adapter (configuration, not a selector).
    * ``supports_static_context`` — whether this judge accepts an immutable, pre-bound context blob
      injected under the reserved context key (ADR-009's static judge-context binding). Defaults to
      ``True``.

    ``factory_ref``/``provider_ref``/``model_ref`` are three independent provenance dimensions; this
    spec keeps them separate rather than folding any of them into another.
    """

    factory_ref: str
    provider_ref: str
    model_ref: str
    supports_static_context: bool = True

    def __post_init__(self) -> None:
        for name in ("factory_ref", "provider_ref", "model_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DeploymentJudgesError(
                    f"JudgeDeploymentSpec.{name} must be a non-empty string, got {value!r}"
                )
        if not isinstance(self.supports_static_context, bool):
            raise DeploymentJudgesError(
                "JudgeDeploymentSpec.supports_static_context must be a bool, "
                f"got {self.supports_static_context!r}"
            )


# The keys of a loaded catalog: exactly the required spec fields plus one optional flag.
_REQUIRED_SPEC_KEYS = frozenset({"factory_ref", "provider_ref", "model_ref"})
_OPTIONAL_SPEC_KEYS = frozenset({"supports_static_context"})
_ALLOWED_SPEC_KEYS = _REQUIRED_SPEC_KEYS | _OPTIONAL_SPEC_KEYS

_REQUIRED_CATALOG_KEYS = frozenset({"available_judges"})
_OPTIONAL_CATALOG_KEYS = frozenset({"context_store_root"})
_ALLOWED_CATALOG_KEYS = _REQUIRED_CATALOG_KEYS | _OPTIONAL_CATALOG_KEYS

_CATALOG_ENV_VAR = "PERCEPT_JUDGE_CATALOG"


@dataclass(frozen=True)
class DeploymentJudges:
    """A deployment's catalog of authorized judges.

    ``available_judges`` maps an opaque **judge ref** (e.g. ``"acme_specialist_judge@0.1"``) to the
    spec for reaching it. Judge refs are opaque strings: this module never parses a version out of a
    ref, never orders refs, and never treats two refs as related. A request is authorized iff its
    exact ref is a key here.

    ``context_store_root`` is an optional filesystem root the static judge-context binding may resolve
    ref+sha256 bindings against (see ADR-009). It is data, not an instruction to read anything at
    construction time — nothing here touches the filesystem.
    """

    available_judges: Mapping[str, JudgeDeploymentSpec] = field(default_factory=dict)
    context_store_root: str | None = None

    def __post_init__(self) -> None:
        # Freeze the mapping so a "frozen" catalog cannot be mutated through the field.
        object.__setattr__(
            self, "available_judges", MappingProxyType(dict(self.available_judges))
        )

    def authorizes(self, judge_ref: str) -> bool:
        """True iff this deployment has explicitly authorized ``judge_ref``.

        The fail-closed test the resolver calls pre-arm: an unknown ref is a refusal, and an empty
        catalog refuses every ref.
        """
        return judge_ref in self.available_judges

    def spec_for(self, judge_ref: str) -> JudgeDeploymentSpec | None:
        """The spec for ``judge_ref`` if authorized, else ``None`` (fail closed at the call site)."""
        return self.available_judges.get(judge_ref)

    @classmethod
    def from_file(cls, path: str) -> "DeploymentJudges":
        """Load a catalog from a JSON file, failing loudly on any malformation.

        Shape::

            {
              "available_judges": {
                "<judge-ref>": {
                  "factory_ref": "...",
                  "provider_ref": "...",
                  "model_ref": "...",
                  "supports_static_context": true   # optional, defaults true
                },
                ...
              },
              "context_store_root": "..."            # optional
            }

        Any of: unreadable file, malformed JSON, missing required keys, unexpected extra keys, or a
        wrong type anywhere → :class:`DeploymentJudgesError`. Nothing is partially loaded.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError as exc:
            raise DeploymentJudgesError(f"judge catalog not found: {path}") from exc
        except OSError as exc:
            raise DeploymentJudgesError(f"could not read judge catalog {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DeploymentJudgesError(f"judge catalog {path} is not valid JSON: {exc}") from exc

        return cls._from_mapping(raw, source=path)

    @classmethod
    def _from_mapping(cls, raw: object, *, source: str) -> "DeploymentJudges":
        if not isinstance(raw, dict):
            raise DeploymentJudgesError(
                f"judge catalog {source} must be a JSON object, got {type(raw).__name__}"
            )

        keys = set(raw)
        missing = _REQUIRED_CATALOG_KEYS - keys
        if missing:
            raise DeploymentJudgesError(
                f"judge catalog {source} missing required key(s): {sorted(missing)}"
            )
        extra = keys - _ALLOWED_CATALOG_KEYS
        if extra:
            raise DeploymentJudgesError(
                f"judge catalog {source} has unexpected key(s): {sorted(extra)}"
            )

        raw_judges = raw["available_judges"]
        if not isinstance(raw_judges, dict):
            raise DeploymentJudgesError(
                f"judge catalog {source} 'available_judges' must be a JSON object, "
                f"got {type(raw_judges).__name__}"
            )

        judges: dict[str, JudgeDeploymentSpec] = {}
        for ref, spec_raw in raw_judges.items():
            judges[ref] = cls._spec_from_mapping(ref, spec_raw, source=source)

        context_store_root = raw.get("context_store_root")
        if context_store_root is not None and not isinstance(context_store_root, str):
            raise DeploymentJudgesError(
                f"judge catalog {source} 'context_store_root' must be a string or absent, "
                f"got {type(context_store_root).__name__}"
            )

        return cls(available_judges=judges, context_store_root=context_store_root)

    @staticmethod
    def _spec_from_mapping(
        ref: str, spec_raw: object, *, source: str
    ) -> JudgeDeploymentSpec:
        if not isinstance(spec_raw, dict):
            raise DeploymentJudgesError(
                f"judge catalog {source} entry {ref!r} must be a JSON object, "
                f"got {type(spec_raw).__name__}"
            )

        spec_keys = set(spec_raw)
        missing = _REQUIRED_SPEC_KEYS - spec_keys
        if missing:
            raise DeploymentJudgesError(
                f"judge catalog {source} entry {ref!r} missing required key(s): {sorted(missing)}"
            )
        extra = spec_keys - _ALLOWED_SPEC_KEYS
        if extra:
            raise DeploymentJudgesError(
                f"judge catalog {source} entry {ref!r} has unexpected key(s): {sorted(extra)}"
            )

        kwargs = {
            "factory_ref": spec_raw["factory_ref"],
            "provider_ref": spec_raw["provider_ref"],
            "model_ref": spec_raw["model_ref"],
        }
        if "supports_static_context" in spec_raw:
            kwargs["supports_static_context"] = spec_raw["supports_static_context"]

        # JudgeDeploymentSpec.__post_init__ raises DeploymentJudgesError on any bad field.
        return JudgeDeploymentSpec(**kwargs)


#: The default catalog: no judge is authorized. Every WatchPlan.judge request fails closed against
#: this until a deployment explicitly configures a catalog (see module docstring).
EMPTY_DEPLOYMENT_JUDGES = DeploymentJudges(available_judges={})


def load_deployment_judges(env: Mapping[str, str] | None = None) -> DeploymentJudges:
    """Resolve the deployment judge catalog from the environment.

    Reads the env var ``PERCEPT_JUDGE_CATALOG`` (a path to a JSON catalog):

    * **set** → :meth:`DeploymentJudges.from_file` on that path (loudly errors on malformation);
    * **unset / empty** → :data:`EMPTY_DEPLOYMENT_JUDGES`.

    An empty catalog is not a degraded state — it is the correct default. It means every
    ``WatchPlan.judge`` request fails closed downstream, which is exactly what "deployment must
    explicitly authorize judges" requires. ``env`` defaults to ``os.environ`` when omitted.
    """
    if env is None:
        import os

        env = os.environ

    path = env.get(_CATALOG_ENV_VAR)
    if not path:
        return EMPTY_DEPLOYMENT_JUDGES
    return DeploymentJudges.from_file(path)
