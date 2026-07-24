"""Judge authorization + resolver integration (ADR-009, PERCEPT-SDK-P06-001).

Covers the resolve-side reconciliation of a ``WatchPlan.judge`` REQUEST against the
deployment-owned catalog: an authorized judge populates ``ResolvedWatch.judge`` and an additive
``components["judge"]`` block (moving ``component_snapshot_hash``); a plan naming NO judge is
byte-for-byte unchanged (no ``"judge"`` key, hash stable); and every fail-closed refusal
(unknown ref, empty default catalog, uninstalled factory, unsupported/unverifiable context,
context-without-judge) raises PRE-ARM. Also: receipt completeness, the module-level adapter cache,
and the generic ``hash_projection`` helper.

No runtime execution — every test constructs inputs and calls ``ConfigResolver.resolve`` / the
``judge_registry`` seam directly; nothing touches worker.py/agent.py/gate.py or the network.
"""

from __future__ import annotations

import hashlib

import pytest

from percept import registry
from percept.config import Settings
from percept.contracts.watchplan import ArtifactBinding, ResolvedJudgeSpec
from percept.deployment_judges import DeploymentJudges, JudgeDeploymentSpec
from percept.judge_context import JudgeContextError
from percept.judge_registry import (
    BoundJudge,
    JudgeContextUnsupportedError,
    JudgeFactoryUnavailableError,
    UnknownJudgeError,
    clear_adapter_cache,
    get_or_create_adapter,
    resolve_authorized,
)
from percept.resolver import ConfigError, ConfigResolver, hash_projection
from percept.tracing import content_hash

# A unique vision-seam factory name so this test never collides with a real backend and can be
# cleaned up deterministically. The factory counts its constructions so the adapter-cache tests can
# assert hit/miss.
_FACTORY_NAME = "p06_test_vision_factory"
_ABSENT_FACTORY = "p06_absent_factory_never_registered"
_JUDGE_REF = "audit_specialist_judge@0.1"


class _CountingVision:
    """A throwaway vision adapter double — identity is all these tests read off it."""

    def __init__(self) -> None:
        self.instance_id = object()


@pytest.fixture
def fake_factory():
    """Register the counting vision factory for the duration of a test, then remove it."""
    calls = {"n": 0}

    def _factory(_settings):
        calls["n"] += 1
        return _CountingVision()

    registry.register("vision", _FACTORY_NAME, _factory)
    clear_adapter_cache()
    try:
        yield calls
    finally:
        registry._REGISTRY["vision"].pop(_FACTORY_NAME, None)
        clear_adapter_cache()


def _catalog(*, supports_static_context: bool = True, factory_ref: str = _FACTORY_NAME,
             context_store_root: str | None = None) -> DeploymentJudges:
    return DeploymentJudges(
        available_judges={
            _JUDGE_REF: JudgeDeploymentSpec(
                factory_ref=factory_ref,
                provider_ref="anthropic",
                model_ref="claude-x",
                supports_static_context=supports_static_context,
            )
        },
        context_store_root=context_store_root,
    )


def _plan(**kwargs):
    from percept.contracts.watchplan import WatchPlan

    kwargs.setdefault("condition", "a person is at the door")
    return WatchPlan(**kwargs)


def _write_context(tmp_path, ref: str, text: str) -> tuple[str, str]:
    """Write a static-context artifact and return (store_root, sha256)."""
    (tmp_path / ref).write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(tmp_path), digest


# ---------------------------------------------------------------------------
# resolve_authorized — the atomic authorization step in isolation
# ---------------------------------------------------------------------------


class TestResolveAuthorized:
    def test_authorized_returns_spec(self, fake_factory) -> None:
        spec = resolve_authorized(_JUDGE_REF, _catalog())
        assert isinstance(spec, ResolvedJudgeSpec)
        assert spec.requested_ref == _JUDGE_REF
        assert spec.resolved_ref == _JUDGE_REF
        assert spec.factory_ref == _FACTORY_NAME
        assert spec.provider_ref == "anthropic"
        assert spec.model_requested == "claude-x"
        assert spec.model_resolved == "claude-x"
        assert spec.prompt_envelope_sha256 is None
        assert spec.context_ref is None
        assert spec.context_sha256 is None

    def test_binding_mirrored_onto_spec(self, fake_factory) -> None:
        binding = ArtifactBinding(ref="rubric.txt", sha256="a" * 64)
        spec = resolve_authorized(_JUDGE_REF, _catalog(), judge_context=binding)
        assert spec.context_ref == "rubric.txt"
        assert spec.context_sha256 == "a" * 64

    def test_unknown_ref_raises(self, fake_factory) -> None:
        with pytest.raises(UnknownJudgeError) as exc:
            resolve_authorized("no_such_judge@9", _catalog())
        # The message must point at the deployment as the thing that must authorize.
        assert "explicitly authorize" in str(exc.value)

    def test_empty_catalog_fails_closed(self, fake_factory) -> None:
        with pytest.raises(UnknownJudgeError):
            resolve_authorized(_JUDGE_REF, DeploymentJudges(available_judges={}))

    def test_factory_not_installed_raises(self) -> None:
        # No fake_factory fixture: the catalog names a factory nothing registered.
        with pytest.raises(JudgeFactoryUnavailableError):
            resolve_authorized(_JUDGE_REF, _catalog(factory_ref=_ABSENT_FACTORY))

    def test_context_unsupported_raises(self, fake_factory) -> None:
        binding = ArtifactBinding(ref="rubric.txt", sha256="a" * 64)
        with pytest.raises(JudgeContextUnsupportedError):
            resolve_authorized(
                _JUDGE_REF,
                _catalog(supports_static_context=False),
                judge_context=binding,
            )

    def test_authz_errors_are_config_errors(self, fake_factory) -> None:
        # All three subclass ConfigError so a resolver caller fails closed uniformly.
        for exc_type in (UnknownJudgeError, JudgeFactoryUnavailableError, JudgeContextUnsupportedError):
            assert issubclass(exc_type, ConfigError)


# ---------------------------------------------------------------------------
# Resolver integration — the judge component + the byte-stability invariant
# ---------------------------------------------------------------------------


class TestResolverIntegration:
    def test_authorized_judge_populates_component(self, fake_factory) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(
            _plan(judge=_JUDGE_REF), settings=Settings(), deployment_judges=_catalog()
        )
        assert isinstance(resolved.judge, ResolvedJudgeSpec)
        assert resolved.judge.resolved_ref == _JUDGE_REF
        block = resolved.components["judge"]
        assert block == {
            "requested_ref": _JUDGE_REF,
            "resolved_ref": _JUDGE_REF,
            "factory_ref": _FACTORY_NAME,
            "provider": "anthropic",
            "model_requested": "claude-x",
            "model_resolved": "claude-x",
            "prompt_envelope_sha256": None,
            "context": None,
        }

    def test_component_hash_changes_when_judge_requested(self, fake_factory) -> None:
        resolver = ConfigResolver()
        baseline = resolver.resolve(_plan(), settings=Settings(), deployment_judges=_catalog())
        with_judge = resolver.resolve(
            _plan(judge=_JUDGE_REF), settings=Settings(), deployment_judges=_catalog()
        )
        assert with_judge.component_snapshot_hash != baseline.component_snapshot_hash

    def test_judge_unset_is_byte_identical(self, fake_factory) -> None:
        """A plan naming no judge: NO 'judge' key, and the hash is exactly the hash over the
        judge-free components (byte-stable vs the pre-ADR-009 path)."""
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), settings=Settings(), deployment_judges=_catalog())
        assert resolved.judge is None
        assert "judge" not in resolved.components
        assert resolved.component_snapshot_hash == content_hash(dict(resolved.components))

    def test_judge_unset_hash_independent_of_catalog(self, fake_factory) -> None:
        """Passing a populated catalog must not perturb a judge-free resolution's hash — the catalog
        only matters once a judge is REQUESTED."""
        resolver = ConfigResolver()
        with_catalog = resolver.resolve(_plan(), settings=Settings(), deployment_judges=_catalog())
        empty_catalog = resolver.resolve(
            _plan(), settings=Settings(), deployment_judges=DeploymentJudges(available_judges={})
        )
        assert with_catalog.component_snapshot_hash == empty_catalog.component_snapshot_hash

    def test_unknown_ref_pre_arm(self, fake_factory) -> None:
        resolver = ConfigResolver()
        with pytest.raises(UnknownJudgeError):
            resolver.resolve(
                _plan(judge="no_such_judge@9"), settings=Settings(), deployment_judges=_catalog()
            )

    def test_empty_default_catalog_fails_closed(self, monkeypatch) -> None:
        # deployment_judges omitted -> load_deployment_judges() with the env var unset -> empty.
        monkeypatch.delenv("PERCEPT_JUDGE_CATALOG", raising=False)
        resolver = ConfigResolver()
        with pytest.raises(UnknownJudgeError):
            resolver.resolve(_plan(judge=_JUDGE_REF), settings=Settings())

    def test_factory_not_installed_pre_arm(self) -> None:
        resolver = ConfigResolver()
        with pytest.raises(JudgeFactoryUnavailableError):
            resolver.resolve(
                _plan(judge=_JUDGE_REF),
                settings=Settings(),
                deployment_judges=_catalog(factory_ref=_ABSENT_FACTORY),
            )

    def test_context_unsupported_pre_arm(self, fake_factory) -> None:
        binding = ArtifactBinding(ref="rubric.txt", sha256="a" * 64)
        resolver = ConfigResolver()
        with pytest.raises(JudgeContextUnsupportedError):
            resolver.resolve(
                _plan(judge=_JUDGE_REF, judge_context=binding),
                settings=Settings(),
                deployment_judges=_catalog(supports_static_context=False),
            )

    def test_context_without_judge_raises(self, fake_factory) -> None:
        binding = ArtifactBinding(ref="rubric.txt", sha256="a" * 64)
        resolver = ConfigResolver()
        with pytest.raises(ConfigError) as exc:
            resolver.resolve(
                _plan(judge_context=binding), settings=Settings(), deployment_judges=_catalog()
            )
        assert "named no judge" in str(exc.value)

    def test_context_bound_without_store_root_raises(self, fake_factory) -> None:
        binding = ArtifactBinding(ref="rubric.txt", sha256="a" * 64)
        resolver = ConfigResolver()
        with pytest.raises(ConfigError) as exc:
            resolver.resolve(
                _plan(judge=_JUDGE_REF, judge_context=binding),
                settings=Settings(),
                deployment_judges=_catalog(context_store_root=None),
            )
        assert "context_store_root" in str(exc.value)

    def test_bad_context_hash_propagates(self, fake_factory, tmp_path) -> None:
        root, _good = _write_context(tmp_path, "rubric.txt", "the rubric bytes")
        wrong = ArtifactBinding(ref="rubric.txt", sha256="b" * 64)  # valid hex, wrong digest
        resolver = ConfigResolver()
        with pytest.raises(JudgeContextError):
            resolver.resolve(
                _plan(judge=_JUDGE_REF, judge_context=wrong),
                settings=Settings(),
                deployment_judges=_catalog(context_store_root=root),
            )

    def test_valid_context_binds_and_projects(self, fake_factory, tmp_path) -> None:
        root, digest = _write_context(tmp_path, "rubric.txt", "the rubric bytes")
        binding = ArtifactBinding(ref="rubric.txt", sha256=digest)
        resolver = ConfigResolver()
        resolved = resolver.resolve(
            _plan(judge=_JUDGE_REF, judge_context=binding),
            settings=Settings(),
            deployment_judges=_catalog(context_store_root=root),
        )
        assert resolved.judge.context_ref == "rubric.txt"
        assert resolved.judge.context_sha256 == digest
        assert resolved.components["judge"]["context"] == {"ref": "rubric.txt", "sha256": digest}


# ---------------------------------------------------------------------------
# Receipt completeness
# ---------------------------------------------------------------------------


class TestReceipt:
    def test_receipt_no_judge(self, fake_factory) -> None:
        resolver = ConfigResolver()
        resolver.resolve(_plan(), settings=Settings(), deployment_judges=_catalog())
        r = resolver.last_receipt
        assert r.judge_requested is None
        assert r.judge_resolved is None
        assert r.judge_context_ref is None
        assert r.judge_context_sha256 is None
        assert r.fallback_used is False

    def test_receipt_authorized_judge(self, fake_factory) -> None:
        resolver = ConfigResolver()
        resolver.resolve(_plan(judge=_JUDGE_REF), settings=Settings(), deployment_judges=_catalog())
        r = resolver.last_receipt
        assert r.judge_requested == _JUDGE_REF
        assert r.judge_resolved == _JUDGE_REF
        assert r.fallback_used is False

    def test_receipt_with_context(self, fake_factory, tmp_path) -> None:
        root, digest = _write_context(tmp_path, "rubric.txt", "policy text")
        binding = ArtifactBinding(ref="rubric.txt", sha256=digest)
        resolver = ConfigResolver()
        resolver.resolve(
            _plan(judge=_JUDGE_REF, judge_context=binding),
            settings=Settings(),
            deployment_judges=_catalog(context_store_root=root),
        )
        r = resolver.last_receipt
        assert r.judge_context_ref == "rubric.txt"
        assert r.judge_context_sha256 == digest


# ---------------------------------------------------------------------------
# Adapter cache
# ---------------------------------------------------------------------------


class TestAdapterCache:
    def _spec(self) -> ResolvedJudgeSpec:
        return ResolvedJudgeSpec(
            requested_ref=_JUDGE_REF,
            resolved_ref=_JUDGE_REF,
            factory_ref=_FACTORY_NAME,
            provider_ref="anthropic",
            model_requested="claude-x",
            model_resolved="claude-x",
            prompt_envelope_sha256=None,
            context_ref=None,
            context_sha256=None,
        )

    def test_hit_miss_and_clear(self, fake_factory) -> None:
        spec = self._spec()
        a = get_or_create_adapter(spec, Settings())
        assert fake_factory["n"] == 1
        b = get_or_create_adapter(spec, Settings())
        assert b is a  # cache HIT: no second construction
        assert fake_factory["n"] == 1
        clear_adapter_cache()
        c = get_or_create_adapter(spec, Settings())
        assert c is not a  # rebuilt after clear
        assert fake_factory["n"] == 2

    def test_bound_judge_shape(self, fake_factory) -> None:
        spec = self._spec()
        adapter = get_or_create_adapter(spec, Settings())
        bound = BoundJudge(spec=spec, adapter=adapter, static_context=None)
        assert bound.spec is spec
        assert bound.adapter is adapter
        assert bound.static_context is None


# ---------------------------------------------------------------------------
# hash_projection — generic content-hash with excluded JSON pointers
# ---------------------------------------------------------------------------


class TestHashProjection:
    def test_removes_pointer(self) -> None:
        doc = {"a": 1, "b": {"c": 2, "d": 3}}
        full = content_hash(doc)
        projected = hash_projection(doc, ["/b/d"])
        assert projected != full
        # Removing /b/d yields the hash of the document with that key gone.
        assert projected == content_hash({"a": 1, "b": {"c": 2}})

    def test_difference_only_within_excluded_hashes_identically(self) -> None:
        base = {"a": 1, "b": {"c": 2, "volatile": 10}}
        other = {"a": 1, "b": {"c": 2, "volatile": 99}}
        assert hash_projection(base, ["/b/volatile"]) == hash_projection(other, ["/b/volatile"])

    def test_deterministic(self) -> None:
        doc = {"x": [1, 2, 3], "y": {"z": "keep"}}
        assert hash_projection(doc, ["/x/1"]) == hash_projection(doc, ["/x/1"])

    def test_different_exclusions_different_hash(self) -> None:
        doc = {"a": 1, "b": 2, "c": 3}
        assert hash_projection(doc, ["/a"]) != hash_projection(doc, ["/b"])

    def test_absent_pointer_is_noop(self) -> None:
        doc = {"a": 1}
        assert hash_projection(doc, ["/does/not/exist"]) == content_hash(doc)

    def test_does_not_mutate_input(self) -> None:
        doc = {"a": 1, "b": {"c": 2}}
        hash_projection(doc, ["/b/c"])
        assert doc == {"a": 1, "b": {"c": 2}}  # deep-copied, original intact

    def test_list_index_removal(self) -> None:
        doc = {"items": [10, 20, 30]}
        assert hash_projection(doc, ["/items/1"]) == content_hash({"items": [10, 30]})
