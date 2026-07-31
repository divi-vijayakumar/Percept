"""Test-suite conftest.

The suite is OFFLINE by construction — one declared property, not 895 implicit fallbacks.

The offline/scripted RFC flips the dangerous default: a bare ``Percept.create()`` no longer
silently falls back to scripted backends — in LIVE mode it raises ``ModeError``. The whole suite
runs keyless against scripted doubles, so it declares that ONCE here (``PERCEPT_MODE=offline``)
rather than editing 895 call sites. If a test needs more than this fixture to pass, the default is
wrong — surface it, do not paper it.

The loud stderr banner is separately suppressed under pytest (see ``percept.agent._emit_banner``),
so this fixture does not need to mute it: 895 banners would train people to scroll past banners,
the exact failure the RFC exists to prevent.

The suite is also ISOLATED by construction, and for the same reason. Percept keeps real
process-global state — the environment ``load_settings()`` reads, the detector registry, the
backend factory table, the source-profile table — and 304 test modules share one interpreter. A
module that leaves one of those dirty does not fail; it makes some *later* module fail, in a file
that never touched the thing. That tax was measured at 25 false failures, every one of which passed
standalone, and it grew by one purely from adding correct tests. The fixtures below snapshot each
global before every test and put it back after, so a leak stays local to the test that caused it
and the suite's result stops depending on collection order.

They restore; they do not police. A test that deliberately registers a backend, seeds an env var,
or reloads a module still sees exactly what it asked for — it just cannot hand that on.
"""
from __future__ import annotations

import os

import pytest

# Imported at conftest scope ON PURPOSE. Each registry below is populated by module-level
# ``register(...)`` / ``seed()`` calls, so the FIRST import is what installs the shipped entries.
# Importing them here makes the seeded table the baseline every snapshot is taken against. Snapshot
# one *before* its seeding module has been imported and the restore would delete the built-ins for
# the rest of the session — an isolation fixture that causes the bug it exists to prevent.
from percept import registry as _backend_registry
from percept import source_profiles_default as _source_profiles_default  # noqa: F401 — seeds on import
from percept.harness import detectors as _detectors
from percept.sourceprofile import DEFAULT_SOURCE_REGISTRY as _source_profile_registry


def _restore(live: dict, snapshot: dict) -> None:
    """Put ``live`` back to ``snapshot`` IN PLACE — the identity matters as much as the contents.

    Every consumer of these registries holds the dict object itself (module globals, a dataclass
    field), so rebinding a fresh dict would restore the contents and still leave half the process
    reading the old one. Compare first: an untouched registry is the overwhelming common case and
    must cost nothing.
    """
    if live != snapshot:
        live.clear()
        live.update(snapshot)


@pytest.fixture(autouse=True)
def _environ_isolation():
    """Snapshot and restore the whole environment around every test.

    This is the big one. ``load_settings()`` re-reads ``PERCEPT_*`` on every call and never caches,
    so an env var that outlives its test silently re-wires every ``Percept.create()`` after it.
    The concrete failure: ``tools/percept_benchmark.py``'s ``_load_dotenv()`` copies the developer's
    gitignored ``.env`` into ``os.environ``, and one test drives that tool in-process — after which
    ``PERCEPT_VISION_BACKEND=gemini`` and a real API key are set for the remaining 280 modules. A
    bare ``Percept.create()`` then builds a live Gemini adapter and a LIVE core stops raising
    ``ModeError``, because a backend really is selected and its key really is present. 24 of the 25
    known false failures were that one leak.

    Restoring the whole mapping rather than a ``PERCEPT_*`` subset is deliberate: the credential
    names (``ANTHROPIC_API_KEY``, ``GOOGLE_AISTUDIO_API_KEY``, ...) are load-bearing for the
    fail-closed checks, and any allowlist would be a list to forget to update.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        if os.environ != snapshot:
            os.environ.clear()
            os.environ.update(snapshot)


@pytest.fixture(autouse=True)
def _offline(_environ_isolation, monkeypatch):
    """Declare offline mode. Ordered INSIDE the env snapshot so the restore is the last word.

    Requesting ``_environ_isolation`` here is not decoration: it fixes the teardown order. The
    snapshot is taken before ``PERCEPT_MODE`` is set and put back after ``monkeypatch`` has undone
    it, so neither fixture can leave the other's edit behind.
    """
    monkeypatch.setenv("PERCEPT_MODE", "offline")


@pytest.fixture(autouse=True)
def _detector_registry_isolation():
    """Snapshot and restore ``percept.harness.detectors`` — its table AND its module namespace.

    Two distinct leaks, one fixture.

    *Contents*: ``register`` / ``unregister`` / ``clear_detectors`` mutate a process-global dict, so
    a module that registers a skill and does not tear it down changes the detector vocabulary every
    later module compiles against.

    *Identity*: one suite tears down by calling ``importlib.reload`` on the module. Reload
    re-executes the source into the same namespace, which mints NEW class objects —
    ``WakePolicy``, ``Boundary``, ``DetectorManifest`` — while ``percept.harness.tier0`` still holds
    the pre-reload ones from its own ``from .detectors import ...``. The ``isinstance(raw,
    WakePolicy)`` guard in ``coerce_wake_policy`` then rejects a perfectly good policy, its weights
    are silently dropped, and the noisy-OR fusion degrades to ``max(signals)`` for the rest of the
    run. Restoring the namespace dict puts the original classes back on both sides of that seam.

    The namespace restore is conditional on a reload actually having happened (a sentinel class
    identity), so ordinary tests — including anything ``monkeypatch`` did to a module attribute —
    are left alone.
    """
    namespace = dict(_detectors.__dict__)
    registry = dict(_detectors._REGISTRY)
    preset = {k: list(v) for k, v in _detectors._TYPE_DETECTOR_PRESET.items()}
    try:
        yield
    finally:
        if _detectors.WakePolicy is not namespace["WakePolicy"]:  # the module was reloaded
            _detectors.__dict__.clear()
            _detectors.__dict__.update(namespace)
        # After the namespace restore these names are the ORIGINAL dict objects again, which is why
        # the contents pass has to come second.
        _restore(_detectors._REGISTRY, registry)
        _restore(_detectors._TYPE_DETECTOR_PRESET, preset)


@pytest.fixture(autouse=True)
def _backend_registry_isolation():
    """Snapshot and restore the seam -> {name -> factory} table in ``percept.registry``.

    ``register("vision", "claude", ...)`` is last-writer-wins by design — that is how a product
    overrides a shipped backend. Left in place by a test, the same call shadows the shipped factory
    for every ``Percept.create()`` afterwards, and the seam that resolves is not the one the later
    test asked for. Restored per seam, in place, for the reason ``_restore`` documents.
    """
    snapshot = {seam: dict(table) for seam, table in _backend_registry._REGISTRY.items()}
    try:
        yield
    finally:
        for seam in [s for s in _backend_registry._REGISTRY if s not in snapshot]:
            del _backend_registry._REGISTRY[seam]
        for seam, table in snapshot.items():
            _restore(_backend_registry._REGISTRY.setdefault(seam, {}), table)


@pytest.fixture(autouse=True)
def _source_profile_registry_isolation():
    """Snapshot and restore ``DEFAULT_SOURCE_REGISTRY``.

    Same shape as the backend table: an open, name-keyed process-global that third parties and tests
    both write to. A fake profile left registered either shadows a shipped one or makes a later
    ``names()`` assertion count an entry that module never registered.
    """
    live = _source_profile_registry._profiles
    snapshot = dict(live)
    try:
        yield
    finally:
        _restore(live, snapshot)
