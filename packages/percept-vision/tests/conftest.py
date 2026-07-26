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
"""
from __future__ import annotations

import sys

import pytest

from percept.agent import (
    Percept,
    _REVIEWED_TEST_IN_PROCESS_PROVIDER_CAPABILITY,
)

if sys.version_info < (3, 11):
    import tomli

    # Python 3.11 added the stdlib name used by the CLI/replay tests. Keep the
    # canonical 3.10 runner on the same import surface without per-test skips.
    sys.modules.setdefault("tomllib", tomli)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("PERCEPT_MODE", "offline")


@pytest.fixture(autouse=True, scope="session")
def _reviewed_shared_memory_provider_fixtures():
    """Grant this repository's legacy shared-memory oracles an opaque capability.

    Production defaults every unmarked adapter to process isolation. Many
    unit/eval fixtures intentionally inspect mutable call ledgers or coordinate
    asyncio.Events, so the central harness authenticates those cores explicitly
    instead of teaching production code to trust test-looking marker strings.
    Process-boundary regressions explicitly remove the capability from the
    exact cores whose production behavior they exercise.
    """
    original_create = Percept.create.__func__
    original_descriptor = Percept.__dict__["create"]

    def create_reviewed_core(cls, *args, **kwargs):
        core = original_create(cls, *args, **kwargs)
        core._reviewed_test_in_process_provider_capability = (
            _REVIEWED_TEST_IN_PROCESS_PROVIDER_CAPABILITY
        )
        return core

    setattr(
        Percept,
        "create",
        classmethod(create_reviewed_core),
    )
    try:
        yield
    finally:
        setattr(Percept, "create", original_descriptor)
