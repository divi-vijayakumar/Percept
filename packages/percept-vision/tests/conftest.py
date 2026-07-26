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

if sys.version_info < (3, 11):
    import tomli

    # Python 3.11 added the stdlib name used by the CLI/replay tests. Keep the
    # canonical 3.10 runner on the same import surface without per-test skips.
    sys.modules.setdefault("tomllib", tomli)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("PERCEPT_MODE", "offline")
