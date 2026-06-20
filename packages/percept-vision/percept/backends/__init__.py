"""Backend adapters. ``fake`` ships in Phase 0; vendor adapters land in later phases.

Resolution is by name through :mod:`percept.registry`; import the fakes directly only in
tests/fixtures.
"""

from .fake import (
    FakeConductor,
    FakeSpecialist,
    FakeSTT,
    FakeTTS,
    FakeVision,
)

__all__ = [
    "FakeVision",
    "FakeSpecialist",
    "FakeConductor",
    "FakeSTT",
    "FakeTTS",
]
