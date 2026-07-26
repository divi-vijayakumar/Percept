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
    ScriptedConductor,
)
from .fake_judge import (
    ErrorInjectionConfig,
    FakeJudge,
    JudgeCallRecord,
    LatencyProfile,
    ProviderError,
)
from .judge_pool import (
    JudgeCapability,
    JudgePool,
    JudgeRoute,
    JudgeRouteReceipt,
)

__all__ = [
    "FakeVision",
    "FakeSpecialist",
    "FakeConductor",
    "ScriptedConductor",
    "FakeSTT",
    "FakeTTS",
    # probe-harness scaffolding — wiring-only, not a production backend
    "FakeJudge",
    "LatencyProfile",
    "ErrorInjectionConfig",
    "ProviderError",
    "JudgeCallRecord",
    "JudgeCapability",
    "JudgePool",
    "JudgeRoute",
    "JudgeRouteReceipt",
]
