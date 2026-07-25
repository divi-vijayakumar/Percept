"""PROBE-HARNESS SCAFFOLDING — ``fake_judge``.

**Wiring-only, not a production backend.** This is the PRE-MVP viability spike's
"architecture-isolation lever": a deterministic, key-free judge double that lets Phase-A probes
(latency, delivery) exercise the runtime's *architecture* — queueing, backpressure,
fail-closed handling — without a real vendor call ever in the loop. Standing rule 2 (see
:mod:`percept.backends.fake`) still applies: it proves wiring and our logic, never model
intelligence. Verdict *content* is delegated to :class:`~percept.backends.fake.FakeVision`'s
scripted dict; this module only controls *when* a call returns and *how* it can fail.

Two probe-harness needs :class:`~percept.backends.fake.FakeVision` intentionally does not cover:

1. **Configurable, seeded latency.** :class:`LatencyProfile` samples a per-call delay (fixed,
   uniform, or lognormal) from a :class:`random.Random` seeded once at construction — never
   ``random``'s global state — so a probe run is byte-for-byte reproducible given the same seed
   and call sequence.
2. **Injectable error/timeout/malformed-output behaviour.** :class:`ErrorInjectionConfig` sets a
   configurable *rate* and a set of failure *modes* (``timeout`` / ``provider_error`` /
   ``malformed`` / ``partial``) so later fail-closed and quality probes can exercise the
   fail-closed paths (malformed/partial output, timeout, provider error -> ASK/SILENT/
   provider-failure) against a judge that misbehaves *on cue* instead of waiting for a real vendor
   outage.

Not wired into ``percept/registry.py`` or the ``load_driver``/``stream_gen`` probe entrypoints —
that wiring is follow-up work once those harness pieces exist.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from ..contracts import Verdict
from .fake import FakeVision

# -- failure surface -------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Injected provider-side failure (rate limit / 5xx / vendor outage) — probe/wiring-only.

    Mirrors the shape of a real backend's outbound-call failure (see
    :mod:`percept.backends._resilience`) so a probe's fail-closed handling can be exercised
    against the same exception surface a real adapter would raise, without a real adapter.
    """


# -- latency ----------------------------------------------------------------------------------

_LATENCY_KINDS = ("fixed", "uniform", "lognormal")


@dataclass(frozen=True)
class LatencyProfile:
    """A seedable latency distribution. Never ``time.sleep``/``random`` global state — sampling
    always draws from the :class:`random.Random` instance passed in, so the same seed + the same
    call sequence reproduces the same latencies (a determinism requirement).

    ``kind``:
      * ``"fixed"``     — always ``value_s``.
      * ``"uniform"``   — ``Random.uniform(low_s, high_s)``.
      * ``"lognormal"`` — ``Random.lognormvariate(mu, sigma)`` (mu/sigma of the underlying normal,
        in log-seconds) — a common shape for real judge-call latency (a long tail of slow calls).
    """

    kind: str = "fixed"
    value_s: float = 0.0
    low_s: float = 0.0
    high_s: float = 0.0
    mu: float = 0.0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _LATENCY_KINDS:
            raise ValueError(f"LatencyProfile.kind must be one of {_LATENCY_KINDS}, got {self.kind!r}")

    def sample(self, rng: random.Random) -> float:
        """Draw one latency sample in seconds, clamped to >= 0. Consumes ``rng`` state."""
        if self.kind == "fixed":
            return max(0.0, self.value_s)
        if self.kind == "uniform":
            return max(0.0, rng.uniform(self.low_s, self.high_s))
        # "lognormal"
        return max(0.0, rng.lognormvariate(self.mu, self.sigma))


# -- error/timeout injection -------------------------------------------------------------------

_ERROR_MODES = ("timeout", "provider_error", "malformed", "partial")


@dataclass(frozen=True)
class ErrorInjectionConfig:
    """Configurable failure injection for :class:`FakeJudge`.

    ``rate`` is the per-call probability (0..1) that a call is injected with a failure; when it
    fires, the mode is drawn uniformly from ``modes``. Both draws consume the shared seeded RNG, so
    which calls fail and with what mode is itself reproducible given the same seed.

    Modes:
      * ``"timeout"``        — raises :class:`asyncio.TimeoutError`.
      * ``"provider_error"`` — raises :class:`ProviderError` (simulated vendor 5xx/rate-limit).
      * ``"malformed"``      — *returns* (does not raise) a non-``Verdict`` object with an
        unexpected shape, simulating a response that doesn't even parse as the judge contract.
      * ``"partial"``        — *returns* a dict missing required ``Verdict`` fields (e.g. only
        ``satisfied``, no ``confidence``), simulating truncated/partial JSON from the vendor.
    """

    rate: float = 0.0
    modes: tuple[str, ...] = ("provider_error",)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"ErrorInjectionConfig.rate must be in [0, 1], got {self.rate!r}")
        if not self.modes:
            raise ValueError("ErrorInjectionConfig.modes must be non-empty")
        bad = set(self.modes) - set(_ERROR_MODES)
        if bad:
            raise ValueError(f"unknown error mode(s) {sorted(bad)}; must be a subset of {_ERROR_MODES}")


# -- the fake judge -----------------------------------------------------------------------------


@dataclass
class JudgeCallRecord:
    """One recorded call — the probe harness's raw telemetry, kept in call order."""

    latency_s: float
    injected: str | None  # None (clean call) or the injected failure mode name


class FakeJudge:
    """A deterministic, seedable, key-free judge double for the probe harness.

    Wraps a :class:`~percept.backends.fake.FakeVision` for verdict *content* (composition, not
    inheritance/rewrite — the Vision/judge seam is untouched) and adds the two probe-harness
    knobs: seeded latency and seeded error injection.

    Determinism contract: construct two instances with the same ``seed`` (and the same
    ``latency``/``errors`` config) and drive them with the same sequence of ``judge()`` calls ->
    :attr:`calls` records the identical latency + injected-mode sequence, and every call that was
    not injected with a failure returns an identical :class:`~percept.contracts.Verdict`.

    Parameters
    ----------
    seed:
        Seeds the private :class:`random.Random` instance. Never touches the global ``random``
        module, so a probe run never perturbs (or is perturbed by) unrelated code.
    vision:
        An existing :class:`FakeVision` to delegate verdict content to. If omitted, one is built
        from ``script`` (or the default scripted dict).
    script:
        Passed through to a freshly built :class:`FakeVision` when ``vision`` is not given.
    latency:
        A :class:`LatencyProfile`; defaults to zero latency.
    errors:
        An :class:`ErrorInjectionConfig`; defaults to never injecting.
    simulate_sleep:
        When True (default), ``judge()`` actually ``await asyncio.sleep(latency)`` so a probe
        observes real elapsed time (needed for queueing/backpressure probes). Set False to record
        sampled latency without paying wall-clock time (useful for fast unit tests / offline
        replay against an injectable clock).
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        vision: FakeVision | None = None,
        script: dict[str, Verdict] | None = None,
        latency: LatencyProfile | None = None,
        errors: ErrorInjectionConfig | None = None,
        simulate_sleep: bool = True,
    ) -> None:
        self._rng = random.Random(seed)
        self._vision = vision if vision is not None else FakeVision(script=script)
        self._latency = latency or LatencyProfile(kind="fixed", value_s=0.0)
        self._errors = errors or ErrorInjectionConfig(rate=0.0)
        self._simulate_sleep = simulate_sleep
        self.calls: list[JudgeCallRecord] = []

    @property
    def latencies(self) -> list[float]:
        """Sampled latency per call, in call order (seconds)."""
        return [c.latency_s for c in self.calls]

    @property
    def injected_modes(self) -> list[str | None]:
        """Injected failure mode per call, in call order (``None`` = clean call)."""
        return [c.injected for c in self.calls]

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict | dict:
        """Same call shape as :meth:`FakeVision.judge` (drops into the existing judge seam).

        Order of operations is fixed (and thus reproducible): sample latency, decide
        inject-or-not, decide mode-if-injecting, sleep (if configured), then either raise/return
        the injected failure or delegate to the wrapped :class:`FakeVision`.
        """
        latency = self._latency.sample(self._rng)
        inject = self._rng.random() < self._errors.rate
        mode = self._rng.choice(self._errors.modes) if inject else None
        self.calls.append(JudgeCallRecord(latency_s=latency, injected=mode))

        if self._simulate_sleep and latency > 0:
            await asyncio.sleep(latency)

        if mode == "timeout":
            raise asyncio.TimeoutError(f"fake_judge: injected timeout after {latency:.3f}s")
        if mode == "provider_error":
            raise ProviderError("fake_judge: injected provider error (simulated 5xx/rate-limit)")
        if mode == "malformed":
            # Not a Verdict at all — a consumer that assumes the judge contract without validating
            # must fail closed here.
            return {"unexpected_shape": True, "raw": "<garbled provider payload>"}
        if mode == "partial":
            # Verdict-shaped but missing required fields (simulated truncated JSON).
            return {"satisfied": True}

        return await self._vision.judge(
            condition,
            frame,
            roi=roi,
            exemplar=exemplar,
            corrections=corrections,
            frames=frames,
            audio=audio,
            context=context,
        )


__all__ = [
    "FakeJudge",
    "LatencyProfile",
    "ErrorInjectionConfig",
    "ProviderError",
    "JudgeCallRecord",
]
