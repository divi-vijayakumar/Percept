"""The PUBLISHED extension-seam conformance suite — the runnable definition of "my plug-in works".

Percept's cognition core never imports a vendor SDK (invariant 7): every model lives behind the
frozen :class:`percept.contracts.Vision` seam and is injected either as an instance
(``Percept.create(vision=my_adapter)``) or by NAME through :mod:`percept.registry`. That makes the
Vision seam the public extension point — and this module is its *executable* specification.

An external adapter author runs::

    python -m percept.conformance mypkg.vision:MyVision

or, in their own test suite::

    from percept.conformance import check_vision_adapter

    def test_conformance():
        report = check_vision_adapter(MyVision(), frame=load_test_frame())
        assert report.ok, report.format()

What this suite IS: a check that the adapter honours the **shape and the failure mode** of the seam
— the method set, the optional-kwarg escalations the brain may pass (``roi`` / ``frames`` / ``audio``
/ ``context`` / ``corrections``), the :class:`~percept.contracts.Verdict` invariants the gate reads,
and that the adapter survives **fail-closed injection** into a live core (it is NOT silently replaced
by a scripted double, and a live core with NO vision refuses to run at all).

What this suite is NOT: a measure of the model's intelligence. Standing rule 2 applies — accuracy is
an L2/L3 eval question (``percept.bench``, the golden sets), never a conformance question. No check
here asserts a particular ``satisfied`` value for a particular frame, because a conformant adapter is
free to be wrong; it is not free to be *shaped* wrong.

Check severity:

``REQUIRED``
    A violation breaks the core. The report is not ``ok`` and the CLI exits non-zero.
``ADVISORY``
    A recommendation (cost accounting, graceful degradation of a modality the backend lacks). It is
    reported and never fails the run.

THE OTHER SEAMS. An earlier decision settled which extension points survive, so this module covers all
three of them with ONE reporting shape (:class:`CheckResult` / :class:`ConformanceReport`) and ONE
severity vocabulary:

``vision``
    :func:`check_vision_adapter` — the cloud judge behind :class:`percept.contracts.Vision`.
``detector``
    :func:`check_detector` — the bring-your-own **admission detector**
    (:func:`percept.harness.detectors.register_detector`): a client's OCR / phash / CLIP model that
    says *when to look closely*. A broken one must fail LOUDLY here rather than silently
    under-admitting in production, where the symptom is "percept never fired" with no error at all.
``structural`` / ``semantic``
    :func:`check_tier0_adapter` — the ``structural_adapter`` / ``semantic_adapter`` callables
    :func:`percept.runtime.run` now accepts, which fill the matching channels of
    :class:`~percept.contracts.Tier0Signal`.

THE SAME SCOPE DISCLAIMER APPLIES TO EVERY ROLE, and every report prints it: this suite proves a
plug-in is WELL-FORMED. It is NOT a measure of the model's intelligence or quality. A green detector
report does NOT mean "my OCR model is good" — it means the SDK can call it, route it, and count what
it emits. Quality is :mod:`percept.eval`'s job (``percept-eval``, a paired differential A/B against
your own fixtures and your own pre-registered bars).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import math
import sys
import textwrap
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Iterable

from .contracts import CONTRACT_VERSION, Box, Goal, Observation, Verdict, Vision

#: Semver of the CONFORMANCE SUITE itself — the set of checks and their severities — distinct from
#: ``percept.contracts.CONTRACT_VERSION`` (the DTO shapes) and ``percept.watch.WATCH_VERSION`` (the
#: entrypoint). Bump MAJOR when an existing check gets stricter or a REQUIRED check is added (an
#: adapter that passed may now fail); MINOR for a new ADVISORY check; PATCH for wording/fixes.
#: 1.1.0 — MINOR: the ``detector`` and ``structural``/``semantic`` roles are ADDITIVE (new suites for
#: seams that had none). No existing Vision check changed or got stricter, so an adapter that passed
#: 1.0.0 still passes.
CONFORMANCE_VERSION = "1.1.0"

REQUIRED = "required"
ADVISORY = "advisory"

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

#: The extension seams this module can conformance-check. One report shape, one severity vocabulary.
ROLE_VISION = "vision"
ROLE_DETECTOR = "detector"
ROLE_STRUCTURAL = "structural"
ROLE_SEMANTIC = "semantic"

#: ``role -> the report's title line``.
ROLE_TITLES: dict[str, str] = {
    ROLE_VISION: "Vision adapter",
    ROLE_DETECTOR: "admission detector",
    ROLE_STRUCTURAL: "tier-0 structural_adapter",
    ROLE_SEMANTIC: "tier-0 semantic_adapter",
}

#: The tier-0 adapter roles — the two channels ``percept.runtime.run()`` exposes.
TIER0_ROLES = (ROLE_STRUCTURAL, ROLE_SEMANTIC)

#: Printed on EVERY report, for EVERY role. Conformance is a well-formedness proof, never a quality
#: score — and the one misreading that costs a client real money is "green conformance = good model".
SCOPE_DISCLAIMER = (
    "SCOPE: this suite proves your plug-in is WELL-FORMED — that percept can call it, route it, "
    "and read what it returns. It is NOT a measure of the model's intelligence or quality: no check "
    "here asserts a correct answer, because a conformant plug-in is free to be WRONG; it is not free "
    "to be SHAPED wrong. A green report does NOT mean 'my model is good'. Quality is percept eval's "
    "job (percept-eval: a paired A/B on your fixtures against your own pre-registered bars)."
)

#: The keyword-only escalations the brain MAY pass to ``judge``. Each must be accepted and must
#: default to ``None`` so a single-frame call is unchanged — see ``Vision.judge``'s docstring.
JUDGE_OPTIONAL_KWARGS = ("roi", "exemplar", "corrections", "frames", "audio", "context")

#: Severity of each ``judge`` escalation. ``roi`` and ``corrections`` are REQUIRED — the fovea crop
#: and assertion-beats-observation are core cognition, not optional modalities. A strip / audio clip /
#: enricher context / exemplar is a MODALITY the backend may genuinely lack: it must be accepted and
#: may be ignored (degrade gracefully), so those are ADVISORY.
ESCALATION_SEVERITY: dict[str, str] = {
    "roi": REQUIRED,
    "corrections": REQUIRED,
    "frames": ADVISORY,
    "audio": ADVISORY,
    "context": ADVISORY,
    "exemplar": ADVISORY,
}

#: Every check that needs an actual CALL into the adapter (or a constructed core), in report order.
#: ``live_calls=False`` marks exactly these ``SKIP``.
CALL_CHECKS: tuple[tuple[str, str], ...] = (
    ("judge.minimal_call_returns_verdict", REQUIRED),
    *((f"judge.escalation_{k}", sev) for k, sev in ESCALATION_SEVERITY.items()),
    ("judge.reports_usage", ADVISORY),
    ("detect.returns_normalised_observations", REQUIRED),
    ("detect.tolerates_empty_goals", REQUIRED),
    ("detect_events.returns_list", REQUIRED),
)

#: The fail-closed INJECTION checks — they construct a real ``Percept`` core, so ``check_injection``
#: turns them off independently of the adapter calls.
INJECTION_CHECKS: tuple[tuple[str, str], ...] = (
    ("injection.instance_survives_create", REQUIRED),
    ("injection.not_a_scripted_double", ADVISORY),
    ("injection.live_core_is_fail_closed", REQUIRED),
)

#: The DETECTOR checks that need to actually PUSH samples through the detector, in report order.
#: A frame-backed detector (``signal == "frame"``) with no caller-supplied samples marks exactly these
#: ``SKIP`` — there is no universal probe frame, and a fabricated one would prove nothing.
DETECTOR_CALL_CHECKS: tuple[tuple[str, str], ...] = (
    ("detector.push_returns_boundaries", REQUIRED),
    ("detector.boundaries_are_attributed", REQUIRED),
    ("detector.attribution_matches_registry_name", ADVISORY),
    ("detector.tolerates_degenerate_signal", REQUIRED),
    ("detector.deterministic_on_fixed_input", REQUIRED),
    ("detector.reset_restores_initial_state", REQUIRED),
)

#: The DETECTOR checks that run without pushing anything (manifest, construction, routing) plus the
#: registry round-trip. Listed for introspection/self-tests; the suite emits them in this order.
DETECTOR_STATIC_CHECKS: tuple[tuple[str, str], ...] = (
    ("detector.manifest_well_formed", REQUIRED),
    ("detector.manifest_is_descriptive", ADVISORY),
    ("detector.manifest_declares_category", ADVISORY),
    ("detector.factory_constructs", REQUIRED),
    ("detector.push_and_reset_present", REQUIRED),
    ("detector.signal_channel_is_routable", REQUIRED),
    ("detector.selectable_by_name_into_watchspec", REQUIRED),
)

#: The tier-0 adapter checks (``structural_adapter`` / ``semantic_adapter``), in report order.
TIER0_STATIC_CHECKS: tuple[tuple[str, str], ...] = (
    ("adapter.is_callable", REQUIRED),
    ("adapter.role_is_exposed_on_run", REQUIRED),
)
TIER0_CALL_CHECKS: tuple[tuple[str, str], ...] = (
    ("adapter.returns_bounded_score", REQUIRED),
    ("adapter.tolerates_degenerate_frame", REQUIRED),
    ("adapter.does_not_crash_the_gate", REQUIRED),
    ("adapter.score_reaches_the_tier0_signal", REQUIRED),
    ("adapter.is_deterministic", ADVISORY),
)

#: The frame handed to the adapter when the caller supplies none. A frame is an OPAQUE per-backend
#: type, so there is no universal probe: this sentinel only exercises the *plumbing*, and a real
#: adapter should be handed a real frame via ``frame=``.
PROBE_FRAME = "percept-conformance-probe"
PROBE_CONDITION = "a person is visible"


def _wrap_disclaimer(width: int = 96) -> list[str]:
    """The scope disclaimer, wrapped and indented — printed on every report, for every role."""
    return ["  " + line for line in textwrap.wrap(SCOPE_DISCLAIMER, width=width)]


@dataclass(frozen=True)
class CheckResult:
    """One conformance check's outcome."""

    name: str
    status: str  # PASS | FAIL | SKIP
    severity: str = REQUIRED
    detail: str = ""

    @property
    def blocking(self) -> bool:
        """True iff this outcome makes the whole report non-conformant."""
        return self.status == FAIL and self.severity == REQUIRED


@dataclass(frozen=True)
class ConformanceReport:
    """The result of running the suite against one plug-in, in any :data:`ROLE_TITLES` role."""

    adapter: str
    results: tuple[CheckResult, ...] = ()
    contract_version: str = CONTRACT_VERSION
    conformance_version: str = CONFORMANCE_VERSION
    frame_was_probe: bool = False
    #: Which extension seam was checked — ``vision`` (default, back-compatible), ``detector``,
    #: ``structural`` or ``semantic``. Only affects reporting; the pass/fail rule is role-independent.
    role: str = ROLE_VISION

    @property
    def ok(self) -> bool:
        return not any(r.blocking for r in self.results)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status == FAIL)

    @property
    def blocking_failures(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.blocking)

    @property
    def skipped(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status == SKIP)

    @property
    def complete(self) -> bool:
        """True iff every REQUIRED check actually RAN. ``ok`` without ``complete`` is a partial pass —
        e.g. a static-only run, which proves shape but never that the adapter answers."""
        return not any(r.status == SKIP and r.severity == REQUIRED for r in self.results)

    def get(self, name: str) -> CheckResult | None:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def format(self) -> str:
        """A human-readable report — what the CLI prints and what an assert message should carry."""
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}
        title = ROLE_TITLES.get(self.role, self.role)
        lines = [
            f"percept {title} conformance — {self.adapter}",
            f"  suite {self.conformance_version} · contract {self.contract_version}",
            *_wrap_disclaimer(),
        ]
        if self.frame_was_probe and self.role in TIER0_ROLES:
            lines.extend(
                "  " + ln
                for ln in textwrap.wrap(
                    "NOTE: no frames= were supplied, so a synthetic grayscale ladder was used. The "
                    "TYPE is exact (the gate always hands you a tier0.Frame) but the CONTENT is not "
                    "your workload — re-run with frames from your own stream.",
                    width=96,
                )
            )
        elif self.frame_was_probe:
            lines.append(
                "  NOTE: no frame= was supplied, so the opaque sentinel frame was used. This "
                "exercises plumbing only — re-run with a real frame for a meaningful result."
            )
        for r in self.results:
            sev = "" if r.severity == REQUIRED else f" ({ADVISORY})"
            lines.append(f"  [{mark.get(r.status, r.status)}] {r.name}{sev}")
            if r.detail:
                lines.append(f"        {r.detail}")
        if not self.ok:
            verdict = "NOT CONFORMANT"
        elif self.complete:
            verdict = "CONFORMANT"
        else:
            verdict = "PARTIALLY CHECKED — conformant so far, but required checks were SKIPPED"
        n_fail = len(self.blocking_failures)
        lines.append(
            f"  => {verdict} ({n_fail} blocking failure(s), {len(self.skipped)} skipped, "
            f"of {len(self.results)} checks)"
        )
        return "\n".join(lines)


class _Recorder:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, name: str, status: str, *, severity: str = REQUIRED, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, status=status, severity=severity, detail=detail))

    async def check(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        severity: str = REQUIRED,
        on_pass: str = "",
    ) -> bool:
        """Run one check. It PASSES unless it raises; an ``AssertionError`` carries the detail."""
        try:
            out = fn()
            if inspect.isawaitable(out):
                await out
        except AssertionError as exc:
            self.add(name, FAIL, severity=severity, detail=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 — any raise is a conformance failure, reported not swallowed
            self.add(name, FAIL, severity=severity, detail=f"{type(exc).__name__}: {exc}")
            return False
        self.add(name, PASS, severity=severity, detail=on_pass)
        return True

    def check_sync(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        severity: str = REQUIRED,
        on_pass: str = "",
    ) -> bool:
        """Sync twin of :meth:`check` — the detector and tier-0 adapter seams are SYNCHRONOUS (the
        gate calls them inline, per frame, on the hot path), so there is nothing to await. Same
        fail-closed rule: any raise is a reported failure, never a swallowed one."""
        try:
            fn()
        except AssertionError as exc:
            self.add(name, FAIL, severity=severity, detail=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            self.add(name, FAIL, severity=severity, detail=f"{type(exc).__name__}: {exc}")
            return False
        self.add(name, PASS, severity=severity, detail=on_pass)
        return True


def _skip_call_checks(rec: "_Recorder", detail: str) -> None:
    """Record every call-based check as ``SKIP`` — an honest partial report, never a silent pass."""
    for check_name, severity in CALL_CHECKS + INJECTION_CHECKS:
        rec.add(check_name, SKIP, severity=severity, detail=detail)


# --- Verdict / Observation invariants -------------------------------------------------------------


def assert_verdict(value: Any, *, where: str) -> Verdict:
    """The gate's contract on a judge return. Raises ``AssertionError`` naming what is wrong."""
    if not isinstance(value, Verdict):
        raise AssertionError(
            f"{where} returned {type(value).__name__}, expected percept.contracts.Verdict "
            "(import it — do not duck-type a look-alike; the gate reads the dataclass)"
        )
    if not isinstance(value.satisfied, bool):
        raise AssertionError(
            f"{where}: Verdict.satisfied is {type(value.satisfied).__name__}, must be a bool "
            "(the gate branches on it; a truthy string or 0/1 int is not conformant)"
        )
    if not isinstance(value.confidence, (int, float)) or isinstance(value.confidence, bool):
        raise AssertionError(
            f"{where}: Verdict.confidence is {type(value.confidence).__name__}, must be a float"
        )
    if not (0.0 <= float(value.confidence) <= 1.0):
        raise AssertionError(
            f"{where}: Verdict.confidence={value.confidence} is outside [0.0, 1.0] — the gate "
            "compares it against a probability threshold, so an unnormalised score breaks firing"
        )
    if not isinstance(value.evidence, str):
        raise AssertionError(f"{where}: Verdict.evidence must be a str (got {type(value.evidence).__name__})")
    if not isinstance(value.usage, dict):
        raise AssertionError(f"{where}: Verdict.usage must be a dict (got {type(value.usage).__name__})")
    if not isinstance(value.events, list):
        raise AssertionError(f"{where}: Verdict.events must be a list (got {type(value.events).__name__})")
    return value


def assert_observations(value: Any, *, where: str) -> list[Observation]:
    """``detect`` must return a list of :class:`Observation` with normalised boxes."""
    if not isinstance(value, list):
        raise AssertionError(f"{where} returned {type(value).__name__}, expected a list of Observation")
    for i, obs in enumerate(value):
        if not isinstance(obs, Observation):
            raise AssertionError(
                f"{where}[{i}] is {type(obs).__name__}, expected percept.contracts.Observation"
            )
        if not isinstance(obs.label, str) or not obs.label:
            raise AssertionError(f"{where}[{i}]: Observation.label must be a non-empty str")
        if not isinstance(obs.box, Box):
            raise AssertionError(f"{where}[{i}]: Observation.box must be a percept.contracts.Box")
        for axis in ("x", "y", "w", "h"):
            v = getattr(obs.box, axis)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise AssertionError(f"{where}[{i}]: Box.{axis} must be a float")
            if not (0.0 <= float(v) <= 1.0):
                raise AssertionError(
                    f"{where}[{i}]: Box.{axis}={v} is outside [0,1] — boxes are NORMALISED image "
                    "coordinates (the entity graph's IoU matching assumes it); divide by width/height"
                )
        if not (0.0 <= float(obs.confidence) <= 1.0):
            raise AssertionError(f"{where}[{i}]: Observation.confidence={obs.confidence} outside [0,1]")
        if not isinstance(obs.attrs, dict):
            raise AssertionError(f"{where}[{i}]: Observation.attrs must be a dict[str, str]")
    return value


# --- the suite ------------------------------------------------------------------------------------


async def acheck_vision_adapter(
    adapter: Any,
    *,
    frame: Any = None,
    frames: Iterable[Any] | None = None,
    condition: str = PROBE_CONDITION,
    goals: list[Goal] | None = None,
    check_injection: bool = True,
    live_calls: bool = True,
    name: str | None = None,
) -> ConformanceReport:
    """Run the Vision conformance suite against ``adapter`` (async form).

    Args:
        adapter: The Vision adapter INSTANCE under test (already constructed — construction is the
            author's business; the seam contract starts at the methods).
        frame: A real frame of whatever opaque type the adapter accepts. Strongly recommended; when
            omitted a sentinel is used and the report says so.
        frames: OPTIONAL earlier context frames (oldest→newest) for the temporal-strip escalation.
        condition: The plain-language condition to judge. NO assertion is made about the ANSWER.
        goals: Goals passed to ``detect``; defaults to one throwaway goal.
        check_injection: Also prove the adapter survives fail-closed injection into a real core.
        live_calls: When ``False``, run only the STATIC checks (method set, protocol, signature) and
            mark every call-based check ``SKIP``. Use it to conformance-check an adapter whose vendor
            SDK or key is absent in this environment — a static pass is a real (partial) result, and
            the report says plainly which checks did not run.
        name: Display name for the report (defaults to the class's qualified name).
    """
    rec = _Recorder()
    probe = frame is None and live_calls
    frame_in = PROBE_FRAME if probe else frame
    strip = list(frames) if frames is not None else [frame_in]
    goal_list = (
        goals
        if goals is not None
        else [Goal(id="conformance", condition=condition, say="conformance probe")]
    )
    display = name or f"{type(adapter).__module__}.{type(adapter).__qualname__}"

    # 1. the method set --------------------------------------------------------------------------
    def _methods() -> None:
        missing = [m for m in ("judge", "detect", "detect_events") if not callable(getattr(adapter, m, None))]
        if missing:
            raise AssertionError(
                f"missing Vision method(s): {missing}. The seam is three methods — judge (the firing "
                "contract), detect (the entity substrate), detect_events (temporal; may return [])."
            )

    has_methods = await rec.check("vision.methods_present", _methods)

    def _protocol() -> None:
        if not isinstance(adapter, Vision):
            raise AssertionError(
                "isinstance(adapter, percept.contracts.Vision) is False — the Protocol is "
                "runtime_checkable, so a missing or non-callable method fails here"
            )

    await rec.check("vision.runtime_protocol", _protocol)

    if not has_methods:
        rec.add("judge.is_coroutine_function", SKIP, detail="skipped: judge() is missing")
        rec.add("judge.accepts_optional_escalations", SKIP, detail="skipped: judge() is missing")
        _skip_call_checks(rec, "skipped: the adapter is missing Vision method(s)")
        return ConformanceReport(adapter=display, results=tuple(rec.results), frame_was_probe=probe)

    # 2. judge is async and accepts the escalation kwargs -----------------------------------------
    def _judge_async() -> None:
        if not inspect.iscoroutinefunction(getattr(adapter, "judge")):
            raise AssertionError(
                "judge() is not a coroutine function — the brain awaits it inside the async "
                "perceive loop. Define it as `async def judge(...)`."
            )

    await rec.check("judge.is_coroutine_function", _judge_async)

    def _judge_kwargs() -> None:
        try:
            sig = inspect.signature(adapter.judge)
        except (TypeError, ValueError) as exc:  # pragma: no cover — exotic callables
            raise AssertionError(f"judge() has no introspectable signature: {exc}") from exc
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return  # **kwargs absorbs every escalation — conformant
        bad_missing = [k for k in JUDGE_OPTIONAL_KWARGS if k not in params]
        if bad_missing:
            raise AssertionError(
                f"judge() does not accept {bad_missing} — the brain passes these keyword-only "
                "escalations (ROI crop, temporal strip, audio, enricher context, user corrections) "
                "per the concern's PerceptionSpec. Accept and ignore what you cannot use, or take "
                "**kwargs; raising TypeError takes the whole watch down."
            )
        bad_default = [
            k for k in JUDGE_OPTIONAL_KWARGS if params[k].default is not None
        ]
        if bad_default:
            raise AssertionError(
                f"judge() params {bad_default} do not default to None — every escalation is OPTIONAL "
                "so a plain single-frame call is byte-for-byte the base case."
            )

    await rec.check("judge.accepts_optional_escalations", _judge_kwargs)

    if not live_calls:
        _skip_call_checks(rec, "skipped: live_calls=False (static conformance only)")
        return ConformanceReport(adapter=display, results=tuple(rec.results), frame_was_probe=probe)

    # 3. the minimal call and the Verdict invariants ----------------------------------------------
    async def _judge_minimal() -> None:
        assert_verdict(await adapter.judge(condition, frame_in), where="judge(condition, frame)")

    judged = await rec.check("judge.minimal_call_returns_verdict", _judge_minimal)

    # 4. each escalation, one at a time — a backend lacking a modality DEGRADES, never raises -----
    payloads: dict[str, dict[str, Any]] = {
        "roi": {"roi": Box(0.25, 0.25, 0.5, 0.5)},
        "corrections": {"corrections": {"the mug": "is a travel mug"}},
        "frames": {"frames": strip},
        "audio": {"audio": b""},
        "context": {"context": {"conformance": "probe"}},
        "exemplar": {"exemplar": frame_in},
    }
    escalations = tuple((kw, payloads[kw], sev) for kw, sev in ESCALATION_SEVERITY.items())
    if judged:
        for kw, payload, severity in escalations:

            async def _call(payload: dict[str, Any] = payload, kw: str = kw) -> None:
                assert_verdict(
                    await adapter.judge(condition, frame_in, **payload),
                    where=f"judge(..., {kw}=...)",
                )

            await rec.check(
                f"judge.escalation_{kw}",
                _call,
                severity=severity,
                on_pass="" if severity == REQUIRED else "accepted (or gracefully ignored)",
            )
    else:
        for kw, _payload, severity in escalations:
            rec.add(
                f"judge.escalation_{kw}",
                SKIP,
                severity=severity,
                detail="skipped: the minimal judge() call already failed",
            )

    # 5. cost accounting — advisory, but it is what $/stream-hour is computed from ----------------
    async def _usage() -> None:
        v = await adapter.judge(condition, frame_in)
        if not v.usage:
            raise AssertionError(
                "Verdict.usage is empty — report input_tokens / output_tokens / cost_usd so "
                "percept.observability.judge_usage can cost the stream. Optional, not free."
            )

    if judged:
        await rec.check("judge.reports_usage", _usage, severity=ADVISORY)
    else:
        rec.add("judge.reports_usage", SKIP, severity=ADVISORY, detail="skipped: judge() failed")

    # 6. detect / detect_events -------------------------------------------------------------------
    async def _detect() -> None:
        assert_observations(await adapter.detect(frame_in, goal_list), where="detect(frame, goals)")

    await rec.check("detect.returns_normalised_observations", _detect)

    async def _detect_no_goals() -> None:
        assert_observations(await adapter.detect(frame_in, []), where="detect(frame, [])")

    await rec.check("detect.tolerates_empty_goals", _detect_no_goals)

    async def _detect_events() -> None:
        out = await adapter.detect_events([(0.0, frame_in)], goal_list)
        if not isinstance(out, list):
            raise AssertionError(
                f"detect_events returned {type(out).__name__}, expected a list (return [] if the "
                "backend does no temporal detection — it is an optional capability, not an error)"
            )

    await rec.check("detect_events.returns_list", _detect_events)

    # 7. fail-closed injection ---------------------------------------------------------------------
    if check_injection:
        rec.results.extend(_injection_checks(adapter))
    else:
        for check_name, severity in INJECTION_CHECKS:
            rec.add(check_name, SKIP, severity=severity, detail="skipped: check_injection=False")

    return ConformanceReport(adapter=display, results=tuple(rec.results), frame_was_probe=probe)


def _injection_checks(adapter: Any) -> list[CheckResult]:
    """Prove the adapter survives injection into a real core, and that the core is fail-closed.

    Two properties, both public guarantees of the seam:

    1. ``Percept.create(vision=adapter)`` keeps YOUR instance on ``core.vision`` and does NOT classify
       it as a scripted double — no silent substitution.
    2. A LIVE core with NO vision selected REFUSES to construct (``ModeError``) rather than quietly
       falling back to a scripted double. That is the fail-closed half of the injection contract, and
       it is what makes "my adapter is the one running" checkable.
    """
    from .agent import ModeError, Percept, _is_scripted_backend

    out: list[CheckResult] = []
    try:
        core = Percept.create(vision=adapter, mode="offline", discover_plugins=False)
        if core.vision is not adapter:
            raise AssertionError(
                f"Percept.create(vision=adapter).vision is {type(core.vision).__name__}, not your "
                "instance — the adapter was substituted"
            )
    except AssertionError as exc:
        out.append(CheckResult("injection.instance_survives_create", FAIL, REQUIRED, str(exc)))
    except Exception as exc:  # noqa: BLE001
        out.append(
            CheckResult(
                "injection.instance_survives_create", FAIL, REQUIRED, f"{type(exc).__name__}: {exc}"
            )
        )
    else:
        out.append(CheckResult("injection.instance_survives_create", PASS, REQUIRED))
        # ADVISORY, and it can only fire for percept's own in-core doubles: ``_is_scripted_backend``
        # decides on the CLASS's module, so a third-party adapter always passes. It is here because the
        # health surface (``percept.agent._scripted_seams``) is what tells an operator "you are running
        # a scripted double" — an adapter author should SEE that their adapter is not one.
        if _is_scripted_backend(core.vision):
            out.append(
                CheckResult(
                    "injection.not_a_scripted_double",
                    FAIL,
                    ADVISORY,
                    "this adapter lives in percept.backends.fake, so the core reports it as a "
                    "SCRIPTED seam (health: scripted_backends). Expected for the in-core doubles; "
                    "a third-party adapter never trips this.",
                )
            )
        else:
            out.append(CheckResult("injection.not_a_scripted_double", PASS, ADVISORY))

    try:
        Percept.create(mode="live", discover_plugins=False)
    except ModeError:
        out.append(CheckResult("injection.live_core_is_fail_closed", PASS, REQUIRED))
    except Exception as exc:  # noqa: BLE001
        out.append(
            CheckResult(
                "injection.live_core_is_fail_closed",
                FAIL,
                REQUIRED,
                f"expected ModeError for a live core with no vision, got {type(exc).__name__}: {exc}",
            )
        )
    else:
        out.append(
            CheckResult(
                "injection.live_core_is_fail_closed",
                FAIL,
                REQUIRED,
                "a LIVE core with no vision constructed successfully — the core is NOT fail-closed "
                "in this environment (check PERCEPT_VISION_BACKEND / PERCEPT_MODE in your env)",
            )
        )
    return out


def check_vision_adapter(adapter: Any, **kwargs: Any) -> ConformanceReport:
    """Synchronous wrapper around :func:`acheck_vision_adapter` — the form a test suite calls."""
    return asyncio.run(acheck_vision_adapter(adapter, **kwargs))


# --- the admission-detector seam --------------------------------------------------------
#
# What a client actually plugs in here: an OCR / phash / embedding model that answers "the document
# changed at t=8.2" — WHEN to look closely. It never answers WHAT the frame says (see
# docs/reference/registration.md, "Admission is not extraction"). The failure mode this suite exists
# to prevent is the silent one: a detector whose `signal` is misspelled gets routed to the motion
# channel by DetectorBank, or one that is never registered is DROPPED by the selection firewall — in
# both cases the watch simply under-admits, with no exception, no log, and no way to see it at 2am.


#: The fixed scalar probe stream: a deterministic damped oscillation in ``[0, 1]``, 40 samples at
#: 10 Hz. Deterministic by construction (no RNG, no clock), which is what makes the determinism and
#: reset checks meaningful. A frame-backed detector needs REAL frames instead — see ``samples=``.
def _probe_samples() -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(40):
        # two full cycles, amplitude 0..1, plus a small fixed offset so a threshold detector arms
        v = 0.5 - 0.5 * math.cos(2.0 * math.pi * (i / 10.0))
        out.append((round(i * 0.1, 6), round(0.05 + 0.9 * v, 6)))
    return out


#: Degenerate scalar streams every detector must survive: dead-flat silence, pinned-high saturation,
#: and a single sample. These are what a real stream looks like when a camera is capped or a mic is
#: muted, and a detector that divides by a range or indexes ``[-2]`` dies here rather than in prod.
def _degenerate_samples() -> list[list[tuple[float, float]]]:
    return [
        [(i * 0.1, 0.0) for i in range(10)],
        [(i * 0.1, 1.0) for i in range(10)],
        [(0.0, 0.0)],
    ]


def _detector_factory(detector: Any) -> Callable[[], Any] | None:
    """The zero-arg factory :func:`percept.harness.detectors.register` needs, or ``None``.

    A class / zero-arg callable IS the factory (the registered form). Given an already-constructed
    instance we try its class; if that needs arguments we return ``None`` and the suite falls back to
    ``reset()`` for the fresh-state checks, saying so in the report.
    """
    if inspect.isclass(detector) or (callable(detector) and not hasattr(detector, "push")):
        return detector
    cls = type(detector)
    try:
        cls()
    except Exception:  # noqa: BLE001 — a ctor needing args is legal, just not a registrable factory
        return None
    return cls


def _boundary_key(b: Any) -> tuple:
    return (b.detector, b.kind, round(float(b.t), 9), round(float(b.value), 9), b.channel)


def check_detector(
    detector: Any,
    *,
    name: str | None = None,
    manifest: Any = None,
    samples: Iterable[tuple[float, Any]] | None = None,
    live_calls: bool = True,
) -> ConformanceReport:
    """Run the ADMISSION-DETECTOR conformance suite (the detector half of the extension-seam settlement).

    This is the seam a client plugs an OCR / phash / CLIP model into for screen content:
    :func:`percept.harness.detectors.register_detector` + a
    :class:`~percept.harness.detectors.DetectorManifest`. The suite verifies the manifest is
    well-formed, the call signature is the one :class:`~percept.harness.detectors.DetectorBank` uses,
    the output is a list of real :class:`~percept.harness.detectors.Boundary` objects attributed to
    this detector, the detector is DETERMINISTIC on a fixed input and ``reset()`` really restores
    initial state, and that the name is selectable end-to-end into a
    :class:`~percept.contracts.WatchSpec` through the real selection firewall.

    IT DOES NOT MEASURE YOUR MODEL. A green report means percept can call, route and count your
    detector — not that its boundaries are the right boundaries. That is ``percept-eval``'s question
    (:mod:`percept.eval.differential`), answered against YOUR fixtures and YOUR pre-registered bars.

    Args:
        detector: The registrable form — a class or zero-arg factory (preferred, this is what
            ``register_detector`` takes), or an already-constructed instance.
        name: Registry name to check selectability under (defaults to the manifest's / the
            detector's ``name`` attribute / the class name).
        manifest: An explicit :class:`~percept.harness.detectors.DetectorManifest`. When omitted, one
            is read from the registry if this name is already registered, else synthesized — and the
            descriptive checks then report what is missing.
        samples: ``(t, value)`` pairs to push. REQUIRED for a frame-backed detector
            (``signal == "frame"``): there is no universal probe frame, so those call checks SKIP
            rather than fabricate one. Defaults to a fixed deterministic scalar oscillation.
        live_calls: ``False`` runs only the static checks (manifest, construction, routing,
            selectability) and marks every push-based check ``SKIP``.
    """
    from .harness.detectors import (
        _REGISTRY,
        Boundary,
        DetectorManifest,
        DetectorBank,
        coerce_authored_detectors,
        make_detector,
        register,
        register_detector,
        registered,
        unregister,
    )

    rec = _Recorder()
    factory = _detector_factory(detector)
    probe_inst = None
    if factory is not None:
        try:
            probe_inst = factory()
        except Exception:  # noqa: BLE001 — reported by detector.factory_constructs below
            probe_inst = None
    elif not inspect.isclass(detector):
        probe_inst = detector

    reg_name = (
        name
        or getattr(manifest, "name", None)
        or getattr(detector, "name", None)
        or getattr(probe_inst, "name", None)
        or (detector.__name__ if inspect.isclass(detector) else type(detector).__name__)
    )
    reg_name = str(reg_name)
    display = name or reg_name
    if manifest is None and reg_name in _REGISTRY:
        manifest = _REGISTRY[reg_name][1]
    mf = manifest if isinstance(manifest, DetectorManifest) else DetectorManifest(
        name=reg_name, senses="", when_to_use="", output=""
    )

    # 1. the manifest — what the brain reads to COMPOSE this skill without hardcoding it -----------
    def _manifest_shape() -> None:
        if manifest is not None and not isinstance(manifest, DetectorManifest):
            raise AssertionError(
                f"manifest is {type(manifest).__name__}, expected "
                "percept.harness.detectors.DetectorManifest (import it — the registry stores the "
                "dataclass and the conductor reads its fields)"
            )
        if not isinstance(mf.name, str) or not mf.name.strip():
            raise AssertionError("DetectorManifest.name must be a non-empty str — it IS the registry key")
        for field_name in ("senses", "when_to_use", "output", "category"):
            v = getattr(mf, field_name)
            if not isinstance(v, str):
                raise AssertionError(
                    f"DetectorManifest.{field_name} must be a str (got {type(v).__name__})"
                )

    rec.check_sync("detector.manifest_well_formed", _manifest_shape)

    def _manifest_descriptive() -> None:
        empty = [f for f in ("senses", "when_to_use", "output") if not getattr(mf, f, "").strip()]
        if empty:
            raise AssertionError(
                f"DetectorManifest fields {empty} are empty. They are not decoration: the conductor "
                "reads `when_to_use` to decide whether to compose this skill for a goal, and a human "
                "debugging an under-admitting watch reads `output` to know what it should have emitted."
            )

    rec.check_sync("detector.manifest_is_descriptive", _manifest_descriptive, severity=ADVISORY)

    def _manifest_category() -> None:
        if not mf.category.strip():
            raise AssertionError(
                "DetectorManifest.category is empty, so this detector is invisible to "
                "detectors_in_category() — the modality-as-a-class discovery. Set one of the "
                "CATEGORY_* constants (e.g. CATEGORY_CHANGE for an OCR document-changed detector)."
            )

    rec.check_sync("detector.manifest_declares_category", _manifest_category, severity=ADVISORY)

    # 2. construction + the call shape DetectorBank uses -------------------------------------------
    def _constructs() -> None:
        if factory is None:
            raise AssertionError(
                f"no zero-arg factory: {type(detector).__name__}() needs arguments. register() takes "
                "a ZERO-ARG factory that builds a fresh stateful instance per goal — bind your "
                "configuration with functools.partial or a classmethod that takes no arguments."
            )
        inst = factory()
        if inst is None:
            raise AssertionError("the factory returned None")

    ok_ctor = rec.check_sync("detector.factory_constructs", _constructs)

    def _push_reset() -> None:
        inst = probe_inst if probe_inst is not None else detector
        for m in ("push", "reset"):
            if not callable(getattr(inst, m, None)):
                raise AssertionError(
                    f"missing detector method {m!r}. The seam is two methods — push(t, value) -> "
                    "list[Boundary] (streaming, called per sample) and reset() (stream "
                    "discontinuity / new source)."
                )
        try:
            sig = inspect.signature(inst.push)
        except (TypeError, ValueError) as exc:  # pragma: no cover — exotic callables
            raise AssertionError(f"push() has no introspectable signature: {exc}") from exc
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        var_pos = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if not var_pos and len(positional) < 2:
            raise AssertionError(
                f"push{sig} does not accept two positional arguments — DetectorBank calls "
                "`push(t, value)` where t is the stream timestamp and value the signal sample "
                "(or the raw frame for a frame-backed detector)."
            )
        if inspect.iscoroutinefunction(inst.push):
            raise AssertionError(
                "push() is a coroutine function — the tier-0 gate is SYNCHRONOUS and calls it inline "
                "per frame on the hot path; an unawaited coroutine silently emits no boundaries."
            )

    ok_push = ok_ctor and rec.check_sync("detector.push_and_reset_present", _push_reset)
    if not ok_ctor:
        rec.add("detector.push_and_reset_present", SKIP, detail="skipped: no instance could be built")

    channel = str(getattr(probe_inst if probe_inst is not None else detector, "signal", "motion"))

    def _routable() -> None:
        known = ("motion", "energy", "frame")
        if channel not in known:
            raise AssertionError(
                f"signal={channel!r} is not one of {known}. DetectorBank routes on this attribute and "
                "everything it does not recognise falls through to the MOTION channel — so a typo "
                "here does not raise, it silently feeds your detector the wrong signal forever."
            )

    rec.check_sync("detector.signal_channel_is_routable", _routable)

    # 3. selectability — register -> firewall -> WatchSpec, through the REAL code path -------------
    def _selectable() -> None:
        if factory is None:
            raise AssertionError("skipped by dependency: no zero-arg factory to register")
        from .contracts import Goal
        from .harness.tier0 import compile_watchspec

        prior = _REGISTRY.get(reg_name)
        try:
            got = register_detector(reg_name, factory, manifest=mf)
            if got.name != reg_name:
                raise AssertionError(
                    f"registered manifest name {got.name!r} != {reg_name!r} — the manifest's name is "
                    "authoritative, so the goal must select THAT name"
                )
            if reg_name not in registered():
                raise AssertionError(f"{reg_name!r} is not in registered() after register_detector")
            if not hasattr(make_detector(reg_name), "push"):
                raise AssertionError("make_detector() built something without push()")
            if coerce_authored_detectors([reg_name]) != [reg_name]:
                raise AssertionError(
                    f"the selection firewall dropped {reg_name!r} — a goal could never activate it"
                )
            goal = Goal(
                id="conformance",
                condition=PROBE_CONDITION,
                say="conformance probe",
                state={"type": "watch", "count": 0, "detectors": [reg_name]},
            )
            spec = compile_watchspec([goal], threshold=0.2)
            selected = spec.conditions[0].detectors if spec.conditions else []
            if reg_name not in selected:
                raise AssertionError(
                    f"compile_watchspec did not select {reg_name!r} for a goal that authored it "
                    f"(got {selected}) — the detector is registered but unreachable"
                )
            DetectorBank([reg_name])  # the per-goal construction the runtime performs
        finally:
            unregister(reg_name)
            if prior is not None:
                register(reg_name, prior[0], prior[1])

    if ok_ctor:
        rec.check_sync(
            "detector.selectable_by_name_into_watchspec",
            _selectable,
            on_pass=f"selectable as {reg_name!r}",
        )
    else:
        rec.add(
            "detector.selectable_by_name_into_watchspec",
            SKIP,
            detail="skipped: no instance could be built",
        )

    # 4. the push-based checks ---------------------------------------------------------------------
    def _skip_calls(detail: str) -> None:
        for check_name, severity in DETECTOR_CALL_CHECKS:
            rec.add(check_name, SKIP, severity=severity, detail=detail)

    def _report() -> ConformanceReport:
        return ConformanceReport(
            adapter=display, results=tuple(rec.results), role=ROLE_DETECTOR
        )

    if not ok_push:  # a bad manifest is REPORTED but never blocks the calls — they are independent
        _skip_calls("skipped: the detector is missing push()/reset() or could not be constructed")
        return _report()
    if not live_calls:
        _skip_calls("skipped: live_calls=False (static conformance only)")
        return _report()
    stream = list(samples) if samples is not None else None
    if stream is None and channel == "frame":
        _skip_calls(
            "skipped: this is a FRAME-backed detector (signal='frame') and no samples= were given. "
            "There is no universal probe frame — re-run with samples=[(t, frame), ...] from your "
            "own stream. A fabricated frame would prove nothing."
        )
        return _report()
    if stream is None:
        stream = _probe_samples()

    def _fresh() -> Any:
        if factory is not None:
            return factory()
        probe_inst.reset()
        return probe_inst

    def _run(inst: Any, pairs: list[tuple[float, Any]]) -> list[Any]:
        out: list[Any] = []
        for t, v in pairs:
            got = inst.push(t, v)
            if not isinstance(got, list):
                raise AssertionError(
                    f"push({t}, ...) returned {type(got).__name__}, expected a LIST of Boundary "
                    "(return [] when this sample is not a boundary — that is the common case)"
                )
            out.extend(got)
        return out

    first: list[Any] = []

    def _push_shape() -> None:
        got = _run(_fresh(), stream)
        for i, b in enumerate(got):
            if not isinstance(b, Boundary):
                raise AssertionError(
                    f"boundary[{i}] is {type(b).__name__}, expected "
                    "percept.harness.detectors.Boundary (import it — the brain reads the dataclass)"
                )
            if not isinstance(b.kind, str) or not b.kind:
                raise AssertionError(f"boundary[{i}].kind must be a non-empty str")
            for axis in ("t", "value"):
                v = getattr(b, axis)
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise AssertionError(f"boundary[{i}].{axis} must be a float (got {type(v).__name__})")
                if not math.isfinite(float(v)):
                    raise AssertionError(
                        f"boundary[{i}].{axis}={v} is not finite — a NaN/inf timestamp poisons the "
                        "accumulator that counts occurrences"
                    )
            if b.channel is not None and not isinstance(b.channel, str):
                raise AssertionError(f"boundary[{i}].channel must be a str or None")
        first.extend(got)

    ok_shape = rec.check_sync(
        "detector.push_returns_boundaries",
        _push_shape,
        on_pass=f"{len(stream)} samples pushed",
    )

    def _attributed() -> None:
        names = {b.detector for b in first}
        bad = sorted(n for n in names if not isinstance(n, str) or not n.strip())
        if bad:
            raise AssertionError(
                "Boundary.detector is empty or not a str. It is how the brain — and every trace, "
                "funnel and doctor panel — tells whose boundary this is; an unattributed boundary is "
                "unattributable at 2am."
            )
        if len(names) > 1:
            raise AssertionError(
                f"one detector emitted boundaries under several names {sorted(names)}. A detector "
                "that wraps another one must RELABEL to a single name of its own, as PoseOpenness "
                "does, or its counts split across two rows."
            )

    def _attribution_name() -> None:
        names = {b.detector for b in first}
        if names and names != {reg_name}:
            raise AssertionError(
                f"boundaries are attributed to {sorted(names)}, not {reg_name!r} — the name this "
                "detector is REGISTERED and selected under. Functionally harmless (the runtime maps "
                "boundaries by goal_id), but every trace row and doctor panel will name something "
                "the client's plan never mentions."
            )

    if ok_shape:
        rec.check_sync(
            "detector.boundaries_are_attributed",
            _attributed,
            on_pass=f"{len(first)} boundary(s) emitted",
        )
        rec.check_sync(
            "detector.attribution_matches_registry_name", _attribution_name, severity=ADVISORY
        )
    else:
        rec.add(
            "detector.boundaries_are_attributed",
            SKIP,
            detail="skipped: push() already failed its shape check",
        )
        rec.add(
            "detector.attribution_matches_registry_name",
            SKIP,
            severity=ADVISORY,
            detail="skipped: push() already failed its shape check",
        )

    if channel == "frame":
        # A frame-backed detector cannot be handed synthetic scalars; degenerate for it means a
        # FROZEN stream (the same real frame repeated — a paused screen share) and a stream of one.
        held = stream[0][1]
        degenerate = [[(i * 0.1, held) for i in range(5)], [(0.0, held)]]
        degenerate_detail = "a frozen (repeated-frame) stream and a single-sample stream survived"
    else:
        degenerate = _degenerate_samples()
        degenerate_detail = "flat-zero, pinned-high and single-sample streams survived"

    def _degenerate() -> None:
        for pairs in degenerate:
            _run(_fresh(), pairs)

    rec.check_sync(
        "detector.tolerates_degenerate_signal", _degenerate, on_pass=degenerate_detail
    )

    def _deterministic() -> None:
        a = [_boundary_key(b) for b in _run(_fresh(), stream)]
        b = [_boundary_key(x) for x in _run(_fresh(), stream)]
        if a != b:
            raise AssertionError(
                f"two runs over the SAME fixed input disagreed ({len(a)} vs {len(b)} boundaries; "
                f"first divergence at index {next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))}). "
                "An admission detector must be a pure function of the stream: a wall clock, an RNG "
                "or shared mutable state here makes every trace, replay and eval un-reproducible."
            )

    rec.check_sync("detector.deterministic_on_fixed_input", _deterministic)

    def _reset() -> None:
        inst = _fresh()
        a = [_boundary_key(b) for b in _run(inst, stream)]
        inst.reset()
        b = [_boundary_key(x) for x in _run(inst, stream)]
        if a != b:
            raise AssertionError(
                "reset() did not restore initial state — the same input after reset() produced "
                f"{len(b)} boundaries instead of {len(a)}. The runtime calls reset() on a stream "
                "discontinuity (new source, seek, reconnect); leftover state there fabricates or "
                "swallows boundaries at exactly the moment a client is least able to see it."
            )

    rec.check_sync("detector.reset_restores_initial_state", _reset)
    return _report()


# --- the tier-0 adapter seam (structural / semantic) ------------------------------------


def _probe_frames() -> list[Any]:
    """A small deterministic frame ladder. NOTE these are real ``percept.harness.tier0.Frame``s —
    exactly the TYPE the gate hands an adapter (``MotionGate`` coerces with ``_as_frame`` first), so
    the plumbing here is genuine even though the CONTENT is synthetic."""
    from .harness.tier0 import Frame, gray_frame

    checker = Frame(w=4, h=4, pixels=tuple(0 if (i + i // 4) % 2 else 255 for i in range(16)))
    return [gray_frame(4, 4, 0), checker, gray_frame(4, 4, 128)]


def _degenerate_frames() -> list[Any]:
    """The frames that break a naive adapter: a 1x1 frame (no neighbourhood to diff), and uniform
    black / uniform white (zero variance — the classic divide-by-range)."""
    from .harness.tier0 import gray_frame

    return [gray_frame(1, 1, 0), gray_frame(4, 4, 0), gray_frame(4, 4, 255)]


def check_tier0_adapter(
    adapter: Any,
    *,
    role: str = ROLE_STRUCTURAL,
    frames: Iterable[Any] | None = None,
    name: str | None = None,
    live_calls: bool = True,
) -> ConformanceReport:
    """Run the TIER-0 ADAPTER conformance suite for ``structural_adapter`` / ``semantic_adapter``.

    Both are exposed on :func:`percept.runtime.run`, so a client can now supplement tier 0 with
    a cheap local model (a CLIP drift score, a pose/privacy classifier, an OCR "the page changed"
    score) without bypassing the supported entrypoint. The adapter is a plain
    ``(frame) -> float in [0, 1]`` callable, invoked ONCE PER FRAME, SYNCHRONOUSLY, INSIDE the gate —
    so an adapter that raises takes the whole watch down and an adapter that returns an unnormalised
    score silently pins or mutes admission.

    This suite verifies exactly that: the score is a bounded normalized float, the adapter survives a
    degenerate frame, and a real :class:`~percept.harness.tier0.MotionGate` carrying it evaluates a
    frame stream without crashing and reports the adapter's value on the matching
    :class:`~percept.contracts.Tier0Signal` channel.

    IT DOES NOT MEASURE YOUR MODEL. A bounded score is not a GOOD score. Whether your adapter admits
    the right frames is ``percept-eval``'s paired A/B question, against your fixtures and your bars.

    Args:
        adapter: the ``(frame) -> float`` callable.
        role: ``"structural"`` or ``"semantic"`` — which ``run()`` parameter this fills.
        frames: real frames from your stream (strongly recommended). The gate coerces whatever the
            source yields with ``_as_frame`` before calling you, so an adapter always receives a
            :class:`~percept.harness.tier0.Frame`; when ``frames`` is omitted a synthetic ladder of
            those is used and the report says so.
        name: display name for the report.
        live_calls: ``False`` runs only the static checks and marks the calling checks ``SKIP``.
    """
    rec = _Recorder()
    if role not in TIER0_ROLES:
        raise ValueError(f"role must be one of {TIER0_ROLES}, got {role!r}")
    display = name or getattr(adapter, "__qualname__", None) or type(adapter).__name__
    probe = frames is None
    frame_list = list(frames) if frames is not None else _probe_frames()

    def _callable() -> None:
        if not callable(adapter):
            raise AssertionError(
                f"{type(adapter).__name__} is not callable — the seam is a plain function "
                "`(frame) -> float in [0, 1]`, not an object with a method"
            )
        try:
            sig = inspect.signature(adapter)
        except (TypeError, ValueError):  # pragma: no cover — builtins without signatures
            return
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        var_pos = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if not var_pos and len(required) != 1:
            raise AssertionError(
                f"{display}{sig} does not take exactly one required positional argument — the gate "
                "calls it as `adapter(frame)` and nothing else"
            )
        if inspect.iscoroutinefunction(adapter):
            raise AssertionError(
                "the adapter is a coroutine function — the gate is SYNCHRONOUS; an unawaited "
                "coroutine is truthy, so float() on it raises inside the gate and stops the watch"
            )

    ok_callable = rec.check_sync("adapter.is_callable", _callable)

    def _exposed() -> None:
        from .runtime import run as runtime_run

        param = f"{role}_adapter"
        if param not in inspect.signature(runtime_run).parameters:
            raise AssertionError(
                f"percept.runtime.run() has no {param!r} parameter in this install — the tier-0 "
                "extension seam is not reachable from the supported entrypoint here"
            )

    rec.check_sync("adapter.role_is_exposed_on_run", _exposed, on_pass=f"run({role}_adapter=...)")

    def _skip_calls(detail: str) -> None:
        for check_name, severity in TIER0_CALL_CHECKS:
            rec.add(check_name, SKIP, severity=severity, detail=detail)

    def _report() -> ConformanceReport:
        return ConformanceReport(
            adapter=display, results=tuple(rec.results), role=role, frame_was_probe=probe
        )

    if not ok_callable:
        _skip_calls("skipped: the adapter is not usable as `(frame) -> float`")
        return _report()
    if not live_calls:
        _skip_calls("skipped: live_calls=False (static conformance only)")
        return _report()

    def _assert_score(v: Any, *, where: str) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AssertionError(
                f"{where} returned {type(v).__name__}, expected a float — the gate does "
                "`float(adapter(frame))` and a non-number raises there, stopping the watch"
            )
        f = float(v)
        if not math.isfinite(f):
            raise AssertionError(
                f"{where} returned {v} — a NaN/inf score compares False against every threshold, so "
                "the channel silently never admits"
            )
        if not (0.0 <= f <= 1.0):
            raise AssertionError(
                f"{where} returned {f}, outside [0.0, 1.0]. The channel is compared against a "
                f"condition threshold that lives in [0,1], so an unnormalised score either pins "
                "admission on forever or mutes it forever — normalize before returning."
            )
        return f

    def _bounded() -> None:
        for i, f in enumerate(frame_list):
            _assert_score(adapter(f), where=f"adapter(frames[{i}])")

    rec.check_sync(
        "adapter.returns_bounded_score",
        _bounded,
        on_pass=f"{len(frame_list)} frame(s) scored in [0,1]",
    )

    def _degenerate() -> None:
        for f in _degenerate_frames():
            _assert_score(adapter(f), where=f"adapter(degenerate {f.w}x{f.h} uniform frame)")

    rec.check_sync(
        "adapter.tolerates_degenerate_frame",
        _degenerate,
        on_pass="1x1, uniform-black and uniform-white frames survived",
    )

    # The gate diffs consecutive frames, so a stream may not change dimensions mid-run (motion_score
    # raises on a size mismatch — a percept invariant, nothing to do with the adapter). Group the
    # probe + degenerate frames by size and drive a FRESH gate over each group.
    gate_groups: dict[tuple[int, int], list[Any]] = {}
    for f in frame_list + _degenerate_frames():
        gate_groups.setdefault((f.w, f.h), []).append(f)
    n_gate_frames = sum(len(g) for g in gate_groups.values())

    def _build_gate() -> Any:
        from .contracts import WatchCondition, WatchSpec
        from .harness.tier0 import MotionGate

        spec = WatchSpec(
            conditions=[
                WatchCondition(
                    goal_id="conformance",
                    anchor_text=PROBE_CONDITION,
                    threshold=0.5,
                    gate_on=role,  # type: ignore[arg-type]
                )
            ]
        )
        kwargs = {f"{role}_adapter": adapter}
        return MotionGate(spec, **kwargs)

    def _gate_survives() -> None:
        for group in gate_groups.values():
            gate = _build_gate()
            for i, f in enumerate(group):
                gate.evaluate(f, ts=i * 0.1)

    ok_gate = rec.check_sync(
        "adapter.does_not_crash_the_gate",
        _gate_survives,
        on_pass=f"{n_gate_frames} frames through a real MotionGate gated on {role!r}",
    )

    def _reaches_signal() -> None:
        for i, f in enumerate(frame_list):
            expected = float(adapter(f))
            # a fresh gate per frame: the tier-0 channels are computed from THIS frame alone, so the
            # prev-frame cursor is irrelevant here and a fresh gate keeps mixed sizes legal.
            got = float(getattr(_build_gate().evaluate(f, ts=i * 0.1).signal, role))
            if got != expected:
                raise AssertionError(
                    f"Tier0Signal.{role} is {got} on frame {i} but the adapter returns {expected} — "
                    "the value a client reads in the trace would not be the value that decided "
                    "admission"
                )

    if ok_gate:
        rec.check_sync("adapter.score_reaches_the_tier0_signal", _reaches_signal)
    else:
        rec.add(
            "adapter.score_reaches_the_tier0_signal",
            SKIP,
            detail="skipped: the gate already failed",
        )

    def _deterministic() -> None:
        for i, f in enumerate(frame_list):
            a, b = float(adapter(f)), float(adapter(f))
            if a != b:
                raise AssertionError(
                    f"adapter(frames[{i}]) returned {a} then {b} for the SAME frame. A stochastic "
                    "admission score makes a run unreproducible: the trace cannot explain why a "
                    "frame was admitted and percept eval cannot A/B it. ADVISORY because a genuinely "
                    "stateful adapter (a running drift baseline) is legitimate — but say so."
                )

    rec.check_sync("adapter.is_deterministic", _deterministic, severity=ADVISORY)
    return _report()


# --- CLI ------------------------------------------------------------------------------------------


def load_target(target: str, *, instantiate: bool = True) -> Any:
    """Resolve ``"pkg.module:Attr"`` to the object under test.

    ``instantiate`` (default ``True``, the Vision behaviour) calls a class / zero-arg factory to get
    an adapter INSTANCE. Pass ``False`` for the roles whose registrable form IS the callable — a
    detector factory (``register_detector`` takes the factory, not an instance) and a tier-0
    ``(frame) -> float`` adapter (calling it would run the model on nothing).
    """
    if ":" not in target:
        raise ValueError(
            f"bad target {target!r} — expected 'package.module:Attr' (a Vision class, a zero-arg "
            "factory, or an already-constructed instance)"
        )
    mod_name, _, attr = target.partition(":")
    obj = import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if instantiate and (inspect.isclass(obj) or (callable(obj) and not hasattr(obj, "judge"))):
        obj = obj()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m percept.conformance",
        description=(
            "Run the published extension-seam conformance suite against a third-party plug-in — a "
            "Vision adapter, an admission detector, or a tier-0 structural/semantic adapter. Exits "
            "0 when conformant, 1 otherwise. This proves WELL-FORMEDNESS, never model quality: use "
            "percept-eval for that."
        ),
    )
    parser.add_argument("target", help="the plug-in to test, as 'package.module:ClassOrFactory'")
    parser.add_argument(
        "--role",
        default=ROLE_VISION,
        choices=sorted(ROLE_TITLES),
        help=(
            "which extension seam TARGET plugs into: 'vision' (default, the cloud judge), 'detector' "
            "(register_detector — a factory), or 'structural'/'semantic' (a run() tier-0 adapter)"
        ),
    )
    parser.add_argument(
        "--no-injection",
        action="store_true",
        help="vision only: skip the fail-closed injection checks (they construct a real Percept core)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="run only the static checks (shape, signature, registration) — no calls into the plug-in",
    )
    parser.add_argument("--condition", default=PROBE_CONDITION, help="plain-language condition to judge")
    parser.add_argument(
        "--name",
        default=None,
        help="detector only: the registry name to check selectability under (default: the manifest's)",
    )
    args = parser.parse_args(argv)

    try:
        obj = load_target(args.target, instantiate=args.role == ROLE_VISION)
    except Exception as exc:  # noqa: BLE001 — the CLI reports, never tracebacks at the user
        sys.stderr.write(f"could not load {args.target!r}: {type(exc).__name__}: {exc}\n")
        return 2

    if args.role == ROLE_DETECTOR:
        report = check_detector(obj, name=args.name, live_calls=not args.static)
    elif args.role in TIER0_ROLES:
        report = check_tier0_adapter(
            obj, role=args.role, name=args.target, live_calls=not args.static
        )
    else:
        report = check_vision_adapter(
            obj,
            condition=args.condition,
            check_injection=not (args.no_injection or args.static),
            live_calls=not args.static,
            name=args.target,
        )
    sys.stdout.write(report.format() + "\n")
    return 0 if report.ok else 1


__all__ = [
    "CONFORMANCE_VERSION",
    "REQUIRED",
    "ADVISORY",
    "PASS",
    "FAIL",
    "SKIP",
    "JUDGE_OPTIONAL_KWARGS",
    "ESCALATION_SEVERITY",
    "CALL_CHECKS",
    "INJECTION_CHECKS",
    "DETECTOR_STATIC_CHECKS",
    "DETECTOR_CALL_CHECKS",
    "TIER0_STATIC_CHECKS",
    "TIER0_CALL_CHECKS",
    "ROLE_VISION",
    "ROLE_DETECTOR",
    "ROLE_STRUCTURAL",
    "ROLE_SEMANTIC",
    "ROLE_TITLES",
    "TIER0_ROLES",
    "SCOPE_DISCLAIMER",
    "PROBE_FRAME",
    "PROBE_CONDITION",
    "CheckResult",
    "ConformanceReport",
    "assert_verdict",
    "assert_observations",
    "acheck_vision_adapter",
    "check_vision_adapter",
    "check_detector",
    "check_tier0_adapter",
    "load_target",
    "main",
]


if __name__ == "__main__":  # pragma: no cover — exercised via main() in tests
    raise SystemExit(main())
