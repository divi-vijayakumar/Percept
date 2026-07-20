"""The PUBLISHED Vision-adapter conformance suite — the runnable definition of "my adapter works".

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
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Iterable

from .contracts import CONTRACT_VERSION, Box, Goal, Observation, Verdict, Vision

#: Semver of the CONFORMANCE SUITE itself — the set of checks and their severities — distinct from
#: ``percept.contracts.CONTRACT_VERSION`` (the DTO shapes) and ``percept.watch.WATCH_VERSION`` (the
#: entrypoint). Bump MAJOR when an existing check gets stricter or a REQUIRED check is added (an
#: adapter that passed may now fail); MINOR for a new ADVISORY check; PATCH for wording/fixes.
CONFORMANCE_VERSION = "1.0.0"

REQUIRED = "required"
ADVISORY = "advisory"

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

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

#: The frame handed to the adapter when the caller supplies none. A frame is an OPAQUE per-backend
#: type, so there is no universal probe: this sentinel only exercises the *plumbing*, and a real
#: adapter should be handed a real frame via ``frame=``.
PROBE_FRAME = "percept-conformance-probe"
PROBE_CONDITION = "a person is visible"


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
    """The result of running the suite against one adapter."""

    adapter: str
    results: tuple[CheckResult, ...] = ()
    contract_version: str = CONTRACT_VERSION
    conformance_version: str = CONFORMANCE_VERSION
    frame_was_probe: bool = False

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
        lines = [
            f"percept Vision conformance — {self.adapter}",
            f"  suite {self.conformance_version} · contract {self.contract_version}",
        ]
        if self.frame_was_probe:
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


# --- CLI ------------------------------------------------------------------------------------------


def load_target(target: str) -> Any:
    """Resolve ``"pkg.module:Attr"`` to an adapter INSTANCE (calling ``Attr`` if it is a factory)."""
    if ":" not in target:
        raise ValueError(
            f"bad target {target!r} — expected 'package.module:Attr' (a Vision class, a zero-arg "
            "factory, or an already-constructed instance)"
        )
    mod_name, _, attr = target.partition(":")
    obj = import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if inspect.isclass(obj) or (callable(obj) and not hasattr(obj, "judge")):
        obj = obj()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m percept.conformance",
        description=(
            "Run the published Vision-adapter conformance suite against a third-party adapter. "
            "Exits 0 when conformant, 1 otherwise."
        ),
    )
    parser.add_argument("target", help="adapter to test, as 'package.module:ClassOrFactory'")
    parser.add_argument(
        "--no-injection",
        action="store_true",
        help="skip the fail-closed injection checks (they construct a real Percept core)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="run only the static checks (method set, protocol, signature) — no calls into the adapter",
    )
    parser.add_argument("--condition", default=PROBE_CONDITION, help="plain-language condition to judge")
    args = parser.parse_args(argv)

    try:
        adapter = load_target(args.target)
    except Exception as exc:  # noqa: BLE001 — the CLI reports, never tracebacks at the user
        sys.stderr.write(f"could not load {args.target!r}: {type(exc).__name__}: {exc}\n")
        return 2
    report = check_vision_adapter(
        adapter,
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
    "PROBE_FRAME",
    "PROBE_CONDITION",
    "CheckResult",
    "ConformanceReport",
    "assert_verdict",
    "assert_observations",
    "acheck_vision_adapter",
    "check_vision_adapter",
    "load_target",
    "main",
]


if __name__ == "__main__":  # pragma: no cover — exercised via main() in tests
    raise SystemExit(main())
