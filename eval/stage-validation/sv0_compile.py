"""SV-0.7 .. SV-0.10 — Stage-0 PLANNER/COMPILER validation for the Percept SDK.

Exercises the real chain ``WatchPlan -> ConfigResolver -> ResolvedWatch -> RuntimeCompiler ->
runtime`` (plus the public ``apply_watch_plan`` arm boundary) entirely OFFLINE — no provider call,
no network, no frame decode.

Why this suite exists: a mis-compiled Watch is indistinguishable from a bad judge in every
end-to-end metric the project publishes. If a "count how many X" goal silently compiles to a
boolean presence Watch, every downstream precision/recall number is measuring the wrong thing.

Tests
-----
SV-0.7   Compile fidelity over the 100-case corpus. Does ``goal_text`` compile to the primitive
         its ``goal_kind`` implies? Measured on two paths:
           A. DEFAULT  — ``WatchPlan(condition=goal_text)``, what a client actually gets.
           B. AUTHORED — the caller hand-authors the ``Goal.state.*`` explicit paths for that kind;
              measures whether the primitive is even REACHABLE and EXECUTABLE.
SV-0.8   Compile determinism (same process, fresh subprocess, different cwd) + hash coverage:
         mutate every ``Goal.state.*`` registry field in turn and record which of
         ``plan_hash`` / ``component_snapshot_hash`` / ``effective`` / compiled state moves.
         Also probes the reverse: ``Goal.state.*`` written by a layer OTHER than the plan.
SV-0.9   Effective-config inspectability: every value in the COMPILED artifact must be
         attributable to a resolution layer via ``ResolvedWatch.provenance``.
SV-0.10  Unsupported-goal honesty: out-of-1.x-envelope goals (person identity, distinct-person
         counting, intent inference) must fail closed with a typed reason, never silently
         degrade to a weaker Watch.

Run
---
    /private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv0_compile.py

Emits ``eval/stage-validation/results/sv0_compile.json`` and prints a PASS/FAIL table.
Importable: ``from sv0_compile import run_all; report = run_all()``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

# --------------------------------------------------------------------------------------------
# Runtime under test: the launch wheel in site-packages. NEVER the repo's packages/percept-vision.
# --------------------------------------------------------------------------------------------
import percept  # noqa: E402
from percept.behavior_routes import BehaviorRoute, behavior_route  # noqa: E402
from percept.config import Settings  # noqa: E402
from percept.contracts.watchplan import WatchPlan  # noqa: E402
from percept.resolver import PRECEDENCE, ConfigError, ConfigResolver  # noqa: E402
from percept.runtime_compiler import runtime_compile  # noqa: E402
from percept.sourceprofile import UNTUNED, SourceProfile  # noqa: E402
from percept.tracing import content_hash  # noqa: E402
from percept.tunables import REGISTRY  # noqa: E402
from percept.watchplan_apply import apply_watch_plan  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.json"
RESULTS = HERE / "results"
JSON_OUT = RESULTS / "sv0_compile.json"

#: The tunables-registry ``Goal.state.*`` paths — the full authored-state surface.
STATE_SPECS = tuple(s for s in REGISTRY if s.path.startswith("Goal.state."))

#: Layer names a ``SettingOrigin.owner_layer`` may legally carry (ADR-008 precedence tiers + Gate).
LEGAL_OWNER_LAYERS = frozenset(PRECEDENCE) | {"Gate"}


# ============================================================================================
# Shared helpers
# ============================================================================================


def _fresh_core() -> Any:
    """An offline ``Percept`` core (scripted backends, no key, no network).

    ``mode="offline"`` is the runtime's own documented keyless path; the banner it prints is
    swallowed so the PASS/FAIL table stays readable.
    """
    from percept.agent import Percept

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return Percept.create(settings=Settings(), mode="offline")


def compile_plan(
    plan: WatchPlan,
    *,
    source_profile: Any = None,
    settings: Settings | None = None,
) -> tuple[Any, Any]:
    """``(ResolvedWatch, CompiledWatch)`` for ``plan``. Pure — no I/O, no arm.

    ``settings=Settings()`` is passed by default so the corpus sweep is independent of whatever
    ``PERCEPT_*`` env the operator happens to have set (env sensitivity is probed separately in
    SV-0.8).
    """
    resolved = ConfigResolver().resolve(
        plan,
        source_profile=source_profile,
        settings=settings if settings is not None else Settings(),
    )
    return resolved, runtime_compile(resolved)


def _has_accumulate(state: Mapping[str, Any]) -> bool:
    """``fire_when`` present in EITHER the raw compiler shape or the coerced armed shape.

    ``runtime_compile`` leaves it at ``state["behavior"]["accumulate"]["fire_when"]``;
    ``apply_watch_plan``'s coercion firewall flattens it to ``state["fire_when"]``.
    """
    if "fire_when" in state:
        return True
    behavior = state.get("behavior")
    if isinstance(behavior, Mapping):
        accumulate = behavior.get("accumulate")
        if isinstance(accumulate, Mapping) and "fire_when" in accumulate:
            return True
    return False


def primitive_of(state: Mapping[str, Any]) -> str:
    """Which runtime primitive a ``Goal.state`` selects, in the corpus's own vocabulary.

    Reads the CLOSED route table (``percept.behavior_routes``) plus the two frame-Watch
    refinements the runtime actually executes (``hold_s`` -> sustained duration,
    ``fire_when`` -> accumulate/nth-distinct counting). Nothing is inferred from text.
    Accepts both the raw ``runtime_compile`` shape and the coerced armed shape.
    """
    route = behavior_route(state)
    if route is BehaviorRoute.FRAME_TRANSITION:
        return "transition"
    if route is BehaviorRoute.EXTERNAL_BOUNDARY:
        return "count"
    if route is BehaviorRoute.EXTERNAL_RHYTHM:
        return "rhythm"
    if route is BehaviorRoute.FRAME_PROCESS:
        return "process"
    if route in (BehaviorRoute.EXTERNAL_EVENT_WINDOW, BehaviorRoute.EXTERNAL_SCHEDULER):
        return route.value
    if route is BehaviorRoute.FRAME_WATCH:
        if _has_accumulate(state):
            return "count"  # accumulate / nth-distinct — the executable counting shape
        if "hold_s" in state:
            return "duration"
        return "state"  # a plain boolean presence Watch
    return "unsupported"


#: What a caller must AUTHOR, per goal kind, to reach that kind's primitive. Read straight off the
#: tunables registry vocabulary (Section C) — nothing invented here.
AUTHORED_FOR_KIND: dict[str, dict[str, Any]] = {
    "duration": {
        "Goal.state.hold_s": 3.0,
        "Goal.state.cadence.min_judge_interval": 1.0,
        "Goal.state.cadence.gate_on": "fixed",
    },
    "transition": {
        "Goal.state.type": "transition",
        "Goal.state.from": "the condition does not hold",
        "Goal.state.to": "the condition holds",
    },
    "count": {"Goal.state.type": "count"},
    "state": {},
}

#: The fallback counting shape, for when the NAMED ``type="count"`` primitive turns out not to be
#: executable by the frame worker.
COUNT_ACCUMULATE = {"Goal.state.fire_when": "n >= 2", "Goal.state.window_s": 30.0}


def _apply_verdict(core: Any, plan: WatchPlan) -> dict[str, Any]:
    """Drive the PUBLIC arm boundary and reduce the receipt to a comparable verdict.

    Also reads back the CANONICAL armed ``Goal.state`` (post-coercion) from the registry — that,
    not the raw ``runtime_compile`` output, is what the frame worker actually executes.
    """
    receipt = apply_watch_plan(core, plan)
    armed_state: dict[str, Any] | None = None
    if receipt.watch_ids:
        goal = core.goals.get(receipt.watch_ids[0])
        if goal is not None:
            armed_state = dict(goal.state)
    return {
        "status": receipt.status,
        "armed": list(receipt.watch_ids),
        "armed_state": armed_state,
        "dropped": [(d.path, d.code) for d in receipt.dropped_fields],
        "coerced": [(d.path, d.code) for d in receipt.coerced_fields],
        "warnings": [(d.path, d.code) for d in receipt.warnings],
    }


# ============================================================================================
# SV-0.7 — Compile fidelity corpus
# ============================================================================================


def sv_0_7(cases: list[dict]) -> dict[str, Any]:
    """Compile all 100 corpus ``goal_text`` values; assert the primitive matches ``goal_kind``."""
    per_case: list[dict[str, Any]] = []
    default_core = _fresh_core()
    authored_core = _fresh_core()

    # A single compiled fingerprint per case, to answer "does ANY compiled output vary with kind?"
    fingerprints: dict[str, set[str]] = {}

    for i, case in enumerate(cases):
        kind = case["goal_kind"]
        text = case["goal_text"]

        # -- Path A: DEFAULT. What a client typing this condition actually gets. ---------------
        plan_a = WatchPlan(condition=text, id=f"sv07a_{i}")
        try:
            res_a, comp_a = compile_plan(plan_a)
            state_a = dict(comp_a.goal.state)
            prim_a = primitive_of(state_a)
            err_a = None
        except Exception as exc:  # noqa: BLE001 — a compile crash is itself the finding
            state_a, prim_a, err_a = {}, "compile_error", f"{type(exc).__name__}: {exc}"
            res_a = comp_a = None
        apply_a = _apply_verdict(default_core, plan_a) if err_a is None else None
        if apply_a and apply_a["armed_state"] is not None:
            # The ARMED state is authoritative — it is what the frame worker executes.
            prim_a = primitive_of(apply_a["armed_state"])

        # Fingerprint every knob the compiler produced, minus per-goal identity.
        if comp_a is not None:
            fp = content_hash(
                {
                    "settings": {
                        sec: dict(vars(val)) for sec, val in vars(comp_a.settings).items()
                    },
                    "state": state_a,
                    "watchspec": dict(comp_a.watchspec_kwargs),
                    "plan_mode": comp_a.plan_mode,
                    "timebase": comp_a.timebase,
                    "coverage_floor": comp_a.coverage_floor,
                    "sense_policy": repr(comp_a.sense_policy),
                }
            )
            fingerprints.setdefault(kind, set()).add(fp)

        # -- Path B: AUTHORED. Is the kind's primitive reachable + executable at all? ----------
        authored = dict(AUTHORED_FOR_KIND[kind])
        plan_b = WatchPlan(condition=text, id=f"sv07b_{i}", explicit=authored)
        try:
            res_b, comp_b = compile_plan(plan_b)
            state_b = dict(comp_b.goal.state)
            prim_b = primitive_of(state_b)
            err_b = None
        except Exception as exc:  # noqa: BLE001
            state_b, prim_b, err_b = {}, "compile_error", f"{type(exc).__name__}: {exc}"
        apply_b = _apply_verdict(authored_core, plan_b) if err_b is None else None
        if apply_b and apply_b["armed_state"] is not None:
            prim_b = primitive_of(apply_b["armed_state"])

        # Fallback counting shape when the named ``count`` primitive did not arm.
        apply_c = None
        prim_c = None
        if kind == "count" and (apply_b is None or apply_b["status"] == "rejected"):
            plan_c = WatchPlan(
                condition=text, id=f"sv07c_{i}", explicit=dict(COUNT_ACCUMULATE)
            )
            _, comp_c = compile_plan(plan_c)
            prim_c = primitive_of(dict(comp_c.goal.state))
            apply_c = _apply_verdict(authored_core, plan_c)
            if apply_c["armed_state"] is not None:
                prim_c = primitive_of(apply_c["armed_state"])

        per_case.append(
            {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "goal_kind": kind,
                "goal_text": text,
                "default_primitive": prim_a,
                "default_state": state_a,
                "default_apply": apply_a,
                "default_match": prim_a == kind,
                "default_error": err_a,
                "authored_explicit": authored,
                "authored_primitive": prim_b,
                "authored_state": state_b,
                "authored_apply": apply_b,
                "authored_match": prim_b == kind,
                "authored_armed": bool(apply_b and apply_b["status"] != "rejected"),
                "authored_error": err_b,
                "fallback_primitive": prim_c,
                "fallback_apply": apply_c,
                "fallback_match": prim_c == kind,
                "fallback_armed": bool(apply_c and apply_c["status"] != "rejected"),
            }
        )

    # -- Per-kind breakdown -----------------------------------------------------------------
    kinds = sorted({c["goal_kind"] for c in cases})
    breakdown: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        rows = [r for r in per_case if r["goal_kind"] == kind]
        breakdown[kind] = {
            "n": len(rows),
            "default_match": sum(r["default_match"] for r in rows),
            "default_primitives": _tally(r["default_primitive"] for r in rows),
            "authored_match": sum(r["authored_match"] for r in rows),
            "authored_armed": sum(r["authored_armed"] for r in rows),
            "authored_reject_codes": _tally(
                code
                for r in rows
                if r["authored_apply"]
                for _p, code in r["authored_apply"]["dropped"]
            ),
            "fallback_armed": sum(r["fallback_armed"] for r in rows),
            "fallback_match_and_armed": sum(
                r["fallback_match"] and r["fallback_armed"] for r in rows
            ),
            "fallback_primitives": _tally(
                r["fallback_primitive"] for r in rows if r["fallback_primitive"]
            ),
            # The bottom line a client cares about: is there ANY way to arm this kind's primitive?
            "reachable_and_executable": sum(
                (r["authored_match"] and r["authored_armed"])
                or (r["fallback_match"] and r["fallback_armed"])
                for r in rows
            ),
        }

    # -- Window / refractory sanity on the compiled artifact ---------------------------------
    _, base_comp = compile_plan(WatchPlan(condition="a probe condition"))
    gate = base_comp.settings.gate
    events = base_comp.settings.events
    window_checks = {
        "gate.refractory_s": gate.refractory_s,
        "gate.validity_window_s": gate.validity_window_s,
        "gate.key_ttl_s": gate.key_ttl_s,
        "events.window": events.window,
        "events.merge_gap_s": events.merge_gap_s,
        "episodes.window_s": base_comp.settings.episodes.window_s,
    }
    window_sane = (
        isinstance(gate.refractory_s, (int, float))
        and gate.refractory_s >= 0
        and gate.validity_window_s > 0
        and events.window >= 1
        and events.merge_gap_s >= 0
    )
    # A duration Watch must observe at least once inside its own hold window.
    dur_state = next(
        (r["authored_state"] for r in per_case if r["goal_kind"] == "duration"), {}
    )
    dur_cadence = dur_state.get("cadence") or {}
    duration_window_sane = bool(
        dur_state.get("hold_s", 0) > 0
        and dur_cadence.get("min_judge_interval", 0) > 0
        and dur_state["hold_s"] >= dur_cadence["min_judge_interval"]
        and dur_cadence.get("gate_on") == "fixed"
    )

    # -- Does the compiled artifact vary with goal_kind at all on the DEFAULT path? ----------
    all_fps = set().union(*fingerprints.values()) if fingerprints else set()
    kind_sensitive = len(all_fps) > 1

    total_default = sum(r["default_match"] for r in per_case)
    total_authored = sum(r["authored_match"] for r in per_case)
    total_armed = sum(r["authored_armed"] for r in per_case)
    total_reachable = sum(v["reachable_and_executable"] for v in breakdown.values())

    failures: list[str] = []
    if total_default != len(per_case):
        failures.append(
            f"DEFAULT compile fidelity {total_default}/{len(per_case)}: "
            f"{len(per_case) - total_default} goals compiled to a primitive their goal_kind "
            f"does not imply (percept/goalsugar.py:3-11,66-90 + percept/runtime_compiler.py:310 — the "
            f"condition text is never read for meaning, so every plan compiles to type='watch')"
        )
    if not kind_sensitive:
        failures.append(
            "DEFAULT compile output is BYTE-IDENTICAL across all four goal kinds "
            "(one compiled fingerprint for 100 cases) — nothing in the compile chain "
            "distinguishes a duration goal from a count goal"
        )
    if total_authored != len(per_case):
        failures.append(
            f"AUTHORED compile fidelity {total_authored}/{len(per_case)}: even with the "
            "kind's own Goal.state.* paths authored explicitly, the compiled primitive "
            "does not match"
        )
    if total_armed != len(per_case):
        failures.append(
            f"AUTHORED arm reachability {total_armed}/{len(per_case)}: the primitive a "
            "goal_kind implies cannot be armed on percept.run "
            "(percept/behavior_routes.py:24-34 routes type='count' to EXTERNAL_BOUNDARY, which "
            "worker_can_drive() rejects; percept/watchplan_apply.py:221-241 refuses the arm)"
        )
    if total_reachable != len(per_case):
        failures.append(
            f"NO executable primitive exists for {len(per_case) - total_reachable} case(s) "
            "even counting the accumulate/nth-distinct fallback"
        )
    if not window_sane:
        failures.append(f"gate/event window semantics are not sane: {window_checks}")
    if not duration_window_sane:
        failures.append(
            f"duration Watch window semantics unsound: state={dur_state}"
        )

    return {
        "id": "SV-0.7",
        "name": "Compile fidelity corpus",
        "passed": not failures,
        "failures": failures,
        "n_cases": len(per_case),
        "default_fidelity": f"{total_default}/{len(per_case)}",
        "authored_fidelity": f"{total_authored}/{len(per_case)}",
        "authored_armed": f"{total_armed}/{len(per_case)}",
        "reachable_and_executable": f"{total_reachable}/{len(per_case)}",
        "kind_sensitive_compile": kind_sensitive,
        "distinct_compiled_fingerprints": len(all_fps),
        "breakdown": breakdown,
        "window_checks": window_checks,
        "window_sane": window_sane,
        "duration_window_sane": duration_window_sane,
        "per_case": per_case,
    }


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ============================================================================================
# SV-0.8 — Compile determinism + hash coverage
# ============================================================================================

#: A fixed, non-trivial plan set the determinism probe compiles identically everywhere.
DETERMINISM_PLANS: tuple[dict[str, Any], ...] = (
    {"condition": "the milk is boiling over"},
    {
        "condition": "the door goes from closed to open",
        "explicit": {
            "Goal.state.type": "transition",
            "Goal.state.from": "door closed",
            "Goal.state.to": "door open",
        },
    },
    {
        "condition": "someone stays seated for a while",
        "explicit": {
            "Goal.state.hold_s": 4.0,
            "Goal.state.cadence.min_judge_interval": 1.0,
            "Goal.state.cadence.gate_on": "fixed",
        },
        "target": "person",
        "rearm": "never",
        "consequence": {"miss_cost": "severe", "false_fire_cost": "mild"},
    },
    {
        "condition": "three pallets have gone past",
        "explicit": {"Goal.state.fire_when": "n >= 3", "Goal.state.window_s": 45.0},
        "detectors": ("motion",),
    },
)

#: Executed in a FRESH interpreter. Prints one JSON line of compile digests.
_SUBPROCESS_SRC = r"""
import json, sys
from percept.config import Settings
from percept.contracts.watchplan import WatchPlan
from percept.resolver import ConfigResolver
from percept.runtime_compiler import runtime_compile
from percept.tracing import content_hash

specs = json.loads(sys.argv[1])
out = []
R = ConfigResolver()
for spec in specs:
    spec = dict(spec)
    spec["detectors"] = tuple(spec.get("detectors", ()))
    plan = WatchPlan(**spec)
    r = R.resolve(plan, settings=Settings())
    c = runtime_compile(r)
    out.append({
        "plan_hash": r.plan_hash,
        "component_snapshot_hash": r.component_snapshot_hash,
        "effective_hash": content_hash(dict(sorted(r.effective.items()))),
        "provenance_hash": content_hash(
            {k: v.to_dict() for k, v in sorted(r.provenance.items())}
        ),
        "goal_id": c.goal.id,
        "state_hash": content_hash(c.goal.state),
        "settings_hash": content_hash(
            {s: dict(vars(v)) for s, v in sorted(vars(c.settings).items())}
        ),
        "watchspec_hash": content_hash(dict(c.watchspec_kwargs)),
    })
print(json.dumps(out))
"""


def _compile_digests(specs) -> list[dict[str, str]]:
    """In-process twin of ``_SUBPROCESS_SRC``."""
    out = []
    resolver = ConfigResolver()
    for spec in specs:
        spec = dict(spec)
        spec["detectors"] = tuple(spec.get("detectors", ()))
        plan = WatchPlan(**spec)
        r = resolver.resolve(plan, settings=Settings())
        c = runtime_compile(r)
        out.append(
            {
                "plan_hash": r.plan_hash,
                "component_snapshot_hash": r.component_snapshot_hash,
                "effective_hash": content_hash(dict(sorted(r.effective.items()))),
                "provenance_hash": content_hash(
                    {k: v.to_dict() for k, v in sorted(r.provenance.items())}
                ),
                "goal_id": c.goal.id,
                "state_hash": content_hash(c.goal.state),
                "settings_hash": content_hash(
                    {s: dict(vars(v)) for s, v in sorted(vars(c.settings).items())}
                ),
                "watchspec_hash": content_hash(dict(c.watchspec_kwargs)),
            }
        )
    return out


def _run_subprocess(cwd: str, hashseed: str) -> list[dict[str, str]]:
    specs = json.dumps(
        [{**s, "detectors": list(s.get("detectors", ()))} for s in DETERMINISM_PLANS]
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PERCEPT_")}
    env["PYTHONHASHSEED"] = hashseed
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SRC, specs],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess compile failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _sample_for(spec) -> Any:
    """A legal-but-different probe value for one ``Goal.state.*`` tunable spec."""
    if spec.choices:
        for choice in spec.choices:
            if choice != spec.default:
                return choice
        return spec.choices[0]
    vt = spec.value_type
    if vt == "bool":
        return True
    if vt == "int":
        return 3
    if vt == "float":
        lo, hi = spec.bounds if spec.bounds else (0.0, 10.0)
        return (lo + 2.0) if hi == float("inf") else (lo + hi) / 2.0
    if vt == "list":
        return ["done"]
    if spec.path.endswith("fire_when"):
        return "n >= 4"
    if spec.path.endswith("episode.policy"):
        return "document_presence"
    return "sv08_probe"


def sv_0_8() -> dict[str, Any]:
    """Determinism across processes and working directories + ``state.*`` hash coverage."""
    failures: list[str] = []

    # -- determinism ------------------------------------------------------------------------
    in_proc_1 = _compile_digests(DETERMINISM_PLANS)
    in_proc_2 = _compile_digests(DETERMINISM_PLANS)
    same_process = in_proc_1 == in_proc_2
    if not same_process:
        failures.append("compile is not deterministic within one process")

    determinism_runs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        for label, cwd, seed in (
            ("subprocess_cwd_repo", str(HERE), "0"),
            ("subprocess_cwd_tmp_a", tmp_a, "12345"),
            ("subprocess_cwd_tmp_b", tmp_b, "999"),
            ("subprocess_cwd_root", os.sep, "7"),
        ):
            try:
                determinism_runs[label] = _run_subprocess(cwd, seed)
            except Exception as exc:  # noqa: BLE001
                determinism_runs[label] = {"error": f"{type(exc).__name__}: {exc}"}

    cross_process_ok = True
    cross_process_diffs: list[dict[str, Any]] = []
    for label, digests in determinism_runs.items():
        if isinstance(digests, dict) and "error" in digests:
            cross_process_ok = False
            cross_process_diffs.append({"run": label, "error": digests["error"]})
            continue
        if digests != in_proc_1:
            cross_process_ok = False
            for i, (got, want) in enumerate(zip(digests, in_proc_1)):
                differing = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
                if differing:
                    cross_process_diffs.append(
                        {"run": label, "plan_index": i, "fields": differing}
                    )
    if not cross_process_ok:
        failures.append(
            f"compile is not reproducible across processes/working directories: "
            f"{cross_process_diffs}"
        )

    # -- hash coverage: every Goal.state.* registry path, mutated in turn --------------------
    base_plan = WatchPlan(condition="sv08 coverage probe")
    base_res, base_comp = compile_plan(base_plan)
    base = {
        "plan_hash": base_res.plan_hash,
        "component_snapshot_hash": base_res.component_snapshot_hash,
        "effective_hash": content_hash(dict(sorted(base_res.effective.items()))),
        "state_hash": content_hash(base_comp.goal.state),
    }

    # The ARMED (post-coercion) baseline, read back through the public arm boundary.
    coverage_core = _fresh_core()
    base_armed = _apply_verdict(
        coverage_core, WatchPlan(condition="sv08 coverage probe", id="sv08_base")
    )
    base["armed_state_hash"] = content_hash(base_armed["armed_state"] or {})

    coverage_rows: list[dict[str, Any]] = []
    for i, spec in enumerate(STATE_SPECS):
        value = _sample_for(spec)
        plan = WatchPlan(condition="sv08 coverage probe", explicit={spec.path: value})
        try:
            res, comp = compile_plan(plan)
        except (ConfigError, ValueError) as exc:
            coverage_rows.append(
                {
                    "path": spec.path,
                    "probe_value": repr(value),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        armed = _apply_verdict(
            coverage_core,
            WatchPlan(
                condition="sv08 coverage probe",
                id=f"sv08_{i}",
                explicit={spec.path: value},
            ),
        )
        armed_moved = (
            content_hash(armed["armed_state"] or {}) != base["armed_state_hash"]
            if armed["armed_state"] is not None
            else None
        )
        coverage_rows.append(
            {
                "path": spec.path,
                "probe_value": repr(value),
                "plan_hash_moved": res.plan_hash != base["plan_hash"],
                "component_snapshot_hash_moved": (
                    res.component_snapshot_hash != base["component_snapshot_hash"]
                ),
                "effective_moved": content_hash(dict(sorted(res.effective.items())))
                != base["effective_hash"],
                "compiled_state_moved": content_hash(comp.goal.state)
                != base["state_hash"],
                "armed_state_moved": armed_moved,
                "arm_status": armed["status"],
                "arm_dropped": armed["dropped"],
                "arm_coerced": armed["coerced"],
                "in_effective": spec.path in res.effective,
            }
        )

    ok_rows = [r for r in coverage_rows if "error" not in r]
    plan_hash_gaps = [r["path"] for r in ok_rows if not r["plan_hash_moved"]]
    effective_gaps = [r["path"] for r in ok_rows if not r["effective_moved"]]
    comp_hash_gaps = [r["path"] for r in ok_rows if not r["component_snapshot_hash_moved"]]
    # A field the hash claims changed but the ARMED runtime did NOT: the inverse hazard.
    # A drop the receipt REPORTS (coerced_dropped / a rejection) is loud, not silent.
    inert_reported: list[str] = []
    inert_silent: list[str] = []
    for r in ok_rows:
        if not r["plan_hash_moved"] or r["armed_state_moved"] is not False:
            continue
        reported = bool(r["arm_coerced"]) or bool(r["arm_dropped"])
        (inert_reported if reported else inert_silent).append(r["path"])
    if inert_silent:
        failures.append(
            f"{len(inert_silent)} Goal.state.* field(s) moved plan_hash but changed NOTHING in "
            f"the armed Watch, with no diagnostic on the receipt: {inert_silent}"
        )

    if plan_hash_gaps:
        failures.append(
            f"plan_hash does not cover {len(plan_hash_gaps)} Goal.state.* path(s) authored via "
            f"WatchPlan.explicit: {plan_hash_gaps}"
        )
    if effective_gaps:
        failures.append(
            f"ResolvedWatch.effective does not carry {len(effective_gaps)} authored "
            f"Goal.state.* path(s): {effective_gaps}"
        )

    # -- reverse probe: Goal.state.* written by a layer OTHER than the plan -------------------
    # A SourceProfile binds Goal.state.sense.* / Goal.state.cadence.* onto the compiled Goal
    # (percept/resolver.py:249-258). Neither hash on ResolvedWatch is taken over that.
    untuned = SourceProfile(
        name="sv08_untuned_coast",
        modalities=("video",),
        temporal_unit="frame",
        admission_strategy=UNTUNED,
        evidence_strategy="frames",
        identity_strategy="bbox_iou",
        parameters={"coast_interval_s": 4.0},
    )
    prof_res, prof_comp = compile_plan(base_plan, source_profile=untuned)
    profile_injected = {
        "compiled_state": dict(prof_comp.goal.state),
        "state_changed": content_hash(prof_comp.goal.state) != base["state_hash"],
        "plan_hash_moved": prof_res.plan_hash != base["plan_hash"],
        "component_snapshot_hash_moved": (
            prof_res.component_snapshot_hash != base["component_snapshot_hash"]
        ),
        "effective_moved": content_hash(dict(sorted(prof_res.effective.items())))
        != base["effective_hash"],
    }
    if profile_injected["state_changed"] and not (
        profile_injected["plan_hash_moved"]
        or profile_injected["component_snapshot_hash_moved"]
    ):
        failures.append(
            "a SourceProfile injected Goal.state.sense.* / Goal.state.cadence.* into the "
            "compiled Goal while BOTH ResolvedWatch.plan_hash and .component_snapshot_hash "
            "stayed identical (percept/resolver.py:839,883; percept/runtime_compiler.py:255-270)"
        )

    # -- reverse probe: Settings.* (the admission the Watch actually runs) --------------------
    base_settings = Settings()
    hot_settings = dataclasses.replace(
        base_settings,
        edge=dataclasses.replace(base_settings.edge, motion_gate=0.9, salience_stat="block_max"),
        gate=dataclasses.replace(base_settings.gate, threshold=0.95, refractory_s=120.0),
    )
    set_res, set_comp = compile_plan(base_plan, settings=hot_settings)
    settings_injected = {
        "watchspec_kwargs": dict(set_comp.watchspec_kwargs),
        "gate_threshold": set_comp.settings.gate.threshold,
        "watchspec_changed": content_hash(dict(set_comp.watchspec_kwargs))
        != content_hash(dict(base_comp.watchspec_kwargs)),
        "plan_hash_moved": set_res.plan_hash != base["plan_hash"],
        "component_snapshot_hash_moved": (
            set_res.component_snapshot_hash != base["component_snapshot_hash"]
        ),
        "effective_moved": content_hash(dict(sorted(set_res.effective.items())))
        != base["effective_hash"],
    }
    if settings_injected["watchspec_changed"] and not (
        settings_injected["plan_hash_moved"]
        or settings_injected["component_snapshot_hash_moved"]
    ):
        failures.append(
            "a Deployment/env Settings override changed the compiled admission "
            "(watchspec_kwargs + gate threshold) while BOTH ResolvedWatch hashes stayed "
            "identical (percept/resolver.py:839,883)"
        )

    return {
        "id": "SV-0.8",
        "name": "Compile determinism + hash coverage",
        "passed": not failures,
        "failures": failures,
        "same_process_deterministic": same_process,
        "cross_process_deterministic": cross_process_ok,
        "cross_process_diffs": cross_process_diffs,
        "determinism_runs": sorted(determinism_runs),
        "n_state_paths_probed": len(coverage_rows),
        "plan_hash_gaps": plan_hash_gaps,
        "effective_gaps": effective_gaps,
        "component_snapshot_hash_gaps": comp_hash_gaps,
        "hash_moved_but_runtime_inert_reported": inert_reported,
        "hash_moved_but_runtime_inert_silent": inert_silent,
        "profile_injected_state": profile_injected,
        "settings_injected": settings_injected,
        "coverage_rows": coverage_rows,
    }


# ============================================================================================
# SV-0.9 — Effective-config inspectability
# ============================================================================================


def _flat_state_paths(state: Mapping[str, Any], prefix: str = "Goal.state.") -> list[str]:
    out: list[str] = []
    for key, value in state.items():
        if isinstance(value, dict):
            out.extend(_flat_state_paths(value, f"{prefix}{key}."))
        else:
            out.append(f"{prefix}{key}")
    return out


def sv_0_9() -> dict[str, Any]:
    """Every value in the COMPILED Watch must be attributable to a resolution layer."""
    failures: list[str] = []

    scenarios: list[tuple[str, WatchPlan, Any]] = [
        ("bare", WatchPlan(condition="sv09 bare probe"), None),
        (
            "authored_explicit",
            WatchPlan(
                condition="sv09 authored probe",
                explicit={
                    "Goal.state.hold_s": 3.0,
                    "Goal.state.cadence.min_judge_interval": 1.0,
                    "Goal.state.cadence.gate_on": "fixed",
                    "WatchProfile.salience_stat": "block_max",
                },
                rearm="never",
                consequence={"miss_cost": "severe"},
                target="person",
            ),
            None,
        ),
        (
            "typed_plan_fields",
            WatchPlan(
                condition="sv09 typed-field probe",
                sense="audio",
                detectors=("motion",),
                schedule={"windows": [["09:00", "17:00"]]},
                target="doorway",
            ),
            None,
        ),
        (
            "source_profile",
            WatchPlan(condition="sv09 profile probe"),
            SourceProfile(
                name="sv09_untuned_coast",
                modalities=("video",),
                temporal_unit="frame",
                admission_strategy=UNTUNED,
                evidence_strategy="frames",
                identity_strategy="bbox_iou",
                parameters={"coast_interval_s": 4.0},
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    all_unattributed: dict[str, list[str]] = {}
    for label, plan, profile in scenarios:
        resolved, compiled = compile_plan(plan, source_profile=profile)
        effective = dict(resolved.effective)
        provenance = dict(resolved.provenance)

        key_parity = set(effective) == set(provenance)
        value_parity = all(effective[k] == provenance[k].value for k in effective)
        bad_layers = sorted(
            {o.owner_layer for o in provenance.values()} - LEGAL_OWNER_LAYERS
        )
        blank_sources = [k for k, o in provenance.items() if not str(o.source_id).strip()]
        blank_reasons = [k for k, o in provenance.items() if not str(o.reason).strip()]

        # Compiled Settings leaves.
        settings_leaves = {
            f"Settings.{sec}.{name}": val
            for sec, secval in vars(compiled.settings).items()
            for name, val in vars(secval).items()
        }
        settings_unattributed = sorted(k for k in settings_leaves if k not in effective)

        # Compiled Goal.state leaves.
        state_paths = _flat_state_paths(compiled.goal.state)
        state_unattributed = sorted(p for p in state_paths if p not in effective)

        unattributed = settings_unattributed + state_unattributed
        if unattributed:
            all_unattributed[label] = unattributed

        explain = resolved.explain()
        explain_covers = all(
            path in explain for path in list(provenance)[: min(len(provenance), 200)]
        )

        rows.append(
            {
                "scenario": label,
                "n_effective": len(effective),
                "n_provenance": len(provenance),
                "key_parity": key_parity,
                "value_parity": value_parity,
                "owner_layers": sorted({o.owner_layer for o in provenance.values()}),
                "illegal_owner_layers": bad_layers,
                "blank_source_ids": blank_sources,
                "blank_reasons": blank_reasons,
                "n_settings_leaves": len(settings_leaves),
                "settings_unattributed": settings_unattributed,
                "compiled_state_paths": state_paths,
                "state_unattributed": state_unattributed,
                "explain_covers_all_fields": explain_covers,
            }
        )

        if not key_parity:
            failures.append(f"[{label}] effective/provenance key sets differ")
        if not value_parity:
            failures.append(f"[{label}] effective[k] != provenance[k].value for some k")
        if bad_layers:
            failures.append(f"[{label}] owner_layer outside the declared vocabulary: {bad_layers}")
        if blank_sources:
            failures.append(f"[{label}] provenance entries with a blank source_id: {blank_sources}")
        if settings_unattributed:
            failures.append(
                f"[{label}] compiled Settings leaves with no provenance entry: "
                f"{settings_unattributed}"
            )
        if state_unattributed:
            failures.append(
                f"[{label}] compiled Goal.state values with no provenance entry: "
                f"{state_unattributed} (injected by percept/runtime_compiler.py:310,314,317-321,330,375)"
            )
        if not explain_covers:
            failures.append(f"[{label}] ResolvedWatch.explain() omits at least one resolved field")

    return {
        "id": "SV-0.9",
        "name": "Effective-config inspectability",
        "passed": not failures,
        "failures": failures,
        "scenarios": rows,
        "unattributed_by_scenario": all_unattributed,
    }


# ============================================================================================
# SV-0.10 — Unsupported-goal honesty
# ============================================================================================

#: Ten goals deliberately outside supported 1.x semantics. Each names the capability it needs.
OUT_OF_SCOPE_GOALS: tuple[tuple[str, str, str], ...] = (
    (
        "named_person_identity",
        "tell me when Divi specifically walks into the room",
        "person identity — requires biometric re-identification of a named individual",
    ),
    (
        "who_is_x",
        "who is the person standing at the front door right now",
        "person identity — answering 'who' requires an identity resolver percept refuses to ship",
    ),
    (
        "distinct_person_count",
        "count how many distinct people entered the building today",
        "counting DISTINCT people — requires durable cross-episode person re-identification",
    ),
    (
        "unique_visitors_now",
        "tell me how many unique visitors are in the queue right now",
        "de-duplicated person cardinality — requires person individuation",
    ),
    (
        "intent_inference",
        "alert me when someone looks like they intend to shoplift",
        "intent inference — predicting a future act from appearance",
    ),
    (
        "drowsiness_prediction",
        "let me know when the driver seems drowsy and is about to fall asleep",
        "intent/state prediction — inferring an unobserved future internal state",
    ),
    (
        "same_person_return",
        "notify me when the same person returns for a second time",
        "person re-identification across visits",
    ),
    (
        "role_discrimination",
        "tell me when an employee rather than a customer is behind the counter",
        "person-role attribution — requires identity or roster linkage",
    ),
    (
        "named_animal_identity",
        "recognise my dog Rex and tell me when he is on the sofa",
        "named-individual identity (non-human) — same re-identification capability",
    ),
    (
        "which_of_my_children",
        "identify which of my three children is at the fridge",
        "person disambiguation among known individuals",
    ),
)

#: An explicitly out-of-envelope identity resolver named through the supported knob.
OUT_OF_SCOPE_EXPLICIT: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "identity_policy_face_recognition",
        "tell me when the same face reappears",
        {"Goal.state.identity_policy": "face_recognition"},
    ),
)


def sv_0_10() -> dict[str, Any]:
    """Out-of-envelope goals must fail closed with a typed reason — never silently degrade."""
    core = _fresh_core()
    rows: list[dict[str, Any]] = []
    silent_degradations: list[dict[str, str]] = []

    def _probe(key: str, text: str, why: str, explicit: dict[str, Any] | None) -> None:
        plan = WatchPlan(condition=text, id=f"sv10_{key}", explicit=explicit or {})
        compile_error = None
        primitive = None
        state: dict[str, Any] = {}
        try:
            _resolved, compiled = compile_plan(plan)
            state = dict(compiled.goal.state)
            primitive = primitive_of(state)
        except Exception as exc:  # noqa: BLE001
            compile_error = f"{type(exc).__name__}: {exc}"

        verdict = None
        if compile_error is None:
            verdict = _apply_verdict(core, plan)

        failed_closed = compile_error is not None or (
            verdict is not None and verdict["status"] == "rejected"
        )
        typed_reason = None
        if verdict is not None and verdict["dropped"]:
            typed_reason = verdict["dropped"][0][1]

        row = {
            "key": key,
            "goal_text": text,
            "capability_required": why,
            "explicit": explicit or {},
            "compile_error": compile_error,
            "compiled_primitive": primitive,
            "compiled_state": state,
            "apply": verdict,
            "failed_closed": failed_closed,
            "typed_reason": typed_reason,
        }
        rows.append(row)
        if not failed_closed:
            silent_degradations.append(
                {
                    "key": key,
                    "goal_text": text,
                    "capability_required": why,
                    "degraded_to": f"primitive={primitive} state={state}",
                    "receipt_status": verdict["status"] if verdict else "n/a",
                }
            )

    for key, text, why in OUT_OF_SCOPE_GOALS:
        _probe(key, text, why, None)
    for key, text, explicit in OUT_OF_SCOPE_EXPLICIT:
        _probe(
            key,
            text,
            "person-identifying identity resolver named through Goal.state.identity_policy",
            explicit,
        )

    failures: list[str] = []
    if silent_degradations:
        failures.append(
            f"{len(silent_degradations)}/{len(rows)} out-of-scope goals compiled and ARMED "
            "silently as an ordinary boolean frame Watch — no typed refusal at any point in "
            "WatchPlan -> ConfigResolver -> RuntimeCompiler -> apply_watch_plan "
            "(percept/goalsugar.py:3-11,66-90; percept/watchplan_apply.py:417-443 refuses only "
            "admission/evidence/extractors/uncertainty; percept/identity_policy.py:139-155 "
            "checks the envelope only for an already-REGISTERED resolver)"
        )
    typed = [r for r in rows if r["failed_closed"] and not r["typed_reason"]]
    if typed:
        failures.append(
            f"{len(typed)} refusal(s) carried no typed reason code: {[r['key'] for r in typed]}"
        )

    return {
        "id": "SV-0.10",
        "name": "Unsupported-goal honesty",
        "passed": not failures,
        "failures": failures,
        "n_probed": len(rows),
        "n_failed_closed": sum(r["failed_closed"] for r in rows),
        "n_silent_degradations": len(silent_degradations),
        "silent_degradations": silent_degradations,
        "rows": rows,
    }


# ============================================================================================
# Driver
# ============================================================================================


def load_corpus() -> list[dict]:
    doc = json.loads(CORPUS.read_text())
    return list(doc["cases"])


def run_all() -> dict[str, Any]:
    cases = load_corpus()
    tests: list[dict[str, Any]] = []
    for label, fn in (
        ("SV-0.7", lambda: sv_0_7(cases)),
        ("SV-0.8", sv_0_8),
        ("SV-0.9", sv_0_9),
        ("SV-0.10", sv_0_10),
    ):
        try:
            tests.append(fn())
        except Exception as exc:  # noqa: BLE001 — a harness crash is reported, never hidden
            import traceback

            tests.append(
                {
                    "id": label,
                    "name": "<harness error>",
                    "passed": False,
                    "failures": [f"{type(exc).__name__}: {exc}"],
                    "traceback": traceback.format_exc(),
                }
            )
    return {
        "suite": "sv0_compile",
        "runtime": {
            "percept_version": getattr(percept, "__version__", "?"),
            "percept_file": percept.__file__,
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "corpus": {"path": str(CORPUS), "n_cases": len(cases)},
        "tests": tests,
        "passed": all(t["passed"] for t in tests),
    }


def _print_table(report: dict[str, Any]) -> None:
    print()
    print(f"percept {report['runtime']['percept_version']}  <- {report['runtime']['percept_file']}")
    print(f"corpus  {report['corpus']['n_cases']} cases")
    print()
    print(f"{'TEST':<9} {'RESULT':<7} {'NAME':<38} SUMMARY")
    print("-" * 118)
    for t in report["tests"]:
        result = "PASS" if t["passed"] else "FAIL"
        summary = ""
        if t["id"] == "SV-0.7":
            summary = (
                f"default {t.get('default_fidelity')} | authored {t.get('authored_fidelity')} "
                f"| armed {t.get('authored_armed')} | executable "
                f"{t.get('reachable_and_executable')} | kind-sensitive="
                f"{t.get('kind_sensitive_compile')}"
            )
        elif t["id"] == "SV-0.8":
            summary = (
                f"deterministic in-proc={t.get('same_process_deterministic')} "
                f"cross-proc={t.get('cross_process_deterministic')} | "
                f"plan_hash gaps={len(t.get('plan_hash_gaps', []))}/"
                f"{t.get('n_state_paths_probed')} | "
                f"comp_hash gaps={len(t.get('component_snapshot_hash_gaps', []))}"
            )
        elif t["id"] == "SV-0.9":
            n = sum(len(v) for v in t.get("unattributed_by_scenario", {}).values())
            summary = f"{n} compiled value(s) with no provenance entry"
        elif t["id"] == "SV-0.10":
            summary = (
                f"{t.get('n_failed_closed')}/{t.get('n_probed')} failed closed | "
                f"{t.get('n_silent_degradations')} silent degradation(s)"
            )
        print(f"{t['id']:<9} {result:<7} {t['name']:<38} {summary}")
    print("-" * 118)

    for t in report["tests"]:
        if t["passed"]:
            continue
        print(f"\n{t['id']} failures:")
        for f in t["failures"]:
            print(f"  - {f}")

    sv07 = next((t for t in report["tests"] if t["id"] == "SV-0.7"), None)
    if sv07 and "breakdown" in sv07:
        print("\nSV-0.7 per-goal-kind compile breakdown")
        print(
            f"{'goal_kind':<12}{'n':>4}{'default OK':>12}{'authored OK':>13}"
            f"{'armed':>8}{'executable':>12}  default primitives / reject codes"
        )
        for kind, row in sv07["breakdown"].items():
            prims = ", ".join(f"{k}={v}" for k, v in row["default_primitives"].items())
            rejects = ", ".join(f"{k}={v}" for k, v in row["authored_reject_codes"].items())
            note = prims + (f"  [authored reject: {rejects}]" if rejects else "")
            print(
                f"{kind:<12}{row['n']:>4}{row['default_match']:>12}{row['authored_match']:>13}"
                f"{row['authored_armed']:>8}{row['reachable_and_executable']:>12}  {note}"
            )

    sv10 = next((t for t in report["tests"] if t["id"] == "SV-0.10"), None)
    if sv10 and sv10.get("silent_degradations"):
        print("\nSV-0.10 silent degradations")
        for d in sv10["silent_degradations"]:
            print(f"  - {d['key']}: {d['degraded_to']}  (needs: {d['capability_required']})")

    print()
    print("SUITE:", "PASS" if report["passed"] else "FAIL")


def main() -> int:
    report = run_all()
    RESULTS.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=False, default=str))
    _print_table(report)
    print(f"\nJSON -> {JSON_OUT}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
