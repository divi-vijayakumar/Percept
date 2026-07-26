"""Snapshot Card emitter: turn TrackResults and run metadata into a stamped benchmark card.

A benchmark card is DRAFT unless: mode=full, no non-dense modeled cells, no diagnostic/excluded tracks, all
coverage gates pass, and a pinned data-manifest is present. That is intentionally strict: benchmark
code should fail closed rather than accidentally publishing diagnostics as product E2E.
"""

from __future__ import annotations

from .types import CONFIGS, TrackResult

SPEC_VERSION = "v1.0"


def _references() -> list[str]:
    try:
        from eval.bench_matrix import REFERENCES
        return list(REFERENCES)
    except Exception:  # pragma: no cover
        return ['Thinking Machines Lab, "Interaction Models", Connectionism, May 2026. doi:10.64434/tml.20260511']


def stamp_of(mode: str, results: list[TrackResult], manifest: str) -> str:
    if mode != "full" or not manifest:
        return "DRAFT"
    return "RELEASE" if not release_blockers(results) else "DRAFT"


def release_blockers(results: list[TrackResult]) -> list[str]:
    if not results:
        return ["no benchmark track results"]
    blockers: list[str] = []
    for result in results:
        blockers.extend(result.invalid_reasons())
    return blockers


def execution_blockers(results: list[TrackResult]) -> list[str]:
    """Minimum truthfulness bar for writing any scorecard, including an OSS DRAFT."""
    if not results:
        return ["no benchmark track results"]
    names = [result.track for result in results]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    blockers = [f"duplicate track result: {name}" for name in duplicates]
    blockers.extend(
        f"{result.track}: status={result.status}"
        for result in results
        if result.status in {"blocked", "failed"}
    )
    if not any(any(value is not None for value in result.cells.values()) for result in results):
        blockers.append("all benchmark result cells are empty/incomplete")
    return blockers


def render_card(results: list[TrackResult], *, mode: str = "sampled", backend: str = "?", commit: str = "?",
                manifest: str = "", fps: str = "?", device: str = "?", seeds: int = 1,
                ceiling: str = "none", auroc=None, budget_plan: str = "") -> str:
    stamp = stamp_of(mode, results, manifest)
    auroc_s = f"{auroc:.2f}" if isinstance(auroc, (int, float)) else "?"
    L = [
        "=" * 84,
        "# Percept Benchmark Snapshot",
        f"spec: {SPEC_VERSION} · mode: {mode} · stamp: {stamp}",
        f"commit: {commit} · data-manifest: {manifest or '(none — DRAFT)'}",
        f"backbone: {backend}   ceiling: {ceiling}",
        f"budget-plan: {budget_plan or '(not recorded)'}",
        f"fps: {fps} · device: {device} · seeds: {seeds}",
        f"gates: G3 OFF (no world model) · consequence-model OFF (R3: AUROC {auroc_s} vs 0.5)",
        "=" * 84,
        "",
        "## Headline (vector — no composite)",
    ]
    for r in results:
        for label, val in r.headline.items():
            L.append(f"  {label:<36}: {val}")
    L.append(f"  {'total vision calls':<36}: {sum(r.calls for r in results)}")

    track_w = max(14, *(len(r.track) + 2 for r in results))
    metric_w = max(24, *(len(metric) + 2 for r in results for metric in r.metrics()))
    lane_w = max(14, max(len(c) for c in CONFIGS) + 2)
    L += ["", "## Per-track × benchmark lane   (ref = published SOTA, where the metric is comparable)",
          f"  {'track':<{track_w}}{'metric':<{metric_w}}" + "".join(f"{c:>{lane_w}}" for c in CONFIGS) + f"   {'ref (SOTA)':<24}"]
    for r in results:
        for metric in r.metrics():
            cells = []
            for c in CONFIGS:
                v = r.cells.get((c, metric))
                if v is None:
                    cells.append(f"{'n/a':>10}")
                    continue
                s = v if isinstance(v, str) else r.fmt(metric).format(v)
                if f"{c}:{metric}" in r.modeled:
                    s = "~" + s
                cells.append(f"{s:>10}")
            ref = r.reference.get(metric)
            ref_s = ""
            if ref:
                rv = ref["value"]
                rv = rv if isinstance(rv, str) else r.fmt(metric).format(rv)
                ref_s = f"{rv} ({ref.get('by', '')})"
            L.append(f"  {r.track:<{track_w}}{metric:<{metric_w}}" + "".join(f"{c:>{lane_w}}" for c in cells) + f"   {ref_s:<24}")

    measured = sorted({m for r in results for m in r.measured})
    modeled = sorted({m for r in results for m in r.modeled})
    excluded = [e for r in results for e in r.excluded]
    statuses = [f"{r.track}:{r.status}" for r in results]
    schemas = [f"{r.track}:{r.schema_version}" for r in results]
    quality_gates = [
        f"{r.track}:{gate.get('id', '(unnamed)')}={gate.get('status', '?')}"
        for r in results
        for gate in r.quality_gates
    ]
    measurements = [
        (
            f"{r.track}:{label} "
            f"[{measurement.unit}; {measurement.conditioning}; "
            f"denominator={measurement.denominator}]"
        )
        for r in results
        for label, measurement in r.normalized_measurements().items()
    ]
    coverage = []
    for r in results:
        for c in r.normalized_coverage():
            status = "pass" if c.passed else "fail"
            coverage.append(
                f"{r.track}:{c.dataset} {status} expected={c.expected} observed={c.observed} "
                f"failed={c.failed} skipped={c.skipped} case_ids={len(c.case_ids)}"
                + (f" ({c.notes})" if c.notes else "")
            )
    blockers = release_blockers(results)
    L += [
        "", "## Provenance",
        f"  measured : {measured}",
        f"  modeled  : {modeled}   # non-dense modeled cells block RELEASE; dense-modeled is cost reference only",
        f"  excluded : {excluded}",
        f"  track-status : {statuses}",
        f"  result-schema : {schemas}",
        f"  typed-measurements : {measurements}",
        f"  quality-gates : {quality_gates}",
        f"  coverage : {coverage}",
        f"  release-blockers : {blockers}",
        "", "## References",
    ]
    L += [f"  {ref}" for ref in _references()]
    L += ["=" * 84, ""]
    return "\n".join(L)
