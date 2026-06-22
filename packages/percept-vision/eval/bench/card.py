"""The Snapshot Card emitter (spec §7) — turns a list of TrackResults + run metadata into a stamped card.

A card is DRAFT unless: mode=full AND no modeled cells AND a pinned data-manifest (spec §7.2). Phase A is
sampled + modeled, so it emits DRAFT — by design.
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
    any_modeled = any(r.modeled for r in results)
    return "RELEASE" if (mode == "full" and not any_modeled and manifest) else "DRAFT"


def render_card(results: list[TrackResult], *, mode: str = "sampled", backend: str = "?", commit: str = "?",
                manifest: str = "", fps: str = "?", device: str = "?", seeds: int = 1,
                ceiling: str = "none", auroc=None) -> str:
    stamp = stamp_of(mode, results, manifest)
    auroc_s = f"{auroc:.2f}" if isinstance(auroc, (int, float)) else "?"
    L = [
        "=" * 84,
        f"# Percept Benchmark Snapshot",
        f"spec: {SPEC_VERSION} · mode: {mode} · stamp: {stamp}",
        f"commit: {commit} · data-manifest: {manifest or '(none — DRAFT)'}",
        f"backbone: {backend}   ceiling: {ceiling}",
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

    L += ["", "## Per-track × config", f"  {'track':<12}{'metric':<18}" + "".join(f"{c:>11}" for c in CONFIGS)]
    for r in results:
        for metric in r.metrics():
            cells = []
            for c in CONFIGS:
                v = r.cells.get((c, metric))
                if v is None:
                    cells.append(f"{'n/a':>11}")
                    continue
                s = v if isinstance(v, str) else r.fmt(metric).format(v)
                if f"{c}:{metric}" in r.modeled:
                    s = "~" + s
                cells.append(f"{s:>11}")
            L.append(f"  {r.track:<12}{metric:<18}" + "".join(cells))

    measured = sorted({m for r in results for m in r.measured})
    modeled = sorted({m for r in results for m in r.modeled})
    excluded = [e for r in results for e in r.excluded]
    L += [
        "", "## Provenance",
        f"  measured : {measured}",
        f"  modeled  : {modeled}   # must be EMPTY for a RELEASE stamp",
        f"  excluded : {excluded}",
        "", "## References",
    ]
    L += [f"  {ref}" for ref in _references()]
    L += ["=" * 84, ""]
    return "\n".join(L)
