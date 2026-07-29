"""Aggregate the per-stage SV results into one report.

Each stage suite writes `results/<suite>.json`; this collects them into a single stage-ordered
view. The point is localisation: an end-to-end number cannot say which stage moved it, so the
report is organised by stage and refuses to average across them.

A suite that has not run is reported as MISSING, never as passing. Silence is not success.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CORPUS = HERE / "corpus.json"
LONG_SOURCES = HERE / "long_sources.json"

#: Suite -> (stage label, human title). Ordered as the pipeline runs, because a Stage-0 defect
#: invalidates every downstream number and the report must read in that order.
SUITES: dict[str, tuple[str, str]] = {
    "sv0_ingest": ("STAGE 0", "Ingest + normalize"),
    "sv0_compile": ("STAGE 0", "Planner + compiler"),
    "sv1_admission": ("STAGE 1", "Admission (VID/VAD, dwell)"),
    "sv2_senses": ("STAGE 2", "Local perception (senses, reducers)"),
    "sv3_pool": ("STAGE 3", "Semantic judges (boundedness)"),
    "sv4_cognition": ("STAGE 4", "Cognition + disposition"),
    # Audio spans stages 1/2/4 rather than sitting in one, so it reports as its own suite instead of
    # being split across three and losing the cross-stage picture.
    "sv_audio": ("AUDIO", "Audio admission, senses, cognition"),
}


def _load(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _tests_of(payload: Any) -> list[dict[str, Any]]:
    """Pull a test list out of a suite payload without demanding one exact schema.

    The suites are written independently; insisting on a single shape would make the report brittle
    against the very thing it exists to collect.
    """
    if isinstance(payload, dict):
        for key in ("tests", "results", "cases"):
            value = payload.get(key)
            if isinstance(value, list):
                return [t for t in value if isinstance(t, dict)]
            if isinstance(value, dict):
                return [{"id": k, **v} for k, v in value.items() if isinstance(v, dict)]
    if isinstance(payload, list):
        return [t for t in payload if isinstance(t, dict)]
    return []


def _status_of(test: dict[str, Any]) -> str:
    for key in ("status", "verdict", "result", "outcome"):
        value = test.get(key)
        if isinstance(value, str):
            return value.strip().upper()
        if isinstance(value, bool):
            return "PASS" if value else "FAIL"
    if isinstance(test.get("passed"), bool):
        return "PASS" if test["passed"] else "FAIL"
    return "UNKNOWN"


def build() -> dict[str, Any]:
    corpus = _load(CORPUS) or {}
    long_sources = _load(LONG_SOURCES) or {}
    stages: dict[str, list[dict[str, Any]]] = {}
    totals = {"PASS": 0, "FAIL": 0, "UNKNOWN": 0, "MISSING": 0}

    for suite, (stage, title) in SUITES.items():
        payload = _load(RESULTS / f"{suite}.json")
        entry: dict[str, Any] = {"suite": suite, "title": title}
        if payload is None:
            entry.update(status="MISSING", tests=[], counts={})
            totals["MISSING"] += 1
        else:
            tests = _tests_of(payload)
            counts: dict[str, int] = {}
            for test in tests:
                status = _status_of(test)
                counts[status] = counts.get(status, 0) + 1
                totals[status] = totals.get(status, 0) + 1
            entry.update(
                status="FAIL" if counts.get("FAIL") else ("PASS" if tests else "EMPTY"),
                tests=[
                    {
                        "id": t.get("id") or t.get("name") or "?",
                        "status": _status_of(t),
                        "detail": t.get("detail") or t.get("summary") or t.get("message") or "",
                        "measured": t.get("measured") or t.get("metrics") or {},
                    }
                    for t in tests
                ],
                counts=counts,
            )
        stages.setdefault(stage, []).append(entry)

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {
            "n_cases": corpus.get("n_cases"),
            "coverage": corpus.get("coverage"),
        },
        "long_sources": {
            "n_sources": long_sources.get("n_sources"),
            "sources": [
                {
                    "goal_key": s["goal_key"],
                    "duration_s": s["total_duration_s"],
                    "expected_occurrences": s["expected_occurrences"],
                }
                for s in long_sources.get("sources", [])
            ],
        },
        "totals": totals,
        "stages": stages,
    }


def render(report: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("# Percept stage-validation report")
    add("")
    add(f"Generated: {report['generated']}")
    add(f"Runtime under test: exact launch wheel `percept 1.0.0` (`bba572a`)")
    add("")

    corpus = report["corpus"]
    if corpus.get("n_cases"):
        add(f"Corpus: **{corpus['n_cases']} cases**")
        coverage = corpus.get("coverage") or {}
        if coverage:
            add("")
            add("| Dataset | Cases |")
            add("|---|---|")
            for name, info in coverage.items():
                add(f"| {name} | {info.get('built')}/{info.get('requested')} |")
        add("")

    long_sources = report["long_sources"]
    if long_sources.get("n_sources"):
        add(f"Long-horizon sources: **{long_sources['n_sources']}**")
        add("")
        add("| Goal | Duration | Expected occurrences |")
        add("|---|---|---|")
        for s in long_sources["sources"]:
            add(f"| `{s['goal_key']}` | {s['duration_s']:.1f}s | {s['expected_occurrences']} |")
        add("")

    totals = report["totals"]
    add("## Totals")
    add("")
    add(f"PASS **{totals.get('PASS', 0)}** · FAIL **{totals.get('FAIL', 0)}** · "
        f"UNKNOWN {totals.get('UNKNOWN', 0)} · suites MISSING {totals.get('MISSING', 0)}")
    add("")

    for stage in ("STAGE 0", "STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "AUDIO"):
        entries = report["stages"].get(stage)
        if not entries:
            continue
        add(f"## {stage}")
        add("")
        for entry in entries:
            add(f"### {entry['title']} (`{entry['suite']}`)")
            add("")
            if entry["status"] == "MISSING":
                add("**MISSING** — suite did not produce results. Not counted as passing.")
                add("")
                continue
            counts = ", ".join(f"{k} {v}" for k, v in sorted(entry["counts"].items()))
            add(f"Status: **{entry['status']}** ({counts})")
            add("")
            if entry["tests"]:
                add("| Test | Status | Detail |")
                add("|---|---|---|")
                for test in entry["tests"]:
                    detail = str(test["detail"]).replace("|", "\\|")[:160]
                    add(f"| {test['id']} | {test['status']} | {detail} |")
                add("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "REPORT.md")
    ap.add_argument("--json-out", type=Path, default=HERE / "results" / "report.json")
    args = ap.parse_args()

    report = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    args.out.write_text(render(report))

    totals = report["totals"]
    print(f"report -> {args.out}")
    print(f"  PASS {totals.get('PASS',0)}  FAIL {totals.get('FAIL',0)}  "
          f"UNKNOWN {totals.get('UNKNOWN',0)}  MISSING suites {totals.get('MISSING',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
