"""Derive the supported, human-adjudicated v1.2 qualification corpus from frozen v1.1.

This is intentionally not a general corpus builder.  It applies the checked-in adjudication ledger
only after verifying the exact v1.1 source hash and each case's expected text/kind.  Media pins and
positive intervals are immutable.  Unsupported or unresolved tasks are retained in an exclusion
ledger, not silently relabelled into the nearest shipped primitive.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name("corpus.v1.1.json")
DEFAULT_ADJUDICATION = Path(__file__).with_name("corpus.v1.2-adjudication.json")
DEFAULT_OUT = Path(__file__).with_name("corpus.v1.2.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(source: dict[str, Any], adjudication: dict[str, Any], *, source_sha: str) -> dict[str, Any]:
    if source.get("corpus_version") != "1.1":
        raise ValueError(f"expected corpus v1.1, got {source.get('corpus_version')!r}")
    if adjudication.get("base_corpus_sha256") != source_sha:
        raise ValueError(
            f"adjudication pins {adjudication.get('base_corpus_sha256')}, source is {source_sha}"
        )

    cases = {str(case["case_id"]): copy.deepcopy(case) for case in source.get("cases") or []}
    decisions = adjudication.get("decisions") or []
    if len({row.get("case_id") for row in decisions}) != len(decisions):
        raise ValueError("adjudication contains duplicate case decisions")

    excluded: list[dict[str, Any]] = []
    corrected: list[dict[str, Any]] = []
    for decision in decisions:
        case_id = str(decision["case_id"])
        if case_id not in cases:
            raise ValueError(f"adjudication case is absent from v1.1: {case_id}")
        case = cases[case_id]
        if case.get("goal_text") != decision.get("goal_text"):
            raise ValueError(f"{case_id}: goal text changed since adjudication")
        if case.get("goal_kind") != decision.get("from"):
            raise ValueError(
                f"{case_id}: expected kind {decision.get('from')!r}, got {case.get('goal_kind')!r}"
            )
        action = decision.get("action")
        audit = {
            "reason_code": decision["reason_code"],
            "evidence": decision["evidence"],
            "adjudication": "eval/stage-validation/corpus.v1.2-adjudication.json",
        }
        if action == "goal_kind_override":
            case["goal_kind"] = decision["to"]
            case.setdefault("provenance", {})["mechanic_adjudication"] = audit | {
                "from": decision["from"],
                "to": decision["to"],
            }
            corrected.append({"case_id": case_id, "from": decision["from"],
                              "to": decision["to"], "reason_code": decision["reason_code"]})
        elif action == "exclude":
            excluded.append({
                "case_id": case_id,
                "goal_text": case["goal_text"],
                "prior_goal_kind": case["goal_kind"],
                **audit,
            })
            del cases[case_id]
        else:
            raise ValueError(f"{case_id}: unsupported adjudication action {action!r}")

    ordered = [cases[str(case["case_id"])] for case in source["cases"] if str(case["case_id"]) in cases]
    counts = Counter(str(case["dataset"]) for case in ordered)
    requested = int(source.get("per_dataset") or 0)
    coverage = {
        name: {
            "requested": requested,
            "built": counts.get(name, 0),
            "excluded_by_adjudication": sum(1 for row in excluded if row["case_id"].startswith(name + "::")),
            "short_by": max(0, requested - counts.get(name, 0)),
        }
        for name in sorted((source.get("coverage") or {}).keys())
    }

    out = copy.deepcopy(source)
    out.update(
        corpus_version="1.2",
        cases=ordered,
        n_cases=len(ordered),
        coverage=coverage,
        derived_from={
            "corpus_version": "1.1",
            "path": "eval/stage-validation/corpus.v1.1.json",
            "sha256": source_sha,
            "method": "checked-in independent mechanic adjudication; media pins and positive intervals unchanged",
        },
        qualification_scope={
            "supported": "single shipped perception primitive with scoreable source-clock ground truth",
            "statistical_launch_eligibility": False,
            "reason": "96 cases are below the 299 independent zero-miss minimum and are not deployment cells",
        },
        mechanic_adjudication={
            "path": "eval/stage-validation/corpus.v1.2-adjudication.json",
            "corrected": corrected,
            "excluded": excluded,
        },
    )
    out["changelog"] = list(source.get("changelog") or []) + [
        "v1.2: corrected four source-backed ProactiveQA activity labels and excluded four unresolved or unsupported single-primitive tasks; media hashes and positives unchanged."
    ]
    return out


def render(source_path: Path, adjudication_path: Path) -> str:
    source = json.loads(source_path.read_text())
    adjudication = json.loads(adjudication_path.read_text())
    return json.dumps(derive(source, adjudication, source_sha=sha256(source_path)), indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = render(args.source, args.adjudication)
    if args.check:
        if not args.out.is_file() or args.out.read_text() != rendered:
            raise SystemExit(f"{args.out} is missing or stale")
        print(f"verified {args.out}")
        return 0
    args.out.write_text(rendered)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
