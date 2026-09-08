"""Derive a human-adjudicated frozen plan for supported corpus v1.2 with zero provider calls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name("corpus-v1.1.frozen.json")
DEFAULT_CORPUS = REPO / "eval" / "stage-validation" / "corpus.v1.2.json"
DEFAULT_ADJUDICATION = REPO / "eval" / "stage-validation" / "corpus.v1.2-adjudication.json"
DEFAULT_OUT = Path(__file__).with_name("corpus-v1.2.frozen.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_payload(
    source: dict[str, Any], corpus: dict[str, Any], adjudication: dict[str, Any]
) -> dict[str, Any]:
    from percept.conductor import AuthoredPrimitive, authoring_key, goal_kind_of

    if corpus.get("corpus_version") != "1.2":
        raise ValueError(f"expected corpus v1.2, got {corpus.get('corpus_version')!r}")
    if source.get("corpus", {}).get("version") != "1.1":
        raise ValueError("source pin is not the frozen v1.1 artifact")

    model = str(source["model"])
    cases = list(corpus.get("cases") or [])
    goal_texts = {str(case["goal_text"]) for case in cases}
    entries: dict[str, dict[str, Any]] = {}
    for text in sorted(goal_texts):
        key = authoring_key(text, model=model)
        if key not in (source.get("entries") or {}):
            raise ValueError(f"v1.1 pin has no decision for supported goal {text!r}")
        entries[key] = copy.deepcopy(source["entries"][key])

    override = dict(adjudication.get("conductor_override") or {})
    override_case = next(
        (case for case in cases if case.get("case_id") == override.get("case_id")), None
    )
    if override_case is None or override_case.get("goal_text") != override.get("goal_text"):
        raise ValueError("conductor override does not identify one supported v1.2 case")
    override_key = authoring_key(str(override["goal_text"]), model=model)
    prior = entries[override_key]
    if prior.get("behavior_type") != override.get("from_behavior_type"):
        raise ValueError("conductor decision changed since adjudication")
    entries[override_key] = {
        "behavior_type": override["to_behavior_type"],
        "explicit": copy.deepcopy(override["explicit"]),
        "goal_text": override["goal_text"],
        "key": override_key,
        "model": model,
        "origin": "human_adjudication",
    }

    hits = 0
    per_kind: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, str]] = []
    for case in cases:
        text = str(case["goal_text"])
        primitive = AuthoredPrimitive.from_record(entries[authoring_key(text, model=model)])
        predicted, expected = goal_kind_of(primitive), str(case["goal_kind"])
        cell = per_kind.setdefault(expected, {"n": 0, "hit": 0})
        cell["n"] += 1
        if predicted == expected:
            hits += 1
            cell["hit"] += 1
        else:
            mismatches.append({
                "case_id": str(case["case_id"]), "expected": expected,
                "predicted": predicted, "behavior_type": primitive.behavior_type,
            })
    if mismatches:
        raise ValueError(f"v1.2 frozen plan is not mechanics-valid: {mismatches}")

    return {
        "kind": "percept-authoring-cache",
        "prompt_version": source["prompt_version"],
        "model": model,
        "entries": dict(sorted(entries.items())),
        "corpus": {
            "path": "eval/stage-validation/corpus.v1.2.json",
            "version": "1.2",
            "n_cases": len(cases),
            "n_unique_goals": len(entries),
        },
        "migration": {
            "kind": "independent-human-mechanic-adjudication",
            "source": "eval/primitive-authoring/corpus-v1.1.frozen.json",
            "adjudication": "eval/stage-validation/corpus.v1.2-adjudication.json",
            "provider_calls": 0,
            "entries_reused": len(entries) - 1,
            "entries_removed_as_out_of_scope": len(source.get("entries") or {}) - len(entries),
            "overrides": [{
                "case_id": override["case_id"],
                "key": override_key,
                "from": override["from_behavior_type"],
                "to": override["to_behavior_type"],
                "reason_code": override["reason_code"],
            }],
        },
        "scorecard": {
            "n": len(cases), "hit": hits,
            "accuracy": hits / len(cases) if cases else None,
            "per_kind": per_kind, "mismatches": mismatches,
            "scope": "human-adjudicated supported single-primitive corpus; not model-only selection accuracy",
        },
    }


def render(source_path: Path, corpus_path: Path, adjudication_path: Path) -> str:
    payload = migrate_payload(
        json.loads(source_path.read_text()),
        json.loads(corpus_path.read_text()),
        json.loads(adjudication_path.read_text()),
    )
    payload["corpus"]["sha256"] = sha256(corpus_path)
    payload["migration"]["source_sha256"] = sha256(source_path)
    payload["migration"]["adjudication_sha256"] = sha256(adjudication_path)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = render(args.source, args.corpus, args.adjudication)
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
