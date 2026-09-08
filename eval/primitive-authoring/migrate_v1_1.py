"""Migrate the frozen v1.0 conductor pin onto corpus v1.1's text-only normalizations.

This script does not author.  It re-keys an existing frozen decision only when the checked-in corpus
diff proves case membership, media hashes, positive intervals, and goal kinds are unchanged and
names the exact before/after text.  Genuine primitive disagreements remain in the resulting
scorecard and are intentionally left for adjudication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name("corpus-100.frozen.json")
DEFAULT_DIFF = REPO / "eval" / "stage-validation" / "corpus.version-diff.json"
DEFAULT_CORPUS = REPO / "eval" / "stage-validation" / "corpus.v1.1.json"
DEFAULT_OUT = Path(__file__).with_name("corpus-v1.1.frozen.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_payload(source: dict[str, Any], diff: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    from percept.conductor import AuthoredPrimitive, authoring_key, goal_kind_of

    required_invariants = {
        "case_membership_identical": True,
        "positives_identical": True,
        "video_sha256_identical": True,
    }
    bad = [key for key, expected in required_invariants.items() if diff.get(key) is not expected]
    if bad or diff.get("goal_kind_changed"):
        raise ValueError(
            "v1.1 pin migration requires a text-only corpus diff; violated: "
            + ", ".join(bad + (["goal_kind_changed"] if diff.get("goal_kind_changed") else []))
        )
    if str(diff.get("to_version")) != str(corpus.get("corpus_version")):
        raise ValueError(
            f"diff target {diff.get('to_version')!r} != corpus {corpus.get('corpus_version')!r}"
        )

    source_records = list((source.get("entries") or {}).values())
    by_text: dict[str, dict[str, Any]] = {}
    for record in source_records:
        text = str(record.get("goal_text") or "")
        if text in by_text and by_text[text] != record:
            raise ValueError(f"source pin has conflicting records for {text!r}")
        by_text[text] = copy.deepcopy(record)

    before_for_after: dict[str, str] = {}
    for change in diff.get("goal_text_changes") or []:
        before, after = str(change["goal_text_before"]), str(change["goal_text_after"])
        prior = before_for_after.setdefault(after, before)
        if prior != before:
            raise ValueError(f"multiple v1.0 texts map to {after!r}: {prior!r}, {before!r}")

    entries: dict[str, dict[str, Any]] = {}
    migrated: list[dict[str, str]] = []
    reused = 0
    cases = list(corpus.get("cases") or [])
    for goal_text in sorted({str(case["goal_text"]) for case in cases}):
        record = by_text.get(goal_text)
        if record is None:
            before = before_for_after.get(goal_text)
            if before is None or before not in by_text:
                raise ValueError(f"no frozen v1.0 decision or proven rename for {goal_text!r}")
            record = copy.deepcopy(by_text[before])
            record["goal_text"] = goal_text
            record["key"] = authoring_key(goal_text, model=str(record["model"]))
            migrated.append({"before": before, "after": goal_text, "key": record["key"]})
        else:
            reused += 1
        expected_key = authoring_key(goal_text, model=str(record["model"]))
        if record.get("key") != expected_key:
            raise ValueError(f"record key mismatch for {goal_text!r}")
        entries[expected_key] = record

    hits = 0
    per_kind: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, str]] = []
    for case in cases:
        text = str(case["goal_text"])
        record = entries[authoring_key(text, model=str(source["model"]))]
        primitive = AuthoredPrimitive.from_record(record)
        predicted, expected = goal_kind_of(primitive), str(case["goal_kind"])
        cell = per_kind.setdefault(expected, {"n": 0, "hit": 0})
        cell["n"] += 1
        if predicted == expected:
            hits += 1
            cell["hit"] += 1
        else:
            mismatches.append(
                {
                    "case_id": str(case["case_id"]),
                    "goal_text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "behavior_type": primitive.behavior_type,
                }
            )

    return {
        "kind": "percept-authoring-cache",
        "prompt_version": source["prompt_version"],
        "model": source["model"],
        "entries": dict(sorted(entries.items())),
        "corpus": {
            "path": "eval/stage-validation/corpus.v1.1.json",
            "version": corpus.get("corpus_version"),
            "n_cases": len(cases),
            "n_unique_goals": len(entries),
        },
        "migration": {
            "kind": "proven-text-normalization",
            "source": "eval/primitive-authoring/corpus-100.frozen.json",
            "diff": "eval/stage-validation/corpus.version-diff.json",
            "provider_calls": 0,
            "reused_exact": reused,
            "rekeyed": migrated,
            "invariants": required_invariants | {"goal_kind_changed": 0},
        },
        "scorecard": {
            "n": len(cases),
            "hit": hits,
            "accuracy": hits / len(cases) if cases else None,
            "per_kind": per_kind,
            "mismatches": mismatches,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--diff", type=Path, default=DEFAULT_DIFF)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="verify --out is byte-identical")
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text())
    diff = json.loads(args.diff.read_text())
    corpus = json.loads(args.corpus.read_text())
    payload = migrate_payload(source, diff, corpus)
    payload["corpus"]["sha256"] = _sha(args.corpus)
    payload["migration"]["source_sha256"] = _sha(args.source)
    payload["migration"]["diff_sha256"] = _sha(args.diff)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != rendered:
            raise SystemExit(f"{args.out} is missing or stale; regenerate with this script")
        print(f"verified {args.out}: {len(payload['entries'])} frozen decisions")
        return 0
    args.out.write_text(rendered)
    print(
        f"wrote {args.out}: {len(payload['entries'])} decisions, "
        f"{len(payload['migration']['rekeyed'])} proven rekeys, "
        f"score {payload['scorecard']['hit']}/{payload['scorecard']['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
