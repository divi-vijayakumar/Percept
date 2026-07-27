#!/usr/bin/env python3
"""Author a validation corpus with :class:`percept.conductor.PlanConductor` and FREEZE the result.

Answers "does the conductor pick the right primitive?" against a labelled goal set. It makes real
provider calls, so it is a deliberate, out-of-band run — never part of the test suite. What it LANDS
is the frozen artifact: a content-addressed record of every authoring decision, plus the scorecard
those decisions produced. A benchmark (or the offline replay test) then binds to the artifact and
re-runs the exact same plans with no provider call and no drift.

    python tools/author_corpus.py \
        --corpus <corpus.json> \
        --out eval/primitive-authoring/corpus-100.frozen.json

The corpus must be a JSON object with a ``cases`` list (or a bare list) whose entries carry
``goal_text`` and a ``goal_kind`` label.

Re-running with an existing ``--out`` re-uses every decision already pinned there and only authors
the goals that are missing (the cache is content-addressed by prompt version + model + goal text, so
a prompt edit correctly invalidates everything).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "percept-vision"))

from percept.conductor import (  # noqa: E402
    CONDUCTOR_MODEL,
    PROMPT_VERSION,
    AuthoringCache,
    PlanConductor,
    goal_kind_of,
    selection_accuracy,
)


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    cases = payload["cases"] if isinstance(payload, dict) else payload
    return [
        {"case_id": c.get("case_id"), "goal_text": c["goal_text"], "goal_kind": c["goal_kind"]}
        for c in cases
    ]


async def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=CONDUCTOR_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="author only the first N cases (a smoke run)")
    args = ap.parse_args()

    cases = _load_cases(args.corpus)
    if args.limit:
        cases = cases[: args.limit]

    # Resume from whatever is already pinned; frozen=False because this script is the WRITER.
    cache = AuthoringCache(frozen=False, model=args.model)
    if args.out.exists():
        cache = AuthoringCache.load(args.out, frozen=False)

    conductor = PlanConductor(model=args.model, cache=cache)
    authored = await conductor.author_all([c["goal_text"] for c in cases], concurrency=args.concurrency)

    scorecard = selection_accuracy(zip(authored, (c["goal_kind"] for c in cases)))
    payload = cache.to_dict()
    payload["corpus"] = {
        "path": str(args.corpus),
        "sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "n_cases": len(cases),
    }
    payload["prompt_version"] = PROMPT_VERSION
    payload["model"] = args.model
    payload["cases"] = [
        {
            "case_id": case["case_id"],
            "goal_text": case["goal_text"],
            "goal_kind": case["goal_kind"],
            "authored_type": primitive.behavior_type,
            "predicted_kind": goal_kind_of(primitive),
            "origin": primitive.origin,
        }
        for case, primitive in zip(cases, authored)
    ]
    payload["scorecard"] = scorecard
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"model={args.model} prompt={PROMPT_VERSION} n={scorecard['total']}")
    print(f"overall {scorecard['hit']}/{scorecard['total']} = {scorecard['accuracy']:.3f}")
    for kind, bucket in sorted(scorecard["per_kind"].items()):
        print(f"  {kind:<11} {bucket['hit']:>3}/{bucket['total']:<3} = {bucket['accuracy']:.3f}")
    print("confusion (label -> predicted):")
    for label, row in sorted(scorecard["confusion"].items()):
        print(f"  {label:<11} {dict(sorted(row.items()))}")
    print(f"frozen artifact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
