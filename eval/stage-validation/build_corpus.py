"""Build the stage-validation corpus: N cases per dataset, each with a natural-language goal.

The stage-validation suite (SV-0 .. SV-4) needs client-shaped inputs: a plain-language condition
plus ground-truth intervals on the source clock. Every dataset here already carries both, in a
different shape; this module normalizes them into one case schema and pins each case to the exact
media bytes it was derived from.

This is an INTERNAL validation corpus. It is deliberately NOT the licensed publication manifest
(`eval/golden-v1/MANIFEST.json`) and must never be presented as one.

Case schema
-----------
    case_id        stable "<dataset>::<video>::<ordinal>"
    dataset        charades | proactiveqa | perception_test | omnipro | repcount
    video_path     repo-relative media path
    video_sha256   content hash of the exact media bytes
    duration_s     source duration where the dataset declares it
    goal_text      the natural-language condition handed to the Watch
    goal_kind      transition | duration | count | state  (drives which primitive should compile)
    positives      [[start_s, end_s], ...] ground-truth positive intervals on the source clock
    provenance     the raw annotation fields the case was derived from
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING = REPO_ROOT / "eval" / "bench-staging"
DEFAULT_OUT = REPO_ROOT / "eval" / "stage-validation" / "corpus.json"

#: Fixed so corpus selection is reproducible across machines and checkouts. Selection must never
#: depend on filesystem ordering, which differs between the main worktree and a launch worktree.
SELECTION_SEED = 20260726


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _media_fps(path: Path) -> float | None:
    """Frame rate straight from the container, via ffprobe — no decode, no guessed constant.

    RepCount annotates cycle bounds as frame indices, so a wrong fps silently shifts every
    ground-truth interval rather than failing loudly.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den or 1)
        return fps if fps > 0 else None
    except (subprocess.SubprocessError, ValueError, ZeroDivisionError):
        return None


def _media_index(dataset: str, *, by_name: bool = False) -> dict[str, Path]:
    """Map annotation id -> media path for one dataset."""
    root = STAGING / dataset
    return {(p.name if by_name else p.stem): p for p in root.rglob("*.mp4")}


# --------------------------------------------------------------------------------------------
# charades — actions are already natural language with explicit [start, end] on the source clock
# --------------------------------------------------------------------------------------------
def build_charades(limit: int) -> list[dict[str, Any]]:
    rows = json.loads((STAGING / "charades/raw/selection.json").read_text())
    media = _media_index("charades")
    rng = random.Random(SELECTION_SEED)
    cases: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda r: r["id"]):
        path = media.get(row["id"])
        if path is None:
            continue
        actions = [a for a in row.get("actions") or [] if len(a) == 3 and a[2] > a[1]]
        if not actions:
            continue
        # One goal per case keeps recall attributable to a single condition. Pick the longest
        # action: short ones are dominated by annotation boundary error, which would show up as a
        # judge failure rather than the annotation artifact it is.
        text, start, end = max(actions, key=lambda a: a[2] - a[1])
        # Charades action intervals routinely overrun the declared clip length (the annotators
        # worked past the encoded end). An interval extending beyond the media would score as
        # unreachable recall — clamp to the source rather than blaming the runtime for it.
        length = row.get("length")
        if isinstance(length, (int, float)) and length > 0:
            end = min(float(end), float(length))
            start = min(float(start), float(end))
        if not end > start:
            continue
        cases.append(
            {
                "case_id": f"charades::{row['id']}::0",
                "dataset": "charades",
                "video_path": _rel(path),
                "duration_s": row.get("length"),
                "goal_text": str(text).strip(),
                "goal_kind": "duration",
                "positives": [[round(float(start), 3), round(float(end), 3)]],
                "provenance": {"scene": row.get("scene"), "n_actions": len(actions)},
            }
        )
    rng.shuffle(cases)
    return cases[:limit]


# --------------------------------------------------------------------------------------------
# proactiveqa — assistant replies carry `related_timespan`; the reply text is the condition
# --------------------------------------------------------------------------------------------
def build_proactiveqa(limit: int) -> list[dict[str, Any]]:
    rows = json.loads((STAGING / "proactiveqa/raw/EGO_subset.json").read_text())
    media = _media_index("proactiveqa", by_name=True)
    rng = random.Random(SELECTION_SEED)
    cases: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda r: str(r.get("video"))):
        path = media.get(row.get("video"))
        if path is None:
            continue
        answers = [
            a
            for a in row.get("answer") or []
            if isinstance(a.get("related_timespan"), list)
            and len(a["related_timespan"]) == 2
            and a["related_timespan"][1] > a["related_timespan"][0]
            and str(a.get("content") or "").strip()
        ]
        if not answers:
            continue
        first = answers[0]
        start, end = first["related_timespan"]
        cases.append(
            {
                "case_id": f"proactiveqa::{path.stem}::0",
                "dataset": "proactiveqa",
                "video_path": _rel(path),
                "duration_s": row.get("duration"),
                "goal_text": str(first["content"]).strip(),
                "goal_kind": "transition",
                "positives": [[round(float(start), 3), round(float(end), 3)]],
                "provenance": {"type": first.get("type"), "n_steps": len(answers)},
            }
        )
    rng.shuffle(cases)
    return cases[:limit]


# --------------------------------------------------------------------------------------------
# omnipro — questions are natural language; task name maps onto the expected Watch primitive
# --------------------------------------------------------------------------------------------
_OMNIPRO_KIND = {
    "cumulative_counting": "count",
    "dedup_counting": "count",
    "snapshot_counting": "count",
    "instant_event_alert": "transition",
    "semantic_condition_alert": "transition",
    "realtime_state_monitor": "state",
    "explicit_target_grounding": "state",
    "event_narration": "transition",
    "sequential_step_instruction": "transition",
}


def build_omnipro(limit: int) -> list[dict[str, Any]]:
    rows = json.loads((STAGING / "omnipro/raw/benchmark_subset.json").read_text())
    media = _media_index("omnipro")
    rng = random.Random(SELECTION_SEED)
    cases: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda r: str(r.get("id"))):
        path = media.get(row.get("video_id"))
        if path is None:
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        # `question_time_sec` is when the QUESTION is asked — it is 0 for every row in this subset
        # and is NOT ground truth. The answerable moments live in `ground_truth[*].trigger_time_sec`,
        # one entry per occurrence. Using the former yields a silent near-zero recall that reads as
        # a runtime failure.
        triggers = [
            g.get("trigger_time_sec")
            for g in (row.get("ground_truth") or [])
            if isinstance(g.get("trigger_time_sec"), (int, float))
        ]
        positives = [[round(float(t), 3), round(float(t), 3)] for t in sorted(triggers)]
        if not positives:
            continue
        cases.append(
            {
                "case_id": f"omnipro::{row['video_id']}::{row['id'].rsplit('::', 1)[-1]}",
                "dataset": "omnipro",
                "video_path": _rel(path),
                "duration_s": row.get("duration"),
                "goal_text": question,
                "goal_kind": _OMNIPRO_KIND.get(row.get("task"), "transition"),
                "positives": positives,
                "provenance": {
                    "task": row.get("task"),
                    "ground_truth": row.get("ground_truth"),
                    "audio_dependency": row.get("audio_dependency"),
                    "instant_annotation": True,
                },
            }
        )
    rng.shuffle(cases)
    return cases[:limit]


# --------------------------------------------------------------------------------------------
# perception_test — action_localisation segments; annotation set is fetched/cached separately
# --------------------------------------------------------------------------------------------
def _perception_test_annotations() -> dict[str, Any]:
    """Full cached action annotations when available, else the tiny checked-in sample.

    ``sample.json`` holds 8 videos; the real valid-split set (~5.3k videos) is fetched and cached
    by ``eval.bench.perception_test.load_annotations``. Preferring the cache is what lets this
    dataset reach its quota — the sample alone caps the corpus at 8.
    """
    try:  # the benchmark harness lives beside the runtime, not inside the wheel
        import sys

        harness = REPO_ROOT / "packages" / "percept-vision"
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        from eval.bench import perception_test as pt  # noqa: PLC0415

        full = pt.load_annotations("action")
        if full:
            return full
    except Exception:  # noqa: BLE001 - corpus build must not require network
        pass
    sample = STAGING / "perception_test/raw/sample.json"
    return json.loads(sample.read_text()) if sample.is_file() else {}


def build_perception_test(limit: int) -> list[dict[str, Any]]:
    rows = _perception_test_annotations()
    if not rows:
        return []
    media = _media_index("perception_test")
    rng = random.Random(SELECTION_SEED)
    cases: list[dict[str, Any]] = []

    for video_id, row in sorted(rows.items()):
        path = media.get(video_id)
        if path is None:
            continue
        meta = row.get("metadata") or {}
        fps = float(meta.get("frame_rate") or 0.0)
        segments = [s for s in row.get("action_localisation") or [] if s.get("label")]
        if not segments or fps <= 0:
            continue
        seg = max(segments, key=lambda s: float(s.get("timestamps", [0, 0])[1]) - float(s.get("timestamps", [0, 0])[0]))
        start, end = seg.get("timestamps", [None, None])
        if start is None or end is None or not end > start:
            continue
        # Perception Test stores action_localisation timestamps in MICROSECONDS, not seconds: a
        # 33.7-second clip carries values like 18541983. Feeding those through as seconds puts
        # every positive interval past the end of the media and scores recall as a flat zero.
        start, end = float(start) / 1e6, float(end) / 1e6
        duration_s = float(meta.get("num_frames", 0)) / fps
        if duration_s and end > duration_s * 1.05:
            continue  # unit assumption failed for this row — drop rather than guess
        cases.append(
            {
                "case_id": f"perception_test::{video_id}::0",
                "dataset": "perception_test",
                "video_path": _rel(path),
                "duration_s": round(duration_s, 3) if fps else None,
                "goal_text": str(seg["label"]).strip(),
                "goal_kind": "duration",
                "positives": [[round(float(start), 3), round(float(end), 3)]],
                "provenance": {
                    "is_camera_moving": meta.get("is_camera_moving"),
                    "resolution": meta.get("resolution"),
                    "n_segments": len(segments),
                },
            }
        )
    rng.shuffle(cases)
    return cases[:limit]


# --------------------------------------------------------------------------------------------
# repcount — ground-truth rep counts + per-cycle bounds; the natural Stage-2 counting corpus
# --------------------------------------------------------------------------------------------
#: RepCount action_type -> a condition phrased the way a client would state it.
#:
#: KEYED ON A NORMALIZED action name (see ``_normalize_action``).  The dataset is inconsistent about
#: separators — it ships BOTH ``frontraise`` and ``front_raise``, ``jump_jack`` and ``jumpjack`` —
#: and this table originally held only the unseparated spellings.  Every underscored row therefore
#: missed the lookup and fell through to the ``f"a person completes a {action}"`` fallback, which
#: put the raw dataset token into the condition a live judge was asked to evaluate: 12 of the 20
#: repcount goals literally read "a person completes a jump_jack".  That is not a phrasing blemish,
#: it is a different experiment — the judge was shown a token no client would ever write, on the one
#: goal_kind (``count``) the dataset exists to test.  Normalizing the key makes both spellings hit
#: the same intended sentence.
_REPCOUNT_PHRASING = {
    "situp": "a person completes a sit-up",
    "pushup": "a person completes a push-up",
    "pullup": "a person completes a pull-up",
    "squat": "a person completes a squat",
    "jumpjack": "a person completes a jumping jack",
    "benchpressing": "a person completes a bench press repetition",
    "frontraise": "a person completes a front raise",
    "pommelhorse": "a person completes a pommel horse circle",
    "battlerope": "a person completes a battle rope wave",
    "bespin": "a person completes a spin",
    "soccer": "a person completes a soccer juggle",
}


def _normalize_action(action: str) -> str:
    """``"front_raise"``, ``"Front-Raise"``, ``"frontraise"`` -> ``"frontraise"``."""
    return "".join(ch for ch in str(action).lower() if ch.isalnum())


def repcount_goal_text(action: str) -> str:
    """The client-shaped condition for one RepCount action type.

    The fallback still exists for an action the table has never seen, but it now spells the token
    out as words instead of pasting a snake_case identifier into a natural-language condition.
    """
    action = str(action or "").strip()
    known = _REPCOUNT_PHRASING.get(_normalize_action(action))
    if known:
        return known
    spelled = " ".join(part for part in action.replace("-", "_").split("_") if part)
    return f"a person completes a {spelled or action}"


def build_repcount(limit: int) -> list[dict[str, Any]]:
    lance_path = STAGING / "repcount/raw/data/test.lance"
    if not lance_path.exists():
        return []
    try:
        import lance  # noqa: PLC0415 - optional analysis dependency
    except ImportError:
        return []

    dataset = lance.dataset(str(lance_path))
    rows = dataset.to_table(
        columns=["video_id", "action_type", "count", "cycle_bounds_json", "video_ext"]
    ).to_pylist()
    media = _media_index("repcount")
    rng = random.Random(SELECTION_SEED)
    cases: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda r: str(r["video_id"])):
        path = media.get(str(row["video_id"]))
        if path is None:
            continue
        count = row.get("count")
        if not (isinstance(count, (int, str)) and str(count).isdigit()):
            continue
        count = int(count)
        # Bounded reps: a 2-rep clip cannot distinguish a counter from a lucky edge detector, and a
        # 60-rep clip is dominated by drift rather than by the counting rule under test.
        if not 4 <= count <= 20:
            continue
        try:
            bounds = json.loads(row.get("cycle_bounds_json") or "[]")
        except json.JSONDecodeError:
            bounds = []
        action = str(row.get("action_type") or "").strip()
        # cycle_bounds are FRAME indices in flat (start, end) pairs. Convert here using the media's
        # OWN fps — leaving `positives` empty makes every consumer read a flat-zero recall and
        # blame the runtime, which is precisely the failure this corpus exists to rule out.
        fps = _media_fps(path)
        positives: list[list[float]] = []
        if fps and len(bounds) >= 2:
            for index in range(0, len(bounds) - 1, 2):
                start_f, end_f = bounds[index], bounds[index + 1]
                if isinstance(start_f, (int, float)) and isinstance(end_f, (int, float)) and end_f > start_f:
                    positives.append([round(start_f / fps, 3), round(end_f / fps, 3)])
        cases.append(
            {
                "case_id": f"repcount::{row['video_id']}::0",
                "dataset": "repcount",
                "video_path": _rel(path),
                "duration_s": None,
                "goal_text": repcount_goal_text(action),
                "goal_kind": "count",
                "positives": positives,
                "provenance": {
                    "action_type": action,
                    "expected_count": count,
                    "cycle_bounds_frames": bounds,
                    "bounds_unit": "frame_index",
                },
            }
        )
    rng.shuffle(cases)
    return cases[:limit]


BUILDERS = {
    "charades": build_charades,
    "proactiveqa": build_proactiveqa,
    "omnipro": build_omnipro,
    "perception_test": build_perception_test,
    "repcount": build_repcount,
}


def build(datasets: list[str], per_dataset: int, *, hash_media: bool = True) -> dict[str, Any]:
    corpus: list[dict[str, Any]] = []
    report: dict[str, Any] = {}

    for name in datasets:
        builder = BUILDERS.get(name)
        if builder is None:
            report[name] = {"requested": per_dataset, "built": 0, "note": "no builder"}
            continue
        cases = builder(per_dataset)
        for case in cases:
            path = REPO_ROOT / case["video_path"]
            case["video_sha256"] = sha256_file(path) if hash_media else None
        corpus.extend(cases)
        report[name] = {
            "requested": per_dataset,
            "built": len(cases),
            "short_by": max(0, per_dataset - len(cases)),
        }

    return {
        "version": 1,
        "kind": "stage-validation-corpus",
        "note": "INTERNAL validation corpus. Not the licensed publication manifest.",
        "selection_seed": SELECTION_SEED,
        "per_dataset": per_dataset,
        "coverage": report,
        "n_cases": len(corpus),
        "cases": corpus,
    }


# --------------------------------------------------------------------------------------------
# corpus VERSIONING
# --------------------------------------------------------------------------------------------
#: The revision ``corpus.json`` holds.  It is FROZEN: the file is never edited in place, because
#: every result already on disk was scored against those exact bytes and must stay re-derivable
#: from them.  A correction ships as a NEW file with a new version, selected explicitly through
#: ``PERCEPT_SV_CORPUS``, and the version is stamped into every artifact's meta so a table can
#: never silently mix revisions.
FROZEN_VERSION = "1.0"
NEXT_VERSION = "1.1"


def revise(frozen: dict[str, Any], *, version: str = NEXT_VERSION) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the next corpus revision from the frozen one, and report exactly what moved.

    DERIVED, not rebuilt.  Re-running the builders would re-select from live dataset files and a
    changed snapshot, ffprobe result or annotation cache could silently alter case MEMBERSHIP —
    at which point v1.0 and v1.1 would no longer be the same experiment and no result could be
    carried across them.  Applying the corrected phrasing to the frozen cases guarantees the two
    revisions differ in the goal wording and in nothing else, which is the only way a v1.0 result
    and a v1.1 result are comparable at all.
    """
    cases = json.loads(json.dumps(frozen["cases"]))  # deep copy: the frozen doc is never mutated
    changes: list[dict[str, Any]] = []
    for case in cases:
        if case.get("dataset") != "repcount":
            continue
        action = str((case.get("provenance") or {}).get("action_type") or "")
        before = case["goal_text"]
        after = repcount_goal_text(action)
        if after != before:
            case["goal_text"] = after
            changes.append({"case_id": case["case_id"], "action_type": action,
                            "goal_text_before": before, "goal_text_after": after})

    revised = dict(frozen)
    revised["cases"] = cases
    revised["corpus_version"] = version
    revised["derived_from"] = {
        "corpus_version": frozen.get("corpus_version") or FROZEN_VERSION,
        "sha256": hashlib.sha256(
            json.dumps(frozen, indent=2, sort_keys=True).encode()).hexdigest(),
        "method": "goal_text re-derived from provenance.action_type; every other field copied "
                  "verbatim, so case membership, media pins and ground truth are identical",
    }
    revised["changelog"] = [
        f"repcount goal_text: {len(changes)} of "
        f"{sum(1 for c in cases if c.get('dataset') == 'repcount')} cases. _REPCOUNT_PHRASING was "
        "keyed on unseparated action names only, so every underscored spelling missed the lookup "
        "and the raw dataset token reached the judge (\"a person completes a jump_jack\")."
    ]

    def _kinds(doc_cases: list[dict[str, Any]]) -> dict[str, str]:
        return {c["case_id"]: c["goal_kind"] for c in doc_cases}

    before_kinds, after_kinds = _kinds(frozen["cases"]), _kinds(cases)
    diff = {
        "from_version": frozen.get("corpus_version") or FROZEN_VERSION,
        "to_version": version,
        "n_cases_before": len(frozen["cases"]),
        "n_cases_after": len(cases),
        "case_membership_identical": [c["case_id"] for c in frozen["cases"]] == [c["case_id"] for c in cases],
        "goal_text_changed": len(changes),
        "goal_text_changes": changes,
        "goal_text_changes_by_action": dict(sorted(
            {a: sum(1 for c in changes if c["action_type"] == a)
             for a in {c["action_type"] for c in changes}}.items())),
        # Reported even though it is expected to be empty: a phrasing fix that silently re-typed a
        # goal would change which PRIMITIVE compiles, and that is a different experiment.
        "goal_kind_changed": sorted(k for k in after_kinds if before_kinds.get(k) != after_kinds[k]),
        "goal_kind_diff_note": "goal_kind is derived from the dataset/task, never from goal_text; "
                               "an empty list is the expected and required result.",
        "positives_identical": all(
            a["positives"] == b["positives"] for a, b in zip(frozen["cases"], cases)),
        "video_sha256_identical": all(
            a.get("video_sha256") == b.get("video_sha256") for a, b in zip(frozen["cases"], cases)),
    }
    return revised, diff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="cases per dataset")
    ap.add_argument("--datasets", default=",".join(BUILDERS), help="comma-separated dataset names")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-hash", action="store_true", help="skip media hashing (faster, unpinned)")
    ap.add_argument("--revise", action="store_true",
                    help=f"derive corpus v{NEXT_VERSION} from the frozen v{FROZEN_VERSION} at "
                         f"--out instead of rebuilding from the datasets. Writes "
                         f"corpus.v{NEXT_VERSION}.json and corpus.version-diff.json; never touches "
                         f"the frozen file.")
    args = ap.parse_args()

    if args.revise:
        frozen = json.loads(args.out.read_text())
        revised, diff = revise(frozen)
        out = args.out.with_name(f"{args.out.stem}.v{NEXT_VERSION}.json")
        out.write_text(json.dumps(revised, indent=2, sort_keys=True))
        diff_path = args.out.with_name(f"{args.out.stem}.version-diff.json")
        diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True))
        print(f"corpus v{NEXT_VERSION} -> {out}")
        print(f"  frozen v{FROZEN_VERSION} left untouched at {args.out}")
        print(f"  case membership identical: {diff['case_membership_identical']}")
        print(f"  positives identical:       {diff['positives_identical']}")
        print(f"  goal_text changed:         {diff['goal_text_changed']} "
              f"{diff['goal_text_changes_by_action']}")
        print(f"  goal_kind changed:         {diff['goal_kind_changed'] or 'none'}")
        print(f"  diff -> {diff_path}")
        return 0

    manifest = build(
        [d.strip() for d in args.datasets.split(",") if d.strip()],
        args.n,
        hash_media=not args.no_hash,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"corpus -> {args.out}")
    print(f"  cases: {manifest['n_cases']}")
    for name, info in manifest["coverage"].items():
        flag = "" if not info.get("short_by") else f"   SHORT BY {info['short_by']}"
        print(f"  {name:<18} {info['built']:>3}/{info['requested']}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
