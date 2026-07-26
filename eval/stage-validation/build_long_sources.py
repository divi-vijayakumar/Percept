"""Concatenate same-goal corpus cases into long sources with one Watch per goal.

Short clips cannot exercise the long-running behaviour the runtime actually claims: repeated
occurrence, re-arming after an occurrence closes, episode continuity across gaps, refractory state,
and steady-state memory. Prior work established that episode memory in particular is meaningless to
evaluate on short clips — it is for long-running, multi-occurrence horizons.

So this builds the missing corpus shape: N clips that share a goal are concatenated into ONE source
and bound to ONE Watch, whose ground truth is the union of the constituent positives shifted by each
clip's cumulative offset. A goal that fires once per clip should now fire N times on one source,
with N-1 re-arms — which is precisely what SV-4.3 measures.

Clips are re-encoded to a uniform profile because the ffmpeg concat demuxer requires matching codec
parameters, and the corpus spans five datasets with different encodings.

Output: `long_sources/<goal_key>.mp4` plus `long_sources.json` describing each source, its Watch
goal, and the shifted ground-truth intervals.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "eval" / "stage-validation" / "corpus.json"
OUT_DIR = REPO_ROOT / "eval" / "stage-validation" / "long_sources"
OUT_MANIFEST = REPO_ROOT / "eval" / "stage-validation" / "long_sources.json"

#: Uniform re-encode profile. Deliberately modest: these sources exist to exercise temporal
#: cognition over a long horizon, not to test decode fidelity, and the machine building them is
#: disk-constrained.
SCALE_WIDTH = 640
TARGET_FPS = 15
CRF = 28


def _norm_goal_key(case: dict[str, Any]) -> str:
    """Group key for 'the same goal'.

    RepCount groups on ``action_type`` because its phrasing is generated from that field anyway;
    the raw values are inconsistently spelled (``front_raise`` vs ``frontraise``), so underscores
    are stripped before comparison. Everything else groups on the normalized goal text.
    """
    if case["dataset"] == "repcount":
        action = str(case["provenance"].get("action_type") or "")
        return f"repcount::{re.sub(r'[^a-z0-9]', '', action.lower())}"
    text = re.sub(r"[^a-z0-9 ]", "", case["goal_text"].lower()).strip()
    slug = re.sub(r"\s+", "_", text)
    return f"{case['dataset']}::{slug}"


def probe_duration(path: Path) -> float | None:
    """Exact container duration, which is what the offset arithmetic must use."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def _reencode(src: Path, dst: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale={SCALE_WIDTH}:-2,fps={TARGET_FPS}",
         "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
         "-an",  # audio is dropped: these sources exercise the video temporal path
         str(dst)],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0 and dst.is_file()


def build_group(key: str, cases: list[dict[str, Any]], work: Path) -> dict[str, Any] | None:
    """Concatenate one goal group and shift its ground truth onto the joined clock."""
    work.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    members: list[dict[str, Any]] = []
    offset = 0.0

    for index, case in enumerate(sorted(cases, key=lambda c: c["case_id"])):
        src = REPO_ROOT / case["video_path"]
        if not src.is_file():
            continue
        part = work / f"{index:03d}.mp4"
        if not _reencode(src, part):
            continue
        # Measure the RE-ENCODED part, not the original: fps/scale normalization can shift the
        # container duration, and an offset computed from the source would drift across members.
        duration = probe_duration(part)
        if duration is None:
            continue

        shifted = [
            [round(start + offset, 3), round(end + offset, 3)]
            for start, end in case.get("positives") or []
        ]
        members.append(
            {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "offset_s": round(offset, 3),
                "duration_s": round(duration, 3),
                "positives_shifted": shifted,
                "expected_count": case["provenance"].get("expected_count"),
            }
        )
        parts.append(part)
        offset += duration

    if len(parts) < 2:  # a "long source" of one clip measures nothing new
        return None

    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{p.name}'\n" for p in parts))
    output = OUT_DIR / f"{key.replace('::', '__')}.mp4"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", str(output)],
        capture_output=True, text=True, timeout=600,
    )
    if joined.returncode != 0 or not output.is_file():
        return None

    exemplar = cases[0]
    return {
        "goal_key": key,
        "video_path": str(output.relative_to(REPO_ROOT)),
        "total_duration_s": round(offset, 3),
        "n_members": len(members),
        # One Watch for the whole source: the point of the exercise is that a single goal must
        # re-arm across N occurrences rather than firing once and latching.
        "watch": {"goal_text": exemplar["goal_text"], "goal_kind": exemplar["goal_kind"]},
        "expected_occurrences": len(members),
        "positives": [span for m in members for span in m["positives_shifted"]],
        "members": members,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-group", type=int, default=2, help="minimum clips to form a long source")
    ap.add_argument("--max-groups", type=int, default=6, help="cap on sources built (disk/time)")
    ap.add_argument("--work", type=Path,
                    default=Path("/private/tmp/claude-501/-Users-dhivyavijayakumar-git-Percept-sdk"
                                 "/3d8c8120-cb64-4426-8af7-8eb60fc4a085/scratchpad/longsrc"))
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in corpus["cases"]:
        groups[_norm_goal_key(case)].append(case)

    eligible = sorted(
        ((k, v) for k, v in groups.items() if len(v) >= args.min_group),
        key=lambda kv: -len(kv[1]),
    )[: args.max_groups]

    print(f"eligible goal groups: {len(eligible)}")
    built: list[dict[str, Any]] = []
    for key, cases in eligible:
        print(f"  building {key} ({len(cases)} clips) ...", flush=True)
        entry = build_group(key, cases, args.work / key.replace("::", "__"))
        if entry is None:
            print("    skipped (encode/concat failed or too few usable clips)")
            continue
        built.append(entry)
        print(f"    -> {entry['video_path']}  {entry['total_duration_s']}s  "
              f"{entry['expected_occurrences']} occurrences")

    OUT_MANIFEST.write_text(
        json.dumps(
            {"version": 1, "kind": "stage-validation-long-sources",
             "profile": {"scale_width": SCALE_WIDTH, "fps": TARGET_FPS, "crf": CRF},
             "n_sources": len(built), "sources": built},
            indent=2, sort_keys=True,
        )
    )
    print(f"\nmanifest -> {OUT_MANIFEST}  ({len(built)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
