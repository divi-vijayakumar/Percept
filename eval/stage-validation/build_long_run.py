"""Build ONE long source out of the whole corpus, for the long-horizon state run.

``build_long_sources.py`` builds one source *per goal group* — N clips that share a goal, so a single
Watch has to re-arm N times.  That answers "does one goal survive repetition".  It cannot answer the
question this file exists for: **what happens to the shared state machinery when MANY watches run
concurrently over a horizon longer than any bound in the config.**  Entity tracks, the per-goal
episode logs, the event ledger and the judge scheduler are all process-wide; none of them is
observable on a 30-second clip with one goal armed.

So this concatenates the corpus into a SINGLE source and the runner arms every distinct goal against
it at once.

Three disciplines are inherited from ``build_long_sources.py`` and one is added:

1. Clips are re-encoded to a uniform profile, because the ffmpeg concat demuxer requires matching
   codec parameters and the corpus spans five datasets with five encodings.
2. Offsets are measured on the **re-encoded part**, never on the source: fps/scale normalisation
   shifts container duration, and a source-derived offset drifts cumulatively across ~100 members.
3. Positives are shifted onto the joined clock by the offset of the part they came from.
4. NEW — each member is **trimmed to ``--max-part-s``** (default 90 s).  The corpus is 181 minutes
   of media, dominated by proactiveqa (120 min over 20 clips); replay is real-time, so the whole
   corpus is a 3-hour run.  Trimming keeps every distinct video (and therefore every goal) in the
   source while landing the total near the approved wall budget.  A positive interval that starts
   after the cut is DROPPED and one that straddles it is CLIPPED, both counted in the manifest, so
   the ground truth never claims coverage the media does not contain.

Member ORDER is a deterministic round-robin over datasets rather than corpus order.  Contiguous
per-dataset blocks would put every occurrence of a goal inside one 10-minute stretch; round-robin
spreads each goal's occurrences across the full horizon, which is the only way the >1 h episode
window, the re-arm path and judge allocation over time are exercised at all.

Output: the concatenated mp4 plus a manifest describing the joined clock, under
``eval/stage-validation/long_sources/``.  Intermediate re-encoded parts are deleted after the
concat (the build machine is disk-constrained).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "eval" / "stage-validation" / "corpus.json"
OUT_DIR = REPO_ROOT / "eval" / "stage-validation" / "long_sources"

#: Uniform re-encode profile, identical in spirit to ``build_long_sources.py``: modest on purpose.
#: These sources exist to exercise long-horizon state, not decode fidelity.
#:
#: The geometry is FIXED (letterboxed), not ``scale=W:-2``.  ``-2`` preserves each source's aspect
#: ratio, so a corpus spanning 4:3 and 16:9 produces parts of DIFFERENT heights; the concat demuxer
#: accepts them, ``-c copy`` writes them, and the resulting mp4 decodes the first part and then
#: repeats one frozen frame for the rest of the run.  Nothing downstream complains — the container
#: duration, the frame count and the timestamps are all correct — so the only way this is caught is
#: by looking at the PIXELS, which ``--verify`` below now does before the manifest is written.
SCALE_WIDTH = 640
SCALE_HEIGHT = 360
TARGET_FPS = 15
CRF = 28
VFILTER = (
    f"scale={SCALE_WIDTH}:{SCALE_HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={SCALE_WIDTH}:{SCALE_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}"
)


def probe_duration(path: Path) -> float | None:
    """Exact container duration — the only thing the offset arithmetic is allowed to use."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def _reencode(src: Path, dst: Path, max_s: float) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", f"{max_s:.3f}",
         "-vf", VFILTER,
         "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-an",
         str(dst)],
        capture_output=True, text=True, timeout=900,
    )
    return result.returncode == 0 and dst.is_file()


def verify_content(path: Path, *, probes: int = 60) -> dict[str, Any]:
    """Decode across the whole joined clock and prove the PIXELS actually change.

    Every cheap check — container duration, frame count, timestamp monotonicity — passes on a
    concat that silently froze at the first part boundary.  This is the check that does not: sample
    ``probes`` frames uniformly, signature each one, and report how many are distinct and how long
    the longest identical run is.  A source whose content is frozen for most of its length is
    unusable no matter how correct its metadata looks, and the builder refuses to publish it.
    """
    import cv2  # the [runtime] extra, already required by the source decoder

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        return {"ok": False, "reason": "no frames reported"}
    step = max(1, total // probes)
    sigs: list[tuple] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 16))
            sigs.append(tuple(int(v) for v in small.flatten()[::8]))
        i += 1
    cap.release()
    longest = 0
    run = 0
    for a, b in zip(sigs, sigs[1:]):
        run = run + 1 if a == b else 0
        longest = max(longest, run + 1 if run else 0)
    distinct = len(set(sigs))
    return {
        "ok": len(sigs) > 4 and distinct >= max(4, len(sigs) // 4),
        "frames_read": i,
        "frames_reported": total,
        "fps": fps,
        "probes": len(sigs),
        "distinct_signatures": distinct,
        "longest_identical_run_probes": longest,
        "longest_identical_run_s": (round(longest * step / fps, 1) if fps else None),
    }


def order_members(cases: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Distinct video paths in deterministic dataset round-robin order.

    Returns ``[(video_path, [cases that use it]), ...]``.  Several corpus cases can share one video
    (different goals over the same footage); the video is encoded ONCE and every case that uses it is
    shifted onto the same offset.
    """
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_path[case["video_path"]].append(case)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for path, group in by_path.items():
        by_dataset[group[0]["dataset"]].append(path)
    for paths in by_dataset.values():
        paths.sort()
    datasets = sorted(by_dataset)
    out: list[tuple[str, list[dict[str, Any]]]] = []
    i = 0
    while True:
        progressed = False
        for ds in datasets:
            paths = by_dataset[ds]
            if i < len(paths):
                out.append((paths[i], sorted(by_path[paths[i]], key=lambda c: c["case_id"])))
                progressed = True
        if not progressed:
            break
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-part-s", type=float, default=90.0,
                    help="hard trim per member; 0 = no trim (the full 181-minute corpus)")
    ap.add_argument("--max-total-s", type=float, default=0.0,
                    help="stop adding members once the joined clock passes this (0 = no cap)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--name", default="long_run_all")
    ap.add_argument("--work", type=Path,
                    default=Path("/private/tmp/claude-501/-Users-dhivyavijayakumar-git-Percept-sdk"
                                 "/3d8c8120-cb64-4426-8af7-8eb60fc4a085/scratchpad/longrun_parts"))
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    members_in = order_members(corpus["cases"])
    max_s = args.max_part_s if args.max_part_s > 0 else 1e9
    args.work.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build] {len(members_in)} distinct videos, trim {max_s:.0f}s, {args.jobs} encoders",
          flush=True)

    def encode(idx_path: tuple[int, str]) -> tuple[int, Path | None, float | None]:
        idx, rel = idx_path
        src = REPO_ROOT / rel
        part = args.work / f"{idx:04d}.mp4"
        if part.is_file() and probe_duration(part):
            return idx, part, probe_duration(part)
        if not src.is_file() or not _reencode(src, part, max_s):
            return idx, None, None
        return idx, part, probe_duration(part)

    t0 = time.monotonic()
    encoded: dict[int, tuple[Path | None, float | None]] = {}
    with ThreadPoolExecutor(args.jobs) as ex:
        for n, (idx, part, dur) in enumerate(
            ex.map(encode, [(i, p) for i, (p, _) in enumerate(members_in)]), start=1
        ):
            encoded[idx] = (part, dur)
            if n % 10 == 0:
                print(f"[build]   encoded {n}/{len(members_in)} ({time.monotonic() - t0:.0f}s)",
                      flush=True)

    # ---- lay the joined clock out on the RE-ENCODED durations --------------------------------
    parts: list[Path] = []
    members: list[dict[str, Any]] = []
    offset = 0.0
    positives_dropped = 0
    positives_clipped = 0
    skipped: list[str] = []
    for idx, (rel, cases) in enumerate(members_in):
        part, duration = encoded.get(idx, (None, None))
        if part is None or duration is None:
            skipped.append(rel)
            continue
        if args.max_total_s and offset + duration > args.max_total_s:
            continue
        member_cases = []
        for case in cases:
            shifted = []
            for start, end in case.get("positives") or []:
                if start >= duration:
                    positives_dropped += 1
                    continue
                if end > duration:
                    positives_clipped += 1
                    end = duration
                shifted.append([round(start + offset, 3), round(end + offset, 3)])
            member_cases.append({
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "goal_kind": case["goal_kind"],
                "goal_text": case["goal_text"],
                "positives_shifted": shifted,
                "n_positives_source": len(case.get("positives") or []),
            })
        members.append({
            "index": len(members),
            "video_path": rel,
            "offset_s": round(offset, 3),
            "duration_s": round(duration, 3),
            "trimmed": duration < (probe_duration(REPO_ROOT / rel) or duration) - 0.05,
            "cases": member_cases,
        })
        parts.append(part)
        offset += duration

    if len(parts) < 2:
        print("[build] refusing to build: fewer than two usable parts")
        return 1

    concat_list = args.work / f"{args.name}.concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    output = OUT_DIR / f"{args.name}.mp4"
    print(f"[build] concatenating {len(parts)} parts -> {output}", flush=True)
    joined = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", str(output)],
        capture_output=True, text=True, timeout=1800,
    )
    if joined.returncode != 0 or not output.is_file():
        print(f"[build] concat FAILED: {joined.stderr[:2000]}")
        return 1

    measured = probe_duration(output)
    verified = verify_content(output)
    print(f"[build] content verification: {verified}", flush=True)
    if not verified.get("ok"):
        print("[build] REFUSING to publish a manifest: the concatenated source's pixels do not "
              "change across the joined clock. A mismatched-geometry concat produces exactly this "
              "(correct duration, frozen content).")
        return 1

    goals: dict[str, dict[str, Any]] = {}
    for member in members:
        for case in member["cases"]:
            entry = goals.setdefault(case["goal_text"], {
                "goal_text": case["goal_text"],
                "goal_kinds": [],
                "datasets": [],
                "case_ids": [],
                "positives": [],
            })
            entry["goal_kinds"].append(case["goal_kind"])
            entry["datasets"].append(case["dataset"])
            entry["case_ids"].append(case["case_id"])
            entry["positives"].extend(case["positives_shifted"])
    for entry in goals.values():
        entry["goal_kind"] = max(set(entry["goal_kinds"]), key=entry["goal_kinds"].count)
        entry["dataset"] = max(set(entry["datasets"]), key=entry["datasets"].count)
        entry["positives"].sort()
        entry["n_occurrences"] = len(entry["positives"])
        del entry["goal_kinds"], entry["datasets"]

    manifest = {
        "version": 1,
        "kind": "stage-validation-single-long-source",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_path": str(output.relative_to(REPO_ROOT)),
        "profile": {"scale_width": SCALE_WIDTH, "scale_height": SCALE_HEIGHT,
                    "vfilter": VFILTER, "fps": TARGET_FPS, "crf": CRF,
                    "max_part_s": args.max_part_s, "audio": False},
        "content_verification": verified,
        "clock": {
            "sum_of_part_durations_s": round(offset, 3),
            "measured_container_duration_s": None if measured is None else round(measured, 3),
            "drift_s": None if measured is None else round(measured - offset, 3),
        },
        "n_members": len(members),
        "n_videos_skipped": len(skipped),
        "videos_skipped": skipped,
        "positives_dropped_past_trim": positives_dropped,
        "positives_clipped_at_trim": positives_clipped,
        "n_goals": len(goals),
        "goals": sorted(goals.values(), key=lambda g: g["goal_text"]),
        "members": members,
    }
    manifest_path = OUT_DIR / f"{args.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    if not args.keep_parts:
        shutil.rmtree(args.work, ignore_errors=True)

    print(f"[build] {output}  {offset / 60:.1f} min joined clock "
          f"(container {(measured or 0) / 60:.1f} min, drift {manifest['clock']['drift_s']}s)")
    print(f"[build] {len(members)} members, {len(goals)} distinct goals, "
          f"{sum(g['n_occurrences'] for g in goals.values())} GT occurrences")
    print(f"[build] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
