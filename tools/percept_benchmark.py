#!/usr/bin/env python3
"""Reproducible local benchmark runner for Percept.

The built-in ``percept-bench scorecard`` is intentionally simple. This wrapper is
for longer local/prod-ish runs: stable data/run folders, track-level subprocess
isolation, retries, progress logs, and scorecard collection from partial results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "packages" / "percept-vision"
HARNESS = ROOT / "packages" / "percept-harness"
CONTRACTS = ROOT / "packages" / "percept-contracts"

DEFAULT_DATA_DIR = ROOT / "eval" / "bench-staging"
DEFAULT_RUNS_DIR = ROOT / "eval" / "local-benchmark" / "runs"
DEFAULT_CACHE_DIR = ROOT / "eval" / "local-benchmark" / "cache"
DEFAULT_MANIFEST = ROOT / "eval" / "golden-v1" / "MANIFEST.json"

TRACK_MODULES = {
    "relational": "eval.bench.track_relational",
    "counting": "eval.bench.track_counting",
    "acoustic": "eval.bench.track_acoustic_pt",
    "timeliness": "eval.bench.track_timeliness",
    "edge": "eval.bench.track_edge_recall",
    "escalation": "eval.bench.track_escalation_policy",
    "cuespeak": "eval.bench.track_cuespeak",
    "timespeak": "eval.bench.track_timespeak",
}
ALIASES = {
    "all": tuple(TRACK_MODULES),
    "no-counting": tuple(k for k in TRACK_MODULES if k != "counting"),
    "stable": tuple(k for k in TRACK_MODULES if k != "counting"),
}

CHARADES_VIDEO_ZIP = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
PROACTIVEQA_EGO_ZIP = "https://huggingface.co/datasets/wangyueqian/ProactiveVideoQA/resolve/main/EGO/videos.zip"
SAMPLE_SEED = os.environ.get("PERCEPT_BENCH_SAMPLE_SEED", "percept-golden-v1")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_rank(*parts: object) -> str:
    key = "|".join(str(p) for p in (SAMPLE_SEED, *parts))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def _bench_paths() -> dict[str, Path]:
    _load_dotenv()
    return {
        "data_dir": _path_from_env("PERCEPT_BENCH_DATA_DIR", DEFAULT_DATA_DIR),
        "runs_dir": _path_from_env("PERCEPT_BENCH_RUNS_DIR", DEFAULT_RUNS_DIR),
        "cache_dir": _path_from_env("PERCEPT_BENCH_CACHE_DIR", DEFAULT_CACHE_DIR),
    }


def _pythonpath() -> str:
    parts = [str(SDK), str(HARNESS), str(CONTRACTS)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _env() -> dict[str, str]:
    _load_dotenv()
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    if env.get("GOOGLE_AISTUDIO_API_KEY") and not env.get("GEMINI_API_KEY"):
        env["GEMINI_API_KEY"] = env["GOOGLE_AISTUDIO_API_KEY"]
    paths = _bench_paths()
    env.setdefault("PERCEPT_BENCH_DATA_DIR", _display_path(paths["data_dir"]))
    env.setdefault("PERCEPT_BENCH_RUNS_DIR", _display_path(paths["runs_dir"]))
    env.setdefault("PERCEPT_BENCH_CACHE_DIR", _display_path(paths["cache_dir"]))
    return env


def _manifest_ref(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{_display_path(path)} sha256:{digest}"


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "?"


def _json_safe_result(result: Any) -> dict[str, Any]:
    data = asdict(result)
    data["cells"] = {f"{cfg}:{metric}": value for (cfg, metric), value in result.cells.items()}
    return data


def _track_result_from_json(data: dict[str, Any]):
    from eval.bench.types import TrackResult

    cells = {}
    for key, value in data.get("cells", {}).items():
        cfg, _, metric = key.partition(":")
        cells[(cfg, metric)] = value
    data = dict(data)
    data["cells"] = cells
    return TrackResult(**data)


def _resolve_tracks(raw: str, exclude: str = "") -> list[str]:
    selected: list[str] = []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        names = ALIASES.get(part, (part,))
        for name in names:
            if name not in TRACK_MODULES:
                raise SystemExit(f"unknown track {name!r}; choices: {', '.join(TRACK_MODULES)}")
            if name not in selected:
                selected.append(name)
    excluded = {p.strip() for p in exclude.split(",") if p.strip()}
    unknown = excluded - set(TRACK_MODULES)
    if unknown:
        raise SystemExit(f"unknown excluded track(s): {', '.join(sorted(unknown))}")
    return [name for name in selected if name not in excluded]


def _write_event(run_dir: Path, event: dict[str, Any]) -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with (run_dir / "events.jsonl").open("a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _data_inventory(data_dir: Path) -> dict[str, Any]:
    entries = []
    for rel in [
        "perception_test/raw",
        "charades/raw",
        "omnipro",
        "repcount/raw",
        "proactiveqa/raw",
    ]:
        path = data_dir / rel
        files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
        size = sum(p.stat().st_size for p in files)
        entries.append(
            {
                "name": rel,
                "path": _display_path(path),
                "exists": path.exists(),
                "files": len(files),
                "size_mb": round(size / 1_000_000, 1),
            }
        )
    return {"data_dir": _display_path(data_dir), "entries": entries}


def _copy_remotezip_members(*, url: str, members: list[tuple[str, Path]], label: str) -> int:
    from remotezip import RemoteZip

    done = 0
    with RemoteZip(url) as rz:
        names = set(rz.namelist())
        for idx, (member, dest) in enumerate(members, 1):
            if dest.exists() and dest.stat().st_size > 0:
                done += 1
                continue
            if member not in names:
                print(f"  {label} missing in zip: {member}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(rz.read(member))
            except OSError:
                if dest.exists() and dest.stat().st_size == 0:
                    dest.unlink()
                raise
            done += 1
            if done % 25 == 0 or idx == len(members):
                print(f"  {label}: cached {done}/{len(members)}")
    return done


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_prepare(_args: argparse.Namespace) -> int:
    paths = _bench_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    inventory = _data_inventory(paths["data_dir"])
    inventory_path = paths["cache_dir"] / "data_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print("prepared benchmark workspace")
    for key, path in paths.items():
        print(f"  {key}: {_display_path(path)}")
    print(f"  inventory: {_display_path(inventory_path)}")
    for entry in inventory["entries"]:
        status = "ok" if entry["exists"] else "missing"
        print(f"  {status:<7} {entry['path']:<48} {entry['files']:>5} files {entry['size_mb']:>8} MB")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Materialize benchmark media caches via the existing SDK fetcher."""
    paths = _bench_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "eval.bench.fetch_all",
        "--videos",
        str(args.videos),
        "--repcount",
        str(args.repcount),
    ]
    print(f"fetching benchmark data into {_display_path(paths['data_dir'])}")
    return subprocess.call(cmd, cwd=ROOT, env=_env())


def _charades_class_map(raw: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (raw / "Charades" / "Charades_v1_classes.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls, _, desc = line.partition(" ")
        out[cls] = desc.strip()
    return out


def _charades_rows(raw: Path, n: int) -> list[dict[str, Any]]:
    cls = _charades_class_map(raw)
    csv_path = raw / "Charades" / "Charades_v1_test.csv"
    rows = list(csv.DictReader(csv_path.open()))
    candidates = []
    for row in rows:
        if row.get("verified") != "Yes" or not row.get("actions"):
            continue
        actions = []
        for part in row["actions"].split(";"):
            bits = part.split()
            if len(bits) != 3 or bits[0] not in cls:
                continue
            actions.append([cls[bits[0]], float(bits[1]), float(bits[2])])
        if not actions:
            continue
        candidates.append(
            {
                "id": row["id"],
                "length": float(row["length"]),
                "scene": row.get("scene"),
                "quality": int(row.get("quality") or 0),
                "relevance": int(row.get("relevance") or 0),
                "actions": actions,
            }
        )
    candidates.sort(key=lambda r: _stable_rank("charades", r["id"]))
    return candidates[:n]


def cmd_fetch_charades(args: argparse.Namespace) -> int:
    raw = _bench_paths()["data_dir"] / "charades" / "raw"
    video_dir = raw / "Charades_v1_480"
    raw.mkdir(parents=True, exist_ok=True)
    rows = _charades_rows(raw, args.n)
    members = [(f"Charades_v1_480/{row['id']}.mp4", video_dir / f"{row['id']}.mp4") for row in rows]
    print(f"fetching Charades videos into {_display_path(video_dir)}")
    cached = _copy_remotezip_members(url=CHARADES_VIDEO_ZIP, members=members, label="charades")
    local_rows = [row for row in rows if (video_dir / f"{row['id']}.mp4").exists()]
    (raw / "selection.json").write_text(json.dumps(local_rows, indent=2, sort_keys=True) + "\n")
    print(f"Charades selection -> {_display_path(raw / 'selection.json')} ({len(local_rows)} rows, {cached} cached)")
    return 0 if len(local_rows) == len(rows) else 2


def _load_or_fetch_proactiveqa_anno(raw: Path) -> list[dict[str, Any]]:
    anno = raw / "EGO_anno.json"
    if anno.exists():
        return json.loads(anno.read_text())
    from huggingface_hub import hf_hub_download

    src = Path(hf_hub_download("wangyueqian/ProactiveVideoQA", "EGO/anno.json", repo_type="dataset"))
    raw.mkdir(parents=True, exist_ok=True)
    anno.write_bytes(src.read_bytes())
    return json.loads(anno.read_text())


def _proactiveqa_rows(raw: Path, n: int) -> list[dict[str, Any]]:
    rows = _load_or_fetch_proactiveqa_anno(raw)
    rows = [r for r in rows if r.get("video") and r.get("answer")]
    rows.sort(key=lambda r: _stable_rank("proactiveqa", r.get("question_id", r["video"])))
    return rows[:n]


def cmd_fetch_proactiveqa(args: argparse.Namespace) -> int:
    raw = _bench_paths()["data_dir"] / "proactiveqa" / "raw"
    video_dir = raw / "EGO_videos"
    rows = _proactiveqa_rows(raw, args.n)
    members = [(f"videos/{row['video']}", video_dir / row["video"]) for row in rows]
    print(f"fetching ProactiveVideoQA EGO videos into {_display_path(video_dir)}")
    cached = _copy_remotezip_members(url=PROACTIVEQA_EGO_ZIP, members=members, label="proactiveqa")
    local_rows = [row for row in rows if (video_dir / row["video"]).exists()]
    (raw / "EGO_subset.json").write_text(json.dumps(local_rows, indent=2, sort_keys=True) + "\n")
    print(f"ProactiveQA subset -> {_display_path(raw / 'EGO_subset.json')} ({len(local_rows)} rows, {cached} cached)")
    return 0 if len(local_rows) == len(rows) else 2


def cmd_index_proactiveqa(_args: argparse.Namespace) -> int:
    raw = _bench_paths()["data_dir"] / "proactiveqa" / "raw"
    video_dir = raw / "EGO_videos"
    rows = _load_or_fetch_proactiveqa_anno(raw)
    local = [row for row in rows if (video_dir / row["video"]).exists() and (video_dir / row["video"]).stat().st_size > 0]
    local.sort(key=lambda r: _stable_rank("proactiveqa", r.get("question_id", r["video"])))
    (raw / "EGO_subset.json").write_text(json.dumps(local, indent=2, sort_keys=True) + "\n")
    print(f"ProactiveQA local subset -> {_display_path(raw / 'EGO_subset.json')} ({len(local)} rows)")
    return 0


def _pt_candidate_rows(*, action_n: int, sound_n: int) -> list[dict[str, Any]]:
    sys.path[:0] = [str(SDK), str(HARNESS), str(CONTRACTS)]
    from eval.bench import perception_test as pt

    rows: list[dict[str, Any]] = []

    def add(task: str, events: list[dict[str, Any]]) -> None:
        ann_path = pt.STAGE / pt.ANN_ZIP[task]
        ann_sha = _sha256(ann_path) if ann_path.exists() else ""
        data = pt.load_annotations(task)
        key = pt.TASK_KEY[task]
        for idx, ev in enumerate(events):
            media_path = pt.VIDEO_DIR / f"{ev['video_id']}.mp4"
            if not media_path.exists():
                continue
            video = data.get(ev["video_id"], {})
            segments = video.get(key, [])
            rows.append(
                {
                    "id": f"pt-valid-{task}-{idx:04d}-{ev['video_id']}",
                    "source": "google-deepmind/perception_test valid split",
                    "license": "CC-BY-4.0",
                    "split": "test",
                    "track": "perception-test-action" if task == "action" else "perception-test-acoustic",
                    "task": pt.TASK_KEY[task],
                    "video_id": ev["video_id"],
                    "event_label": ev.get("label"),
                    "onset_s": round(float(ev.get("onset_s", 0.0)), 3),
                    "end_s": round(float(ev.get("end_s", 0.0)), 3),
                    "media_path": _display_path(media_path),
                    "media_sha256": _sha256(media_path),
                    "annotation_path": _display_path(ann_path),
                    "annotation_sha256": ann_sha,
                    "segments": len(segments),
                    "is_visible": ev.get("is_visible"),
                }
            )

    add("action", pt.sample_events("action", action_n))
    add("sound", pt.sample_events("sound", sound_n, is_visible=0))
    return rows


def _charades_candidate_rows(n: int) -> list[dict[str, Any]]:
    raw = _bench_paths()["data_dir"] / "charades" / "raw"
    selection = raw / "selection.json"
    if not selection.exists():
        return []
    rows = json.loads(selection.read_text())[:n]
    out = []
    ann = raw / "Charades" / "Charades_v1_test.csv"
    license_path = raw / "Charades" / "license.txt"
    for row in rows:
        media = raw / "Charades_v1_480" / f"{row['id']}.mp4"
        if not media.exists():
            continue
        out.append(
            {
                "id": f"charades-{row['id']}",
                "source": "Charades v1 480p test split",
                "license": "Charades non-commercial evaluation license",
                "split": "test",
                "track": "charades-miou",
                "task": "temporal_action_localization",
                "video_id": row["id"],
                "scene": row.get("scene"),
                "media_path": _display_path(media),
                "media_sha256": _sha256(media),
                "annotation_path": _display_path(ann),
                "annotation_sha256": _sha256(ann) if ann.exists() else "",
                "license_path": _display_path(license_path),
                "actions": len(row.get("actions", [])),
                "duration_s": row.get("length"),
            }
        )
    return out


def _proactiveqa_candidate_rows(n: int) -> list[dict[str, Any]]:
    raw = _bench_paths()["data_dir"] / "proactiveqa" / "raw"
    subset = raw / "EGO_subset.json"
    if not subset.exists():
        return []
    rows = json.loads(subset.read_text())[:n]
    ann = raw / "EGO_anno.json"
    out = []
    for row in rows:
        media = raw / "EGO_videos" / row["video"]
        if not media.exists():
            continue
        out.append(
            {
                "id": f"proactiveqa-ego-{row['question_id']}",
                "source": "ProactiveVideoQA EGO subset",
                "license": "ProactiveVideoQA dataset license unspecified in local cache",
                "split": "test",
                "track": "proactiveqa-pauc",
                "task": "proactive_video_qa",
                "video_id": row["video"],
                "question_id": row["question_id"],
                "media_path": _display_path(media),
                "media_sha256": _sha256(media),
                "annotation_path": _display_path(ann),
                "annotation_sha256": _sha256(ann) if ann.exists() else "",
                "answer_turns": len(row.get("answer", [])),
                "duration_s": row.get("duration"),
            }
        )
    return out


def _repcount_candidate_rows(n: int) -> list[dict[str, Any]]:
    sys.path[:0] = [str(SDK), str(HARNESS), str(CONTRACTS)]
    from eval.bench import repcount as rc

    out = []
    for row in rc.sample_repcount(n):
        media = rc.fetch_video(row["index"], row["video_id"])
        if not media.exists():
            continue
        out.append(
            {
                "id": f"repcount-a-{row['video_id']}",
                "source": "RepCount-A via lmms-lab-eval/repcounta-lance",
                "license": "RepCount-A reference assets",
                "split": "test",
                "track": "repcount-a",
                "task": "repetition_counting",
                "scope": "pose-friendly subset selected before scoring; not full RepCount-A",
                "video_id": row["video_id"],
                "media_path": _display_path(media),
                "media_sha256": _sha256(media),
                "action": row["action"],
                "count": row["count"],
                "lance_index": row["index"],
            }
        )
    return out


def cmd_manifest_candidate(args: argparse.Namespace) -> int:
    paths = _bench_paths()
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    rows = _pt_candidate_rows(action_n=args.pt_action, sound_n=args.pt_sound)
    rows.extend(_charades_candidate_rows(args.charades))
    rows.extend(_proactiveqa_candidate_rows(args.proactiveqa))
    rows.extend(_repcount_candidate_rows(args.repcount))
    manifest = {
        "version": "golden-v1-candidate",
        "status": "candidate",
        "created": datetime.now(timezone.utc).date().isoformat(),
        "release_rule": "Promote only after human review, split freeze, no modeled cells, and release audit.",
        "tracks": [
            {
                "id": "perception-test-action",
                "source": "Perception Test CC-BY-4.0 valid split",
                "status": "candidate_public",
                "metrics": ["event_recall", "p_pauc", "reaction_p50", "reaction_p90"],
            },
            {
                "id": "perception-test-acoustic",
                "source": "Perception Test CC-BY-4.0 valid split",
                "status": "candidate_public",
                "metrics": ["recall", "fusion_fp", "recall_flip"],
            },
            {
                "id": "charades-miou",
                "source": "Charades v1 test split",
                "status": "reference_noncommercial",
                "metrics": ["miou", "precision", "recall"],
            },
            {
                "id": "proactiveqa-pauc",
                "source": "ProactiveVideoQA EGO",
                "status": "candidate_reference",
                "metrics": ["pauc@0", "pauc@0.5", "pauc@1", "interruptions"],
            },
            {
                "id": "repcount-a",
                "source": "RepCount-A",
                "status": "candidate_reference",
                "metrics": ["obo", "mae"],
            },
        ],
        "splits": {"dev": [], "test": rows},
        "notes": [
            "Generated from local PERCEPT_BENCH_DATA_DIR cache.",
            "Candidate is ignored by git until reviewed and promoted into eval/golden-v1/MANIFEST.json.",
            f"Sampling seed: {SAMPLE_SEED}. Dataset rows are deterministic hash samples unless a track declares a narrower scope.",
            "Perception Test rows are one event per video from valid annotations.",
            "Charades rows are deterministic hash samples from verified test videos with actions; Charades is non-commercial reference data.",
            "ProactiveQA rows are deterministic hash samples from EGO annotations with answers.",
            "RepCount rows use the current pose-friendly/count-bounded scoring subset; do not present as full RepCount-A until the runner supports all action types.",
        ],
    }
    out = Path(args.out) if args.out else paths["cache_dir"] / "golden-v1-candidate.json"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["track"]] = counts.get(row["track"], 0) + 1
    print(f"candidate manifest -> {_display_path(out)}")
    for track, count in sorted(counts.items()):
        print(f"  {track}: {count} rows")
    return 0


def cmd_run_track(args: argparse.Namespace) -> int:
    _load_dotenv()
    module = importlib.import_module(TRACK_MODULES[args.track])
    result = module.score(live=args.live, n=args.n)
    payload = _json_safe_result(result)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _attempt_track(
    *,
    run_dir: Path,
    track: str,
    live: bool,
    n: int,
    attempts: int,
    timeout_s: int,
) -> dict[str, Any]:
    result_path = run_dir / "results" / f"{track}.json"
    log_path = run_dir / "logs" / f"{track}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run-track",
        "--track",
        track,
        "--n",
        str(n),
        "--out",
        str(result_path),
    ]
    if live:
        cmd.append("--live")

    for attempt in range(1, attempts + 1):
        started = time.time()
        _write_event(run_dir, {"event": "track_start", "track": track, "attempt": attempt})
        try:
            with log_path.open("a") as log:
                log.write(f"\n### attempt {attempt} {datetime.now(timezone.utc).isoformat()}\n")
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_s if timeout_s > 0 else None,
                )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            with log_path.open("a") as log:
                log.write(f"\nTIMEOUT after {timeout_s}s\n")
            returncode = 124
        elapsed = round(time.time() - started, 2)
        if returncode == 0 and result_path.exists():
            _write_event(run_dir, {"event": "track_ok", "track": track, "attempt": attempt, "elapsed_s": elapsed})
            return {"track": track, "ok": True, "attempts": attempt, "elapsed_s": elapsed, "result": str(result_path)}
        _write_event(
            run_dir,
            {
                "event": "track_failed",
                "track": track,
                "attempt": attempt,
                "returncode": returncode,
                "elapsed_s": elapsed,
                "log": str(log_path),
            },
        )
        if attempt < attempts:
            time.sleep(min(30, 2 ** attempt))
    return {"track": track, "ok": False, "attempts": attempts, "log": str(log_path)}


def cmd_run(args: argparse.Namespace) -> int:
    paths = _bench_paths()
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or _now()
    run_dir = paths["runs_dir"] / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    tracks = _resolve_tracks(args.tracks, args.exclude)
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "live": args.live,
        "n": args.n,
        "tracks": tracks,
        "attempts": args.attempts,
        "workers": args.workers,
        "data_dir": _display_path(paths["data_dir"]),
        "cache_dir": _display_path(paths["cache_dir"]),
        "manifest": _display_path(Path(args.manifest).resolve()) if args.manifest else None,
        "commit": _commit(),
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (run_dir / "data_inventory.json").write_text(json.dumps(_data_inventory(paths["data_dir"]), indent=2) + "\n")
    _write_event(run_dir, {"event": "run_start", **meta})

    print(f"run: {_display_path(run_dir)}")
    print(f"tracks: {', '.join(tracks)}")
    print(f"mode: {'live' if args.live else 'offline'} · n={args.n} · workers={args.workers} · attempts={args.attempts}")

    outcomes: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _attempt_track,
                run_dir=run_dir,
                track=track,
                live=args.live,
                n=args.n,
                attempts=args.attempts,
                timeout_s=args.timeout_s,
            ): track
            for track in tracks
        }
        total = len(futures)
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            outcome = future.result()
            outcomes.append(outcome)
            status = "ok" if outcome["ok"] else "failed"
            print(f"[{done}/{total}] {outcome['track']}: {status}")

    (run_dir / "outcomes.json").write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    _write_event(run_dir, {"event": "run_done", "outcomes": outcomes})

    collect_args = argparse.Namespace(
        run_dir=str(run_dir),
        manifest=args.manifest,
        mode=args.mode,
        out=str(run_dir / "scorecard.txt"),
        device=args.device,
        fps=args.fps,
        seeds=args.seeds,
        require_release=False,
    )
    rc = cmd_collect(collect_args)
    failed = [o["track"] for o in outcomes if not o["ok"]]
    if failed:
        print(f"failed tracks: {', '.join(failed)}")
        return 2
    return rc


def cmd_collect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    result_dir = run_dir / "results"
    if not result_dir.exists():
        raise SystemExit(f"missing result dir: {result_dir}")

    sys.path[:0] = [str(SDK), str(HARNESS), str(CONTRACTS)]
    from eval.bench import card

    results = []
    for path in sorted(result_dir.glob("*.json")):
        results.append(_track_result_from_json(json.loads(path.read_text())))
    if not results:
        raise SystemExit(f"no track results found in {result_dir}")

    manifest = Path(args.manifest).resolve() if args.manifest else DEFAULT_MANIFEST
    manifest_ref = _manifest_ref(manifest)
    backend = "gemini-2.5-flash" if _run_meta(run_dir).get("live") else "fake (offline stub)"
    auroc = next((r.cells.get(("core", "auroc")) for r in results if r.track == "relational"), None)
    rendered = card.render_card(
        results,
        mode=args.mode,
        backend=backend,
        commit=_run_meta(run_dir).get("commit", _commit()),
        manifest=manifest_ref,
        fps=args.fps,
        device=args.device,
        seeds=args.seeds,
        auroc=auroc,
    )
    out = Path(args.out) if args.out else run_dir / "scorecard.txt"
    out.write_text(rendered)
    print(rendered)
    print(f"scorecard written -> {_display_path(out)}")
    if args.require_release and card.stamp_of(args.mode, results, manifest_ref) != "RELEASE":
        return 1
    return 0


def _run_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, run, and collect local Percept benchmark runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="create local benchmark folders and data inventory").set_defaults(func=cmd_prepare)

    fetch = sub.add_parser("fetch", help="pre-cache benchmark data into PERCEPT_BENCH_DATA_DIR")
    fetch.add_argument("--videos", type=int, default=12, help="Perception Test videos to fetch per task")
    fetch.add_argument("--repcount", type=int, default=12, help="RepCount videos to materialize")
    fetch.set_defaults(func=cmd_fetch)

    manifest = sub.add_parser("manifest-candidate", help="write ignored content-hashed candidate golden manifest")
    manifest.add_argument("--pt-action", type=int, default=200, help="Perception Test action rows")
    manifest.add_argument("--pt-sound", type=int, default=100, help="Perception Test invisible-sound rows")
    manifest.add_argument("--charades", type=int, default=100, help="Charades rows from local selection")
    manifest.add_argument("--proactiveqa", type=int, default=100, help="ProactiveVideoQA EGO rows from local subset")
    manifest.add_argument("--repcount", type=int, default=100, help="RepCount-A rows from local sampler")
    manifest.add_argument("--out", default=None)
    manifest.set_defaults(func=cmd_manifest_candidate)

    charades = sub.add_parser("fetch-charades", help="range-fetch selected Charades videos and write selection.json")
    charades.add_argument("--n", type=int, default=100, help="number of Charades test videos")
    charades.set_defaults(func=cmd_fetch_charades)

    proactive = sub.add_parser("fetch-proactiveqa", help="range-fetch selected ProactiveVideoQA EGO videos")
    proactive.add_argument("--n", type=int, default=100, help="number of ProactiveVideoQA EGO videos")
    proactive.set_defaults(func=cmd_fetch_proactiveqa)

    pindex = sub.add_parser("index-proactiveqa", help="rewrite EGO_subset.json from locally cached videos")
    pindex.set_defaults(func=cmd_index_proactiveqa)

    run = sub.add_parser("run", help="run benchmark tracks with retries and per-track logs")
    run.add_argument("--live", action="store_true", help="use live Gemini-backed track scorers")
    run.add_argument("--n", type=int, default=8, help="sample size per track")
    run.add_argument("--tracks", default="all", help="comma list or alias: all, stable, no-counting")
    run.add_argument("--exclude", default="", help="comma list of tracks to skip")
    run.add_argument("--workers", type=int, default=2, help="parallel track subprocesses")
    run.add_argument("--attempts", type=int, default=2, help="attempts per track")
    run.add_argument("--timeout-s", type=int, default=1800, help="timeout per track attempt; 0 disables")
    run.add_argument("--run-id", default=None, help="stable run id/folder name")
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest stamped on the scorecard")
    run.add_argument("--mode", default="sampled", choices=["sampled", "full"], help="scorecard mode")
    run.add_argument("--device", default="local", help="device/runtime descriptor for scorecard")
    run.add_argument("--fps", default="1", help="sampling/capture fps descriptor for scorecard")
    run.add_argument("--seeds", type=int, default=1, help="number of seeds represented")
    run.set_defaults(func=cmd_run)

    collect = sub.add_parser("collect", help="collect a scorecard from an existing run directory")
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    collect.add_argument("--mode", default="sampled", choices=["sampled", "full"])
    collect.add_argument("--out", default=None)
    collect.add_argument("--device", default="local")
    collect.add_argument("--fps", default="1")
    collect.add_argument("--seeds", type=int, default=1)
    collect.add_argument("--require-release", action="store_true")
    collect.set_defaults(func=cmd_collect)

    internal = sub.add_parser("_run-track")
    internal.add_argument("--track", required=True, choices=sorted(TRACK_MODULES))
    internal.add_argument("--live", action="store_true")
    internal.add_argument("--n", type=int, required=True)
    internal.add_argument("--out", required=True)
    internal.set_defaults(func=cmd_run_track)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
