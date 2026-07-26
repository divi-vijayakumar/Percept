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
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "packages" / "percept-vision"

DEFAULT_DATA_DIR = ROOT / "eval" / "bench-staging"
DEFAULT_RUNS_DIR = ROOT / "eval" / "local-benchmark" / "runs"
DEFAULT_CACHE_DIR = ROOT / "eval" / "local-benchmark" / "cache"
DEFAULT_MANIFEST = ROOT / "eval" / "golden-v1" / "MANIFEST.json"

TRACK_MODULES = {
    "pt-action-e2e": "eval.bench.track_perception_test_e2e",
    "charades-e2e": "eval.bench.track_charades_e2e",
    "proactiveqa-e2e": "eval.bench.track_proactiveqa_e2e",
    "relational": "eval.bench.track_relational",
    "counting": "eval.bench.track_counting",
    "acoustic": "eval.bench.track_acoustic_pt",
    "timeliness": "eval.bench.track_timeliness",
    "edge": "eval.bench.track_edge_recall",
    "escalation": "eval.bench.track_escalation_policy",
    "cuespeak": "eval.bench.track_cuespeak",
    "timespeak": "eval.bench.track_timespeak",
}
SAFE_LIVE_TRACKS = ("pt-action-e2e", "edge", "escalation", "counting", "cuespeak", "timespeak")
LEGACY_MEDIA_DENSE_TRACKS = ("relational", "acoustic", "timeliness")
DERIVED_ARTIFACT_TRACKS = ("cuespeak", "timespeak")
ALIASES = {
    "all": tuple(TRACK_MODULES),
    "safe-live": SAFE_LIVE_TRACKS,
    "no-counting": tuple(k for k in SAFE_LIVE_TRACKS if k != "counting"),
    "stable": SAFE_LIVE_TRACKS,
    "legacy-all": tuple(TRACK_MODULES),
}

CHARADES_VIDEO_ZIP = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
PROACTIVEQA_EGO_ZIP = "https://huggingface.co/datasets/wangyueqian/ProactiveVideoQA/resolve/main/EGO/videos.zip"
SAMPLE_SEED = os.environ.get("PERCEPT_BENCH_SAMPLE_SEED", "percept-golden-v1")

# Front-load the pose checkpoint download (RTMO via MMPose) in a child that has the
# benchmark PYTHONPATH + certifi TLS bundle, so the counting track never blocks on it.
POSE_WARM_SNIPPET = (
    "from percept.harness.pose import ensure_pose_ready; "
    "print('pose provider ready:', ensure_pose_ready())"
)


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
    parts = [str(SDK)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _ensure_tls_certs(env: dict[str, str]) -> dict[str, str]:
    """Default SSL_CERT_FILE/REQUESTS_CA_BUNDLE to certifi's bundle.

    macOS python.org builds ship without a usable system CA chain, so torch-hub
    (urllib) RTMO checkpoint downloads fail with CERTIFICATE_VERIFY_FAILED. Point the
    download paths at certifi unless the caller already pinned a bundle.
    """
    if env.get("SSL_CERT_FILE") and env.get("REQUESTS_CA_BUNDLE"):
        return env
    try:
        import certifi
    except ModuleNotFoundError:
        return env
    bundle = certifi.where()
    env.setdefault("SSL_CERT_FILE", bundle)
    env.setdefault("REQUESTS_CA_BUNDLE", bundle)
    return env


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
    _ensure_tls_certs(env)
    return env


def _backend_label(live: bool) -> str:
    if not live:
        return "fake (offline stub)"
    _load_dotenv()
    return os.environ.get("PERCEPT_VISION_MODEL") or os.environ.get("PERCEPT_GEMINI_MODEL") or "gemini"


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


def _order_tracks_for_dependencies(tracks: list[str]) -> list[str]:
    """Run derived artifact tracks after their E2E source tracks when they share one run.

    CueSpeak/TimeSpeak read result JSON from PT/Charades/PVQA artifacts. If they execute before those
    source tracks, they can accidentally derive from older runs. Keep user selection but move derived
    tracks to the end so a sequential run is coherent.
    """

    derived = [name for name in tracks if name in DERIVED_ARTIFACT_TRACKS]
    base = [name for name in tracks if name not in DERIVED_ARTIFACT_TRACKS]
    return base + derived


def _guard_live_tracks(tracks: list[str]) -> None:
    blocked = [t for t in tracks if t in LEGACY_MEDIA_DENSE_TRACKS]
    if blocked and os.environ.get("PERCEPT_ALLOW_RAW_LIVE") != "1":
        raise SystemExit(
            "refusing live media-bearing dense/raw benchmark track(s): "
            + ", ".join(blocked)
            + ". Use E2E/edge tracks, or set PERCEPT_ALLOW_RAW_LIVE=1 for an explicit legacy diagnostic."
        )


def _guard_live_prerequisites(
    *,
    manifest: str | None,
    release: bool,
    n: int | None = None,
    tracks: list[str] | None = None,
    workers: int = 1,
    attempts: int = 1,
) -> None:
    """Fail before starting workers when provider spend/media prerequisites are absent."""
    sys.path[:0] = [str(SDK)]
    from eval.bench.run import _live_prerequisite_blockers

    blockers = _live_prerequisite_blockers(
        manifest=manifest,
        release=release,
        n=n,
        tracks=tracks or [],
    )
    if os.environ.get("PERCEPT_VISION_BACKEND", "gemini") == "openrouter":
        if workers != 1:
            blockers.append("OpenRouter live benchmark requires --workers 1 for one budget ledger")
        if attempts != 1:
            blockers.append("OpenRouter live benchmark requires --attempts 1; retries are budgeted per request")
    if blockers:
        raise SystemExit("BENCHMARK LIVE GATE: BLOCKED — " + "; ".join(blockers))


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
    sys.path[:0] = [str(SDK)]
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
    sys.path[:0] = [str(SDK)]
    from eval.bench import repcount as rc

    out = []
    for row in rc.sample_repcount_unbiased(n):
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
                "scope": "unbiased action sample; edge-boundary detector(s) feed the Percept core count; no raw VLM baseline",
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
            "RepCount rows use an unbiased action sample. Counting scorecards must not use a raw per-frame VLM baseline; publish detector/provider failures and unresolved rates.",
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


def _prefetch_pose(run_dir: Path) -> bool:
    """Warm the configured pose provider checkpoint up front. Returns True on success.

    A failure is non-fatal: the counting track will still run and report the real
    unresolved reason. We only front-load so a slow/broken download surfaces early.
    """
    provider = os.environ.get("PERCEPT_POSE_PROVIDER", "rtmo")
    log_path = run_dir / "logs" / "pose-prefetch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_event(run_dir, {"event": "pose_prefetch_start", "provider": provider})
    started = time.time()
    with log_path.open("a") as log:
        log.write(f"\n### pose prefetch ({provider}) {datetime.now(timezone.utc).isoformat()}\n")
        proc = subprocess.run(
            [sys.executable, "-c", POSE_WARM_SNIPPET],
            cwd=ROOT,
            env=_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = round(time.time() - started, 2)
    ok = proc.returncode == 0
    _write_event(
        run_dir,
        {
            "event": "pose_prefetch_done",
            "provider": provider,
            "ok": ok,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "log": str(log_path),
        },
    )
    return ok


def cmd_prefetch_pose(_args: argparse.Namespace) -> int:
    """Download/warm the configured pose provider checkpoint (RTMO by default)."""
    provider = os.environ.get("PERCEPT_POSE_PROVIDER", "rtmo")
    print(f"warming pose provider: {provider}")
    proc = subprocess.run([sys.executable, "-c", POSE_WARM_SNIPPET], cwd=ROOT, env=_env(), text=True)
    return proc.returncode


def cmd_run_track(args: argparse.Namespace) -> int:
    _load_dotenv()
    if args.live:
        _guard_live_tracks([args.track])
        _guard_live_prerequisites(
            manifest=args.manifest,
            release=False,
            n=args.n,
            tracks=[args.track],
        )
    if args.manifest:
        os.environ["PERCEPT_BENCH_MANIFEST"] = str(Path(args.manifest).resolve())
    module = importlib.import_module(TRACK_MODULES[args.track])
    result = module.score(live=args.live, n=args.n)
    payload = _json_safe_result(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    return 0


def _existing_result_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        result = _track_result_from_json(json.loads(path.read_text()))
        return bool(result.cells) and any(value is not None for value in result.cells.values())
    except Exception:
        return False


def _attempt_track(
    *,
    run_dir: Path,
    track: str,
    live: bool,
    n: int,
    attempts: int,
    timeout_s: int,
    resume: bool,
    track_index: int,
    start_delay_s: float,
    manifest: str,
) -> dict[str, Any]:
    result_path = run_dir / "results" / f"{track}.json"
    log_path = run_dir / "logs" / f"{track}.log"
    if resume and _existing_result_ok(result_path):
        _write_event(run_dir, {"event": "track_skip_existing", "track": track, "result": str(result_path)})
        return {"track": track, "ok": True, "attempts": 0, "elapsed_s": 0, "result": str(result_path), "skipped": True}

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
        "--manifest",
        manifest,
    ]
    if live:
        cmd.append("--live")

    if start_delay_s > 0:
        time.sleep(track_index * start_delay_s)

    for attempt in range(1, attempts + 1):
        started = time.time()
        _write_event(run_dir, {"event": "track_start", "track": track, "attempt": attempt})
        try:
            with log_path.open("a") as log:
                log.write(f"\n### attempt {attempt} {datetime.now(timezone.utc).isoformat()}\n")
                env = _env()
                env["PERCEPT_BENCH_RUN_DIR"] = str(run_dir)
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
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


def _background_cmd(args: argparse.Namespace, run_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--run-id",
        run_id,
        "--n",
        str(args.n),
        "--tracks",
        args.tracks,
        "--workers",
        str(args.workers),
        "--attempts",
        str(args.attempts),
        "--timeout-s",
        str(args.timeout_s),
        "--manifest",
        args.manifest,
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--fps",
        args.fps,
        "--seeds",
        str(args.seeds),
        "--start-delay-s",
        str(args.start_delay_s),
    ]
    if args.live:
        cmd.append("--live")
    if args.exclude:
        cmd.extend(["--exclude", args.exclude])
    cmd.append("--resume" if args.resume else "--no-resume")
    cmd.append("--prefetch-pose" if args.prefetch_pose else "--no-prefetch-pose")
    return cmd


def cmd_run(args: argparse.Namespace) -> int:
    paths = _bench_paths()
    tracks = _order_tracks_for_dependencies(_resolve_tracks(args.tracks, args.exclude))
    if args.live:
        _guard_live_tracks(tracks)
        _guard_live_prerequisites(
            manifest=args.manifest,
            release=args.mode == "full",
            n=args.n,
            tracks=tracks,
            workers=args.workers,
            attempts=args.attempts,
        )
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or _now()
    run_dir = paths["runs_dir"] / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    if args.background and os.environ.get("PERCEPT_BENCH_BACKGROUND_CHILD") != "1":
        log_path = run_dir / "benchmark.log"
        cmd = _background_cmd(args, run_id)
        env = _env()
        env["PERCEPT_BENCH_BACKGROUND_CHILD"] = "1"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        payload = {
            "run_id": run_id,
            "pid": proc.pid,
            "cmd": cmd,
            "cmd_display": shlex.join(cmd),
            "log": _display_path(log_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "background.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"background run: {_display_path(run_dir)}")
        print(f"pid: {proc.pid}")
        print(f"log: {_display_path(log_path)}")
        print(f"resume/collect: python3 tools/percept_benchmark.py collect --run-dir {_display_path(run_dir)}")
        return 0

    existing_meta = _run_meta(run_dir) if args.resume else {}
    meta = {
        "run_id": run_id,
        "created_at": existing_meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "resumed_at": datetime.now(timezone.utc).isoformat() if existing_meta else None,
        "live": args.live,
        "n": args.n,
        "tracks": tracks,
        "attempts": args.attempts,
        "workers": args.workers,
        "resume": args.resume,
        "start_delay_s": args.start_delay_s,
        "data_dir": _display_path(paths["data_dir"]),
        "cache_dir": _display_path(paths["cache_dir"]),
        "manifest": _display_path(Path(args.manifest).resolve()) if args.manifest else None,
        "mode": args.mode,
        "device": args.device,
        "fps": args.fps,
        "seeds": args.seeds,
        "commit": _commit(),
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (run_dir / "data_inventory.json").write_text(json.dumps(_data_inventory(paths["data_dir"]), indent=2) + "\n")
    _write_event(run_dir, {"event": "run_start", **meta})

    print(f"run: {_display_path(run_dir)}")
    print(f"tracks: {', '.join(tracks)}")
    print(
        f"mode: {'live' if args.live else 'offline'} · n={args.n} · workers={args.workers} "
        f"· attempts={args.attempts} · resume={args.resume} · start_delay_s={args.start_delay_s}"
    )

    pose_needed = (
        args.live and "counting" in tracks and os.environ.get("PERCEPT_ENABLE_POSE_COUNT") == "1"
    )
    if pose_needed and args.prefetch_pose:
        provider = os.environ.get("PERCEPT_POSE_PROVIDER", "rtmo")
        print(f"prefetching pose model ({provider}) ...")
        ok = _prefetch_pose(run_dir)
        print(f"pose prefetch: {'ok' if ok else 'failed (counting track will report the unresolved reason)'}")

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
                resume=args.resume,
                track_index=idx,
                start_delay_s=args.start_delay_s,
                manifest=args.manifest,
            ): track
            for idx, track in enumerate(tracks)
        }
        total = len(futures)
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            outcome = future.result()
            outcomes.append(outcome)
            status = "skipped" if outcome.get("skipped") else "ok" if outcome["ok"] else "failed"
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
        allow_partial=False,
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

    sys.path[:0] = [str(SDK)]
    from eval.bench import card
    from eval.bench.validate import (
        ValidationError,
        manifest_runner_map,
        validate_golden_manifest,
    )

    manifest = Path(args.manifest).resolve() if args.manifest else DEFAULT_MANIFEST
    try:
        manifest_data = validate_golden_manifest(manifest, release=args.require_release)
    except ValidationError as exc:
        raise SystemExit(f"benchmark collection manifest audit failed: {exc}") from exc
    runner_map = manifest_runner_map(manifest_data)
    if not runner_map:
        raise SystemExit(
            "pinned manifest has no explicit runners mapping; collection qualification is BLOCKED"
        )
    declared_tracks = {str(row["id"]) for row in manifest_data["tracks"]}
    required_tracks = {
        str(row["id"]) for row in manifest_data["tracks"] if row.get("required") is True
    }
    meta = _run_meta(run_dir)
    sample_n = meta.get("n")
    if args.mode == "sampled" and sample_n is not None and (
        isinstance(sample_n, bool) or not isinstance(sample_n, int) or sample_n <= 0
    ):
        raise SystemExit(f"sampled run metadata has invalid n: {sample_n!r}")
    sample_limit = sample_n if args.mode == "sampled" and sample_n is not None else None
    manifest_cases = {
        track: [
            str(item["id"])
            for item in manifest_data["splits"]["test"]
            if item["track"] == track
        ][:sample_limit]
        for track in declared_tracks
    }

    results_by_runner = {}
    for path in sorted(result_dir.glob("*.json")):
        runner = path.stem
        try:
            results_by_runner[runner] = _track_result_from_json(json.loads(path.read_text()))
        except Exception as exc:
            raise SystemExit(f"invalid track result {path}: {exc}") from exc
    results = list(results_by_runner.values())
    if not results:
        raise SystemExit(f"no track results found in {result_dir}")

    expected_runners = meta.get("tracks")
    if not isinstance(expected_runners, list) or not expected_runners:
        raise SystemExit(f"run metadata has no non-empty tracks list: {run_dir / 'run.json'}")
    expected_runners = [str(runner) for runner in expected_runners]
    undeclared_selected = sorted(set(expected_runners) - set(runner_map))
    if undeclared_selected:
        raise SystemExit(
            "run selected runner(s) not declared by pinned manifest mapping: "
            + ", ".join(undeclared_selected)
        )
    selected_logical_tracks = {
        runner_map[runner]["manifest_track"] for runner in expected_runners
    }
    missing_required = sorted(required_tracks - selected_logical_tracks)
    if missing_required:
        raise SystemExit(
            "run selection omits required pinned-manifest track(s): "
            + ", ".join(missing_required)
        )
    actual_runners = list(results_by_runner)
    missing_tracks = sorted(set(expected_runners) - set(actual_runners))
    unexpected_tracks = sorted(set(actual_runners) - set(expected_runners))
    incomplete = []
    if missing_tracks:
        incomplete.append("missing=" + ",".join(missing_tracks))
    if unexpected_tracks:
        incomplete.append("unexpected=" + ",".join(unexpected_tracks))
    outcomes_path = run_dir / "outcomes.json"
    if outcomes_path.exists():
        outcomes = json.loads(outcomes_path.read_text())
        failed = sorted(str(row.get("track")) for row in outcomes if not row.get("ok"))
        if failed:
            incomplete.append("failed=" + ",".join(failed))
    if incomplete and (args.require_release or not getattr(args, "allow_partial", False)):
        raise SystemExit("benchmark collection incomplete: " + "; ".join(incomplete))
    execution_blockers = card.execution_blockers(results)
    if execution_blockers:
        raise SystemExit("benchmark collection has no usable result artifacts: " + "; ".join(execution_blockers))

    coverage_errors: list[str] = []
    for runner, result in results_by_runner.items():
        mapping = runner_map.get(runner)
        if mapping is None:
            coverage_errors.append(f"{runner}: result artifact runner is not manifest-declared")
            continue
        if result.track != mapping["result_track"]:
            coverage_errors.append(
                f"{runner}: TrackResult identity {result.track!r} does not match pinned "
                f"result_track {mapping['result_track']!r}"
            )
        logical_track = mapping["manifest_track"]
        expected_case_ids = manifest_cases[logical_track]
        coverage = result.normalized_coverage()
        actual_case_ids = [case_id for item in coverage for case_id in item.case_ids]
        expected_count = sum(item.expected for item in coverage)
        if not expected_case_ids:
            coverage_errors.append(
                f"{runner}/{logical_track}: selected runner has no pinned test cases"
            )
        if expected_count != len(expected_case_ids):
            coverage_errors.append(
                f"{runner}/{logical_track}: coverage expected={expected_count}, "
                f"pinned-manifest cases={len(expected_case_ids)}"
            )
        if len(actual_case_ids) != len(set(actual_case_ids)):
            coverage_errors.append(f"{runner}/{logical_track}: duplicate coverage case IDs")
        expected_case_set = set(expected_case_ids)
        missing_cases = sorted(expected_case_set - set(actual_case_ids))
        extra_cases = sorted(set(actual_case_ids) - expected_case_set)
        if missing_cases:
            coverage_errors.append(
                f"{runner}/{logical_track}: coverage missing manifest case IDs={missing_cases}"
            )
        if extra_cases:
            coverage_errors.append(
                f"{runner}/{logical_track}: coverage has undeclared case IDs={extra_cases}"
            )
    if coverage_errors:
        raise SystemExit(
            "benchmark collection coverage does not match pinned manifest: "
            + "; ".join(coverage_errors)
        )

    manifest_ref = _manifest_ref(manifest)
    backend = _backend_label(bool(_run_meta(run_dir).get("live")))
    auroc = next((r.cells.get(("core-replay", "auroc")) for r in results if r.track == "relational"), None)
    rendered = card.render_card(
        results,
        mode=args.mode,
        backend=backend,
        commit=meta.get("commit", _commit()),
        manifest=manifest_ref,
        fps=args.fps,
        device=args.device,
        seeds=args.seeds,
        auroc=auroc,
    )
    out = Path(args.out) if args.out else run_dir / "scorecard.txt"
    out.write_text(rendered)
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"scorecard artifact is missing or empty: {out}")
    print(rendered)
    print(f"scorecard written -> {_display_path(out)}")
    if incomplete:
        print("COLLECTION STATUS: INCOMPLETE (explicit --allow-partial) — " + "; ".join(incomplete))
    else:
        print(
            f"COLLECTION STATUS: PASS — {len(results)} selected track artifact(s) audited "
            "against pinned manifest tracks and case IDs"
        )
    if args.require_release and card.stamp_of(args.mode, results, manifest_ref) != "RELEASE":
        print("COLLECTION RELEASE GATE: BLOCKED — scorecard is DRAFT")
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

    posewarm = sub.add_parser("prefetch-pose", help="download/warm the configured pose provider checkpoint (RTMO by default)")
    posewarm.set_defaults(func=cmd_prefetch_pose)

    run = sub.add_parser("run", help="run benchmark tracks with retries and per-track logs")
    run.add_argument("--live", action="store_true", help="use the configured live vision backend")
    run.add_argument("--n", type=int, default=8, help="sample size per track")
    run.add_argument(
        "--tracks",
        default="safe-live",
        help="comma list or alias: all, safe-live, stable, no-counting, legacy-all",
    )
    run.add_argument("--exclude", default="", help="comma list of tracks to skip")
    run.add_argument("--workers", type=int, default=1, help="parallel track subprocesses")
    run.add_argument("--attempts", type=int, default=1, help="attempts per track")
    run.add_argument("--timeout-s", type=int, default=1800, help="timeout per track attempt; 0 disables")
    run.add_argument("--run-id", default=None, help="stable run id/folder name")
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest stamped on the scorecard")
    run.add_argument("--mode", default="sampled", choices=["sampled", "full"], help="scorecard mode")
    run.add_argument("--device", default="local", help="device/runtime descriptor for scorecard")
    run.add_argument("--fps", default="1", help="sampling/capture fps descriptor for scorecard")
    run.add_argument("--seeds", type=int, default=1, help="number of seeds represented")
    run.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip valid existing per-track result JSON files in the run folder",
    )
    run.add_argument(
        "--start-delay-s",
        type=float,
        default=0.0,
        help="stagger track subprocess starts by this many seconds per track index",
    )
    run.add_argument("--background", action="store_true", help="launch the run in the background and return immediately")
    run.add_argument(
        "--prefetch-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm the pose model (download RTMO checkpoint) before the counting track runs",
    )
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
    collect.add_argument(
        "--allow-partial",
        action="store_true",
        help="render an explicitly INCOMPLETE draft when selected track artifacts are missing",
    )
    collect.set_defaults(func=cmd_collect)

    internal = sub.add_parser("_run-track")
    internal.add_argument("--track", required=True, choices=sorted(TRACK_MODULES))
    internal.add_argument("--live", action="store_true")
    internal.add_argument("--n", type=int, required=True)
    internal.add_argument("--out", required=True)
    internal.add_argument("--manifest", required=True)
    internal.set_defaults(func=cmd_run_track)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
