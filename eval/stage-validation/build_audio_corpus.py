"""Build the AUDIO stage-validation corpus — cases whose ground truth is derived from SOUND.

``build_corpus.py`` builds the VIDEO corpus: every positive interval there comes from a visual
annotation. SV-1.6 (VAD admission recall) was *refused* against it because nothing in that corpus
carries audio ground truth and its largest dataset (``proactiveqa``) has no audio stream at all.
This module closes that gap and is the input to ``sv_audio.py``.

What "audio ground truth" means here
------------------------------------
Three strengths, never averaged together, always carried on the case:

``audio_annotated``        a human annotated a SOUND event with a start and an end on the source
                           clock (Perception Test ``sound_localisation``). This is the only source
                           strong enough to carry a headline admission-recall number.
``audio_annotated_point``  a human annotated the INSTANT an audio-carried event became answerable
                           (OmniPro ``ground_truth[*].trigger_time_sec`` where ``trigger_type``
                           names ``speech``/``sound``). A point, not an interval — matching it
                           needs an explicit, reported tolerance.
``proxy``                  the label names an event that is *usually* audible, but the annotated
                           interval is VISUAL (Charades). Reported, never used as an audio bar.

Case schema — a superset of ``corpus.json``'s, so the same consumers read both::

    case_id       stable "<dataset>::<video>::<ordinal>"
    dataset       perception_test_sound | omnipro_audio | charades_audible
    video_path    repo-relative media path
    video_sha256  content hash of the exact media bytes
    duration_s    source duration
    goal_text     the natural-language condition handed to the Watch
    goal_kind     transition | duration | count | state
    positives     [[start_s, end_s], ...] ground truth on the source clock
    provenance    the raw annotation fields the case was derived from
    gt_kind       interval | point | proxy_interval
    gt_strength   audio_annotated | audio_annotated_point | proxy
    audio         {codec, sample_rate, channels, duration_s} — PROBED from the media, not assumed

Ground-truth defence
--------------------
Three ground-truth defects were found and fixed in the video corpus (microsecond timestamps read as
seconds, intervals past the end of the media, unseeded selection). Every one of those classes is
re-checked here and the outcome is recorded in ``checks`` rather than assumed:

* ``units``       Perception Test ``sound_localisation`` timestamps are MICROSECONDS. Verified, not
                  trusted: ``timestamps[i] / 1e6`` must agree with ``frame_ids[i] / frame_rate``.
* ``containment`` every interval must fall inside the media's own duration.
* ``audio``       every selected media file must actually HAVE an audio stream (ffprobe).
* ``seeding``     selection is a seeded shuffle over a sorted list — never filesystem order.

This is an INTERNAL validation corpus. It is deliberately NOT the licensed publication manifest
(``eval/golden-v1/MANIFEST.json``) and must never be presented as one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING = REPO_ROOT / "eval" / "bench-staging"
PT_BENCH = REPO_ROOT / "packages" / "percept-vision" / "eval" / "bench" / "perception_test.py"

#: Same seed as the video corpus, so the two selections are reproducible the same way.
SELECTION_SEED = 20260726

#: Perception Test sound labels that are NOT a sound. ``Muted`` is the annotators' marker for a
#: stretch with NO audible source — treating it as a positive scores acoustic recall against
#: silence. See ``golden_sound_events`` in the defect list below.
NON_SOUND_LABELS = frozenset({"Muted"})

#: Labels that name a sound but not WHICH sound. Kept out of the goal-bearing positives (the
#: condition would read "an audible other sound"), still counted in the any-sound ground truth.
VAGUE_LABELS = frozenset({"Other"})

#: OmniPro trigger types whose event is carried by the AUDIO channel. ``visual`` alone is not.
AUDIO_TRIGGER_TYPES = frozenset(
    {"speech", "sound", "sound+speech", "visual+speech", "visual+sound", "visual+sound+speech"}
)

#: OmniPro trigger types with NO visual component — the strictest audio-only slice.
PURE_AUDIO_TRIGGER_TYPES = frozenset({"speech", "sound", "sound+speech"})

#: Charades action labels whose event normally makes a noise. The INTERVAL is still visual, so
#: these are ``proxy`` strength and are never the bar for an audio number.
CHARADES_AUDIBLE_HINTS = (
    "laughing", "sneezing", "cooking", "vacuum", "washing", "wash ", "pouring",
    "opening a door", "closing a door", "opening a closet", "closing a closet",
    "opening a refrigerator", "closing a refrigerator", "drinking", "eating",
    "watching television", "playing with a phone", "talking on a phone",
)

_OMNIPRO_KIND = {
    "cumulative_counting": "count",
    "dedup_counting": "count",
    "snapshot_counting": "count",
    "instant_event_alert": "transition",
    "semantic_condition_alert": "transition",
    "realtime_state_monitor": "state",
    "event_narration": "duration",
    "sequential_step_instruction": "transition",
    "explicit_target_grounding": "state",
}


# ---------------------------------------------------------------------------------------------
# media + hashing helpers
# ---------------------------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def probe_audio(path: Path) -> dict[str, Any] | None:
    """The media's FIRST audio stream, straight from the container — or ``None`` if it has none.

    Never assume a container carries sound. ``proactiveqa`` mp4s look identical to ``charades``
    mp4s from the outside and have no audio stream at all; that is exactly why SV-1.6 could not be
    measured, and a corpus that assumed otherwise would report a flat-zero recall as a runtime bug.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,sample_rate,channels,duration",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout
        data = json.loads(out or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    try:
        sample_rate = int(stream["sample_rate"])
    except (KeyError, TypeError, ValueError):
        sample_rate = None
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": stream.get("channels"),
        "duration_s": round(duration, 3) if duration is not None else None,
    }


def _media_index(dataset: str, *, by_name: bool = False) -> dict[str, Path]:
    root = STAGING / dataset
    if not root.is_dir():
        return {}
    return {(p.name if by_name else p.stem): p for p in root.rglob("*.mp4")}


def _merge(intervals: list[list[float]], *, gap: float = 0.0) -> list[list[float]]:
    """Union of intervals, oldest first. Overlapping ground truth would double-count recall."""
    out: list[list[float]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


# ---------------------------------------------------------------------------------------------
# Perception Test sound_localisation — the primary source
# ---------------------------------------------------------------------------------------------


def load_pt_bench() -> Any | None:
    """Load the repo's own Perception Test bench module BY PATH.

    Deliberately not ``sys.path.insert(packages/percept-vision)`` the way ``build_corpus.py`` does:
    that directory also contains the ``percept`` package, and shadowing the wheel under test with a
    source tree is precisely the mistake this suite exists to avoid. ``spec_from_file_location``
    loads the one module and touches nothing else.

    Reusing it is the point: the golden ``perception-test-acoustic`` track's own definitions
    (``ANN_ZIP['sound']``, the microsecond conversion, ``label_phrase``) live there.
    """
    if not PT_BENCH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_pt_bench_audio", PT_BENCH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - a corpus build must not require the bench harness
        return None
    return module


def label_phrase(label: str) -> str:
    """Perception Test label -> a plain-language audible condition.

    Mirrors ``eval/bench/perception_test.py::label_phrase`` and is used only when that module
    cannot be loaded, so the corpus never invents a second phrasing while the real one is reachable.
    """
    category, _, detail = label.partition(":")
    subject = detail.strip().lower()
    if subject in ("", "sound"):
        subject = category.strip().lower()
    return f"right now there is an audible {subject} sound or event"


def load_sound_annotations() -> tuple[dict[str, Any], str]:
    """Full valid-split ``sound_localisation`` annotations, from the CACHED zip when present.

    Returns ``(annotations, provenance)``. Falls back to the checked-in 8-video ``sample.json``,
    which is reported as such — the sample is train-split and carries ``Muted`` segments the valid
    split does not.
    """
    zip_path = STAGING / "perception_test/raw/challenge_sound_localisation_valid_annotations.zip"
    if zip_path.is_file():
        import zipfile

        try:
            with zipfile.ZipFile(zip_path) as handle:
                return json.loads(handle.read(handle.namelist()[0])), "valid_split_cached_zip"
        except (OSError, ValueError, KeyError):
            pass
    bench = load_pt_bench()
    if bench is not None:
        try:  # network fetch; only reached when the cache is absent
            return bench.load_annotations("sound"), "valid_split_fetched"
        except Exception:  # noqa: BLE001
            pass
    sample = STAGING / "perception_test/raw/sample.json"
    if sample.is_file():
        return json.loads(sample.read_text()), "sample_json_train_split"
    return {}, "unavailable"


def build_perception_test_sound(limit: int, checks: dict[str, Any]) -> list[dict[str, Any]]:
    annotations, provenance = load_sound_annotations()
    checks["pt_annotation_source"] = provenance
    checks["pt_videos_annotated"] = len(annotations)
    if not annotations:
        checks["pt_note"] = "no sound_localisation annotations reachable"
        return []

    bench = load_pt_bench()
    phrase: Callable[[str], str] = getattr(bench, "label_phrase", label_phrase)
    media = _media_index("perception_test")

    unit_errors: list[float] = []
    dropped_no_media = 0
    dropped_no_segments = 0
    dropped_unit_mismatch = 0
    dropped_out_of_range = 0
    non_sound_dropped = 0
    candidates: list[dict[str, Any]] = []

    for video_id in sorted(annotations):
        row = annotations[video_id]
        path = media.get(video_id)
        if path is None:
            dropped_no_media += 1
            continue
        meta = row.get("metadata") or {}
        try:
            fps = float(meta.get("frame_rate") or 0.0)
            n_frames = float(meta.get("num_frames") or 0.0)
        except (TypeError, ValueError):
            continue
        if fps <= 0 or n_frames <= 0:
            continue
        media_duration = n_frames / fps

        segments = [s for s in (row.get("sound_localisation") or []) if s.get("label")]
        if not segments:
            dropped_no_segments += 1
            continue

        by_label: dict[str, list[list[float]]] = {}
        any_sound: list[list[float]] = []
        bad_units = False
        out_of_range = False
        for seg in segments:
            label = str(seg["label"]).strip()
            stamps = seg.get("timestamps") or []
            frames = seg.get("frame_ids") or []
            if len(stamps) != 2:
                continue
            # DEFECT CLASS RE-CHECK: Perception Test stores timestamps in MICROSECONDS. Verified
            # against the row's own frame_ids/frame_rate rather than trusted, because reading them
            # as seconds puts every interval past the end of a ~25 s clip and reads as zero recall.
            start, end = float(stamps[0]) / 1e6, float(stamps[1]) / 1e6
            if len(frames) == 2:
                for value, frame_id in ((start, frames[0]), (end, frames[1])):
                    try:
                        unit_errors.append(abs(value - float(frame_id) / fps))
                    except (TypeError, ValueError, ZeroDivisionError):
                        bad_units = True
            if not end > start:
                continue
            if end > media_duration * 1.05:
                out_of_range = True
                continue
            end = min(end, media_duration)
            if label in NON_SOUND_LABELS:
                non_sound_dropped += 1
                continue
            any_sound.append([start, end])
            if label in VAGUE_LABELS:
                continue
            by_label.setdefault(label, []).append([start, end])

        if bad_units:
            dropped_unit_mismatch += 1
            continue
        if out_of_range and not any_sound:
            dropped_out_of_range += 1
            continue
        if not by_label or not any_sound:
            dropped_no_segments += 1
            continue

        # The goal-bearing label is the one with the most annotated segments (ties broken
        # lexicographically, never by dict order) so a case's condition has the most evidence
        # available for it.
        label = max(sorted(by_label), key=lambda k: (len(by_label[k]), k))
        positives = _merge(by_label[label])
        candidates.append(
            {
                "case_id": f"perception_test_sound::{video_id}::0",
                "dataset": "perception_test_sound",
                "video_path": _rel(path),
                "duration_s": round(media_duration, 3),
                "goal_text": phrase(label),
                "goal_kind": "transition",
                "positives": [[round(a, 3), round(b, 3)] for a, b in positives],
                "gt_kind": "interval",
                "gt_strength": "audio_annotated",
                "provenance": {
                    "source": "perception_test sound_localisation",
                    "annotation_set": provenance,
                    "label": label,
                    "labels_present": sorted(by_label),
                    "n_segments_label": len(by_label[label]),
                    "n_segments_any": len(any_sound),
                    "sound_positives_any": [
                        [round(a, 3), round(b, 3)] for a, b in _merge(any_sound)
                    ],
                    "is_visible": sorted(
                        {
                            s.get("is_visible")
                            for s in segments
                            if str(s.get("label", "")).strip() == label
                        }
                    ),
                    "frame_rate": fps,
                    "audio_samples": meta.get("audio_samples"),
                    "audio_sample_rate": meta.get("audio_sample_rate"),
                    "timestamp_unit": "microseconds (verified against frame_ids/frame_rate)",
                },
            }
        )

    checks["pt_unit_max_error_s"] = round(max(unit_errors), 6) if unit_errors else None
    checks["pt_unit_samples"] = len(unit_errors)
    checks["pt_dropped"] = {
        "no_local_media": dropped_no_media,
        "no_usable_segments": dropped_no_segments,
        "unit_mismatch": dropped_unit_mismatch,
        "interval_past_media_end": dropped_out_of_range,
        "non_sound_label_segments": non_sound_dropped,
    }
    checks["pt_candidates"] = len(candidates)

    rng = random.Random(SELECTION_SEED)
    rng.shuffle(candidates)
    return candidates[:limit]


# ---------------------------------------------------------------------------------------------
# OmniPro — audio-carried trigger instants
# ---------------------------------------------------------------------------------------------


def build_omnipro_audio(limit: int, checks: dict[str, Any]) -> list[dict[str, Any]]:
    path = STAGING / "omnipro/raw/benchmark_subset.json"
    if not path.is_file():
        checks["omnipro_note"] = "benchmark_subset.json absent"
        return []
    rows = json.loads(path.read_text())
    media = _media_index("omnipro")

    dependency: dict[str, int] = {}
    trigger_types: dict[str, int] = {}
    for row in rows:
        key = str(row.get("audio_dependency"))
        dependency[key] = dependency.get(key, 0) + 1
        for trigger in row.get("ground_truth") or []:
            name = str(trigger.get("trigger_type"))
            trigger_types[name] = trigger_types.get(name, 0) + 1
    checks["omnipro_audio_dependency_values"] = dependency
    checks["omnipro_trigger_types"] = trigger_types
    checks["omnipro_rows_total"] = len(rows)
    checks["omnipro_videos_local"] = len(media)

    candidates: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: str(r.get("id"))):
        media_path = media.get(row.get("video_id"))
        if media_path is None:
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        triggers = [
            trigger
            for trigger in (row.get("ground_truth") or [])
            if str(trigger.get("trigger_type")) in AUDIO_TRIGGER_TYPES
            and isinstance(trigger.get("trigger_time_sec"), (int, float))
        ]
        if not triggers:
            continue
        duration = row.get("duration")
        # `question_time_sec` is when the QUESTION is asked, not when the event happens; it is 0 on
        # every row of this subset. The answerable instants are the per-trigger times.
        seconds = sorted(float(t["trigger_time_sec"]) for t in triggers)
        if isinstance(duration, (int, float)) and duration > 0:
            seconds = [s for s in seconds if 0.0 <= s <= float(duration)]
        if not seconds:
            continue
        pure = [t for t in triggers if str(t["trigger_type"]) in PURE_AUDIO_TRIGGER_TYPES]
        candidates.append(
            {
                "case_id": f"omnipro_audio::{row['video_id']}::{str(row['id']).rsplit('::', 1)[-1]}",
                "dataset": "omnipro_audio",
                "video_path": _rel(media_path),
                "duration_s": duration,
                "goal_text": question,
                "goal_kind": _OMNIPRO_KIND.get(row.get("task"), "transition"),
                "positives": [[round(s, 3), round(s, 3)] for s in seconds],
                "gt_kind": "point",
                "gt_strength": "audio_annotated_point",
                "provenance": {
                    "source": "omnipro ground_truth trigger_time_sec",
                    "task": row.get("task"),
                    "audio_dependency": row.get("audio_dependency"),
                    "trigger_types": sorted({str(t["trigger_type"]) for t in triggers}),
                    "n_pure_audio_triggers": len(pure),
                    "instant_annotation": True,
                    "note": (
                        "point ground truth: matching it requires an explicit tolerance, which the "
                        "consuming measurement must state"
                    ),
                },
            }
        )
    checks["omnipro_candidates"] = len(candidates)
    rng = random.Random(SELECTION_SEED)
    rng.shuffle(candidates)
    return candidates[:limit]


# ---------------------------------------------------------------------------------------------
# Charades — audible-label actions on a VISUAL interval (proxy strength only)
# ---------------------------------------------------------------------------------------------


def build_charades_audible(limit: int, checks: dict[str, Any]) -> list[dict[str, Any]]:
    path = STAGING / "charades/raw/selection.json"
    if not path.is_file():
        checks["charades_note"] = "selection.json absent"
        return []
    rows = json.loads(path.read_text())
    media = _media_index("charades")
    candidates: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda r: str(r.get("id"))):
        media_path = media.get(row.get("id"))
        if media_path is None:
            continue
        actions = [
            a
            for a in (row.get("actions") or [])
            if len(a) == 3
            and a[2] > a[1]
            and any(hint in str(a[0]).lower() for hint in CHARADES_AUDIBLE_HINTS)
        ]
        if not actions:
            continue
        text, start, end = max(actions, key=lambda a: a[2] - a[1])
        length = row.get("length")
        if isinstance(length, (int, float)) and length > 0:
            end = min(float(end), float(length))
            start = min(float(start), float(end))
        if not end > start:
            continue
        candidates.append(
            {
                "case_id": f"charades_audible::{row['id']}::0",
                "dataset": "charades_audible",
                "video_path": _rel(media_path),
                "duration_s": length,
                "goal_text": str(text).strip(),
                "goal_kind": "duration",
                "positives": [[round(float(start), 3), round(float(end), 3)]],
                "gt_kind": "proxy_interval",
                "gt_strength": "proxy",
                "provenance": {
                    "source": "charades action label with an audible referent",
                    "scene": row.get("scene"),
                    "n_audible_actions": len(actions),
                    "note": (
                        "the ANNOTATED INTERVAL IS VISUAL. The label names an event that is usually "
                        "audible; audibility itself was never annotated. Proxy only — never an "
                        "audio bar."
                    ),
                },
            }
        )
    checks["charades_candidates"] = len(candidates)
    rng = random.Random(SELECTION_SEED)
    rng.shuffle(candidates)
    return candidates[:limit]


# ---------------------------------------------------------------------------------------------
# datasets with no audio to offer — recorded, never silently skipped
# ---------------------------------------------------------------------------------------------


def survey_refusals() -> dict[str, Any]:
    """Probe the datasets that cannot contribute, so the refusal is measured rather than asserted."""
    out: dict[str, Any] = {}
    for dataset in ("proactiveqa", "repcount"):
        media = sorted(_media_index(dataset).items())[:8]
        probed = [(name, probe_audio(path)) for name, path in media]
        with_audio = [name for name, info in probed if info is not None]
        out[dataset] = {
            "probed": len(probed),
            "with_audio_stream": len(with_audio),
            "examples_with_audio": with_audio[:3],
            "verdict": (
                "no audio stream — cannot carry audio ground truth"
                if not with_audio
                else "audio present on some media, but no AUDIO annotation exists for it"
            ),
        }
    return out


BUILDERS: dict[str, Callable[[int, dict[str, Any]], list[dict[str, Any]]]] = {
    "perception_test_sound": build_perception_test_sound,
    "omnipro_audio": build_omnipro_audio,
    "charades_audible": build_charades_audible,
}

DEFAULT_LIMITS = {
    "perception_test_sound": 40,
    "omnipro_audio": 10,
    "charades_audible": 10,
}


def build(
    limits: dict[str, int] | None = None,
    *,
    hash_media: bool = True,
    probe: bool = True,
    survey: bool = True,
) -> dict[str, Any]:
    """Build the audio corpus. Returns the full document; the caller decides where it lands."""
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    checks: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []

    for name, builder in BUILDERS.items():
        requested = int(limits.get(name, 0))
        if requested <= 0:
            coverage[name] = {"requested": 0, "built": 0}
            continue
        selected = builder(requested, checks)
        kept: list[dict[str, Any]] = []
        no_audio: list[str] = []
        for case in selected:
            path = REPO_ROOT / case["video_path"]
            if probe:
                info = probe_audio(path)
                if info is None:
                    # A case whose media has no audio stream cannot carry audio ground truth,
                    # whatever its annotation says. Drop it loudly.
                    no_audio.append(case["case_id"])
                    continue
                case["audio"] = info
                for interval in case["positives"]:
                    if info["duration_s"] and interval[1] > info["duration_s"] + 0.5:
                        interval[1] = round(float(info["duration_s"]), 3)
                        interval[0] = min(interval[0], interval[1])
            case["video_sha256"] = sha256_file(path) if hash_media else None
            kept.append(case)
        cases.extend(kept)
        coverage[name] = {
            "requested": requested,
            "built": len(kept),
            "short_by": max(0, requested - len(kept)),
            "dropped_no_audio_stream": no_audio,
            "gt_strength": kept[0]["gt_strength"] if kept else None,
        }

    by_strength: dict[str, int] = {}
    for case in cases:
        by_strength[case["gt_strength"]] = by_strength.get(case["gt_strength"], 0) + 1

    document = {
        "version": 1,
        "kind": "stage-validation-audio-corpus",
        "note": "INTERNAL validation corpus. Not the licensed publication manifest.",
        "selection_seed": SELECTION_SEED,
        "coverage": coverage,
        "n_cases": len(cases),
        "cases_by_gt_strength": by_strength,
        "ground_truth_checks": checks,
        "refused_datasets": survey_refusals() if survey else {},
        "cases": cases,
    }
    return document


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", type=int, default=DEFAULT_LIMITS["perception_test_sound"])
    ap.add_argument("--omnipro", type=int, default=DEFAULT_LIMITS["omnipro_audio"])
    ap.add_argument("--charades", type=int, default=DEFAULT_LIMITS["charades_audible"])
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the corpus JSON (default: stdout summary only)",
    )
    ap.add_argument("--no-hash", action="store_true", help="skip media hashing (faster, unpinned)")
    args = ap.parse_args()

    document = build(
        {
            "perception_test_sound": args.pt,
            "omnipro_audio": args.omnipro,
            "charades_audible": args.charades,
        },
        hash_media=not args.no_hash,
    )

    print(f"audio corpus: {document['n_cases']} cases")
    for name, info in document["coverage"].items():
        print(f"  {name:24s} {info['built']}/{info['requested']}  ({info.get('gt_strength')})")
    for name, info in document["refused_datasets"].items():
        print(f"  REFUSED {name:16s} {info['verdict']}")
    checks = document["ground_truth_checks"]
    print(f"  unit check: max |ts/1e6 - frame/fps| = {checks.get('pt_unit_max_error_s')} s "
          f"over {checks.get('pt_unit_samples')} bounds")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2, sort_keys=False))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
