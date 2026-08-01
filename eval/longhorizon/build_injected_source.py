"""Build a long-horizon source with FRAME-EXACT ground truth by injecting labelled clips.

Why this exists
===============
The longest evaluation source in this repo is 81.6 minutes. The runtime is described as running
continuously. Nothing here measures a horizon longer than a lunch break, and the reason is
economic, not technical: you cannot hand-label hours of video, and a precision number taken from a
partially-labelled soak is exactly the "precision without ground-truth coverage" error that
``docs/evaluation/benchmarks.md`` forbids.

Injection is the way out. Take clips whose ground truth is already annotated frame-by-frame, splice
them into a long background stream at scheduled offsets, and shift their intervals onto the joined
clock. The result is **frame-exact ground truth at an arbitrary horizon** for the price of the
labels you already have. The background is not labelled and is never claimed to be.

Relationship to the existing builders
-------------------------------------
``eval/stage-validation/build_long_sources.py`` concatenates same-goal clips end to end;
``build_long_run.py`` concatenates the whole corpus in a dataset round-robin. Both establish the
arithmetic this module reuses:

1. **Uniform re-encode.** The ffmpeg concat demuxer requires matching codec parameters, and the
   corpus spans five datasets with five encodings.
2. **Fixed geometry, never ``scale=W:-2``.** ``-2`` preserves each source's aspect ratio, so a
   corpus spanning 4:3 and 16:9 yields parts of different heights. The demuxer accepts them,
   ``-c copy`` writes them, and the result decodes the first part and then holds one frozen frame
   for the rest of the run — with a correct duration, a correct frame count, and correct
   timestamps. Only looking at the pixels catches it, which is why ``verify_content`` runs before a
   manifest is written.
3. **Offsets are measured on the RE-ENCODED part, never on the source.** fps and scale
   normalisation shift container duration by fractions of a frame; a source-derived offset drifts
   cumulatively across a hundred members.
4. **Positives are shifted by the offset of the part they came from**, and an interval that falls
   past a trim is dropped or clipped, counted either way, so the ground truth never claims coverage
   the media does not contain.

What is new here is the **schedule**: the clips do not abut, they are placed at chosen offsets with
unlabelled background between them, which is what makes a two-hour horizon out of twenty minutes of
labels.

The manifest schema
===================
One JSON document per built source. ``kind`` is ``"longhorizon-injected-source"``, ``version`` is 1.

Top level::

    source_id           stable slug; the mp4 is "<source_id>.mp4" beside the manifest
    kind, version       schema discriminators
    build               {tool, seed, policy, argv-derived params} — everything needed to rebuild
    video_path          repo-relative path to the media
    video_sha256        hash of the built media, so a manifest cannot drift from its pixels
    total_duration_s    the joined clock: SUM of measured part durations (authoritative)
    container_duration_s / drift_s
                        what ffprobe reports on the concatenated file, and the difference. Non-zero
                        drift is normal and small; a large value means the concat did not do what
                        the arithmetic assumed.
    profile             the uniform encode profile: {scale_width, scale_height, vfilter, fps, crf,
                        pix_fmt, audio: false}. Two manifests are only comparable if this matches.
    clock               wall-clock anchor semantics — see below
    license             see below
    redistributable     bool — see below
    segments            the full tiling of the clock, in order (see below)
    injections          the injected clips, with their labels (see below)
    watches             one entry per (goal_text, gt_strength) — what a runner arms
    totals              per-tier counts and labelled seconds, NEVER a single blended number
    checks              collision, containment, tiling, and pixel-liveness results
    content_verification
                        the pixel-liveness probe from build_long_run.py

``clock`` — wall-clock anchor semantics
---------------------------------------
::

    clock: {
      "media_epoch_s": 0.0,
      "wall_clock_anchor": null | "2026-07-31T02:00:00Z",
      "anchor_semantics": "none" | "declared" | "recorder_utc",
      "monotonic": true,
      "unobserved": [[start_s, end_s], ...]
    }

``anchor_semantics`` is the field that stops a whole class of wrong claim:

``none``          this source is spliced from archival clips. Media ``t=0`` corresponds to no real
                  instant. Any diurnal, sunrise/sunset, or schedule-derived reasoning on this
                  source is **invalid** — there is no wall clock to reason about.
``declared``      the builder was handed an anchor for convenience (e.g. to exercise a scheduler).
                  It is a fiction and is labelled as one; it still licenses no diurnal claim.
``recorder_utc``  media ``t=0`` is the UTC instant a recorder began writing. This is the value a
                  real recorded stream carries, and the only one under which schedule-derived
                  ground truth — sunrise, scheduled transit, timed lighting — means anything.

``unobserved`` is first-class and normally empty here: a spliced source has no outage. It exists so
a replay adapter reads the same shape whether the source was spliced or recorded, and so a real
recorder gap surfaces as an interval that is neither positive nor negative — the semantics
``eval/synthetic/intervals/degraded_judge_gap_is_not_continued_truth`` already pins.

``license`` / ``redistributable``
----------------------------------
Per source::

    license: {
      "fixture": {"id": ..., "redistributable": bool},
      "contributors": [ {dataset, license_id, license_note, redistributable, role, n_parts} ]
    }
    redistributable: bool     # AND over every contributor, background included

``redistributable`` is the AND, not the majority and not the headline dataset: one non-clearable
background clip makes the whole spliced artefact non-publishable, because the artefact contains its
pixels. The publish gate refuses a corpus containing a non-redistributable source, so ``--require-
redistributable`` makes the builder refuse to write the manifest at all rather than emit something
that has to be caught downstream. An unknown licence is treated as **not** redistributable; there
is no permissive default.

``segments`` — the tiling of the clock
---------------------------------------
Every second of the source belongs to exactly one segment, in order, with no hole and no overlap.
Two roles::

    {role: "background", index, offset_s, duration_s, gt: "none",
     media: {video_path, video_sha256, src_in_s, src_out_s}, labels_discarded: {...}}

    {role: "injection", index, offset_s, duration_s, injection_id, case_id, goal_text}

``gt: "none"`` on a background segment means **unlabelled**. It does not mean "no event occurred".
Background is drawn from real footage that was never annotated for the injected conditions, so an
injected goal may well be true somewhere in it. That is the price of the method and it has one
consequence, stated here so no consumer has to infer it: **recall and latency are computable on
this source; precision and false-positive rate are not.** A fire during a background segment is not
evidence of a false positive. If a background clip carried labels of its own (it was lifted from a
labelled corpus case), those labels are discarded and ``labels_discarded`` records what was dropped
and why, so the omission is visible rather than silent.

``injections`` — the labelled events
-------------------------------------
::

    {
      injection_id            "<ordinal>::<case_id>"
      case_id, dataset        origin case in the stage-validation corpus (see ``--corpus``)
      origin_video_path
      origin_video_sha256     the exact bytes the labels were annotated against
      goal_text, goal_kind
      scheduled_offset_s      what the schedule policy asked for
      offset_s                where it actually landed on the joined clock (cumulative measured
                              durations of the preceding parts) — THIS is the authoritative offset
      schedule_drift_s        offset_s - scheduled_offset_s
      source_duration_s       origin clip duration
      duration_s              duration of the RE-ENCODED part, which is what the clock uses
      positives_source        intervals on the origin clip's clock
      positives               the same intervals shifted onto the joined clock
      n_positives_dropped / n_positives_clipped
      gt_strength             one of the three tiers below
      gt_kind                 interval | point
      point_tolerance_s       present iff gt_kind == "point"; matching a point REQUIRES a stated
                              tolerance, and the manifest states it rather than leaving each
                              consumer to invent one
      provenance              the origin case's provenance, carried through unmodified
    }

Ground-truth strength — three tiers, never averaged
----------------------------------------------------
The discipline is the one ``eval/stage-validation/build_audio_corpus.py`` established for audio:
the strength is carried on every interval, tiers are reported separately, and no number is ever
computed across them.

``annotated_interval``  a human annotated the START and the END of the event itself, on the origin
                        clip's clock. Perception Test ``action_localisation`` segments, Charades
                        action spans, RepCount per-cycle bounds. This is the only tier strong
                        enough to carry a headline recall or latency number.
``annotated_point``     a human annotated the INSTANT the event became answerable. A point, not an
                        interval. OmniPro ``ground_truth[*].trigger_time_sec``. Matching it needs an
                        explicit tolerance, which the manifest states; it is never fed to an
                        interval IoU.
``derived_span``        the span exists but was never annotated as the event's extent — it is a
                        related or whole-clip span standing in for one. ProactiveQA
                        ``related_timespan`` is the span an assistant reply *relates to*, not the
                        moment its condition became true. Reported, never a bar.

``totals`` therefore carries ``labelled_seconds_by_strength`` and ``n_intervals_by_strength`` and
no blended total. ``watches`` are keyed on ``(goal_text, gt_strength)``: if the same sentence
arrives at two strengths it becomes two watch entries, so a scorer cannot merge them by accident.

Determinism
===========
Regeneration is a pure function of ``--seed`` and the arguments recorded in ``build``. Selection is
a seeded shuffle over a **sorted** list, never filesystem order; jitter comes from the same seeded
generator; nothing reads the wall clock. The one value that is not reproducible from the seed alone
is ``generated_utc``, which is recorded for provenance and is excluded from any comparison — pass
``--no-timestamp`` for a byte-identical rebuild.

Refusals
========
The builder refuses rather than degrades:

* **Collision.** Two clips whose intervals overlap (or violate ``--min-gap-s``) abort the build and
  the offending pair is reported with both offsets, both durations, and the shortfall in seconds.
  Silently nudging one clip would put the manifest's offsets and the media's offsets out of
  agreement, which is undetectable downstream and poisons every latency number taken from it. The
  built-in policies cannot collide by construction; ``--schedule-json`` can, and it is the path a
  real recording takes, where the offsets come from the recorder's clock rather than a policy.
* **Infeasible schedule.** N clips of measured total D cannot be placed in a horizon T with
  ``--min-gap-s`` between them if the arithmetic does not close. Reported with the numbers.
* **Frozen pixels.** The concat verification from ``build_long_run.py`` runs before the manifest is
  written; a source whose content does not change is not published.
* **Non-redistributable under ``--require-redistributable``.**

Verification
============
``--verify <manifest.json>`` re-derives every claim from the media on disk: media hash, container
duration against the joined clock, the segment tiling (exact cover, no hole, no overlap), every
positive interval inside the media, decodability at each injection offset, and the origin media
hashes still matching the bytes the labels were annotated against. It exits non-zero on any
mismatch.

Disk
====
Intermediate re-encoded parts are deleted after the concat unless ``--keep-parts`` is passed; the
build machine is disk-constrained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The label source. ``corpus.v1.1.json``, not ``corpus.json``, for two reasons: it is the TRACKED
#: one (``corpus.json`` is git-ignored, so a fresh clone cannot build against it), and v1.0's
#: repcount conditions paste the raw dataset token into the sentence a judge is asked to evaluate
#: ("a person completes a jump_jack"). ``corpus.version-diff.json`` records that case membership,
#: media pins and ground truth are byte-identical between the two — only the wording changed — so
#: nothing about the offsets or the labels depends on which is used.
CORPUS = REPO_ROOT / "eval" / "stage-validation" / "corpus.v1.1.json"
OUT_DIR = REPO_ROOT / "eval" / "longhorizon" / "sources"

SCHEMA_VERSION = 1
SCHEMA_KIND = "longhorizon-injected-source"

# --------------------------------------------------------------------------------------------
# uniform encode profile
# --------------------------------------------------------------------------------------------
#: Deliberately modest, and identical in spirit to the two stage-validation builders: these sources
#: exercise temporal cognition over a long horizon, not decode fidelity, and the build machine is
#: disk-constrained.
#:
#: The geometry is FIXED and letterboxed rather than ``scale=W:-2``. See the module docstring: an
#: aspect-preserving scale across a mixed-aspect corpus produces a concat that freezes on the first
#: part while reporting a perfectly correct duration.
SCALE_WIDTH = 640
SCALE_HEIGHT = 360
TARGET_FPS = 15
CRF = 28
PIX_FMT = "yuv420p"
VFILTER = (
    f"scale={SCALE_WIDTH}:{SCALE_HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={SCALE_WIDTH}:{SCALE_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}"
)

# --------------------------------------------------------------------------------------------
# ground-truth strength, per contributing dataset
# --------------------------------------------------------------------------------------------
#: How each corpus dataset's positives were produced, and therefore which tier they may be scored
#: in. Derived from ``eval/stage-validation/build_corpus.py``, not guessed:
#:
#: * ``perception_test`` — ``action_localisation`` segment timestamps (microseconds, converted).
#: * ``charades``        — action span (start, end) from the Charades annotation.
#: * ``repcount``        — per-cycle frame bounds converted with the media's own fps.
#: * ``omnipro``         — ``ground_truth[*].trigger_time_sec``: an instant, stored as [t, t].
#: * ``proactiveqa``     — ``answer[0].related_timespan``: the span the reply RELATES TO. It is not
#:                         an annotation of when the condition became true, so it cannot be an
#:                         interval bar; it is a stand-in and is labelled as one.
GT_STRENGTH: dict[str, tuple[str, str]] = {
    "perception_test": ("annotated_interval", "interval"),
    "charades": ("annotated_interval", "interval"),
    "repcount": ("annotated_interval", "interval"),
    "omnipro": ("annotated_point", "point"),
    "proactiveqa": ("derived_span", "interval"),
}

#: Tiers in descending strength. Only the first may carry a headline number.
GT_TIERS = ("annotated_interval", "annotated_point", "derived_span")

#: Default tolerance attached to a point-annotated positive. It is stated on the manifest so every
#: consumer scores the same window; matching a point without a declared tolerance is meaningless.
POINT_TOLERANCE_S = 2.0

# --------------------------------------------------------------------------------------------
# licensing
# --------------------------------------------------------------------------------------------
#: Redistribution status per contributing dataset. An unknown dataset is NOT redistributable —
#: there is no permissive default, because the failure mode is publishing pixels that cannot be
#: published. A spliced artefact is redistributable only if every contributor is, background
#: included, since the artefact contains their pixels.
LICENSES: dict[str, dict[str, Any]] = {
    "perception_test": {
        "license_id": "CC-BY-4.0",
        "redistributable": True,
        "license_note": "Perception Test is released CC-BY-4.0; attribution required.",
    },
    "charades": {
        "license_id": "charades-license-non-commercial",
        "redistributable": False,
        "license_note": "Charades ships under its own non-commercial agreement; pixels are not "
                        "ours to redistribute.",
    },
    "repcount": {
        "license_id": "unknown-research-use",
        "redistributable": False,
        "license_note": "RepCount aggregates third-party clips; redistribution not established.",
    },
    "proactiveqa": {
        "license_id": "unknown-research-use",
        "redistributable": False,
        "license_note": "Egocentric source footage; redistribution not established.",
    },
    "omnipro": {
        "license_id": "unknown-third-party",
        "redistributable": False,
        "license_note": "Platform-hosted source footage; redistribution not established.",
    },
}
UNKNOWN_LICENSE = {
    "license_id": "unknown",
    "redistributable": False,
    "license_note": "No licence record; treated as not redistributable.",
}


def license_for(dataset: str) -> dict[str, Any]:
    return dict(LICENSES.get(dataset, UNKNOWN_LICENSE))


# --------------------------------------------------------------------------------------------
# media helpers
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


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


def _encode(src: Path, dst: Path, *, ss: float | None = None, t: float | None = None) -> bool:
    """Re-encode ``src`` (optionally a [ss, ss+t) window) to the uniform profile.

    ``-ss`` is placed BEFORE ``-i`` so ffmpeg seeks rather than decoding-and-discarding; because
    the output is re-encoded the cut is frame-accurate anyway, and the whole point of a
    disk-constrained two-hour build is not to decode 12 GB to reach the interesting minute.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", str(src)]
    if t is not None:
        cmd += ["-t", f"{t:.3f}"]
    cmd += ["-vf", VFILTER, "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
            "-pix_fmt", PIX_FMT, "-an", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return result.returncode == 0 and dst.is_file()


def verify_content(path: Path, *, probes: int = 90) -> dict[str, Any]:
    """Decode across the joined clock and prove the PIXELS actually change.

    Adapted from ``build_long_run.py``. Every cheap check — container duration, frame count,
    timestamp monotonicity — passes on a concat that silently froze at the first part boundary.
    This is the check that does not: sample ``probes`` frames uniformly, signature each, and report
    how many are distinct and how long the longest identical run is.
    """
    try:
        import cv2  # the [runtime] extra, already required by the source decoder
    except ImportError:
        return {"ok": False, "reason": "opencv unavailable; cannot verify pixels"}

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        return {"ok": False, "reason": "no frames reported"}
    step = max(1, total // probes)
    sigs: list[tuple] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 16))
            sigs.append(tuple(int(v) for v in small.flatten()[::8]))
        index += 1
    cap.release()
    longest = 0
    run = 0
    for a, b in zip(sigs, sigs[1:]):
        run = run + 1 if a == b else 0
        longest = max(longest, run + 1 if run else 0)
    distinct = len(set(sigs))
    return {
        "ok": len(sigs) > 4 and distinct >= max(4, len(sigs) // 4),
        "frames_read": index,
        "frames_reported": total,
        "fps": fps,
        "probes": len(sigs),
        "distinct_signatures": distinct,
        "longest_identical_run_probes": longest,
        "longest_identical_run_s": (round(longest * step / fps, 1) if fps else None),
    }


# --------------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------------


def eligible_injections(
    cases: list[dict[str, Any]],
    *,
    datasets: set[str] | None,
    strengths: set[str],
    min_s: float,
    max_s: float,
    durations: dict[str, float],
) -> list[dict[str, Any]]:
    """Corpus cases that can be injected, annotated with their GT tier.

    A case is eligible when its media is on disk, its probed duration falls in the requested band,
    and it carries at least one positive. Cases with no positives are useless as injections: the
    whole point of the method is that the injected span is labelled.
    """
    out: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda c: c["case_id"]):
        dataset = case["dataset"]
        if datasets is not None and dataset not in datasets:
            continue
        strength, gt_kind = GT_STRENGTH.get(dataset, ("derived_span", "interval"))
        if strength not in strengths:
            continue
        if not (case.get("positives") or []):
            continue
        path = REPO_ROOT / case["video_path"]
        duration = durations.get(case["video_path"])
        if duration is None or not path.is_file():
            continue
        if not (min_s <= duration <= max_s):
            continue
        entry = dict(case)
        entry["_duration_s"] = duration
        entry["_gt_strength"] = strength
        entry["_gt_kind"] = gt_kind
        out.append(entry)
    return out


def select_injections(pool: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded, dataset round-robin selection.

    Round-robin over datasets rather than a flat shuffle, for the same reason ``build_long_run.py``
    orders its members that way: a flat shuffle over a corpus with one dominant dataset puts most
    of one dataset's goals in one stretch of the horizon. Interleaving spreads goal kinds and scene
    types across the whole clock, which is the only way the long episode window and the re-arm path
    see varied input rather than one block of squats.
    """
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in pool:
        by_dataset[case["dataset"]].append(case)
    for dataset in sorted(by_dataset):
        # Shuffle a SORTED list with a per-dataset stream derived from the seed: filesystem or dict
        # order must never reach the selection.
        by_dataset[dataset].sort(key=lambda c: c["case_id"])
        random.Random(f"{seed}:{dataset}").shuffle(by_dataset[dataset])
    order: list[dict[str, Any]] = []
    i = 0
    while len(order) < len(pool):
        progressed = False
        for dataset in sorted(by_dataset):
            bucket = by_dataset[dataset]
            if i < len(bucket):
                order.append(bucket[i])
                progressed = True
        if not progressed:
            break
        i += 1
    return order[:n]


def select_background(
    cases: list[dict[str, Any]],
    *,
    datasets: set[str],
    exclude_paths: set[str],
    durations: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """Background media, seeded-shuffled over a sorted, de-duplicated list of paths.

    A path already used as an injection is excluded: splicing the same clip in twice, once labelled
    and once not, would make the same footage simultaneously ground truth and unlabelled.
    """
    seen: dict[str, dict[str, Any]] = {}
    for case in sorted(cases, key=lambda c: c["case_id"]):
        if case["dataset"] not in datasets:
            continue
        path = case["video_path"]
        if path in exclude_paths or path in seen:
            continue
        if durations.get(path) is None or not (REPO_ROOT / path).is_file():
            continue
        seen[path] = case
    ordered = [seen[key] for key in sorted(seen)]
    random.Random(f"{seed}:background").shuffle(ordered)
    return ordered


# --------------------------------------------------------------------------------------------
# scheduling
# --------------------------------------------------------------------------------------------


def schedule_offsets(
    durations: list[float],
    *,
    horizon_s: float,
    policy: str,
    min_gap_s: float,
    lead_in_s: float,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    """Deterministic injection offsets on the joined clock.

    ``even``      clips at the start of equal slots. Reproducible with no randomness at all; the
                  gaps are uniform, which is the least interesting arrangement for re-arm.
    ``jittered``  (default) equal slots, seeded uniform jitter inside the part of each slot that
                  still leaves ``min_gap_s`` of background after the clip. Varied gap lengths are
                  what actually exercise re-arm, refractory and episode-window behaviour.
    ``bursty``    clips paired into clusters separated by ``min_gap_s``, clusters spread across the
                  horizon. Deliberately stresses back-to-back occurrence; it is also the policy
                  most likely to trip the collision refusal, which is the point.

    Returns ``(offsets, feasibility)``. Raises ``ValueError`` when the arithmetic does not close —
    a schedule that does not fit is refused with the numbers, never squeezed.
    """
    n = len(durations)
    if n == 0:
        raise ValueError("no clips to schedule")
    total_clip_s = sum(durations)
    required = total_clip_s + min_gap_s * n + lead_in_s
    feasibility = {
        "policy": policy,
        "n_clips": n,
        "horizon_s": round(horizon_s, 3),
        "total_clip_s": round(total_clip_s, 3),
        "min_gap_s": min_gap_s,
        "lead_in_s": lead_in_s,
        "required_s": round(required, 3),
        "slack_s": round(horizon_s - required, 3),
    }
    if required > horizon_s:
        raise ValueError(
            f"schedule infeasible: {n} clips totalling {total_clip_s:.1f}s plus "
            f"{min_gap_s:.1f}s gaps and {lead_in_s:.1f}s lead-in need {required:.1f}s, "
            f"but the horizon is {horizon_s:.1f}s (short by {required - horizon_s:.1f}s). "
            f"Reduce --n-inject, --min-gap-s, or raise --target-s."
        )

    usable = horizon_s - lead_in_s
    rng = random.Random(f"{seed}:schedule:{policy}")
    offsets: list[float] = []

    if policy in ("even", "jittered"):
        slot = usable / n
        for index, duration in enumerate(durations):
            slot_start = lead_in_s + index * slot
            room = slot - duration - min_gap_s
            if room < 0:
                raise ValueError(
                    f"schedule infeasible: clip {index} is {duration:.1f}s and its "
                    f"{slot:.1f}s slot cannot also hold a {min_gap_s:.1f}s gap"
                )
            jitter = rng.uniform(0.0, room) if policy == "jittered" else 0.0
            offsets.append(round(slot_start + jitter, 3))
    elif policy == "bursty":
        # Pairs of adjacent clips separated by exactly min_gap_s, the pairs themselves spread over
        # the horizon. Exercises "two occurrences inside one episode window" against "one
        # occurrence per long idle" in the same run.
        n_clusters = max(1, (n + 1) // 2)
        cluster_slot = usable / n_clusters
        index = 0
        for cluster in range(n_clusters):
            members = durations[index:index + 2]
            span = sum(members) + min_gap_s * (len(members) - 1)
            room = cluster_slot - span - min_gap_s
            if room < 0:
                raise ValueError(
                    f"schedule infeasible: cluster {cluster} needs {span:.1f}s inside a "
                    f"{cluster_slot:.1f}s slot with a {min_gap_s:.1f}s trailing gap"
                )
            cursor = lead_in_s + cluster * cluster_slot + rng.uniform(0.0, room)
            for duration in members:
                offsets.append(round(cursor, 3))
                cursor += duration + min_gap_s
            index += len(members)
    else:
        raise ValueError(f"unknown schedule policy: {policy!r}")

    return offsets, feasibility


def detect_collisions(
    offsets: list[float], durations: list[float], *, min_gap_s: float, labels: list[str]
) -> list[dict[str, Any]]:
    """Every pair of injections whose spans overlap, or sit closer than ``min_gap_s``.

    Reported, never repaired. Nudging a colliding clip would leave the manifest claiming an offset
    the media does not have; a latency measured against a wrong offset is wrong by exactly the nudge
    and there is nothing downstream that can notice.
    """
    order = sorted(range(len(offsets)), key=lambda i: offsets[i])
    collisions: list[dict[str, Any]] = []
    for a, b in zip(order, order[1:]):
        end_a = offsets[a] + durations[a]
        gap = offsets[b] - end_a
        if gap < min_gap_s:
            collisions.append({
                "kind": "overlap" if gap < 0 else "gap_below_minimum",
                "first": {"label": labels[a], "offset_s": round(offsets[a], 3),
                          "duration_s": round(durations[a], 3), "end_s": round(end_a, 3)},
                "second": {"label": labels[b], "offset_s": round(offsets[b], 3),
                           "duration_s": round(durations[b], 3)},
                "gap_s": round(gap, 3),
                "shortfall_s": round(min_gap_s - gap, 3),
            })
    return collisions


# --------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------


def _shift_positives(
    positives: list[list[float]], *, offset: float, part_duration: float
) -> tuple[list[list[float]], int, int]:
    """Shift onto the joined clock, dropping/clipping anything past the encoded part's end.

    The re-encode does not trim, but fps normalisation can move the container end by a fraction of
    a frame, and an annotation that already overran its source (Charades does this routinely) must
    not be allowed to claim coverage the media does not contain.
    """
    shifted: list[list[float]] = []
    dropped = 0
    clipped = 0
    for start, end in positives:
        start = float(start)
        end = float(end)
        if start >= part_duration:
            dropped += 1
            continue
        if end > part_duration:
            end = part_duration
            clipped += 1
        shifted.append([round(start + offset, 3), round(end + offset, 3)])
    return shifted, dropped, clipped


def build(args: argparse.Namespace) -> int:
    corpus_path: Path = args.corpus
    if not corpus_path.is_file():
        print(f"[build] refusing: corpus not found at {corpus_path}")
        return 1
    corpus = json.loads(corpus_path.read_text())
    cases = corpus["cases"]
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- probe every candidate's true duration -----------------------------------------------
    # The corpus's ``duration_s`` is absent for repcount and comes from annotation metadata
    # elsewhere; the schedule must be laid out on measured media, so probe once and reuse.
    paths = sorted({c["video_path"] for c in cases})
    print(f"[build] probing {len(paths)} media files", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        probed = list(pool.map(lambda p: (p, probe_duration(REPO_ROOT / p)), paths))
    durations = {p: d for p, d in probed if d is not None}

    inject_datasets = set(args.inject_datasets.split(",")) if args.inject_datasets else None
    strengths = set(args.gt_strengths.split(","))
    unknown = strengths - set(GT_TIERS)
    if unknown:
        print(f"[build] unknown gt strength tier(s): {sorted(unknown)}; known: {list(GT_TIERS)}")
        return 2

    pool_cases = eligible_injections(
        cases, datasets=inject_datasets, strengths=strengths,
        min_s=args.min_inject_s, max_s=args.max_inject_s, durations=durations,
    )
    print(f"[build] {len(pool_cases)} eligible injection cases", flush=True)
    if len(pool_cases) < args.n_inject:
        print(f"[build] refusing: asked for {args.n_inject} injections, only {len(pool_cases)} "
              f"cases are eligible under the current filters")
        return 1
    chosen = select_injections(pool_cases, args.n_inject, args.seed)

    # ---- encode the injections and MEASURE them ----------------------------------------------
    print(f"[build] encoding {len(chosen)} injection clips ({args.jobs} encoders)", flush=True)

    def encode_injection(item: tuple[int, dict[str, Any]]) -> tuple[int, Path | None, float | None]:
        index, case = item
        part = work / f"inj_{index:04d}.mp4"
        if not part.is_file() and not _encode(REPO_ROOT / case["video_path"], part):
            return index, None, None
        return index, part, probe_duration(part)

    encoded: dict[int, tuple[Path | None, float | None]] = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for index, part, duration in pool.map(encode_injection, list(enumerate(chosen))):
            encoded[index] = (part, duration)

    usable: list[tuple[dict[str, Any], Path, float]] = []
    failed: list[str] = []
    for index, case in enumerate(chosen):
        part, duration = encoded.get(index, (None, None))
        if part is None or duration is None or duration <= 0:
            failed.append(case["case_id"])
            continue
        usable.append((case, part, duration))
    if failed:
        print(f"[build] {len(failed)} clip(s) failed to encode: {failed}")
    if len(usable) < 2:
        print("[build] refusing: fewer than two usable injection clips")
        return 1

    # ---- schedule, on the MEASURED durations --------------------------------------------------
    inj_durations = [d for _, _, d in usable]
    labels = [c["case_id"] for c, _, _ in usable]
    if args.schedule_json is not None:
        # An explicit schedule is how injection meets a REAL recording: the offsets come from the
        # recorder's clock, not from a policy. It is also the only way the collision refusal is
        # reachable in practice, which is why it exists here rather than only in the policies.
        raw = json.loads(args.schedule_json.read_text())
        given = raw["offsets_s"] if isinstance(raw, dict) else raw
        if len(given) != len(inj_durations):
            print(f"[build] REFUSING: --schedule-json supplies {len(given)} offsets but "
                  f"{len(inj_durations)} clips were selected. The schedule is applied in the "
                  f"seeded selection order; it cannot be partially specified.")
            return 1
        scheduled = [round(float(v), 3) for v in given]
        feasibility = {"policy": "explicit", "n_clips": len(scheduled),
                       "horizon_s": round(args.target_s, 3),
                       "total_clip_s": round(sum(inj_durations), 3),
                       "min_gap_s": args.min_gap_s, "lead_in_s": args.lead_in_s,
                       "source": str(args.schedule_json)}
    else:
        try:
            scheduled, feasibility = schedule_offsets(
                inj_durations, horizon_s=args.target_s, policy=args.policy,
                min_gap_s=args.min_gap_s, lead_in_s=args.lead_in_s, seed=args.seed,
            )
        except ValueError as exc:
            print(f"[build] REFUSING: {exc}")
            return 1

    collisions = detect_collisions(
        scheduled, inj_durations, min_gap_s=args.min_gap_s, labels=labels
    )
    if collisions:
        print(f"[build] REFUSING: {len(collisions)} schedule collision(s). Injections are never "
              f"silently shifted — a nudged clip makes every offset in the manifest a lie.")
        for collision in collisions[:10]:
            print(f"  {collision['kind']}: {collision['first']['label']} "
                  f"[{collision['first']['offset_s']}..{collision['first']['end_s']}] then "
                  f"{collision['second']['label']} @{collision['second']['offset_s']} "
                  f"— gap {collision['gap_s']}s, short by {collision['shortfall_s']}s")
        return 1

    order = sorted(range(len(usable)), key=lambda i: scheduled[i])

    # ---- fill the gaps with background --------------------------------------------------------
    exclude = {c["video_path"] for c, _, _ in usable}
    bg_datasets = {d for d in args.background_dataset.split(",") if d}
    background = select_background(
        cases, datasets=bg_datasets, exclude_paths=exclude,
        durations=durations, seed=args.seed,
    )
    if not background:
        print(f"[build] refusing: no background media available from "
              f"{sorted(bg_datasets)} after excluding injected clips")
        return 1
    print(f"[build] {len(background)} background clips available "
          f"({sum(durations[c['video_path']] for c in background) / 60:.1f} min)", flush=True)

    #: Background is consumed as one continuous tape: a cursor walks each clip from its start and
    #: moves to the next clip when it runs out. Deterministic, and it keeps background runs as long
    #: as the source allows, which minimises the number of hard cuts a spliced source introduces.
    bg_index = 0
    bg_cursor = 0.0
    bg_requests: list[tuple[int, dict[str, Any], float, float]] = []  # (part#, case, ss, want)
    timeline: list[dict[str, Any]] = []  # ordered plan; "part" filled in below
    part_counter = 0
    clock = 0.0

    def request_background(want: float) -> bool:
        """Queue background cuts totalling ``want`` seconds. False when the tape runs out."""
        nonlocal bg_index, bg_cursor, part_counter, clock
        remaining = want
        while remaining > args.min_bg_part_s:
            if bg_index >= len(background):
                return False
            case = background[bg_index]
            available = durations[case["video_path"]] - bg_cursor
            if available <= args.min_bg_part_s:
                bg_index += 1
                bg_cursor = 0.0
                continue
            take = min(remaining, available)
            bg_requests.append((part_counter, case, bg_cursor, take))
            timeline.append({"role": "background", "part": part_counter, "case": case,
                             "src_in_s": round(bg_cursor, 3), "want_s": round(take, 3)})
            part_counter += 1
            bg_cursor += take
            remaining -= take
        return True

    exhausted = False
    for position, index in enumerate(order):
        case, part, duration = usable[index]
        want = scheduled[index] - clock
        if want > args.min_bg_part_s and not request_background(want):
            exhausted = True
            break
        clock = scheduled[index] + duration
        timeline.append({"role": "injection", "part": part_counter, "case": case,
                         "path": part, "duration_s": duration,
                         "scheduled_offset_s": scheduled[index], "position": position})
        part_counter += 1
    tail = args.target_s - clock
    if not exhausted and tail > args.min_bg_part_s:
        request_background(tail)

    if exhausted:
        print(f"[build] refusing: background media exhausted before the horizon was filled. "
              f"{sum(durations[c['video_path']] for c in background) / 60:.1f} min of background "
              f"is available; the schedule needs "
              f"{(args.target_s - sum(inj_durations)) / 60:.1f} min.")
        return 1

    # ---- encode the background cuts -----------------------------------------------------------
    print(f"[build] encoding {len(bg_requests)} background segments", flush=True)

    def encode_background(
        item: tuple[int, dict[str, Any], float, float]
    ) -> tuple[int, Path | None, float | None]:
        number, case, ss, want = item
        part = work / f"bg_{number:04d}.mp4"
        if not part.is_file() and not _encode(REPO_ROOT / case["video_path"], part, ss=ss, t=want):
            return number, None, None
        return number, part, probe_duration(part)

    bg_encoded: dict[int, tuple[Path | None, float | None]] = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for number, part, duration in pool.map(encode_background, bg_requests):
            bg_encoded[number] = (part, duration)

    # ---- lay the REALIZED clock out on measured part durations --------------------------------
    parts: list[Path] = []
    segments: list[dict[str, Any]] = []
    injections: list[dict[str, Any]] = []
    offset = 0.0
    dropped_total = 0
    clipped_total = 0
    contributors: dict[str, dict[str, Any]] = {}

    def note_contributor(dataset: str, role: str) -> None:
        entry = contributors.setdefault(
            f"{dataset}::{role}",
            {"dataset": dataset, "role": role, "n_parts": 0, **license_for(dataset)},
        )
        entry["n_parts"] += 1

    for item in timeline:
        if item["role"] == "background":
            part, duration = bg_encoded.get(item["part"], (None, None))
            if part is None or duration is None or duration <= 0:
                continue
            case = item["case"]
            segments.append({
                "index": len(segments),
                "role": "background",
                "offset_s": round(offset, 3),
                "duration_s": round(duration, 3),
                "gt": "none",
                "media": {
                    "video_path": case["video_path"],
                    "video_sha256": case.get("video_sha256"),
                    "src_in_s": item["src_in_s"],
                    "src_out_s": round(item["src_in_s"] + duration, 3),
                },
                # Background lifted from a labelled corpus case keeps NO labels. Recording what was
                # discarded keeps the omission visible: a reader can see that this footage has
                # annotations elsewhere and that this manifest deliberately does not use them.
                "labels_discarded": {
                    "origin_case_id": case["case_id"],
                    "n_positives_in_origin": len(case.get("positives") or []),
                    "reason": "background is UNLABELLED, not negative: its annotations describe a "
                              "different condition than the injected goals, and admitting them "
                              "would imply coverage this source does not have",
                },
            })
            note_contributor(case["dataset"], "background")
            parts.append(part)
            offset += duration
            continue

        case = item["case"]
        part = item["path"]
        duration = item["duration_s"]
        strength, gt_kind = GT_STRENGTH.get(case["dataset"], ("derived_span", "interval"))
        shifted, dropped, clipped = _shift_positives(
            case.get("positives") or [], offset=offset, part_duration=duration
        )
        dropped_total += dropped
        clipped_total += clipped
        injection_id = f"{item['position']:03d}::{case['case_id']}"
        entry = {
            "injection_id": injection_id,
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "origin_video_path": case["video_path"],
            "origin_video_sha256": case.get("video_sha256"),
            "goal_text": case["goal_text"],
            "goal_kind": case["goal_kind"],
            "scheduled_offset_s": round(item["scheduled_offset_s"], 3),
            "offset_s": round(offset, 3),
            "schedule_drift_s": round(offset - item["scheduled_offset_s"], 3),
            "source_duration_s": round(case["_duration_s"], 3),
            "duration_s": round(duration, 3),
            "positives_source": [[float(a), float(b)] for a, b in (case.get("positives") or [])],
            "positives": shifted,
            "n_positives_dropped": dropped,
            "n_positives_clipped": clipped,
            "gt_strength": strength,
            "gt_kind": gt_kind,
            "provenance": case.get("provenance"),
        }
        if gt_kind == "point":
            entry["point_tolerance_s"] = args.point_tolerance_s
        injections.append(entry)
        segments.append({
            "index": len(segments),
            "role": "injection",
            "offset_s": round(offset, 3),
            "duration_s": round(duration, 3),
            "injection_id": injection_id,
            "case_id": case["case_id"],
            "goal_text": case["goal_text"],
        })
        note_contributor(case["dataset"], "injection")
        parts.append(part)
        offset += duration

    if len(injections) < 2:
        print("[build] refusing: fewer than two injections survived encoding")
        return 1

    # ---- redistribution gate -------------------------------------------------------------------
    contributor_rows = sorted(contributors.values(), key=lambda r: (r["dataset"], r["role"]))
    redistributable = all(row["redistributable"] for row in contributor_rows)
    if args.require_redistributable and not redistributable:
        blocking = sorted({r["dataset"] for r in contributor_rows if not r["redistributable"]})
        print(f"[build] REFUSING to write a manifest under --require-redistributable: "
              f"non-redistributable contributor(s) {blocking}. A spliced artefact contains the "
              f"pixels of every contributor, background included.")
        return 1

    # ---- concat ---------------------------------------------------------------------------------
    concat_list = work / f"{args.source_id}.concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    output = OUT_DIR / f"{args.source_id}.mp4"
    print(f"[build] concatenating {len(parts)} parts -> {output}", flush=True)
    joined = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", str(output)],
        capture_output=True, text=True, timeout=3600,
    )
    if joined.returncode != 0 or not output.is_file():
        print(f"[build] concat FAILED: {joined.stderr[:2000]}")
        return 1

    container = probe_duration(output)
    content = verify_content(output)
    print(f"[build] content verification: {content}", flush=True)
    if not content.get("ok") and not args.allow_frozen:
        print("[build] REFUSING to publish a manifest: the concatenated source's pixels do not "
              "change across the joined clock. A mismatched-geometry concat produces exactly this "
              "(correct duration, frozen content).")
        return 1

    # ---- watches, keyed on (goal_text, gt_strength) so tiers can never be merged ---------------
    watches: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in injections:
        key = (entry["goal_text"], entry["gt_strength"])
        watch = watches.setdefault(key, {
            "goal_text": entry["goal_text"],
            "goal_kind": entry["goal_kind"],
            "gt_strength": entry["gt_strength"],
            "gt_kind": entry["gt_kind"],
            "datasets": [],
            "injection_ids": [],
            "positives": [],
        })
        watch["datasets"].append(entry["dataset"])
        watch["injection_ids"].append(entry["injection_id"])
        watch["positives"].extend(entry["positives"])
    for watch in watches.values():
        watch["positives"].sort()
        watch["n_occurrences"] = len(watch["positives"])
        watch["n_injections"] = len(watch["injection_ids"])
        watch["datasets"] = sorted(set(watch["datasets"]))

    # ---- totals, per tier, never blended --------------------------------------------------------
    by_strength: dict[str, dict[str, Any]] = {}
    for entry in injections:
        row = by_strength.setdefault(entry["gt_strength"], {
            "n_injections": 0, "n_intervals": 0, "labelled_seconds": 0.0, "datasets": set(),
        })
        row["n_injections"] += 1
        row["n_intervals"] += len(entry["positives"])
        row["labelled_seconds"] += sum(b - a for a, b in entry["positives"])
        row["datasets"].add(entry["dataset"])
    for row in by_strength.values():
        row["labelled_seconds"] = round(row["labelled_seconds"], 3)
        row["datasets"] = sorted(row["datasets"])

    # ---- advisory: does an injected goal read like the background's own content? ----------------
    # Background is unlabelled, so an injected condition that is ALSO plausibly true in the
    # background weakens the source further: an early fire may be a correct detection of unlabelled
    # footage rather than a defect. Cheap lexical overlap is not a semantic check, but it points at
    # the goals where that risk is highest instead of leaving it unstated.
    stop = {"a", "an", "the", "is", "are", "of", "to", "in", "on", "at", "with", "from", "and",
            "someone", "something", "person", "people", "it", "they", "their", "into", "for",
            "by", "up", "down", "out", "off", "over", "completes", "how", "many", "count", "when"}

    def content_words(text: str) -> set[str]:
        return {
            token.strip(".,!?'\"").lower()
            for token in str(text).split()
            if len(token) > 3 and token.strip(".,!?'\"").lower() not in stop
        }

    bg_words: dict[str, set[str]] = {}
    for segment in segments:
        if segment["role"] != "background":
            continue
        origin = segment["labels_discarded"]["origin_case_id"]
        bg_case = next((c for c in cases if c["case_id"] == origin), None)
        if bg_case is not None:
            bg_words[origin] = content_words(bg_case["goal_text"])
    overlaps = []
    for entry in injections:
        words = content_words(entry["goal_text"])
        hits = sorted({origin for origin, other in bg_words.items() if words & other})
        if hits:
            overlaps.append({"injection_id": entry["injection_id"],
                             "goal_text": entry["goal_text"],
                             "background_case_ids": hits[:5],
                             "n_background_cases": len(hits)})

    manifest = {
        "kind": SCHEMA_KIND,
        "version": SCHEMA_VERSION,
        "source_id": args.source_id,
        "generated_utc": (None if args.no_timestamp
                          else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "build": {
            "tool": "eval/longhorizon/build_injected_source.py",
            "seed": args.seed,
            "policy": args.policy,
            "target_s": args.target_s,
            "n_inject_requested": args.n_inject,
            "min_gap_s": args.min_gap_s,
            "lead_in_s": args.lead_in_s,
            "min_inject_s": args.min_inject_s,
            "max_inject_s": args.max_inject_s,
            "min_bg_part_s": args.min_bg_part_s,
            "background_dataset": args.background_dataset,
            "inject_datasets": args.inject_datasets,
            "gt_strengths": args.gt_strengths,
            "point_tolerance_s": args.point_tolerance_s,
            "corpus": str(corpus_path.relative_to(REPO_ROOT)),
            "corpus_version": corpus.get("corpus_version"),
            "corpus_sha256": sha256_file(corpus_path),
            "corpus_selection_seed": corpus.get("selection_seed"),
        },
        "video_path": str(output.relative_to(REPO_ROOT)),
        "video_sha256": sha256_file(output),
        "total_duration_s": round(offset, 3),
        "container_duration_s": None if container is None else round(container, 3),
        "drift_s": None if container is None else round(container - offset, 3),
        "profile": {
            "scale_width": SCALE_WIDTH, "scale_height": SCALE_HEIGHT, "vfilter": VFILTER,
            "fps": TARGET_FPS, "crf": CRF, "pix_fmt": PIX_FMT, "audio": False,
        },
        "clock": {
            "media_epoch_s": 0.0,
            "wall_clock_anchor": args.wall_clock_anchor,
            "anchor_semantics": "declared" if args.wall_clock_anchor else "none",
            "monotonic": True,
            "unobserved": [],
            "note": ("Spliced from archival clips: media t=0 corresponds to no real instant, and "
                     "schedule-derived ground truth — sunrise, transit, timed lighting — is NOT "
                     "valid on this source. `unobserved` stays empty because a spliced source has "
                     "no recorder outage; the field exists so a replay adapter reads the same "
                     "shape for a recorded source, where a gap is neither positive nor negative."),
        },
        "license": {
            "fixture": {"id": f"{args.source_id}", "redistributable": redistributable},
            "contributors": contributor_rows,
        },
        "redistributable": redistributable,
        "totals": {
            "n_segments": len(segments),
            "n_background_segments": sum(1 for s in segments if s["role"] == "background"),
            "n_injections": len(injections),
            "injected_media_s": round(sum(e["duration_s"] for e in injections), 3),
            "background_media_s": round(
                sum(s["duration_s"] for s in segments if s["role"] == "background"), 3),
            "n_watches": len(watches),
            "n_intervals_by_strength": {k: v["n_intervals"] for k, v in by_strength.items()},
            "labelled_seconds_by_strength": {
                k: v["labelled_seconds"] for k, v in by_strength.items()},
            "by_strength": by_strength,
            "note": ("There is deliberately no blended labelled-seconds total. The tiers measure "
                     "different things and a number computed across them means nothing — the "
                     "discipline build_audio_corpus.py established for audio ground truth."),
        },
        "checks": {
            "schedule_feasibility": feasibility,
            "collisions": collisions,
            "n_injections_failed_encode": len(failed),
            "case_ids_failed_encode": failed,
            "positives_dropped_past_part_end": dropped_total,
            "positives_clipped_at_part_end": clipped_total,
            "max_schedule_drift_s": round(
                max(abs(e["schedule_drift_s"]) for e in injections), 3),
            "tiling": _check_tiling(segments, offset),
            "goal_background_lexical_overlap": {
                "n_flagged": len(overlaps),
                "note": ("ADVISORY, not a defect. Background is unlabelled; where an injected "
                         "goal shares content words with a background clip's own annotation, a "
                         "fire outside an injected interval is more likely a correct detection of "
                         "unlabelled footage than a false positive."),
                "flagged": overlaps[:40],
            },
        },
        "content_verification": content,
        "watches": sorted(watches.values(), key=lambda w: (w["gt_strength"], w["goal_text"])),
        "injections": injections,
        "segments": segments,
    }

    manifest_path = OUT_DIR / f"{args.source_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    if not args.keep_parts:
        shutil.rmtree(work, ignore_errors=True)

    size_mb = output.stat().st_size / 1e6
    print(f"[build] {output}  {offset / 60:.1f} min joined clock "
          f"(container {(container or 0) / 60:.1f} min, drift {manifest['drift_s']}s, "
          f"{size_mb:.1f} MB)")
    print(f"[build] {len(injections)} injections, {len(watches)} watches, "
          f"{len(segments)} segments, redistributable={redistributable}")
    for tier in GT_TIERS:
        row = by_strength.get(tier)
        if row:
            print(f"[build]   {tier:20s} {row['n_injections']:3d} injections, "
                  f"{row['n_intervals']:4d} intervals, {row['labelled_seconds']:.1f} labelled s")
    print(f"[build] manifest -> {manifest_path}")
    return 0


def _check_tiling(segments: list[dict[str, Any]], total: float) -> dict[str, Any]:
    """The segments must tile [0, total] exactly: no hole, no overlap, in order."""
    holes: list[dict[str, Any]] = []
    cursor = 0.0
    for segment in segments:
        delta = segment["offset_s"] - cursor
        if abs(delta) > 0.002:
            holes.append({"index": segment["index"], "expected_offset_s": round(cursor, 3),
                          "declared_offset_s": segment["offset_s"], "delta_s": round(delta, 3)})
        cursor = segment["offset_s"] + segment["duration_s"]
    return {
        "ok": not holes and abs(cursor - total) <= 0.002,
        "n_discontinuities": len(holes),
        "discontinuities": holes[:10],
        "end_s": round(cursor, 3),
        "total_duration_s": round(total, 3),
    }


# --------------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------------


def verify(manifest_path: Path, *, check_origins: bool, deep: bool) -> int:
    """Re-derive every manifest claim from the media on disk. Non-zero exit on any mismatch."""
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    warnings: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
        if not ok:
            failures.append(message)

    print(f"verify {manifest_path}")
    check(manifest.get("kind") == SCHEMA_KIND, f"kind == {SCHEMA_KIND}")
    check(manifest.get("version") == SCHEMA_VERSION, f"version == {SCHEMA_VERSION}")

    media = REPO_ROOT / manifest["video_path"]
    check(media.is_file(), f"media present: {manifest['video_path']}")
    if not media.is_file():
        return 1

    digest = sha256_file(media)
    check(digest == manifest["video_sha256"],
          f"media sha256 matches the manifest ({digest[:12]}…)")

    container = probe_duration(media)
    total = float(manifest["total_duration_s"])
    check(container is not None and abs(container - total) < 1.0,
          f"container duration {container:.3f}s vs joined clock {total:.3f}s "
          f"(drift {None if container is None else round(container - total, 3)}s)")

    tiling = _check_tiling(manifest["segments"], total)
    check(tiling["ok"], f"segments tile [0, {total:.1f}] exactly "
                        f"({tiling['n_discontinuities']} discontinuities)")

    roles = {s["role"] for s in manifest["segments"]}
    check(roles <= {"background", "injection"}, f"segment roles are known: {sorted(roles)}")
    check(all(s.get("gt") == "none" for s in manifest["segments"] if s["role"] == "background"),
          "every background segment is gt='none' (unlabelled, not negative)")

    injection_ids = {s["injection_id"] for s in manifest["segments"] if s["role"] == "injection"}
    declared = {e["injection_id"] for e in manifest["injections"]}
    check(injection_ids == declared,
          f"segment injections match the injections list ({len(declared)})")

    bad_interval = []
    outside_own_span = []
    for entry in manifest["injections"]:
        span = (entry["offset_s"], entry["offset_s"] + entry["duration_s"])
        for start, end in entry["positives"]:
            if not (0.0 <= start <= end <= total + 0.05):
                bad_interval.append((entry["injection_id"], start, end))
            if start < span[0] - 0.05 or end > span[1] + 0.05:
                outside_own_span.append((entry["injection_id"], start, end, span))
    check(not bad_interval, f"every positive lies inside [0, {total:.1f}] "
                            f"({len(bad_interval)} violations)")
    check(not outside_own_span,
          f"every positive lies inside its own injected clip's span "
          f"({len(outside_own_span)} violations)")

    strengths = {e["gt_strength"] for e in manifest["injections"]}
    check(strengths <= set(GT_TIERS), f"gt_strength tiers are known: {sorted(strengths)}")
    check(all("point_tolerance_s" in e for e in manifest["injections"]
              if e["gt_kind"] == "point"),
          "every point-annotated injection carries an explicit tolerance")
    check("labelled_seconds_by_strength" in manifest["totals"],
          "totals are reported per tier, not blended")

    watch_keys = {(w["goal_text"], w["gt_strength"]) for w in manifest["watches"]}
    check(len(watch_keys) == len(manifest["watches"]),
          "watches are keyed on (goal_text, gt_strength) with no duplicates")

    contributors = manifest["license"]["contributors"]
    expected = all(c["redistributable"] for c in contributors)
    check(manifest["redistributable"] == expected,
          f"redistributable is the AND over all contributors ({expected})")

    check(bool(manifest.get("content_verification", {}).get("ok")),
          "pixels change across the joined clock (recorded at build time)")

    if check_origins:
        missing = 0
        mismatched = 0
        for entry in manifest["injections"]:
            origin = REPO_ROOT / entry["origin_video_path"]
            if not origin.is_file():
                missing += 1
                continue
            if entry.get("origin_video_sha256") and sha256_file(origin) != entry["origin_video_sha256"]:
                mismatched += 1
        check(mismatched == 0,
              f"origin media still hashes to the bytes the labels were annotated against "
              f"({mismatched} mismatched)")
        if missing:
            warnings.append(f"{missing} origin clip(s) not on this machine — hashes unchecked")

    if deep:
        try:
            import cv2

            cap = cv2.VideoCapture(str(media))
            unreadable = []
            for entry in manifest["injections"]:
                probe_at = entry["offset_s"] + min(1.0, entry["duration_s"] / 2)
                cap.set(cv2.CAP_PROP_POS_MSEC, probe_at * 1000.0)
                ok, _frame = cap.read()
                if not ok:
                    unreadable.append(entry["injection_id"])
            cap.release()
            check(not unreadable,
                  f"a frame decodes at every injection offset ({len(unreadable)} unreadable)")
        except ImportError:
            warnings.append("opencv unavailable — skipped per-injection decode probe")

    for warning in warnings:
        print(f"  [WARN] {warning}")
    print(f"verify: {'OK' if not failures else str(len(failures)) + ' FAILURE(S)'}")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build (or verify) a long-horizon source with injected frame-exact ground truth."
    )
    ap.add_argument("--verify", type=Path, default=None,
                    help="verify an existing manifest against its media and exit")
    ap.add_argument("--verify-origins", action="store_true",
                    help="with --verify, also re-hash every origin clip (slow, thorough)")
    ap.add_argument("--verify-deep", action="store_true",
                    help="with --verify, also decode a frame at every injection offset")

    ap.add_argument("--source-id", default="inject_2h_v1", help="slug for the mp4 and manifest")
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                    help="label source; defaults to the tracked corpus.v1.1.json")
    ap.add_argument("--target-s", type=float, default=7200.0, help="horizon in seconds")
    ap.add_argument("--n-inject", type=int, default=32, help="number of clips to inject")
    ap.add_argument("--seed", type=int, default=20260731,
                    help="every random choice derives from this; a build is a pure function of it")
    ap.add_argument("--policy", default="jittered", choices=("even", "jittered", "bursty"))
    ap.add_argument("--schedule-json", type=Path, default=None,
                    help="explicit offsets ({\"offsets_s\": [...]} or a bare list), applied in the "
                         "seeded selection order and checked for collisions like any other")
    ap.add_argument("--min-gap-s", type=float, default=45.0,
                    help="minimum unlabelled background between two injections")
    ap.add_argument("--lead-in-s", type=float, default=60.0,
                    help="background before the first injection, so a watch has arming room")
    ap.add_argument("--min-inject-s", type=float, default=8.0)
    ap.add_argument("--max-inject-s", type=float, default=60.0)
    ap.add_argument("--min-bg-part-s", type=float, default=2.0,
                    help="background cuts shorter than this are not worth a part")
    ap.add_argument("--background-dataset", default="proactiveqa,omnipro",
                    help="comma-separated dataset(s) supplying unlabelled background footage")
    ap.add_argument("--inject-datasets", default="",
                    help="comma-separated allow-list; empty means every eligible dataset")
    ap.add_argument("--gt-strengths", default="annotated_interval",
                    help="comma-separated tiers eligible for injection. The default is the single "
                         "strongest tier: mixing tiers in one source invites a scorer to average "
                         "across them.")
    ap.add_argument("--point-tolerance-s", type=float, default=POINT_TOLERANCE_S)
    ap.add_argument("--wall-clock-anchor", default=None,
                    help="ISO-8601 UTC that media t=0 stands for. Recorded as 'declared' — a "
                         "fiction on a spliced source, and it licenses no diurnal claim.")
    ap.add_argument("--require-redistributable", action="store_true",
                    help="refuse to write a manifest if any contributor cannot be redistributed")
    ap.add_argument("--allow-frozen", action="store_true",
                    help="escape hatch: publish even if the pixel-liveness probe fails")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="omit generated_utc so a rebuild is byte-identical")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--work", type=Path, default=Path("/tmp/percept-longhorizon-parts"),
                    help="scratch for intermediate re-encoded parts; deleted after the concat")
    args = ap.parse_args()

    if args.verify is not None:
        return verify(args.verify, check_origins=args.verify_origins, deep=args.verify_deep)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
