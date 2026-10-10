"""Dataset adapters. Each returns ``VideoCase`` lists and declares what it can't express.

* ``perception-test``: Perception Test (CC-BY 4.0) action-localisation labels, one case per
  clip. Each label present in the clip becomes the event condition "<label> begins", plus
  ``negatives`` labels absent from the clip so precision and false events per hour are defined.
  Cannot express: states (every label is an onset event), or anything a label doesn't name.
* ``perception-test-sparse``: a declared-synthetic long-sparse construction. Clips are joined
  with idle gaps (a held frame with sensor noise) and distractor clips (none of the watched
  labels) so the target clips fill a chosen share of the timeline (``occupancy``). It supports
  the cost-versus-occupancy curve and is never natural footage.
* ``meva``: MEVA (CC BY 4.0), fixed indoor and outdoor cameras, official activity annotations
  (``*.activities.yml``) next to the videos fetched by ``fetch_meva.py``. The primary real
  continuous set. Condition text is mechanical: "<activity with spaces> begins". Only entry
  activities are detector-expressible (as "a person appears").
* ``local:<dir>``: a directory of MP4s plus ``labels.json`` (or ``labels.yaml``), for private
  clips evaluated locally; nothing from it is committed. Format::

      {"conditions": [{"label": "person falls", "text": "a person falls",
                       "kind": "event", "detector": "person"}],
       "clips": [{"file": "a.mp4", "occurrences": [{"label": "person falls", "start": 12.0, "end": 14.5}]}]}

* ``synthetic``: generated clips (coloured squares appearing on a textured wall) for CI.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

from benchmarks.cases import CondSpec, Occurrence, Segment, VideoCase, merge_overlapping

REPO = Path(__file__).resolve().parents[1]
PT_ROOT = REPO / "eval" / "bench-staging" / "perception_test" / "raw"
PT_LABELS = "challenge_action_localisation_valid_annotations.zip"
MEVA_ROOT = REPO / "eval" / "bench-staging" / "meva"
MEVA_PERSON = ("person_enters_scene_through_structure",)    # "a person appears" for a person detector


def _video_seconds(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    finally:
        cap.release()


def _with_negatives(case_id: str, present: list[str], vocab: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(f"{seed}:{case_id}")
    absent = sorted(set(vocab) - set(present))
    return sorted(set(present)) + sorted(rng.sample(absent, min(n, len(absent))))


# --------------------------------------------------------------------------- Perception Test


def _pt_annotations(root: Path) -> dict:
    import zipfile
    with zipfile.ZipFile(root / PT_LABELS) as z:
        return json.loads(z.read(z.namelist()[0]))


def perception_test(root: Path = PT_ROOT, *, limit: int | None = None, negatives: int = 2, seed: int = 0,
                    detector_map: dict | None = None) -> list[VideoCase]:
    labels = _pt_annotations(root)
    vocab = sorted({a["label"] for v in labels.values() for a in v.get("action_localisation", [])})
    cases = []
    for vid in sorted(labels):
        path = root / "videos" / f"{vid}.mp4"
        acts = labels[vid].get("action_localisation", [])
        if not path.exists() or not acts:
            continue
        meta = labels[vid]["metadata"]
        duration = meta["num_frames"] / meta["frame_rate"]
        occs = merge_overlapping([Occurrence(a["label"], a["timestamps"][0] / 1e6, a["timestamps"][1] / 1e6)
                                  for a in acts])
        watched = _with_negatives(vid, [o.label for o in occs], vocab, negatives, seed)
        conds = tuple(CondSpec(lb, f"{lb} begins", "event", (detector_map or {}).get(lb)) for lb in watched)
        cases.append(VideoCase(vid, "perception-test", (Segment(str(path), 0.0, duration),), conds, tuple(occs)))
        if limit and len(cases) >= limit:
            break
    return cases


def long_sparse(pool: list[VideoCase], *, occupancies=(0.05, 0.2, 0.5), length_s: float = 600.0,
                per_level: int = 1, negatives: int = 2, seed: int = 0) -> list[VideoCase]:
    """Declared-synthetic long timelines at several occupancy levels (see the module docstring)."""
    rng = random.Random(seed)
    vocab = sorted({c.label for case in pool for c in case.conditions})
    out = []
    for occ in occupancies:
        for k in range(per_level):
            targets, used = [], 0.0
            for case in rng.sample(pool, len(pool)):
                if used >= occ * length_s:
                    break
                targets.append(case)
                used += case.duration
            watched = sorted({o.label for c in targets for o in c.occurrences})
            distract = [c for c in pool if c not in targets and not {o.label for o in c.occurrences} & set(watched)]
            gap = max(0.0, length_s - used) / max(1, len(targets))
            segments, occs, t = [], [], 0.0
            for case in targets:
                fill = gap
                if distract and rng.random() < 0.5:     # half the gaps carry a distractor clip
                    d = rng.choice(distract)
                    take = min(fill, d.duration)
                    segments.append(Segment(d.segments[0].path, 0.0, take))
                    fill -= take
                    t += take
                if fill > 0:
                    segments.append(Segment(case.segments[0].path, 0.0, 0.0, hold=round(fill, 1)))
                    t += round(fill, 1)
                segments.append(case.segments[0])
                occs += [Occurrence(o.label, t + o.start, t + o.end) for o in case.occurrences]
                t += case.duration
            cid = f"sparse-{int(occ * 100):02d}pct-{k}"
            conds = tuple(CondSpec(lb, f"{lb} begins") for lb in _with_negatives(cid, watched, vocab, negatives, seed))
            out.append(VideoCase(cid, "perception-test-sparse", tuple(segments), conds, tuple(occs), synthetic=True,
                                 notes=f"synthetic: {len(targets)} clips at {used / max(t, 1):.0%} occupancy"))
    return out


# --------------------------------------------------------------------------- MEVA


_ACT = re.compile(r"'?act2'?\s*:\s*\{\s*'?([A-Za-z_]+)'?")
_SPAN = re.compile(r"'?tsr0'?\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def meva_activities(text: str, fps: float) -> list[Occurrence]:
    """Occurrences from one ``*.activities.yml`` (one flow-style record per line)."""
    out = []
    for line in text.splitlines():
        name, spans = _ACT.search(line), _SPAN.findall(line)
        if name and spans:
            start = min(int(a) for a, _ in spans) / fps
            end = max(int(b) for _, b in spans) / fps
            out.append(Occurrence(name.group(1), start, end))
    return out


def meva(root: Path = MEVA_ROOT, *, limit: int | None = None, negatives: int = 2, seed: int = 0) -> list[VideoCase]:
    import cv2
    raw = []
    for yml in sorted((root / "annotations").rglob("*.activities.yml")):
        stem = yml.name[: -len(".activities.yml")]
        video = next((p for p in (root / "videos").glob(f"{stem}*") if p.suffix in (".avi", ".mp4")), None)
        if video is None:
            continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        raw.append((stem, str(video), merge_overlapping(meva_activities(yml.read_text(), fps))))
    vocab = sorted({o.label for _, _, occs in raw for o in occs})
    cases = []
    for stem, video, occs in raw[:limit]:
        watched = _with_negatives(stem, [o.label for o in occs], vocab, negatives, seed)
        conds = tuple(CondSpec(lb, f"{lb.replace('_', ' ')} begins", "event",
                               "person" if lb in MEVA_PERSON else None) for lb in watched)
        cases.append(VideoCase(stem, "meva", (Segment(video, 0.0, _video_seconds(video)),), conds, tuple(occs)))
    return cases


# --------------------------------------------------------------------------- local clips


def local(directory: Path) -> list[VideoCase]:
    directory = Path(directory)
    path = next((directory / n for n in ("labels.json", "labels.yaml", "labels.yml") if (directory / n).exists()), None)
    if path is None:
        raise FileNotFoundError(f"{directory} needs labels.json or labels.yaml")
    if path.suffix == ".json":
        spec = json.loads(path.read_text())
    else:
        import yaml  # optional: pip install pyyaml
        spec = yaml.safe_load(path.read_text())
    conds = tuple(CondSpec(c["label"], c.get("text", c["label"]), c.get("kind", "event"), c.get("detector"))
                  for c in spec["conditions"])
    cases = []
    for clip in spec["clips"]:
        video = str(directory / clip["file"])
        occs = merge_overlapping([Occurrence(o["label"], float(o["start"]), float(o["end"]))
                                  for o in clip.get("occurrences", [])])
        cases.append(VideoCase(Path(clip["file"]).stem, f"local:{directory.name}",
                               (Segment(video, 0.0, _video_seconds(video)),), conds, tuple(occs)))
    return cases


# --------------------------------------------------------------------------- synthetic (CI)


SYNTH = {"red square appears": (40, 40, 220), "blue square appears": (220, 60, 40), "green square appears": (40, 200, 40)}


def synthetic(directory: Path, *, clips: int = 2, seconds: float = 12.0, fps: int = 10) -> list[VideoCase]:
    """Generated MP4s: a textured wall where a coloured square appears and stays a while. The
    green square never appears: an absent-condition row. All three are motion-expressible."""
    import cv2
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:120, 0:160]
    wall = ((x * 7 + y * 13) % 200 + 20).astype(np.uint8)
    cases = []
    for i in range(clips):
        rng = random.Random(i)
        occs = [Occurrence("red square appears", 2.0 + i, 5.0 + i),
                Occurrence("blue square appears", seconds - 5.0 - rng.random(), seconds - 2.0)]
        path = directory / f"synthetic_{i}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
        for n in range(int(seconds * fps)):
            t, frame = n / fps, np.dstack([wall] * 3).copy()
            for o, (x0, y0) in zip(occs, ((20, 20), (100, 60))):
                if o.start <= t < o.end:
                    frame[y0:y0 + 40, x0:x0 + 40] = SYNTH[o.label]
            writer.write(frame)
        writer.release()
        conds = tuple(CondSpec(lb, f"a {lb.replace(' appears', '')} appears", "event", "motion") for lb in SYNTH)
        cases.append(VideoCase(f"synthetic-{i}", "synthetic", (Segment(str(path), 0.0, seconds),), conds,
                               tuple(occs), synthetic=True, notes="generated for CI"))
    return cases


def load(spec: str, *, limit: int | None = None, root: str | None = None, negatives: int = 2, seed: int = 0,
         occupancies=(0.05, 0.2, 0.5), workdir: Path | None = None) -> list[VideoCase]:
    """Cases for a ``--dataset`` spec."""
    if spec == "perception-test":
        return perception_test(Path(root) if root else PT_ROOT, limit=limit, negatives=negatives, seed=seed)
    if spec == "perception-test-sparse":
        pool = perception_test(Path(root) if root else PT_ROOT, limit=200, negatives=0, seed=seed)
        return long_sparse(pool, occupancies=occupancies, negatives=negatives, seed=seed)[:limit]
    if spec == "meva":
        return meva(Path(root) if root else MEVA_ROOT, limit=limit, negatives=negatives, seed=seed)
    if spec.startswith("local:"):
        return local(Path(spec[6:]))[:limit]
    if spec == "synthetic":
        return synthetic((workdir or Path(os.getcwd())) / "synthetic-media", clips=limit or 2)
    raise ValueError(f"unknown dataset {spec!r}")
