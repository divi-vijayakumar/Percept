"""Evaluating watch plans on your own labelled clips, offline.

Experimental: ``percept.plans.run`` may change before it is stable.

The flow: labels -> a frozen split -> the trigger stage alone on dev (no judge calls) -> two
configurations you declare, side by side on validation -> register the one you chose -> score the
test split once. Choosing between configurations is yours; the toolkit only reports.

The clips are drawn here (a red square appears for a few seconds) and the judge is ``FakeJudge``,
so nothing leaves the machine. It reads and writes video files with OpenCV, the ``video`` extra:
``pip install "percept-vision[video]"``, then ``python examples/plan_evaluation.py``.
"""

import tempfile
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit('This example needs OpenCV, the video extra: pip install "percept-vision[video]"')

from percept import PriceModel, plans
from percept.judges import FakeJudge
from percept.plans import run as evaluation

FPS, SECONDS = 10, 8.0


def draw_clip(path: Path) -> None:
    y, x = np.mgrid[0:120, 0:160]
    wall = ((x * 7 + y * 13) % 200 + 20).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 120))
    for n in range(int(SECONDS * FPS)):
        frame = np.dstack([wall] * 3).copy()
        if 2.0 <= n / FPS < 5.0:
            frame[20:60, 20:60] = (40, 40, 220)
        writer.write(frame)
    writer.release()


def judge_script(request):
    """Stands in for a vision model: a red square is visible in the newest image."""
    im = request.images[-1]
    seen = bool(((im[..., 2] > 180) & (im[..., 0] < 90)).any())
    return {i.condition.id: ("yes", 0.95, "a red square") if seen else ("no", 0.9, "nothing") for i in request.items}


def plan(every_s=None) -> dict:
    concern = {"id": "red", "source": "cam", "description": "a red square appears",
               "judgment": {"route": "fast"}, "decision": {"kind": "event"}}
    if every_s is not None:
        concern["attention"] = {"any": [{"every": every_s}]}
    return {"schema_version": plans.SCHEMA_VERSION, "plan_id": "square-watch", "revision": 1,
            "sources": [{"id": "cam", "kind": "video", "max_side": 160}],
            "budget": {"max_requests_per_min": 600},     # a demo budget
            "concerns": [concern]}


def groups_for_each_split() -> list:
    """Recording groups that land in each split (splits are by a hash of the group name)."""
    out = {}
    for i in range(1000):
        out.setdefault(plans.labels.split_of(f"day-{i}"), f"day-{i}")
    return [out[s] for s in plans.labels.SPLITS]


def main() -> None:
    root = Path(tempfile.mkdtemp())
    clips = []
    for group in groups_for_each_split():
        draw_clip(root / f"{group}.avi")
        clips.append({"file": f"{group}.avi", "group": group,
                      "occurrences": [{"label": "red", "start": 2.0, "end": 5.0}]})
    labels = plans.load_labels({"conditions": [{"label": "red", "text": "a red square appears"}], "clips": clips})
    manifest = plans.split(labels)
    print("split:", {s: len(f) for s, f in manifest.clips.items()}, "manifest", manifest.manifest_hash[:12])

    def compiled(every_s=None):
        return plans.compile(plan(every_s), {"fast": FakeJudge(judge_script, price=PriceModel(0.10, 0.40, 258))})

    # 1. The trigger stage alone: which occurrences would any look have seen? No judge calls.
    runs = evaluation.attention_only(compiled(), labels, manifest.clips["dev"], video_root=root)
    s = plans.evaluate(labels, runs)
    print(f"dev, attention only: candidate {s.funnel['candidate']}/{s.occurrences}, ${s.cost_usd:.4f}")

    # 2. Two configurations we declare, side by side on validation. Nothing is ranked.
    configs = {"on_change (default)": compiled(), "every 3 s": compiled(every_s=3.0)}
    scores = evaluation.sweep(configs, labels, manifest, split="validation", video_root=root,
                              judge_cap_usd=0.05, speed=4.0)
    for name, score in scores.items():
        print(f"validation, {name}: recall {score.recall}, false alerts {score.fp}, ${score.cost_usd:.4f}")

    # 3. Freeze the configuration we chose, then score the test split once.
    chosen = configs["on_change (default)"]
    registration = evaluation.register(chosen, manifest, labels, split="test", created="2026-10-12T00:00:00Z")
    test = evaluation.score_split(chosen, labels, manifest, split="test", video_root=root, judge_cap_usd=0.05,
                                  registration=registration, speed=4.0)
    print(f"test: recall {test.recall}, false alerts {test.fp}, registration {registration['registration_hash'][:12]}")


if __name__ == "__main__":
    main()
