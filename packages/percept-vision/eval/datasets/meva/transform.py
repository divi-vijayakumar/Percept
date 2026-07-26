"""MEVA KPF activities -> Percept ``facility_observation`` fixtures + a pre-registered split.

The transform, end to end:

1. Read every ``*.activities.yml`` clip in a MEVA KPF directory (:mod:`eval.datasets.meva.kpf`).
2. Map each recognized activity type onto a ``facility_observation`` condition
   (:mod:`eval.datasets.meva.pack_ext`); drop the rest.
3. Emit one :class:`percept.eval.differential.Fixture` per ``(clip, condition)`` — its episodes are
   that condition's activities, and each episode's fire timestamp is the activity's BEGIN
   (``begin_frame / fps`` seconds), exactly as :class:`percept.harness.golden_eval.GroundTruthEpisode`
   wants it.
4. Write each fixture's on-disk ``ground_truth.json`` so the whole tree loads back through
   :func:`percept.eval.differential.load_fixtures`.
5. Partition the fixtures into a 50/50 held-out split BY SCENE (camera-location recording session),
   never by clip — MEVA is multiview, so a clip-level split would leak the same event across
   train/test. Written as ``splits.json`` (fixture-id level) so it round-trips through
   :func:`percept.eval.differential.load_held_out_split`.
6. Write the acceptance ``bars.json`` (:data:`DEFAULT_BARS`) and a ``registration.json`` — a
   :class:`MevaSplitRegistration` (a :class:`percept.eval.derivation.DerivationPreRegistration`-style
   content-addressed record) freezing the fixture-set hash, the bars hash and the exact scene split,
   so a later reader can prove none of them moved after the numbers were produced.

Run it on a real MEVA checkout::

    python -m eval.datasets.meva.transform --kpf-dir /path/to/meva/kpf --out ./meva-fixtures

or regenerate the committed synthetic sample (``testdata/`` -> ``fixtures/``)::

    python -m eval.datasets.meva.transform --sample
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from percept.contracts import GroundTruth
from percept.eval.derivation import bars_hash, fixture_set_hash
from percept.eval.differential import (
    Bar,
    BarsDocument,
    Fixture,
    HeldOutSplit,
    load_fixtures,
    load_held_out_split,
    scope_for_goal_shape,
)
from percept.harness.golden_eval import GroundTruthEpisode
from percept.tracing import content_hash

from .kpf import MEVA_DEFAULT_FPS, MevaActivity, MevaClip, iter_kpf_dir
from .pack_ext import ConditionSpec, condition_for

__all__ = [
    "MEVA_ANNOTATION_VERSION",
    "DEFAULT_ALLOWED_LATENESS_S",
    "DEFAULT_BARS",
    "DATASET_ROOT",
    "TESTDATA_DIR",
    "FIXTURES_DIR",
    "MevaFixture",
    "MevaSplitRegistration",
    "sanitize",
    "activity_to_episode",
    "build_fixtures",
    "scene_split",
    "write_fixture_tree",
    "write_dataset",
    "regenerate_sample",
    "main",
]

#: Stamped onto every episode's ground truth so a relabel changes the fixture-set hash.
MEVA_ANNOTATION_VERSION = "meva-kpf-1"

#: Grace (seconds) past an episode's end a still-correct fire may land in (judge/queue latency
#: headroom). A documented default, overridable per run; it never moves the BEGIN fire anchor.
DEFAULT_ALLOWED_LATENESS_S = 2.0

#: Acceptance bars for a MEVA ``facility_observation`` differential. Client-style
#: bars: candidate recall must clear 0.80; the paired recall improvement must be >= 0.05 AND its CI
#: must exclude zero (a point estimate is not enough); and the candidate must add NO false fires.
DEFAULT_BARS = BarsDocument(
    source="eval.datasets.meva.transform.DEFAULT_BARS",
    declared_by="percept DS1 (MEVA -> facility_observation acceptance bars)",
    declared_at="2026-07-23",
    bars=(
        Bar(metric="candidate_recall", direction="min", value=0.80),
        Bar(metric="recall_delta", direction="min", value=0.05, require_ci_excludes_zero=True),
        Bar(metric="false_fire_delta", direction="max", value=0.0),
    ),
)

#: Locations of the committed synthetic sample (dev-tooling only).
DATASET_ROOT = Path(__file__).resolve().parent
TESTDATA_DIR = DATASET_ROOT / "testdata"
FIXTURES_DIR = DATASET_ROOT / "fixtures"


def sanitize(token: str) -> str:
    """Make a MEVA id filesystem/split safe: ``.`` -> ``_`` (dir names + split ids stay flat)."""
    return token.replace(".", "_")


@dataclass(frozen=True)
class MevaFixture:
    """A produced :class:`~percept.eval.differential.Fixture` plus the MEVA provenance the on-disk
    ``Fixture`` cannot carry — the SCENE it belongs to (the split unit) and the source clip/condition.
    """

    fixture: Fixture
    scene_id: str
    clip_id: str
    condition: ConditionSpec

    @property
    def fixture_id(self) -> str:
        return self.fixture.fixture_id


def activity_to_episode(
    activity: MevaActivity,
    *,
    fixture_id: str,
    fps: float,
    allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
) -> GroundTruthEpisode:
    """Turn one MEVA activity into a :class:`~percept.harness.golden_eval.GroundTruthEpisode`.

    The activity BEGIN (``begin_frame / fps``) becomes BOTH ``event_start_ts`` and
    ``decision_eligible_ts`` — the earliest instant a correct fire may legally land, i.e. the fire
    timestamp the dataset contract pins to the activity begin. The activity END becomes ``episode_end_ts``.
    """
    begin_ts = activity.begin_ts(fps)
    end_ts = activity.end_ts(fps)
    return GroundTruthEpisode(
        episode_id=f"{activity.activity_type}-{activity.activity_id}",
        ground_truth=GroundTruth(
            event_start_ts=begin_ts,
            decision_eligible_ts=begin_ts,
            annotation_version=MEVA_ANNOTATION_VERSION,
            annotator_or_fixture_id=fixture_id,
        ),
        episode_end_ts=end_ts,
        allowed_lateness_s=allowed_lateness_s,
    )


def _fixture_id(clip_id: str, template: str) -> str:
    return f"{sanitize(clip_id)}__{template}"


def build_fixtures(
    clips: Sequence[MevaClip],
    *,
    allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
) -> tuple[MevaFixture, ...]:
    """Map clips -> fixtures: one fixture per ``(clip, mapped condition)``.

    A fixture is per-condition so its ``goal_shape`` is homogeneous (one scope verdict per fixture)
    and its episode ids are the join keys the one-to-one matcher scores against. Activities whose
    type does not map (:func:`eval.datasets.meva.pack_ext.condition_for` -> ``None``) are dropped.
    """
    out: list[MevaFixture] = []
    for clip in clips:
        # Group this clip's mapped activities by target template (== condition).
        by_template: dict[str, tuple[ConditionSpec, list[MevaActivity]]] = {}
        for activity in clip.activities:
            spec = condition_for(activity.activity_type)
            if spec is None:
                continue
            by_template.setdefault(spec.template, (spec, []))[1].append(activity)

        for template in sorted(by_template):
            spec, activities = by_template[template]
            fixture_id = _fixture_id(clip.clip_id, template)
            episodes = tuple(
                activity_to_episode(
                    a, fixture_id=fixture_id, fps=clip.fps, allowed_lateness_s=allowed_lateness_s
                )
                for a in sorted(activities, key=lambda a: (a.begin_frame, a.activity_id))
            )
            out.append(
                MevaFixture(
                    fixture=Fixture(
                        fixture_id=fixture_id,
                        goal_shape=spec.goal_shape,
                        episodes=episodes,
                        scope=scope_for_goal_shape(spec.goal_shape),
                    ),
                    scene_id=sanitize(clip.scene_id),
                    clip_id=clip.clip_id,
                    condition=spec,
                )
            )
    if not out:
        raise ValueError("no fixtures produced — no clip carried a mapped MEVA activity type")
    return tuple(sorted(out, key=lambda mf: mf.fixture_id))


def scene_split(scene_ids: Sequence[str]) -> dict[str, list[str]]:
    """Deterministic 50/50 held-out split OVER SCENES (never clips).

    Scenes are sorted, then assigned by parity of their sorted index: even -> ``train``, odd ->
    ``test``. This is reproducible (no wall clock, no RNG), balanced (exactly 50/50 for an even
    scene count; the extra scene goes to ``train`` for an odd count), and — because the unit is the
    SCENE (all camera views of one recording session) — no physical event can appear on both sides.
    ``dev`` stays empty: this is a two-way qualify split, and every non-test scene is training data.
    """
    ordered = sorted(set(scene_ids))
    train = [s for i, s in enumerate(ordered) if i % 2 == 0]
    test = [s for i, s in enumerate(ordered) if i % 2 == 1]
    return {"train": train, "dev": [], "test": test}


def _fixture_level_splits(
    fixtures: Sequence[MevaFixture], scenes: Mapping[str, list[str]]
) -> dict[str, list[str]]:
    """Expand a scene-level split to the fixture ids each scene owns (what ``splits.json`` stores)."""
    scene_to_side = {scene: side for side, ss in scenes.items() for scene in ss}
    out: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for mf in fixtures:
        out[scene_to_side[mf.scene_id]].append(mf.fixture_id)
    return {side: sorted(ids) for side, ids in out.items()}


@dataclass(frozen=True)
class MevaSplitRegistration:
    """A :class:`percept.eval.derivation.DerivationPreRegistration`-style frozen record.

    Content-addresses everything that had to be fixed BEFORE any numbers exist: the fixture-set
    hash (the labels themselves), the bars, and the EXACT scene split (both the scene->side map and
    the derived held-out fixture ids). ``registration_hash`` is taken over that payload with the same
    :func:`percept.tracing.content_hash` the eval harness uses, so drift is detectable and the seed
    is derived, never wall-clock.
    """

    fixture_set_hash: str
    split: HeldOutSplit
    scenes: Mapping[str, list[str]]
    fixture_to_scene: Mapping[str, str]
    bars: BarsDocument
    declared_at: str = ""
    expected_registration_hash: str | None = None

    def _split_content(self) -> dict[str, Any]:
        """A PATH-INDEPENDENT view of the split — its content, never the absolute ``splits.json``
        location. ``HeldOutSplit.to_dict()`` carries the on-disk source path, which would make the
        frozen registration hash (and the committed file) differ per checkout; the split's identity
        is the held-out/exploratory membership, not where the file happens to live."""
        return {
            "declared": self.split.declared,
            "split_name": self.split.split_name,
            "held_out": list(self.split.held_out),
            "exploratory": list(self.split.exploratory),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "fixture_set_hash": self.fixture_set_hash,
            "split": self._split_content(),
            "scenes": {side: sorted(ss) for side, ss in self.scenes.items()},
            "bars": [b.to_dict() for b in self.bars.bars],
        }

    @property
    def registration_hash(self) -> str:
        return content_hash(self.payload())

    @property
    def bars_hash(self) -> str:
        return bars_hash(self.bars)

    @property
    def seed(self) -> int:
        """Deterministic bootstrap seed derived from the registration — never wall time."""
        return int(self.registration_hash[:12], 16)

    def assert_no_drift(self) -> None:
        """Raise if the presented configuration hashes differently from what was pre-registered."""
        if self.expected_registration_hash is None:
            return
        actual = self.registration_hash
        if actual != self.expected_registration_hash:
            raise ValueError(
                "REFUSING: MEVA fixture set / split / bars changed after registration.\n"
                f"  pre-registered: {self.expected_registration_hash}\n"
                f"  presented now : {actual}"
            )

    def to_dict(self) -> dict[str, Any]:
        # Written form: the same path-independent payload (via ``payload()``) plus provenance. The
        # split is the normalized content view — ``splits.json`` is the sibling file, referenced by
        # name, never by absolute path, so the committed record is portable across checkouts.
        return {
            "schema": "eval.datasets.meva.registration/1",
            "declared_at": self.declared_at,
            "registration_hash": self.registration_hash,
            "bars_hash": self.bars_hash,
            "bootstrap_seed": self.seed,
            "split_source": "splits.json",
            "fixture_to_scene": dict(sorted(self.fixture_to_scene.items())),
            **self.payload(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ground_truth_payload(fixture: Fixture) -> dict[str, Any]:
    return {
        "goal_shape": fixture.goal_shape,
        "episodes": [
            {
                "episode_id": e.episode_id,
                "event_start_ts": e.ground_truth.event_start_ts,
                "decision_eligible_ts": e.ground_truth.decision_eligible_ts,
                "episode_end_ts": e.episode_end_ts,
                "allowed_lateness_s": e.allowed_lateness_s,
                "annotation_version": e.ground_truth.annotation_version,
                "annotator_or_fixture_id": e.ground_truth.annotator_or_fixture_id,
            }
            for e in fixture.episodes
        ],
    }


def write_fixture_tree(
    fixtures: Sequence[MevaFixture],
    out_dir: str | Path,
    *,
    bars: BarsDocument = DEFAULT_BARS,
    declared_at: str = "2026-07-23",
) -> MevaSplitRegistration:
    """Write the whole loadable tree under ``out_dir`` and return the frozen registration.

    Writes ``<fixture_id>/ground_truth.json`` per fixture, then ``splits.json`` (fixture level,
    derived from the scene split), ``bars.json``, ``scene_index.json`` (fixture->scene, for the
    scene-disjointness check) and ``registration.json``.
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    for mf in fixtures:
        fixture_dir = base / mf.fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        _write_json(fixture_dir / "ground_truth.json", _ground_truth_payload(mf.fixture))

    # Reload from disk so the frozen hash is over the CANONICAL loaded labels (what a consumer sees).
    loaded = load_fixtures(base)

    scenes = scene_split([mf.scene_id for mf in fixtures])
    splits = _fixture_level_splits(fixtures, scenes)
    _write_json(base / "splits.json", splits)
    _write_json(base / "bars.json", bars.to_dict())
    fixture_to_scene = {mf.fixture_id: mf.scene_id for mf in fixtures}
    _write_json(base / "scene_index.json", dict(sorted(fixture_to_scene.items())))

    split = load_held_out_split(base, loaded, split_name="test")
    registration = MevaSplitRegistration(
        fixture_set_hash=fixture_set_hash(loaded),
        split=split,
        scenes=scenes,
        fixture_to_scene=fixture_to_scene,
        bars=bars,
        declared_at=declared_at,
    )
    (base / "registration.json").write_text(registration.to_json() + "\n")
    return registration


def write_dataset(
    kpf_dir: str | Path,
    out_dir: str | Path,
    *,
    fps: float = MEVA_DEFAULT_FPS,
    allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
    bars: BarsDocument = DEFAULT_BARS,
) -> MevaSplitRegistration:
    """Full pipeline: read a KPF dir, build fixtures, write the loadable tree + registration."""
    clips = iter_kpf_dir(kpf_dir, fps=fps)
    fixtures = build_fixtures(clips, allowed_lateness_s=allowed_lateness_s)
    return write_fixture_tree(fixtures, out_dir, bars=bars)


def regenerate_sample() -> MevaSplitRegistration:
    """Regenerate the committed synthetic sample (``testdata/`` -> ``fixtures/``)."""
    return write_dataset(TESTDATA_DIR, FIXTURES_DIR)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m eval.datasets.meva.transform`` entrypoint."""
    parser = argparse.ArgumentParser(
        prog="eval.datasets.meva.transform",
        description="Transform MEVA KPF activity annotations into Percept facility_observation "
        "fixtures with a pre-registered, scene-disjoint 50/50 held-out split.",
    )
    parser.add_argument("--kpf-dir", help="directory of MEVA *.activities.yml files")
    parser.add_argument("--out", help="output fixtures directory")
    parser.add_argument("--fps", type=float, default=MEVA_DEFAULT_FPS, help="clip frame rate (default 30)")
    parser.add_argument(
        "--sample", action="store_true",
        help="regenerate the committed synthetic sample (testdata/ -> fixtures/)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.sample:
        registration = regenerate_sample()
        out = FIXTURES_DIR
    else:
        if not args.kpf_dir or not args.out:
            parser.error("--kpf-dir and --out are required unless --sample is given")
        registration = write_dataset(args.kpf_dir, args.out, fps=args.fps)
        out = Path(args.out)

    print(f"wrote fixtures -> {out}")
    print(f"  registration_hash : {registration.registration_hash}")
    print(f"  fixture_set_hash  : {registration.fixture_set_hash}")
    print(f"  bars_hash         : {registration.bars_hash}")
    print(f"  held-out (test)   : {', '.join(registration.split.held_out) or '—'}")
    print(f"  scenes train/test : {len(registration.scenes['train'])}/{len(registration.scenes['test'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
