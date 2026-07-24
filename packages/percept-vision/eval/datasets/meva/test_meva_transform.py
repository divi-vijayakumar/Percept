"""Unit tests for the MEVA -> facility_observation transform (DEV TOOLING).

Run from the SDK package root::

    cd packages/percept-vision && python3 -m pytest eval/datasets/meva/test_meva_transform.py -q

Everything runs offline on the committed synthetic KPF sample — no dataset download, no model call,
no network. The tests prove: annotation->fixture mapping is correct, fire timestamps land on the
activity BEGIN, the bars/split files load through Percept's OWN loaders, and the held-out split is
SCENE-disjoint (no camera view of a scene leaks across train/test).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.domain_pack import DomainPack, get_domain_pack, watch_template
from percept.eval import assert_disjoint, load_splits
from percept.eval.derivation import fixture_set_hash
from percept.eval.differential import load_bars, load_fixtures, load_held_out_split
from percept.harness.golden_eval import build_golden_timeline, score

from eval.datasets.meva import kpf, pack_ext, transform

# The pre-registered, frozen hashes (path-independent by construction). A silent relabel of the
# committed sample changes the fixture-set hash and fails these loudly — which is the point.
FROZEN_FIXTURE_SET_HASH = "c490894b6a8f56184ee37fa883120b16fb0a832a40786f834ca89173eb754b22"
FROZEN_REGISTRATION_HASH = "5e2e71b31482c47adbb74dbeefd06da3f92ca81e8bd75bae38aba576f10587c0"

FIXTURES = transform.FIXTURES_DIR
TESTDATA = transform.TESTDATA_DIR


# ---------------------------------------------------------------------------
# KPF reader
# ---------------------------------------------------------------------------


def test_kpf_reader_parses_types_frames_and_scene() -> None:
    clip = kpf.load_kpf_file(TESTDATA / "2018-03-05.11-25-00.11-30-00.school.G330.activities.yml")
    assert clip.scene_id == "2018-03-05.11-25-00.11-30-00.school"
    assert clip.camera == "G330"
    assert clip.site == "school"
    assert clip.fps == kpf.MEVA_DEFAULT_FPS
    by_type = {a.activity_type: a for a in clip.activities}
    assert set(by_type) == {"person_opens_facility_door", "person_loiters", "vehicle_turns_left"}
    door = by_type["person_opens_facility_door"]
    assert (door.begin_frame, door.end_frame) == (300, 480)
    assert door.begin_ts() == 10.0  # 300 / 30 fps


def test_scene_id_groups_all_camera_views() -> None:
    g330 = kpf.load_kpf_file(TESTDATA / "2018-03-05.11-25-00.11-30-00.school.G330.activities.yml")
    g420 = kpf.load_kpf_file(TESTDATA / "2018-03-05.11-25-00.11-30-00.school.G420.activities.yml")
    # Two camera VIEWS, one recording session -> one scene id (the split unit).
    assert g330.scene_id == g420.scene_id
    assert g330.camera != g420.camera


def test_activity_type_encoding_variants() -> None:
    # Alias `abandonment` (act2 dict) ...
    g420 = kpf.load_kpf_file(TESTDATA / "2018-03-05.13-10-00.13-15-00.school.G420.activities.yml")
    assert any(a.activity_type == "abandonment" for a in g420.activities)
    # ... and the older DIVA-style `activity: <name>` string form both parse.
    g506 = kpf.load_kpf_file(TESTDATA / "2018-03-07.09-05-00.09-10-00.bus.G506.activities.yml")
    assert any(a.activity_type == "person_abandons_package" for a in g506.activities)


def test_clip_without_camera_token_is_rejected() -> None:
    with pytest.raises(kpf.KpfError):
        kpf.parse_clip_id("2018-03-05.11-25-00.11-30-00.school.activities.yml")


# ---------------------------------------------------------------------------
# Mapping: annotation -> fixture
# ---------------------------------------------------------------------------


def test_mapping_targets_expected_conditions_and_shapes() -> None:
    clips = kpf.iter_kpf_dir(TESTDATA)
    fixtures = transform.build_fixtures(clips)
    by_id = {mf.fixture_id: mf for mf in fixtures}

    # 4 scenes x 2 cameras, mapped conditions per scene -> 14 fixtures (see the sample layout).
    assert len(fixtures) == 14

    door = by_id["2018-03-05_11-25-00_11-30-00_school_G330__door_left_open"]
    assert door.condition.template == "door_left_open"
    assert door.fixture.goal_shape == "episode"

    zone = by_id["2018-03-07_14-40-00_14-45-00_bus_G505__zone_occupied"]
    assert zone.condition.template == "zone_occupied"
    assert zone.fixture.goal_shape == "episode"

    package = by_id["2018-03-05_13-10-00_13-15-00_school_G330__package_appeared"]
    assert package.condition.template == "package_appeared"
    assert package.fixture.goal_shape == "point"


def test_unmapped_activity_is_dropped() -> None:
    clips = kpf.iter_kpf_dir(TESTDATA)
    fixtures = transform.build_fixtures(clips)
    # `vehicle_turns_left` is present in the KPF but maps to no facility_observation condition,
    # so it yields no fixture and no episode.
    assert pack_ext.condition_for("vehicle_turns_left") is None
    assert not any("vehicle_turns_left" in mf.fixture_id for mf in fixtures)
    for mf in fixtures:
        for ep in mf.fixture.episodes:
            assert "vehicle_turns_left" not in ep.episode_id


def test_fire_timestamp_lands_on_activity_begin() -> None:
    clips = kpf.iter_kpf_dir(TESTDATA)
    fixtures = {mf.fixture_id: mf for mf in transform.build_fixtures(clips)}

    door = fixtures["2018-03-05_11-25-00_11-30-00_school_G330__door_left_open"].fixture
    (ep,) = door.episodes
    assert ep.ground_truth.event_start_ts == 10.0        # begin frame 300 / 30 fps
    assert ep.ground_truth.decision_eligible_ts == 10.0  # the fire anchor IS the begin
    assert ep.episode_end_ts == 16.0                     # end frame 480 / 30 fps

    package = fixtures["2018-03-05_13-10-00_13-15-00_school_G330__package_appeared"].fixture
    (pep,) = package.episodes
    assert pep.ground_truth.decision_eligible_ts == 50.0  # begin frame 1500 / 30 fps

    # The canonical golden timeline fires at exactly the pinned begin instant.
    golden = build_golden_timeline(door.episodes)
    assert golden[0].timing.decision_emitted_ts == 10.0


# ---------------------------------------------------------------------------
# The committed tree loads through Percept's OWN loaders
# ---------------------------------------------------------------------------


def test_fixtures_load_through_percept_loader() -> None:
    fixtures = load_fixtures(FIXTURES)
    assert len(fixtures) == 14
    # Every mapped condition is a SUPPORTED scope shape -> it can be scored (not dropped by scope).
    assert all(f.scope.scorable for f in fixtures)


def test_golden_timeline_scores_recall_1_on_every_fixture() -> None:
    # Proves the on-disk fixtures are matcher-compatible: the ideal fire timeline recovers every
    # labeled episode under Percept's own one-to-one event-level matcher.
    for f in load_fixtures(FIXTURES):
        result = score(f.episodes, build_golden_timeline(f.episodes))
        assert result.recall == 1.0
        assert result.false_fires == 0


def test_bars_load_through_percept_loader() -> None:
    bars = load_bars(FIXTURES / "bars.json")
    assert bars.declared
    by_metric = {b.metric: b for b in bars.bars}
    assert by_metric["candidate_recall"].direction == "min"
    assert by_metric["candidate_recall"].value == pytest.approx(0.80)
    assert by_metric["recall_delta"].direction == "min"
    assert by_metric["recall_delta"].value == pytest.approx(0.05)
    assert by_metric["recall_delta"].require_ci_excludes_zero is True
    assert by_metric["false_fire_delta"].direction == "max"
    assert by_metric["false_fire_delta"].value == pytest.approx(0.0)


def test_split_loads_and_is_5050_by_scene() -> None:
    fixtures = load_fixtures(FIXTURES)
    split = load_held_out_split(FIXTURES, fixtures, split_name="test")
    assert split.declared
    registration = json.loads((FIXTURES / "registration.json").read_text())
    # 4 scenes, split 2/2 (a true 50/50 by scene).
    assert len(registration["scenes"]["train"]) == 2
    assert len(registration["scenes"]["test"]) == 2
    assert set(split.held_out) == set(registration["split"]["held_out"])


# ---------------------------------------------------------------------------
# The split is SCENE-disjoint (the anti-leak invariant)
# ---------------------------------------------------------------------------


def test_split_is_scene_disjoint() -> None:
    splits = load_splits(FIXTURES / "splits.json")  # percept's own loader: raises if fixtures overlap
    assert_disjoint(splits)  # fixture-level disjointness (percept's own check)

    fixture_to_scene = json.loads((FIXTURES / "scene_index.json").read_text())

    # THE required assertion: no SCENE id appears on both sides of the split. MEVA is multiview, so
    # a clip-level split could put one camera's view of an event in train and another in test; the
    # scene grouping is what forbids that.
    scenes_by_side = {
        side: {fixture_to_scene[fid] for fid in ids}
        for side, ids in splits.items()
        if ids
    }
    train_scenes = scenes_by_side.get("train", set())
    test_scenes = scenes_by_side.get("test", set())
    assert train_scenes.isdisjoint(test_scenes), (
        f"scene(s) {train_scenes & test_scenes} appear in BOTH train and test — multiview leak"
    )
    # And both cameras of every scene stay together (each scene wholly on one side).
    assert train_scenes and test_scenes


# ---------------------------------------------------------------------------
# Pre-registration: frozen, content-addressed, drift-detecting
# ---------------------------------------------------------------------------


def test_registration_is_frozen_and_round_trips() -> None:
    fixtures = load_fixtures(FIXTURES)
    committed = json.loads((FIXTURES / "registration.json").read_text())

    # The committed frozen hashes match the recomputed content hashes (nothing drifted).
    assert committed["fixture_set_hash"] == FROZEN_FIXTURE_SET_HASH
    assert committed["registration_hash"] == FROZEN_REGISTRATION_HASH
    assert fixture_set_hash(fixtures) == FROZEN_FIXTURE_SET_HASH

    # Rebuild the registration from the loaded tree and confirm it reproduces the frozen hash.
    split = load_held_out_split(FIXTURES, fixtures, split_name="test")
    registration = transform.MevaSplitRegistration(
        fixture_set_hash=fixture_set_hash(fixtures),
        split=split,
        scenes=committed["scenes"],
        fixture_to_scene=committed["fixture_to_scene"],
        bars=transform.DEFAULT_BARS,
        declared_at=committed["declared_at"],
        expected_registration_hash=FROZEN_REGISTRATION_HASH,
    )
    assert registration.registration_hash == FROZEN_REGISTRATION_HASH
    registration.assert_no_drift()  # does not raise


def test_registration_drift_is_refused() -> None:
    fixtures = load_fixtures(FIXTURES)
    split = load_held_out_split(FIXTURES, fixtures, split_name="test")
    committed = json.loads((FIXTURES / "registration.json").read_text())
    drifted = transform.MevaSplitRegistration(
        fixture_set_hash="deadbeef" * 8,  # a relabeled fixture set
        split=split,
        scenes=committed["scenes"],
        fixture_to_scene=committed["fixture_to_scene"],
        bars=transform.DEFAULT_BARS,
        expected_registration_hash=FROZEN_REGISTRATION_HASH,
    )
    with pytest.raises(ValueError):
        drifted.assert_no_drift()


def test_regeneration_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    # The committed tree is reproducible: regenerating from the same KPF sample yields identical
    # bytes, so the transform has no hidden nondeterminism (ordering, wall-clock, RNG).
    transform.write_dataset(TESTDATA, tmp_path)
    committed_files = sorted(p for p in FIXTURES.rglob("*") if p.is_file())
    assert committed_files, "no committed fixtures found"
    for committed in committed_files:
        rel = committed.relative_to(FIXTURES)
        regenerated = tmp_path / rel
        assert regenerated.exists(), f"regeneration missing {rel}"
        assert regenerated.read_bytes() == committed.read_bytes(), f"{rel} differs after regen"


# ---------------------------------------------------------------------------
# Pack scaffolding (self-contained; never edits the shared registry)
# ---------------------------------------------------------------------------


def test_shipped_templates_render_the_mapped_conditions() -> None:
    # The mapping targets templates the shipped facility_observation@1 pack already defines.
    pack = get_domain_pack("facility_observation@1")
    for spec in pack_ext.MEVA_ACTIVITY_TO_CONDITION.values():
        assert spec.template in pack.templates
    plan = watch_template("facility_observation@1/door_left_open")
    assert "open" in plan.condition


def test_meva_pack_scaffolding_builds_and_registers_cleanly() -> None:
    pack = pack_ext.build_meva_facility_pack()
    assert isinstance(pack, DomainPack)
    assert set(pack.templates) == {
        "person_opens_facility_door",
        "person_loiters",
        "person_abandons_package",
    }
    # A MEVA-named template renders to the same door condition shape.
    plan = pack.template("person_opens_facility_door").render()
    assert "open" in plan.condition

    # register() is opt-in and reversible — it must not permanently mutate the shared registry.
    from percept.domain_pack import DEFAULT_DOMAIN_PACK_REGISTRY

    try:
        registered = pack_ext.register(overwrite=True)
        assert get_domain_pack("facility_observation_meva@1") == registered
    finally:
        DEFAULT_DOMAIN_PACK_REGISTRY.unregister("facility_observation_meva@1")
    assert "facility_observation_meva@1" not in DEFAULT_DOMAIN_PACK_REGISTRY
