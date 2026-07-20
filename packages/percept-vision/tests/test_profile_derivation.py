"""Card L1 — the profile-derivation job (fixtures, bars) -> provenance-stamped WatchProfile.

The properties under test are the card's own promises:

* the whole ladder is reported, losers included;
* NO winner and NO profile without operator-supplied bars (and none when nothing meets them);
* the emitted profile is a card-P4 VersionedProfile, content-addressed and round-trippable;
* the bars, the split and the FIXTURE LABELS are pre-registered and hashed before scoring, and a
  configuration that moved after the bars were declared refuses to score;
* ADR-005: a rung names its SourceProfile, and overloading WatchProfile.environment with a
  source-class value is rejected before any number exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.eval.differential import Bar, BarsDocument, ConfigDriftError
from percept.eval.derivation import (
    DerivationInputError,
    DerivationVerdict,
    Ladder,
    LadderRung,
    assert_source_class_not_overloaded,
    bars_hash,
    derive_profile,
    fixture_set_hash,
    load_ladder,
    main,
    render_scorecard,
)
from percept.profile_versioning import (
    VersionedProfile,
    content_hash,
    profile_to_dict,
)
from percept.sourceprofile import get_source_profile
from percept.watchprofile import GateProfile, WatchProfile

ANN = "eligibility-rubric-v1"
PHYSICAL = get_source_profile("physical_scene")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _episode(eid: str, start: float, end: float, lateness: float = 1.0) -> dict:
    return {
        "episode_id": eid,
        "event_start_ts": start,
        "decision_eligible_ts": start,
        "episode_end_ts": end,
        "allowed_lateness_s": lateness,
        "annotation_version": ANN,
        "annotator_or_fixture_id": "fixture-gen",
    }


def _event(key: str, ts: float, seq: int = 0) -> dict:
    return {"key": key, "evidence_seq": seq, "action": "fire",
            "timing": {"decision_emitted_ts": ts}}


def _trace(run_id: str, *, cost: float) -> dict:
    return {
        "manifest": {"run_id": run_id},
        "records": [],
        "summary": {"run_id": run_id, "judge_completed": 1,
                    "cost": {"input_tokens": 10.0, "output_tokens": 0.0, "usd": cost}},
    }


def write_fixture(root: Path, fixture_id: str, *, goal_shape: str, episodes: list[dict],
                  arms: dict[str, list[dict]], costs: dict[str, float] | None = None) -> None:
    d = root / fixture_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "ground_truth.json").write_text(json.dumps({"goal_shape": goal_shape,
                                                     "episodes": episodes}))
    for arm, events in arms.items():
        (d / arm).mkdir(parents=True, exist_ok=True)
        (d / arm / "events.json").write_text(json.dumps(events))
        if costs and arm in costs:
            (d / arm / "trace.json").write_text(json.dumps(_trace(f"{fixture_id}-{arm}",
                                                                 cost=costs[arm])))


def _profile(name: str, **overrides) -> WatchProfile:
    base = dict(
        name=name,
        environment="fixed_camera",
        goal_type="stative",
        state="start_anchored",
        salience_stat="mean",
        plan_mode="single",
    )
    base.update(overrides)
    return WatchProfile(**base)  # type: ignore[arg-type]


BASE_RUNG = _profile("mean-single")
BLOCK_RUNG = _profile("block-max", salience_stat="block_max",
                      stat_thresholds={"block_max": 0.20})
WINDOW_RUNG = _profile("block-max-window", salience_stat="block_max",
                       stat_thresholds={"block_max": 0.20}, plan_mode="window",
                       gate=GateProfile(require_false_first=False, refractory_s=15.0))


def _ladder() -> Ladder:
    return Ladder(
        rungs=(
            LadderRung("baseline", BASE_RUNG, PHYSICAL, description="shipped mean-stat defaults"),
            LadderRung("blockmax", BLOCK_RUNG, PHYSICAL, description="+ localized salience"),
            LadderRung("window", WINDOW_RUNG, PHYSICAL, description="+ window judging + gate relax"),
        ),
        source="test-ladder",
    )


def _corpus(root: Path, *, declare_split: bool = True) -> None:
    """Four fixtures. baseline misses two episodes, blockmax misses one, window misses none."""
    plans = {
        "f1": ({"baseline": [], "blockmax": [_event("k", 1.5)], "window": [_event("k", 1.5)]}),
        "f2": ({"baseline": [], "blockmax": [], "window": [_event("k", 1.5)]}),
        "f3": ({"baseline": [_event("k", 1.5)], "blockmax": [_event("k", 1.5)],
                "window": [_event("k", 1.5)]}),
        "f4": ({"baseline": [_event("k", 1.5)], "blockmax": [_event("k", 1.5)],
                "window": [_event("k", 1.5)]}),
    }
    for fid, arms in plans.items():
        write_fixture(root, fid, goal_shape="episode", episodes=[_episode("e1", 1.0, 2.0)],
                      arms=arms,
                      costs={"baseline": 0.01, "blockmax": 0.02, "window": 0.03})
    if declare_split:
        (root / "splits.json").write_text(json.dumps({"train": ["f1"], "dev": [],
                                                      "test": ["f2", "f3", "f4"]}))


def _bars(*bars: Bar) -> BarsDocument:
    return BarsDocument(source="test-bars.json", bars=bars, declared_by="ops",
                        declared_at="2026-07-20")


# ---------------------------------------------------------------------------
# 1. The whole ladder is reported, losers included
# ---------------------------------------------------------------------------


def test_every_rung_is_scored_including_the_losers(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(fixtures_root=tmp_path, ladder=_ladder(),
                            bars=BarsDocument(source="(none supplied)"))
    assert [r.rung.name for r in report.rungs] == ["baseline", "blockmax", "window"]
    # A scored loser is data: the baseline rung is present with real numbers, not elided.
    baseline = report.rung("baseline")
    assert baseline is not None and baseline.is_baseline
    assert baseline.mean("recall") == pytest.approx(2 / 3)
    assert report.rung("window").mean("recall") == pytest.approx(1.0)
    payload = json.loads(report.to_json())
    assert len(payload["ladder"]) == 3


def test_deltas_are_measured_against_the_ladder_baseline(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(fixtures_root=tmp_path, ladder=_ladder(),
                            bars=BarsDocument(source="(none)"))
    assert report.rung("baseline").delta("recall").mean_delta == pytest.approx(0.0)
    assert report.rung("window").delta("recall").mean_delta == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 2. No bars -> report, never a winner
# ---------------------------------------------------------------------------


def test_without_bars_it_reports_and_refuses_to_declare_a_winner(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(fixtures_root=tmp_path, ladder=_ladder(),
                            bars=BarsDocument(source="(none supplied)"))
    assert report.verdict is DerivationVerdict.NOT_ASSESSED
    assert report.winner is None
    assert report.derived is None
    assert "NO BARS SUPPLIED" in report.verdict_reason
    assert "DERIVED PROFILE: none emitted." in render_scorecard(report)


def test_bars_but_undeclared_split_refuses(tmp_path: Path) -> None:
    _corpus(tmp_path, declare_split=False)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.5)),
    )
    assert report.verdict is DerivationVerdict.NOT_ASSESSED
    assert report.derived is None
    assert "UNDECLARED" in report.verdict_reason


def test_no_rung_meets_the_bars_is_a_fail_with_no_profile(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=1.01)),
    )
    assert report.verdict is DerivationVerdict.FAIL
    assert report.derived is None
    assert all(r.bars_met is False for r in report.rungs)
    assert "NO RUNG met every supplied bar" in report.verdict_reason


# ---------------------------------------------------------------------------
# 3. Winner + the emitted artifact
# ---------------------------------------------------------------------------


def test_winner_is_the_earliest_rung_in_declared_order_that_meets_every_bar(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.99)),
    )
    assert report.verdict is DerivationVerdict.PASS
    assert report.winner.rung.name == "window"
    assert report.rung("baseline").bars_met is False
    assert report.rung("blockmax").bars_met is False


def test_earliest_wins_when_several_rungs_pass(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.6)),
    )
    assert report.verdict is DerivationVerdict.PASS
    assert report.winner.rung.name == "baseline"
    assert "also met the bars" in report.verdict_reason


def test_derived_profile_is_a_stamped_roundtrippable_versioned_profile(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.99)),
        eval_run_id="run-42", created_at="2026-07-20T00:00:00+00:00",
    )
    derived = report.derived
    assert derived is not None
    vp = derived.watch_profile
    assert isinstance(vp, VersionedProfile)
    assert vp.content_hash == content_hash(WINDOW_RUNG)
    assert vp.provenance.eval_run_id == "run-42"
    assert vp.provenance.harness_version  # the harness that measured it
    # Card P4 round-trip through the wire format, checksum verified on load.
    assert VersionedProfile.from_dict(vp.to_dict()).profile == WINDOW_RUNG

    payload = json.loads(derived.to_json())
    assert payload["source_profile"]["name"] == "physical_scene"
    prov = payload["derivation"]
    assert prov["fixture_set_hash"] == report.registration.fixture_set_hash
    assert prov["bars_hash"] == report.registration.bars_hash
    assert prov["registration_hash"] == report.registration.registration_hash
    assert prov["winning_rung"] == "window"
    assert prov["ladder"] == ["baseline", "blockmax", "window"]
    assert prov["split"]["held_out"] == ["f2", "f3", "f4"]


def test_a_partially_measurable_bar_set_is_never_a_pass(tmp_path: Path) -> None:
    """cost_usd is unmeasurable when no arm wrote a trace: unassessed, never optimistic."""
    for fid in ("f1", "f2"):
        write_fixture(tmp_path, fid, goal_shape="episode", episodes=[_episode("e1", 1.0, 2.0)],
                      arms={"baseline": [_event("k", 1.5)], "blockmax": [_event("k", 1.5)],
                            "window": [_event("k", 1.5)]})
    (tmp_path / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["f1", "f2"]}))
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.5),
                   Bar(metric="cost_usd_delta", direction="max", value=0.0)),
    )
    assert all(r.bars_met is None for r in report.rungs)
    assert report.verdict is DerivationVerdict.FAIL
    assert report.derived is None


# ---------------------------------------------------------------------------
# 4. Pre-registration — the anti-tune-to-green record
# ---------------------------------------------------------------------------


def test_registration_hash_covers_bars_ladder_order_and_labels(tmp_path: Path) -> None:
    _corpus(tmp_path)
    bars = _bars(Bar(metric="candidate_recall", direction="min", value=0.99))
    base = derive_profile(fixtures_root=tmp_path, ladder=_ladder(), bars=bars)
    h = base.registration.registration_hash

    # same inputs -> same hash (and the same run id / bootstrap seed)
    again = derive_profile(fixtures_root=tmp_path, ladder=_ladder(), bars=bars)
    assert again.registration.registration_hash == h
    assert again.registration.eval_run_id == base.registration.eval_run_id

    # moving a bar changes it
    moved = derive_profile(fixtures_root=tmp_path, ladder=_ladder(),
                           bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.5)))
    assert moved.registration.registration_hash != h

    # reordering the ladder changes it — the order is the winner rule, so it is load-bearing
    rungs = _ladder().rungs
    reordered = Ladder(rungs=(rungs[0], rungs[2], rungs[1]), source="test-ladder")
    assert derive_profile(fixtures_root=tmp_path, ladder=reordered,
                          bars=bars).registration.registration_hash != h


def test_relabelling_a_fixture_changes_the_fixture_set_hash(tmp_path: Path) -> None:
    _corpus(tmp_path)
    before = derive_profile(fixtures_root=tmp_path, ladder=_ladder(),
                            bars=BarsDocument(source="(none)")).registration.fixture_set_hash
    gt = tmp_path / "f2" / "ground_truth.json"
    gt.write_text(json.dumps({"goal_shape": "episode",
                              "episodes": [_episode("e1", 1.0, 9.0)]}))
    after = derive_profile(fixtures_root=tmp_path, ladder=_ladder(),
                           bars=BarsDocument(source="(none)")).registration.fixture_set_hash
    assert before != after


def test_fixture_set_hash_is_order_independent(tmp_path: Path) -> None:
    from percept.eval.differential import load_fixtures

    _corpus(tmp_path)
    fixtures = load_fixtures(tmp_path)
    assert fixture_set_hash(fixtures) == fixture_set_hash(tuple(reversed(fixtures)))


def test_bars_hash_moves_with_the_bars() -> None:
    a = _bars(Bar(metric="candidate_recall", direction="min", value=0.9))
    b = _bars(Bar(metric="candidate_recall", direction="min", value=0.8))
    assert bars_hash(a) != bars_hash(b)
    assert bars_hash(a) == bars_hash(_bars(Bar(metric="candidate_recall", direction="min",
                                               value=0.9)))


def test_config_drift_refuses_to_score(tmp_path: Path) -> None:
    _corpus(tmp_path)
    bars = _bars(Bar(metric="candidate_recall", direction="min", value=0.99))
    good = derive_profile(fixtures_root=tmp_path, ladder=_ladder(), bars=bars)
    with pytest.raises(ConfigDriftError) as exc:
        derive_profile(
            fixtures_root=tmp_path, ladder=_ladder(),
            bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.1)),
            expected_registration_hash=good.registration.registration_hash,
        )
    assert "REFUSING TO DERIVE" in str(exc.value)


# ---------------------------------------------------------------------------
# 5. ADR-005 — two axes, and no overloading of the camera axis
# ---------------------------------------------------------------------------


def test_a_rung_names_its_source_profile() -> None:
    rung = LadderRung("baseline", BASE_RUNG, get_source_profile("screen_document"))
    assert rung.identity_payload()["source_profile"]["name"] == "screen_document"


def test_environment_may_not_carry_a_source_class_value() -> None:
    bad = _profile("oops", environment="screen_document")
    with pytest.raises(DerivationInputError) as exc:
        assert_source_class_not_overloaded(bad)
    assert "ADR-005" in str(exc.value)
    with pytest.raises(DerivationInputError):
        LadderRung("bad", bad, PHYSICAL)


def test_a_ladder_may_not_mix_source_classes() -> None:
    with pytest.raises(DerivationInputError) as exc:
        Ladder(rungs=(
            LadderRung("a", BASE_RUNG, PHYSICAL),
            LadderRung("b", BLOCK_RUNG, get_source_profile("screen_document")),
        ))
    assert "SAME source profile" in str(exc.value)


def test_a_non_camera_source_derives_without_touching_the_camera_axis(tmp_path: Path) -> None:
    """The screen_document probe: a non-camera source class still derives a profile, and the
    WatchProfile's camera-shaped axes are untouched."""
    screen = get_source_profile("screen_document")
    ladder = Ladder(rungs=(
        LadderRung("baseline", BASE_RUNG, screen),
        LadderRung("window", WINDOW_RUNG, screen),
    ))
    _corpus(tmp_path)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=ladder,
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.99)),
    )
    assert report.derived.source_profile.name == "screen_document"
    assert report.derived.watch_profile.profile.environment == "fixed_camera"
    # the scorecard tells the 2am reader the source is admission-untuned
    assert "UNTUNED" in render_scorecard(report)


# ---------------------------------------------------------------------------
# 6. Ladder validation + loading
# ---------------------------------------------------------------------------


def test_a_single_rung_is_not_a_ladder() -> None:
    with pytest.raises(DerivationInputError):
        Ladder(rungs=(LadderRung("only", BASE_RUNG, PHYSICAL),))


def test_duplicate_fixture_dirs_are_rejected() -> None:
    with pytest.raises(DerivationInputError) as exc:
        Ladder(rungs=(
            LadderRung("a", BASE_RUNG, PHYSICAL, fixture_dir="same"),
            LadderRung("b", BLOCK_RUNG, PHYSICAL, fixture_dir="same"),
        ))
    assert "unique" in str(exc.value)


def test_load_ladder_roundtrips_profile_bodies(tmp_path: Path) -> None:
    doc = {
        "source_profile": "physical_scene",
        "declared_at": "2026-07-20",
        "rungs": [
            {"name": "baseline", "description": "a", "profile": profile_to_dict(BASE_RUNG)},
            {"name": "window", "description": "b", "profile": profile_to_dict(WINDOW_RUNG)},
        ],
    }
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(doc))
    ladder = load_ladder(path)
    assert [r.name for r in ladder.rungs] == ["baseline", "window"]
    assert ladder.rungs[1].watch_profile == WINDOW_RUNG
    assert ladder.source_profile.name == "physical_scene"
    assert ladder.declared_at == "2026-07-20"


def test_load_ladder_requires_a_source_profile(tmp_path: Path) -> None:
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"rungs": [{"name": "a", "profile": profile_to_dict(BASE_RUNG)},
                                          {"name": "b", "profile": profile_to_dict(BLOCK_RUNG)}]}))
    with pytest.raises(DerivationInputError) as exc:
        load_ladder(path)
    assert "ADR-005" in str(exc.value)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    _corpus(tmp_path)
    ladder_path = tmp_path / "ladder.json"
    ladder_path.write_text(json.dumps({
        "source_profile": "physical_scene",
        "rungs": [
            {"name": "baseline", "profile": profile_to_dict(BASE_RUNG)},
            {"name": "blockmax", "profile": profile_to_dict(BLOCK_RUNG)},
            {"name": "window", "profile": profile_to_dict(WINDOW_RUNG)},
        ],
    }))
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({
        "declared_by": "ops",
        "bars": [{"metric": "candidate_recall", "direction": "min", "value": 0.99}],
    }))
    return ladder_path, bars_path


def test_cli_writes_the_derived_profile_and_the_scorecard(tmp_path: Path, capsys) -> None:
    ladder_path, bars_path = _write_cli_inputs(tmp_path)
    out_profile = tmp_path / "derived.json"
    out_json = tmp_path / "report.json"
    rc = main(["--fixtures", str(tmp_path), "--ladder", str(ladder_path),
               "--bars", str(bars_path), "--out-profile", str(out_profile),
               "--json", str(out_json)])
    assert rc == 0
    assert "VERDICT: PASS" in capsys.readouterr().out
    body = json.loads(out_profile.read_text())
    assert body["derivation"]["winning_rung"] == "window"
    assert VersionedProfile.from_dict(body["watch_profile"]).profile.name == "block-max-window"
    assert len(json.loads(out_json.read_text())["ladder"]) == 3


def test_cli_writes_no_profile_without_bars(tmp_path: Path, capsys) -> None:
    ladder_path, _ = _write_cli_inputs(tmp_path)
    out_profile = tmp_path / "derived.json"
    rc = main(["--fixtures", str(tmp_path), "--ladder", str(ladder_path),
               "--out-profile", str(out_profile)])
    assert rc == 0
    assert not out_profile.exists()
    captured = capsys.readouterr()
    assert "NOT_ASSESSED" in captured.out
    assert "no profile written" in captured.err


def test_cli_refuses_on_bad_input(tmp_path: Path, capsys) -> None:
    rc = main(["--fixtures", str(tmp_path / "nope"), "--ladder", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err


def test_scorecard_shows_the_provenance_stamp_and_the_whole_ladder(tmp_path: Path) -> None:
    _corpus(tmp_path)
    report = derive_profile(
        fixtures_root=tmp_path, ladder=_ladder(),
        bars=_bars(Bar(metric="candidate_recall", direction="min", value=0.99)),
    )
    text = render_scorecard(report)
    for needle in ("registration_hash", "fixture_set_hash", "bars_hash", "eval_run_id",
                   "baseline", "blockmax", "window", "physical_scene", "DERIVED PROFILE"):
        assert needle in text
