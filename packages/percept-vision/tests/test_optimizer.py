"""— the optimizer: a typed SearchSpace over declared-safe tunables, scored through the
same ladder discipline percept.eval.derivation already runs, emitting an OptimizationLock.

The properties under test are the card's own promises:

* a SearchSpace may only cover tunables.json paths with safe_to_optimize=true — an unsafe path
  raises at construction, and a hard-coded guardrail deny-list blocks gate/goal/consent/retention
  paths even when the registry itself would say they are safe (data cannot bypass this);
* the optimizer reuses percept.eval.derivation's ladder/scorecard machinery unchanged (no
  reimplementation) — expand_templates() just materializes more LadderRung objects for it to score;
* ACCEPTANCE: on a localized-salience fixture corpus, the optimizer selects block_max@0.20 over
  mean@0.10, writes an OptimizationLock, and a locked re-run reproduces the selection
  deterministically (same values, same content_hash, and — fed back through ConfigResolver — the
  same effective runtime values);
* the CLI (`percept optimize`) writes the same lock artifact end to end, scripted/offline, zero API
  spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.eval.derivation import DerivationVerdict
from percept.eval.differential import Bar, BarsDocument
from percept.optimization_lock import OptimizationLock
from percept.optimizer import (
    OptimizerTemplate,
    SearchDimension,
    SearchSpace,
    SearchSpaceError,
    apply_search_point,
    expand_templates,
    load_optimizer_document,
    main,
    optimize,
)
from percept.resolver import ConfigResolver, LAYER_OPTIMIZATION_LOCK
from percept.sourceprofile import get_source_profile
from percept.tracing import content_hash as tracing_content_hash
from percept.watchprofile import WatchProfile

ANN = "eligibility-rubric-v1"
PHYSICAL = get_source_profile("physical_scene")


# ---------------------------------------------------------------------------
# fixture builders — same shape tests/test_profile_derivation.py uses
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
    return {"key": key, "evidence_seq": seq, "action": "fire", "timing": {"decision_emitted_ts": ts}}


def write_fixture(root: Path, fixture_id: str, *, goal_shape: str, episodes: list[dict],
                  arms: dict[str, list[dict]]) -> None:
    d = root / fixture_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "ground_truth.json").write_text(json.dumps({"goal_shape": goal_shape, "episodes": episodes}))
    for arm, events in arms.items():
        (d / arm).mkdir(parents=True, exist_ok=True)
        (d / arm / "events.json").write_text(json.dumps(events))


def _localized_salience_corpus(root: Path) -> None:
    """The localized-screen-salience scenario the viability ladder already established (memory:
    block_max nearly doubles recall over the mean-stat baseline): a mean@0.10 arm that misses most
    localized motion, and a block_max@0.20 arm that catches it. Mirrors
    tests/test_profile_derivation.py's `_corpus` builder exactly (four fixtures, one held-out
    baseline-only miss, three in the 'test' split)."""
    plans = {
        "f1": {"mean": [], "block_max": [_event("e1", 1.5)]},
        "f2": {"mean": [], "block_max": [_event("e1", 1.5)]},
        "f3": {"mean": [_event("e1", 1.5)], "block_max": [_event("e1", 1.5)]},
        "f4": {"mean": [_event("e1", 1.5)], "block_max": [_event("e1", 1.5)]},
    }
    for fid, arms in plans.items():
        write_fixture(root, fid, goal_shape="episode", episodes=[_episode("e1", 1.0, 2.0)], arms=arms)
    (root / "splits.json").write_text(json.dumps({"train": ["f1"], "dev": [], "test": ["f2", "f3", "f4"]}))


def _templates() -> tuple[OptimizerTemplate, OptimizerTemplate]:
    mean_base = WatchProfile(
        name="mean-base", environment="fixed_camera", goal_type="stative", state="start_anchored",
        salience_stat="mean",
    )
    block_base = WatchProfile(
        name="block-base", environment="fixed_camera", goal_type="stative", state="start_anchored",
        salience_stat="block_max",
    )
    return (
        OptimizerTemplate(name="mean", base_profile=mean_base, source_profile=PHYSICAL, fixture_dir="mean"),
        OptimizerTemplate(name="block_max", base_profile=block_base, source_profile=PHYSICAL,
                          fixture_dir="block_max"),
    )


def _space() -> SearchSpace:
    return SearchSpace(dimensions=(
        SearchDimension(path="Settings.edge.motion_gate", choices=(0.10,)),
        SearchDimension(path="Settings.edge.motion_gate_block_max", choices=(0.20,)),
    ))


def _bars() -> BarsDocument:
    return BarsDocument(source="test-bars", declared_by="ops", declared_at="2026-07-22",
                        bars=(Bar(metric="candidate_recall", direction="min", value=0.99),))


# ---------------------------------------------------------------------------
# 1. SearchSpace — safe-only, guardrailed even against data
# ---------------------------------------------------------------------------


class TestSearchSpaceGuardrails:
    def test_unsafe_path_raises_at_construction(self):
        """Settings.entity.max_entities is safe_to_optimize=False for registry reasons (an eviction
        resource ceiling, not a guardrailed vocabulary) — exercises the REGISTRY check specifically,
        distinct from the hard-coded guardrail prefixes exercised below."""
        with pytest.raises(SearchSpaceError, match="safe_to_optimize=False"):
            SearchDimension(path="Settings.entity.max_entities", choices=(32,))

    def test_unknown_path_raises(self):
        with pytest.raises(SearchSpaceError, match="not a known"):
            SearchDimension(path="Totally.Bogus.Path", choices=(1,))

    @pytest.mark.parametrize("path", [
        "Settings.gate.threshold",
        "Settings.gate.refractory_s",
        "WatchProfile.gate.refractory_s",
        "Goal.state.fire_when",
        "Goal.state.miss_cost",
        "Goal.state.type",
        "Settings.consent.allow_computer_use",
        "Settings.consent.actuation",
    ])
    def test_guardrail_blocks_gate_goal_consent_paths(self, path):
        with pytest.raises(SearchSpaceError, match="guardrailed"):
            SearchDimension(path=path, choices=(1,))

    def test_guardrail_blocks_retention_path_even_though_registry_marks_it_safe(self):
        """Settings.episodes.window_s is safe_to_optimize=True in percept.tunables.REGISTRY today —
        the hard-coded guardrail must still refuse it (never trusting the registry flag alone)."""
        from percept.tunables import REGISTRY

        spec = next(s for s in REGISTRY if s.path == "Settings.episodes.window_s")
        assert spec.safe_to_optimize is True, "test assumption: this path IS marked safe in data"
        with pytest.raises(SearchSpaceError, match="guardrailed"):
            SearchDimension(path="Settings.episodes.window_s", choices=(3600.0,))

    def test_safe_numeric_path_is_accepted(self):
        dim = SearchDimension(path="Settings.edge.motion_gate", choices=(0.10, 0.15))
        assert dim.choices == (0.10, 0.15)

    def test_choices_and_range_are_mutually_exclusive(self):
        with pytest.raises(SearchSpaceError, match="EITHER"):
            SearchDimension(path="Settings.edge.motion_gate", choices=(0.1,), low=0.0, high=1.0)

    def test_needs_choices_or_range(self):
        with pytest.raises(SearchSpaceError, match="needs either"):
            SearchDimension(path="Settings.edge.motion_gate")

    def test_duplicate_dimension_paths_rejected(self):
        with pytest.raises(SearchSpaceError, match="unique"):
            SearchSpace(dimensions=(
                SearchDimension(path="Settings.edge.motion_gate", choices=(0.1,)),
                SearchDimension(path="Settings.edge.motion_gate", choices=(0.2,)),
            ))

    def test_empty_space_rejected(self):
        with pytest.raises(SearchSpaceError, match="at least one dimension"):
            SearchSpace(dimensions=())


class TestSearchSpaceSampling:
    def test_grid_is_the_cross_product(self):
        space = SearchSpace(dimensions=(
            SearchDimension(path="Settings.edge.motion_gate", choices=(0.10, 0.12)),
            SearchDimension(path="Settings.cache.ttl_s", choices=(4.0, 8.0)),
        ))
        points = space.grid()
        assert len(points) == 4
        assert {(p["Settings.edge.motion_gate"], p["Settings.cache.ttl_s"]) for p in points} == {
            (0.10, 4.0), (0.10, 8.0), (0.12, 4.0), (0.12, 8.0),
        }

    def test_grid_requires_discrete_choices(self):
        space = SearchSpace(dimensions=(SearchDimension(path="Settings.edge.motion_gate", low=0.0, high=1.0),))
        with pytest.raises(SearchSpaceError, match="grid search needs discrete choices"):
            space.grid()

    def test_random_is_deterministic_given_a_seed(self):
        space = SearchSpace(dimensions=(SearchDimension(path="Settings.edge.motion_gate", low=0.0, high=1.0),))
        a = space.random(5, seed=42)
        b = space.random(5, seed=42)
        assert a == b

    def test_random_seed_changes_the_draw(self):
        space = SearchSpace(dimensions=(SearchDimension(path="Settings.edge.motion_gate", low=0.0, high=1.0),))
        a = space.random(5, seed=1)
        b = space.random(5, seed=2)
        assert a != b

    def test_unknown_strategy_raises(self):
        space = SearchSpace(dimensions=(SearchDimension(path="Settings.edge.motion_gate", choices=(0.1,)),))
        with pytest.raises(SearchSpaceError, match="unknown search strategy"):
            space.sample("bogus")

    def test_to_dict_from_dict_round_trips(self):
        space = SearchSpace(dimensions=(
            SearchDimension(path="Settings.edge.motion_gate", choices=(0.10, 0.12)),
        ))
        again = SearchSpace.from_dict(space.to_dict())
        assert again.grid() == space.grid()


# ---------------------------------------------------------------------------
# 2. Materialization + template expansion
# ---------------------------------------------------------------------------


class TestApplySearchPoint:
    def test_motion_gate_maps_onto_stat_thresholds(self):
        base = WatchProfile(name="p", environment="fixed_camera", goal_type="stative",
                            state="start_anchored", salience_stat="block_max")
        candidate = apply_search_point(base, {"Settings.edge.motion_gate_block_max": 0.20})
        assert candidate.stat_thresholds == {"block_max": 0.20}
        assert candidate.salience_stat == "block_max"  # structural axis untouched

    def test_cadence_paths_map_onto_the_cadence_sub_object(self):
        base = WatchProfile(name="p", environment="fixed_camera", goal_type="stative", state="start_anchored")
        candidate = apply_search_point(base, {"WatchProfile.cadence.base_fps": 6.0})
        assert candidate.cadence.base_fps == 6.0
        assert candidate.cadence.max_fps == base.cadence.max_fps  # untouched sibling field

    def test_unmapped_path_raises(self):
        base = WatchProfile(name="p", environment="fixed_camera", goal_type="stative", state="start_anchored")
        with pytest.raises(SearchSpaceError, match="no materialization mapping"):
            apply_search_point(base, {"Settings.entity.iou_match": 0.7})


class TestExpandTemplates:
    def test_single_point_keeps_the_template_name(self):
        expanded = expand_templates(_templates(), _space())
        names = [r.name for r, _ in expanded]
        assert names == ["mean", "block_max"]

    def test_multi_point_disambiguates_with_a_suffix(self):
        space = SearchSpace(dimensions=(SearchDimension(path="Settings.edge.motion_gate", choices=(0.10, 0.12)),))
        templates = (_templates()[0],)
        expanded = expand_templates(templates, space)
        names = [r.name for r, _ in expanded]
        assert len(names) == 2
        assert names[0] != names[1]
        assert all(n.startswith("mean@") for n in names)

    def test_no_templates_raises(self):
        with pytest.raises(SearchSpaceError, match="at least one OptimizerTemplate"):
            expand_templates((), _space())


# ---------------------------------------------------------------------------
# 3. ACCEPTANCE — block_max@0.20 wins over mean@0.10, lock written, locked re-run reproduces
# ---------------------------------------------------------------------------


class TestAcceptanceBlockMaxWinsOverMean:
    def test_optimizer_selects_block_max_over_mean(self, tmp_path: Path):
        _localized_salience_corpus(tmp_path)
        report = optimize(
            fixtures_root=tmp_path, templates=_templates(), search_space=_space(), bars=_bars(),
            base_plan_hash="plan-abc123", eval_run_id="run-accept-1",
            created_at="2026-07-22T00:00:00+00:00",
        )
        assert report.derivation.verdict is DerivationVerdict.PASS
        assert report.derivation.winner is not None
        assert report.derivation.winner.rung.name == "block_max"
        # the losing "mean" rung is still reported, scored, not elided (a scored loser is data).
        mean_rung = report.derivation.rung("mean")
        assert mean_rung is not None
        assert mean_rung.bars_met is False

        # held-out recall: mean misses 2/3, block_max catches all 3 (matches the corpus builder).
        assert mean_rung.mean("recall") == pytest.approx(2 / 3)
        assert report.derivation.winner.mean("recall") == pytest.approx(1.0)

    def test_lock_is_written_with_the_winning_values(self, tmp_path: Path):
        _localized_salience_corpus(tmp_path)
        report = optimize(
            fixtures_root=tmp_path, templates=_templates(), search_space=_space(), bars=_bars(),
            base_plan_hash="plan-abc123", eval_run_id="run-accept-2",
            created_at="2026-07-22T00:00:00+00:00",
        )
        lock = report.lock
        assert lock is not None
        assert isinstance(lock, OptimizationLock)
        assert lock.values == {
            "Settings.edge.motion_gate": 0.10,
            "Settings.edge.motion_gate_block_max": 0.20,
        }
        assert lock.base_plan_hash == "plan-abc123"
        assert lock.fixture_hash == report.derivation.registration.fixture_set_hash
        assert lock.catalog_component_hash  # non-empty: a real catalog snapshot was read
        assert lock.held_out_metrics["recall"] == pytest.approx(1.0)
        assert lock.locked_hash == lock.content_hash  # validated-at-emission, matches immediately
        assert lock.created_at == "2026-07-22T00:00:00+00:00"

    def test_locked_rerun_reproduces_the_selection_deterministically(self, tmp_path: Path):
        """Re-running the SAME optimizer inputs must reproduce the exact same lock — the whole point
        of an OptimizationLock is that its selection is not a one-off, non-reproducible artifact."""
        _localized_salience_corpus(tmp_path)
        first = optimize(
            fixtures_root=tmp_path, templates=_templates(), search_space=_space(), bars=_bars(),
            base_plan_hash="plan-abc123", eval_run_id="run-accept-3",
            created_at="2026-07-22T00:00:00+00:00",
        )
        second = optimize(
            fixtures_root=tmp_path, templates=_templates(), search_space=_space(), bars=_bars(),
            base_plan_hash="plan-abc123", eval_run_id="run-accept-3",
            created_at="2026-07-22T00:00:00+00:00",
        )
        assert first.lock is not None and second.lock is not None
        assert first.lock.content_hash == second.lock.content_hash
        assert dict(first.lock.values) == dict(second.lock.values)
        assert first.derivation.registration.registration_hash == second.derivation.registration.registration_hash

    def test_locked_values_reproduce_through_the_resolver(self, tmp_path: Path):
        """'applied explicitly via the resolver's existing precedence layer' — feed the winning
        lock back through ConfigResolver and confirm the SAME numeric values come out the other
        side, at the OptimizationLock precedence tier."""
        from percept.contracts import WatchPlan

        _localized_salience_corpus(tmp_path)
        report = optimize(
            fixtures_root=tmp_path, templates=_templates(), search_space=_space(), bars=_bars(),
            eval_run_id="run-accept-4", created_at="2026-07-22T00:00:00+00:00",
        )
        lock = report.lock
        assert lock is not None

        resolver = ConfigResolver()
        resolved = resolver.resolve(WatchPlan(condition="motion at the door"), optimization_lock=lock)
        assert resolved.effective["Settings.edge.motion_gate"] == 0.10
        assert resolved.effective["Settings.edge.motion_gate_block_max"] == 0.20
        assert resolved.provenance["Settings.edge.motion_gate_block_max"].owner_layer == LAYER_OPTIMIZATION_LOCK
        assert resolved.optimization_lock_hash == lock.content_hash
        # no drift: the lock's own claim still matches its values, so no invalidation diagnostic.
        codes = [w.code for w in resolver.last_receipt.warnings]
        assert "optimization_lock_hash_mismatch" not in codes

    def test_expected_registration_hash_guards_against_config_drift(self, tmp_path: Path):
        """The optimizer inherits derive_profile's pre-registration discipline unchanged: a
        configuration that moved after the bars were declared refuses to score, exactly as it does
        for a hand-authored ladder."""
        from percept.eval.differential import ConfigDriftError

        _localized_salience_corpus(tmp_path)
        good = optimize(fixtures_root=tmp_path, templates=_templates(), search_space=_space(), bars=_bars())
        with pytest.raises(ConfigDriftError):
            optimize(
                fixtures_root=tmp_path, templates=_templates(), search_space=_space(),
                bars=_bars(),
                expected_registration_hash=good.derivation.registration.registration_hash + "x",
            )


# ---------------------------------------------------------------------------
# 4. CLI — `percept optimize` end to end
# ---------------------------------------------------------------------------


def _write_cli_inputs(tmp_path: Path) -> Path:
    _localized_salience_corpus(tmp_path)
    space_path = tmp_path / "space.json"
    space_path.write_text(json.dumps({
        "source_profile": "physical_scene",
        "declared_at": "2026-07-22",
        "templates": [
            {"name": "mean", "fixture_dir": "mean",
             "profile": {"name": "mean-base", "environment": "fixed_camera", "goal_type": "stative",
                         "state": "start_anchored", "salience_stat": "mean"}},
            {"name": "block_max", "fixture_dir": "block_max",
             "profile": {"name": "block-base", "environment": "fixed_camera", "goal_type": "stative",
                         "state": "start_anchored", "salience_stat": "block_max"}},
        ],
        "search_space": {
            "dimensions": [
                {"path": "Settings.edge.motion_gate", "choices": [0.10]},
                {"path": "Settings.edge.motion_gate_block_max", "choices": [0.20]},
            ],
        },
        "strategy": "grid",
    }))
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({
        "declared_by": "ops",
        "bars": [{"metric": "candidate_recall", "direction": "min", "value": 0.99}],
    }))
    return space_path


class TestCLI:
    def test_cli_writes_the_lock(self, tmp_path: Path, capsys):
        space_path = _write_cli_inputs(tmp_path)
        out_lock = tmp_path / "lock.json"
        rc = main([
            "--fixtures", str(tmp_path), "--space", str(space_path),
            "--bars", str(tmp_path / "bars.json"), "--out-lock", str(out_lock),
            "--run-id", "run-cli-accept", "--declared-at", "2026-07-22",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "VERDICT: PASS" in out
        assert "block_max" in out
        body = json.loads(out_lock.read_text())
        assert body["schema_version"] == 1
        assert body["values"] == {
            "Settings.edge.motion_gate": 0.10,
            "Settings.edge.motion_gate_block_max": 0.20,
        }

    def test_cli_computes_base_plan_hash_from_a_plan_file(self, tmp_path: Path):
        from percept.contracts import WatchPlan

        space_path = _write_cli_inputs(tmp_path)
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps({"condition": "motion at the door"}))
        out_lock = tmp_path / "lock.json"
        rc = main([
            "--fixtures", str(tmp_path), "--space", str(space_path),
            "--bars", str(tmp_path / "bars.json"), "--plan", str(plan_path),
            "--out-lock", str(out_lock), "--run-id", "run-cli-plan",
        ])
        assert rc == 0
        body = json.loads(out_lock.read_text())
        expected = tracing_content_hash(WatchPlan(condition="motion at the door").to_dict())
        assert body["base_plan_hash"] == expected

    def test_cli_writes_no_lock_without_bars(self, tmp_path: Path, capsys):
        space_path = _write_cli_inputs(tmp_path)
        out_lock = tmp_path / "lock.json"
        rc = main(["--fixtures", str(tmp_path), "--space", str(space_path), "--out-lock", str(out_lock)])
        assert rc == 0
        assert not out_lock.exists()
        assert "NOT_ASSESSED" in capsys.readouterr().out

    def test_cli_refuses_on_bad_input(self, tmp_path: Path, capsys):
        rc = main(["--fixtures", str(tmp_path / "nope"), "--space", str(tmp_path / "nope.json")])
        assert rc == 2
        assert "REFUSED" in capsys.readouterr().err

    def test_load_optimizer_document_round_trips(self, tmp_path: Path):
        space_path = _write_cli_inputs(tmp_path)
        templates, space, strategy, declared_at = load_optimizer_document(space_path)
        assert [t.name for t in templates] == ["mean", "block_max"]
        assert strategy == "grid"
        assert declared_at == "2026-07-22"
        assert space.grid() == _space().grid()
