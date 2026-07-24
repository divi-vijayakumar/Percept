"""Unit tests: bars + fixtures load through Percept's OWN loaders, and the screen pack is valid."""

from __future__ import annotations

from pathlib import Path

from percept.domain_pack import DomainPackRegistry
from percept.eval.differential import load_bars, load_fixtures, load_held_out_split

from eval.datasets.guiworld.pack import SCREEN_OBSERVATION_V1, register_entrypoint_pack

HERE = Path(__file__).resolve().parent
BARS = HERE / "bars.json"
FIXTURES_DIR = HERE / "testdata" / "fixtures"


# --- bars load through percept.eval.differential.load_bars ----------------------------------


def test_bars_load_through_percept_loader():
    doc = load_bars(BARS)
    assert doc.declared
    by_metric = {b.metric: b for b in doc.bars}
    assert set(by_metric) == {"candidate_recall", "recall_delta", "false_fire_delta"}
    assert by_metric["candidate_recall"].direction == "min"
    assert by_metric["recall_delta"].direction == "min"
    assert by_metric["recall_delta"].require_ci_excludes_zero is True
    assert by_metric["false_fire_delta"].direction == "max"
    assert by_metric["false_fire_delta"].value == 0.0


# --- the committed fixture tree loads through the eval loaders -------------------------------


def test_committed_fixtures_load_through_load_fixtures():
    fixtures = load_fixtures(FIXTURES_DIR)
    assert len(fixtures) == 13
    for fixture in fixtures:
        assert fixture.episodes
        assert fixture.scope.scorable


def test_committed_split_loads_through_load_held_out_split():
    fixtures = load_fixtures(FIXTURES_DIR)
    split = load_held_out_split(FIXTURES_DIR, fixtures, split_name="test")
    assert split.declared
    assert split.held_out and split.exploratory
    # Every named fixture exists (load_held_out_split raises otherwise); belt-and-braces here.
    ids = {f.fixture_id for f in fixtures}
    assert set(split.held_out) <= ids
    assert set(split.exploratory) <= ids


# --- the screen_observation pack -------------------------------------------------------------


def test_pack_identity_and_templates():
    assert SCREEN_OBSERVATION_V1.qualified_name == "screen_observation@1"
    assert set(SCREEN_OBSERVATION_V1.templates) == {
        "modal_appeared",
        "error_shown",
        "page_changed",
    }


def test_pack_templates_render_screen_watch_plans():
    modal = SCREEN_OBSERVATION_V1.template("modal_appeared").render()
    assert modal.condition == "a modal dialog appeared"
    assert modal.source == "screen"
    error = SCREEN_OBSERVATION_V1.template("error_shown").render()
    assert error.condition == "an error message is shown"
    page = SCREEN_OBSERVATION_V1.template("page_changed").render()
    assert page.condition == "the page changed"


def test_pack_notes_preserve_the_shipped_screen_finding():
    notes = SCREEN_OBSERVATION_V1.notes.lower()
    # no salience stat admits a page-scale edit ...
    assert "salience statistic" in notes and "page-scale" in notes
    # ... the coast floor stays mandatory ...
    assert "coast floor stays mandatory" in notes
    # ... the client change signal remains A/B-able ...
    assert "percept eval" in notes and "register_detector" in notes
    # ... identity is content, never geometry; window, never montage.
    assert "content key, never geometry" in notes
    assert "ordered frame window, never a sprite montage" in notes


def test_register_entrypoint_pack_registers_into_a_fresh_registry():
    registry = DomainPackRegistry()
    register_entrypoint_pack(registry.register)
    assert registry.resolve("screen_observation@1") is SCREEN_OBSERVATION_V1


def test_pack_module_has_no_import_side_effect_on_default_registry():
    # The scaffold must NOT auto-register (that is the documented one-line follow-up), so a fresh
    # default registry does not already carry the pack.
    from percept.domain_pack import DEFAULT_DOMAIN_PACK_REGISTRY

    assert "screen_observation@1" not in DEFAULT_DOMAIN_PACK_REGISTRY
