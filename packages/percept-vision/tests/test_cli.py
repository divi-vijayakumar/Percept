"""``percept`` umbrella CLI — delegation to the existing scripts, plus the new resolve/config/
components subcommands.

Delegation is tested by asserting the SAME module's ``main`` runs (argparse's own ``--help`` exit
is proof enough that the real parser was reached — this module invents no parsing of its own for
delegated commands, per its docstring). ``resolve``/``config``/``components`` are the new surface
and get direct coverage here, mirroring ``tests/test_replay_cli.py``'s own
import-the-module-and-call-``main``-with-an-argv-list convention (no subprocess spawning).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.cli import main

#: Absolute, so these tests pass regardless of pytest's working directory (repo root, package
#: root, or anywhere else a runner invokes it from).
_EXAMPLE_PLAN = str(Path(__file__).resolve().parent.parent / "examples" / "watch_plan.json")


def test_no_args_prints_help_and_exits_2(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 2
    assert "percept resolve" in out or "resolve" in out
    assert "watch" in out and "doctor" in out


def test_top_help_exits_0(capsys):
    rc = main(["-h"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "percept-doctor" in out


def test_unknown_command_exits_2(capsys):
    rc = main(["not-a-command"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_delegation_reaches_the_real_doctor_parser():
    """`percept doctor --help` must hit percept.doctor's OWN argparse parser (proof the two
    commands are byte-identical) — argparse's --help path raises SystemExit(0)."""
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--help"])
    assert exc.value.code == 0


def test_delegation_reaches_the_real_replay_parser():
    with pytest.raises(SystemExit) as exc:
        main(["replay", "--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# percept resolve
# ---------------------------------------------------------------------------


def test_resolve_bare_condition_explain(capsys):
    rc = main(["resolve", "--condition", "a person is at the door", "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ResolvedWatch (plan_hash=" in out
    assert "Settings.gate.threshold" in out


def test_resolve_shipped_example_end_to_end(capsys):
    """The shipped examples/watch_plan.json resolves against the physical_scene SourceProfile —
    the exit-criteria demo path ('percept resolve --explain works on a shipped profile')."""
    rc = main(["resolve", _EXAMPLE_PLAN, "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SourceProfile.name = 'physical_scene'" in out
    assert "<- SourceProfile (physical_scene)" in out


def test_resolve_json_output_round_trips(capsys):
    rc = main(["resolve", _EXAMPLE_PLAN, "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["source_profile"]["SourceProfile.name"] == "physical_scene"
    assert payload["component_snapshot_hash"]
    assert "providers" in payload["components"]
    assert "catalog_component_hash" in payload["components"]


def test_resolve_default_output_is_a_summary_not_full_provenance(capsys):
    rc = main(["resolve", "--condition", "a dog barks"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved" in out
    assert "plan_hash=" in out
    assert "ResolvedWatch (plan_hash=" not in out  # that's --explain's format, not the default


def test_resolve_missing_condition_and_config_is_an_error(capsys):
    rc = main(["resolve"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "config file or --condition" in err


def test_resolve_unknown_source_profile_is_an_error(capsys):
    rc = main(["resolve", "--condition", "x", "--source-profile", "not_a_real_profile"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not_a_real_profile" in err


def test_resolve_missing_file_is_an_error():
    with pytest.raises(SystemExit):
        main(["resolve", "no/such/file.json"])


# ---------------------------------------------------------------------------
# percept config tunables
# ---------------------------------------------------------------------------


def test_config_tunables_lists_every_registry_entry(capsys):
    from percept.tunables import REGISTRY

    rc = main(["config", "tunables"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{len(REGISTRY)} tunable(s)" in out
    assert "Settings.gate.threshold" in out


def test_config_tunables_owner_filter(capsys):
    rc = main(["config", "tunables", "--owner", "Gate"])
    assert rc == 0
    out = capsys.readouterr().out
    for line in out.splitlines():
        if "tunable(s)" in line:
            continue
        assert "(Gate," in line


def test_config_tunables_json_matches_generate(capsys):
    from percept.tunables import generate

    rc = main(["config", "tunables", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == generate() + "\n"  # print() adds one more trailing newline on top of generate()'s own


# ---------------------------------------------------------------------------
# percept components
# ---------------------------------------------------------------------------


def test_components_list_all(capsys):
    from percept import catalog

    rc = main(["components", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{len(catalog.list_components())} component(s)" in out


def test_components_list_filtered_by_role(capsys):
    rc = main(["components", "list", "--role", "sense"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sense:physical_scene" in out
    assert "3 component(s)" in out


def test_components_describe_role_qualified(capsys):
    rc = main(["components", "describe", "sense:physical_scene"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "role:      sense" in out
    assert "name:      physical_scene" in out


def test_components_describe_json(capsys):
    rc = main(["components", "describe", "sense:physical_scene", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["role"] == "sense"
    assert payload["name"] == "physical_scene"


def test_components_describe_not_found(capsys):
    rc = main(["components", "describe", "sense:nope_not_real"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no component matches" in err


def test_components_describe_ambiguous_bare_name_lists_roles(capsys):
    """physical_scene exists only under 'sense' today, so this is a control: an ambiguous bare
    name across roles must ask for a role: prefix rather than silently pick one."""
    from percept import catalog

    same_name_roles = [
        c.role for c in catalog.list_components() if c.name == "physical_scene"
    ]
    if len(same_name_roles) > 1:  # pragma: no cover - only if the catalog grows a name clash
        rc = main(["components", "describe", "physical_scene"])
        assert rc == 2
    else:
        rc = main(["components", "describe", "physical_scene"])
        assert rc == 0
