"""— the ADRs are a map of the code, so the map must be checkable.

Four stale-citation classes shipped in the ADR set and misled readers who trusted them:

* ADR-006 pinned ``CONTRACT_VERSION = "1.2.0"`` and discussed ``Incident.entity_id`` as unshipped,
  when 1.3.0 was bumped precisely FOR that field and ``make_incident`` already wires it;
* ADR-006's ``percept/gate.py`` line cites had drifted off by tens of lines;
* ADR-005 cited the private ``percept/harness/detectors.py`` path for ``register_detector`` while
  the contracts module documents ``percept.edge.detectors`` as the public one;
* ADR-004 routed an integrator needing liveness to a ``WatchHealth`` channel that does not exist.

These tests do not re-check prose. They check the small set of load-bearing FACTS a reader acts on,
each against the code rather than against another document — so the next drift fails CI instead of
reaching a client debugging at 2am.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import percept.gate as gate_mod
import percept.incidents as incidents_mod
from percept.contracts import CONTRACT_VERSION, Incident

ADR_DIR = Path(__file__).resolve().parents[3] / ".adr"
PKG = Path(inspect.getfile(gate_mod)).resolve().parent


def _adr(name: str) -> str:
    p = ADR_DIR / name
    if not p.exists():  # docs/ live outside the installed package
        pytest.skip(f"{p} not present in this checkout")
    return p.read_text()


def _line(rel: str, n: int) -> str:
    """Return 1-indexed line ``n`` of ``percept/<rel>``."""
    return (PKG / rel).read_text().splitlines()[n - 1]


# --------------------------------------------------------------------------- (a) contract version


def test_adr006_quotes_the_live_contract_version():
    """ADR-006 pins CONTRACT_VERSION in two places; both must equal the code's value."""
    text = _adr("ADR-006-incident-relations.md")
    quoted = set(re.findall(r'CONTRACT_VERSION = "([0-9.]+)"', text))
    assert quoted, "ADR-006 no longer quotes CONTRACT_VERSION at all"
    assert quoted == {CONTRACT_VERSION}, (
        f"ADR-006 quotes CONTRACT_VERSION {sorted(quoted)} but the code says {CONTRACT_VERSION!r}"
    )


def test_entity_id_is_shipped_and_wired_not_prospective():
    """The field ADR-006 §7 documents as shipped really is shipped AND populated."""
    assert "entity_id" in Incident.__dataclass_fields__
    assert Incident.__dataclass_fields__["entity_id"].default is None

    # make_incident must READ the fire's own entity_id — that is what "wired" means here.
    src = inspect.getsource(incidents_mod.make_incident)
    assert 'getattr(fire, "entity_id", None)' in src

    class _Fire:
        goal_id = "g1"
        verdict = None
        text = "hi"
        entity_id = "e7"

    assert incidents_mod.make_incident(_Fire(), ts=1.0).entity_id == "e7"


def test_adr006_does_not_describe_entity_id_as_unshipped():
    text = _adr("ADR-006-incident-relations.md")
    assert "SHIPPED in 1.3.0" in text, "ADR-006 §7 must state entity_id's shipped status"


# --------------------------------------------------------------------------- (b) gate.py cites


#: (cited line, a token that MUST appear on it). Every gate.py anchor ADR-006 leans on.
GATE_ANCHORS = [
    (13, "Rising-edge"),
    (16, "confident-false and re-arms"),
    (32, "Suppression is observable"),
    (79, "_DEFAULT_MAX_KEYS"),
    (80, "_DEFAULT_KEY_TTL_S"),
    (82, "GateAction = Literal"),
    (123, "GATE_REASONS"),
    (134, ")"),
    (138, "REASON_ACTION"),
    (162, "SUPPRESSION_REASONS"),
    (239, "def classify("),
    (318, "evaluate`` is the only firing path"),
    # The six per-key facts. They were six parallel dicts; they are now the fields of ONE
    # `_KeyState` record, which is what made `_drop`/`reset_goal`/`reset_entity` and the `max_keys`
    # denominator agree. Each field's comment still names the dict it replaced, so the ADR's cite
    # resolves to the state it is talking about.
    (281, "_armed"),
    (282, "_dwell_since"),
    (285, "_no_evict"),
    (286, "_last_fire"),
    (517, "def decide("),
    (607, "confident-false re-arms"),
    (632, "effective_refractory"),
    (638, '"refractory"'),
    (774, "def reset_entity("),
    (808, "def snapshot("),
]


@pytest.mark.parametrize("lineno,token", GATE_ANCHORS)
def test_gate_line_anchors_still_hold(lineno, token):
    """Every gate.py line ADR-006 cites must still contain what the ADR says it does."""
    assert token in _line("gate.py", lineno), (
        f"gate.py:{lineno} no longer contains {token!r} — ADR-006's cites need re-deriving"
    )


#: The corrected gate.py cites, verbatim, as ADR-006 now writes them.
GATE_CITES = [
    ":13-16", ":32-42", ":79-80", ":82", ":123-134", ":138-149", ":162",
    ":239-255", ":318", ":281-286", ":282", ":359-360", ":285",
    ":517", ":607", ":632-638", ":774", ":808-970",
]


@pytest.mark.parametrize("cite", GATE_CITES)
def test_adr006_still_carries_the_rederived_gate_cite(cite):
    """...and ADR-006 must still carry the re-derived cite, not slide back to a stale neighbour."""
    text = _adr("ADR-006-incident-relations.md")
    assert f"`{cite}`" in text or f"gate.py{cite}`" in text, (
        f"ADR-006 no longer cites gate.py{cite}"
    )


def test_adr006_no_longer_cites_the_stale_gate_ranges():
    """The corrected stale gate citations must not come back."""
    text = _adr("ADR-006-incident-relations.md")
    for stale in ("gate.py:48-49", ":101-139", "gate.py:107-139", ":338-453", "gate.py:255-266"):
        assert stale not in text, f"stale gate.py cite {stale!r} is back in ADR-006"


# --------------------------------------------------------------------------- (c) public detector path


def test_register_detector_public_path_resolves():
    """percept.edge.detectors is the documented public path and must really re-export it."""
    from percept.contracts import __file__ as contracts_file

    assert "percept.edge.detectors.register_detector" in Path(contracts_file).read_text()

    edge = pytest.importorskip("percept.edge.detectors")
    harness = pytest.importorskip("percept.harness.detectors")
    assert edge.register_detector is harness.register_detector


def test_adr005_cites_the_public_detector_path():
    text = _adr("ADR-005-source-profiles.md")
    assert "`percept/edge/detectors.py` (`register_detector`" in text
    assert "percept.edge.detectors.register_detector" in text


# --------------------------------------------------------------------------- (d) WatchHealth is unbuilt


def test_watchhealth_does_not_exist_anywhere_under_percept():
    """The premise of the ADR-004 fix. If someone builds it, this test tells them to update ADR-004."""
    hits = [
        p.relative_to(PKG)
        for p in PKG.rglob("*.py")
        if "WatchHealth" in p.read_text(errors="ignore")
    ]
    assert hits == [], (
        f"WatchHealth now exists ({hits}) — ADR-004 §1 and ADR-002's status must be updated"
    )


def test_the_health_channel_adr004_now_points_at_is_real():
    from percept.runtime.worker import RuntimeHealthEvent, run

    assert {"kind", "ts", "data"} <= set(RuntimeHealthEvent.__dataclass_fields__)
    assert "on_health" in inspect.signature(run).parameters
    assert "on_frame" in inspect.signature(run).parameters

    # The terminal coverage signal makes this channel worth routing an integrator to.
    worker_src = (PKG / "runtime" / "worker.py").read_text()
    assert 'RuntimeHealthEvent("runtime_ended"' in worker_src
    assert 'RuntimeHealthEvent("zero_admission_coverage"' in worker_src


def test_adr004_no_longer_routes_liveness_to_watchhealth():
    text = _adr("ADR-004-delivery-flow-replay.md")
    assert "`WatchHealth`/`on_health` channel ADR-002 defines" not in text
    assert "UNBUILT" in text, "ADR-004 must mark ADR-002's WatchHealth as unbuilt"
    assert "RuntimeHealthEvent" in text, "ADR-004 must name the channel that does exist"
