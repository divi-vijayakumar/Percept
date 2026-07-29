"""Drift guard for ADR-003's premise.

ADR-003 previously declared ``Depends on: ADR-001 §4 (defines evidence.frame_ref as a field on the
judgment contract)``. Neither half of that resolved: ADR-001 §4 is *Dwell and refractory*, and no
evidence pointer exists on the judgment contract at all. These tests pin the *true* starting
position in code, so the ADR cannot silently drift back onto a false premise.

The code is the ground truth here; the ADR is the thing under test.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from percept.contracts import JudgmentEvent, LabelRecord, Verdict, WatchEvent

_ADR_DIR = Path(__file__).resolve().parents[3] / ".adr"
_ADR_001 = _ADR_DIR / "ADR-001-judgment-and-gate-contract.md"
_ADR_003 = _ADR_DIR / "ADR-003-evidence-artifacts-retention.md"

_needs_docs = pytest.mark.skipif(
    not _ADR_003.is_file(), reason="ADR tree not present (installed package, not a source checkout)"
)


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


# --------------------------------------------------------------------------------------------
# The code half: what the judgment contract actually carries today.
# --------------------------------------------------------------------------------------------


def test_verdict_evidence_is_plain_text_not_a_handle() -> None:
    """``Verdict.evidence`` is free-text rationale, not a pointer to an artifact."""
    ann = {f.name: f.type for f in dataclasses.fields(Verdict)}
    assert "evidence" in ann
    assert "str" in str(ann["evidence"])
    assert Verdict(satisfied=True, confidence=1.0).evidence == ""
    assert "frame_ref" not in _fields(Verdict)
    assert "evidence_ref" not in _fields(Verdict)


def test_judgment_event_carries_no_evidence_pointer() -> None:
    """``JudgmentEvent`` has no evidence payload; ``evidence_seq`` is an ordering int, not a ref."""
    names = _fields(JudgmentEvent)
    # the implementation enriched the envelope with scalar OBSERVABILITY metadata (reason/watchplan_id/roi/
    # modality/provenance); evidence_origin names which path produced the evidence and occurrence_id
    # identifies one temporal occurrence. Neither scalar is a handle to raw evidence bytes. This test
    # guards that evidence_seq remains an ordering int, no field is a ``*_ref`` pointer, and there is
    # no media/frame handle. A Box ``roi`` is coordinates, not a pixel pointer.
    assert names == {
        "key", "evidence_seq", "action", "timing",
        "reason", "watchplan_id", "roi", "modality", "provenance", "evidence_origin",
        "occurrence_id",
    }
    assert not any(n.endswith("_ref") for n in names)
    assert not ({"frame_ref", "evidence_ref", "media", "frames"} & names)


def test_watch_event_carries_no_evidence_handle() -> None:
    names = _fields(WatchEvent)
    assert "frame_ref" not in names
    assert "evidence_ref" not in names
    assert not any(n.endswith("_ref") for n in names)


def test_label_record_frame_ref_is_the_only_one_and_is_egress_excluded() -> None:
    """The single in-tree ``frame_ref`` is a LOCAL path on a label, structurally un-egressable."""
    assert "frame_ref" in _fields(LabelRecord)
    assert "frame_ref" not in LabelRecord.EGRESS_FIELDS
    rec = LabelRecord(
        incident_id="i1", verdict_ref="v1", goal_id="g1", ts=0.0, good=True, frame_ref="/tmp/a.jpg"
    )
    assert "frame_ref" not in rec.to_egress()
    assert rec.to_local_dict()["frame_ref"] == "/tmp/a.jpg"


# --------------------------------------------------------------------------------------------
# The docs half: ADR-003 must not re-assert the false premise.
# --------------------------------------------------------------------------------------------


@_needs_docs
def test_adr_001_section_4_is_dwell_and_refractory() -> None:
    """The section ADR-003 used to cite as the frame_ref definition is about something else."""
    heads = re.findall(r"^## §(\d+) — (.+)$", _ADR_001.read_text(), flags=re.M)
    by_num = dict(heads)
    assert "dwell" in by_num["4"].lower()
    assert "frame_ref" not in _ADR_001.read_text()


@_needs_docs
def test_adr_003_does_not_claim_adr_001_defines_frame_ref() -> None:
    text = _ADR_003.read_text()
    depends = text.split("- **Related:**", 1)[0]
    assert "ADR-001 §4" not in depends
    # The false claim may only survive inside the explicit correction note.
    correction = [ln for ln in text.splitlines() if ln.startswith(">")]
    for line in text.splitlines():
        if "defines `evidence.frame_ref`" in line:
            assert line in correction, "false premise restated outside the correction note"


@_needs_docs
def test_adr_003_scopes_section_1_to_a_new_field() -> None:
    text = _ADR_003.read_text()
    section_1 = text.split("### 1. ", 1)[1].split("### 2. ", 1)[0]
    assert "evidence_ref" in section_1
    assert "new" in section_1.lower()
    # It must explicitly disclaim the existing LabelRecord field it would otherwise contradict.
    assert "LabelRecord.frame_ref" in section_1


@_needs_docs
def test_adr_003_cites_the_label_record_precedent_by_line() -> None:
    text = _ADR_003.read_text()
    assert "L-1269" in text, "LabelRecord.frame_ref must be cited by line"
    assert "L-1277" in text or "EGRESS_FIELDS" in text
    src = (Path(__file__).resolve().parents[1] / "percept" / "contracts" / "__init__.py").read_text()
    for needle in ("class LabelRecord", "frame_ref", "EGRESS_FIELDS"):
        # assert the SYMBOL EXISTS, not the line it sits on — additive contract edits (SubjectRef,
        # the claim-mutation channel) shift every line below and must not break a citation test.
        assert needle in src, f"ADR-003 premise: {needle!r} vanished from contracts source"



@_needs_docs
def test_adr_003_status_is_still_proposed() -> None:
    assert "- **Status:** Proposed" in _ADR_003.read_text()
