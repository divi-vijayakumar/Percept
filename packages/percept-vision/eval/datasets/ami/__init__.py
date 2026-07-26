"""AMI Meeting Corpus (NXT) -> Percept recorded-meeting fixtures and domain pack.

Public surface:

- :func:`~eval.datasets.ami.transform.transform_corpus` — read an AMI NXT export directory and emit
  one :class:`~eval.datasets.ami.transform.MeetingFixture` per meeting.
- :func:`~eval.datasets.ami.transform.write_fixture_tree` — serialise those to the
  ``<fixtures>/<meeting>/ground_truth.json`` (+ ``splits.json``) tree Percept's loaders read.
- :func:`~eval.datasets.ami.split.build_meeting_split` — a deterministic 50/50 held-out split BY
  MEETING; :class:`~eval.datasets.ami.split.MeetingSplitRegistration` freezes it + the fixture-set
  hash + the bars (a ``DerivationPreRegistration``-style pre-registration).
- :data:`~eval.datasets.ami.pack.MEETING_OBSERVATION_V1` — the self-contained meeting DomainPack
  (register it via :func:`~eval.datasets.ami.pack.register_meeting_pack`; NOT auto-registered).
- :func:`bars_path` / :func:`load_meeting_bars` — the committed acceptance-bars document.

Everything here is dev tooling (NOT wheel-shipped) and touches NO real dataset — a small synthetic
NXT sample + the fixtures it produces live under ``testdata/`` so the whole path is unit-testable
offline.
"""

from __future__ import annotations

from pathlib import Path

from percept.eval.differential import BarsDocument, load_bars

from .pack import MEETING_OBSERVATION_V1, register_meeting_pack
from .split import (
    MeetingSplitRegistration,
    build_meeting_split,
    freeze_registration,
    meeting_split_for,
)
from .transform import (
    AMI_ALLOWED_LATENESS_S,
    AMI_ANNOTATION_VERSION,
    AMI_GOAL_SHAPE,
    SECTION_CONDITIONS,
    AmiTransformError,
    MeetingFixture,
    WatchEpisode,
    discover_meeting_ids,
    read_meeting_annotations,
    transform_corpus,
    transform_meeting,
    write_fixture_tree,
)

#: Directory holding the committed synthetic NXT sample + the fixtures it produces.
TESTDATA_ROOT = Path(__file__).resolve().parent / "testdata"


def bars_path() -> Path:
    """Absolute path to the committed acceptance-bars document (``bars.json``)."""
    return Path(__file__).resolve().parent / "bars.json"


def load_meeting_bars() -> BarsDocument:
    """The committed bars, loaded through Percept's own :func:`percept.eval.differential.load_bars`."""
    return load_bars(bars_path())


__all__ = [
    "AMI_ALLOWED_LATENESS_S",
    "AMI_ANNOTATION_VERSION",
    "AMI_GOAL_SHAPE",
    "SECTION_CONDITIONS",
    "AmiTransformError",
    "MeetingFixture",
    "WatchEpisode",
    "discover_meeting_ids",
    "read_meeting_annotations",
    "transform_meeting",
    "transform_corpus",
    "write_fixture_tree",
    "build_meeting_split",
    "meeting_split_for",
    "MeetingSplitRegistration",
    "freeze_registration",
    "MEETING_OBSERVATION_V1",
    "register_meeting_pack",
    "TESTDATA_ROOT",
    "bars_path",
    "load_meeting_bars",
]
