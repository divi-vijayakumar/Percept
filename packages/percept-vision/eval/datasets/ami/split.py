"""Pre-registered, MEETING-disjoint held-out split for the AMI fixtures (board card DS2).

The one rule that protects the trust metric on a meeting corpus: split by MEETING, never by
utterance or dialogue act. Two dialogue acts from the same meeting share a speaker, a room and a
topic; letting one land in ``dev`` and its neighbour in ``test`` leaks the held-out set. Because the
transform emits exactly one fixture per meeting (:mod:`eval.datasets.ami.transform`), a
fixture-level split IS a meeting-level split — :func:`build_meeting_split` assigns whole fixture ids,
and :func:`percept.eval.differential.load_held_out_split` (via ``percept.eval.load_splits``) enforces
disjointness across ``train``/``dev``/``test`` on load.

:class:`MeetingSplitRegistration` is the ``DerivationPreRegistration``-style freeze: it content-
addresses the fixture set's LABELS (via :func:`percept.eval.derivation.fixture_set_hash`), the exact
held-out split, and the declared bars into one ``registration_hash`` (via
:func:`percept.tracing.content_hash`, the same hasher the shipped pre-registrations use). Pass the
hash back in as ``expected_registration_hash`` and a corpus, split or bars that moved after the bars
were declared raises :class:`~percept.eval.differential.ConfigDriftError` — the same anti-tune-to-
green refusal ``percept.eval.derivation`` performs, applied to this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from percept.eval.derivation import fixture_set_hash
from percept.eval.differential import (
    BarsDocument,
    ConfigDriftError,
    Fixture,
    HeldOutSplit,
    load_fixtures,
    load_held_out_split,
)
from percept.tracing import content_hash

from .transform import MeetingFixture

__all__ = [
    "SOURCE_PROFILE_NAME",
    "build_meeting_split",
    "meeting_split_for",
    "MeetingSplitRegistration",
    "freeze_registration",
]

#: The source class these fixtures were transformed for (ADR-005). A recorded meeting is a
#: conversation, not a camera or a screen — and that profile is EXPERIMENTAL (see the pack notes).
SOURCE_PROFILE_NAME = "conversation"


def build_meeting_split(
    meeting_ids: Sequence[str], *, split_name: str = "test"
) -> dict[str, list[str]]:
    """A deterministic 50/50 held-out split over WHOLE meetings.

    Sort the meeting ids, then interleave: even ranks -> the held-out ``split_name`` (default
    ``test``), odd ranks -> ``dev`` (exploratory). ``train`` stays empty. Deterministic (no RNG, no
    wall time), exactly balanced to within one meeting, and meeting-disjoint by construction — no id
    can be assigned to two sides. Duplicated ids are rejected loudly rather than silently collapsed.
    """
    ids = list(meeting_ids)
    if len(set(ids)) != len(ids):
        dupes = sorted({m for m in ids if ids.count(m) > 1})
        raise ValueError(f"meeting ids must be unique to split by meeting; duplicated: {dupes}")
    if split_name not in ("dev", "test"):
        raise ValueError(f"split_name must be 'dev' or 'test', got {split_name!r}")
    other = "test" if split_name == "dev" else "dev"
    held: list[str] = []
    exploratory: list[str] = []
    for rank, meeting in enumerate(sorted(ids)):
        (held if rank % 2 == 0 else exploratory).append(meeting)
    return {"train": [], other: exploratory, split_name: held}


@dataclass(frozen=True)
class MeetingSplitRegistration:
    """Everything fixed BEFORE any score exists, content-addressed into one ``registration_hash``.

    Mirrors :class:`percept.eval.derivation.DerivationPreRegistration`: same ``payload()`` /
    ``registration_hash`` / ``seed`` / ``assert_no_drift`` shape, without pulling in a WatchProfile
    ladder (this card owns the fixtures and the split, not the eye-tuning ladder a later derivation
    card supplies).
    """

    fixture_set_hash: str
    split: HeldOutSplit
    bars: BarsDocument
    source_profile: str = SOURCE_PROFILE_NAME
    goal_shape: str = "episode"
    split_name: str = "test"
    declared_at: str = ""
    expected_registration_hash: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "dataset": "ami-meeting-corpus",
            "source_profile": self.source_profile,
            "goal_shape": self.goal_shape,
            "fixture_set_hash": self.fixture_set_hash,
            "split": self.split.to_dict(),
            "bars": [b.to_dict() for b in self.bars.bars],
        }

    @property
    def registration_hash(self) -> str:
        return content_hash(self.payload())

    @property
    def bars_hash(self) -> str:
        return content_hash([b.to_dict() for b in self.bars.bars])

    @property
    def seed(self) -> int:
        """Deterministic bootstrap seed — derived from the registration, never from wall time."""
        return int(self.registration_hash[:12], 16)

    def assert_no_drift(self) -> None:
        """Refuse if the corpus/split/bars moved after the bars were declared."""
        if self.expected_registration_hash is None:
            return
        actual = self.registration_hash
        if actual != self.expected_registration_hash:
            raise ConfigDriftError(
                "REFUSING: the AMI fixture set, held-out split or bars changed after the bars were "
                "declared.\n"
                f"  pre-registered registration_hash: {self.expected_registration_hash}\n"
                f"  configuration presented now:      {actual}\n"
                "The fixture labels, the meeting-disjoint split and the bars are all covered by this "
                "hash. Re-declare the bars against the new corpus and re-run."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_at": self.declared_at,
            "registration_hash": self.registration_hash,
            "expected_registration_hash": self.expected_registration_hash,
            "bootstrap_seed": self.seed,
            "bars_hash": self.bars_hash,
            "bars_source": self.bars.source,
            **self.payload(),
        }


def freeze_registration(
    fixtures_root: str | Path,
    bars: BarsDocument,
    *,
    split_name: str = "test",
    declared_at: str = "",
    expected_registration_hash: str | None = None,
    fixtures: Sequence[Fixture] | None = None,
) -> MeetingSplitRegistration:
    """Load the fixture tree + its ``splits.json`` through Percept's OWN loaders and freeze them.

    ``fixtures`` may be passed to reuse an already-loaded set; otherwise
    :func:`percept.eval.differential.load_fixtures` reads ``fixtures_root``. The split is read by
    :func:`percept.eval.differential.load_held_out_split` (which enforces disjointness), so the
    frozen split is exactly the one a ``percept eval`` run would score against.
    """
    loaded = list(fixtures) if fixtures is not None else list(load_fixtures(fixtures_root))
    split = load_held_out_split(fixtures_root, loaded, split_name=split_name)
    return MeetingSplitRegistration(
        fixture_set_hash=fixture_set_hash(loaded),
        split=split,
        bars=bars,
        split_name=split_name,
        declared_at=declared_at,
        expected_registration_hash=expected_registration_hash,
    )


def meeting_split_for(meetings: Sequence[MeetingFixture], *, split_name: str = "test") -> dict[str, list[str]]:
    """Convenience: build the split straight from transformed :class:`MeetingFixture` objects."""
    return build_meeting_split([m.meeting_id for m in meetings], split_name=split_name)
