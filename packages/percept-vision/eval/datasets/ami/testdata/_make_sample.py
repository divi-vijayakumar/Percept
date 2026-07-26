"""Regenerate the committed synthetic AMI NXT sample and the fixtures it produces.

This is NOT the AMI corpus. It is a handful of NXT-format-REPRESENTATIVE records — words +
dialogue-acts + an abstractive summary + summary links — small enough to commit, real enough in
shape to exercise the whole transform (href ranges, section headings, extractive links, the
skip-ABSTRACT rule, the same-act-two-headings case, and unlinked acts). It downloads nothing.

Run from the package root::

    python3 -m eval.datasets.ami.testdata._make_sample

It rewrites ``testdata/nxt/`` (the sample) and ``testdata/fixtures/`` (the produced fixture tree +
splits.json), so a reviewer can diff both and confirm the fixtures ARE what the transform emits.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from eval.datasets.ami.split import build_meeting_split
from eval.datasets.ami.transform import transform_corpus, write_fixture_tree

_HERE = Path(__file__).resolve().parent
NXT_DIR = _HERE / "nxt"
FIXTURES_DIR = _HERE / "fixtures"

_WORD_DUR = 0.4  # seconds per word
_GAP = 1.0  # seconds of silence between two dialogue acts


@dataclass
class Act:
    """One dialogue act to synthesise: its id suffix, speaker, and the words it spans."""

    suffix: str
    speaker: str
    words: list[str]


@dataclass
class Meeting:
    """One synthetic meeting: dialogue acts, summary sentences per heading, and the links."""

    meeting_id: str
    acts: list[Act]
    #: heading (ABSTRACT/DECISIONS/ACTIONS/PROBLEMS) -> list[(sentence_suffix, text)]
    summary: dict[str, list[tuple[str, str]]]
    #: (act_suffix, sentence_suffix) extractive->abstractive links
    links: list[tuple[str, str]] = field(default_factory=list)


# --- the corpus: 4 meetings, so a 50/50 meeting split is 2 held-out / 2 exploratory -------------

MEETINGS: list[Meeting] = [
    Meeting(
        meeting_id="ES0001a",
        acts=[
            Act("1", "A", ["Let", "us", "go", "with", "the", "rubber", "case"]),
            Act("2", "B", ["The", "battery", "life", "worries", "me"]),
            Act("3", "A", ["Anyway", "welcome", "everyone"]),  # -> ABSTRACT (skipped)
            Act("4", "B", ["I", "think", "so", "too"]),  # unlinked
        ],
        summary={
            "ABSTRACT": [("s1", "The team opened the design review.")],
            "DECISIONS": [("s2", "The team decided on a rubber case.")],
            "ACTIONS": [("s3", "The designer will source rubber casing.")],
            "PROBLEMS": [("s4", "Battery life is an unresolved concern.")],
        },
        # act 1 is linked under BOTH decisions AND actions -> two key-disjoint episodes.
        links=[("1", "s2"), ("1", "s3"), ("2", "s4"), ("3", "s1")],
    ),
    Meeting(
        meeting_id="ES0001b",
        acts=[
            Act("1", "A", ["We", "will", "ship", "in", "March"]),
            Act("2", "B", ["And", "we", "pick", "the", "blue", "theme"]),
            Act("3", "A", ["Marketing", "will", "draft", "the", "brief"]),
        ],
        summary={
            "ABSTRACT": [("s1", "A planning meeting.")],
            "DECISIONS": [
                ("s2", "The team decided to ship in March."),
                ("s3", "The team decided on the blue theme."),
            ],
            "ACTIONS": [("s4", "Marketing will draft the brief.")],
        },
        links=[("1", "s2"), ("2", "s3"), ("3", "s4")],
    ),
    Meeting(
        meeting_id="IS0001a",
        acts=[
            Act("1", "A", ["Let", "us", "adopt", "the", "new", "logo"]),
            Act("2", "B", ["The", "schedule", "is", "too", "tight"]),
        ],
        summary={
            "DECISIONS": [("s1", "The team adopted the new logo.")],
            "PROBLEMS": [("s2", "The schedule is too tight.")],
        },
        links=[("1", "s1"), ("2", "s2")],
    ),
    Meeting(
        meeting_id="TS0001a",
        acts=[
            Act("1", "A", ["Engineering", "will", "run", "the", "load", "test"]),
            Act("2", "B", ["The", "vendor", "keeps", "missing", "deadlines"]),
            Act("3", "A", ["We", "go", "with", "supplier", "two"]),
        ],
        summary={
            "ACTIONS": [("s1", "Engineering will run the load test.")],
            "PROBLEMS": [("s2", "The vendor keeps missing deadlines.")],
            "DECISIONS": [("s3", "The team chose supplier two.")],
        },
        links=[("1", "s1"), ("2", "s2"), ("3", "s3")],
    ),
]

_NITE_ROOT = '<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="{root_id}">'


def _words_xml(meeting: Meeting, speaker: str) -> tuple[str, list[str]]:
    """The words.xml body for one speaker + the ordered word ids it declares."""
    lines = [_NITE_ROOT.format(root_id=f"{meeting.meeting_id}.{speaker}.words")]
    ids: list[str] = []
    clock = 1.0
    idx = 0
    for act in meeting.acts:
        if act.speaker != speaker:
            continue
        for token in act.words:
            wid = f"{meeting.meeting_id}.{speaker}.words{idx}"
            start = round(clock, 2)
            end = round(clock + _WORD_DUR, 2)
            lines.append(
                f'  <w nite:id="{wid}" starttime="{start}" endtime="{end}">{token}</w>'
            )
            ids.append(wid)
            clock = end
            idx += 1
        clock += _GAP
    lines.append("</nite:root>")
    return "\n".join(lines) + "\n", ids


def _dialogue_act_xml(meeting: Meeting, speaker: str, word_ids: dict[str, list[str]]) -> str:
    """The dialog-act.xml body for one speaker, spanning its acts' words via a range href."""
    lines = [_NITE_ROOT.format(root_id=f"{meeting.meeting_id}.{speaker}.dialog-act")]
    cursor = 0
    ids = word_ids[speaker]
    for act in meeting.acts:
        if act.speaker != speaker:
            continue
        n = len(act.words)
        span = ids[cursor : cursor + n]
        cursor += n
        da_id = f"{meeting.meeting_id}.{speaker}.dialog-act.{act.suffix}"
        words_file = f"{meeting.meeting_id}.{speaker}.words.xml"
        if len(span) == 1:
            href = f"{words_file}#id({span[0]})"
        else:
            href = f"{words_file}#id({span[0]})..id({span[-1]})"
        lines.append(f'  <dact nite:id="{da_id}">')
        lines.append(f'    <nite:child href="{href}"/>')
        lines.append("  </dact>")
    lines.append("</nite:root>")
    return "\n".join(lines) + "\n"


def _abssumm_xml(meeting: Meeting) -> str:
    lines = [_NITE_ROOT.format(root_id=f"{meeting.meeting_id}.abssumm")]
    for heading in ("ABSTRACT", "DECISIONS", "ACTIONS", "PROBLEMS"):
        sentences = meeting.summary.get(heading)
        if not sentences:
            continue
        tag = heading.lower()
        lines.append(f"  <{tag}>")
        for suffix, text in sentences:
            sid = f"{meeting.meeting_id}.abssumm.{suffix}"
            lines.append(f'    <sentence nite:id="{sid}">{text}</sentence>')
        lines.append(f"  </{tag}>")
    lines.append("</nite:root>")
    return "\n".join(lines) + "\n"


def _summlink_xml(meeting: Meeting) -> str:
    lines = [_NITE_ROOT.format(root_id=f"{meeting.meeting_id}.summlink")]
    speaker_of = {act.suffix: act.speaker for act in meeting.acts}
    for i, (act_suffix, sent_suffix) in enumerate(meeting.links):
        speaker = speaker_of[act_suffix]
        da_id = f"{meeting.meeting_id}.{speaker}.dialog-act.{act_suffix}"
        sid = f"{meeting.meeting_id}.abssumm.{sent_suffix}"
        da_file = f"{meeting.meeting_id}.{speaker}.dialog-act.xml"
        lines.append(f'  <summlink nite:id="{meeting.meeting_id}.summlink.{i}">')
        lines.append(f'    <nite:pointer role="extractive" href="{da_file}#id({da_id})"/>')
        lines.append(
            f'    <nite:pointer role="abstractive" href="{meeting.meeting_id}.abssumm.xml#id({sid})"/>'
        )
        lines.append("  </summlink>")
    lines.append("</nite:root>")
    return "\n".join(lines) + "\n"


def write_sample() -> None:
    if NXT_DIR.exists():
        shutil.rmtree(NXT_DIR)
    NXT_DIR.mkdir(parents=True)
    for meeting in MEETINGS:
        speakers = sorted({act.speaker for act in meeting.acts})
        word_ids: dict[str, list[str]] = {}
        for speaker in speakers:
            body, ids = _words_xml(meeting, speaker)
            word_ids[speaker] = ids
            (NXT_DIR / f"{meeting.meeting_id}.{speaker}.words.xml").write_text(body)
        for speaker in speakers:
            (NXT_DIR / f"{meeting.meeting_id}.{speaker}.dialog-act.xml").write_text(
                _dialogue_act_xml(meeting, speaker, word_ids)
            )
        (NXT_DIR / f"{meeting.meeting_id}.abssumm.xml").write_text(_abssumm_xml(meeting))
        (NXT_DIR / f"{meeting.meeting_id}.summlink.xml").write_text(_summlink_xml(meeting))


def write_fixtures() -> None:
    if FIXTURES_DIR.exists():
        shutil.rmtree(FIXTURES_DIR)
    fixtures = transform_corpus(NXT_DIR)
    splits = build_meeting_split([f.meeting_id for f in fixtures])
    write_fixture_tree(fixtures, FIXTURES_DIR, splits=splits)


def main() -> None:
    write_sample()
    write_fixtures()
    print(f"wrote synthetic NXT sample -> {NXT_DIR}")
    print(f"wrote produced fixtures    -> {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
