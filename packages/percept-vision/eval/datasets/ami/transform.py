"""AMI Meeting Corpus (NXT format) -> Percept ``Fixture`` transform (board card DS2).

WHAT THIS READS (reference: groups.inf.ed.ac.uk/ami — the NXT annotation stand-off format):

- ``<meeting>.<speaker>.words.xml`` — the token layer. Each ``<w nite:id=... starttime=...
  endtime=...>token</w>`` carries the wall-clock span of one word.
- ``<meeting>.<speaker>.dialog-act.xml`` — dialogue acts. Each ``<dact nite:id=...>`` points at a
  span of words via ``<nite:child href="...words.xml#id(a)..id(b)"/>``. A dact's ``start`` is the
  earliest linked word's ``starttime``; its ``end`` is the latest linked word's ``endtime``.
- ``<meeting>.abssumm.xml`` — the ABSTRACTIVE summary. ``<sentence>`` elements sit under one of four
  headings: ``ABSTRACT`` / ``DECISIONS`` / ``PROBLEMS`` / ``ACTIONS``.
- ``<meeting>.summlink.xml`` — the EXTRACTIVE link layer. Each ``<summlink>`` joins one abstractive
  summary sentence (``role="abstractive"``) to the dialogue act that evidences it
  (``role="extractive"``).

WHAT IT EMITS: one Percept :class:`~percept.eval.differential.Fixture` PER MEETING. A summary-linked
dialogue act becomes a :class:`~percept.harness.golden_eval.GroundTruthEpisode` whose watch
condition is decided by the heading of the abstractive sentence it links to:

======================  ================================
abstractive heading     watch condition
======================  ================================
DECISIONS               "a decision was made"
ACTIONS                 "an action item was assigned"
PROBLEMS                "a problem was raised"
ABSTRACT                (no watch condition — skipped)
======================  ================================

The linked dialogue act's START time becomes the episode's fire timestamp
(``decision_eligible_ts``, which :func:`percept.harness.golden_eval.build_golden_timeline` pins the
golden fire to). The act's END time is ``episode_end_ts``, so the closed matching window is the
utterance's own span plus :data:`AMI_ALLOWED_LATENESS_S` of judge/queue grace.

This is RECORDED-MEETING evaluation tooling only. The shipped ``conversation``
:class:`~percept.sourceprofile.SourceProfile` is ``stability="experimental"`` — no diarization
behind ``identity_strategy="speaker_label"``, and goals judge serially at ~24s/turn — so a live
conversation outruns the watch. Nothing here claims otherwise (see :mod:`eval.datasets.ami.pack`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from xml.etree import ElementTree as ET

from percept.contracts import ANNOTATION_RUBRIC_VERSION, GroundTruth
from percept.eval.differential import Fixture, scope_for_goal_shape
from percept.harness.golden_eval import GroundTruthEpisode

__all__ = [
    "NITE_NS",
    "AMI_ALLOWED_LATENESS_S",
    "AMI_GOAL_SHAPE",
    "AMI_ANNOTATION_VERSION",
    "SECTION_CONDITIONS",
    "AmiTransformError",
    "NxtWord",
    "NxtDialogueAct",
    "NxtSummaryLink",
    "MeetingAnnotations",
    "WatchEpisode",
    "MeetingFixture",
    "parse_href",
    "read_meeting_annotations",
    "discover_meeting_ids",
    "transform_meeting",
    "transform_corpus",
    "write_fixture_tree",
]

#: The XML namespace every ``nite:*`` attribute/element lives in (NXT stand-off annotation).
NITE_NS = "http://nite.sourceforge.net/"
_NITE = f"{{{NITE_NS}}}"

#: Grace past ``episode_end_ts`` a still-correct fire may land in, in seconds. Set to one measured
#: serial judge turn (~24s/turn — the shipped ``conversation`` profile's own documented limit): on a
#: serially-judged meeting watch, a correct fire can legitimately trail the utterance by about one
#: turn's judge latency. Documented, not tuned; tighten it on your own traces.
AMI_ALLOWED_LATENESS_S = 24.0

#: Every emitted fixture declares this goal shape. A summary-linked dialogue act is a continuous
#: labeled span with a rising edge (the moment the point is verbalised), which the ADR-006 scope
#: table marks ``SUPPORTED`` under "episode": one outward event per (goal, entity) per episode.
AMI_GOAL_SHAPE = "episode"

#: The rubric version stamped on every produced ``GroundTruth`` (Percept's shipped eligibility
#: rubric — an AMI fixture is scored by the SAME rubric a physical-scene fixture is).
AMI_ANNOTATION_VERSION = ANNOTATION_RUBRIC_VERSION

#: abstractive-summary heading -> (watch condition text, episode-id slug). ABSTRACT is deliberately
#: absent: the general abstract is not a fire-worthy event, so an ABSTRACT-only link yields no
#: episode.
SECTION_CONDITIONS: dict[str, tuple[str, str]] = {
    "DECISIONS": ("a decision was made", "decision"),
    "ACTIONS": ("an action item was assigned", "action_item"),
    "PROBLEMS": ("a problem was raised", "problem"),
}

#: abssumm element tag (lower-cased) -> canonical heading key used by :data:`SECTION_CONDITIONS`.
_SECTION_TAGS: dict[str, str] = {
    "abstract": "ABSTRACT",
    "decisions": "DECISIONS",
    "actions": "ACTIONS",
    "problems": "PROBLEMS",
}

_HREF_RE = re.compile(r"^(?P<file>[^#]*)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$")


class AmiTransformError(ValueError):
    """An NXT annotation file is missing, malformed, or links to something that does not exist."""


# ---------------------------------------------------------------------------
# NXT record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NxtWord:
    """One token of the words layer, with its wall-clock span (``None`` for untimed punctuation)."""

    word_id: str
    text: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class NxtDialogueAct:
    """One dialogue act: the words it spans, and the derived ``[start, end]`` of that span."""

    da_id: str
    word_ids: tuple[str, ...]
    start: float
    end: float


@dataclass(frozen=True)
class NxtSummaryLink:
    """One extractive->abstractive link: a dialogue act evidencing a summary sentence."""

    dialogue_act_id: str
    sentence_id: str


@dataclass(frozen=True)
class MeetingAnnotations:
    """The four NXT layers of ONE meeting, already parsed and cross-resolvable."""

    meeting_id: str
    words: Mapping[str, NxtWord]
    dialogue_acts: Mapping[str, NxtDialogueAct]
    sentence_sections: Mapping[str, str]
    summary_links: tuple[NxtSummaryLink, ...]


# ---------------------------------------------------------------------------
# Transformed (Percept-facing) record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchEpisode:
    """One summary-linked dialogue act mapped to a watch condition + its matching window.

    ``episode_id`` is ``"{dialogue_act_id}:{slug}"`` so the same act linked under two headings (a
    decision AND an action) yields two distinct, key-disjoint episodes, and the id is unique within
    the meeting fixture (the join key :func:`percept.harness.golden_eval.score` matches on).
    """

    episode_id: str
    dialogue_act_id: str
    section: str
    condition: str
    start: float
    end: float

    def to_ground_truth_episode(self) -> GroundTruthEpisode:
        """This watch episode as a Percept :class:`~percept.harness.golden_eval.GroundTruthEpisode`.

        ``event_start_ts == decision_eligible_ts == start`` (the fire timestamp is the linked act's
        start); ``episode_end_ts`` is the act's end (clamped to never precede the start).
        """
        end = self.end if self.end >= self.start else self.start
        return GroundTruthEpisode(
            episode_id=self.episode_id,
            ground_truth=GroundTruth(
                event_start_ts=self.start,
                decision_eligible_ts=self.start,
                annotation_version=AMI_ANNOTATION_VERSION,
                annotator_or_fixture_id=self.dialogue_act_id,
            ),
            episode_end_ts=end,
            allowed_lateness_s=AMI_ALLOWED_LATENESS_S,
        )

    def to_dict(self) -> dict[str, object]:
        end = self.end if self.end >= self.start else self.start
        return {
            "episode_id": self.episode_id,
            "event_start_ts": self.start,
            "decision_eligible_ts": self.start,
            "episode_end_ts": end,
            "allowed_lateness_s": AMI_ALLOWED_LATENESS_S,
            "annotation_version": AMI_ANNOTATION_VERSION,
            "annotator_or_fixture_id": self.dialogue_act_id,
        }


@dataclass(frozen=True)
class MeetingFixture:
    """One meeting's worth of watch episodes — the unit that becomes ONE Percept fixture."""

    meeting_id: str
    episodes: tuple[WatchEpisode, ...]
    conditions: Mapping[str, str] = field(default_factory=dict)

    def ground_truth_episodes(self) -> tuple[GroundTruthEpisode, ...]:
        return tuple(e.to_ground_truth_episode() for e in self.episodes)

    def to_fixture(self, path: str = "") -> Fixture:
        """A Percept :class:`~percept.eval.differential.Fixture` for this meeting."""
        return Fixture(
            fixture_id=self.meeting_id,
            goal_shape=AMI_GOAL_SHAPE,
            episodes=self.ground_truth_episodes(),
            scope=scope_for_goal_shape(AMI_GOAL_SHAPE),
            path=path,
        )

    def ground_truth_json(self) -> dict[str, object]:
        """The ``ground_truth.json`` body Percept's :func:`percept.eval.differential.load_fixtures`
        reads: ``goal_shape`` + the episode window records (sorted by episode id, so the same corpus
        always serialises byte-for-byte)."""
        return {
            "goal_shape": AMI_GOAL_SHAPE,
            "episodes": [e.to_dict() for e in sorted(self.episodes, key=lambda e: e.episode_id)],
        }


# ---------------------------------------------------------------------------
# href parsing
# ---------------------------------------------------------------------------


def parse_href(href: str) -> tuple[str, str, str | None]:
    """Split an NXT href into ``(file, first_id, last_id_or_None)``.

    Accepts a single reference ``file#id(a)`` and a contiguous range ``file#id(a)..id(b)`` — the two
    forms NXT ``nite:child`` / ``nite:pointer`` hrefs use. Raises :class:`AmiTransformError` on any
    other shape rather than silently resolving to nothing.
    """
    match = _HREF_RE.match((href or "").strip())
    if match is None:
        raise AmiTransformError(f"unparseable NXT href {href!r} (expected 'file#id(a)' or 'file#id(a)..id(b)')")
    return match.group("file"), match.group("a"), match.group("b")


# ---------------------------------------------------------------------------
# Layer readers
# ---------------------------------------------------------------------------


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_words_file(path: Path) -> list[NxtWord]:
    """Parse one ``*.words.xml`` into its timed tokens IN DOCUMENT ORDER (order the range syntax spans).

    Includes EVERY stand-off token carrying a ``nite:id`` — not just ``<w>`` but the timed non-lexical
    tokens the REAL AMI words layer interleaves (``<vocalsound>``, ``<disfmarker>``, ``<gap>``,
    ``<pause>``, ``<nonvocalsound>``, ``<transformerror>`` …). A dialogue act's ``nite:child`` range can
    legitimately START or END on one of these (measured: ``ES2002a.A.dialog-act.dharshi.6``'s sole child
    is ``id(ES2002a.A.words49)`` — a ``<vocalsound>``), so they MUST be in the ordered id index or
    :func:`_resolve_word_range` raises :class:`AmiTransformError` on a valid real act and the whole
    meeting read fails. Only ``<w>`` carries transcript text (non-lexical tokens get ``text=""``), so
    they never pollute a transcript window; but they DO carry ``starttime``/``endtime`` and are valid
    span boundaries. An element with no ``nite:id`` cannot be a range target and is skipped.

    The synthetic committed sample is ``<w>``-only, so this is a strict superset — the offline tests are
    unaffected (verified: the 18-test suite stays green).
    """
    root = _parse_xml(path)
    out: list[NxtWord] = []
    for elem in root:
        wid = elem.get(f"{_NITE}id")
        if not wid:
            continue
        out.append(
            NxtWord(
                word_id=wid,
                text=(elem.text or "").strip() if _localname(elem.tag) == "w" else "",
                start=_float_or_none(elem.get("starttime")),
                end=_float_or_none(elem.get("endtime")),
            )
        )
    return out


def _read_dialogue_acts_file(
    path: Path, words_by_file: Mapping[str, list[str]], words: Mapping[str, NxtWord]
) -> list[NxtDialogueAct]:
    """Parse one ``*.dialog-act.xml``, resolving each dact's word span into a ``[start, end]``."""
    root = _parse_xml(path)
    acts: list[NxtDialogueAct] = []
    for dact in root:
        if _localname(dact.tag) != "dact":
            continue
        da_id = dact.get(f"{_NITE}id")
        if not da_id:
            raise AmiTransformError(f"{path}: a <dact> element is missing nite:id")
        word_ids: list[str] = []
        for child in dact:
            if _localname(child.tag) != "child":
                continue
            href = child.get("href")
            if not href:
                continue
            word_ids.extend(_resolve_word_range(href, words_by_file, path))
        timed = [words[wid] for wid in word_ids if words[wid].start is not None]
        if not timed:
            # A dact with no timed words cannot be placed on the timeline — skip it rather than
            # invent a timestamp.
            continue
        start = min(w.start for w in timed)  # type: ignore[type-var]
        end = max((w.end for w in timed if w.end is not None), default=start)
        acts.append(
            NxtDialogueAct(da_id=da_id, word_ids=tuple(word_ids), start=float(start), end=float(end))
        )
    return acts


def _resolve_word_range(
    href: str, words_by_file: Mapping[str, list[str]], source: Path
) -> list[str]:
    """Expand an ``nite:child`` href into the ordered word ids it spans (inclusive range)."""
    file, first, last = parse_href(href)
    order = words_by_file.get(Path(file).name)
    if order is None:
        raise AmiTransformError(f"{source}: dact links to unknown words file {file!r}")
    index = {wid: i for i, wid in enumerate(order)}
    if first not in index:
        raise AmiTransformError(f"{source}: dact links to unknown word id {first!r}")
    if last is None:
        return [first]
    if last not in index:
        raise AmiTransformError(f"{source}: dact links to unknown word id {last!r}")
    lo, hi = index[first], index[last]
    if hi < lo:
        lo, hi = hi, lo
    return order[lo : hi + 1]


def _read_abssumm_file(path: Path) -> dict[str, str]:
    """Parse one ``*.abssumm.xml`` into ``sentence_id -> heading`` (ABSTRACT/DECISIONS/...)."""
    root = _parse_xml(path)
    sections: dict[str, str] = {}
    for heading in root:
        section = _SECTION_TAGS.get(_localname(heading.tag).lower())
        if section is None:
            continue
        for sentence in heading:
            if _localname(sentence.tag) != "sentence":
                continue
            sid = sentence.get(f"{_NITE}id")
            if not sid:
                raise AmiTransformError(f"{path}: a <sentence> under {section} is missing nite:id")
            sections[sid] = section
    return sections


def _read_summlink_file(path: Path) -> list[NxtSummaryLink]:
    """Parse one ``*.summlink.xml`` into extractive(dact) -> abstractive(sentence) links."""
    root = _parse_xml(path)
    links: list[NxtSummaryLink] = []
    for summlink in root:
        if _localname(summlink.tag) != "summlink":
            continue
        extractive: str | None = None
        abstractive: str | None = None
        for pointer in summlink:
            if _localname(pointer.tag) != "pointer":
                continue
            role = pointer.get("role")
            href = pointer.get("href")
            if not href:
                continue
            _file, target, _last = parse_href(href)
            if role == "extractive":
                extractive = target
            elif role == "abstractive":
                abstractive = target
        if extractive and abstractive:
            links.append(NxtSummaryLink(dialogue_act_id=extractive, sentence_id=abstractive))
    return links


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _parse_xml(path: Path) -> ET.Element:
    if not path.exists():
        raise AmiTransformError(f"NXT file not found: {path}")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise AmiTransformError(f"{path}: malformed XML — {exc}") from exc


def _localname(tag: str) -> str:
    """The tag name with any ``{namespace}`` prefix stripped."""
    return tag.rsplit("}", 1)[-1]


# ---------------------------------------------------------------------------
# Meeting assembly + discovery
# ---------------------------------------------------------------------------


def _find(base: Path, pattern: str) -> list[Path]:
    """``rglob`` ``pattern`` under ``base``, skipping hidden / AppleDouble (``._*``) sidecar files.

    ``rglob`` (not ``glob``) so BOTH layouts resolve from one code path: the committed synthetic sample
    is a FLAT ``testdata/nxt`` dir, while the REAL AMI NXT export splits layers into per-``layer``
    subdirectories (``words/``, ``dialogueActs/``, ``abstractive/``, ``extractive/`` …). ``rglob``
    matches a flat top-level file too, so the offline tests are unaffected. The ``._*`` filter drops the
    macOS AppleDouble sidecars an extraction onto a non-HFS volume (e.g. an exFAT external drive) leaves
    next to every real file — those would otherwise mint bogus meeting ids like ``._ES2002a``.
    """
    return sorted(p for p in base.rglob(pattern) if not p.name.startswith("."))


def discover_meeting_ids(root: str | Path) -> list[str]:
    """Every meeting id under ``root``, derived from the one ``<meeting>.abssumm.xml`` per meeting."""
    base = Path(root)
    if not base.is_dir():
        raise AmiTransformError(f"AMI corpus root is not a directory: {base}")
    ids = sorted({p.name[: -len(".abssumm.xml")] for p in _find(base, "*.abssumm.xml")})
    if not ids:
        raise AmiTransformError(f"no *.abssumm.xml under {base} — is this an AMI NXT export?")
    return ids


def read_meeting_annotations(root: str | Path, meeting_id: str) -> MeetingAnnotations:
    """Read and cross-resolve all four NXT layers of ``meeting_id`` under ``root``."""
    base = Path(root)

    words: dict[str, NxtWord] = {}
    words_by_file: dict[str, list[str]] = {}
    for wf in _find(base, f"{meeting_id}.*.words.xml"):
        file_words = _read_words_file(wf)
        words_by_file[wf.name] = [w.word_id for w in file_words]
        for w in file_words:
            words[w.word_id] = w

    dialogue_acts: dict[str, NxtDialogueAct] = {}
    for df in _find(base, f"{meeting_id}.*.dialog-act.xml"):
        for act in _read_dialogue_acts_file(df, words_by_file, words):
            dialogue_acts[act.da_id] = act

    abssumm_matches = _find(base, f"{meeting_id}.abssumm.xml")
    abssumm_path = abssumm_matches[0] if abssumm_matches else base / f"{meeting_id}.abssumm.xml"
    sentence_sections = _read_abssumm_file(abssumm_path)

    summlink_matches = _find(base, f"{meeting_id}.summlink.xml")
    summary_links = tuple(_read_summlink_file(summlink_matches[0])) if summlink_matches else ()

    return MeetingAnnotations(
        meeting_id=meeting_id,
        words=words,
        dialogue_acts=dialogue_acts,
        sentence_sections=sentence_sections,
        summary_links=summary_links,
    )


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def transform_meeting(annotations: MeetingAnnotations) -> MeetingFixture:
    """Map one meeting's summary-linked dialogue acts to watch episodes.

    A link is kept only when (a) its abstractive sentence sits under DECISIONS/ACTIONS/PROBLEMS and
    (b) its extractive dialogue act was timed. Episodes are de-duplicated by ``episode_id`` (the same
    act linked twice under the same heading is one episode) and sorted by ``(start, episode_id)``.
    """
    episodes: dict[str, WatchEpisode] = {}
    conditions: dict[str, str] = {}
    for link in annotations.summary_links:
        section = annotations.sentence_sections.get(link.sentence_id)
        mapping = SECTION_CONDITIONS.get(section or "")
        if mapping is None:
            continue  # ABSTRACT-only (or an unknown heading) is not a watch condition
        condition, slug = mapping
        act = annotations.dialogue_acts.get(link.dialogue_act_id)
        if act is None:
            continue  # link points at an untimed / missing act
        episode_id = f"{act.da_id}:{slug}"
        conditions[slug] = condition
        if episode_id in episodes:
            continue
        episodes[episode_id] = WatchEpisode(
            episode_id=episode_id,
            dialogue_act_id=act.da_id,
            section=section or "",
            condition=condition,
            start=act.start,
            end=act.end,
        )
    ordered = tuple(sorted(episodes.values(), key=lambda e: (e.start, e.episode_id)))
    return MeetingFixture(meeting_id=annotations.meeting_id, episodes=ordered, conditions=conditions)


def transform_corpus(root: str | Path) -> list[MeetingFixture]:
    """Transform every meeting under ``root`` into a :class:`MeetingFixture` (empty ones dropped).

    A meeting whose summary links produced no decision/action/problem episode is skipped: Percept's
    ``load_fixtures`` refuses a fixture with zero episodes, so emitting one would be a broken tree.
    """
    fixtures: list[MeetingFixture] = []
    for meeting_id in discover_meeting_ids(root):
        fixture = transform_meeting(read_meeting_annotations(root, meeting_id))
        if fixture.episodes:
            fixtures.append(fixture)
    if not fixtures:
        raise AmiTransformError(
            f"no meeting under {root} produced any decision/action/problem episode "
            "(no summary links resolved to a timed dialogue act)"
        )
    return fixtures


def write_fixture_tree(
    fixtures: Sequence[MeetingFixture],
    out_root: str | Path,
    *,
    splits: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    """Write ``fixtures`` as the ``<fixtures>/<meeting_id>/ground_truth.json`` tree Percept loads.

    When ``splits`` is given, a ``splits.json`` is written at the root so the tree round-trips
    through :func:`percept.eval.differential.load_held_out_split`. Returns ``out_root``.
    """
    base = Path(out_root)
    base.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        d = base / fixture.meeting_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "ground_truth.json").write_text(
            json.dumps(fixture.ground_truth_json(), indent=2, sort_keys=False) + "\n"
        )
    if splits is not None:
        payload = {name: list(splits.get(name, [])) for name in ("train", "dev", "test")}
        (base / "splits.json").write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return base
