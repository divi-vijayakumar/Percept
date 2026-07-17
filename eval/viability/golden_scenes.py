"""Card A32 (Viability §5a SECONDARY real-clip validation set) — golden real-clip loader + scorer.

PRE-MVP VIABILITY / SCAFFOLDING. This module turns the 48 labeled real scenes in
``eval/fixtures-golden/labels.jsonl`` (+ ``eval/fixtures-golden/scenes/<scene_id>/``) into
:class:`~percept.harness.golden_eval.GroundTruthEpisode` records (card A14) and scores predicted
:class:`~percept.contracts.JudgmentEvent` sets against them via
:func:`percept.harness.golden_eval.score`.

It works ENTIRELY from the JSON labels — it never decodes ``clip.mp4``/``frames/`` pixels and never
runs a judge. The videos exist for a *future*, separately-gated, for-record real-VLM validation run
(bars-locked; needs provider keys); this module is the loader/scoring wiring for that run, not the
run itself.

################################################################################################
§5a CAVEAT — THIS IS A SECONDARY VALIDATION SUBSTRATE, NOT THE PRIMARY GATE SUBSTRATE
################################################################################################

Viability §5a names TWO substrates for ``decision_eligible_ts`` (see
``percept.contracts.ANNOTATION_RUBRIC``):

1. BY CONSTRUCTION (primary, core gating archetypes) — a synthetic fixture generator IS the ground
   truth (``eval/viability/fixtures/core_archetypes.py``). This is the PRIMARY substrate the core
   gating decision is measured against, and it is NOT touched by this module.
2. ANNOTATED REAL CLIPS (secondary, two-annotator) — full §5a rigor wants TWO independent human
   annotators plus adjudication of any disagreement (``GroundTruth.from_two_annotator``), with a
   reported inter-annotator agreement number.

The 48 real scenes loaded here have only a SINGLE reviewer's approved/pinned label per
``rising_edges[]`` record (``review_status``/``verify`` in ``labels.jsonl``) — there is no second
annotator and no adjudication trail. That single-reviewer provenance is recorded structurally, not
just in this docstring: every :class:`~percept.contracts.GroundTruth` built here carries
``annotator_or_fixture_id="single-reviewer:<reviewer>"`` rather than pretending to be a
two-annotator record. Do NOT wire the output of this module into the core PRE-MVP VIABILITY gating
decision — it is a secondary real-clip validation signal to watch alongside the primary synthetic
gate, not a replacement for it.

################################################################################################
Rising edges -> GroundTruthEpisode
################################################################################################

Per ``rising_edges[]`` record:

* ``decision_eligible_ts`` = ``rising_edges[].t`` — the reviewer-corrected/pinned instant the
  watched condition becomes true. This is exactly Percept's rising-edge FIRE model / the runtime's
  ``decision_eligible_ts`` concept.
* The episode window's END comes from the ``events[]`` record whose ``type`` string equals the
  rising edge's ``condition`` string (the join key ``labels.jsonl`` uses between the two lists —
  there is no explicit foreign key). When a match is found, ``episode_end_ts = events[].t_end`` and
  ``event_start_ts = events[].t_start``. When NO ``events[]`` record shares the rising edge's exact
  condition text (a handful of scenes describe more rising edges than labeled event spans, e.g.
  ``eval3_s27``'s "mail truck leaves the scene" has no matching event), there is no labeled span to
  fall back to: both ``event_start_ts`` and ``episode_end_ts`` default to the rising edge's own
  ``t`` (a zero-width span at the pinned instant), and the caller-supplied ``allowed_lateness_s`` is
  the ONLY source of window width for that episode.
* Defensive clamp: a few single-reviewer labels are internally inconsistent — e.g. ``eval2_s01``'s
  pinned rising edge (``t=130.0``) lands strictly AFTER its own matched event's labeled
  ``t_end=94.0``. Rather than raising (which would hard-fail the loader on one noisy label out of
  48 real scenes), this is clamped: ``event_start_ts = min(events_t_start, t)`` and
  ``episode_end_ts = max(events_t_end, t)``, so the two frozen-contract invariants
  (``GroundTruth.decision_eligible_ts >= event_start_ts`` and
  ``GroundTruthEpisode.episode_end_ts >= ground_truth.decision_eligible_ts``) always hold. Each
  :class:`RealClipEpisode` records whether this clamp fired via ``label_inconsistent`` so a caller
  can filter these out if it wants a stricter set — another concrete symptom of the single-reviewer
  §5a caveat above, not something this loader silently launders away.

Reused, never edited: :class:`percept.contracts.GroundTruth`, :class:`percept.contracts.JudgmentEvent`,
:data:`percept.contracts.ANNOTATION_RUBRIC_VERSION`,
:class:`percept.harness.golden_eval.GroundTruthEpisode`, :func:`percept.harness.golden_eval.score`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Make `percept` importable the same way whether this module is imported as part of a package or
# run/loaded standalone (mirrors eval/viability/test_ground_truth.py's convention).
_PKG_SRC = Path(__file__).resolve().parents[2] / "packages" / "percept-vision"
if str(_PKG_SRC) not in sys.path and _PKG_SRC.is_dir():
    sys.path.insert(0, str(_PKG_SRC))

from percept.contracts import ANNOTATION_RUBRIC_VERSION, GroundTruth, JudgmentEvent  # noqa: E402
from percept.harness.golden_eval import (  # noqa: E402
    DEFAULT_PREDICTED_ACTIONS,
    GoldenEvalResult,
    GroundTruthEpisode,
    score as _golden_score,
)

#: Default path to the real-clip labels this loader reads (repo-root-relative:
#: ``eval/fixtures-golden/labels.jsonl``), resolved from this file's own location so the loader
#: works regardless of the caller's cwd.
DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[1] / "fixtures-golden" / "labels.jsonl"

#: Grace period (seconds) added past an episode's own (possibly zero-width, see module docstring)
#: window when the caller does not supply its own. Real clips have noisier/coarser event
#: boundaries than the synthetic-by-construction core archetypes, so this default is intentionally
#: more generous than ``GroundTruthEpisode.allowed_lateness_s``'s own zero-grace default.
DEFAULT_ALLOWED_LATENESS_S = 1.0

#: §5a substrate caveat, spelled out as a data value (not just prose in the module docstring) so a
#: caller/report can surface it programmatically, and so :func:`test_golden_scenes` can assert on
#: it directly rather than parsing the docstring.
SECONDARY_SUBSTRATE_CAVEAT = (
    "eval/viability/golden_scenes.py loads SINGLE-REVIEWER approved/pinned real-clip labels "
    "(review_status/verify in eval/fixtures-golden/labels.jsonl). Full Viability §5a rigor for "
    "real (annotated) clips wants TWO independent annotators plus adjudication of any "
    "disagreement (see percept.contracts.GroundTruth.from_two_annotator) with a reported "
    "inter-annotator agreement number -- neither exists for these 48 scenes. This is therefore a "
    "SECONDARY validation substrate: it must never be wired into the core PRE-MVP VIABILITY gating "
    "decision, which is measured against the synthetic-by-construction core archetypes in "
    "eval/viability/fixtures/core_archetypes.py instead."
)


@dataclass(frozen=True)
class RealClipEpisode:
    """One real-clip ``rising_edges[]`` record, joined into a :class:`GroundTruthEpisode` plus the
    scene-level metadata (`scene_id`, `condition`, `split`, ...) the raw ``GroundTruthEpisode``
    (card A14, imported/never edited here) has no field for.

    ``episode`` is what :func:`score_real_clip_predictions` (and, directly,
    ``percept.harness.golden_eval.score``) actually scores against; the remaining fields are
    provenance/filtering metadata carried alongside it.
    """

    episode: GroundTruthEpisode
    scene_id: str
    condition: str
    split: str
    review_status: str
    verify: str
    reviewer: str
    #: True iff a labeled ``events[]`` record shared this rising edge's exact ``condition`` text
    #: (see module docstring) — False means the episode window is the degenerate/zero-width
    #: fallback driven only by ``allowed_lateness_s``.
    matched_event: bool
    #: True iff the labeled event span and the pinned rising-edge time were mutually inconsistent
    #: and had to be clamped to satisfy the GroundTruth/GroundTruthEpisode invariants (see module
    #: docstring) — a concrete single-reviewer §5a data-quality symptom, not a loader bug.
    label_inconsistent: bool


def load_scene_records(labels_path: Path | str = DEFAULT_LABELS_PATH) -> list[dict]:
    """Parse ``labels.jsonl`` into a list of raw per-scene dict records (one line = one scene).

    No filtering, no interpretation — this is the raw JSON, one dict per ``scene_id``. Use
    :func:`load_real_clip_episodes` to get scored-ready :class:`RealClipEpisode` objects instead.
    """
    records: list[dict] = []
    with open(labels_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _match_event(events: Sequence[dict], condition: str) -> dict | None:
    """The join between one ``rising_edges[]`` record and ``events[]``: first event whose ``type``
    string equals the rising edge's ``condition`` string (see module docstring — there is no
    explicit foreign key in the label schema)."""
    for event in events:
        if event.get("type") == condition:
            return event
    return None


def _build_episode(
    *,
    scene_id: str,
    idx: int,
    rising_edge: dict,
    events: Sequence[dict],
    reviewer: str,
    allowed_lateness_s: float,
) -> tuple[GroundTruthEpisode, bool, bool]:
    """Build one :class:`GroundTruthEpisode` for a single ``rising_edges[]`` record.

    Returns ``(episode, matched_event, label_inconsistent)`` — see module docstring for the
    matching/clamping rules.
    """
    t = float(rising_edge["t"])
    matched = _match_event(events, rising_edge["condition"])

    if matched is not None:
        raw_start = float(matched["t_start"])
        raw_end = float(matched["t_end"])
    else:
        # No labeled event span shares this rising edge's condition text -- degenerate,
        # zero-width fallback; allowed_lateness_s is the only source of window width.
        raw_start = t
        raw_end = t

    event_start_ts = min(raw_start, t)
    episode_end_ts = max(raw_end, t)
    label_inconsistent = (raw_start > t) or (raw_end < t)

    ground_truth = GroundTruth(
        event_start_ts=event_start_ts,
        decision_eligible_ts=t,
        annotation_version=ANNOTATION_RUBRIC_VERSION,
        # Single-reviewer provenance, structurally distinct from a two-annotator
        # `annotator_a+annotator_b` id (see GroundTruth.from_two_annotator) -- the §5a caveat is
        # encoded in the record itself, not just in this module's docstring.
        annotator_or_fixture_id=f"single-reviewer:{reviewer}",
    )
    episode = GroundTruthEpisode(
        episode_id=f"{scene_id}#{idx}",
        ground_truth=ground_truth,
        episode_end_ts=episode_end_ts,
        allowed_lateness_s=allowed_lateness_s,
    )
    return episode, matched is not None, label_inconsistent


def load_real_clip_episodes(
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    *,
    approved_only: bool = True,
    pinned_only: bool = True,
    allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
) -> list[RealClipEpisode]:
    """Load the real-clip ``labels.jsonl`` into scored-ready :class:`RealClipEpisode` records.

    Defaults to the STRICTEST usable secondary-validation slice (§5a caveat above still applies
    regardless): ``approved_only=True`` keeps only scenes with ``review_status == "approved"``,
    and ``pinned_only=True`` keeps only ``rising_edges[]`` records with ``verify == "pinned"``
    (reviewer-corrected, not merely proposed/unverified). Pass ``approved_only=False`` and/or
    ``pinned_only=False`` to widen the set for exploratory use — but the PRIMARY gate substrate
    remains the synthetic core archetypes regardless of these flags (see module docstring).

    A scene with zero ``rising_edges[]`` (13/48 scenes have none at all) contributes zero episodes,
    not an error.
    """
    episodes: list[RealClipEpisode] = []
    for rec in load_scene_records(labels_path):
        if approved_only and rec.get("review_status") != "approved":
            continue
        scene_id = rec["scene_id"]
        split = rec["split"]
        review_status = rec["review_status"]
        reviewer = rec.get("reviewer", "unknown-reviewer")
        events = rec.get("events", [])

        for idx, rising_edge in enumerate(rec.get("rising_edges", [])):
            verify = rising_edge.get("verify", "unverified")
            if pinned_only and verify != "pinned":
                continue
            episode, matched_event, label_inconsistent = _build_episode(
                scene_id=scene_id,
                idx=idx,
                rising_edge=rising_edge,
                events=events,
                reviewer=reviewer,
                allowed_lateness_s=allowed_lateness_s,
            )
            episodes.append(
                RealClipEpisode(
                    episode=episode,
                    scene_id=scene_id,
                    condition=rising_edge["condition"],
                    split=split,
                    review_status=review_status,
                    verify=verify,
                    reviewer=reviewer,
                    matched_event=matched_event,
                    label_inconsistent=label_inconsistent,
                )
            )
    return episodes


def episodes_for_scene(
    episodes: Sequence[RealClipEpisode], scene_id: str
) -> list[RealClipEpisode]:
    """Filter a :func:`load_real_clip_episodes` result down to one scene, preserving order."""
    return [ep for ep in episodes if ep.scene_id == scene_id]


def score_real_clip_predictions(
    episodes: Sequence[RealClipEpisode],
    predicted_events: Sequence[JudgmentEvent],
    *,
    predicted_actions=DEFAULT_PREDICTED_ACTIONS,
) -> GoldenEvalResult:
    """Score ``predicted_events`` against ``episodes`` via ``percept.harness.golden_eval.score``
    UNCHANGED (card A14's §5a event-level matching rule) — this is a thin unwrap of
    :class:`RealClipEpisode` -> :class:`GroundTruthEpisode` plus the pass-through call, not a
    reimplementation of any scoring logic. See ``golden_eval.score`` for the matching semantics.

    Raises the same :class:`ValueError` ``golden_eval.score``/``GroundTruthEpisode`` would if
    ``episodes`` contains duplicate ``episode_id``s (e.g. two different scenes' loads concatenated
    carelessly, or the same scene loaded twice) -- episode_id is `f"{scene_id}#{idx}"`, unique
    within one :func:`load_real_clip_episodes` call, but a caller must not silently merge two
    independently-loaded lists that could collide.
    """
    return _golden_score(
        [ep.episode for ep in episodes],
        predicted_events,
        predicted_actions=predicted_actions,
    )


__all__ = [
    "DEFAULT_ALLOWED_LATENESS_S",
    "DEFAULT_LABELS_PATH",
    "SECONDARY_SUBSTRATE_CAVEAT",
    "RealClipEpisode",
    "episodes_for_scene",
    "load_real_clip_episodes",
    "load_scene_records",
    "score_real_clip_predictions",
]
