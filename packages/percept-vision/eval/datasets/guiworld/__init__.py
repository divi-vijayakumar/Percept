"""GUI-World -> Percept ``screen_document`` fixtures (dev tooling; NOT wheel-shipped).

GUI-World (ICLR 2025; HuggingFace ``ONE-Lab/GUI-World``; https://gui-world.github.io) annotates
keyframe/temporal GUI events across desktop / web / mobile / multi-window recordings under
CC BY 4.0 (shippable). This package reads GUI-World-shaped keyframe annotations and emits Percept
:class:`percept.eval.differential.Fixture` objects for three screen watch conditions:

* a dialog / modal appearing  -> ``"a modal dialog appeared"``
* an error surface            -> ``"an error message is shown"``
* a navigation / page change  -> ``"the page changed"``

The annotated event time becomes the :class:`~percept.harness.golden_eval.GroundTruthEpisode`
fire timestamp. Fixtures honor ``screen_document`` semantics: identity is a CONTENT key (never
geometry), episodes are an ORDERED time window (never a sprite montage), and every episode's match
window is widened by the mandatory ``screen_document`` coast floor so admission never depends on a
salience statistic alone.

SeeAction is deliberately NOT used here: it is CC BY-NC-ND (non-commercial), unusable for a shipped
pack. Only GUI-World (CC BY 4.0) is transformed.

No network, no model calls, no dataset download happens in this package: the transform is pure and
operates on already-parsed annotation records. A small synthetic, GUI-World-representative sample
lives under ``testdata/`` so the whole path is unit-testable with no real dataset.
"""

from .transform import (
    ERROR_SHOWN,
    MODAL_APPEARED,
    PAGE_CHANGED,
    SCREEN_CONDITIONS,
    GuiWorldKeyframe,
    GuiWorldSession,
    ScreenCondition,
    classify_event,
    fixture_to_ground_truth_dict,
    load_sessions,
    parse_session,
    session_to_fixtures,
    sessions_to_fixtures,
    write_fixture_tree,
)

__all__ = [
    "ScreenCondition",
    "MODAL_APPEARED",
    "ERROR_SHOWN",
    "PAGE_CHANGED",
    "SCREEN_CONDITIONS",
    "GuiWorldKeyframe",
    "GuiWorldSession",
    "classify_event",
    "parse_session",
    "load_sessions",
    "session_to_fixtures",
    "sessions_to_fixtures",
    "fixture_to_ground_truth_dict",
    "write_fixture_tree",
]
