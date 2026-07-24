"""GUI-World keyframe annotations -> Percept ``screen_document`` fixtures.

Input shape (a GUI-World-representative session record; one JSON object per recording)::

    {
      "video_id": "gw-web-01",          # SESSION id — the split unit (must contain no "__")
      "platform": "web",                # desktop | web | mobile | multi_window (provenance only)
      "fps": 30.0,                      # optional; used only to resolve a keyframe given as "frame"
      "duration_s": 58.0,              # optional; provenance only
      "annotator": "guiworld-keyframe", # optional; provenance only
      "keyframes": [
        {"t": 6.2, "event": "modal_open",
         "caption": "A cookie-consent modal dialog appears over the article",
         "content_key": "cookie-consent"},
        {"t": 21.8, "event": "navigate",
         "caption": "The user clicks a link and the page changes to the pricing page",
         "content_key": "pricing"},
        {"t": 40.1, "event": "error_toast",
         "caption": "A red error toast reads 'Payment failed'"},
        {"t": 12.0, "event": "scroll", "caption": "The user scrolls the article"}
      ]
    }

A keyframe's ``event`` type (and, when it is unknown, keyword fallback over its ``caption``) is
mapped to at most one of three :class:`ScreenCondition`. A keyframe that maps to none (``scroll``,
``hover``, ``type``, ...) is DROPPED — not every GUI event is a watch condition.

One :class:`~percept.eval.differential.Fixture` is emitted per ``(session, condition)`` with at
least one mapped keyframe. Its ``fixture_id`` is ``f"{video_id}__{condition_id}"`` so the split unit
(the session) is recoverable with :func:`session_of` and a whole session is never split across the
held-out boundary.

Screen semantics honored here (see ``percept.source_profiles_default.SCREEN_DOCUMENT``):

* **content_key identity, never geometry.** The per-episode key is derived from the keyframe's
  ``content_key`` (document / page / dialog id), never from any screen coordinate — no geometry
  field is ever read. Two visually-identical modals in the same region but with different content
  keys become two distinct episodes.
* **ordered window, never a sprite montage.** Episodes within a fixture are ordered by event time.
* **the coast floor is preserved.** Each episode's match window is widened by the mandatory
  ``screen_document`` coast interval (``allowed_lateness_s = SCREEN_COAST_INTERVAL_S``) so a fire
  admitted by the fixed low-rate coast — not only one admitted by the ``block_max`` salience
  statistic — still lands inside the window. Admission never depends on the salience stat alone.

Pure: no I/O beyond the explicit ``load_sessions`` / ``write_fixture_tree`` helpers, no network, no
model calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from percept.contracts import GroundTruth
from percept.eval.differential import Fixture, scope_for_goal_shape
from percept.harness.golden_eval import GroundTruthEpisode
from percept.source_profiles_default import SCREEN_COAST_INTERVAL_S

#: Provenance stamped onto every emitted episode's ``GroundTruth.annotation_version``. A run that
#: silently mixed this with another annotation source is a scoring bug this field exists to catch.
GUIWORLD_ANNOTATION_VERSION = "guiworld-keyframe-v1"

#: Separator between the session id and the condition id inside a ``fixture_id``. A session id must
#: never contain it, so :func:`session_of` can recover the split unit unambiguously.
FIXTURE_SEP = "__"


# ---------------------------------------------------------------------------
# 1. The three screen watch conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenCondition:
    """One screen watch condition an emitted fixture is scored against.

    - ``condition_id`` — the slug used in the ``fixture_id`` and as the pack template name.
    - ``condition`` — the plain-language watch condition text.
    - ``goal_shape`` — a shape the one-to-one event-level matcher SUPPORTS (``point`` / ``episode``);
      it drives :func:`percept.eval.differential.scope_for_goal_shape`.
    - ``eligible_after_s`` — offset from the annotated event time to the first-eligible-judgment
      instant. ``0.0`` for a discrete UI event: the instant a modal/error/page renders IS when it is
      eligible to be judged, so the golden fire lands exactly on the annotated event time.
    - ``hold_s`` — how long the state persists after the event (the episode's own duration); the raw
      match window is ``[event_time, event_time + hold_s]`` before the coast grace is added.
    """

    condition_id: str
    condition: str
    goal_shape: str
    eligible_after_s: float
    hold_s: float


#: A dialog / modal appearing. An opened modal persists until dismissed -> a true ``episode`` with a
#: rising edge.
MODAL_APPEARED = ScreenCondition(
    condition_id="modal_appeared",
    condition="a modal dialog appeared",
    goal_shape="episode",
    eligible_after_s=0.0,
    hold_s=6.0,
)

#: An error surface (dialog / toast / banner). An error message stays on screen -> ``episode``.
ERROR_SHOWN = ScreenCondition(
    condition_id="error_shown",
    condition="an error message is shown",
    goal_shape="episode",
    eligible_after_s=0.0,
    hold_s=6.0,
)

#: A navigation / page change. An instantaneous transition between two page identities -> ``point``.
PAGE_CHANGED = ScreenCondition(
    condition_id="page_changed",
    condition="the page changed",
    goal_shape="point",
    eligible_after_s=0.0,
    hold_s=1.5,
)

#: Every condition, keyed by ``condition_id``.
SCREEN_CONDITIONS: dict[str, ScreenCondition] = {
    c.condition_id: c for c in (MODAL_APPEARED, ERROR_SHOWN, PAGE_CHANGED)
}

#: GUI-World event-type vocabulary -> condition. Authoritative when a keyframe declares an ``event``.
_EVENT_TYPE_TO_CONDITION: dict[str, ScreenCondition] = {
    # modal / dialog / popup
    "modal_open": MODAL_APPEARED,
    "modal": MODAL_APPEARED,
    "dialog_open": MODAL_APPEARED,
    "dialog": MODAL_APPEARED,
    "popup_open": MODAL_APPEARED,
    "popup": MODAL_APPEARED,
    "alert_open": MODAL_APPEARED,
    "confirm_dialog": MODAL_APPEARED,
    "prompt_dialog": MODAL_APPEARED,
    # error surfaces
    "error": ERROR_SHOWN,
    "error_dialog": ERROR_SHOWN,
    "error_toast": ERROR_SHOWN,
    "error_banner": ERROR_SHOWN,
    "toast_error": ERROR_SHOWN,
    "validation_error": ERROR_SHOWN,
    "exception": ERROR_SHOWN,
    # navigation / page change
    "navigate": PAGE_CHANGED,
    "navigation": PAGE_CHANGED,
    "page_change": PAGE_CHANGED,
    "page_load": PAGE_CHANGED,
    "url_change": PAGE_CHANGED,
    "route_change": PAGE_CHANGED,
    "tab_switch": PAGE_CHANGED,
    "screen_transition": PAGE_CHANGED,
}

#: Keyword fallback over a free-text caption, used ONLY when the event type is absent/unknown.
#: Ordered: the first condition whose any keyword is a substring of the lowercased caption wins.
#: ERROR precedes MODAL precedes PAGE so an "error dialog" caption reads as an error, not a bare
#: modal.
_CAPTION_KEYWORDS: tuple[tuple[ScreenCondition, tuple[str, ...]], ...] = (
    (ERROR_SHOWN, ("error", "failed", "failure", "invalid", "exception", "denied", "could not")),
    (MODAL_APPEARED, ("modal", "dialog", "pop-up", "popup", "confirmation", "alert", "overlay")),
    (PAGE_CHANGED, ("navigat", "page change", "redirect", "url ", "switches to", "loads the")),
)


def classify_event(event: str, caption: str = "") -> ScreenCondition | None:
    """Map one keyframe's ``event`` type (authoritative) or ``caption`` (fallback) to a condition.

    Returns ``None`` for a GUI event that is none of the three screen watch conditions (a scroll, a
    hover, a keystroke) — such a keyframe is dropped, never turned into an episode.
    """
    key = (event or "").strip().lower()
    if key in _EVENT_TYPE_TO_CONDITION:
        return _EVENT_TYPE_TO_CONDITION[key]
    text = (caption or "").lower()
    if text:
        for condition, keywords in _CAPTION_KEYWORDS:
            if any(kw in text for kw in keywords):
                return condition
    return None


# ---------------------------------------------------------------------------
# 2. Parsed input records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiWorldKeyframe:
    """One temporal GUI event annotation.

    - ``t`` — the annotated event time, in seconds, on the recording's own timeline.
    - ``event`` — the GUI-World event type (may be empty; then the caption classifies).
    - ``caption`` — the free-text keyframe caption.
    - ``content_key`` — an OPTIONAL content/document/page/dialog id. Screen identity keys on this,
      never on geometry. Absent -> the episode key falls back to its ordinal within the fixture.
    """

    t: float
    event: str = ""
    caption: str = ""
    content_key: str | None = None


@dataclass(frozen=True)
class GuiWorldSession:
    """One GUI-World recording (video) — the split unit."""

    video_id: str
    platform: str = ""
    keyframes: tuple[GuiWorldKeyframe, ...] = ()
    fps: float | None = None
    duration_s: float | None = None
    annotator: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError(f"GuiWorldSession.video_id must be a non-empty string, got {self.video_id!r}")
        if FIXTURE_SEP in self.video_id:
            raise ValueError(
                f"GuiWorldSession.video_id {self.video_id!r} may not contain {FIXTURE_SEP!r} — it is "
                "the fixture_id session/condition separator, so a session id carrying it would make "
                "the split unit unrecoverable."
            )


def _keyframe_time(raw: Mapping[str, Any], fps: float | None) -> float:
    """Resolve a keyframe's event time in seconds. Prefer ``t`` (seconds); else ``frame`` / fps."""
    if "t" in raw and raw["t"] is not None:
        return float(raw["t"])
    if "frame" in raw and raw["frame"] is not None:
        if not fps or fps <= 0:
            raise ValueError(
                "keyframe gives 'frame' but the session declares no positive 'fps' to resolve it "
                "to seconds"
            )
        return float(raw["frame"]) / float(fps)
    raise ValueError(f"keyframe has neither 't' (seconds) nor 'frame': {dict(raw)!r}")


def parse_session(raw: Mapping[str, Any]) -> GuiWorldSession:
    """Parse one GUI-World-representative record into a :class:`GuiWorldSession`."""
    fps = raw.get("fps")
    fps_val = float(fps) if fps is not None else None
    keyframes = tuple(
        GuiWorldKeyframe(
            t=_keyframe_time(kf, fps_val),
            event=str(kf.get("event", "")),
            caption=str(kf.get("caption", "")),
            content_key=(str(kf["content_key"]) if kf.get("content_key") is not None else None),
        )
        for kf in raw.get("keyframes", ())
    )
    return GuiWorldSession(
        video_id=str(raw["video_id"]),
        platform=str(raw.get("platform", "")),
        keyframes=keyframes,
        fps=fps_val,
        duration_s=(float(raw["duration_s"]) if raw.get("duration_s") is not None else None),
        annotator=str(raw.get("annotator", "")),
    )


def load_sessions(path: str | Path) -> tuple[GuiWorldSession, ...]:
    """Load a JSON array (or ``{"sessions": [...]}``) of GUI-World-representative records."""
    data = json.loads(Path(path).read_text() or "[]")
    if isinstance(data, Mapping):
        data = data.get("sessions", [])
    return tuple(parse_session(rec) for rec in data)


# ---------------------------------------------------------------------------
# 3. Session -> fixtures
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str | None) -> str:
    """A filesystem/key-safe slug of a content key; ``"kf"`` when there is no content key."""
    if not value:
        return "kf"
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "kf"


def session_of(fixture_id: str) -> str:
    """Recover the SESSION id (the split unit) from a ``fixture_id``."""
    return fixture_id.split(FIXTURE_SEP, 1)[0]


def condition_of(fixture_id: str) -> str:
    """Recover the ``condition_id`` from a ``fixture_id``."""
    parts = fixture_id.split(FIXTURE_SEP, 1)
    return parts[1] if len(parts) == 2 else ""


def session_to_fixtures(
    session: GuiWorldSession,
    *,
    allowed_lateness_s: float = SCREEN_COAST_INTERVAL_S,
) -> tuple[Fixture, ...]:
    """Transform one session into a fixture per ``(session, condition)`` with a mapped keyframe.

    Episodes within a fixture are ordered by event time (an ordered window, never a montage). Each
    episode's match window is widened by ``allowed_lateness_s`` (defaulting to the mandatory
    ``screen_document`` coast interval) so a coast-admitted fire still lands inside it.
    """
    grouped: dict[str, list[GuiWorldKeyframe]] = {}
    for kf in session.keyframes:
        condition = classify_event(kf.event, kf.caption)
        if condition is None:
            continue
        grouped.setdefault(condition.condition_id, []).append(kf)

    fixtures: list[Fixture] = []
    for condition_id in sorted(grouped):
        condition = SCREEN_CONDITIONS[condition_id]
        # Ordered window: sort by event time, then content-key slug for a deterministic tie-break.
        entries = sorted(grouped[condition_id], key=lambda kf: (kf.t, _slug(kf.content_key)))
        fixture_id = f"{session.video_id}{FIXTURE_SEP}{condition_id}"
        episodes: list[GroundTruthEpisode] = []
        for index, kf in enumerate(entries):
            ground_truth = GroundTruth.by_construction(
                event_start_ts=kf.t,
                eligible_after_s=condition.eligible_after_s,
                fixture_id=f"guiworld:{session.video_id}",
                annotation_version=GUIWORLD_ANNOTATION_VERSION,
            )
            # Content-keyed episode id (never geometry); the ordinal only guarantees uniqueness when
            # a content key recurs within the same fixture.
            episode_id = f"{fixture_id}::{_slug(kf.content_key)}::{index:02d}"
            episodes.append(
                GroundTruthEpisode(
                    episode_id=episode_id,
                    ground_truth=ground_truth,
                    episode_end_ts=kf.t + condition.hold_s,
                    allowed_lateness_s=allowed_lateness_s,
                )
            )
        fixtures.append(
            Fixture(
                fixture_id=fixture_id,
                goal_shape=condition.goal_shape,
                episodes=tuple(episodes),
                scope=scope_for_goal_shape(condition.goal_shape),
            )
        )
    return tuple(fixtures)


def sessions_to_fixtures(
    sessions: Sequence[GuiWorldSession],
    *,
    allowed_lateness_s: float = SCREEN_COAST_INTERVAL_S,
) -> tuple[Fixture, ...]:
    """Transform every session and return all fixtures, sorted by ``fixture_id``."""
    fixtures: list[Fixture] = []
    for session in sessions:
        fixtures.extend(session_to_fixtures(session, allowed_lateness_s=allowed_lateness_s))
    return tuple(sorted(fixtures, key=lambda f: f.fixture_id))


# ---------------------------------------------------------------------------
# 4. Serialization to the on-disk fixture layout percept.eval.load_fixtures reads
# ---------------------------------------------------------------------------


def fixture_to_ground_truth_dict(fixture: Fixture) -> dict[str, Any]:
    """Render a :class:`~percept.eval.differential.Fixture` to a ``ground_truth.json`` payload.

    Exactly the keys :func:`percept.eval.differential.load_fixtures` reads back
    (``goal_shape`` + a list of episode windows).
    """
    return {
        "goal_shape": fixture.goal_shape,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "event_start_ts": episode.ground_truth.event_start_ts,
                "decision_eligible_ts": episode.ground_truth.decision_eligible_ts,
                "episode_end_ts": episode.episode_end_ts,
                "allowed_lateness_s": episode.allowed_lateness_s,
                "annotation_version": episode.ground_truth.annotation_version,
                "annotator_or_fixture_id": episode.ground_truth.annotator_or_fixture_id,
            }
            for episode in fixture.episodes
        ],
    }


def write_fixture_tree(
    fixtures: Sequence[Fixture],
    out_dir: str | Path,
    *,
    splits: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    """Write ``<out_dir>/<fixture_id>/ground_truth.json`` for each fixture (and ``splits.json``).

    Returns ``out_dir``. Deterministic: the JSON is written sorted-key so two builds over the same
    fixtures are byte-for-byte identical.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        fixture_dir = root / fixture.fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        (fixture_dir / "ground_truth.json").write_text(
            json.dumps(fixture_to_ground_truth_dict(fixture), indent=2, sort_keys=True) + "\n"
        )
    if splits is not None:
        payload = {name: sorted(splits.get(name, [])) for name in ("train", "dev", "test")}
        (root / "splits.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return root
