"""The shipped :class:`~percept.sourceprofile.SourceProfile` set (card D7, ADR-005).

Three profiles ship. Two are stable; one is explicitly EXPERIMENTAL and says so in its own
``stability`` field, not only in prose:

- :data:`PHYSICAL_SCENE` — the source class percept's stock tier-0 eye was calibrated on. Cameras.
  Admission is the measured H0 stack (localized ``block_max`` salience over the 48x36 gate frame).
- :data:`SCREEN_DOCUMENT` — screen recordings, document/page content. Percept ships NO calibrated
  admission signal for this class and this profile DECLARES that
  (``parameters["change_signal"] == UNTUNED``). Its honest floor is a fixed low-rate coast.
- :data:`CONVERSATION` — EXPERIMENTAL. Speaker identity is off-contract and goals judge serially
  (~24s/turn measured), so this profile is for evaluation, not production.

Why SCREEN_DOCUMENT is untuned, in numbers (the reason this card exists)
-----------------------------------------------------------------------

``percept-doctor`` (card D1) on the real client fixture ``probes/01-screen-walkthrough/data/clean.mp4``
— a 36s accounting screen recording, 180 frames, 1920x1080, containing a HARD CUT between two
different documents:

===========  ========  ========  ==================  ========
stat         max       median    its own threshold   admitted
===========  ========  ========  ==================  ========
mean         0.0163    0.0000    0.100               0
block_max     0.0919    0.0001    0.200               0
p95          0.0743    0.0000    0.150               0
===========  ========  ========  ==================  ========

No statistic admits a single frame, including across the document cut. ``block_max`` amplifies the
localized signal 5.6x over the ``mean`` exactly as designed — and still lands at under HALF its own
calibrated cutoff. The cause is resolution, not statistic selection: the gate frame is 48x36
greyscale (``percept/runtime/source.py``, ``GATE_W, GATE_H = 48, 36``) and ``block_max`` reads an 8x6
grid over it, so one block is ~6x6 px of a downsampled 1080p screen and a document-sized edit is
diluted away.

Lowering a threshold to make this fixture admit would wreck physical-scene precision, which is
calibrated on the SAME cutoffs. So this profile does not lower one. It declares itself untuned, coasts
at a fixed low rate so a static-but-important page is still sampled, and points at
``docs/how-to/tune-admission.md`` — where the CLIENT registers their own page-change / phash / OCR
signal and A/Bs it with ``percept eval`` on THEIR fixtures against THEIR bars.

Importing this module seeds :data:`percept.sourceprofile.DEFAULT_SOURCE_REGISTRY` (see :func:`seed`,
called at the bottom). Tests wanting isolation should ``seed(SourceProfileRegistry())`` instead.
"""

from __future__ import annotations

from .sourceprofile import (
    DEFAULT_SOURCE_REGISTRY,
    UNTUNED,
    SourceProfile,
    SourceProfileRegistry,
)

#: The measured doctor readout that justifies :data:`SCREEN_DOCUMENT` being untuned. Quoted verbatim
#: by ADR-005 and docs/how-to/tune-admission.md so the three cannot drift apart.
SCREEN_EVIDENCE = (
    "Measured with percept-doctor (card D1) on probes/01-screen-walkthrough/data/clean.mp4 "
    "(36s accounting screen recording, 180 frames, 1920x1080, containing a hard cut between two "
    "different documents): mean max 0.0163 vs threshold 0.100 -> 0 admitted; block_max max 0.0919 "
    "vs threshold 0.200 -> 0 admitted; p95 max 0.0743 vs threshold 0.150 -> 0 admitted. block_max "
    "amplifies the localized signal 5.6x over the mean as designed and still lands under HALF its "
    "own cutoff. Cause: the tier-0 gate frame is 48x36 greyscale (runtime/source.py) and block_max "
    "reads an 8x6 grid over it, so one block is ~6x6 px of a downsampled 1080p screen — a "
    "document-sized edit is diluted away. This is an under-resolved EYE, not a mis-selected "
    "statistic, and it is NOT fixed by lowering a threshold (the same cutoffs carry physical-scene "
    "precision)."
)

#: H0 provenance for the physical-scene admission stack (the same eval WatchProfile's defaults cite).
PHYSICAL_PROVENANCE = (
    "Provenance: H0 viability eval, charades-e2e ablation ladder — localized block_max salience @ "
    "0.20 nearly doubled recall over the whole-frame mean baseline. This is the ONE source class "
    "percept's stock tier-0 eye is calibrated on; see percept/profiles_default.py for the matching "
    "WatchProfile knobs."
)


# ---------------------------------------------------------------------------
# physical_scene — cameras. The calibrated case.
# ---------------------------------------------------------------------------

PHYSICAL_SCENE = SourceProfile(
    name="physical_scene",
    modalities=frozenset({"video"}),
    temporal_unit="frame",
    admission_strategy="tier0_motion_salience",
    evidence_strategy="frame_window",
    identity_strategy="track_iou_label",
    default_timebase="auto",
    parameters={
        "salience_stat": "block_max",
        # Not a new knob: the cutoff percept.config.EdgeSettings already carries, restated here so a
        # reader can see WHICH admission the strategy name means. The runtime keeps reading it from
        # EdgeSettings / the WatchProfile, never from this dict.
        "salience_threshold": 0.20,
        "coast_interval_s": None,   # motion IS the signal here; no coast needed
        "episode_unit": "activity_interval",
        "change_signal": "tier0_motion_salience",
    },
    stability="stable",
    notes=(
        "Cameras: fixed, handheld or wearable. Motion on the 48x36 tier-0 gate frame is a REAL "
        "admission signal for this source class. " + PHYSICAL_PROVENANCE
    ),
)


# ---------------------------------------------------------------------------
# screen_document — screen recordings / document + page content. The untuned case, declared.
# ---------------------------------------------------------------------------

#: The fixed low-rate coast for :data:`SCREEN_DOCUMENT`, in seconds. This is a FLOOR chosen to FAIL
#: SAFE, NOT a tuned value and not a claim about any client's content: it guarantees a
#: static-but-important page is still sampled when no change signal fires. It is deliberately slow
#: enough to be affordable on a long session and fast enough that a page cannot live and die entirely
#: between two samples in a human-paced walkthrough. Tune it on YOUR content
#: (docs/how-to/tune-admission.md); the coast is NOT optional and must not be set to None.
SCREEN_COAST_INTERVAL_S = 2.0

SCREEN_DOCUMENT = SourceProfile(
    name="screen_document",
    modalities=frozenset({"video", "text"}),
    temporal_unit="page",
    # The strategy NAMES what should decide admission; parameters["change_signal"] == UNTUNED says
    # percept ships no calibrated implementation of it. Doctor reads exactly that.
    admission_strategy="page_change_signal_or_coast",
    evidence_strategy="screenshot_plus_changed_region",
    # NEVER geometry alone. IoU+label cannot separate INV-2291 from INV-2294: two invoices rendered
    # in the same screen region carrying the same label are the SAME entity by construction under a
    # geometric identity strategy. Screen identity must key on CONTENT (document id / page key).
    identity_strategy="content_key",
    default_timebase="media",
    parameters={
        "change_signal": UNTUNED,
        "coast_interval_s": SCREEN_COAST_INTERVAL_S,
        "coast_required": True,
        "episode_unit": "page_interval",
        "evidence_parts": ("full_screenshot", "changed_region_crop", "previous_screenshot"),
        "evidence_optional_parts": ("previous_screenshot",),
        "identity_key_hint": "document_or_page_key",
        "measured_evidence": SCREEN_EVIDENCE,
    },
    stability="stable",
    notes=(
        "Screen recordings, document and page content. percept's STOCK EYE DOES NOT ADMIT ON THIS "
        "CLASS OF CONTENT and this profile does not pretend otherwise: the change signal is declared "
        "UNTUNED and the honest floor is a fixed low-rate coast, so a static-but-important page is "
        "still sampled. The client supplies and tunes the admission signal (their phash / "
        "page-change detector / OCR model, registered with register_detector) — see "
        "docs/how-to/tune-admission.md. " + SCREEN_EVIDENCE
    ),
)


# ---------------------------------------------------------------------------
# conversation — EXPERIMENTAL ONLY.
# ---------------------------------------------------------------------------

CONVERSATION = SourceProfile(
    name="conversation",
    modalities=frozenset({"audio", "text"}),
    temporal_unit="turn",
    admission_strategy="utterance_boundary",
    evidence_strategy="turn_transcript_window",
    identity_strategy="speaker_label",
    default_timebase="wall",
    parameters={
        "change_signal": "utterance_boundary",
        "coast_interval_s": None,
        "episode_unit": "turn_interval",
        "experimental_reasons": (
            "speaker identity is OFF-CONTRACT: identity_strategy='speaker_label' has no "
            "diarization behind it in this build, so two speakers can collapse to one entity",
            "goals judge SERIALLY at ~24s/turn measured, so a multi-goal conversation watch falls "
            "behind a live conversation",
        ),
    },
    stability="experimental",
    notes=(
        "EXPERIMENTAL — evaluation only, do NOT build a product on this profile. Two known, "
        "measured problems: (1) speaker identity is off-contract — there is no diarization behind "
        "identity_strategy='speaker_label' in this build; (2) goals judge serially at ~24s/turn "
        "measured, so a live conversation outruns the watch. Shipped so the shape can be evaluated "
        "and so nobody has to guess whether percept 'supports conversation' — it does not, yet."
    ),
)


#: Every shipped profile. Extending percept means ADDING to the registry from your own package, not
#: editing this tuple (see percept.sourceprofile.register_source_profile / the
#: ``percept.source_profiles`` entry-point group).
ALL_SOURCE_PROFILES: tuple[SourceProfile, ...] = (
    PHYSICAL_SCENE,
    SCREEN_DOCUMENT,
    CONVERSATION,
)

#: What doctor / the runtime assume when the caller names no source profile. physical_scene is the
#: honest default because it is the one class the stock eye is calibrated on — a client on screen
#: content is running the WRONG profile until they say so, and doctor tells them.
DEFAULT_SOURCE_PROFILE_NAME = "physical_scene"


def seed(registry: SourceProfileRegistry = DEFAULT_SOURCE_REGISTRY, *, overwrite: bool = False) -> None:
    """Register every profile in :data:`ALL_SOURCE_PROFILES` onto ``registry``. Idempotent."""
    for profile in ALL_SOURCE_PROFILES:
        registry.register(profile, overwrite=overwrite)


seed()


__all__ = [
    "SCREEN_EVIDENCE",
    "PHYSICAL_PROVENANCE",
    "SCREEN_COAST_INTERVAL_S",
    "PHYSICAL_SCENE",
    "SCREEN_DOCUMENT",
    "CONVERSATION",
    "ALL_SOURCE_PROFILES",
    "DEFAULT_SOURCE_PROFILE_NAME",
    "seed",
]
