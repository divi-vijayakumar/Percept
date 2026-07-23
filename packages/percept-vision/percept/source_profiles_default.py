"""The shipped :class:`~percept.sourceprofile.SourceProfile` set (ADR-005, extended with a tuned
localized admission default for screen_document).

Three profiles ship. Two are stable; one is explicitly EXPERIMENTAL and says so in its own
``stability`` field, not only in prose:

- :data:`PHYSICAL_SCENE` — the source class percept's stock tier-0 eye was calibrated on. Cameras.
  Admission is the measured admission stack (localized ``block_max`` salience over the 48x36 gate frame).
- :data:`SCREEN_DOCUMENT` — screen recordings, document/page content. TUNED @1: the SAME localized
  ``block_max``/0.20 signal :data:`PHYSICAL_SCENE` carries — never a
  lowered cutoff — is the declared default admission for screen content large enough to register on
  the 48x36 gate frame (a highlighted cell, a popup, a UI element). A fixed low-rate coast
  (1.0-2.0s) remains the floor UNDERNEATH it: the measured evidence below shows a full-PAGE,
  text-scale edit still dilutes below every calibrated cutoff, so the coast is what keeps that
  harder residual case observed, not the tuned default admitting it.
- :data:`CONVERSATION` — EXPERIMENTAL. Speaker identity is off-contract and goals judge serially
  (~24s/turn measured), so this profile is for evaluation, not production.

Why SCREEN_DOCUMENT still carries a coast floor, in numbers
-------------------------------------------------------------

``percept-doctor`` on the real client fixture ``probes/01-screen-walkthrough/data/clean.mp4``
— a 36s accounting screen recording, 180 frames, 1920x1080, containing a HARD CUT between two
different documents:

===========  ========  ========  ==================  ========
stat         max       median    its own threshold   admitted
===========  ========  ========  ==================  ========
mean         0.0163    0.0000    0.100               0
block_max     0.0919    0.0001    0.200               0
p95          0.0743    0.0000    0.150               0
===========  ========  ========  ==================  ========

No statistic admits a single frame of THIS fixture, including across the document cut. ``block_max``
amplifies the localized signal 5.6x over the ``mean`` exactly as designed — and still lands at under
HALF its own calibrated cutoff for a full-page, text-scale edit. The cause is resolution, not
statistic selection: the gate frame is 48x36 greyscale (``percept/runtime/source.py``,
``GATE_W, GATE_H = 48, 36``) and ``block_max`` reads an 8x6 grid over it, so one block is ~6x6 px of
a downsampled 1080p screen and a document-sized edit is diluted away.

This is why the coast is not optional even once ``block_max`` is the declared default: a
LOCALIZED change (a cell, a highlighted region, a modal — enough of the 8x6 grid lights up) clears
0.20 the same way a physical-scene event does, but a page-scale text edit does not, and percept does
not know in advance which one a client's content will produce. Lowering the cutoff to force the
harder case to admit would wreck physical-scene precision, which is calibrated on the SAME cutoffs —
so this profile does not lower one; it keeps ``block_max``/0.20 as the tuned default and leans on the
coast for what that eye cannot resolve. A client whose content is dominated by full-page edits should
still register their own page-change / phash / OCR signal — see ``docs/how-to/tune-admission.md`` —
and A/B it with ``percept eval`` on THEIR fixtures against THEIR bars.

Importing this module seeds :data:`percept.sourceprofile.DEFAULT_SOURCE_REGISTRY` (see :func:`seed`,
called at the bottom). Tests wanting isolation should ``seed(SourceProfileRegistry())`` instead.
"""

from __future__ import annotations

from .sourceprofile import (
    DEFAULT_SOURCE_REGISTRY,
    SourceProfile,
    SourceProfileRegistry,
)

#: The measured doctor readout that justifies :data:`SCREEN_DOCUMENT` being untuned. Quoted verbatim
#: by ADR-005 and docs/how-to/tune-admission.md so the three cannot drift apart.
SCREEN_EVIDENCE = (
    "Measured with percept-doctor on probes/01-screen-walkthrough/data/clean.mp4 "
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

#: Provenance for the physical-scene admission stack (the same eval WatchProfile's defaults cite).
PHYSICAL_PROVENANCE = (
    "Provenance: viability eval, charades-e2e ablation ladder — localized block_max salience @ "
    "0.20 nearly doubled recall over the whole-frame mean baseline. This is the ONE source class "
    "percept's stock tier-0 eye is calibrated on; see percept/profiles_default.py for the matching "
    "WatchProfile knobs."
)


# ---------------------------------------------------------------------------
# physical_scene — cameras. The calibrated case.
# ---------------------------------------------------------------------------

#: PHYSICAL_SCENE's conservative default_episode_policy capability — the SAME
#: "profile declares, concern selects" mechanism SCREEN_DOCUMENT uses
#: (``SourceProfile.default_episode_policy``), but a DIFFERENT one of its two
#: independent knobs: single-focus ATTACH, never unknown-to-unknown temporal continuity.
#:
#: ``unknown_continuity_gap_s`` stays UNSET (``None``) here on purpose — that capability is
#: SCREEN-ONLY (see SCREEN_DOCUMENT / SCREEN_UNKNOWN_CONTINUITY_GAP_S above): a physical scene
#: routinely holds several distinct, fast-moving unidentified subjects (two strangers crossing a
#: doorway a second apart are not "probably the same thing seen again" the way two illegible page
#: samples are), so collapsing unknown sightings by elapsed time alone would silently merge them.
#:
#: ``unknown_attaches_to_active_bound=True`` is the conservative knob that DOES apply: it is
#: self-limiting by construction (``EpisodePolicy`` docstring, ``percept/episodes.py``) — it only
#: ever continues a sighting when the source has EXACTLY ONE open bound episode, and stays unbound
#: the moment a second one opens. That is precisely the common single-subject watch pattern these
#: demos ship (a doorway, a workbench, one inspected part): a brief occlusion of the ONE tracked
#: subject continues its episode instead of minting a rival, while a multi-subject scene is
#: completely unaffected (two-or-more open episodes -> the flag is a no-op, sightings stay unbound).
PHYSICAL_UNKNOWN_ATTACHES_TO_ACTIVE_BOUND = True

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
        # See PHYSICAL_UNKNOWN_ATTACHES_TO_ACTIVE_BOUND's docstring above for why this is the
        # conservative choice and unknown_continuity_gap_s is deliberately absent.
        "unknown_attaches_to_active_bound": PHYSICAL_UNKNOWN_ATTACHES_TO_ACTIVE_BOUND,
    },
    stability="stable",
    notes=(
        "Cameras: fixed, handheld or wearable. Motion on the 48x36 tier-0 gate frame is a REAL "
        "admission signal for this source class. " + PHYSICAL_PROVENANCE + " Default episode policy: "
        "a brief occlusion of the sole tracked subject (unknown_attaches_to_active_bound=True) "
        "continues its episode rather than minting a rival, self-limited to scenes with exactly one "
        "open bound episode; unknown-to-unknown temporal continuity stays OFF (screen_document-only) "
        "because a physical scene routinely holds multiple distinct unidentified subjects."
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

#: The default unknown-to-unknown continuity gap for :data:`SCREEN_DOCUMENT`, in seconds
#: (``percept.episodes.EpisodePolicy.unknown_continuity_gap_s``). A FLOOR pick, not a tuned value: a
#: document whose content key is unreadable this coast tick and unreadable the NEXT coast tick
#: (:data:`SCREEN_COAST_INTERVAL_S` apart) is, on a slow-changing screen source, far more likely to be
#: the same still-illegible page than a brand-new one — so it is defensible to fold consecutive
#: illegible sightings into one provisional subject rather than minting a fresh one every coast tick.
#: Set to a few coast intervals so a couple of missed samples do not manufacture a new subject; a
#: gap this small still splits genuinely distinct occurrences on any real page-to-page cadence.
SCREEN_UNKNOWN_CONTINUITY_GAP_S = 3.0 * SCREEN_COAST_INTERVAL_S

SCREEN_DOCUMENT = SourceProfile(
    name="screen_document",
    modalities=frozenset({"video", "text"}),
    temporal_unit="page",
    # TUNED @1: the same localized tier-0 salience stack physical_scene ships, applied to
    # screen content. parameters["change_signal"] names a REAL signal (never UNTUNED) at the SAME
    # cutoff physical_scene carries — see the module docstring for the scope this does and does not
    # cover, and why the coast below stays mandatory alongside it.
    admission_strategy="tier0_localized_salience_or_coast",
    # Ordered frame WINDOW, never a sprite montage (ADR-008 Corrections(a): an ordered frame window
    # beat a sprite montage unconditionally in the dynamic-goal viability evaluation). parameters
    # ["plan_mode"] states this explicitly so a caller reading this profile's parameters sees the
    # same rule the resolver enforces globally.
    evidence_strategy="screenshot_plus_changed_region",
    # NEVER geometry alone. IoU+label cannot separate INV-2291 from INV-2294: two invoices rendered
    # in the same screen region carrying the same label are the SAME entity by construction under a
    # geometric identity strategy. Screen identity must key on CONTENT (document id / page key).
    identity_strategy="content_key",
    default_timebase="media",
    parameters={
        # The tuned default: the SAME statistic + cutoff physical_scene carries (never lowered to
        # force an admit — see the module docstring's "still carries a coast floor" section).
        "change_signal": "tier0_block_max_salience",
        "salience_stat": "block_max",
        "salience_threshold": 0.20,
        # A localized change large enough to clear 0.20 escalates on its own; the coast below is the
        # floor UNDERNEATH that tuned default, not a replacement for it (SCREEN_COAST_INTERVAL_S is
        # in [1.0, 2.0]s — a fail-safe FLOOR pick, not a tuned value, see its own docstring above).
        "coast_interval_s": SCREEN_COAST_INTERVAL_S,
        # This profile's own assertion that its coast floor is not optional. It is READ, not inert:
        # the resolver (percept.resolver) and percept.doctor consult it so that a strict_admission
        # ExecutionPolicy (Settings.edge.coast_enabled=False) which unbinds the coast reports the
        # override LOUDLY — a receipt warning coded coast_required_overridden, and a doctor finding
        # when such a run then admits zero frames — rather than silently dropping the floor.
        "coast_required": True,
        "episode_unit": "page_interval",
        "evidence_parts": ("full_screenshot", "changed_region_crop", "previous_screenshot"),
        "evidence_optional_parts": ("previous_screenshot",),
        "plan_mode": "window",
        "identity_key_hint": "document_or_page_key",
        "measured_evidence": SCREEN_EVIDENCE,
        # An unreadable content key on this source defaults to TEMPORAL-PROXIMITY continuity rather
        # than minting a fresh provisional subject every tick — see SCREEN_UNKNOWN_CONTINUITY_GAP_S
        # and SourceProfile.default_episode_policy. A per-profile capability, not a global: a source
        # with many distinct fast-moving unidentified subjects (physical_scene) does not set this.
        "unknown_continuity_gap_s": SCREEN_UNKNOWN_CONTINUITY_GAP_S,
    },
    stability="stable",
    notes=(
        "Screen recordings, document and page content. TUNED @1: the declared default "
        "admission is localized block_max salience at 0.20 — the SAME cutoff physical_scene carries, "
        "never lowered — which clears on a screen change large enough to register on the 48x36 gate "
        "frame (a highlighted cell, a popup, a UI element). A fixed low-rate coast remains mandatory "
        "underneath it: a full-PAGE, text-scale edit still dilutes below every calibrated cutoff (see "
        "the measured evidence below), so the coast — not the tuned admission — is what keeps that "
        "harder residual case observed. A client whose content is dominated by page-scale edits "
        "should still register their own page-change / phash / OCR signal (register_detector) and "
        "A/B it with `percept eval` — see docs/how-to/tune-admission.md. An unreadable content key "
        "defaults to temporal-proximity unknown-continuity (SCREEN_UNKNOWN_CONTINUITY_GAP_S) rather "
        "than a fresh provisional subject per tick. " + SCREEN_EVIDENCE
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
    "PHYSICAL_UNKNOWN_ATTACHES_TO_ACTIVE_BOUND",
    "PHYSICAL_SCENE",
    "SCREEN_DOCUMENT",
    "CONVERSATION",
    "ALL_SOURCE_PROFILES",
    "DEFAULT_SOURCE_PROFILE_NAME",
    "seed",
]
