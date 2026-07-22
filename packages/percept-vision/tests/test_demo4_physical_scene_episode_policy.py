"""Card DEMO4 (Finding 3a) — PHYSICAL_SCENE gets a real, non-default ``default_episode_policy`` — the same
"profile declares, concern selects" mechanism ``SCREEN_DOCUMENT`` uses
(``SourceProfile.default_episode_policy``, ``percept/sourceprofile.py``, the Y17 pattern
``test_c2f_source_profile_defaults.py`` exercises for screen_document) — but a DIFFERENT one of the
policy's two independent knobs:

- ``unknown_attaches_to_active_bound=True`` — the conservative, SELF-LIMITING knob: it only ever
  continues a sighting when the source holds exactly ONE open bound episode (two open bound episodes
  -> refused, stays unbound), so a brief occlusion of the sole tracked subject in a single-focus watch
  (a doorway, a workbench, one inspected part) continues its episode instead of minting a rival, while
  a genuinely multi-subject scene is completely unaffected.
- ``unknown_continuity_gap_s`` stays UNSET (``None``) — that unknown-to-unknown temporal-proximity
  capability stays SCREEN-ONLY: a physical scene routinely holds several distinct, fast-moving
  unidentified subjects, so collapsing them by elapsed time alone would silently merge different
  people/objects.

These tests protect both halves of that declaration and prove the self-limiting attach behaviorally
(not just as a policy-value assertion), mirroring ``test_c2f_source_profile_defaults.py``'s shape.
"""

from __future__ import annotations

import unittest

from percept.contracts import SubjectRef
from percept.episodes import EpisodeLog, EpisodePolicy
from percept.source_profiles_default import (
    PHYSICAL_SCENE,
    PHYSICAL_UNKNOWN_ATTACHES_TO_ACTIVE_BOUND,
    SCREEN_DOCUMENT,
)


def _ref(key: "str | None", *, resolution: str) -> SubjectRef:
    return SubjectRef(
        scheme="track", namespace="camera", key=key, resolution=resolution,
        scope="session", strategy="test", confidence=0.9,
    )


class _Exemplar:
    """A minimal stand-in for a judged observation — episodes.py only ever reads ``.subject``."""

    def __init__(self, subject: "SubjectRef | None") -> None:
        self.subject = subject


# ----------------------------------------------------------------------------------------------
# 1. the declared policy VALUE — conservative, and distinct from both EpisodePolicy() and
#    SCREEN_DOCUMENT's own default.
# ----------------------------------------------------------------------------------------------


class TestPhysicalSceneDefaultEpisodePolicyValue(unittest.TestCase):
    def test_single_focus_attach_is_declared_on(self):
        pol = PHYSICAL_SCENE.default_episode_policy
        self.assertTrue(pol.unknown_attaches_to_active_bound)
        self.assertEqual(pol.unknown_attaches_to_active_bound, PHYSICAL_UNKNOWN_ATTACHES_TO_ACTIVE_BOUND)

    def test_unknown_to_unknown_temporal_continuity_stays_screen_only(self):
        # NO unknown_continuity_gap_s by default for physical_scene -- that capability stays
        # screen_document-only (a physical scene can hold many distinct unidentified subjects).
        pol = PHYSICAL_SCENE.default_episode_policy
        self.assertIsNone(pol.unknown_continuity_gap_s)

    def test_differs_from_the_byte_for_byte_default(self):
        # A profile that declares NEITHER parameter gets EpisodePolicy() verbatim (the documented
        # no-capability default). physical_scene now declares one -- this is an intentional,
        # non-trivial choice, not an accident of construction.
        self.assertNotEqual(PHYSICAL_SCENE.default_episode_policy, EpisodePolicy())

    def test_differs_from_screen_documents_default(self):
        # The two profiles pick DIFFERENT capabilities of the same mechanism -- proving neither is
        # copy-pasted from the other.
        physical = PHYSICAL_SCENE.default_episode_policy
        screen = SCREEN_DOCUMENT.default_episode_policy
        self.assertNotEqual(physical, screen)
        self.assertTrue(physical.unknown_attaches_to_active_bound)
        self.assertFalse(screen.unknown_attaches_to_active_bound)
        self.assertIsNone(physical.unknown_continuity_gap_s)
        self.assertIsNotNone(screen.unknown_continuity_gap_s)


# ----------------------------------------------------------------------------------------------
# 2. behavioral proof: a brief occlusion of the SOLE tracked subject continues its episode.
# ----------------------------------------------------------------------------------------------


class TestSingleFocusAttachBehavior(unittest.TestCase):
    def test_unknown_sighting_attaches_to_the_sole_open_bound_episode(self):
        log = EpisodeLog("g1", policy=PHYSICAL_SCENE.default_episode_policy)
        bound = _Exemplar(_ref("person-1", resolution="bound"))
        episode, is_new = log.observe(0.0, bound)
        self.assertTrue(is_new)
        # a brief occlusion: the same subject is seen but not re-identified a moment later
        unknown = _Exemplar(_ref(None, resolution="unknown"))
        continued_episode, is_new2 = log.observe(0.5, unknown)
        self.assertFalse(is_new2, "the sole open bound episode must continue, not mint a rival")
        self.assertEqual(continued_episode.id, episode.id)
        self.assertEqual(log.count(), 1)

    def test_two_open_bound_episodes_refuse_the_attach(self):
        # Self-limiting: with MORE THAN ONE thing in focus, an unknown sighting could belong to
        # either, so it stays unbound (mints its own fresh episode) -- exactly the multi-subject
        # safety net the policy's docstring promises.
        log = EpisodeLog("g1", policy=PHYSICAL_SCENE.default_episode_policy)
        log.observe(0.0, _Exemplar(_ref("person-1", resolution="bound")))
        log.observe(0.0, _Exemplar(_ref("person-2", resolution="bound")))
        self.assertEqual(log.count(), 2)
        _episode, is_new = log.observe(0.5, _Exemplar(_ref(None, resolution="unknown")))
        self.assertTrue(is_new, "two open bound episodes must refuse the single-focus attach")
        self.assertEqual(log.count(), 3)

    def test_without_the_profile_default_two_unknowns_never_collapse(self):
        # Contrast: the byte-for-byte EpisodePolicy() (no profile, or a profile that declares
        # nothing) never attaches -- every unidentified sighting mints its own episode.
        log = EpisodeLog("g1", policy=EpisodePolicy())
        log.observe(0.0, _Exemplar(_ref("person-1", resolution="bound")))
        _episode, is_new = log.observe(0.5, _Exemplar(_ref(None, resolution="unknown")))
        self.assertTrue(is_new)
        self.assertEqual(log.count(), 2)

    def test_two_distinct_unknowns_still_never_collapse_under_the_physical_default(self):
        # unknown_continuity_gap_s is unset for physical_scene, so even with NO bound episode open,
        # two back-to-back unidentified sightings must NOT collapse into one (unlike
        # screen_document's SCREEN_UNKNOWN_CONTINUITY_GAP_S behavior).
        log = EpisodeLog("g1", policy=PHYSICAL_SCENE.default_episode_policy)
        log.observe(0.0, _Exemplar(_ref(None, resolution="unknown")))
        _episode, is_new = log.observe(0.5, _Exemplar(_ref(None, resolution="unknown")))
        self.assertTrue(is_new, "two distinct unknowns must never collapse for physical_scene")
        self.assertEqual(log.count(), 2)


if __name__ == "__main__":
    unittest.main()
