"""— an open episode is KEYED into the verdict cache instead of switching the cache off.

The hole this closes. ``Percept._judge`` serialises a concern's open episode into the judge
``context``, and the first remedy for the state that creates was to bypass the verdict cache whenever
a context was present. The reasoning was right — "a verdict formed under one episode state must never
be re-served under another" — and the remedy was backwards: an open episode is the state a watch
spends nearly all of its ticks in, so bypassing on its presence disabled the cache exactly when it was
most useful. Measured on 180s of real footage: one continuous occurrence, 187 judged frames, 187 judge
calls, zero cache hits. The better episodes worked, the more completely the cache was off.

The codebase had already answered this once. A re-armed ``condition`` is also state the frame hash
cannot see, and the answer there was to put it in the entry SLOT (``cache/framehash.py``), not to stop
caching. ``CacheKey.context_fingerprint`` does the same one level up, over an opaque projection of the
answer-affecting context state minted by ``percept.episode_memory``.

What this suite pins:

1. **the projection is total and classified** — every field of the judge projection is declared either
   answer-affecting or answer-neutral, so a field added later cannot silently escape the key
   (``TestTheProjectionIsTotal``);
2. **the encoding is injective** — equal fingerprints mean equal inputs, which is what carries "same
   key implies same state" (``TestTheEncodingIsInjective``);
3. **the decisive property: stability over a continuous true period** — one ``opened`` revision,
   heartbeats coalescing, no annotation: exactly ONE fingerprint value, however long it runs
   (``TestStabilityOverAContinuousPeriod``);
4. **it moves on a genuine state transition** — and specifically on the transitions a revision COUNT
   would miss (``TestGenuineTransitionsChangeIt``);
5. **the cache layer keys on it and invalidation is unaffected** (``TestTheLayerKeysOnIt``);
6. **end to end** — a camera watch inside one open occurrence serves from the cache, and does not
   fragment the episode, duplicate the fire, or lose the close (``TestEndToEndInsideAnOpenEpisode``);
7. **the confirmation-count feedback loop is closed** — a cheap serve advances ``observations`` but
   NOT the count the next prompt quotes (``TestTheConfirmationCountLoop``).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.cache.framehash import CacheKey, FrameHashLayer
from percept.config import CacheSettings, GateSettings, Settings
from percept.episode_memory import (
    ANSWER_NEUTRAL_EPISODE_FIELDS,
    ANSWER_NEUTRAL_REVISION_FIELDS,
    FINGERPRINTED_EPISODE_FIELDS,
    FINGERPRINTED_REVISION_FIELDS,
    _framed,
    _opt,
    episode_context_and_fingerprint,
)
from percept.episodes import EpisodeLog, EpisodePolicy
from percept.identity_policy import SCENE_IDENTITY
from percept.source_profiles_default import PHYSICAL_SCENE

_YES = Verdict(True, 0.95, "a person is at the door")
_NO = Verdict(False, 0.95, "the doorway is empty")


def _log(**policy_kw) -> EpisodeLog:
    """A scene-keyed log with a generous continuity gap — the shape a camera watch keeps."""
    return EpisodeLog(
        "g1", window_s=10_000.0, max_episodes=8,
        policy=EpisodePolicy(continuity_gap_s=1000.0, **policy_kw),
    )


def _fp(log: EpisodeLog, **kw) -> "str | None":
    return episode_context_and_fingerprint(log, **kw)[1]


def _text(log: EpisodeLog, **kw) -> "str | None":
    return episode_context_and_fingerprint(log, **kw)[0]


# --------------------------------------------------------------------------------------------------
# 1 — the projection is TOTAL: every field is classified, so none can escape the key by accident.
# --------------------------------------------------------------------------------------------------


class TestTheProjectionIsTotal(unittest.TestCase):
    def _judge_dict(self) -> dict:
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        ep = log.current_open()
        log.annotate(ep, value="a duffel bag", incident_id="inc-1")
        log.record_revision(ep, "updated", t=1.0, value="a duffel bag")
        return ep.to_judge_dict()

    def test_every_projected_field_is_classified_exactly_once(self):
        """The guard that keeps this honest as the projection grows.

        A field added to ``to_judge_dict`` and forgotten here would ride into the prompt WITHOUT
        riding into the cache key — the precise bug the fingerprint exists to prevent. Failing this
        test is the instruction to decide which column the new field belongs in.
        """
        keys = set(self._judge_dict())
        self.assertEqual(
            keys, FINGERPRINTED_EPISODE_FIELDS | ANSWER_NEUTRAL_EPISODE_FIELDS,
            "a judge-projection field is unclassified: put it in FINGERPRINTED_EPISODE_FIELDS if it "
            "can change the answer, ANSWER_NEUTRAL_EPISODE_FIELDS if it provably cannot",
        )
        self.assertFalse(FINGERPRINTED_EPISODE_FIELDS & ANSWER_NEUTRAL_EPISODE_FIELDS)

    def test_every_revision_row_field_is_classified_exactly_once(self):
        rows = self._judge_dict()["revisions"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(
                set(row), FINGERPRINTED_REVISION_FIELDS | ANSWER_NEUTRAL_REVISION_FIELDS
            )
        self.assertFalse(FINGERPRINTED_REVISION_FIELDS & ANSWER_NEUTRAL_REVISION_FIELDS)

    def test_nothing_open_and_no_static_context_leaves_the_key_untouched(self):
        """A goal with no episode gets ``None`` — the pre-existing cache key, byte for byte."""
        self.assertEqual(episode_context_and_fingerprint(None), (None, None))
        self.assertEqual(episode_context_and_fingerprint(_log()), (None, None))

    def test_a_bound_static_context_alone_still_fingerprints(self):
        """The projection is of the whole answer-affecting context, not only of the episode."""
        text, fp = episode_context_and_fingerprint(_log(), static_context="the lobby is unstaffed")
        self.assertIsNone(text)  # nothing open -> nothing to inject
        self.assertIsNotNone(fp)
        _, other = episode_context_and_fingerprint(_log(), static_context="the lobby is staffed")
        self.assertNotEqual(fp, other)

    def test_the_text_and_the_fingerprint_arrive_together(self):
        """One reading of the log produces both, so they can never describe different states."""
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        text, fp = episode_context_and_fingerprint(log)
        self.assertIsNotNone(text)
        self.assertIsNotNone(fp)
        self.assertIn("already-established episode", text)


# --------------------------------------------------------------------------------------------------
# 2 — the encoding is INJECTIVE. This is what "same key implies same state" rests on.
# --------------------------------------------------------------------------------------------------


class TestTheEncodingIsInjective(unittest.TestCase):
    def test_framing_is_reversible_so_no_two_field_lists_collide(self):
        """Length-prefixing makes the concatenation parseable back into its fields.

        Without it, a caller-owned ``value`` containing the separator could forge a different field
        list that encodes identically — an aliased key, i.e. a verdict served under state it was
        never formed under.
        """
        self.assertNotEqual(_framed(["ab", "c"]), _framed(["a", "bc"]))
        self.assertNotEqual(_framed(["a:b"]), _framed(["a", "b"]))
        self.assertNotEqual(_framed(["", "a"]), _framed(["a"]))

    def test_absence_never_encodes_as_a_present_value(self):
        self.assertNotEqual(_opt(None), _opt(""))
        self.assertNotEqual(_opt(None), _opt("0"))
        self.assertNotEqual(_opt("1"), _opt("11"))

    def test_a_value_cannot_forge_another_episodes_projection(self):
        """The end-to-end form of the two properties above, on real episodes."""
        a, b = _log(), _log()
        a.observe_decision(0.0, _YES, subject_key="cam:1")
        b.observe_decision(0.0, _YES, subject_key="cam:11")
        self.assertNotEqual(_fp(a), _fp(b))
        c, d = _log(), _log()
        c.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        d.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        c.annotate(c.current_open(), value="ab")
        d.annotate(d.current_open(), value="a", incident_id="b")
        self.assertNotEqual(_fp(c), _fp(d))


# --------------------------------------------------------------------------------------------------
# 3 — THE DECISIVE PROPERTY. A continuous true period must hold ONE fingerprint value.
# --------------------------------------------------------------------------------------------------


class TestStabilityOverAContinuousPeriod(unittest.TestCase):
    """If the fingerprint churns here, the projection is wrong and the cache is off again.

    The property is not a hope about model behaviour, it is a property of the log: ``confirmed`` is
    deliberately absent from ``STATE_CHANGING_REVISION_KINDS``, so a heartbeat coalesces in place —
    appending no timeline row and moving only ``last_seen`` / ``last_confirmed`` / the observation
    counts, every one of which is in the answer-neutral column.
    """

    def test_two_hundred_heartbeats_produce_exactly_one_fingerprint(self):
        log = _log()
        seen = set()
        for i in range(200):
            log.observe_decision(float(i), _YES, subject_key=SCENE_IDENTITY)
            seen.add(_fp(log))
        self.assertEqual(len(seen), 1, f"the fingerprint churned across a continuous period: {seen}")
        ep = log.current_open()
        self.assertEqual([r.kind for r in ep.revisions], ["opened"])  # heartbeats appended nothing
        self.assertEqual(ep.observations, 200)                        # while time really did pass

    def test_the_answer_neutral_fields_cannot_move_it(self):
        """Stated directly on the fields, not only through the fold that advances them."""
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        ep = log.current_open()
        before = _fp(log)
        ep.last_seen = 9_999.0
        ep.last_confirmed = 9_999.0
        ep.started_at = -1.0
        ep.observations = 10_000
        ep.judged_observations = 9_999
        self.assertEqual(_fp(log), before)

    def test_a_timeline_at_its_bound_still_holds_one_value(self):
        """The saturating case: ``max_revisions`` reached, then a long continuous period."""
        log = EpisodeLog(
            "g1", window_s=10_000.0, max_episodes=4,
            policy=EpisodePolicy(continuity_gap_s=1000.0), max_revisions=3,
        )
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        ep = log.current_open()
        for i in range(6):  # overflow the timeline so rows are being dropped
            log.record_revision(ep, "updated", t=float(i), value=f"v{i}")
        seen = set()
        for i in range(50):
            log.observe_decision(10.0 + i, _YES, subject_key=SCENE_IDENTITY)
            seen.add(_fp(log))
        self.assertEqual(len(seen), 1)


# --------------------------------------------------------------------------------------------------
# 4 — and it MOVES on a genuine transition, including the ones a count would miss.
# --------------------------------------------------------------------------------------------------


class TestGenuineTransitionsChangeIt(unittest.TestCase):
    def _open(self) -> tuple[EpisodeLog, str]:
        log = _log(allow_reopen=True)
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        return log, _fp(log)

    def test_a_close_changes_it(self):
        log, before = self._open()
        log.close(log.current_open().id, t=5.0)
        self.assertNotEqual(_fp(log), before)

    def test_a_reopen_changes_it(self):
        log, _ = self._open()
        ep_id = log.current_open().id
        log.close(ep_id, t=5.0)
        closed = _fp(log)
        log.observe_decision(6.0, _YES, subject_key=SCENE_IDENTITY)  # reopen -> an "opened" row
        self.assertNotEqual(_fp(log), closed)

    def test_a_new_occurrence_changes_it(self):
        log, before = self._open()
        log.close(log.current_open().id, t=5.0)
        log.observe_decision(6.0, _YES, subject_key=SCENE_IDENTITY)
        self.assertNotEqual(_fp(log), before)  # a different episode id, at minimum

    def test_an_updated_revision_changes_it(self):
        log, before = self._open()
        log.record_revision(log.current_open(), "updated", t=1.0, value="a duffel bag")
        self.assertNotEqual(_fp(log), before)

    def test_annotate_changes_it_although_the_revision_count_does_not_move(self):
        """The reason a count is not a version: ``annotate`` is a single-writer field write.

        It moves ``value`` / ``incident_id`` — both of which reach the prompt — and touches
        ``revisions`` not at all. Anything keyed on ``len(revisions)`` would serve the old verdict.
        """
        log, before = self._open()
        ep = log.current_open()
        n = len(ep.revisions)
        log.annotate(ep, value="a duffel bag")
        after_value = _fp(log)
        self.assertEqual(len(ep.revisions), n)
        self.assertNotEqual(after_value, before)
        log.annotate(ep, incident_id="inc-1")
        self.assertEqual(len(ep.revisions), n)
        self.assertNotEqual(_fp(log), after_value)

    def test_another_open_episode_alongside_changes_it(self):
        """``other_open`` is reported to the model as a count, so it is part of the key."""
        log, before = self._open()
        log.observe_decision(1.0, _YES, subject_key="cam:2")
        self.assertNotEqual(_fp(log), before)

    def test_a_truncated_timeline_changes_it_even_when_the_retained_pairs_repeat(self):
        """``revisions_omitted`` reaches the prompt ("most recent 2 of 4 changes"), so it is keyed.

        The subtle case, and the reason the count is in the projection rather than only the pairs:
        with identical rows the retained ``(kind, value)`` slice does not move, and only the omitted
        count tells the model its view of the timeline got shorter.
        """
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        ep = log.current_open()
        for i in range(2):
            log.record_revision(ep, "updated", t=float(i), value="same")
        before = _fp(log, max_revisions=2)
        self.assertEqual(ep.to_judge_dict(max_revisions=2)["revisions_omitted"], 1)
        log.record_revision(ep, "updated", t=9.0, value="same")  # retained pairs identical
        d = ep.to_judge_dict(max_revisions=2)
        self.assertEqual([(r["kind"], r["value"]) for r in d["revisions"]],
                         [("updated", "same"), ("updated", "same")])
        self.assertEqual(d["revisions_omitted"], 2)
        self.assertNotEqual(_fp(log, max_revisions=2), before)


# --------------------------------------------------------------------------------------------------
# 5 — the cache LAYER keys on it, and the two invalidation paths are unaffected.
# --------------------------------------------------------------------------------------------------


class TestTheLayerKeysOnIt(unittest.TestCase):
    def _layer(self) -> FrameHashLayer:
        clock = [0.0]
        return FrameHashLayer(
            CacheSettings(enabled=True, ttl_s=100.0, tau_bits=0), clock=lambda: clock[0]
        )

    def _frame(self) -> list:
        return [[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]] * 8

    def _key(self, fp: "str | None", *, entity_id: "str | None" = "e1") -> CacheKey:
        return CacheKey(
            goal_id="g1", frame=self._frame(), entity_id=entity_id, condition="c",
            context_fingerprint=fp,
        )

    def test_the_same_fingerprint_hits_and_a_different_one_misses(self):
        layer = self._layer()
        layer.store(self._key("fp-a"), _YES, 0.9)
        self.assertEqual(layer.lookup(self._key("fp-a")).status, "HIT")
        self.assertEqual(layer.lookup(self._key("fp-b")).status, "DEFER")
        self.assertEqual(layer.lookup(self._key(None)).status, "DEFER")

    def test_a_fingerprinted_entry_never_shadows_an_unfingerprinted_one(self):
        layer = self._layer()
        layer.store(self._key(None), _YES, 0.9)
        layer.store(self._key("fp-a"), _NO, 0.9)
        self.assertTrue(layer.lookup(self._key(None)).verdict.satisfied)
        self.assertFalse(layer.lookup(self._key("fp-a")).verdict.satisfied)

    def test_invalidate_still_drops_a_goals_entries_whatever_the_fingerprint(self):
        layer = self._layer()
        for fp in (None, "fp-a", "fp-b"):
            layer.store(self._key(fp), _YES, 0.9)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer.invalidate("g1"), 3)
        self.assertEqual(len(layer), 0)

    def test_invalidate_entity_still_drops_a_corrected_entitys_entries(self):
        layer = self._layer()
        for fp in (None, "fp-a"):
            layer.store(self._key(fp), _YES, 0.9)
        layer.store(self._key("fp-a", entity_id="e2"), _YES, 0.9)
        self.assertEqual(layer.invalidate_entity("e1", goal_id="g1"), 2)
        self.assertEqual(len(layer), 1)

    def test_the_layer_never_parses_the_token(self):
        """The cache stays a contract: an arbitrary opaque string is a usable fingerprint."""
        layer = self._layer()
        layer.store(self._key("  ;:not-a-hash;:  "), _YES, 0.9)
        self.assertEqual(layer.lookup(self._key("  ;:not-a-hash;:  ")).status, "HIT")


# --------------------------------------------------------------------------------------------------
# 6 — end to end: a camera watch inside ONE open occurrence.
# --------------------------------------------------------------------------------------------------


class _CountingVision:
    """Returns a scripted verdict per tick and counts the calls it was actually asked to make."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls = 0
        self.contexts: list = []

    async def judge(self, condition, frame, **kw):  # noqa: D401 — a double, not an API
        self.contexts.append(kw.get("context"))
        verdict = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return verdict

    async def detect(self, frame, goals):
        return []


def _core(vision, *, ttl_s: float = 100.0, audit_rate: float = 0.0) -> Percept:
    settings = Settings(
        gate=GateSettings(threshold=0.7, validity_window_s=1000.0),
        cache=CacheSettings(enabled=True, ttl_s=ttl_s, audit_rate=audit_rate),
    )
    core = Percept.create(
        vision=vision, settings=settings, discover_plugins=False,
        source_profile=PHYSICAL_SCENE,
    )
    core.add_goal(
        Goal(
            id="g1", condition="a person is at the door", say="someone is at the door",
            state={"type": "watch", "count": 0},
        )
    )
    return core


def _still() -> list:
    """One unchanging griddable frame — the case the ROI dHash is supposed to recognise."""
    return [[float(x * 8) for x in range(9)] for _ in range(8)]


class TestEndToEndInsideAnOpenEpisode(unittest.TestCase):
    def _run(self, ticks: int, *, script=None, ttl_s: float = 100.0):
        vision = _CountingVision(script or [_YES])
        core = _core(vision, ttl_s=ttl_s)
        actions = []
        frame = _still()
        for i in range(ticks):
            events = asyncio.run(core.perceive_judged(frame, t=float(i)))
            actions.append([e.action for e in events])
        return core, vision, actions

    def test_the_cache_serves_inside_an_open_episode(self):
        """The measured defect, inverted: judge calls now stop when the picture stops changing."""
        core, vision, _ = self._run(40)
        self.assertGreater(core.metrics.get("cache.hit"), 30)
        self.assertLess(vision.calls, 10, "the episode bypass is still disabling the cache")

    def test_a_served_tick_does_not_fragment_the_episode(self):
        core, _, _ = self._run(40)
        log = core.episodes.get("g1")
        self.assertEqual(len(log.open_episodes()), 1)
        self.assertEqual(len(log._episodes), 1)

    def test_the_fire_count_stays_one(self):
        core, _, actions = self._run(40)
        self.assertEqual(sum(a.count("fire") for a in actions), 1)
        self.assertEqual(core.metrics.get("gate.fire"), 1)

    def test_the_open_episode_still_reaches_the_judge_on_a_cache_miss(self):
        """Keying is not a licence to stop injecting: a miss inside an episode still carries it."""
        core, vision, _ = self._run(40)
        with_episode = [
            c for c in vision.contexts
            if c and any("already-established episode" in str(v) for v in c.values())
        ]
        self.assertTrue(with_episode, "the episode text stopped reaching the adapter")

    def test_a_close_is_detected_when_the_picture_changes(self):
        """The occurrence boundary must survive caching, or the watch fires once and never again.

        A real scene change moves the ROI dHash past ``tau_bits``, which is the L0 frame-diff backstop
        the layer has always had. Keying the episode changed nothing about it.
        """
        vision = _CountingVision([_YES])
        core = _core(vision, ttl_s=100.0)
        frame = _still()
        for i in range(12):
            asyncio.run(core.perceive_judged(frame, t=float(i)))
        log = core.episodes.get("g1")
        self.assertEqual(len(log.open_episodes()), 1)
        vision.script = [_NO]
        moved = [[float(200 - x * 8) for x in range(9)] for _ in range(8)]
        asyncio.run(core.perceive_judged(moved, t=12.0))
        self.assertEqual(log.open_episodes(), [])
        self.assertEqual(core.metrics.get("episode.scene.closed"), 1)

    def test_a_close_is_still_detected_within_the_ttl_on_an_unchanged_picture(self):
        """The hard case: the ONLY thing that changed is the answer.

        With an identical frame and a live entry, the dHash has nothing to report — so what catches
        the close is the false-serve audit, the mechanism that was already there for exactly this
        (``DeterministicAuditor``: a stratified 1-in-N re-perceive whose fresh read wins on
        disagreement). It bounds how long a stale TRUE can be served inside the TTL, and it does so
        without the episode fold or the gate learning anything new.
        """
        vision = _CountingVision([_YES])
        core = _core(vision, ttl_s=1000.0, audit_rate=0.05)  # 1 hit in 20 re-perceived
        frame = _still()
        for i in range(10):
            asyncio.run(core.perceive_judged(frame, t=float(i)))
        log = core.episodes.get("g1")
        self.assertEqual(len(log.open_episodes()), 1)
        vision.script = [_NO]
        for i in range(10, 60):  # still well inside the TTL, and the frame never changes
            asyncio.run(core.perceive_judged(frame, t=float(i)))
            if not log.open_episodes():
                break
        self.assertEqual(log.open_episodes(), [], "a stale TRUE was served for the whole run")
        self.assertGreaterEqual(core.metrics.get("cache.false_serve"), 1)
        self.assertEqual(core.metrics.get("episode.scene.closed"), 1)

    def test_the_ttl_still_bounds_a_served_episode(self):
        """Past the TTL every entry is a hard miss, episode or no episode."""
        vision = _CountingVision([_YES])
        core = _core(vision, ttl_s=0.0)
        frame = _still()
        for i in range(10):
            asyncio.run(core.perceive_judged(frame, t=float(i)))
        self.assertEqual(core.metrics.get("cache.hit"), 0)
        self.assertEqual(vision.calls, 10)


class TestTheUnkeyedContextFence(unittest.TestCase):
    """Only the context the fingerprint COVERS may take the cache path.

    Removing the episode bypass is safe because the episode is keyed. The fence makes that a
    structural property rather than an audit of today's call sites: a context key nobody has taught
    the fingerprint about is answer-affecting state outside the key, so it bypasses instead of being
    served for. Every context the shipped sites build already trips one of the older bypass reasons,
    so this fires for nobody today — which is exactly why it needs a test of its own.
    """

    def test_a_context_key_the_fingerprint_does_not_cover_bypasses_the_cache(self):
        vision = _CountingVision([_YES])
        core = _core(vision, ttl_s=1000.0)
        goal = core.goals.get("g1")
        frame = _still()
        for _ in range(6):
            asyncio.run(
                core._judge(goal, frame, None, {}, context={"a_context_nobody_keyed": 1})
            )
        self.assertEqual(vision.calls, 6)
        self.assertEqual(core.metrics.get("cache.hit"), 0)

    def test_the_same_call_with_no_such_key_caches_as_normal(self):
        """The control: the fence is what makes the difference, not the direct ``_judge`` drive."""
        vision = _CountingVision([_YES])
        core = _core(vision, ttl_s=1000.0)
        goal = core.goals.get("g1")
        frame = _still()
        for _ in range(6):
            asyncio.run(core._judge(goal, frame, None, {}))
        self.assertEqual(vision.calls, 1)
        self.assertEqual(core.metrics.get("cache.hit"), 5)


# --------------------------------------------------------------------------------------------------
# 7 — the feedback loop a cache-served tick would otherwise close.
# --------------------------------------------------------------------------------------------------


class TestTheConfirmationCountLoop(unittest.TestCase):
    """A served tick must not inflate the observation count the next PROMPT quotes.

    The loop, if it did: the prompt says "confirmed across N observations"; a free serve advances N;
    the larger N strengthens the model's continuity prior; the stronger prior makes the next
    confirmation likelier. Nothing in that circuit looks at a pixel.
    """

    def test_the_prompt_quotes_the_judged_count_not_the_fold_count(self):
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        for i in range(1, 20):
            log.observe_decision(float(i), _YES, subject_key=SCENE_IDENTITY, judged=False)
        ep = log.current_open()
        self.assertEqual(ep.observations, 20)
        self.assertEqual(ep.judged_observations, 1)
        self.assertIn("confirmed across 1 observations", _text(log))
        self.assertNotIn("across 20 observations", _text(log))

    def test_a_judged_fold_advances_both_counts(self):
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        log.observe_decision(1.0, _YES, subject_key=SCENE_IDENTITY)
        ep = log.current_open()
        self.assertEqual((ep.observations, ep.judged_observations), (2, 2))

    def test_neither_count_is_in_the_fingerprint(self):
        """Counting is observability. Keying on it would re-disable the cache every tick."""
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY)
        before = _fp(log)
        log.observe_decision(1.0, _YES, subject_key=SCENE_IDENTITY, judged=False)
        log.observe_decision(2.0, _YES, subject_key=SCENE_IDENTITY)
        self.assertEqual(_fp(log), before)

    def test_a_cache_served_tick_does_not_advance_the_judged_count_end_to_end(self):
        vision = _CountingVision([_YES])
        core = _core(vision, ttl_s=100.0)
        frame = _still()
        for i in range(40):
            asyncio.run(core.perceive_judged(frame, t=float(i)))
        ep = core.episodes.get("g1").current_open()
        self.assertEqual(ep.observations, 40)
        self.assertEqual(
            ep.judged_observations, vision.calls,
            "the judged count drifted from the number of judge calls actually spent",
        )
        self.assertLess(ep.judged_observations, ep.observations)

    def test_the_judged_count_is_carried_but_the_split_is_visible(self):
        log = _log()
        log.observe_decision(0.0, _YES, subject_key=SCENE_IDENTITY, judged=False)
        ep = log.current_open()
        self.assertEqual(ep.observations, 1)
        self.assertEqual(ep.judged_observations, 0)  # opened off a serve — nothing was paid for
        d = ep.to_judge_dict()
        self.assertEqual(d["observations"], 1)
        self.assertEqual(d["judged_observations"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
