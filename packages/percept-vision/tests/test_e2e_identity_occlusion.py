"""Continuous identity across an occlusion — one product function, end to end.

Two subjects share a scene. One is hidden behind the other for a while, then both are visible
again. The capability under test is not a mechanism: it is whether the whole chain
**measurement -> identity -> episodes** keeps each subject as one subject, and whether the state a
subject accumulated downstream dies with it.

Five bars, and every one of them is shown to be able to FAIL before it is trusted — a bar that
cannot fail is worth less than no bar:

1. an id survives an occlusion shorter than the graph's coast window: one id, not two.
   *Falsifier:* the same track with a longer gap mints a second id.
2. two subjects never merge: distinct ids for every frame both are visible.
   *Falsifier:* the graph's other matcher, on the same track, holds them apart — so the merge is a
   property of one branch rather than of the fixture.
3. one episode per subject-occurrence.
   *Falsifier:* a re-minted id opens a third episode for two people.
4. an identity switch is reported.
   *Falsifier:* a control shows the counter surface reports the run's other decisions, so the
   silence around identity is specific rather than a dead surface.
5. a dead entity leaves no orphan gate, dwell or cache state.
   *Falsifier:* unwiring the eviction hook leaves all three behind.

**Scope, stated plainly.** The data is a scripted box track: hand-authored, deterministic, no
download, no model. That qualifies WIRING and geometry only. It says nothing about how a real
detector behaves on a real crossing, and it supports no accuracy number — a re-identification claim
needs labelled person tracks that do not exist in this tree.

**Driver.** The stages this file traverses are measurement, identity and episodes, so frames go in
through ``Percept.perceive_judged`` rather than the tier-0 admission loop: every scripted frame must
reach the entity graph or the track under test is not the track that was authored. Admission is a
different capability with its own file. One consequence shows up in the funnel below — with no
``edge.decision`` in the trace the shared stage-attribution helper counts admissions as
observations, which it documents, and ``incident.created`` is emitted by the runtime worker rather
than by this driver, so ``fired`` reads 0 here and is not asserted on.
"""

from __future__ import annotations

import asyncio
import dataclasses
import unittest
import warnings

from percept import Goal, Percept, UndrivenBehaviorRouteWarning
from percept.adapters import Box, Observation, Verdict
from percept.backends.fake import FakeVision
from percept.config import Settings
from percept.entity import EntityGraph
from percept.eval.stage_attribution import FUNNEL_POINTS, attribute
from percept.identity import AppearanceIdentity, ContinuityIdentity, cosine
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

# --------------------------------------------------------------------------------------------
# The scripted scene. Every box is the same size and sits on the same row, so the intersection
# over union between any two of them is a pure function of their horizontal distance — which is
# what makes each geometry below arguable on paper rather than only observable in a debugger.
# --------------------------------------------------------------------------------------------

BOX_W, BOX_H, ROW = 0.18, 0.40, 0.30

#: Seconds between frames. Fast enough that a subject can cross the scene inside the graph's coast
#: window, which is what the occlusion arm needs.
FRAME_DT = 0.1

#: The stationary subject's left edge — the thing the other subject passes behind.
ANCHORED_X = 0.40

#: A displacement of this much per frame keeps overlap with the subject's OWN previous box at
#: 0.565, just above the shipped 0.5 continuity threshold, so plain continuity holds the track.
WALK_STEP = 0.05

#: A displacement this large drops that self-overlap to 0.333, BELOW the threshold. A subject whose
#: own continuity has failed is exactly the one another box can capture, which is what the crossing
#: arm needs.
STRIDE_STEP = 0.09

#: The shipped ``Settings.identity.appearance_threshold``.
APPEARANCE_MATCH = 0.8


def box(x: float) -> Box:
    return Box(round(x, 4), ROW, BOX_W, BOX_H)


def person(x: float) -> Observation:
    return Observation("person", box(x), 0.9)


def grid_frame(seed: int) -> list[list[int]]:
    """A frame with real horizontal structure, so the perception cache can actually hash it.

    ``FakeVision`` keys its scripted verdicts and detections off ``str(frame)``, and the cache's
    difference-hash accepts a plain 2-D grid, so one deterministic object serves both without
    importing numpy or reading a file.
    """
    return [[(x * 37 + y * 11 + seed * 53) % 251 for x in range(8)] for y in range(8)]


Scene = list[tuple[list[list[int]], float, list[Observation]]]


def occlusion_track(hidden_frames: int, *, keeps_walking: bool = True) -> Scene:
    """One subject parked at ``ANCHORED_X``; a second walks left and is hidden for a stretch.

    ``keeps_walking`` is the difference between the two things an occlusion can mean. True: the
    walker crosses behind the anchored subject and reappears on the far side, clear of it, so
    nothing but its own identity can explain the id it comes back with. False: it stops behind the
    anchored subject and reappears exactly where it went in, so plain continuity re-acquires it.
    Both shapes run for the same number of frames and present the same number of detections, which
    is what makes them comparable.
    """
    frames: Scene = []
    x = 0.80
    for i in range(5 + hidden_frames + 3):
        hidden = 5 <= i < 5 + hidden_frames
        seen = [person(ANCHORED_X)]
        if not hidden:
            seen.append(person(x))
        frames.append((grid_frame(i), round(i * FRAME_DT, 4), seen))
        if i < 5 or keeps_walking:
            x -= WALK_STEP
    return frames


def crossing_track(n: int = 9) -> Scene:
    """Both subjects visible throughout; the mover strides across the anchored one.

    At the fifth frame the mover's box sits 0.03 from the anchored subject's and 0.09 from its own
    previous one, so its overlap with the STRANGER (0.714) beats the overlap with itself (0.333)
    and clears the continuity threshold. That is the whole of a crossing: for one frame, the
    nearest box is not your own.
    """
    frames: Scene = []
    x = 0.79
    for i in range(n):
        frames.append((grid_frame(i), round(i * FRAME_DT, 4), [person(ANCHORED_X), person(x)]))
        x -= STRIDE_STEP
    return frames


def passing_track(n: int = 9) -> Scene:
    """The same two subjects and the same frame count, but the mover never comes close.

    The counterfactual for the crossing: identical in every respect a counter or a trace stage can
    see, except that no frame merges.
    """
    frames: Scene = []
    x = 0.95
    for i in range(n):
        frames.append((grid_frame(i), round(i * FRAME_DT, 4), [person(ANCHORED_X), person(x)]))
        x -= WALK_STEP
    return frames


def churn_track(n: int = 3) -> Scene:
    """One subject at a time, each at a disjoint position — a scene whose subjects turn over."""
    return [
        (grid_frame(i), round(i * 0.5, 4), [person(0.05 + 0.30 * (i % 3))])
        for i in range(n)
    ]


# --------------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------------


def scripted_vision(frames: Scene) -> FakeVision:
    """A keyless judge/detector scripted off the frame itself — wiring, never intelligence."""
    return FakeVision(
        script={
            str(f): Verdict(satisfied=True, confidence=0.95, evidence="a person")
            for f, _t, _obs in frames
        },
        detect_script={str(f): obs for f, _t, obs in frames},
    )


@dataclasses.dataclass
class RunResult:
    """Everything one scripted run produced, so each bar reads a fact rather than re-running."""

    core: Percept
    frames: Scene
    ids_per_frame: list[list[str]]
    records: tuple
    manifest: object
    health: list[str]
    cache_layer: object

    @property
    def merged_frames(self) -> list[list[str]]:
        """Frames on which two detections were bound to one entity id."""
        return [ids for ids in self.ids_per_frame if len(ids) != len(set(ids))]

    @property
    def live_ids(self) -> set[str]:
        return {e.id for e in self.core.graph.active()}

    def observability_signature(self) -> tuple:
        """Everything this run said about itself that a client could act on.

        Counters, health events, the entity each gate decision was scoped by, and which trace
        stages were emitted at all. Wall-clock totals are excluded deliberately: they differ
        between any two runs of anything and would make every comparison below trivially true.

        The counter tuple is the union of the runtime's own registry and the entity graph's
        identity counters (``core.graph.identity_counters()``), because both are published surfaces
        a client reads off the same object it already holds. They are merged rather than compared
        separately so a bar below cannot pass on one surface while the other stays silent.
        """
        counters = tuple(
            sorted(
                [(k, v) for k, v in self.core.metrics.snapshot().items() if isinstance(v, int)]
                + list(self.core.graph.identity_counters().items())
            )
        )
        gate_entities = tuple(
            r.data.get("entity_id") for r in self.records if r.stage == "gate.classified"
        )
        stages = tuple(sorted({r.stage for r in self.records}))
        return counters, tuple(self.health), gate_entities, stages


def drive(frames: Scene, *, goals, settings=None, trace=False, mutate=None) -> RunResult:
    """Push every scripted frame through one core and record what identity did with it."""

    async def _go() -> RunResult:
        recorder = (
            TraceRecorder(
                RunTraceManifest.create(source="cam", policy=TracePolicy(level="full"))
            )
            if trace
            else None
        )
        health: list[str] = []
        core = Percept.create(
            vision=scripted_vision(frames),
            discover_plugins=False,
            settings=settings,
            on_trace=recorder.tracer() if recorder is not None else None,
            on_health=health.append,
        )
        if mutate is not None:
            mutate(core)
        for goal in goals:
            # The counting goal is HOST-driven on purpose (``with_counting`` folds each subject in
            # by id below), which is exactly what the runtime's undriven-route warning tells a
            # caller to do. Silenced here so the diagnostic keeps its signal value elsewhere;
            # nothing else is filtered.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndrivenBehaviorRouteWarning)
                core.add_goal(goal)
        ids_per_frame: list[list[str]] = []
        for frame, t, _obs in frames:
            await core.perceive_judged(frame, source="cam", t=t)
            ids_per_frame.append(list(core.graph._last_frame_ids))
        if recorder is not None:
            recorder.close()
        return RunResult(
            core=core,
            frames=frames,
            ids_per_frame=ids_per_frame,
            records=recorder.records if recorder is not None else (),
            manifest=recorder.manifest if recorder is not None else None,
            health=health,
            cache_layer=core.cache.layers[0] if core.cache.layers else None,
        )

    return asyncio.run(_go())


WATCH = Goal(id="zone", condition="a person is in the zone", say="someone is here")
DWELL_WATCH = Goal(
    id="zone",
    condition="a person is in the zone",
    say="someone is here",
    state={"hold_s": 0.2},
)


def counting_goal() -> Goal:
    return Goal(id="visits", condition="a person visits", say="a visit", state={"type": "count"})


def small_scene_settings(max_entities: int) -> Settings:
    """A scene the graph can only hold ``max_entities`` subjects in, with the cache switched on.

    Both halves matter for the eviction bar: the cap is what kills a subject mid-run, and the cache
    has to hold a real slot keyed on that subject for its sweep to be observable at all.
    """
    base = Settings()
    return dataclasses.replace(
        base,
        entity=dataclasses.replace(base.entity, max_entities=max_entities),
        cache=dataclasses.replace(base.cache, enabled=True),
    )


def with_counting(frames: Scene, goals) -> RunResult:
    """Run ``frames``, then fold one observation per SUBJECT into the counting log by its id.

    This is the seam the whole cross-cutting concern rests on: the episode log is told who an
    observation is about using nothing but the id identity handed up. If identity is wrong here the
    episode is wrong, and nothing downstream can tell.
    """
    result = drive(frames, goals=goals)
    for (frame, t, _obs), ids in zip(frames, result.ids_per_frame):
        for entity_id in ids:
            result.core.count_event("visits", str(frame), subject_key=entity_id, t=t)
    return result


def episodes_of(result: RunResult, goal_id: str = "visits"):
    return result.core.episodes[goal_id]


# --------------------------------------------------------------------------------------------
# Bar 1 — an id survives an occlusion shorter than the coast window
# --------------------------------------------------------------------------------------------


class IdentitySurvivesOcclusionTests(unittest.TestCase):
    """The capability itself: hidden for a moment is not the same as gone."""

    def test_the_hidden_subject_comes_back_with_the_id_it_left_with(self):
        result = drive(occlusion_track(7), goals=[WATCH])
        first, last = result.ids_per_frame[0], result.ids_per_frame[-1]

        self.assertEqual(len(first), 2, "the scene opens with two detected subjects")
        self.assertEqual(len(last), 2, "and closes with two")
        self.assertEqual(
            set(last), set(first), "the occluded subject came back as a different subject"
        )
        self.assertEqual(
            len(result.live_ids),
            2,
            "a re-minted id would leave the scene holding three subjects for two people",
        )

    def test_the_same_track_with_a_longer_gap_does_mint_a_second_id(self):
        """The falsifier for the bar above.

        Fourteen hidden frames is 1.4 s, past the graph's 1.0 s coast window. If the assertion
        above could not tell the two runs apart it would be measuring nothing.
        """
        result = drive(occlusion_track(14), goals=[WATCH])
        self.assertNotEqual(
            set(result.ids_per_frame[-1]),
            set(result.ids_per_frame[0]),
            "past the coast window the graph must NOT still claim continuity",
        )
        self.assertEqual(len(result.live_ids), 3, "two subjects, three ids")

    def test_a_subject_that_returns_to_where_it_hid_keeps_its_id_at_any_gap(self):
        """The other half of the coast rule: the window bounds PREDICTION, not memory.

        Same 1.4 s gap, same frame count. The walker stopped rather than crossing, so plain
        continuity re-acquires it and no coast is needed. Without this, the falsifier above could
        be read as 'a long gap always loses the subject', which is not what the graph does.
        """
        result = drive(occlusion_track(14, keeps_walking=False), goals=[WATCH])
        self.assertEqual(set(result.ids_per_frame[-1]), set(result.ids_per_frame[0]))
        self.assertEqual(len(result.live_ids), 2)

    def test_both_subjects_hold_distinct_ids_on_every_frame_both_are_visible(self):
        result = drive(occlusion_track(7), goals=[WATCH])
        both_visible = [ids for ids in result.ids_per_frame if len(ids) == 2]
        self.assertGreater(len(both_visible), 1, "the fixture must show both subjects repeatedly")
        self.assertEqual(
            result.merged_frames, [], "two subjects in one frame collapsed onto one id"
        )

    def test_the_anchored_subject_is_never_disturbed_by_the_walker(self):
        result = drive(occlusion_track(7), goals=[WATCH])
        anchored = result.ids_per_frame[0][0]
        for ids in result.ids_per_frame:
            self.assertIn(anchored, ids, "the subject that never moved must never change id")


# --------------------------------------------------------------------------------------------
# Cross-cutting: the stages this run traverses, read through the shared helper
# --------------------------------------------------------------------------------------------


class StageAttributionAcrossTheOcclusionTests(unittest.TestCase):
    """The funnel and the per-segment spans for the same run, composed rather than re-derived."""

    def setUp(self):
        self.result = drive(occlusion_track(7), goals=[WATCH], trace=True)
        self.attribution = attribute(self.result.records, manifest=self.result.manifest)

    def test_the_funnel_is_monotone_and_nothing_was_lost_reaching_a_verdict(self):
        counts = self.attribution.funnel.counts
        ordered = [counts[point] for point in FUNNEL_POINTS]
        self.assertEqual(
            ordered, sorted(ordered, reverse=True), f"funnel is not monotone: {dict(counts)}"
        )
        self.assertEqual(counts["observed"], len(self.result.ids_per_frame))
        self.assertEqual(
            self.attribution.funnel.step("judge").ratio,
            1.0,
            "a judge span that opened and never closed would hide exactly the slow calls",
        )

    def test_the_spans_are_quotable_because_no_deadline_truncated_them(self):
        self.assertTrue(
            self.attribution.publishable,
            f"unquotable segments: {self.attribution.unpublishable_segments}",
        )
        self.assertIn("judge", self.attribution.measured_segments)

    def test_every_entity_the_gate_and_trace_were_keyed_on_is_a_subject_that_exists(self):
        """The cross-cutting rule as one assertion: a wrong id must not poison a slot.

        Nothing was evicted in this run, so every key a downstream store holds must name a subject
        the graph still knows about. A key naming an unknown id is an orphan however it got there —
        by eviction, by a re-mint, or by a merge.
        """
        core = self.result.core
        live = self.result.live_ids
        gate_orphans = [k for k in core.gate._keys if k[1] not in live]
        self.assertEqual(gate_orphans, [], f"gate keys naming no live subject: {gate_orphans}")

        traced = {
            r.data.get("entity_id") for r in self.result.records if r.stage == "gate.classified"
        }
        self.assertTrue(traced, "the trace must carry the entity each decision was scoped by")
        self.assertTrue(
            traced <= live, f"the trace scoped a decision by an unknown id: {traced - live}"
        )


# --------------------------------------------------------------------------------------------
# Bar 3 — one episode per subject-occurrence
# --------------------------------------------------------------------------------------------


class EpisodePerSubjectOccurrenceTests(unittest.TestCase):
    """Two people present once each is two episodes — through the occlusion, not around it."""

    def test_two_subjects_across_one_occlusion_produce_exactly_two_episodes(self):
        result = with_counting(occlusion_track(7), [WATCH, counting_goal()])
        log = episodes_of(result)
        self.assertEqual(log.count(), 2, "one occurrence per subject, and no more")
        self.assertEqual(len({ep.subject_key for ep in log.open_episodes()}), 2)
        for episode in log.open_episodes():
            self.assertEqual(
                episode.started_at,
                0.0,
                "both subjects were present from the first frame; neither episode may start late",
            )

    def test_an_id_lost_past_the_coast_window_opens_a_third_episode(self):
        """The falsifier — and the cross-cutting failure the bar exists to catch.

        The scene still holds two people. The episode log says three occurrences, because the
        walker's second id is a second subject as far as everything downstream can tell. That is
        'a wrong id silently poisons a downstream episode', measured rather than argued.
        """
        result = with_counting(occlusion_track(14), [WATCH, counting_goal()])
        log = episodes_of(result)
        self.assertEqual(log.count(), 3, "two subjects, three episodes")
        starts = sorted(ep.started_at for ep in log.open_episodes())
        self.assertGreater(
            starts[-1], 0.0, "the third episode opened mid-run, when the id was re-minted"
        )
        self.assertEqual(
            result.core.metrics.get("episode.new"),
            3,
            "and the log counted it as a genuinely new occurrence",
        )


# --------------------------------------------------------------------------------------------
# Bar 2 — two subjects never merge
# --------------------------------------------------------------------------------------------


class _SwatchEmbedder:
    """A stand-in appearance embedder: a crop's look is whatever the fixture says it is.

    The shipped embedder needs opencv, and what is under test here is not colour histograms — it is
    what the graph does with two observations whose signatures both point at one entity. Keying the
    look off the box position keeps the whole arm in the standard library and deterministic.
    """

    def __init__(self, looks: dict[float, tuple[float, ...]]) -> None:
        self.looks = dict(looks)

    def restage(self, looks: dict[float, tuple[float, ...]]) -> None:
        """Move the same wardrobe to new positions — the subjects moved, their looks did not."""
        self.looks = dict(looks)

    def __call__(self, _frame, at: Box):
        # The graph calls ``embedder(frame, box)``; this stand-in reads only the box, because the
        # fixture — not the pixels — decides what each crop looks like.
        return self.looks.get(round(at.x, 4))


class SameFrameAssignmentTests(unittest.TestCase):
    """One frame, two detections, one entity: the failure this bar names.

    A frame is the strongest identity evidence there is. Two simultaneous detections are two
    subjects — whatever they look like, however close they are. The graph has no rule that says so:
    each observation is asked the 'same thing?' question against the whole entity set with no memory
    of which entities this frame already claimed.

    Read the two runs in the first pair together. Same track, same graph, same data; only which
    matcher answers differs. The recovery matcher excludes an entity another observation already
    claimed this frame and holds the subjects apart. The primary matcher has no such refusal. The
    invariant is therefore stated inside the implementation — in one branch of it.
    """

    def _continuity_ids(self, matcher) -> list[list[str]]:
        graph = EntityGraph(identity=matcher)
        return [[e.id for e in graph.ingest(obs, t=t)] for _f, t, obs in crossing_track()]

    # -- the crossing, under the shipped geometric matcher ------------------------------------
    def test_the_crossing_frame_carries_two_detections_the_detector_called_separate(self):
        """Fixture control, stating the geometry honestly rather than flatteringly.

        On the merge frame the two boxes overlap heavily — that is what a subject passing directly
        in front of another looks like, and it is the only shape a single symmetric overlap
        threshold can produce a merge from. The detector still reported TWO subjects at TWO
        positions, which is the fact the graph discards.
        """
        anchored, mover = (obs.box for obs in crossing_track()[4][2])
        self.assertNotEqual(anchored.x, mover.x, "two detections, two positions")
        self.assertGreater(anchored.iou(mover), 0.5, "and a near-total occlusion between them")

    def test_the_recovery_matcher_refuses_an_entity_another_observation_already_claimed(self):
        """Control: with the primary matcher unable to answer, no frame ever merges."""
        never_matches = ContinuityIdentity(iou_match=1.01)  # no overlap can reach this
        for ids in self._continuity_ids(never_matches):
            self.assertEqual(len(set(ids)), 2, f"the recovery path merged two subjects: {ids}")

    def test_two_subjects_never_bind_to_one_id_in_one_frame(self):
        for ids in self._continuity_ids(ContinuityIdentity()):
            self.assertEqual(len(set(ids)), 2, f"two subjects collapsed onto one id: {ids}")

    def test_the_shipped_default_pipeline_never_merges_two_detections(self):
        """REWRITTEN (was ``test_the_merge_is_reachable_through_the_shipped_default_pipeline``).

        It characterized the defect: exactly one frame of a full ``Percept`` run bound two
        detections to one id. Same-frame exclusivity landed, so the assertion is now that no frame
        does — reached through the shipped default rather than a hand-built graph, so the pin above
        is not a lab curiosity in either direction. The refusal is reported rather than silent,
        which is the other half of the same fix.
        """
        result = drive(crossing_track(), goals=[WATCH])
        self.assertEqual(
            result.merged_frames,
            [],
            "the shipped default bound both detections to one id on the crossing frame",
        )
        self.assertEqual(
            result.core.graph.identity_counters()["identity.contested"],
            1,
            "the crossing was refused silently — a near-merge is the identity event most worth "
            "seeing, and the run must say it happened",
        )

    # -- the same gap through appearance re-id, where the boxes need not touch at all ---------
    def _appearance_ids(self, other_look: tuple[float, ...]) -> list[list[str]]:
        """Two subjects at disjoint boxes, both re-acquired by appearance on the second frame."""
        mine = (1.0, 0.0, 0.0)
        wardrobe = _SwatchEmbedder({0.10: mine, 0.70: other_look})
        graph = EntityGraph(
            identity=AppearanceIdentity(threshold=APPEARANCE_MATCH), embedder=wardrobe
        )
        out = [[e.id for e in graph.ingest([person(0.10), person(0.70)], t=0.0)]]
        # Both subjects displace beyond their own overlap, so continuity cannot answer for either
        # and the appearance fallback decides both — still on ONE frame, still at disjoint boxes.
        wardrobe.restage({0.40: mine, 0.95: other_look})
        out.append([e.id for e in graph.ingest([person(0.40), person(0.95)], t=0.5)])
        return out

    def test_two_differently_dressed_subjects_are_held_apart_by_appearance(self):
        """Control: the appearance arm resolves two subjects correctly when they differ."""
        for ids in self._appearance_ids((0.0, 1.0, 0.0)):
            self.assertEqual(len(set(ids)), 2)

    def test_two_similar_looking_subjects_in_one_frame_stay_two_subjects(self):
        alike = (0.97, 0.24, 0.0)
        self.assertGreater(
            cosine((1.0, 0.0, 0.0), alike), APPEARANCE_MATCH, "the fixture must be a real near-match"
        )
        for ids in self._appearance_ids(alike):
            self.assertEqual(len(set(ids)), 2, f"two subjects side by side became one: {ids}")


class MergePoisonsTheDownstreamEpisodeTests(unittest.TestCase):
    """What a merge would cost once identity is handed up — the concern the bar protects.

    This is the layer the same-frame invariant exists for. A merge never raises: both episodes
    still exist and the log still counts 2, so the only trace of it is an observation ledger in
    which one subject's episode absorbed a frame it was never seen on.
    """

    def _run(self) -> tuple[RunResult, int]:
        frames = crossing_track()
        return with_counting(frames, [WATCH, counting_goal()]), len(frames)

    def test_no_episode_absorbs_an_observation_of_the_other_subject(self):
        result, n_frames = self._run()
        for episode in episodes_of(result).open_episodes():
            self.assertLessEqual(
                episode.observations,
                n_frames,
                f"episode {episode.subject_key} folded in more observations than there were frames",
            )

    def test_the_shape_of_the_log_could_never_have_shown_the_damage(self):
        """Control: why the bar above is the one that has to hold.

        Two episodes exist and the count is 2 whether or not a frame merged — a merge moves an
        observation between episodes, it does not drop one. So the log's SHAPE is blind to it, and
        the observation ledger asserted above is the only place it was ever visible.
        """
        result, n_frames = self._run()
        log = episodes_of(result)
        self.assertEqual(log.count(), 2)
        folded = [ep.observations for ep in log.open_episodes()]
        self.assertEqual(
            sum(folded),
            2 * n_frames,
            "every detection was folded somewhere; the merge moved one, it did not drop it",
        )


# --------------------------------------------------------------------------------------------
# Bar 4 — an identity switch is reported
# --------------------------------------------------------------------------------------------


class IdentityChurnObservabilityTests(unittest.TestCase):
    """A merge and a re-mint are the two identity events a client has to be able to see.

    Each bar here is a counterfactual rather than a keyword search: run the scene where identity
    went wrong, run the closest possible scene where it did not, and ask whether anything the
    runtime published can tell them apart. If the two signatures are equal, no dashboard, alert or
    trace reader can recover the switch — whatever the fields happen to be called.
    """

    def test_the_counter_surface_does_report_the_runs_other_decisions(self):
        """Control: the surface the bars below read is alive.

        Without this, 'the two runs published the same thing' could just mean 'neither published
        anything at all'.
        """
        result = with_counting(occlusion_track(14), [WATCH, counting_goal()])
        counters = result.core.metrics.snapshot()
        self.assertEqual(counters.get("episode.new"), 3)
        self.assertGreater(counters.get("episode.continuation", 0), 0)
        self.assertGreater(counters.get("gate.silent", 0), 0)

    def test_the_two_counterfactual_scenes_really_do_differ_in_identity(self):
        """Fixture control: the pairs below are only meaningful if identity actually diverged.

        REWRITTEN with the same-frame invariant. The crossing scene no longer merges — that is the
        sibling fix — so what now separates it from the passing scene is the CONTESTED claim it
        forces: on the crossing frame the mover's best match is a subject another detection already
        holds, and the graph refuses it. A near-merge is the identity event, and a run that has one
        is not the same run as one that does not.
        """
        crossed = drive(crossing_track(), goals=[WATCH], trace=True)
        passed = drive(passing_track(), goals=[WATCH], trace=True)
        self.assertEqual(crossed.merged_frames, [])
        self.assertEqual(passed.merged_frames, [])
        self.assertEqual(crossed.core.graph.identity_counters()["identity.contested"], 1)
        self.assertEqual(passed.core.graph.identity_counters()["identity.contested"], 0)

        remint = drive(occlusion_track(14), goals=[WATCH], trace=True)
        kept = drive(occlusion_track(14, keeps_walking=False), goals=[WATCH], trace=True)
        self.assertEqual(len(remint.live_ids), 3)
        self.assertEqual(len(kept.live_ids), 2)

    def test_a_run_containing_a_near_merge_reports_something_a_clean_run_does_not(self):
        """The bar: two subjects contended for one id, and the run has to say so.

        Nothing else in the two runs differs — same frame count, same detections, same goal, same
        gate decisions. ``gate.classified`` carries the id of the ONE in-focus entity per goal per
        frame, so it reads identically on both; the counterfactual is what proves the signal is the
        identity event and not an artefact of the scene.
        """
        crossed = drive(crossing_track(), goals=[WATCH], trace=True)
        passed = drive(passing_track(), goals=[WATCH], trace=True)
        self.assertNotEqual(
            crossed.observability_signature(),
            passed.observability_signature(),
            "a run that nearly merged two subjects is indistinguishable from a clean one in every "
            "published surface",
        )

    def test_a_run_that_re_mints_an_id_reports_something_a_run_that_keeps_it_does_not(self):
        remint = drive(occlusion_track(14), goals=[WATCH], trace=True)
        kept = drive(occlusion_track(14, keeps_walking=False), goals=[WATCH], trace=True)
        self.assertNotEqual(
            remint.observability_signature(),
            kept.observability_signature(),
            "a subject silently became a different subject and the run said nothing",
        )


# --------------------------------------------------------------------------------------------
# Bar 5 — a dead entity leaves nothing behind
# --------------------------------------------------------------------------------------------


class EvictionHygieneTests(unittest.TestCase):
    """When the scene turns over, the state a dead subject accumulated must die with it.

    The run is a real one rather than hand-planted state: the goal carries a hold, so each
    subject's key is DWELL-PENDING — a dwell timer and an eviction stamp and no armed bit at all.
    That is the shape a sweep over the armed bit alone would miss, and the shape a re-minted id
    would then be measured against.
    """

    def _run(self, *, unwire_hook: bool) -> RunResult:
        return drive(
            churn_track(),
            goals=[DWELL_WATCH],
            settings=small_scene_settings(2),
            mutate=(lambda core: setattr(core.graph, "on_evict", None)) if unwire_hook else None,
        )

    @staticmethod
    def _orphans(result: RunResult) -> dict[str, list]:
        core, live = result.core, result.live_ids

        def dead(keys):
            return [k for k in keys if isinstance(k, tuple) and len(k) >= 2 and k[1] not in live]

        return {
            "keys": dead(core.gate._keys),
            "armed": dead(core.gate._armed),
            "dwell_since": dead(core.gate._dwell_since),
            "dwell_last_supported": dead(core.gate._dwell_last_supported),
            "touched": dead(core.gate._touched),
            "no_evict": dead(core.gate._no_evict),
            "last_fire": dead(core.gate._last_fire),
            "cache": dead(result.cache_layer._entries),
        }

    def test_the_scene_really_turned_over_and_the_state_under_test_really_existed(self):
        """Fixture control — an empty gate and an empty cache would pass the bar vacuously."""
        result = self._run(unwire_hook=True)
        self.assertEqual(len(result.live_ids), 2, "the cap must have killed a subject")
        orphans = self._orphans(result)
        self.assertEqual(len(orphans["keys"]), 1, "exactly one subject died")
        self.assertEqual(
            orphans["dwell_since"],
            orphans["keys"],
            "the dead subject's key must carry the dwell timer the sweep has to reach",
        )
        self.assertEqual(orphans["touched"], orphans["keys"])
        self.assertEqual(orphans["armed"], [], "a dwell-pending key never wrote an armed bit")
        self.assertEqual(len(orphans["cache"]), 1, "and it must own a real cache slot")

    def test_a_dead_entity_leaves_no_gate_dwell_or_cache_state(self):
        orphans = self._orphans(self._run(unwire_hook=False))
        self.assertEqual(
            {store: found for store, found in orphans.items() if found},
            {},
            "an evicted subject left state behind",
        )

    def test_the_same_assertion_fails_when_the_eviction_hook_is_unwired(self):
        """The falsifier: with the hook gone the sweep never runs and every store keeps the id."""
        orphans = self._orphans(self._run(unwire_hook=True))
        self.assertTrue(orphans["keys"], "unwired, the gate must keep the dead subject's key")
        self.assertTrue(orphans["dwell_since"], "including its dwell timer")
        self.assertTrue(orphans["cache"], "and the cache must keep its slot")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
