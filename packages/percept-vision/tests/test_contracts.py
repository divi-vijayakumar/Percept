"""L1: percept-contracts — DTO defaults, Concern.from_goal, Incident, version negotiation,
and JSON-schema generation. Stdlib only, no backends, no keys."""

import unittest

import percept.contracts as pc
from percept.contracts import (
    ANNOTATION_RUBRIC,
    ANNOTATION_RUBRIC_VERSION,
    CONTRACT_VERSION,
    AudioFrame,
    Box,
    ClipEvent,
    Concern,
    Consequence,
    ContractVersionMismatch,
    Decision,
    Evidence,
    Goal,
    GroundTruth,
    Incident,
    IncidentSnippet,
    JudgmentEvent,
    JudgmentTiming,
    JudgmentTimingBuilder,
    Schedule,
    Subject,
    Tier0Signal,
    TrackingBehavior,
    Verdict,
    WatchCondition,
    WatchEvent,
    WatchSpec,
    minute_of_day,
    negotiate,
)


class TestDTODefaults(unittest.TestCase):
    def test_verdict_defaults(self):
        v = Verdict(satisfied=True, confidence=0.8)
        self.assertEqual(v.evidence, "")

    def test_goal_default_state_is_watch(self):
        g = Goal(id="g1", condition="user drinks coffee", say="careful")
        self.assertEqual(g.state["type"], "watch")
        self.assertEqual(g.state["count"], 0)
        self.assertEqual(g.modality, "speak")
        self.assertEqual(g.source, "camera")
        self.assertEqual(g.expiry, "until_cancelled")

    def test_decision_defaults(self):
        d = Decision(route="ignore")
        self.assertEqual(d.look, "none")
        self.assertEqual(d.goal_ops, [])
        self.assertEqual(d.endpoint, "n/a")

    def test_box_is_frozen_and_iou(self):
        b = Box(0.0, 0.0, 1.0, 1.0)
        with self.assertRaises(Exception):
            b.x = 0.5  # frozen
        # identical boxes -> IoU 1.0; disjoint -> 0.0
        self.assertAlmostEqual(b.iou(Box(0.0, 0.0, 1.0, 1.0)), 1.0)
        self.assertAlmostEqual(b.iou(Box(5.0, 5.0, 1.0, 1.0)), 0.0)

    def test_consequence_defaults(self):
        c = Consequence()
        self.assertEqual(c.miss_cost, "mild")
        self.assertEqual(c.false_fire_cost, "mild")
        self.assertFalse(c.protected_floor)

    def test_tracking_defaults(self):
        t = TrackingBehavior()
        self.assertEqual(t.pattern, "watch")
        self.assertEqual(t.rearm, "on_false")
        self.assertIsNone(t.cadence_s)
        self.assertFalse(t.exhaustive)

    def test_watchspec_wire_version_unchanged(self):
        # WatchSpec.version is the wire integer (NOT the contract semver) and must stay 1.
        ws = WatchSpec(conditions=[])
        self.assertEqual(ws.version, 1)

    def test_evidence_default_coverage(self):
        e = Evidence(t=1.0, source_id="cam", confidence=0.9, satisfied=True)
        self.assertEqual(e.coverage, 1.0)


class TestSchedule(unittest.TestCase):
    def test_default_is_always_armed(self):
        # No windows + all seven days -> armed at every civil moment (the "always" default).
        s = Schedule()
        self.assertEqual(s.windows, ())
        self.assertEqual(s.days, (1, 2, 3, 4, 5, 6, 7))
        for wd in range(1, 8):
            self.assertTrue(s.is_active(weekday=wd, minute_of_day=0))
            self.assertTrue(s.is_active(weekday=wd, minute_of_day=12 * 60))
            self.assertTrue(s.is_active(weekday=wd, minute_of_day=1439))

    def test_is_frozen(self):
        s = Schedule()
        with self.assertRaises(Exception):
            s.windows = ((0, 60),)  # frozen

    def test_day_filter(self):
        weekdays_only = Schedule(days=(1, 2, 3, 4, 5))
        self.assertTrue(weekdays_only.is_active(weekday=1, minute_of_day=600))  # Monday
        self.assertTrue(weekdays_only.is_active(weekday=5, minute_of_day=600))  # Friday
        self.assertFalse(weekdays_only.is_active(weekday=6, minute_of_day=600))  # Saturday
        self.assertFalse(weekdays_only.is_active(weekday=7, minute_of_day=600))  # Sunday

    def test_time_of_day_window(self):
        # Armed 09:00 (540) inclusive .. 17:00 (1020) exclusive.
        office = Schedule(windows=((540, 1020),))
        self.assertFalse(office.is_active(weekday=1, minute_of_day=539))  # 08:59
        self.assertTrue(office.is_active(weekday=1, minute_of_day=540))  # 09:00 (inclusive start)
        self.assertTrue(office.is_active(weekday=1, minute_of_day=1019))  # 16:59
        self.assertFalse(office.is_active(weekday=1, minute_of_day=1020))  # 17:00 (exclusive end)

    def test_window_wraps_past_midnight(self):
        # Overnight watch: 22:00 (1320) -> 06:00 (360).
        overnight = Schedule(windows=((1320, 360),))
        self.assertTrue(overnight.is_active(weekday=1, minute_of_day=1320))  # 22:00
        self.assertTrue(overnight.is_active(weekday=1, minute_of_day=1439))  # 23:59
        self.assertTrue(overnight.is_active(weekday=1, minute_of_day=0))  # 00:00
        self.assertTrue(overnight.is_active(weekday=1, minute_of_day=359))  # 05:59
        self.assertFalse(overnight.is_active(weekday=1, minute_of_day=360))  # 06:00 (exclusive)
        self.assertFalse(overnight.is_active(weekday=1, minute_of_day=720))  # noon

    def test_days_and_windows_AND_together(self):
        # Weekend mornings only: armed iff day in {Sat,Sun} AND time in 06:00..12:00.
        s = Schedule(windows=((360, 720),), days=(6, 7))
        self.assertTrue(s.is_active(weekday=6, minute_of_day=400))  # Sat morning
        self.assertFalse(s.is_active(weekday=6, minute_of_day=800))  # Sat afternoon (bad time)
        self.assertFalse(s.is_active(weekday=1, minute_of_day=400))  # Mon morning (bad day)

    def test_minute_of_day_helper(self):
        self.assertEqual(minute_of_day(0, 0), 0)
        self.assertEqual(minute_of_day(9, 30), 570)
        self.assertEqual(minute_of_day(23, 59), 1439)

    def test_watchcondition_schedule_defaults_none(self):
        # Optional field defaults to None (-> always armed) so the default path is unchanged.
        c = WatchCondition(goal_id="g", anchor_text="x", threshold=0.3)
        self.assertIsNone(c.schedule)
        c2 = WatchCondition(goal_id="g", anchor_text="x", threshold=0.3, schedule=Schedule(days=(1,)))
        self.assertEqual(c2.schedule.days, (1,))


class TestConcernFromGoal(unittest.TestCase):
    def test_from_goal_equivalence(self):
        g = Goal(id="g1", condition="the baby is awake", say="check on baby", source="crib")
        c = Concern.from_goal(g)
        self.assertIsInstance(c, Concern)
        self.assertEqual(c.id, g.id)
        self.assertEqual(c.condition, g.condition)
        self.assertEqual(c.say, g.say)
        self.assertEqual(c.source, g.source)
        self.assertEqual(c.identity, Subject(kind="entity", ref=g.condition))
        # defaults pinned to today's gate behavior (watch / on_false / mild)
        self.assertEqual(c.consequence, Consequence())
        self.assertEqual(c.tracking, TrackingBehavior())
        self.assertEqual(c.observations, [])

    def test_from_goal_independent_observation_lists(self):
        a = Concern.from_goal(Goal(id="a", condition="x", say="y"))
        b = Concern.from_goal(Goal(id="b", condition="x", say="y"))
        a.observations.append(Evidence(t=0.0, source_id="s", confidence=1.0, satisfied=True))
        self.assertEqual(b.observations, [])  # default_factory, not shared

    def test_clip_declaration_defaults_off(self):
        # B1: the clip/window_s fields default off, so an un-declared concern is single-frame unchanged.
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        self.assertFalse(c.clip)
        self.assertIsNone(c.window_s)
        self.assertFalse(c.wants_clip)

    def test_wants_clip_reflects_declaration(self):
        ident = Subject(kind="entity", ref="x")
        self.assertTrue(Concern(id="a", identity=ident, condition="c", say="s", clip=True).wants_clip)
        self.assertTrue(Concern(id="b", identity=ident, condition="c", say="s", window_s=4.0).wants_clip)
        self.assertFalse(Concern(id="c", identity=ident, condition="c", say="s", window_s=0.0).wants_clip)


class TestContract130Additions(unittest.TestCase):
    """1.3 additive fields: every one defaults inert, so a 1.2-shaped construction is unchanged."""

    def test_verdict_usage_defaults_empty(self):
        v = Verdict(satisfied=True, confidence=0.8)
        self.assertEqual(v.usage, {})
        self.assertEqual(v.events, [])

    def test_verdict_usage_independent_dicts(self):
        a = Verdict(satisfied=True, confidence=0.5)
        b = Verdict(satisfied=True, confidence=0.5)
        a.usage["input_tokens"] = 100
        self.assertEqual(b.usage, {})  # default_factory, not shared

    def test_verdict_carries_usage_and_events(self):
        v = Verdict(
            satisfied=True,
            confidence=0.9,
            usage={"input_tokens": 258, "output_tokens": 40, "cost_usd": 0.002},
            events=[ClipEvent(t_start=4.0, t_end=7.5, label="item concealed in bag")],
        )
        self.assertEqual(v.usage["input_tokens"], 258)
        self.assertEqual(v.events[0].label, "item concealed in bag")

    def test_clip_event_defaults_and_frozen(self):
        e = ClipEvent(t_start=2.0)
        self.assertIsNone(e.t_end)  # instantaneous event
        self.assertEqual(e.label, "")
        with self.assertRaises(Exception):
            e.t_start = 3.0  # frozen

    def test_concern_clip_knobs_default_inert(self):
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        self.assertIsNone(c.fps)
        self.assertFalse(c.verify_only)
        self.assertEqual(c.version, 1)

    def test_concern_clip_knobs_declared(self):
        ident = Subject(kind="entity", ref="x")
        c = Concern(
            id="a", identity=ident, condition="c", say="s",
            clip=True, window_s=22.0, fps=1.0, verify_only=True, version=3,
        )
        self.assertTrue(c.wants_clip)
        self.assertEqual(c.fps, 1.0)
        self.assertTrue(c.verify_only)
        self.assertEqual(c.version, 3)

    def test_tier0signal_energy_default_none(self):
        # a 1.2-shaped positional construction is unchanged; None != 0.0 (measured silence)
        s = Tier0Signal(0.5, 0.0, 0.0, 123.0)
        self.assertIsNone(s.energy)
        s2 = Tier0Signal(0.0, 0.0, 0.0, 124.0, energy=0.0)
        self.assertEqual(s2.energy, 0.0)


class TestIncident(unittest.TestCase):
    def test_incident_minimal_defaults(self):
        v = Verdict(satisfied=True, confidence=0.9, evidence="baby eyes open")
        inc = Incident(id="inc-1", concern_id="c-baby-awake", verdict=v, ts=123.0)
        self.assertIsNone(inc.label)
        self.assertIsNone(inc.snippet)
        self.assertIs(inc.verdict, v)

    def test_incident_with_clip_snippet(self):
        # the motivating example: baby-monitor "baby awake" fires with a short clip
        snip = IncidentSnippet(
            kind="clip",
            media=b"\x00\x01fakeMP4",
            mime="video/mp4",
            t_start=10.0,
            t_end=13.0,
        )
        inc = Incident(
            id="inc-2",
            concern_id="c-baby-awake",
            verdict=Verdict(satisfied=True, confidence=0.95),
            ts=200.0,
            label="real",
            snippet=snip,
        )
        self.assertEqual(inc.snippet.kind, "clip")
        self.assertEqual(inc.snippet.mime, "video/mp4")
        self.assertEqual(inc.label, "real")

    def test_incident_snippet_frame_and_audio(self):
        frame = IncidentSnippet(kind="frame", frames=[object()])
        audio = IncidentSnippet(kind="audio", media=b"RIFF", mime="audio/wav", t_start=1.0, t_end=2.0)
        self.assertEqual(frame.kind, "frame")
        self.assertIsNone(frame.media)
        self.assertEqual(audio.kind, "audio")

    def test_redaction_clears_media_keeps_record(self):
        # privacy redaction: null the media in place, keep kind/mime/timestamps + the Incident.
        snip = IncidentSnippet(kind="clip", media=b"secret", mime="video/mp4", t_start=1.0, t_end=4.0)
        inc = Incident(id="i", concern_id="c", verdict=Verdict(True, 0.9), ts=1.0, snippet=snip)
        inc.snippet.media = None
        inc.snippet.frames = None
        self.assertIsNone(inc.snippet.media)
        self.assertEqual(inc.snippet.kind, "clip")
        self.assertEqual(inc.snippet.t_end, 4.0)


class TestLabelRecord(unittest.TestCase):
    """L-1: the flywheel keystone — schema, round-trip, and the media-free egress guarantee."""

    def _rec(self, **over):
        base = dict(
            incident_id="inc-1", verdict_ref="abc123", goal_id="stove", ts=42.0, good=False,
            note="that was steam", tags=["steam", "kitchen"], frame_ref="/home/u/.percept/frames/x.png",
            calibration_version="v3", source="api",
        )
        base.update(over)
        return pc.LabelRecord(**base)

    def test_minimal_defaults(self):
        r = pc.LabelRecord(incident_id="i", verdict_ref="v", goal_id="g", ts=1.0, good=True)
        self.assertIsNone(r.note)
        self.assertEqual(r.tags, [])
        self.assertIsNone(r.frame_ref)
        self.assertIsNone(r.calibration_version)
        self.assertEqual(r.source, "api")

    def test_is_frozen(self):
        r = self._rec()
        with self.assertRaises(Exception):
            r.good = True  # frozen

    def test_source_enum_is_the_four_frozen_values(self):
        # The four producers (BUILD-BOTH rule 1) — do not redesign.
        import typing
        from percept.contracts import LabelRecord

        hints = typing.get_type_hints(LabelRecord)
        self.assertEqual(
            set(typing.get_args(hints["source"])),
            {"api", "voice", "ask_answer", "watch_calibration"},
        )

    def test_tags_independent_default(self):
        a = pc.LabelRecord(incident_id="a", verdict_ref="v", goal_id="g", ts=1.0, good=True)
        b = pc.LabelRecord(incident_id="b", verdict_ref="v", goal_id="g", ts=1.0, good=True)
        a.tags.append("x")
        self.assertEqual(b.tags, [])  # default_factory, not shared

    def test_local_round_trip(self):
        r = self._rec()
        r2 = pc.LabelRecord.from_dict(r.to_local_dict())
        self.assertEqual(r, r2)  # frozen dataclass __eq__ — full field-for-field round-trip

    def test_egress_cannot_emit_frame_ref(self):
        # THE media-free guarantee: the egress serializer structurally cannot carry the local path.
        r = self._rec(frame_ref="/very/secret/local/pixels.png")
        egress = r.to_egress()
        self.assertNotIn("frame_ref", egress)
        self.assertNotIn("frame_ref", pc.LabelRecord.EGRESS_FIELDS)
        # everything ABOVE frame_ref (the metadata head) IS present
        for name in ("incident_id", "verdict_ref", "goal_id", "ts", "good", "note", "tags",
                     "calibration_version", "source"):
            self.assertIn(name, egress)
        # and the secret value appears nowhere in the serialized head
        import json
        self.assertNotIn("secret", json.dumps(egress))

    def test_local_dict_keeps_frame_ref(self):
        # the LOCAL store keeps the tail — that is the point; only egress drops it.
        r = self._rec(frame_ref="/local/x.png")
        self.assertEqual(r.to_local_dict()["frame_ref"], "/local/x.png")

    def test_in_json_schema(self):
        from percept.contracts import jsonschema as js

        defs = js.generate_all()["$defs"]
        self.assertIn("LabelRecord", defs)
        s = defs["LabelRecord"]
        self.assertEqual(s["properties"]["source"], {"enum": ["api", "voice", "ask_answer", "watch_calibration"]})
        self.assertEqual(s["properties"]["tags"], {"type": "array", "items": {"type": "string"}})
        self.assertNotIn("EGRESS_FIELDS", s["properties"])  # ClassVar is not a field
        for name in ("incident_id", "verdict_ref", "goal_id", "ts", "good"):
            self.assertIn(name, s["required"])
        for name in ("note", "tags", "frame_ref", "calibration_version", "source"):
            self.assertNotIn(name, s.get("required", []))


class TestIncidentLabeling(unittest.TestCase):
    """incident.record_label + label_url (the surfacing half of L-1)."""

    def test_label_url_reserved_default_none(self):
        inc = Incident(id="i", concern_id="c", verdict=Verdict(True, 0.9), ts=1.0)
        self.assertIsNone(inc.label_url)  # reserved, wired by paid

    def test_record_label_without_bind_raises(self):
        inc = Incident(id="i", concern_id="c", verdict=Verdict(True, 0.9), ts=1.0)
        with self.assertRaises(RuntimeError):
            inc.record_label(good=False)

    def test_record_label_writes_to_bound_sink(self):
        captured = []
        inc = Incident(id="inc-9", concern_id="stove", verdict=Verdict(True, 0.9), ts=5.0)
        inc.bind_labeler(captured.append, verdict_ref="vref", goal_id="stove", calibration_version="v2")
        rec = inc.record_label(good=False, note="steam", tags=("kitchen",))
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], rec)
        self.assertEqual(rec.incident_id, "inc-9")
        self.assertEqual(rec.verdict_ref, "vref")
        self.assertEqual(rec.goal_id, "stove")
        self.assertEqual(rec.calibration_version, "v2")
        self.assertFalse(rec.good)
        self.assertEqual(rec.tags, ["kitchen"])
        self.assertEqual(rec.source, "api")

    def test_bind_state_is_not_a_dataclass_field(self):
        # the bound sink/ctx must NOT leak into the schema or asdict (private instance attrs only)
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(Incident)}
        self.assertNotIn("_label_sink", field_names)
        self.assertNotIn("_label_ctx", field_names)
        self.assertIn("label_url", field_names)


class TestJudgmentTiming(unittest.TestCase):
    """A5 — the production timestamp contract: two time domains (ADR-001 §5).

    Monotonic domain (ingest_observed_ts .. decision_emitted_ts) is what ALL latency math reads;
    source_media_ts is optional/untrusted and is never touched by the latency helpers.
    """

    def _clock(self, times):
        """A tiny fake clock: pops the next value off ``times`` on each call."""
        it = iter(times)
        return lambda: next(it)

    def test_builder_freeze_stamps_decision_emitted_and_freezes(self):
        clock = self._clock([10.0, 11.0, 12.0, 14.0, 15.0])
        b = JudgmentTimingBuilder(clock=clock)
        b.mark_ingest_observed()
        b.mark_admitted()
        b.mark_judge_started()
        b.mark_judge_completed()
        timing = b.freeze()

        self.assertIsInstance(timing, JudgmentTiming)
        self.assertEqual(timing.ingest_observed_ts, 10.0)
        self.assertEqual(timing.admitted_ts, 11.0)
        self.assertEqual(timing.judge_started_ts, 12.0)
        self.assertEqual(timing.judge_completed_ts, 14.0)
        self.assertEqual(timing.decision_emitted_ts, 15.0)  # freeze() itself calls the clock

        # frozen — no mutation after decision_emitted_ts
        with self.assertRaises(Exception):
            timing.decision_emitted_ts = 99.0

    def test_freeze_raises_when_a_stage_was_never_marked(self):
        b = JudgmentTimingBuilder(clock=self._clock([1.0]))
        b.mark_ingest_observed()
        # admitted_ts / judge_started_ts / judge_completed_ts never marked
        with self.assertRaises(ValueError):
            b.freeze()

    def test_monotonic_stamps_must_be_non_decreasing(self):
        # A later funnel stage stamped BEFORE an earlier one must raise — this is the invariant
        # that keeps every derived latency non-negative.
        with self.assertRaises(ValueError):
            JudgmentTiming(
                ingest_observed_ts=10.0,
                admitted_ts=9.0,  # decreasing vs ingest_observed_ts -> invalid
                judge_started_ts=11.0,
                judge_completed_ts=12.0,
                decision_emitted_ts=13.0,
            )
        with self.assertRaises(ValueError):
            JudgmentTiming(
                ingest_observed_ts=10.0,
                admitted_ts=11.0,
                judge_started_ts=12.0,
                judge_completed_ts=13.0,
                decision_emitted_ts=12.5,  # decreasing vs judge_completed_ts -> invalid
            )
        # equal (non-decreasing, not strictly increasing) stamps are valid
        flat = JudgmentTiming(
            ingest_observed_ts=5.0,
            admitted_ts=5.0,
            judge_started_ts=5.0,
            judge_completed_ts=5.0,
            decision_emitted_ts=5.0,
        )
        self.assertEqual(flat.admitted_to_decision_s(), 0.0)

    def test_source_media_ts_optional_and_excluded_from_ordering_check(self):
        # source_media_ts may be None (the common/untrusted case)...
        timing = JudgmentTiming(
            ingest_observed_ts=1.0,
            admitted_ts=2.0,
            judge_started_ts=3.0,
            judge_completed_ts=4.0,
            decision_emitted_ts=5.0,
        )
        self.assertIsNone(timing.source_media_ts)

        # ...and when present it can be on a totally different (even "earlier"/skewed) timeline
        # without tripping the monotonic-domain invariant — it is excluded from that check.
        skewed = JudgmentTiming(
            ingest_observed_ts=100.0,
            admitted_ts=101.0,
            judge_started_ts=102.0,
            judge_completed_ts=103.0,
            decision_emitted_ts=104.0,
            source_media_ts=0.5,  # e.g. a fixture/RTSP-PTS clock that starts near zero
        )
        self.assertEqual(skewed.source_media_ts, 0.5)

    def test_latency_helpers_use_only_the_monotonic_domain(self):
        timing = JudgmentTiming(
            ingest_observed_ts=100.0,
            admitted_ts=101.0,
            judge_started_ts=103.0,
            judge_completed_ts=106.0,
            decision_emitted_ts=107.0,
            source_media_ts=999999.0,  # wildly different value — must not leak into any helper
        )
        self.assertEqual(timing.ingest_to_decision_s(), 7.0)
        self.assertEqual(timing.admitted_to_decision_s(), 6.0)
        self.assertEqual(timing.judge_wait_s(), 2.0)
        self.assertEqual(timing.judge_duration_s(), 3.0)
        self.assertEqual(timing.decision_overhead_s(), 1.0)
        # none of the helpers reference source_media_ts at all — sanity: changing it changes nothing
        other = JudgmentTiming(
            ingest_observed_ts=100.0,
            admitted_ts=101.0,
            judge_started_ts=103.0,
            judge_completed_ts=106.0,
            decision_emitted_ts=107.0,
            source_media_ts=None,
        )
        self.assertEqual(timing.ingest_to_decision_s(), other.ingest_to_decision_s())
        self.assertEqual(timing.admitted_to_decision_s(), other.admitted_to_decision_s())

    def test_judgment_event_carries_timing_and_a8_fields_no_delivery_stamps(self):
        clock = self._clock([1.0, 2.0, 3.0, 4.0, 5.0])
        b = JudgmentTimingBuilder(clock=clock)
        b.mark_ingest_observed()
        b.mark_admitted()
        b.mark_judge_started()
        b.mark_judge_completed()
        timing = b.freeze()

        event = JudgmentEvent(key="goal-1:entity-7", evidence_seq=3, action="fire", timing=timing)

        # A8's fields survive, additively, on the frozen event.
        self.assertEqual(event.key, "goal-1:entity-7")
        self.assertEqual(event.evidence_seq, 3)
        self.assertEqual(event.action, "fire")
        self.assertIs(event.timing, timing)

        # The event itself is frozen — no post-construction mutation.
        with self.assertRaises(Exception):
            event.action = "ask"

        # No delivery timestamp of any kind lives on the event or its timing — those belong to the
        # DeliveryAttempt ledger (A21/B-SINK), never here.
        import dataclasses

        event_fields = {f.name for f in dataclasses.fields(event)}
        timing_fields = {f.name for f in dataclasses.fields(timing)}
        for forbidden in ("delivery_ts", "delivered_ts", "delivery_attempt", "sink_ts", "attempt_ts"):
            self.assertNotIn(forbidden, event_fields)
            self.assertNotIn(forbidden, timing_fields)


class TestGroundTruth(unittest.TestCase):
    """A6 — the evaluation-only ground-truth record (Viability §3b/§5a; ADR-001 §5).

    ``GroundTruth`` is a DISTINCT type from ``JudgmentTiming``/``JudgmentEvent``: a live event built
    from real production traffic must never carry ``event_start_ts``/``decision_eligible_ts`` —
    structurally, not just by convention.
    """

    def test_is_a_separate_type_from_judgment_timing_and_event(self):
        import dataclasses

        self.assertIsNot(GroundTruth, JudgmentTiming)
        self.assertIsNot(GroundTruth, JudgmentEvent)
        self.assertNotIsInstance(GroundTruth(0.0, 0.0, "v1", "fixture-1"), JudgmentTiming)

        gt_fields = {f.name for f in dataclasses.fields(GroundTruth)}
        timing_fields = {f.name for f in dataclasses.fields(JudgmentTiming)}
        self.assertTrue(gt_fields.isdisjoint(timing_fields))

    def test_live_judgment_event_and_timing_carry_no_ground_truth_fields(self):
        import dataclasses

        timing = JudgmentTiming(
            ingest_observed_ts=1.0,
            admitted_ts=2.0,
            judge_started_ts=3.0,
            judge_completed_ts=4.0,
            decision_emitted_ts=5.0,
        )
        event = JudgmentEvent(key="k", evidence_seq=1, action="fire", timing=timing)

        event_fields = {f.name for f in dataclasses.fields(event)}
        timing_fields = {f.name for f in dataclasses.fields(timing)}
        for forbidden in (
            "event_start_ts",
            "decision_eligible_ts",
            "annotation_version",
            "annotator_or_fixture_id",
            "ground_truth",
        ):
            self.assertNotIn(forbidden, event_fields)
            self.assertNotIn(forbidden, timing_fields)

    def test_by_construction_is_deterministic(self):
        a = GroundTruth.by_construction(
            event_start_ts=10.0, eligible_after_s=0.5, fixture_id="state-archetype-001"
        )
        b = GroundTruth.by_construction(
            event_start_ts=10.0, eligible_after_s=0.5, fixture_id="state-archetype-001"
        )
        self.assertEqual(a, b)
        self.assertEqual(a.decision_eligible_ts, 10.5)
        self.assertEqual(a.annotation_version, ANNOTATION_RUBRIC_VERSION)

    def test_by_construction_rejects_negative_offset(self):
        with self.assertRaises(ValueError):
            GroundTruth.by_construction(event_start_ts=0.0, eligible_after_s=-1.0, fixture_id="f")

    def test_annotation_version_is_required(self):
        with self.assertRaises(ValueError):
            GroundTruth(
                event_start_ts=0.0,
                decision_eligible_ts=0.0,
                annotation_version="",
                annotator_or_fixture_id="fixture-1",
            )

    def test_annotator_or_fixture_id_is_required(self):
        with self.assertRaises(ValueError):
            GroundTruth(
                event_start_ts=0.0,
                decision_eligible_ts=0.0,
                annotation_version=ANNOTATION_RUBRIC_VERSION,
                annotator_or_fixture_id="",
            )

    def test_decision_eligible_ts_cannot_precede_event_start_ts(self):
        with self.assertRaises(ValueError):
            GroundTruth(
                event_start_ts=10.0,
                decision_eligible_ts=9.0,
                annotation_version=ANNOTATION_RUBRIC_VERSION,
                annotator_or_fixture_id="fixture-1",
            )

    def test_record_is_frozen(self):
        gt = GroundTruth.by_construction(event_start_ts=0.0, eligible_after_s=1.0, fixture_id="f")
        with self.assertRaises(Exception):
            gt.decision_eligible_ts = 99.0

    def test_from_two_annotator_reconciles_within_tolerance(self):
        gt = GroundTruth.from_two_annotator(
            event_start_ts=5.0,
            annotator_a_id="alice",
            annotator_a_eligible_ts=6.0,
            annotator_b_id="bob",
            annotator_b_eligible_ts=6.4,
            max_disagreement_s=1.0,
        )
        self.assertAlmostEqual(gt.decision_eligible_ts, 6.2)
        self.assertEqual(gt.annotator_or_fixture_id, "alice+bob")

    def test_from_two_annotator_raises_on_disagreement(self):
        with self.assertRaises(ValueError):
            GroundTruth.from_two_annotator(
                event_start_ts=5.0,
                annotator_a_id="alice",
                annotator_a_eligible_ts=6.0,
                annotator_b_id="bob",
                annotator_b_eligible_ts=9.0,
                max_disagreement_s=1.0,
            )

    def test_annotation_rubric_carries_the_version(self):
        self.assertIn(ANNOTATION_RUBRIC_VERSION, ANNOTATION_RUBRIC)
        self.assertIn("decision_eligible_ts", ANNOTATION_RUBRIC)
class TestWatchEvent(unittest.TestCase):
    """M1 — the event ``percept.watch()`` streams: defaults, all three actions, not frozen."""

    def test_minimal_defaults(self):
        ev = WatchEvent(goal_id="zone", action="fire", t=1.0)
        self.assertEqual(ev.text, "")
        self.assertIsNone(ev.entity_id)
        self.assertIsNone(ev.verdict)

    def test_carries_verdict_and_text(self):
        v = Verdict(satisfied=True, confidence=0.9, evidence="in zone")
        ev = WatchEvent(goal_id="zone", action="fire", t=2.0, text="Alert!", entity_id="e1", verdict=v)
        self.assertIs(ev.verdict, v)
        self.assertEqual(ev.entity_id, "e1")
        self.assertEqual(ev.text, "Alert!")

    def test_accepts_all_three_actions(self):
        for action in ("fire", "ask", "silent"):
            ev = WatchEvent(goal_id="g", action=action, t=0.0)
            self.assertEqual(ev.action, action)

    def test_not_frozen(self):
        # mutable, like Verdict/Incident/FireEvent (the other firing-path DTOs) — not Box/Schedule.
        ev = WatchEvent(goal_id="g", action="ask", t=0.0)
        ev.text = "updated"
        self.assertEqual(ev.text, "updated")


class TestAudioFrame(unittest.TestCase):
    """B2 — the typed first-class audio signal: defaults, render-to-pair, duration, frozen."""

    def test_defaults(self):
        af = AudioFrame()
        self.assertEqual(af.pcm, b"")
        self.assertEqual(af.sample_rate, 16000)
        self.assertEqual(af.channels, 1)
        self.assertEqual(af.sample_format, "s16le")
        self.assertIsNone(af.encoded)
        self.assertIsNone(af.mime)

    def test_as_pair_raw_pcm(self):
        af = AudioFrame(pcm=b"\x01\x02", sample_rate=8000, channels=2)
        self.assertEqual(af.as_pair(), ("audio/L16;rate=8000;channels=2", b"\x01\x02"))

    def test_as_pair_prefers_encoded(self):
        af = AudioFrame(pcm=b"raw", encoded=b"RIFF", mime="audio/wav")
        self.assertEqual(af.as_pair(), ("audio/wav", b"RIFF"))

    def test_duration(self):
        self.assertEqual(AudioFrame(t_start=2.0, t_end=5.0).duration_s, 3.0)
        self.assertIsNone(AudioFrame().duration_s)

    def test_frozen(self):
        af = AudioFrame()
        with self.assertRaises(Exception):
            af.sample_rate = 8000


class TestVersionNegotiation(unittest.TestCase):
    def test_contract_version_is_semver(self):
        # 1.9.0: additive evidence-link correlation — judgment_event_id() + EvidenceOrigin, plus
        # JudgmentEvent.evidence_origin and Incident.judgment_event_id/Incident.evidence_origin (all
        # additive, defaulted, gate-irrelevant; the transition path now emits a judgment record too).
        # 1.8.0: additive WatchPlan/SettingOrigin/ResolvedWatch/WatchReceipt/FieldDiagnostic DTOs
        # (percept.contracts.watchplan, ADR-008 configuration ownership/resolution/provenance), plus
        # additive Incident.revision_id/Incident.supersedes (the episode-owned provisional/superseding
        # link put on the wire), same batch.
        # 1.7.0: additive Incident.evidence_seq (the JudgmentEvent evidence_seq copied through to the
        # delivered record) + additive SubjectRef 'provisional' resolution state (UN_<episode_id>),
        # same batch. 1.6.0: additive judgment-envelope enrichment — JudgmentEvent + Incident gain
        # reason/watchplan_id/roi/modality/provenance (Z3). 1.5.0: Verdict claim-mode channel +
        # ConcernAuthority (Y2); 1.4.0 SubjectRef + Verdict.subject (Y5); 1.3.0 Incident.entity_id (D4);
        # 1.2.0 WatchAction/WatchEvent (M1); 1.1.0 AttentionIntent/GoalOrigin (PCPT-026).
        self.assertEqual(CONTRACT_VERSION, "1.10.0")
        parts = CONTRACT_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.isdigit() for p in parts))

    def test_negotiate_compatible_same_version(self):
        self.assertTrue(negotiate(CONTRACT_VERSION, CONTRACT_VERSION))

    def test_negotiate_future_minor_still_compatible(self):
        # 1.0.0 is the launch contract; a future additive 1.x minor negotiates cleanly (same MAJOR).
        self.assertTrue(negotiate(CONTRACT_VERSION, "1.9.0"))
        self.assertTrue(negotiate("1.9.0", CONTRACT_VERSION))

    def test_negotiate_compatible_minor_patch_differences(self):
        self.assertTrue(negotiate("1.0.0", "1.4.2"))
        self.assertTrue(negotiate("1.9.9", "1.0.0"))

    def test_negotiate_raises_on_major_mismatch(self):
        with self.assertRaises(ContractVersionMismatch):
            negotiate("1.0.0", "2.0.0")
        with self.assertRaises(ContractVersionMismatch):
            negotiate("3.1.0", "1.0.0")


class TestJsonSchema(unittest.TestCase):
    def test_generate_all_has_defs_for_every_dto(self):
        from percept.contracts import jsonschema as js

        doc = js.generate_all()
        self.assertEqual(doc["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(doc["x-contract-version"], CONTRACT_VERSION)
        for cls in js.DTOS:
            self.assertIn(cls.__name__, doc["$defs"], f"missing $defs for {cls.__name__}")

    def test_verdict_schema_required_and_types(self):
        from percept.contracts import jsonschema as js

        s = js.generate_all()["$defs"]["Verdict"]
        self.assertEqual(s["type"], "object")
        self.assertEqual(s["properties"]["satisfied"], {"type": "boolean"})
        self.assertEqual(s["properties"]["confidence"], {"type": "number"})
        # evidence has a default -> not required; satisfied/confidence are required
        self.assertIn("satisfied", s["required"])
        self.assertIn("confidence", s["required"])
        self.assertNotIn("evidence", s["required"])

    def test_incident_schema_refs_snippet_and_verdict(self):
        from percept.contracts import jsonschema as js

        s = js.generate_all()["$defs"]["Incident"]
        self.assertEqual(s["properties"]["verdict"], {"$ref": "#/$defs/Verdict"})
        # snippet is Optional[IncidentSnippet] -> anyOf containing the $ref + null
        any_of = s["properties"]["snippet"]["anyOf"]
        self.assertIn({"$ref": "#/$defs/IncidentSnippet"}, any_of)
        self.assertIn({"type": "null"}, any_of)

    def test_incident_schema_carries_optional_entity_id(self):
        # D4 / contract 1.3: additive, optional, nullable — never `required`.
        from percept.contracts import jsonschema as js

        s = js.generate_all()["$defs"]["Incident"]
        any_of = s["properties"]["entity_id"]["anyOf"]
        self.assertIn({"type": "string"}, any_of)
        self.assertIn({"type": "null"}, any_of)
        self.assertNotIn("entity_id", s.get("required", []))

    def test_literal_becomes_enum(self):
        from percept.contracts import jsonschema as js

        snip = js.generate_all()["$defs"]["IncidentSnippet"]
        self.assertEqual(snip["properties"]["kind"], {"enum": ["frame", "clip", "audio"]})

    def test_schedule_in_defs_and_watchcondition_optional(self):
        from percept.contracts import jsonschema as js

        defs = js.generate_all()["$defs"]
        self.assertIn("Schedule", defs)  # the C1 schedule DTO is exported
        wc = defs["WatchCondition"]
        # schedule is Optional[Schedule] -> anyOf containing the $ref + null, and NOT required
        any_of = wc["properties"]["schedule"]["anyOf"]
        self.assertIn({"$ref": "#/$defs/Schedule"}, any_of)
        self.assertIn({"type": "null"}, any_of)
        self.assertNotIn("schedule", wc.get("required", []))

    def test_audioframe_schema(self):
        from percept.contracts import jsonschema as js

        s = js.generate_all()["$defs"]["AudioFrame"]
        self.assertEqual(s["type"], "object")
        # pcm is base64-encoded bytes; sample_format Literal -> enum; all fields default -> none required
        self.assertEqual(s["properties"]["pcm"], {"type": "string", "contentEncoding": "base64"})
        self.assertEqual(s["properties"]["sample_format"], {"enum": ["s16le", "f32le", "u8"]})
        self.assertEqual(s["properties"]["sample_rate"], {"type": "integer"})
        self.assertNotIn("required", s)  # every field has a default

    def test_130_fields_in_schema(self):
        from percept.contracts import jsonschema as js

        defs = js.generate_all()["$defs"]
        self.assertIn("ClipEvent", defs)
        v = defs["Verdict"]
        self.assertEqual(v["properties"]["usage"], {"type": "object", "additionalProperties": {}})
        self.assertEqual(v["properties"]["events"], {"type": "array", "items": {"$ref": "#/$defs/ClipEvent"}})
        self.assertNotIn("usage", v.get("required", []))
        self.assertNotIn("events", v.get("required", []))
        t = defs["Tier0Signal"]
        self.assertIn({"type": "null"}, t["properties"]["energy"]["anyOf"])
        self.assertNotIn("energy", t.get("required", []))
        c = defs["Concern"]
        self.assertEqual(c["properties"]["verify_only"], {"type": "boolean"})
        self.assertEqual(c["properties"]["version"], {"type": "integer"})
        for name in ("fps", "verify_only", "version"):
            self.assertNotIn(name, c.get("required", []))
    def test_watchevent_schema_refs_verdict_and_action_enum(self):
        from percept.contracts import jsonschema as js

        s = js.generate_all()["$defs"]["WatchEvent"]
        self.assertEqual(s["properties"]["action"], {"enum": ["fire", "ask", "silent"]})
        any_of = s["properties"]["verdict"]["anyOf"]
        self.assertIn({"$ref": "#/$defs/Verdict"}, any_of)
        self.assertIn({"type": "null"}, any_of)
        self.assertIn("goal_id", s["required"])
        self.assertIn("action", s["required"])
        self.assertIn("t", s["required"])
        self.assertNotIn("text", s["required"])

    def test_committed_schema_matches_generator(self):
        # the committed files under schema/ must be byte-identical to a fresh generation
        import json
        from pathlib import Path

        from percept.contracts import jsonschema as js

        combined_path = js.schema_dir() / "percept-contracts.schema.json"
        self.assertTrue(combined_path.exists(), "run `python -m percept.contracts.jsonschema`")
        on_disk = json.loads(Path(combined_path).read_text())
        self.assertEqual(on_disk, js.generate_all())


class TestPublicSurface(unittest.TestCase):
    def test_all_names_importable(self):
        for name in pc.__all__:
            self.assertTrue(hasattr(pc, name), f"{name} missing from percept.contracts")


if __name__ == "__main__":
    unittest.main()
