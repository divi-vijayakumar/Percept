"""L1: percept-contracts — DTO defaults, Concern.from_goal, Incident, version negotiation,
and JSON-schema generation. Stdlib only, no backends, no keys."""

import unittest

import percept.contracts as pc
from percept.contracts import (
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
    Incident,
    IncidentSnippet,
    Schedule,
    Subject,
    Tier0Signal,
    TrackingBehavior,
    Verdict,
    WatchCondition,
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
        self.assertEqual(CONTRACT_VERSION, "1.3.0")
        parts = CONTRACT_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.isdigit() for p in parts))

    def test_negotiate_compatible_same_version(self):
        self.assertTrue(negotiate(CONTRACT_VERSION, CONTRACT_VERSION))

    def test_negotiate_prior_minor_still_compatible(self):
        # a 1.2.0 consumer (pre-1.3 additive fields) negotiates cleanly with this release
        self.assertTrue(negotiate("1.2.0", CONTRACT_VERSION))
        self.assertTrue(negotiate(CONTRACT_VERSION, "1.2.0"))

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
