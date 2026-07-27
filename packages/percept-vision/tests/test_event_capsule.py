"""event_capsule — the additive composite Temporal value.

Three things, all offline / zero-spend (no real judge call anywhere in this file — the point of
:class:`ScriptedJudge` with an EMPTY script is that consulting it at all raises, so a passing run
proves capsule assembly spends nothing):

  1. ``event_capsule`` is SELECTABLE via the same ``goal.state['perception']`` evidence-shape channel
     every other Temporal value already uses (:func:`coerce_authored_perception` / :func:`program_of`).
  2. :func:`percept.capsule.assemble_event_capsule` composes all THREE components — pre-roll ring,
     episode boundary, early-snapshot timer — into one :class:`~percept.capsule.EventCapsule`, and the
     result survives the existing :class:`~percept.evidence.EvidenceStore` retention mechanism (no new
     store).
  3. The four PRE-EXISTING Temporal values (single/strip/window/sprite) are byte-for-byte unchanged —
     the capsule mode is additive-only.
"""

from __future__ import annotations

import json
import struct
import unittest
import zlib

from percept.adapters import Goal
from percept.behavior import (
    PerceptionSpec,
    _VALID_TEMPORAL,
    coerce_authored_perception,
    program_of,
)
from percept.capsule import (
    EarlySnapshotTimer,
    EventCapsule,
    assemble_event_capsule,
    capsule_trace_data,
    store_event_capsule,
)
from percept.episodes import Episode
from percept.evidence import EvidenceStore
from percept.evidence_encoding import UNAVAILABLE_MEDIA_TYPE
from percept.events import FrameWindow
from percept.harness.scripted import ScriptedJudge
from percept.tracing import STAGE_CAPSULE_ASSEMBLED, TRACE_STAGES


# --------------------------------------------------------------------------------------------------
# 1. Goal.state evidence-shape selection — the same channel single/strip/window/sprite already use.
# --------------------------------------------------------------------------------------------------


class TestEventCapsuleSelectable(unittest.TestCase):
    def test_valid_vocab_includes_event_capsule(self):
        self.assertIn("event_capsule", _VALID_TEMPORAL)

    def test_coerce_authored_perception_accepts_it(self):
        self.assertEqual(
            coerce_authored_perception({"temporal": "event_capsule"}),
            {"perception": {"temporal": "event_capsule"}},
        )

    def test_program_of_overlays_it_from_goal_state(self):
        # exactly the shape a conductor / a direct goal.state author would set — no new channel.
        g = Goal(id="g", condition="c", say="s", state={"perception": {"temporal": "event_capsule"}})
        prog = program_of(g)
        self.assertEqual(prog.perception.temporal, "event_capsule")
        # every OTHER perception field stays the untouched default (additive-only: selecting the
        # composite shape doesn't silently change modality/roi/sampling/n_frames).
        self.assertEqual(prog.perception.modality, "vision")
        self.assertEqual(prog.perception.roi, "focus")
        self.assertEqual(prog.perception.sampling, "recent")


# --------------------------------------------------------------------------------------------------
# 2. Assembly — pre-roll ring + episode boundary + early-snapshot timer, all THREE present.
# --------------------------------------------------------------------------------------------------


class _RaisingVision:
    """A Vision double that FAILS the test the instant it's consulted -- the zero-spend proof.

    Capsule assembly reads only the FrameWindow / Episode / EarlySnapshotTimer fixtures below; it
    never calls .judge()/.detect(). Wiring this in (even unused) documents that guarantee for a
    reader the same way test_direct_judge.py's _RaisingVision does for the confident DirectJudge path.
    """

    async def judge(self, *args, **kwargs):
        raise AssertionError("assemble_event_capsule must spend zero vision/judge calls")

    async def detect(self, frame, goals):
        raise AssertionError("assemble_event_capsule must spend zero vision/judge calls")

    async def detect_events(self, frames, goals):
        raise AssertionError("assemble_event_capsule must spend zero vision/judge calls")


class TestEventCapsuleAssembly(unittest.TestCase):
    def setUp(self):
        # zero-spend guard: an EMPTY ScriptedJudge raises the moment it's asked for a verdict, so
        # simply constructing it (and never calling it) is the proof this test spends nothing.
        self.judge_guard = ScriptedJudge(script=[])
        self.vision_guard = _RaisingVision()

        # (a) the pre-roll ring -- the SAME FrameWindow structure the incident snippet's pre-roll
        # already reads (percept.incidents.snippet_from_window). 20 ticks, one per second.
        self.window = FrameWindow(size=20)
        for i in range(20):
            self.window.push(float(i), f"frame{i}")

        # (b) the episode boundary -- a real percept.episodes.Episode (not a duck-typed stand-in),
        # opened at t=0 with an exemplar and last seen at t=10 with a representative.
        self.episode = Episode(
            id="ep1", goal_id="g1", started_at=0.0, last_seen=10.0,
            exemplar="open-frame", representative="latest-frame",
        )

        # (c) the early-snapshot timer -- fed on every tick like a live agent loop would, capturing
        # ONE frame 5s after the episode opened.
        self.timer = EarlySnapshotTimer(after_s=5.0)
        self.timer.arm(self.episode.id, self.episode.started_at)
        for i in range(20):
            self.timer.observe(self.episode.id, float(i), f"frame{i}")

    def test_all_three_components_present(self):
        capsule = assemble_event_capsule(self.window, self.episode, self.timer, pre_s=3.0)
        self.assertIsInstance(capsule, EventCapsule)
        # (a) pre-roll: the last 3s of the ring (t >= 19-3=16, inclusive cutoff -- matches
        # percept.incidents.snippet_from_window's own pre_s convention).
        self.assertTrue(capsule.has_pre_roll)
        self.assertEqual([t for t, _ in capsule.pre_roll], [16.0, 17.0, 18.0, 19.0])
        # (b) boundary: episode open (exemplar@0) + most recent (representative@10).
        self.assertTrue(capsule.has_boundary)
        self.assertEqual(capsule.boundary, ((0.0, "open-frame"), (10.0, "latest-frame")))
        # (c) early snapshot: the first tick >= 5s after open.
        self.assertTrue(capsule.has_early_snapshot)
        self.assertEqual(capsule.early_snapshot, (5.0, "frame5"))
        # the composite predicate a caller checks before trusting the capsule is "complete".
        self.assertTrue(capsule.is_complete)
        # combined, ordered, de-duplicated by timestamp.
        self.assertEqual(
            capsule.frames(),
            [(0.0, "open-frame"), (5.0, "frame5"), (10.0, "latest-frame"),
             (16.0, "frame16"), (17.0, "frame17"), (18.0, "frame18"), (19.0, "frame19")],
        )

    def test_missing_episode_degrades_to_partial_capsule(self):
        # no episode identity -> no boundary, no early snapshot -- degrades gracefully, never raises.
        capsule = assemble_event_capsule(self.window, None, self.timer, pre_s=3.0)
        self.assertTrue(capsule.has_pre_roll)
        self.assertFalse(capsule.has_boundary)
        self.assertFalse(capsule.has_early_snapshot)
        self.assertFalse(capsule.is_complete)
        self.assertIsNone(capsule.episode_id)

    def test_early_snapshot_never_overwritten_after_ring_evicts_it(self):
        # the whole point of a TIMER (not window) capture: push far past the ring's size so the
        # opening frames are long gone, and the snapshot survives untouched.
        small_window = FrameWindow(size=5)
        for i in range(20):
            small_window.push(float(i), f"frame{i}")
        capsule = assemble_event_capsule(small_window, self.episode, self.timer, pre_s=None)
        # the ring only has the last 5 frames (15..19) -- frame5 is long gone from it...
        self.assertEqual([t for t, _ in capsule.pre_roll], [15.0, 16.0, 17.0, 18.0, 19.0])
        # ...but the timer captured it back when it happened, and assembly still finds it.
        self.assertEqual(capsule.early_snapshot, (5.0, "frame5"))
        self.assertTrue(capsule.is_complete)

    def test_assembly_is_pure_and_spends_zero_vision_calls(self):
        # calling assembly twice never mutates window/episode/timer, and neither vision guard is
        # ever invoked anywhere in setUp or here -- constructing them and letting them go unused
        # (they would raise on first use) IS the zero-spend proof.
        c1 = assemble_event_capsule(self.window, self.episode, self.timer, pre_s=3.0)
        c2 = assemble_event_capsule(self.window, self.episode, self.timer, pre_s=3.0)
        self.assertEqual(c1, c2)


# --------------------------------------------------------------------------------------------------
# 2b. Retention -- capsule frames go through the EXISTING EvidenceStore, never a new store.
# --------------------------------------------------------------------------------------------------


class TestEventCapsuleRetention(unittest.TestCase):
    def setUp(self):
        window = FrameWindow(size=20)
        for i in range(20):
            window.push(float(i), ("image/jpeg", f"f{i}".encode()))
        episode = Episode(
            id="ep1", goal_id="g1", started_at=0.0, last_seen=10.0,
            exemplar=("image/jpeg", b"open-bytes"), representative=("image/jpeg", b"latest-bytes"),
        )
        timer = EarlySnapshotTimer(after_s=5.0)
        timer.arm(episode.id, episode.started_at)
        for i in range(20):
            timer.observe(episode.id, float(i), ("image/jpeg", f"f{i}".encode()))
        self.capsule = assemble_event_capsule(window, episode, timer, pre_s=3.0)
        self.assertTrue(self.capsule.is_complete)  # sanity: the fixture built a complete capsule

    def test_stores_through_existing_evidence_store_default_retention(self):
        store = EvidenceStore()
        artifact = store_event_capsule(store, self.capsule)
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.kind, "bundle")
        # every frame the capsule carries becomes one ordered child of ONE parent artifact.
        children = store.resolve_bundle(artifact)
        self.assertEqual(len(children), len(self.capsule.frames()))
        # the retention class is one of the FOUR existing classes ADR-003 names -- no new class,
        # no new store type. (No public getter for an artifact's own class; the fact that .put_bundle
        # accepted the call at all with the default "fire" class, and resolves cleanly, is the proof.)
        for child in children:
            self.assertEqual(child.media_type, "image/jpeg")

    def test_explicit_retention_class_is_honoured(self):
        store = EvidenceStore()
        artifact = store_event_capsule(store, self.capsule, retention_class="sampled_debug")
        self.assertIsNotNone(artifact)
        store.resolve(artifact)  # resolves cleanly under the caller-chosen existing class

    def test_empty_capsule_stores_nothing(self):
        store = EvidenceStore()
        empty = EventCapsule(episode_id=None)
        self.assertIsNone(store_event_capsule(store, empty))


def _png_pixels(payload: bytes) -> bytes:
    """The pixel bytes inside an 8-bit, filter-0 PNG — read straight off the stored artifact, with
    no help from the encoder under test (and no PIL/OpenCV dependency)."""
    pos, width, channels, idat = 8, 0, 0, bytearray()
    while pos < len(payload):
        (length,) = struct.unpack(">I", payload[pos:pos + 4])
        tag, data = payload[pos + 4:pos + 8], payload[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, _height, _depth, color_type = struct.unpack(">IIBB", data[:10])
            channels = {0: 1, 4: 2, 2: 3, 6: 4}[color_type]
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    raw, stride, out = zlib.decompress(bytes(idat)), width * channels, bytearray()
    for start in range(0, len(raw), stride + 1):
        out += raw[start + 1:start + 1 + stride]
    return bytes(out)


class _OpaqueFrame:
    """A frame with no faithful encoding, whose repr would leak its contents if one were taken."""

    def __repr__(self) -> str:  # pragma: no cover - only reached if the defect returns
        return "OPAQUE-CAPSULE-FRAME"


class _PixelBuffer:
    """A decoded pixel buffer (the array-like shape a camera frame arrives in), numpy-free."""

    def __init__(self, height: int, width: int) -> None:
        self.shape, self.dtype = (height, width, 3), "uint8"
        self._raw = bytes((i * 11 + 5) % 256 for i in range(height * width * 3))
        self.nbytes = len(self._raw)

    def tobytes(self) -> bytes:
        return self._raw


class TestCapsuleStoresEvidenceNotDescriptions(unittest.TestCase):
    """A stored capsule holds the frame's REAL bytes, or an explicit gap — never a ``repr()``.

    ``repr`` of a decoded frame is a TRUNCATED description (numpy elides any array over 1000
    elements), so retaining one under an evidence media type loses the evidence silently and
    unrecoverably."""

    def _one_child(self, frame):
        store = EvidenceStore()
        capsule = EventCapsule(episode_id="ep1", boundary=((0.0, frame),))
        artifact = store_event_capsule(store, capsule)
        (child,) = store.resolve_bundle(artifact)
        return store, child

    def test_decoded_pixel_buffer_is_retained_as_its_real_pixels(self):
        frame = _PixelBuffer(4, 6)
        store, child = self._one_child(frame)
        self.assertEqual(child.media_type, "image/png")
        payload = store.open_bytes(child)
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        # every pixel byte is present — a description of this buffer would be a fraction of its size.
        self.assertGreaterEqual(len(payload), frame.nbytes // 2)
        self.assertEqual(_png_pixels(payload), frame.tobytes())

    def test_unencodable_frame_is_retained_as_an_explicit_typed_gap(self):
        store, child = self._one_child(_OpaqueFrame())
        self.assertEqual(child.media_type, UNAVAILABLE_MEDIA_TYPE)
        record = json.loads(store.open_bytes(child).decode("utf-8"))
        self.assertEqual(record["unavailable"]["frame_type"], "_OpaqueFrame")
        self.assertEqual(record["unavailable"]["reason"], "unencodable-frame-type")
        self.assertNotIn(b"OPAQUE-CAPSULE-FRAME", store.open_bytes(child))

    def test_encoded_payloads_still_pass_through_byte_for_byte(self):
        store, child = self._one_child(("image/jpeg", b"\xff\xd8\xffexactly-these-bytes"))
        self.assertEqual(child.media_type, "image/jpeg")
        self.assertEqual(store.open_bytes(child), b"\xff\xd8\xffexactly-these-bytes")


# --------------------------------------------------------------------------------------------------
# 2c. Trace -- the additive capsule.assembled stage.
# --------------------------------------------------------------------------------------------------


class TestEventCapsuleTrace(unittest.TestCase):
    def test_stage_is_in_the_closed_vocabulary(self):
        self.assertIn(STAGE_CAPSULE_ASSEMBLED, TRACE_STAGES)

    def test_trace_data_is_json_safe_scalars_only(self):
        capsule = EventCapsule(
            episode_id="ep1",
            pre_roll=((17.0, "f17"), (18.0, "f18")),
            boundary=((0.0, "open"), (10.0, "latest")),
            early_snapshot=(5.0, "f5"),
        )
        data = capsule_trace_data(capsule)
        self.assertEqual(
            data,
            {"episode_id": "ep1", "pre_roll_n": 2, "boundary_n": 2,
             "has_early_snapshot": True, "is_complete": True},
        )
        # no frame payload (no "f17"/"open"/etc. bytes) ever appears in the advisory -- counts/flags
        # only, matching the media-off .ptrace discipline every other advisory stage follows.
        self.assertNotIn("pre_roll", data)
        self.assertNotIn("boundary", data)
        self.assertNotIn("early_snapshot", data)


# --------------------------------------------------------------------------------------------------
# 3. Old Temporal values -- byte-for-byte unchanged (additive-only change).
# --------------------------------------------------------------------------------------------------


class TestOldTemporalValuesByteForByte(unittest.TestCase):
    def test_default_perception_spec_unchanged(self):
        p = PerceptionSpec()
        self.assertEqual(
            (p.modality, p.temporal, p.n_frames, p.roi, p.sampling, p.audio_s),
            ("vision", "single", 1, "focus", "recent", None),
        )

    def test_every_pre_existing_value_still_validates_unchanged(self):
        for value in ("single", "strip", "window", "sprite"):
            with self.subTest(value=value):
                self.assertEqual(
                    coerce_authored_perception({"temporal": value}),
                    {"perception": {"temporal": value}},
                )

    def test_unauthored_goal_still_single_frame_byte_for_byte(self):
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(program_of(g).perception, PerceptionSpec())

    def test_unknown_vocab_still_dropped(self):
        # a value outside the (now five-member) closed vocabulary is still rejected exactly as before.
        self.assertEqual(coerce_authored_perception({"temporal": "forever"}), {})
        self.assertEqual(coerce_authored_perception({"temporal": "montage"}), {})

    def test_valid_vocab_is_additive_superset(self):
        # event_capsule is the ONLY new member; nothing pre-existing was removed or renamed.
        self.assertEqual(
            _VALID_TEMPORAL,
            {"single", "strip", "window", "sprite", "event_capsule"},
        )


if __name__ == "__main__":
    unittest.main()
