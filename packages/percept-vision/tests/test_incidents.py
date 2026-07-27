"""Incident producer (L1) — FireEvent -> Incident + snippet assembly from the rolling FrameWindow."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import unittest
import zlib
from unittest.mock import patch

from percept.edge.tier0 import gray_frame
from percept.events import FrameWindow
from percept.incidents import incident_judgment, make_incident, snippet_from_window
from percept.contracts import Incident, IncidentSnippet, Verdict
from percept.runtime.sinks import incident_payload

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is an extra, absent in the L1 stdlib lane
    _HAVE_NUMPY = False


class _Fire:
    """Duck-typed stand-in for a FireEvent (goal_id / text / verdict)."""

    def __init__(self, goal_id, text, verdict=None, entity_id=None):
        self.goal_id = goal_id
        self.text = text
        self.verdict = verdict
        self.entity_id = entity_id


class MakeIncidentTests(unittest.TestCase):
    def test_maps_fire_to_incident(self):
        v = Verdict(satisfied=True, confidence=0.9, evidence="person on the floor, not moving")
        inc = make_incident(_Fire("fall", "Possible fall - check on them", v), ts=12.5)
        self.assertEqual(inc.concern_id, "fall")
        self.assertEqual(inc.verdict, v)
        self.assertEqual(inc.message, "Possible fall - check on them")
        self.assertEqual(inc.ts, 12.5)
        self.assertRegex(inc.id, r"^inc-fall-12\.500-[0-9a-f]{8}$")
        self.assertIsNone(inc.label)  # unlabelled until a human / the calibration loop classifies it
        self.assertIsNone(inc.entity_id)  # not entity-scoped -> the back-compatible default

    def test_none_verdict_gets_a_placeholder(self):
        inc = make_incident(_Fire("g", "hi", None), ts=1.0)
        self.assertEqual(inc.verdict.confidence, 0.0)
        self.assertEqual(inc.message, "hi")

    def test_explicit_id_and_snippet_passthrough(self):
        snip = IncidentSnippet(kind="frame", frames=["f"])
        inc = make_incident(_Fire("g", "hi"), ts=1.0, id="X-1", snippet=snip)
        self.assertEqual(inc.id, "X-1")
        self.assertIs(inc.snippet, snip)

    def test_preserves_exact_runtime_local_judgment(self):
        judgment = object()
        fire = _Fire("g", "hi")
        fire.judgment = judgment

        incident = make_incident(fire, ts=1.0)

        self.assertIs(incident_judgment(incident), judgment)
        self.assertNotIn("_judgment", incident_payload(incident))


class IncidentEntityIdTests(unittest.TestCase):
    """The gate key is (goal_id, entity_id); identity must survive the emission boundary."""

    def test_entity_id_carries_from_the_fire(self):
        inc = make_incident(_Fire("caffeine", "stay off caffeine", None, entity_id="cup2"), ts=3.0)
        self.assertEqual(inc.entity_id, "cup2")

    def test_two_subjects_on_one_concern_stay_distinguishable(self):
        # the client-reported failure: interleaved fires from two subjects read as out-of-order
        # divergence because every incident looked identical downstream.
        a = make_incident(_Fire("zone", "left the zone", None, entity_id="person-1"), ts=1.0)
        b = make_incident(_Fire("zone", "left the zone", None, entity_id="person-2"), ts=1.5)
        c = make_incident(_Fire("zone", "left the zone", None, entity_id="person-1"), ts=2.0)
        self.assertEqual([i.entity_id for i in (a, b, c)], ["person-1", "person-2", "person-1"])
        self.assertEqual(a.concern_id, b.concern_id)  # same concern...
        self.assertNotEqual(a.entity_id, b.entity_id)  # ...different subject

    def test_absent_entity_id_defaults_to_none(self):
        class _Legacy:  # a legacy duck-typed fire with NO entity_id attribute at all
            goal_id, text, verdict = "g", "hi", None

        self.assertIsNone(make_incident(_Legacy(), ts=1.0).entity_id)

    def test_explicit_override_wins_and_non_str_is_coerced(self):
        fire = _Fire("g", "hi", None, entity_id="from-fire")
        self.assertEqual(make_incident(fire, ts=1.0, entity_id="explicit").entity_id, "explicit")
        self.assertEqual(make_incident(_Fire("g", "hi", None, entity_id=7), ts=1.0).entity_id, "7")

    def test_positional_construction_is_unchanged(self):
        # entity_id is appended LAST on the dataclass, so every pre-1.3 positional build still means
        # what it meant (a silent field reorder would corrupt existing callers).
        inc = Incident("i", "c", Verdict(satisfied=True, confidence=1.0), 2.0, "real")
        self.assertEqual((inc.label, inc.entity_id), ("real", None))

    def test_end_to_end_through_the_real_gate(self):
        """The gate key IS (goal_id, entity_id) — drive the REAL firing path and prove the entity the
        gate keyed on is the entity the emitted Incident carries."""
        from percept.adapters import Goal
        from percept.agent import Percept
        from percept.backends.fake import FakeVision
        from percept.gate import FIRE

        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False)
        g = Goal(id="caffeine", condition="c", say="stay off caffeine")
        ev = a._fire_concern(g, Verdict(True, 0.95, "yes"), ("caffeine", "cup2"), freshness=1.0, t=0.0)
        self.assertEqual(ev.action, FIRE)
        inc = make_incident(ev, ts=0.0)
        self.assertEqual((inc.concern_id, inc.entity_id), ("caffeine", "cup2"))

    def test_serializes_through_the_webhook_payload(self):
        inc = make_incident(_Fire("g", "hi", None, entity_id="e9"), ts=1.0)
        self.assertEqual(incident_payload(inc)["entity_id"], "e9")
        self.assertIsNone(incident_payload(make_incident(_Fire("g", "hi"), ts=1.0))["entity_id"])


class SnippetFromWindowTests(unittest.TestCase):
    def _window(self, n):
        w = FrameWindow(size=8)
        for t in range(n):
            w.push(float(t), f"f{t}")
        return w

    def test_assembles_strip_from_window(self):
        snip = snippet_from_window(self._window(5), kind="frame")
        self.assertEqual(snip.kind, "frame")
        self.assertEqual(snip.frames, ["f0", "f1", "f2", "f3", "f4"])
        self.assertEqual(snip.t_start, 0.0)
        self.assertEqual(snip.t_end, 4.0)

    def test_pre_s_bounds_by_time(self):
        snip = snippet_from_window(self._window(5), pre_s=2.0)  # t_end=4 -> keep t>=2
        self.assertEqual(snip.frames, ["f2", "f3", "f4"])

    def test_empty_window_is_none(self):
        self.assertIsNone(snippet_from_window(FrameWindow()))


# --------------------------------------------------------------------------------------------------
# The clip bundle carries EVIDENCE — or says it does not. ``kind="clip"`` is the mode chosen exactly
# when evidence leaves the machine, and it used to fall back to ``{"repr": repr(frame)}`` for any
# frame that was not an already-encoded ``(mime, bytes)`` pair. numpy elides the middle of any array
# over 1000 elements, so a decoded 64x64x3 camera frame — 12,288 pixel bytes — was shipped as a
# ~1KB truncated string under an evidence media type, unrecoverably and without a word to anyone.
# --------------------------------------------------------------------------------------------------


def _decode_png(payload: bytes) -> tuple[int, int, int, bytes]:
    """Minimal stdlib reader for the 8-bit, filter-0 PNGs the evidence encoder writes.

    Deliberately independent of the code under test (and of PIL/OpenCV): it re-derives the pixel
    bytes straight off the wire, so an assertion about them is a statement about the payload a
    consumer receives, not a re-run of the encoder."""
    assert payload.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG"
    pos, width, height, channels = 8, 0, 0, 0
    idat = bytearray()
    while pos < len(payload):
        (length,) = struct.unpack(">I", payload[pos:pos + 4])
        tag, data = payload[pos + 4:pos + 8], payload[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", data[:10])
            assert depth == 8, depth
            channels = {0: 1, 4: 2, 2: 3, 6: 4}[color_type]
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    raw, stride, out = zlib.decompress(bytes(idat)), width * channels, bytearray()
    for row in range(height):
        start = row * (stride + 1)
        assert raw[start] == 0, "unexpected scanline filter"
        out += raw[start + 1:start + 1 + stride]
    return width, height, channels, bytes(out)


class _Opaque:
    """A frame with no faithful encoding, whose repr would leak its contents if one were taken."""

    def __repr__(self) -> str:  # pragma: no cover - only reached if the defect returns
        return "OPAQUE-FRAME-CONTENTS"


class _PurePythonFrame:
    """An array-like pixel buffer with no numpy — the stdlib lane's decoded frame."""

    def __init__(self, height: int, width: int, channels: int) -> None:
        self.shape = (height, width) if channels == 1 else (height, width, channels)
        self.dtype = "uint8"
        self._raw = bytes((i * 7 + 3) % 256 for i in range(height * width * channels))
        self.nbytes = len(self._raw)

    def tobytes(self) -> bytes:
        return self._raw


def _clip_bundle(frames: list) -> dict:
    window = FrameWindow(size=64)
    for i, frame in enumerate(frames):
        window.push(float(i), frame)
    snippet = snippet_from_window(window, kind="clip")
    assert snippet.mime == "application/x-percept-frame-bundle+json"
    return json.loads(snippet.media.decode("utf-8"))


class ClipBundleEvidenceTests(unittest.TestCase):
    def test_encoded_tuple_frames_are_byte_identical_to_the_legacy_bundle(self):
        # the happy path the capture thread actually publishes — unchanged, byte for byte.
        window = FrameWindow(size=8)
        window.push(0.0, ("image/jpeg", b"\xff\xd8\xffframe-one"))
        window.push(1.0, ("image/jpeg", b"\xff\xd8\xffframe-two"))
        snippet = snippet_from_window(window, kind="clip")
        expected = json.dumps({
            "version": 1, "t_start": 0.0, "t_end": 1.0,
            "frames": [
                {"mime": "image/jpeg",
                 "data": base64.b64encode(b"\xff\xd8\xffframe-one").decode("ascii")},
                {"mime": "image/jpeg",
                 "data": base64.b64encode(b"\xff\xd8\xffframe-two").decode("ascii")},
            ],
        }).encode("utf-8")
        self.assertEqual(snippet.media, expected)
        self.assertNotIn("unavailable_frames", json.loads(snippet.media.decode("utf-8")))

    def test_decoded_pixel_buffer_ships_its_real_pixels(self):
        frame = _PurePythonFrame(9, 17, 3)
        entry = _clip_bundle([frame])["frames"][0]
        self.assertNotIn("repr", entry)
        self.assertEqual(entry["mime"], "image/png")
        width, height, channels, pixels = _decode_png(base64.b64decode(entry["data"]))
        self.assertEqual((height, width, channels), (9, 17, 3))
        # the payload IS the pixels: byte-for-byte, and hashing to the same digest.
        self.assertEqual(pixels, frame.tobytes())
        self.assertEqual(
            hashlib.sha256(pixels).hexdigest(), hashlib.sha256(frame.tobytes()).hexdigest()
        )

    def test_greyscale_and_alpha_buffers_round_trip_exactly(self):
        for channels in (1, 4):
            with self.subTest(channels=channels):
                frame = _PurePythonFrame(6, 5, channels)
                entry = _clip_bundle([frame])["frames"][0]
                _w, _h, decoded_channels, pixels = _decode_png(base64.b64decode(entry["data"]))
                self.assertEqual(decoded_channels, channels)
                self.assertEqual(pixels, frame.tobytes())

    def test_grid_frame_ships_its_real_pixels(self):
        frame = gray_frame(4, 3, 42)  # the w/h/pixels signal frame shape
        entry = _clip_bundle([frame])["frames"][0]
        self.assertEqual(entry["mime"], "image/png")
        width, height, channels, pixels = _decode_png(base64.b64decode(entry["data"]))
        self.assertEqual((width, height, channels), (4, 3, 1))
        self.assertEqual(pixels, bytes(frame.pixels))

    def test_unencodable_frame_is_an_explicit_gap_never_a_description(self):
        window = FrameWindow(size=8)
        window.push(0.0, ("image/jpeg", b"real"))
        window.push(1.0, _Opaque())
        snippet = snippet_from_window(window, kind="clip")
        bundle = json.loads(snippet.media.decode("utf-8"))
        gap = bundle["frames"][1]
        # a gap carries no bytes and no rendering of the frame — a consumer keying on "mime"/"data"
        # cannot read it as evidence, and the object's contents never reach the wire.
        self.assertEqual(sorted(gap), ["unavailable"])
        self.assertEqual(gap["unavailable"]["reason"], "unencodable-frame-type")
        self.assertEqual(gap["unavailable"]["frame_type"], "_Opaque")
        self.assertNotIn(b"OPAQUE-FRAME-CONTENTS", snippet.media)
        self.assertEqual(bundle["unavailable_frames"], 1)  # and the bundle SAYS it is incomplete
        self.assertEqual(bundle["frames"][0]["mime"], "image/jpeg")  # the real frame still lands

    def test_no_frame_is_ever_a_repr_stand_in(self):
        bundle = _clip_bundle([
            ("image/jpeg", b"encoded"),
            _PurePythonFrame(4, 4, 3),
            gray_frame(2, 2, 1),
            _Opaque(),
            "a-test-double",
            {"not": "a frame"},
        ])
        for entry in bundle["frames"]:
            with self.subTest(entry=sorted(entry)):
                self.assertNotIn("repr", entry)
                self.assertIn(sorted(entry), (["data", "mime"], ["unavailable"]))
        self.assertEqual(bundle["unavailable_frames"], 3)

    def test_oversized_encoding_is_a_gap_not_a_trimmed_payload(self):
        frame = _PurePythonFrame(20, 20, 3)
        with patch("percept.incidents._MAX_ENCODED_FRAME_BYTES", 16):
            entry = _clip_bundle([frame])["frames"][0]
        self.assertEqual(sorted(entry), ["unavailable"])
        self.assertEqual(entry["unavailable"]["reason"], "exceeds-max-encoded-frame-bytes")
        self.assertEqual(entry["unavailable"]["source_nbytes"], frame.nbytes)

    def test_bundle_budget_stops_at_a_gap_and_keeps_the_frames_it_already_carried(self):
        frames = [_PurePythonFrame(8, 8, 3) for _ in range(3)]
        with patch("percept.incidents._MAX_ENCODED_BUNDLE_BYTES", 400):
            bundle = _clip_bundle(frames)
        kinds = [sorted(entry) for entry in bundle["frames"]]
        self.assertIn(["data", "mime"], kinds)  # what fit is carried in full...
        self.assertIn(["unavailable"], kinds)  # ...and what did not is declared missing
        self.assertEqual(bundle["unavailable_frames"], kinds.count(["unavailable"]))


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy for the decoded-camera-frame shape")
class ClipBundleNumpyEvidenceTests(unittest.TestCase):
    """The reported defect verbatim: a decoded numpy camera frame through ``snippet_kind='clip'``."""

    def _frame(self):
        rng = np.random.default_rng(7)
        return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_ndarray_frame_is_not_shipped_as_a_truncated_repr(self):
        frame = self._frame()
        self.assertIn("...", repr(frame))  # numpy elides the middle — what used to be "evidence"
        entry = _clip_bundle([frame])["frames"][0]
        self.assertNotIn("repr", entry)
        self.assertEqual(entry["mime"], "image/png")
        _w, _h, _c, pixels = _decode_png(base64.b64decode(entry["data"]))
        self.assertEqual(len(pixels), frame.nbytes)  # 12,288 pixel bytes, not a ~1KB string
        self.assertEqual(
            hashlib.sha256(pixels).hexdigest(), hashlib.sha256(frame.tobytes()).hexdigest()
        )

    def test_non_contiguous_view_carries_the_view_it_was_given(self):
        frame = self._frame()[::2, ::2]  # an ROI-style view — C-order bytes, not the parent buffer
        entry = _clip_bundle([frame])["frames"][0]
        _w, _h, _c, pixels = _decode_png(base64.b64decode(entry["data"]))
        self.assertEqual(pixels, frame.tobytes())

    def test_unsupported_dtype_is_a_gap_with_no_pixel_description(self):
        frame = self._frame().astype("float32")
        entry = _clip_bundle([frame])["frames"][0]
        self.assertEqual(sorted(entry), ["unavailable"])
        self.assertEqual(entry["unavailable"]["reason"], "unsupported-array-dtype")
        self.assertEqual(entry["unavailable"]["detail"], "float32")
        self.assertEqual(entry["unavailable"]["source_nbytes"], frame.nbytes)


if __name__ == "__main__":
    unittest.main()
