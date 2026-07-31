"""Activating the AppearanceIdentity seam — re-acquire identity across a tracking GAP via an appearance
embedder (the validated hero-probe mechanism, now in the core graph).

The seam was built but dormant (no embedder injected). Here we inject the harness colour-histogram embedder
and prove the graph re-acquires the same id when a subject leaves tracking and re-enters at a new location —
where pure continuity (geometry) would mint a new id. This is the foundation the entity-scoped rhythm hero
will stand on. numpy/opencv live behind the harness embedder; the core only sees the float signature.
"""

from __future__ import annotations

import builtins
import unittest

from percept.adapters import Box, Observation
from percept.entity import EntityGraph
from percept.identity import AppearanceIdentity, ContinuityIdentity, cosine

try:
    import numpy as np
    from percept.harness.appearance import DEGENERATE_BIN_MASS, MAX_BINS, appearance_signature
    _HAVE = True
except Exception:  # pragma: no cover - opencv/numpy not installed
    _HAVE = False

RED, BLUE = (0, 0, 255), (255, 0, 0)  # BGR
LOGGER = "percept.harness.appearance"

#: How far the second tone of a textured swatch sits along the value axis. Two value bins apart at
#: the shipped ``vbins=8`` (32 levels per bin), so the two tones never quantise together.
_TONE_STEP = 96


def _second_tone(color):
    """A distinctly-valued second tone of ``color`` — same hue and saturation, different value."""
    if max(color) < 128:
        return tuple(min(255, c + _TONE_STEP) for c in color)
    return tuple(max(0, c - _TONE_STEP) for c in color)


def _fill(f, color, x0, y0, x1, y1):
    """Paint ``f``'s region in two tones of ``color``, alternating by row."""
    f[y0:y1, x0:x1] = color
    f[y0 + 1:y1:2, x0:x1] = _second_tone(color)
    return f


def _frame(color, box, size=100):
    """A size×size BGR frame with ``color`` filling ``box``'s region (rest black).

    The fill is TEXTURED — two tones of ``color`` alternating by row — because the embedder refuses a
    crop that is uniform enough to put >98% of its mass in one histogram bin. A flat swatch carries no
    identity information (see ``test_a_uniform_crop_is_refused_as_carrying_no_identity``), so a
    flat-swatch fixture would exercise the refusal path rather than the descriptor. Every crop at least
    two rows tall holds both tones.
    """
    f = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = int(box.x * size), int(box.y * size)
    x1, y1 = int((box.x + box.w) * size), int((box.y + box.h) * size)
    return _fill(f, color, x0, y0, x1, y1)


@unittest.skipUnless(_HAVE, "needs numpy + opencv for the appearance embedder")
class AppearanceReidTests(unittest.TestCase):
    def test_signature_discriminates_colour(self):
        b = Box(0.1, 0.1, 0.3, 0.5)
        from percept.identity import cosine
        red, blue = appearance_signature(_frame(RED, b), b), appearance_signature(_frame(BLUE, b), b)
        self.assertGreater(cosine(red, appearance_signature(_frame(RED, b), b)), 0.95)  # same colour ≈ identical
        self.assertLess(cosine(red, blue), 0.2)                                         # red vs blue ≈ orthogonal

    def test_reacquires_identity_after_a_gap(self):
        g = EntityGraph(identity=AppearanceIdentity(threshold=0.5), embedder=appearance_signature)
        b1, b2, b3 = Box(0.05, 0.1, 0.2, 0.4), Box(0.7, 0.1, 0.2, 0.4), Box(0.4, 0.5, 0.2, 0.4)
        a_id = g.ingest([Observation("person", b1, 0.9)], frame=_frame(RED, b1), t=0.0)[0].id
        g.ingest([Observation("person", b2, 0.9)], frame=_frame(BLUE, b2), t=1.0)  # a blue bystander elsewhere
        # the red subject re-enters at a NEW location (no IoU overlap with b1) — continuity fails, appearance wins
        out = g.ingest([Observation("person", b3, 0.9)], frame=_frame(RED, b3), t=2.0)
        self.assertEqual(out[0].id, a_id)  # re-acquired the SAME identity across the gap

    def test_continuity_alone_loses_identity_across_the_jump(self):
        g = EntityGraph(identity=ContinuityIdentity())  # The continuity-only default drifts here.
        b1, b3 = Box(0.05, 0.1, 0.2, 0.4), Box(0.4, 0.5, 0.2, 0.4)
        a_id = g.ingest([Observation("person", b1, 0.9)], t=0.0)[0].id
        moved = g.ingest([Observation("person", b3, 0.9)], t=1.0)[0].id
        self.assertNotEqual(a_id, moved)  # geometry alone mints a new id on the jump (the failure re-id fixes)


def _solid(color, size=40):
    """A size×size BGR frame filled edge to edge with one colour — a crop with NO internal structure."""
    f = np.zeros((size, size, 3), dtype=np.uint8)
    f[:, :] = color
    return f


def _textured(color, size=40):
    """A size×size BGR frame in two tones of ``color``, alternating by row (see ``_frame``)."""
    return _fill(np.zeros((size, size, 3), dtype=np.uint8), color, 0, 0, size, size)


def _graded(color, size=40, levels=4):
    """A size×size BGR frame cycling ``levels`` brightnesses of ``color`` by row.

    Spread across the value range rather than two adjacent tones, so the crop stays non-degenerate at
    any ``vbins`` — ``_textured``'s two tones can quantise together at a coarse value axis.
    """
    f = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(levels):
        f[i::levels, :] = tuple(int(round(c * (i + 1) / levels)) for c in color)
    return f


FULL = Box(0.0, 0.0, 1.0, 1.0)
DIMS = 16 * 16 * 8  # the shipped bins=16, vbins=8 -> a hue x saturation x value histogram


@unittest.skipUnless(_HAVE, "needs numpy + opencv for the appearance embedder")
class AppearanceSignatureBoundaryTests(unittest.TestCase):
    """Boundaries of ``appearance_signature`` — the embedder ``Percept.create(embedder=...)`` injects.

    Mechanism M-38. The descriptor is a JOINT hue x saturation x value histogram: the value axis is
    what keeps black, grey and white apart, and a crop uniform enough to land almost all of its mass
    in one bin is refused outright rather than returned as a confident match (BUG-A-08-01). Input of
    any dtype is normalised to 8-bit before conversion, so a float frame describes the same scene as
    its 8-bit twin (BUG-A-08-02).

    The return value is still a bare ``None`` for every no-evidence cause — that is the caller
    contract, ``EntityGraph`` reads it as "no appearance evidence" — but each cause now names itself
    on the module's DEBUG log, so an operator can tell them apart.
    """

    # ---- the None return paths, and how a caller tells them apart --------------------------------

    def _without_the_imaging_dependencies(self):
        """Make ``import cv2`` / ``import numpy`` raise inside the function, then restore.

        ``sys.modules`` is deliberately left alone — six other test modules share this process.
        """
        real_import = builtins.__import__

        def refusing_import(name, *args, **kwargs):
            if name in ("cv2", "numpy"):
                raise ImportError(f"no {name} in this deployment")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = refusing_import
        self.addCleanup(setattr, builtins, "__import__", real_import)

    def test_missing_imaging_dependency_returns_none_without_raising_the_alarm(self):
        """Rewritten for the diagnostic: the cause is on the DEBUG log, but nothing at INFO or above.

        A deployment without opencv is a configuration fact, not a per-frame fault; warning on every
        frame would be noise. The graph still just goes quiet.
        """
        b = Box(0.1, 0.1, 0.3, 0.5)
        frame = _frame(RED, b)  # built while the real numpy is still reachable
        self.assertIsNotNone(appearance_signature(frame, b))
        self._without_the_imaging_dependencies()
        with self.assertNoLogs("percept", level="INFO"):  # no warning, no health event
            with self.assertLogs(LOGGER, level="DEBUG") as cm:
                self.assertIsNone(appearance_signature(frame, b))
        self.assertIn("numpy/opencv unavailable", "\n".join(cm.output))

    def test_non_image_frames_return_none(self):
        b = Box(0.1, 0.1, 0.5, 0.5)
        two_d = np.zeros((40, 40), dtype=np.uint8)             # a grayscale buffer, ndim == 2
        one_channel = np.zeros((40, 40, 1), dtype=np.uint8)    # shape[2] < 3
        self.assertIsNone(appearance_signature(two_d, b))
        self.assertIsNone(appearance_signature(one_channel, b))

    def test_a_four_channel_bgra_frame_is_refused_rather_than_stripped(self):
        """The channel guard is exact (``!= 3``), not ``< 3``.

        A BGRA frame used to be handed to ``cvtColor`` as if it were BGR. Its alpha cannot be honoured
        without compositing, and histogramming un-composited pixels counts fully transparent ones as
        black — so it is refused, with the cause on the log, rather than silently described wrong.
        """
        bgra = np.zeros((40, 40, 4), dtype=np.uint8)
        bgra[:, :, :3] = RED
        bgra[1::2, :, :3] = _second_tone(RED)
        with self.assertLogs(LOGGER, level="DEBUG") as cm:
            self.assertIsNone(appearance_signature(bgra, FULL))
        self.assertIn("not an HxWx3 BGR array", "\n".join(cm.output))
        # ...and the caller's remedy works: drop the alpha plane and the same pixels describe fine.
        self.assertIsNotNone(appearance_signature(np.ascontiguousarray(bgra[:, :, :3]), FULL))
        self.assertIsNone(appearance_signature(np.zeros((40, 40, 5), dtype=np.uint8), FULL))

    def test_degenerate_crops_return_none_at_the_two_pixel_boundary(self):
        frame = _textured(RED, size=100)
        self.assertIsNone(appearance_signature(frame, Box(0.5, 0.5, 0.01, 0.01)))     # 1×1 crop
        self.assertIsNotNone(appearance_signature(frame, Box(0.5, 0.5, 0.02, 0.02)))  # 2×2 is admitted
        self.assertIsNone(appearance_signature(frame, Box(0.5, 0.5, 0.02, 0.01)))     # 2 wide, 1 tall
        self.assertIsNone(appearance_signature(frame, Box(0.1, 0.1, 0.0, 0.5)))       # zero width
        self.assertIsNone(appearance_signature(frame, Box(0.1, 0.1, 0.5, 0.0)))       # zero height

    def test_box_entirely_off_frame_returns_none_but_a_straddling_box_is_clamped(self):
        frame = _textured(RED, size=100)
        self.assertIsNone(appearance_signature(frame, Box(1.5, 1.5, 0.2, 0.2)))       # past the far edge
        self.assertIsNone(appearance_signature(frame, Box(-0.5, -0.5, 0.2, 0.2)))     # entirely before it
        # A box that straddles the origin is silently clamped to the visible part, not refused — the
        # signature describes a smaller crop than the caller asked for, with no way to notice.
        straddling = appearance_signature(frame, Box(-0.2, -0.2, 0.5, 0.5))
        self.assertIsNotNone(straddling)
        self.assertEqual(straddling, appearance_signature(frame, Box(0.0, 0.0, 0.3, 0.3)))

    def test_every_none_path_returns_the_same_none_but_names_itself_on_the_log(self):
        """Rewritten for the fix: the RETURN value is still one undifferentiated ``None`` — that is the
        ``EntityGraph`` contract — but each cause is now recoverable from the DEBUG log.

        The old float-frame "empty histogram" cause is gone: a float frame is normalised to 8-bit and
        described like any other, so that input now reaches the descriptor instead of falling off it.
        """
        b = Box(0.1, 0.1, 0.5, 0.5)
        frame = _frame(RED, b)
        causes = {
            "not an HxWx3 BGR array": lambda: appearance_signature(np.zeros((40, 40), dtype=np.uint8), b),
            "smaller than 2x2": lambda: appearance_signature(frame, Box(0.5, 0.5, 0.01, 0.01)),
            "off-frame": lambda: appearance_signature(frame, Box(1.5, 1.5, 0.2, 0.2)),
            "near-uniform": lambda: appearance_signature(_solid(RED), FULL),
            "are not integers": lambda: appearance_signature(frame, b, bins=0),
        }
        outcomes = {}
        for cause, call in causes.items():
            with self.subTest(cause=cause):
                with self.assertLogs(LOGGER, level="DEBUG") as cm:
                    outcomes[cause] = call()
                self.assertIsNone(outcomes[cause])
                self.assertIn(cause, "\n".join(cm.output))  # the log names THIS cause, not a generic one
        self._without_the_imaging_dependencies()
        outcomes["no imaging dependency"] = appearance_signature(frame, b)
        self.assertIsNone(outcomes["no imaging dependency"])
        # The values remain indistinguishable by design: a caller that only checks `is None` is unaffected.
        self.assertEqual(len({id(v) for v in outcomes.values()}), 1)

    # ---- the shape and normalisation of a signature ---------------------------------------------

    def test_default_bins_give_a_joint_hsv_l1_normalised_signature(self):
        """Re-pinned dimensionality: adding the value axis takes 16×16 = 256 to 16×16×8 = 2048.

        Any signature persisted under the old descriptor is invalid — it is a different space, and a
        cosine between the two is meaningless.
        """
        b = Box(0.1, 0.1, 0.3, 0.5)
        sig = appearance_signature(_frame(RED, b), b)
        self.assertEqual(len(sig), DIMS)
        self.assertAlmostEqual(sum(sig), 1.0, places=5)  # L1-normalised
        self.assertTrue(all(0.0 <= v <= 1.0 for v in sig))
        self.assertTrue(all(isinstance(v, float) for v in sig))  # plain floats, not numpy scalars

    def test_bins_and_vbins_set_the_dimension_and_normalisation_holds_at_every_value(self):
        frame = _graded(RED, size=40)
        for bins, vbins in ((1, 2), (2, 2), (3, 5), (4, 8), (16, 8), (32, 16), (MAX_BINS, MAX_BINS)):
            with self.subTest(bins=bins, vbins=vbins):
                sig = appearance_signature(frame, FULL, bins=bins, vbins=vbins)
                self.assertEqual(len(sig), bins * bins * vbins)
                self.assertAlmostEqual(sum(sig), 1.0, places=5)

    def test_bins_of_one_no_longer_makes_every_subject_match_every_other(self):
        """``bins=1`` collapses hue and saturation, but the value axis still separates subjects.

        Under the hue×saturation descriptor this returned the constant ``(1.0,)`` for everything, so
        at ``bins=1`` every detection re-identified as every other one. Now a dark subject and a pale
        one differ on value even with the chroma axes collapsed.
        """
        dark = appearance_signature(_textured((20, 20, 20)), FULL, bins=1)
        pale = appearance_signature(_textured((235, 235, 235)), FULL, bins=1)
        self.assertEqual(len(dark), 8)  # 1 x 1 x vbins
        self.assertLess(cosine(dark, pale), 0.8)  # below identity.appearance_threshold, not equal to it
        # Collapsing the value axis TOO leaves a single bin, which is refused as carrying nothing.
        self.assertIsNone(appearance_signature(_textured(RED), FULL, bins=1, vbins=1))

    def test_bad_bins_return_none_instead_of_an_opaque_error(self):
        """Was: ``bins=0`` surfaced as an ``AttributeError`` on a missing ``.sum``. ``bins`` had no
        validation at all, so the docstring's never-crashes promise did not hold."""
        frame = _textured(RED)
        for kwargs in ({"bins": 0}, {"bins": -3}, {"bins": 2.5}, {"bins": True}, {"bins": None},
                       {"bins": MAX_BINS + 1}, {"bins": 10 ** 9}, {"vbins": 0}, {"vbins": -1},
                       {"vbins": MAX_BINS + 1}):
            with self.subTest(**kwargs):
                self.assertIsNone(appearance_signature(frame, FULL, **kwargs))

    def test_a_two_tone_crop_puts_its_mass_in_exactly_two_bins(self):
        sig = appearance_signature(_textured(RED), FULL)
        populated = [(i, v) for i, v in enumerate(sig) if v > 0.0]
        self.assertEqual(len(populated), 2)
        self.assertAlmostEqual(sum(v for _i, v in populated), 1.0, places=5)
        # A different hue lands in different bins — the descriptor is orthogonal, not equal.
        self.assertAlmostEqual(cosine(sig, appearance_signature(_textured(BLUE), FULL)), 0.0, places=5)

    # ---- pixel-index truncation at the frame edge -----------------------------------------------

    def test_int_truncation_drops_the_final_column_and_row_at_the_frame_edge(self):
        """``int()`` truncates the box edges, so a box reaching 99.9% of the frame loses the last
        pixel line entirely — its signature is byte-identical to a box that stops at 99%."""
        size = 100
        frame = _textured(RED, size=size)
        frame[:, size - 1] = BLUE   # the last column
        frame[size - 1, :] = BLUE   # the last row
        reaching_the_edge = appearance_signature(frame, Box(0.0, 0.0, 0.999, 0.999))
        stopping_short = appearance_signature(frame, Box(0.0, 0.0, 0.99, 0.99))
        whole_frame = appearance_signature(frame, FULL)
        self.assertEqual(reaching_the_edge, stopping_short)   # the truncated line is simply gone
        self.assertNotEqual(reaching_the_edge, whole_frame)   # w=1.0 does include it
        self.assertEqual(len([v for v in reaching_the_edge if v > 0.0]), 2)  # the two red tones only
        self.assertEqual(len([v for v in whole_frame if v > 0.0]), 3)        # + the blue edge

    # ---- BUG-A-08-01: the value channel, and the degeneracy guard --------------------------------

    def test_a_uniform_crop_is_refused_as_carrying_no_identity(self):
        """Was ``test_an_all_black_crop_does_not_return_none``, which pinned the defect.

        The ``total <= 0.0`` guard was unreachable for any admitted 8-bit crop — ``calcHist`` counts
        every pixel, including the zero-hue ones — so a descriptor carrying no identity information
        came back as a fully normalised, confident-looking signature. The guard the docstring promised
        is now the one that fires: more than ``DEGENERATE_BIN_MASS`` of the mass in one bin is refused.
        """
        for color in ((0, 0, 0), (128, 128, 128), (255, 255, 255), RED, BLUE):
            with self.subTest(color=color):
                self.assertIsNone(appearance_signature(_solid(color), FULL))

    def test_the_degeneracy_guard_fires_at_the_documented_mass(self):
        """Just over ``DEGENERATE_BIN_MASS`` in one bin is refused; just under is described."""
        self.assertAlmostEqual(DEGENERATE_BIN_MASS, 0.98, places=3)
        size = 100  # 10_000 px: 150 off-tone pixels -> 0.985 dominant, 250 -> 0.975
        for off_tone, expect_none in ((150, True), (250, False)):
            with self.subTest(off_tone=off_tone):
                frame = _solid(RED, size=size)
                flat = frame.reshape(-1, 3)
                flat[:off_tone] = _second_tone(RED)
                sig = appearance_signature(frame, FULL)
                self.assertEqual(sig is None, expect_none)
                if sig is not None:
                    self.assertGreater(max(sig), 0.97)  # still lopsided, but it does carry a second tone

    def test_a_dark_subject_and_a_white_subject_are_not_one_entity(self):
        """BUG-A-08-01, was xfail(strict=True). The hue×saturation descriptor put black, grey and
        white in one bin, so ``EntityGraph`` merged a near-black subject and a white one into a single
        entity at the shipped ``appearance_threshold`` of 0.8."""
        g = EntityGraph(identity=AppearanceIdentity(threshold=0.8), embedder=appearance_signature)
        b1, b3 = Box(0.05, 0.1, 0.2, 0.4), Box(0.4, 0.5, 0.2, 0.4)
        dark = g.ingest([Observation("person", b1, 0.9)],
                        frame=_frame((10, 10, 10), b1), t=0.0)[0].id
        pale = g.ingest([Observation("person", b3, 0.9)],
                        frame=_frame((255, 255, 255), b3), t=2.0)[0].id
        self.assertNotEqual(dark, pale)

    def test_achromatic_shades_are_distinct_signatures(self):
        """Was ``test_achromatic_signatures_are_currently_identical``, the pin for the defect: black,
        white and mid-grey shared one byte-identical signature at cosine exactly 1.0."""
        black = appearance_signature(_textured((10, 10, 10)), FULL)
        white = appearance_signature(_textured((255, 255, 255)), FULL)
        grey = appearance_signature(_textured((128, 128, 128)), FULL)
        for a, b in ((black, white), (black, grey), (grey, white)):
            self.assertNotEqual(a, b)
            self.assertLess(cosine(a, b), 0.8)  # under identity.appearance_threshold

    # ---- BUG-A-08-02: the float/uint8 hue-range fork ---------------------------------------------

    def test_a_float_frame_describes_the_same_scene_as_its_8_bit_twin(self):
        """BUG-A-08-02, was xfail(strict=True). ``cvtColor`` emits hue in 0..360 for float input and
        0..180 for ``uint8`` while the histogram range was fixed at the 8-bit convention, so every
        pixel above hue 180° was dropped: an all-blue float frame emptied the histogram entirely, and
        a half-red/half-blue float frame scored cosine 0.0 against its own 8-bit twin.

        Fixed by normalising the input dtype, so there is one code path and one histogram range.
        """
        eight_bit = np.zeros((40, 40, 3), dtype=np.uint8)
        eight_bit[:, :20, 2] = 255   # red half
        eight_bit[:, 20:, 0] = 255   # blue half
        reference = appearance_signature(eight_bit, FULL)
        twins = {
            "float32 in 0..1": eight_bit.astype(np.float32) / 255.0,
            "float64 in 0..1": eight_bit.astype(np.float64) / 255.0,
            "float32 already in 0..255": eight_bit.astype(np.float32),
            "int32 holding 0..255": eight_bit.astype(np.int32),
            "uint16 at its dtype's full scale": eight_bit.astype(np.uint16) * 257,
        }
        for name, twin in twins.items():
            with self.subTest(dtype=name):
                self.assertAlmostEqual(cosine(appearance_signature(twin, FULL), reference), 1.0, places=3)

    def test_a_float_frame_whose_hue_exceeds_180_degrees_is_no_longer_dropped(self):
        """Was ``test_the_empty_histogram_path_needs_a_float_frame_to_reach``.

        Blue is hue 240° — the exact input that used to empty the histogram. It now converts, and the
        only reason it still returns ``None`` is that a flat swatch is uniform. Give it structure and
        it describes normally.
        """
        flat_blue = np.zeros((40, 40, 3), dtype=np.float32)
        flat_blue[:, :, 0] = 1.0
        with self.assertLogs(LOGGER, level="DEBUG") as cm:
            self.assertIsNone(appearance_signature(flat_blue, FULL))
        self.assertIn("near-uniform", "\n".join(cm.output))   # NOT "histogram is empty"
        textured_blue = flat_blue.copy()
        textured_blue[1::2, :, 0] = 0.4
        self.assertIsNotNone(appearance_signature(textured_blue, FULL))

    def test_the_dtype_scale_is_decided_per_frame_not_per_crop(self):
        """Two boxes on one float frame must be described on the same scale, or their signatures are
        not comparable — so the 0..1 vs 0..255 decision reads the whole frame's peak, not the crop's."""
        frame = np.zeros((40, 40, 3), dtype=np.float32)
        frame[:, :20] = np.array(RED, dtype=np.float32) / 255.0        # a bright left half
        frame[1::2, :20] = np.array(_second_tone(RED), dtype=np.float32) / 255.0
        frame[:, 20:] = frame[:, :20] * 0.25                           # the same crop, dimmed
        left = appearance_signature(frame, Box(0.0, 0.0, 0.5, 1.0))
        right = appearance_signature(frame, Box(0.5, 0.0, 0.5, 1.0))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        # A per-crop scale decision would renormalise the dim half back up to full brightness and make
        # these two identical; a per-frame one keeps the brightness difference the pixels actually have.
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
