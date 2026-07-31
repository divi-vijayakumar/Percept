"""Direct unit coverage for ``prepare_temporal_images`` upload bounds.

Spec row M-52 (temporal-image preparation caps,
``percept/runtime/video_clip.py:176-232``) lists the shipped knobs
``max_dimension=768``, ``max_input_image_bytes=8 MiB``,
``max_input_total_bytes=32 MiB``, ``max_image_bytes=1 MiB`` and
``max_total_bytes=4 MiB``, and records the function as having zero direct
test references.

Every cap in the implementation is a strict ``>``, so a value exactly at a cap
must be admitted and only cap+1 may raise.  These tests derive each cap from a
JPEG that is actually encoded here rather than hard-coding a byte count, so the
boundary is exact on any OpenCV build.

The row's stated failure mode -- input and output budgets are checked
independently, so a set that would comfortably fit after re-encode is still
rejected on its input size -- is pinned below as the *current documented
contract*, not as a defect.
"""

from __future__ import annotations

import builtins
import inspect
import unittest

try:  # module-scope SkipTest is the only skip both runners honour
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without [runtime]
    raise unittest.SkipTest(
        f"prepare_temporal_images needs percept-vision[runtime]: {exc}"
    ) from exc

from percept.runtime.video_clip import prepare_temporal_images

_HUGE = 1 << 30  # a cap large enough to be out of the way


def _encode(array: "np.ndarray", quality: int = 95) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise AssertionError("test fixture failed to encode a JPEG")
    return encoded.tobytes()


def _noise_jpeg(height: int, width: int, *, seed: int = 7, quality: int = 95) -> bytes:
    """Random noise compresses poorly, so quality moves the payload size a lot."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return _encode(array, quality)


def _decode(payload: bytes) -> "np.ndarray":
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("test fixture produced an undecodable JPEG")
    return image


def _dimensions(payload: bytes) -> tuple[int, int]:
    height, width = _decode(payload).shape[:2]
    return int(height), int(width)


def _reference_payload(
    raw: bytes,
    *,
    max_dimension: int = 768,
    quality: int = 85,
) -> bytes:
    """Mirror the implementation's decode -> optional resize -> re-JPEG path."""
    image = _decode(raw)
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return _encode(image, quality)


def _ladder_payload(
    raw: bytes,
    cap: int,
    *,
    max_dimension: int = 768,
) -> bytes | None:
    """What the 85/70/55/40 ladder should settle on for ``max_image_bytes=cap``."""
    payload = b""
    for quality in (85, 70, 55, 40):
        payload = _reference_payload(raw, max_dimension=max_dimension, quality=quality)
        if len(payload) <= cap:
            return payload
    return None


class TemporalImageShapeTests(unittest.TestCase):
    """The result shape: an empty sequence, and ``("image/jpeg", bytes)`` pairs."""

    def test_empty_sequence_returns_an_empty_list(self) -> None:
        self.assertEqual(prepare_temporal_images([]), [])

    def test_every_element_is_an_image_jpeg_media_tuple(self) -> None:
        raw_bytes = _noise_jpeg(40, 60, seed=1)
        raw_tuple = ("image/jpeg", _noise_jpeg(40, 60, seed=2))

        prepared = prepare_temporal_images([raw_bytes, raw_tuple])

        self.assertEqual(len(prepared), 2)
        for index, item in enumerate(prepared):
            with self.subTest(index=index):
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
                media_type, payload = item
                self.assertEqual(media_type, "image/jpeg")
                self.assertIsInstance(payload, bytes)
                # it really is a JPEG, not an opaque blob
                self.assertEqual(payload[:2], b"\xff\xd8")
                _decode(payload)

    def test_shipped_defaults_are_the_documented_five_knobs(self) -> None:
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(
                prepare_temporal_images
            ).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        self.assertEqual(
            defaults,
            {
                "max_dimension": 768,
                "max_input_image_bytes": 8 * 1024 * 1024,
                "max_input_total_bytes": 32 * 1024 * 1024,
                "max_image_bytes": 1024 * 1024,
                "max_total_bytes": 4 * 1024 * 1024,
            },
        )


class TemporalImageDimensionTests(unittest.TestCase):
    """``max_dimension`` is a strict ``>``: an image already within it is kept."""

    def test_image_already_under_max_dimension_keeps_its_dimensions(self) -> None:
        raw = _noise_jpeg(36, 48, seed=3)
        self.assertEqual(_dimensions(raw), (36, 48))

        (_, payload), = prepare_temporal_images([raw], max_dimension=768)

        # The code always re-JPEGs at quality 85, so byte identity is not the
        # contract -- geometric fidelity is.
        self.assertEqual(_dimensions(payload), (36, 48))

    def test_image_exactly_at_max_dimension_is_not_resized(self) -> None:
        raw = _noise_jpeg(60, 100, seed=4)

        (_, payload), = prepare_temporal_images([raw], max_dimension=100)

        self.assertEqual(_dimensions(payload), (60, 100))

    def test_oversized_image_is_downscaled_onto_max_dimension(self) -> None:
        raw = _noise_jpeg(200, 400, seed=5)

        (_, payload), = prepare_temporal_images(
            [raw],
            max_dimension=100,
            max_image_bytes=_HUGE,
            max_total_bytes=_HUGE,
        )

        height, width = _dimensions(payload)
        self.assertEqual(max(height, width), 100)
        self.assertEqual((height, width), (50, 100))  # aspect ratio preserved


class TemporalImageInputBudgetTests(unittest.TestCase):
    """``max_input_image_bytes`` / ``max_input_total_bytes`` boundaries."""

    def test_image_exactly_at_max_input_image_bytes_is_accepted(self) -> None:
        raw = _noise_jpeg(40, 60, seed=6)

        prepared = prepare_temporal_images(
            [raw],
            max_input_image_bytes=len(raw),
            max_image_bytes=_HUGE,
            max_total_bytes=_HUGE,
        )

        self.assertEqual(len(prepared), 1)

    def test_image_one_byte_over_max_input_image_bytes_raises(self) -> None:
        raw = _noise_jpeg(40, 60, seed=6)

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images(
                [raw],
                max_input_image_bytes=len(raw) - 1,
                max_image_bytes=_HUGE,
                max_total_bytes=_HUGE,
            )

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_input_image_bytes",
        )

    def test_total_exactly_at_max_input_total_bytes_is_accepted(self) -> None:
        raws = [_noise_jpeg(40, 60, seed=7), _noise_jpeg(40, 60, seed=8)]
        total = sum(len(raw) for raw in raws)

        prepared = prepare_temporal_images(
            raws,
            max_input_total_bytes=total,
            max_image_bytes=_HUGE,
            max_total_bytes=_HUGE,
        )

        self.assertEqual(len(prepared), 2)

    def test_total_one_byte_over_max_input_total_bytes_raises(self) -> None:
        raws = [_noise_jpeg(40, 60, seed=7), _noise_jpeg(40, 60, seed=8)]
        total = sum(len(raw) for raw in raws)

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images(
                raws,
                max_input_total_bytes=total - 1,
                max_image_bytes=_HUGE,
                max_total_bytes=_HUGE,
            )

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_input_total_bytes",
        )


class TemporalImageOutputBudgetTests(unittest.TestCase):
    """``max_image_bytes`` / ``max_total_bytes`` boundaries and the quality ladder."""

    def test_payload_exactly_at_max_image_bytes_is_accepted_at_quality_85(self) -> None:
        raw = _noise_jpeg(120, 160, seed=9)
        expected = _reference_payload(raw)

        (_, payload), = prepare_temporal_images(
            [raw],
            max_input_image_bytes=_HUGE,
            max_image_bytes=len(expected),
            max_total_bytes=_HUGE,
        )

        self.assertEqual(len(payload), len(expected))
        self.assertEqual(payload, expected)

    def test_quality_ladder_steps_down_when_85_does_not_fit(self) -> None:
        raw = _noise_jpeg(120, 160, seed=9)
        at_85 = _reference_payload(raw, quality=85)
        cap = len(at_85) - 1
        expected = _ladder_payload(raw, cap)
        self.assertIsNotNone(expected, "fixture must still fit at a lower quality")

        (_, payload), = prepare_temporal_images(
            [raw],
            max_input_image_bytes=_HUGE,
            max_image_bytes=cap,
            max_total_bytes=_HUGE,
        )

        self.assertLessEqual(len(payload), cap)
        self.assertNotEqual(payload, at_85)
        self.assertEqual(payload, expected)

    def test_payload_exactly_at_the_lowest_rung_of_the_ladder_is_accepted(self) -> None:
        raw = _noise_jpeg(120, 160, seed=9)
        at_40 = _reference_payload(raw, quality=40)

        (_, payload), = prepare_temporal_images(
            [raw],
            max_input_image_bytes=_HUGE,
            max_image_bytes=len(at_40),
            max_total_bytes=_HUGE,
        )

        self.assertEqual(payload, at_40)

    def test_payload_one_byte_under_the_lowest_rung_raises(self) -> None:
        raw = _noise_jpeg(120, 160, seed=9)
        at_40 = _reference_payload(raw, quality=40)

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images(
                [raw],
                max_input_image_bytes=_HUGE,
                max_image_bytes=len(at_40) - 1,
                max_total_bytes=_HUGE,
            )

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_image_bytes",
        )

    def test_total_exactly_at_max_total_bytes_is_accepted(self) -> None:
        raws = [_noise_jpeg(60, 80, seed=10), _noise_jpeg(60, 80, seed=11)]
        total = sum(len(_reference_payload(raw)) for raw in raws)

        prepared = prepare_temporal_images(
            raws,
            max_input_image_bytes=_HUGE,
            max_image_bytes=_HUGE,
            max_total_bytes=total,
        )

        self.assertEqual(sum(len(payload) for _, payload in prepared), total)

    def test_total_one_byte_over_max_total_bytes_raises(self) -> None:
        raws = [_noise_jpeg(60, 80, seed=10), _noise_jpeg(60, 80, seed=11)]
        total = sum(len(_reference_payload(raw)) for raw in raws)

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images(
                raws,
                max_input_image_bytes=_HUGE,
                max_image_bytes=_HUGE,
                max_total_bytes=total - 1,
            )

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_total_bytes",
        )

    def test_max_total_bytes_does_not_engage_the_quality_ladder(self) -> None:
        """Only ``max_image_bytes`` drives re-encoding; the total cap just refuses."""
        raws = [_noise_jpeg(60, 80, seed=10), _noise_jpeg(60, 80, seed=11)]
        first = _reference_payload(raws[0])
        total = sum(len(_reference_payload(raw)) for raw in raws)
        self.assertLess(
            len(_reference_payload(raws[1], quality=40)),
            total - len(first),
            "a quality drop on the second image would have fit under the cap",
        )

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images(
                raws,
                max_input_image_bytes=_HUGE,
                max_image_bytes=_HUGE,
                max_total_bytes=total - 1,
            )

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_total_bytes",
        )


class TemporalImageUndecodableTests(unittest.TestCase):
    def test_undecodable_payload_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images([b"this is not a JPEG at all"])

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence contains an undecodable image",
        )

    def test_input_cap_is_checked_before_the_image_is_decoded(self) -> None:
        junk = b"x" * 64

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images([junk], max_input_image_bytes=len(junk) - 1)

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_input_image_bytes",
        )

    def test_a_good_image_after_a_bad_one_does_not_rescue_the_call(self) -> None:
        good = _noise_jpeg(40, 60, seed=12)

        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images([good, b"not-an-image", good])

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence contains an undecodable image",
        )


class TemporalImageInputAndOutputBudgetsAreIndependentTests(unittest.TestCase):
    """Row M-52's stated failure mode, pinned as the current contract.

    The input budget is spent before anything is decoded, so an image that
    would shrink dramatically under the output budget is still refused on the
    size it arrived at.  If this ever starts passing, the contract changed and
    the row must be revisited -- it is not asserted here as desirable.
    """

    def test_a_highly_compressible_input_is_still_rejected_on_its_input_size(
        self,
    ) -> None:
        raw = _noise_jpeg(400, 400, seed=13)

        (_, payload), = prepare_temporal_images(
            [raw],
            max_dimension=100,
            max_input_image_bytes=len(raw),
            max_image_bytes=len(raw),
            max_total_bytes=_HUGE,
        )
        # The prepared evidence is a small fraction of the arriving bytes...
        self.assertLess(len(payload) * 4, len(raw))

        # ...yet one byte of input over the input cap refuses the whole set,
        # even though the output cap here is len(raw), far above len(payload).
        with self.assertRaises(ValueError) as caught:
            prepare_temporal_images(
                [raw],
                max_dimension=100,
                max_input_image_bytes=len(raw) - 1,
                max_image_bytes=len(raw),
                max_total_bytes=_HUGE,
            )

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence exceeds max_input_image_bytes",
        )


class TemporalImageMissingRuntimeExtraTests(unittest.TestCase):
    """A missing ``[runtime]`` extra raises; it never degrades to no evidence."""

    def _block_import(self, blocked: str) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == blocked:
                raise ModuleNotFoundError(f"No module named {blocked!r}", name=blocked)
            return real_import(name, *args, **kwargs)

        # Restore unconditionally: an unrestored global here would pollute every
        # later test module in the process.
        self.addCleanup(setattr, builtins, "__import__", real_import)
        builtins.__import__ = fake_import

    def test_missing_cv2_raises_runtime_error(self) -> None:
        raw = _noise_jpeg(40, 60, seed=14)
        self._block_import("cv2")

        with self.assertRaises(RuntimeError) as caught:
            prepare_temporal_images([raw])

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence needs percept-vision[runtime]",
        )
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_missing_numpy_raises_runtime_error(self) -> None:
        raw = _noise_jpeg(40, 60, seed=15)
        self._block_import("numpy")

        with self.assertRaises(RuntimeError) as caught:
            prepare_temporal_images([raw])

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence needs percept-vision[runtime]",
        )

    def test_missing_extra_raises_even_for_an_empty_sequence(self) -> None:
        self._block_import("cv2")

        with self.assertRaises(RuntimeError) as caught:
            prepare_temporal_images([])

        self.assertEqual(
            str(caught.exception),
            "temporal image evidence needs percept-vision[runtime]",
        )

    def test_the_import_hook_is_restored_after_each_test(self) -> None:
        # Guards the cleanup above: if a sibling test leaked its patch, this
        # would raise instead of preparing evidence.
        raw = _noise_jpeg(40, 60, seed=16)
        self.assertEqual(len(prepare_temporal_images([raw])), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
