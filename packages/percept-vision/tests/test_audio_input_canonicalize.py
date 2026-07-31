"""Direct unit coverage for compiled audio ingress canonicalization and its byte caps.

``percept.audio_input`` is the sole media boundary a compiled Watch crosses before an audio
payload can reach a provider. It never converts first and measures afterwards: it *predicts* the
canonical 16-bit output size from the declared width, charges that prediction against the
provider cap, and only then allocates the converted buffer. An under-estimating prediction would
therefore let an over-cap buffer through, so these tests pin the prediction table against the
payload it actually produces, and pin the cap boundary to the byte.

They also cover the four canonicalizers that had no direct test: the strict container parse, the
unit canonicalizer, the ingress-frame canonicalizer that derives a media interval from a byte
count, and the provider-side homogeneity/continuity validation.
"""

from __future__ import annotations

import math
import struct
import unittest
from dataclasses import replace

import percept.audio_input as audio_input
from percept.audio_input import (
    MAX_AUDIO_INPUT_BYTES,
    MAX_AUDIO_PROVIDER_BYTES,
    AudioInputCapability,
    audio_frame_from_units,
    canonicalize_audio_frame,
    canonicalize_audio_ingress_frame,
    canonicalize_audio_unit,
    validate_audio_units_for_provider,
)
from percept.contracts import AudioFrame
from percept.events import AudioBoundary, AudioUnit

CAP = AudioInputCapability(modes=frozenset({"audio_only"}))
HEADER = audio_input._WAV_HEADER_BYTES


# --------------------------------------------------------------------------------------------
# builders — the same shapes the audio evidence runtime tests use, kept local so a change here
# cannot silently retune that file's expectations.
# --------------------------------------------------------------------------------------------


def _pcm(samples: tuple, fmt: str = "s16le") -> bytes:
    """Little-endian interleaved samples in one of the three accepted input widths."""
    if fmt == "s16le":
        return struct.pack("<" + "h" * len(samples), *samples)
    if fmt == "u8":
        return bytes(samples)
    if fmt == "f32le":
        return struct.pack("<" + "f" * len(samples), *samples)
    raise AssertionError(f"unsupported test format {fmt!r}")


def _chunk(tag: bytes, payload: bytes, *, pad: bool = True) -> bytes:
    return (
        tag
        + struct.pack("<I", len(payload))
        + payload
        + (b"\0" if pad and len(payload) % 2 else b"")
    )


def _fmt_payload(
    *,
    encoding: int = 1,
    channels: int = 1,
    rate: int = 4,
    bits: int = 16,
    byte_rate: int | None = None,
    block_align: int | None = None,
) -> bytes:
    align = channels * (bits // 8) if block_align is None else block_align
    rate_bytes = (
        rate * channels * (bits // 8) if byte_rate is None else byte_rate
    )
    return struct.pack("<HHIIHH", encoding, channels, rate, rate_bytes, align, bits)


def _wav_from_chunks(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav(
    data: bytes,
    *,
    rate: int = 4,
    channels: int = 1,
    bits: int = 16,
    pad: bool = True,
) -> bytes:
    """A canonical container: one 16-byte fmt chunk followed by one data chunk."""
    return _wav_from_chunks(
        _chunk(b"fmt ", _fmt_payload(channels=channels, rate=rate, bits=bits)),
        _chunk(b"data", data, pad=pad),
    )


def _frame(
    samples: tuple = (32767,),
    *,
    start: float | None = None,
    rate: int = 4,
    fmt: str = "s16le",
    channels: int = 1,
) -> AudioFrame:
    raw = _pcm(samples, fmt)
    bounds = {}
    if start is not None:
        bounds = {
            "t_start": start,
            "t_end": start + (len(samples) // channels) / rate,
        }
    return AudioFrame(
        pcm=raw,
        sample_rate=rate,
        channels=channels,
        sample_format=fmt,
        **bounds,
    )


def _unit(
    seq: int = 0,
    *,
    start: float = 0.0,
    samples: tuple = (32767,),
    rate: int = 4,
    fmt: str = "s16le",
    channels: int = 1,
    **fields,
) -> AudioUnit:
    chunk = _frame(samples, start=start, rate=rate, fmt=fmt, channels=channels)
    return AudioUnit(
        seq=seq,
        t_start=chunk.t_start,
        t_end=chunk.t_end,
        chunk=chunk,
        **fields,
    )


def _canonical(candidate, **overrides) -> AudioUnit:
    """Canonicalize with plausible carrier provenance; overrides pin what a test is about.

    The defaults are derived defensively so a deliberately malformed candidate still reaches the
    canonicalizer rather than tripping over this helper.
    """
    seq = getattr(candidate, "seq", 0)
    end = getattr(candidate, "t_end", None)
    kwargs = {
        "capability": CAP,
        "source_id": "mic",
        "source_epoch": 0,
        "source_sequence": seq if type(seq) is int and seq >= 0 else 0,
        "observed_time": 100.0,
        "source_time": (
            float(end)
            if isinstance(end, (int, float)) and not isinstance(end, bool)
            else None
        ),
    }
    kwargs.update(overrides)
    return canonicalize_audio_unit(candidate, **kwargs)


def _run(seq: int, start: float, samples: tuple = (100, 200), **overrides) -> AudioUnit:
    """One canonical unit in a contiguous run: seq n starts where seq n-1 ended."""
    return _canonical(_unit(seq, start=start, samples=samples), **overrides)


class PredictedOutputSizeTests(unittest.TestCase):
    """The cap is charged against a prediction, so the prediction must not under-count."""

    def test_prediction_table_is_exact_for_each_accepted_width(self) -> None:
        predict = audio_input._predict_pcm16_size
        for size in (2, 4, 16, 4096, MAX_AUDIO_INPUT_BYTES):
            with self.subTest(size=size):
                self.assertEqual(predict(size, "s16le"), size)
                self.assertEqual(predict(size, "u8"), size * 2)
                self.assertEqual(predict(size, "f32le"), size // 2)

    def test_an_unknown_width_refuses_rather_than_predicting_zero(self) -> None:
        with self.assertRaises(ValueError) as caught:
            audio_input._predict_pcm16_size(64, "s24le")
        self.assertEqual(
            str(caught.exception), "unsupported audio sample format 's24le'"
        )
        for unsupported in ("", "S16LE", "pcm", None, 16):
            with self.subTest(unsupported=unsupported):
                with self.assertRaises(ValueError):
                    audio_input._predict_pcm16_size(64, unsupported)

    def test_prediction_equals_the_payload_conversion_actually_produces(self) -> None:
        """The row's failure mode: a prediction below the real output size defeats the cap."""
        cases = (
            ("s16le", (1, -1, 32767, -32768), 4, 1),
            ("s16le", (1, -1, 32767, -32768), 2, 2),
            ("u8", (0, 64, 128, 255), 4, 1),
            ("u8", (0, 64, 128, 255, 7, 9), 3, 2),
            ("f32le", (-1.0, -0.5, 0.5, 1.0), 4, 1),
            ("f32le", (-1.0, 0.0, 0.25, 1.0, -0.75, 0.5), 3, 2),
        )
        for fmt, samples, rate, channels in cases:
            with self.subTest(fmt=fmt, channels=channels):
                raw = _pcm(samples, fmt)
                frame = AudioFrame(
                    pcm=raw,
                    sample_rate=rate,
                    channels=channels,
                    sample_format=fmt,
                )
                got = canonicalize_audio_frame(
                    frame, t_start=0.0, t_end=1.0, capability=CAP
                )
                self.assertEqual(
                    len(got.pcm), audio_input._predict_pcm16_size(len(raw), fmt)
                )
                self.assertEqual(got.sample_format, "s16le")

    def test_container_prediction_equals_the_payload_it_yields(self) -> None:
        for bits, samples, fmt in ((16, (7, -7), "s16le"), (8, (0, 128, 255, 64), "u8")):
            with self.subTest(bits=bits):
                encoded = _wav(_pcm(samples, fmt), rate=4, bits=bits)
                info = audio_input._parse_pcm_wav(encoded, capability=CAP)
                got = canonicalize_audio_frame(
                    AudioFrame(
                        pcm=b"",
                        encoded=encoded,
                        mime="audio/wav",
                        sample_rate=4,
                        channels=1,
                        sample_format=fmt,
                        t_start=0.0,
                        t_end=len(samples) / 4,
                    ),
                    t_start=0.0,
                    t_end=len(samples) / 4,
                    capability=CAP,
                )
                self.assertEqual(info.predicted_pcm16_bytes, len(got.pcm))
                self.assertEqual(info.input_bytes, len(encoded))


class ProviderOutputCapTests(unittest.TestCase):
    """``max_provider_bytes`` bounds the container, header included."""

    def test_exactly_the_header_complement_is_admitted_and_one_byte_more_is_not(
        self,
    ) -> None:
        self.assertEqual(HEADER, 44)
        self.assertIsNone(
            audio_input._require_output_cap(
                MAX_AUDIO_PROVIDER_BYTES - HEADER, capability=CAP
            )
        )
        with self.assertRaises(ValueError) as caught:
            audio_input._require_output_cap(
                MAX_AUDIO_PROVIDER_BYTES - HEADER + 1, capability=CAP
            )
        self.assertEqual(
            str(caught.exception),
            "audio provider payload would exceed the compiled output cap",
        )

    def test_the_cap_boundary_holds_for_a_narrowed_capability(self) -> None:
        narrow = AudioInputCapability(
            modes=frozenset({"audio_only"}), max_provider_bytes=HEADER + 100
        )
        audio_input._require_output_cap(100, capability=narrow)
        with self.assertRaises(ValueError):
            audio_input._require_output_cap(101, capability=narrow)

    def test_width_expansion_is_charged_before_the_buffer_is_allocated(self) -> None:
        """Eight u8 bytes fit the input cap but cost sixteen bytes of provider budget."""
        raw = _pcm((0, 1, 2, 3, 4, 5, 6, 7), "u8")
        frame = AudioFrame(pcm=raw, sample_rate=8, channels=1, sample_format="u8")
        exact = AudioInputCapability(
            modes=frozenset({"audio_only"}),
            max_input_bytes=8,
            max_provider_bytes=HEADER + 16,
        )
        got = canonicalize_audio_frame(frame, t_start=0.0, t_end=1.0, capability=exact)
        self.assertEqual(len(got.pcm), 16)
        one_short = AudioInputCapability(
            modes=frozenset({"audio_only"}),
            max_input_bytes=8,
            max_provider_bytes=HEADER + 15,
        )
        with self.assertRaises(ValueError) as caught:
            canonicalize_audio_frame(frame, t_start=0.0, t_end=1.0, capability=one_short)
        self.assertEqual(
            str(caught.exception),
            "audio provider payload would exceed the compiled output cap",
        )

    def test_a_container_at_exactly_the_input_cap_is_admitted(self) -> None:
        """At the shipped defaults the largest admissible container is the cap itself."""
        payload = MAX_AUDIO_INPUT_BYTES - HEADER
        encoded = _wav(bytes(payload))
        self.assertEqual(len(encoded), MAX_AUDIO_INPUT_BYTES)
        info = audio_input._parse_pcm_wav(encoded, capability=CAP)
        self.assertEqual(info.predicted_pcm16_bytes, payload)
        self.assertEqual(info.predicted_pcm16_bytes + HEADER, MAX_AUDIO_PROVIDER_BYTES)
        del encoded, info

        oversize = _wav(bytes(payload + 2))
        with self.assertRaises(ValueError) as caught:
            audio_input._parse_pcm_wav(oversize, capability=CAP)
        self.assertEqual(
            str(caught.exception),
            "AudioFrame.encoded exceeds the compiled audio input cap",
        )
        del oversize

    def test_a_narrowed_input_cap_refuses_a_container_the_global_cap_allows(
        self,
    ) -> None:
        encoded = _wav(_pcm((1, 2)))
        narrow = AudioInputCapability(
            modes=frozenset({"audio_only"}), max_input_bytes=len(encoded) - 1
        )
        with self.assertRaises(ValueError) as caught:
            audio_input._parse_pcm_wav(encoded, capability=narrow)
        self.assertEqual(
            str(caught.exception), "WAV input exceeds the compiled audio input cap"
        )


class WavContainerParseTests(unittest.TestCase):
    """``_parse_pcm_wav`` inspects without copying, and refuses everything non-canonical."""

    def test_canonical_container_is_read_without_copying_its_payload(self) -> None:
        data = _pcm((7, -7))
        info = audio_input._parse_pcm_wav(_wav(data, rate=8), capability=CAP)
        self.assertEqual(info.sample_rate, 8)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.sample_format, "s16le")
        self.assertEqual(bytes(info.data), data)
        self.assertIsInstance(info.data, memoryview)
        self.assertEqual(info.input_bytes, HEADER + len(data))
        self.assertEqual(info.predicted_pcm16_bytes, len(data))

    def test_extra_metadata_chunks_are_skipped_not_refused(self) -> None:
        data = _pcm((9, -9))
        for extra in (
            _chunk(b"LIST", b"INFOhelo"),
            _chunk(b"fact", struct.pack("<I", 2)),
            _chunk(b"LIST", b"INFOodd"),
        ):
            with self.subTest(extra=extra[:4]):
                encoded = _wav_from_chunks(
                    _chunk(b"fmt ", _fmt_payload()),
                    extra,
                    _chunk(b"data", data),
                )
                info = audio_input._parse_pcm_wav(encoded, capability=CAP)
                self.assertEqual(bytes(info.data), data)
                self.assertEqual(info.input_bytes, len(encoded))

    def test_a_header_only_container_carries_no_samples(self) -> None:
        header_only = _wav(b"")
        self.assertEqual(len(header_only), HEADER)
        with self.assertRaises(ValueError) as caught:
            audio_input._parse_pcm_wav(header_only, capability=CAP)
        self.assertEqual(str(caught.exception), "WAV data chunk must be non-empty")

    def test_riff_magic_and_declared_length_must_be_exact(self) -> None:
        valid = _wav(_pcm((1,)))
        magic_attacks = (
            b"",
            b"RIFF",
            b"RIFX" + valid[4:],
            valid[:8] + b"AVI " + valid[12:],
        )
        for raw in magic_attacks:
            with self.subTest(raw=raw[:12]):
                with self.assertRaises(ValueError) as caught:
                    audio_input._parse_pcm_wav(raw, capability=CAP)
                self.assertEqual(
                    str(caught.exception), "encoded audio must be a RIFF/WAVE container"
                )
        for raw in (valid + b"xx", valid[:-2]):
            with self.subTest(length=len(raw)):
                with self.assertRaises(ValueError) as caught:
                    audio_input._parse_pcm_wav(raw, capability=CAP)
                self.assertEqual(
                    str(caught.exception),
                    "WAV RIFF declared length must equal the exact payload length",
                )

    def test_chunk_structure_is_strict(self) -> None:
        fmt = _chunk(b"fmt ", _fmt_payload())
        data = _chunk(b"data", _pcm((1,)))
        cases = (
            (_wav_from_chunks(fmt, fmt, data), "WAV contains duplicate fmt chunks"),
            (_wav_from_chunks(fmt, data, data), "WAV contains duplicate data chunks"),
            (
                _wav_from_chunks(fmt),
                "WAV requires exactly one complete fmt chunk and one data chunk",
            ),
            (
                _wav_from_chunks(data),
                "WAV requires exactly one complete fmt chunk and one data chunk",
            ),
            (
                _wav_from_chunks(_chunk(b"fmt ", _fmt_payload() + b"\0\0"), data),
                "compiled audio requires one canonical 16-byte PCM fmt chunk",
            ),
            (
                _wav_from_chunks(fmt, data, b"pad"),
                "WAV ended inside a chunk header",
            ),
            (
                _wav_from_chunks(fmt, _chunk(b"LIST", b"odd", pad=False), data),
                "WAV chunk exceeds the declared RIFF boundary",
            ),
        )
        for raw, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    audio_input._parse_pcm_wav(raw, capability=CAP)
                self.assertEqual(str(caught.exception), message)

    def test_an_unpadded_odd_data_chunk_is_refused(self) -> None:
        """Three u8 samples need a RIFF pad byte; without one the structure does not close."""
        with self.assertRaises(ValueError) as caught:
            audio_input._parse_pcm_wav(
                _wav(bytes((0, 128, 255)), rate=3, bits=8, pad=False), capability=CAP
            )
        self.assertEqual(
            str(caught.exception), "WAV chunk exceeds the declared RIFF boundary"
        )
        info = audio_input._parse_pcm_wav(
            _wav(bytes((0, 128, 255)), rate=3, bits=8), capability=CAP
        )
        self.assertEqual(bytes(info.data), bytes((0, 128, 255)))

    def test_declared_bit_depth_selects_the_width_and_only_two_are_accepted(
        self,
    ) -> None:
        eight = audio_input._parse_pcm_wav(
            _wav(bytes((0, 128, 255, 64)), rate=4, bits=8), capability=CAP
        )
        self.assertEqual(eight.sample_format, "u8")
        self.assertEqual(eight.predicted_pcm16_bytes, 8)
        sixteen = audio_input._parse_pcm_wav(_wav(_pcm((1, 2))), capability=CAP)
        self.assertEqual(sixteen.sample_format, "s16le")
        self.assertEqual(sixteen.predicted_pcm16_bytes, 4)
        for bits in (4, 24, 32, 64):
            with self.subTest(bits=bits):
                raw = _wav_from_chunks(
                    _chunk(b"fmt ", _fmt_payload(bits=bits)),
                    _chunk(b"data", bytes(8)),
                )
                with self.assertRaises(ValueError) as caught:
                    audio_input._parse_pcm_wav(raw, capability=CAP)
                self.assertEqual(
                    str(caught.exception),
                    "compiled PCM WAV supports only 8-bit unsigned or 16-bit signed samples",
                )

    def test_fmt_metadata_must_be_uncompressed_and_self_consistent(self) -> None:
        data = _chunk(b"data", _pcm((1,)))
        cases = (
            (
                _fmt_payload(encoding=3, bits=32),
                "compiled audio accepts uncompressed integer PCM WAV only",
            ),
            (
                _fmt_payload(encoding=0xFFFE),
                "compiled audio accepts uncompressed integer PCM WAV only",
            ),
            (_fmt_payload(channels=0), "audio channels must be a positive integer"),
            (_fmt_payload(rate=0), "audio sample_rate must be a positive integer"),
            (
                _fmt_payload(block_align=4),
                "WAV fmt rate/alignment metadata is inconsistent",
            ),
            (
                _fmt_payload(byte_rate=9),
                "WAV fmt rate/alignment metadata is inconsistent",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    audio_input._parse_pcm_wav(
                        _wav_from_chunks(_chunk(b"fmt ", payload), data), capability=CAP
                    )
                self.assertEqual(str(caught.exception), message)

    def test_data_must_hold_whole_interleaved_sample_frames(self) -> None:
        raw = _wav_from_chunks(
            _chunk(b"fmt ", _fmt_payload(channels=2)),
            _chunk(b"data", _pcm((7,))),
        )
        with self.assertRaises(ValueError) as caught:
            audio_input._parse_pcm_wav(raw, capability=CAP)
        self.assertEqual(
            str(caught.exception), "WAV data ends inside an interleaved sample frame"
        )
        stereo = audio_input._parse_pcm_wav(
            _wav_from_chunks(
                _chunk(b"fmt ", _fmt_payload(channels=2)),
                _chunk(b"data", _pcm((7, 8))),
            ),
            capability=CAP,
        )
        self.assertEqual((stereo.channels, stereo.predicted_pcm16_bytes), (2, 4))


class CanonicalFrameIntervalTests(unittest.TestCase):
    """The interval a frame claims, and the interval its bytes imply, must agree."""

    def test_non_finite_and_reversed_intervals_are_refused(self) -> None:
        frame = _frame((1, 2, 3, 4))
        for bad in (math.nan, math.inf, -math.inf, True, "0.0", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    canonicalize_audio_frame(
                        frame, t_start=bad, t_end=1.0, capability=CAP
                    )
                self.assertEqual(str(caught.exception), "audio t_start must be finite")
                with self.assertRaises(ValueError) as caught:
                    canonicalize_audio_frame(
                        frame, t_start=0.0, t_end=bad, capability=CAP
                    )
                self.assertEqual(str(caught.exception), "audio t_end must be finite")
        with self.assertRaises(ValueError) as caught:
            canonicalize_audio_frame(frame, t_start=1.0, t_end=0.5, capability=CAP)
        self.assertEqual(str(caught.exception), "audio t_end must be >= t_start")

    def test_a_declared_frame_interval_must_agree_within_a_nanosecond(self) -> None:
        raw = _pcm((1, 2, 3, 4))

        def declared(start: float, end: float) -> AudioFrame:
            return AudioFrame(
                pcm=raw,
                sample_rate=4,
                channels=1,
                sample_format="s16le",
                t_start=start,
                t_end=end,
            )

        canonicalize_audio_frame(
            declared(1.0 + 5e-10, 2.0 - 5e-10), t_start=1.0, t_end=2.0, capability=CAP
        )
        with self.assertRaises(ValueError) as caught:
            canonicalize_audio_frame(
                declared(1.0 + 2e-9, 2.0), t_start=1.0, t_end=2.0, capability=CAP
            )
        self.assertEqual(
            str(caught.exception), "AudioFrame t_start disagrees with its AudioUnit"
        )
        with self.assertRaises(ValueError) as caught:
            canonicalize_audio_frame(
                declared(1.0, 2.0 + 2e-9), t_start=1.0, t_end=2.0, capability=CAP
            )
        self.assertEqual(
            str(caught.exception), "AudioFrame t_end disagrees with its AudioUnit"
        )
        for bad in (math.nan, math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    canonicalize_audio_frame(
                        declared(bad, 2.0), t_start=1.0, t_end=2.0, capability=CAP
                    )
                self.assertEqual(
                    str(caught.exception), "audio frame.t_start must be finite"
                )

    def test_the_interval_must_match_the_sample_count_within_a_microsecond(self) -> None:
        frame = _frame((1, 2, 3, 4))
        canonicalize_audio_frame(frame, t_start=0.0, t_end=1.0 + 5e-7, capability=CAP)
        canonicalize_audio_frame(frame, t_start=0.0, t_end=1.0 - 5e-7, capability=CAP)
        for end in (1.0 + 2e-6, 1.0 - 2e-6, 2.0):
            with self.subTest(end=end):
                with self.assertRaises(ValueError) as caught:
                    canonicalize_audio_frame(
                        frame, t_start=0.0, t_end=end, capability=CAP
                    )
                self.assertEqual(
                    str(caught.exception),
                    "AudioFrame interval disagrees with its sample count and format",
                )

    def test_only_the_two_wav_mime_spellings_are_accepted(self) -> None:
        data = _pcm((1, 2))
        encoded = _wav(data)

        def with_mime(mime) -> AudioFrame:
            return AudioFrame(
                pcm=b"",
                encoded=encoded,
                mime=mime,
                sample_rate=4,
                channels=1,
                sample_format="s16le",
                t_start=0.0,
                t_end=0.5,
            )

        for mime in ("audio/wav", "audio/x-wav", "AUDIO/WAV", "  audio/x-wav  "):
            with self.subTest(mime=mime):
                got = canonicalize_audio_frame(
                    with_mime(mime), t_start=0.0, t_end=0.5, capability=CAP
                )
                self.assertEqual(got.pcm, data)
                self.assertIsNone(got.encoded)
                self.assertIsNone(got.mime)
        for mime in (None, "", "   ", b"audio/wav", "audio/mpeg", "audio/L16"):
            with self.subTest(mime=mime):
                with self.assertRaises(ValueError):
                    canonicalize_audio_frame(
                        with_mime(mime), t_start=0.0, t_end=0.5, capability=CAP
                    )

    def test_exactly_one_carrier_may_be_populated(self) -> None:
        encoded = _wav(_pcm((1, 2)))
        both = AudioFrame(
            pcm=_pcm((1, 2)),
            encoded=encoded,
            mime="audio/wav",
            sample_rate=4,
            channels=1,
            sample_format="s16le",
        )
        neither = AudioFrame(
            pcm=b"", sample_rate=4, channels=1, sample_format="s16le"
        )
        for candidate in (both, neither):
            with self.subTest(candidate=candidate.encoded is not None):
                with self.assertRaises(ValueError) as caught:
                    canonicalize_audio_frame(
                        candidate, t_start=0.0, t_end=0.5, capability=CAP
                    )
                self.assertEqual(
                    str(caught.exception),
                    "AudioFrame requires exactly one non-empty carrier: PCM XOR encoded",
                )


class CanonicalUnitTests(unittest.TestCase):
    """``canonicalize_audio_unit`` — the entry point every ingress path funnels through."""

    def test_only_an_exact_audio_unit_with_an_exact_frame_is_accepted(self) -> None:
        class Subclassed(AudioUnit):
            pass

        chunk = _frame((1,), start=0.0)
        rejects = (
            object(),
            (0.0, chunk, 1.0),
            Subclassed(seq=0, t_start=0.0, t_end=0.25, chunk=chunk),
        )
        for candidate in rejects:
            with self.subTest(candidate=type(candidate).__name__):
                with self.assertRaises(ValueError) as caught:
                    _canonical(candidate)
                self.assertEqual(
                    str(caught.exception),
                    "compiled audio ingress requires an exact AudioUnit",
                )
        with self.assertRaises(ValueError) as caught:
            _canonical(
                AudioUnit(seq=0, t_start=0.0, t_end=0.25, chunk=b"\0\0")
            )
        self.assertEqual(
            str(caught.exception),
            "compiled audio ingress requires an exact AudioFrame chunk",
        )

    def test_sequence_and_provenance_must_be_non_negative_exact_integers(self) -> None:
        for seq in (True, -1, 1.0, "0", None):
            with self.subTest(seq=seq):
                candidate = AudioUnit(
                    seq=seq, t_start=0.0, t_end=0.25, chunk=_frame((1,), start=0.0)
                )
                with self.assertRaises(ValueError) as caught:
                    _canonical(candidate, source_sequence=0)
                self.assertEqual(
                    str(caught.exception),
                    "audio seq must be a non-negative exact integer",
                )
        for field, bad in (
            ("source_epoch", -1),
            ("source_epoch", True),
            ("source_sequence", 1.0),
            ("source_sequence", None),
        ):
            with self.subTest(field=field, bad=bad):
                with self.assertRaises(ValueError) as caught:
                    _canonical(_unit(0), **{field: bad})
                self.assertEqual(
                    str(caught.exception),
                    f"audio {field} must be a non-negative exact integer",
                )
        for bad in ("", None, b"mic"):
            with self.subTest(source_id=bad):
                with self.assertRaises(ValueError) as caught:
                    _canonical(_unit(0), source_id=bad)
                self.assertEqual(
                    str(caught.exception),
                    "audio source_id must be a non-empty exact string",
                )

    def test_direction_participant_and_boundaries_are_validated(self) -> None:
        for direction in ("input", "output", "mixed", "unknown"):
            with self.subTest(direction=direction):
                got = _canonical(_unit(0, direction=direction))
                self.assertEqual(got.direction, direction)
        forged = _unit(0)
        object.__setattr__(forged, "direction", "sideways")
        with self.assertRaises(ValueError) as caught:
            _canonical(forged)
        self.assertEqual(str(caught.exception), "audio direction is invalid")

        with self.assertRaises(ValueError) as caught:
            _canonical(_unit(0, participant_id=""))
        self.assertEqual(
            str(caught.exception),
            "audio participant_id must be a non-empty exact string when present",
        )

        ordered = (
            AudioBoundary("onset", 0.0),
            AudioBoundary("vad_start", 0.0),
            AudioBoundary("endpoint", 0.25),
        )
        self.assertEqual(_canonical(_unit(0, boundaries=ordered)).boundaries, ordered)
        with self.assertRaises(ValueError) as caught:
            _canonical(
                _unit(
                    0,
                    boundaries=(
                        AudioBoundary("onset", 1.0),
                        AudioBoundary("endpoint", 0.5),
                    ),
                )
            )
        self.assertEqual(
            str(caught.exception),
            "audio boundaries must be ordered on their event clock",
        )
        for boundaries, message in (
            ([AudioBoundary("onset", 0.0)], "audio boundaries must be an immutable tuple"),
            (
                (("onset", 0.0),),
                "audio boundaries require exact AudioBoundary values",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    _canonical(_unit(0, boundaries=boundaries))
                self.assertEqual(str(caught.exception), message)
        smuggled = AudioBoundary("onset", 0.0)
        object.__setattr__(smuggled, "t", math.nan)
        with self.assertRaises(ValueError) as caught:
            _canonical(_unit(0, boundaries=(smuggled,)))
        self.assertEqual(str(caught.exception), "audio boundary must be finite")

    def test_outer_metadata_may_not_contradict_the_inner_frame(self) -> None:
        chunk = _frame((1,), start=0.0, rate=4, channels=1)
        for field, bad in (
            ("sample_rate", 8),
            ("channels", 2),
            ("sample_format", "u8"),
        ):
            with self.subTest(field=field):
                candidate = AudioUnit(
                    seq=0, t_start=0.0, t_end=0.25, chunk=chunk, **{field: bad}
                )
                with self.assertRaises(ValueError) as caught:
                    _canonical(candidate)
                self.assertEqual(
                    str(caught.exception),
                    f"AudioUnit {field} contradicts its AudioFrame",
                )
        agreeing = AudioUnit(
            seq=0,
            t_start=0.0,
            t_end=1.0,
            chunk=_frame((0, 128, 255, 64), start=0.0, rate=4, fmt="u8"),
            sample_rate=4,
            channels=1,
            sample_format="u8",
        )
        got = _canonical(agreeing)
        self.assertEqual(got.sample_format, "s16le")
        self.assertEqual(got.chunk.sample_format, "s16le")
        self.assertEqual((got.sample_rate, got.channels), (4, 1))

    def test_a_supplied_energy_must_be_finite_and_within_the_unit_interval(self) -> None:
        for energy in (0.0, 0.5, 1.0):
            with self.subTest(energy=energy):
                self.assertEqual(_canonical(_unit(0, energy=energy)).energy, energy)
        for energy in (1.5, -0.1, math.nan, math.inf, True, "0.5"):
            with self.subTest(energy=energy):
                with self.assertRaises(ValueError) as caught:
                    _canonical(_unit(0, energy=energy))
                self.assertEqual(
                    str(caught.exception),
                    "audio energy must be finite and within [0, 1]",
                )

    def test_an_absent_energy_is_the_mean_square_of_the_canonical_samples(self) -> None:
        full_scale = 32767 / 32768.0
        cases = (
            ((32767,), 4, "s16le", full_scale * full_scale),
            ((-32768,), 4, "s16le", 1.0),
            ((0, 0, 0, 0), 4, "s16le", 0.0),
            ((16384, -16384), 2, "s16le", 0.25),
            (
                (0, 128, 255, 64),
                4,
                "u8",
                (1.0 + 0.0 + (32512 / 32768.0) ** 2 + (16384 / 32768.0) ** 2) / 4,
            ),
        )
        for samples, rate, fmt, expected in cases:
            with self.subTest(samples=samples, fmt=fmt):
                got = _canonical(_unit(0, samples=samples, rate=rate, fmt=fmt))
                self.assertAlmostEqual(got.energy, expected, places=12)
                self.assertIsInstance(got.energy, float)

    def test_the_canonical_unit_is_isolated_from_later_caller_mutation(self) -> None:
        raw = bytearray(_pcm((1, 2, 3, 4)))
        chunk = AudioFrame(
            pcm=raw,
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=0.0,
            t_end=1.0,
        )
        candidate = AudioUnit(
            seq=3, t_start=0.0, t_end=1.0, chunk=chunk, event_t=9.0, energy=0.25
        )
        got = _canonical(
            candidate, source_sequence=3, observed_time=9.0, source_time=1.0
        )
        raw[:] = b"\0" * len(raw)
        self.assertEqual(got.chunk.pcm, _pcm((1, 2, 3, 4)))
        self.assertIsInstance(got.chunk.pcm, bytes)
        self.assertIsNot(got, candidate)
        self.assertIs(candidate.chunk, chunk)
        self.assertEqual(
            (got.source_id, got.source_epoch, got.source_sequence), ("mic", 0, 3)
        )
        self.assertEqual((got.observed_time, got.source_time), (9.0, 1.0))
        self.assertEqual((got.seq, got.t_start, got.t_end, got.event_t), (3, 0.0, 1.0, 9.0))

    def test_an_absent_source_clock_stays_absent(self) -> None:
        got = _canonical(_unit(0), source_time=None)
        self.assertIsNone(got.source_time)
        self.assertIsNone(_canonical(_unit(0)).event_t)
        for bad in (math.nan, math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    _canonical(_unit(0), observed_time=bad)
                self.assertEqual(
                    str(caught.exception), "audio observed_time must be finite"
                )
                with self.assertRaises(ValueError) as caught:
                    _canonical(_unit(0, event_t=bad))
                self.assertEqual(str(caught.exception), "audio event_t must be finite")


class IngressFrameTests(unittest.TestCase):
    """``canonicalize_audio_ingress_frame`` — a bare frame, with or without an interval."""

    def test_a_half_declared_interval_is_refused(self) -> None:
        raw = _pcm((1, 2, 3, 4))
        for bounds in ({"t_start": 0.0}, {"t_end": 1.0}):
            with self.subTest(bounds=tuple(bounds)):
                frame = AudioFrame(
                    pcm=raw,
                    sample_rate=4,
                    channels=1,
                    sample_format="s16le",
                    **bounds,
                )
                with self.assertRaises(ValueError) as caught:
                    self._ingest(frame, event_t=5.0, source_time=None)
                self.assertEqual(
                    str(caught.exception),
                    "AudioFrame must declare both t_start/t_end or neither",
                )

    def test_a_declared_interval_is_taken_verbatim(self) -> None:
        frame = AudioFrame(
            pcm=_pcm((1, 2, 3, 4)),
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=2.0,
            t_end=3.0,
        )
        got = self._ingest(frame, event_t=99.0, source_time=None)
        self.assertEqual((got.t_start, got.t_end, got.event_t), (2.0, 3.0, 99.0))

    def test_a_derived_interval_is_anchored_at_the_source_clock_when_present(
        self,
    ) -> None:
        frame = _frame((1, 2, 3, 4))
        got = self._ingest(frame, event_t=50.0, source_time=7.0)
        self.assertEqual((got.t_start, got.t_end), (6.0, 7.0))
        self.assertEqual(got.event_t, 50.0)
        self.assertEqual(got.source_time, 7.0)

    def test_a_derived_interval_falls_back_to_the_event_clock(self) -> None:
        frame = _frame((1, 2, 3, 4))
        got = self._ingest(frame, event_t=50.0, source_time=None)
        self.assertEqual((got.t_start, got.t_end), (49.0, 50.0))
        self.assertIsNone(got.source_time)

    def test_a_derived_interval_uses_the_declared_width_and_layout(self) -> None:
        cases = (
            (_frame((0, 128, 255, 64), rate=4, fmt="u8"), 1.0),
            (_frame((1, 2, 3, 4, 5, 6, 7, 8), rate=4, channels=2), 1.0),
            (_frame(tuple(range(8)), rate=16), 0.5),
        )
        for frame, duration in cases:
            with self.subTest(fmt=frame.sample_format, channels=frame.channels):
                got = self._ingest(frame, event_t=10.0, source_time=None)
                self.assertAlmostEqual(got.t_start, 10.0 - duration, places=9)
                self.assertEqual(got.t_end, 10.0)

    def test_a_derived_interval_from_an_encoded_container(self) -> None:
        data = _pcm((1, 2, 3, 4))
        frame = AudioFrame(
            pcm=b"",
            encoded=_wav(data),
            mime="audio/x-wav",
            sample_rate=4,
            channels=1,
            sample_format="s16le",
        )
        got = self._ingest(frame, event_t=10.0, source_time=None, energy=0.5)
        self.assertEqual((got.t_start, got.t_end), (9.0, 10.0))
        self.assertEqual(got.chunk.pcm, data)
        self.assertIsNone(got.chunk.encoded)
        self.assertEqual(got.energy, 0.5)

    def test_a_derived_interval_survives_a_wall_clock_anchor(self) -> None:
        """A ten-millisecond chunk anchored at an epoch timestamp still closes its interval."""
        frame = _frame(tuple(range(480)), rate=48000)
        anchor = 1785000000.25
        got = self._ingest(frame, event_t=0.0, source_time=anchor, energy=0.1)
        self.assertEqual(got.t_end, anchor)
        self.assertAlmostEqual(got.t_end - got.t_start, 0.01, places=7)

    def test_exactly_one_non_empty_carrier_is_required_to_derive(self) -> None:
        empty = AudioFrame(pcm=b"", sample_rate=4, channels=1, sample_format="s16le")
        both = AudioFrame(
            pcm=_pcm((1,)),
            encoded=_wav(_pcm((1,))),
            mime="audio/wav",
            sample_rate=4,
            channels=1,
            sample_format="s16le",
        )
        for frame in (empty, both):
            with self.subTest(encoded=frame.encoded is not None):
                with self.assertRaises(ValueError) as caught:
                    self._ingest(frame, event_t=1.0, source_time=None)
                self.assertEqual(
                    str(caught.exception),
                    "AudioFrame requires exactly one non-empty bytes-like carrier",
                )

    def test_the_frame_type_clock_and_format_guards_still_apply(self) -> None:
        frame = _frame((1, 2, 3, 4))
        with self.assertRaises(ValueError) as caught:
            self._ingest(object(), event_t=1.0, source_time=None)
        self.assertEqual(
            str(caught.exception), "compiled audio ingress requires an exact AudioFrame"
        )
        with self.assertRaises(ValueError) as caught:
            self._ingest(frame, event_t=math.nan, source_time=None)
        self.assertEqual(str(caught.exception), "audio event_t must be finite")
        with self.assertRaises(ValueError) as caught:
            self._ingest(frame, event_t=1.0, source_time=None, seq=-1)
        self.assertEqual(
            str(caught.exception), "audio seq must be a non-negative exact integer"
        )
        narrow = AudioInputCapability(
            modes=frozenset({"audio_only"}), pcm_formats=frozenset({"s16le"})
        )
        with self.assertRaises(ValueError) as caught:
            self._ingest(
                _frame((0, 128, 255, 64), rate=4, fmt="u8"),
                event_t=1.0,
                source_time=None,
                capability=narrow,
            )
        self.assertEqual(str(caught.exception), "unsupported compiled PCM format")

    def test_both_byte_caps_are_enforced_before_the_interval_is_trusted(self) -> None:
        """Deriving an interval from a byte count must not skip either cap."""
        frame = _frame((1, 2, 3, 4))
        over_input = AudioInputCapability(
            modes=frozenset({"audio_only"}), max_input_bytes=4, max_provider_bytes=100
        )
        with self.assertRaises(ValueError) as caught:
            self._ingest(frame, event_t=1.0, source_time=None, capability=over_input)
        self.assertEqual(
            str(caught.exception), "AudioFrame exceeds the compiled audio input cap"
        )
        over_output = AudioInputCapability(
            modes=frozenset({"audio_only"}),
            max_input_bytes=100,
            max_provider_bytes=HEADER + 7,
        )
        with self.assertRaises(ValueError) as caught:
            self._ingest(frame, event_t=1.0, source_time=None, capability=over_output)
        self.assertEqual(
            str(caught.exception),
            "audio provider payload would exceed the compiled output cap",
        )
        exact = AudioInputCapability(
            modes=frozenset({"audio_only"}),
            max_input_bytes=100,
            max_provider_bytes=HEADER + 8,
        )
        got = self._ingest(frame, event_t=1.0, source_time=None, capability=exact)
        self.assertEqual(len(got.chunk.pcm), 8)
        self.assertEqual((got.t_start, got.t_end), (0.0, 1.0))

    def test_direction_and_participant_ride_through_to_the_canonical_unit(self) -> None:
        got = self._ingest(
            _frame((1, 2, 3, 4)),
            event_t=4.0,
            source_time=None,
            direction="output",
            participant_id="speaker-1",
        )
        self.assertEqual(got.direction, "output")
        self.assertEqual(got.participant_id, "speaker-1")
        self.assertEqual(got.boundaries, ())
        with self.assertRaises(ValueError):
            self._ingest(
                _frame((1, 2, 3, 4)),
                event_t=4.0,
                source_time=None,
                direction="sideways",
            )

    def _ingest(self, frame, *, event_t, source_time, **overrides):
        kwargs = {
            "event_t": event_t,
            "seq": 0,
            "energy": None,
            "capability": CAP,
            "source_id": "mic",
            "source_epoch": 0,
            "source_sequence": 0,
            "observed_time": 100.0,
            "source_time": source_time,
        }
        kwargs.update(overrides)
        return canonicalize_audio_ingress_frame(frame, **kwargs)


class ProviderEvidenceValidationTests(unittest.TestCase):
    """``validate_audio_units_for_provider`` — one contiguous run, and its exact byte size."""

    def setUp(self) -> None:
        self.units = tuple(_run(i, 0.5 * i) for i in range(3))

    def test_empty_evidence_is_refused(self) -> None:
        for empty in ((), []):
            with self.subTest(empty=type(empty).__name__):
                with self.assertRaises(ValueError) as caught:
                    validate_audio_units_for_provider(empty, capability=CAP)
                self.assertEqual(
                    str(caught.exception),
                    "audio judgment requires at least one retained unit",
                )

    def test_the_returned_size_is_the_container_the_payload_will_occupy(self) -> None:
        predicted = validate_audio_units_for_provider(self.units, capability=CAP)
        self.assertEqual(
            predicted, sum(len(unit.chunk.pcm) for unit in self.units) + HEADER
        )
        built = audio_frame_from_units(self.units, capability=CAP)
        self.assertEqual(len(built.encoded), predicted)
        self.assertEqual(
            validate_audio_units_for_provider(self.units[:1], capability=CAP),
            len(self.units[0].chunk.pcm) + HEADER,
        )

    def test_units_that_are_not_homogeneous_canonical_16_bit_are_refused(self) -> None:
        first, second = self.units[0], self.units[1]
        attacks = (
            replace(second, sample_rate=8),
            replace(second, channels=2),
            replace(second, sample_format="u8"),
            replace(second, sample_format=None),
            replace(second, observed_time=None),
            replace(second, source_id=""),
            replace(second, chunk=replace(second.chunk, encoded=b"RIFF", mime="audio/wav")),
            replace(second, chunk=replace(second.chunk, pcm=bytearray(second.chunk.pcm))),
            replace(second, chunk=replace(second.chunk, pcm=b"")),
            replace(second, chunk=b"\0\0"),
        )
        for index, broken in enumerate(attacks):
            with self.subTest(attack=index):
                with self.assertRaises(ValueError) as caught:
                    validate_audio_units_for_provider((first, broken), capability=CAP)
                self.assertEqual(
                    str(caught.exception),
                    "audio judgment units must be homogeneous canonical s16le PCM",
                )

    def test_a_head_unit_without_rate_or_channel_metadata_is_refused(self) -> None:
        raw = AudioUnit(seq=0, t_start=0.0, t_end=0.5, chunk=_frame((1, 2), start=0.0))
        with self.assertRaises(ValueError) as caught:
            validate_audio_units_for_provider((raw,), capability=CAP)
        self.assertEqual(
            str(caught.exception),
            "canonical audio units require rate and channel metadata",
        )

    def test_any_provenance_or_timeline_break_is_refused(self) -> None:
        first = self.units[0]
        attacks = (
            _run(1, 0.5, source_id="other"),
            _run(1, 0.5, source_epoch=1),
            _run(1, 0.5, source_sequence=9),
            _run(2, 0.5),
            _run(1, 0.75),
            _run(1, 0.25),
        )
        for broken in attacks:
            with self.subTest(seq=broken.seq, start=broken.t_start):
                with self.assertRaises(ValueError) as caught:
                    validate_audio_units_for_provider((first, broken), capability=CAP)
                self.assertEqual(
                    str(caught.exception),
                    "audio judgment evidence may not span a discontinuity",
                )

    def test_the_seam_tolerance_is_one_nanosecond(self) -> None:
        first, second = self.units[0], self.units[1]
        near = replace(second, t_start=second.t_start + 5e-10)
        self.assertEqual(
            validate_audio_units_for_provider((first, near), capability=CAP),
            len(first.chunk.pcm) + len(near.chunk.pcm) + HEADER,
        )
        far = replace(second, t_start=second.t_start + 5e-9)
        with self.assertRaises(ValueError) as caught:
            validate_audio_units_for_provider((first, far), capability=CAP)
        self.assertEqual(
            str(caught.exception),
            "audio judgment evidence may not span a discontinuity",
        )

    def test_the_running_total_is_capped_before_the_payload_is_joined(self) -> None:
        payload = sum(len(unit.chunk.pcm) for unit in self.units)
        exact = AudioInputCapability(
            modes=frozenset({"audio_only"}), max_provider_bytes=HEADER + payload
        )
        self.assertEqual(
            validate_audio_units_for_provider(self.units, capability=exact),
            HEADER + payload,
        )
        one_short = AudioInputCapability(
            modes=frozenset({"audio_only"}), max_provider_bytes=HEADER + payload - 1
        )
        with self.assertRaises(ValueError) as caught:
            validate_audio_units_for_provider(self.units, capability=one_short)
        self.assertEqual(
            str(caught.exception),
            "audio provider payload would exceed the compiled output cap",
        )

    def test_a_non_unit_after_the_head_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_audio_units_for_provider(
                (self.units[0], object()), capability=CAP
            )
        self.assertEqual(
            str(caught.exception),
            "audio judgment units must be homogeneous canonical s16le PCM",
        )

    def test_a_non_unit_at_the_head_is_refused_the_same_way(self) -> None:
        """Position must not select the exception class: the runtime's audio gate catches
        ``(RuntimeError, ValueError, TypeError)``, so an ``AttributeError`` from index 0 escaped a
        gate that is supposed to fail closed."""
        for units in ((object(),), (object(), object()), (object(), self.units[0])):
            with self.subTest(length=len(units)):
                with self.assertRaises(ValueError) as caught:
                    validate_audio_units_for_provider(units, capability=CAP)
                self.assertEqual(
                    str(caught.exception),
                    "audio judgment units must be homogeneous canonical s16le PCM",
                )


if __name__ == "__main__":  # pragma: no cover - parity with the sibling audio tests
    unittest.main()
