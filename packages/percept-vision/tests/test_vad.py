"""EnergyVad — the Python utterance endpointer (port of percept-edge/src/vad.js).

The JS `EnergyVad` is the oracle: this asserts the Python twin fires an endpoint on the SAME frame
indices for the same energy sequences (started on the rising edge, endpoint after `hangover_frames`
trailing-silence frames following a `min_speech_frames`-long speech run), and drops a too-short blip.
"""
from __future__ import annotations

import math
import unittest

from percept.runtime import EnergyVad, endpoint_energy_sequence, frame_energy
from percept.runtime._vad import DEFAULT_CHUNK_S, hangover_frames_for
from percept.runtime.worker import _amplitude_threshold_to_energy


class FrameEnergyTests(unittest.TestCase):
    def test_mean_of_squares(self):
        self.assertEqual(frame_energy([]), 0.0)
        self.assertEqual(frame_energy([0.0, 0.0]), 0.0)
        self.assertAlmostEqual(frame_energy([1.0, -1.0]), 1.0)
        self.assertAlmostEqual(frame_energy([0.5, 0.5]), 0.25)


class EndpointSequenceTests(unittest.TestCase):
    def test_one_utterance_endpoints_once(self):
        # 3 silence, 5 speech (>= min_speech=3), 9 silence (>= hangover=8 -> endpoint on the 8th).
        energies = [0.0] * 3 + [0.5] * 5 + [0.0] * 9
        # started at frame 3; 8th trailing-silence frame is index 3+5+7 = 15.
        self.assertEqual(endpoint_energy_sequence(energies), [15])

    def test_short_blip_dropped(self):
        # Only 2 speech frames (< min_speech=3): the run is dropped, no endpoint, endpointer re-arms.
        energies = [0.0] * 3 + [0.5] * 2 + [0.0] * 9
        self.assertEqual(endpoint_energy_sequence(energies), [])

    def test_two_utterances(self):
        energies = [0.5] * 4 + [0.0] * 8 + [0.5] * 4 + [0.0] * 8
        # first: started@0, endpoint@11 (4 speech + 8 silence -> 4+7); second: started@12, endpoint@23.
        self.assertEqual(endpoint_energy_sequence(energies), [11, 23])

    def test_no_endpoint_without_hangover(self):
        # Speech that never gets enough trailing silence never endpoints (turn still open).
        energies = [0.5] * 5 + [0.0] * 4
        self.assertEqual(endpoint_energy_sequence(energies), [])


class EnergyVadStateTests(unittest.TestCase):
    def test_started_and_speaking_flags(self):
        vad = EnergyVad(threshold=0.01, hangover_frames=8, min_speech_frames=3)
        self.assertFalse(vad.speaking)
        self.assertFalse(vad.push(0.0).started)  # silence: nothing

        ev = vad.push(0.5)  # rising edge
        self.assertTrue(ev.started)
        self.assertTrue(ev.speech)
        self.assertTrue(vad.speaking)

        self.assertFalse(vad.push(0.5).started)  # already speaking: not a new start (speech_run=2)
        self.assertFalse(vad.push(0.5).started)  # speech_run=3 -> long enough to count as an utterance
        for _ in range(7):  # 7 more silence frames -> not yet the 8th
            ev = vad.push(0.0)
            self.assertFalse(ev.endpoint)
        ev = vad.push(0.0)  # the 8th trailing silence frame -> endpoint
        self.assertTrue(ev.endpoint)
        self.assertFalse(vad.speaking)

    def test_custom_threshold(self):
        vad = EnergyVad(threshold=0.3, min_speech_frames=1, hangover_frames=2)
        self.assertFalse(vad.push(0.2).speech)  # below threshold
        self.assertTrue(vad.push(0.4).speech)   # above threshold

    def test_reset_rearms(self):
        vad = EnergyVad()
        vad.push(0.5)
        self.assertTrue(vad.speaking)
        vad.reset()
        self.assertFalse(vad.speaking)
        self.assertEqual(vad.push(0.0).frame, 0)  # frame index restarts

    def test_invalid_config_and_energy_fail_closed_like_javascript(self):
        for kwargs in (
            {"threshold": math.nan},
            {"threshold": 1.1},
            {"hangover_frames": 0},
            {"hangover_frames": 1.5},
            {"min_speech_frames": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EnergyVad(**kwargs)
        vad = EnergyVad()
        for energy in (math.nan, math.inf, -0.1, 1.1, True):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                vad.push(energy)


class HangoverUnitsTests(unittest.TestCase):
    """AUDIO-VAD-HANGOVER-UNITS — the two knobs whose units are not the units they look like.

    ``utterance_threshold`` is an AMPLITUDE that the runtime SQUARES into energy, and
    ``hangover_frames`` is a FRAME COUNT whose wall-clock meaning depends on ``chunk_s``. Neither
    fact was written down anywhere near the parameters. These tests are the executable statement of
    both, so the numbers in the docstrings cannot drift away from the behaviour.
    """

    def test_amplitude_knob_is_squared_into_the_energy_domain(self):
        # The shipped utterance_threshold=0.01 is energy 1e-4 — two orders of magnitude below what
        # the number looks like. A chunk at energy 5e-4 is therefore SPEECH, not silence.
        energy = _amplitude_threshold_to_energy(0.01, name="utterance_threshold")
        self.assertAlmostEqual(energy, 1e-4)
        self.assertTrue(EnergyVad(threshold=energy).push(5e-4).speech)
        self.assertFalse(EnergyVad(threshold=0.01).push(5e-4).speech)

    def test_default_hangover_is_two_seconds_at_the_runtime_chunk_cadence(self):
        # 8 frames x 0.25 s/chunk. This is the number the defect is about: continuous narration does
        # not contain 2 s of silence, so it barely endpoints at all.
        self.assertEqual(DEFAULT_CHUNK_S, 0.25)
        self.assertEqual(EnergyVad().hangover_frames * DEFAULT_CHUNK_S, 2.0)

    def test_hangover_frames_for_seconds_round_trips_against_a_cadence(self):
        self.assertEqual(hangover_frames_for(2.0), 8)
        self.assertEqual(hangover_frames_for(0.6), 2)
        self.assertEqual(hangover_frames_for(0.5, chunk_s=0.1), 5)
        self.assertEqual(hangover_frames_for(0.01), 1)  # never rounds down to a disabled hangover
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                hangover_frames_for(bad)
        with self.assertRaises(ValueError):
            hangover_frames_for(1.0, chunk_s=0)

    def test_continuous_speech_endpoints_only_when_the_hangover_is_stated_in_seconds(self):
        # Narration: long speech runs separated by ~0.75 s pauses (3 chunks) — a real turn boundary a
        # human would hear, and the shape the measured corpus actually has.
        pause = [0.0] * 3
        energies = ([0.5] * 12 + pause) * 3
        # The shipped 2.00 s hangover never sees 2 s of silence, so the whole passage is ONE
        # never-finished turn. No `utterance_threshold` can fix this: it is the hangover, in frames.
        self.assertEqual(endpoint_energy_sequence(energies), [])
        # Stated in seconds against the real cadence, the same audio endpoints at every pause.
        self.assertEqual(
            len(
                endpoint_energy_sequence(
                    energies, hangover_frames=hangover_frames_for(0.6)
                )
            ),
            3,
        )

    def test_the_default_is_unchanged_and_stays_the_javascript_twin(self):
        # Deliberately NOT retuned: this class is a byte-exact port whose JS tests are the shared
        # oracle, so the default cannot move on one side alone.
        self.assertEqual(EnergyVad().hangover_frames, 8)
        self.assertEqual(EnergyVad().min_speech_frames, 3)
        self.assertEqual(EnergyVad().threshold, 0.01)


if __name__ == "__main__":
    unittest.main()
