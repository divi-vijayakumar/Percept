"""EnergyVad — the Python utterance endpointer (port of percept-edge/src/vad.js).

The JS `EnergyVad` is the oracle: this asserts the Python twin fires an endpoint on the SAME frame
indices for the same energy sequences (started on the rising edge, endpoint after `hangover_frames`
trailing-silence frames following a `min_speech_frames`-long speech run), and drops a too-short blip.
"""
from __future__ import annotations

import unittest

from percept.runtime import EnergyVad, endpoint_energy_sequence, frame_energy


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


if __name__ == "__main__":
    unittest.main()
