"""Sprite montage primitive (Build 6) — L1 tests.

make_sprite is exercised on synthetic numpy frames (the montage math); the agent's assembly is exercised
with a recording vision stub to prove temporal="sprite" sends ONE montage image for the context (bounded
token cost) while non-image frames degrade gracefully to a plain strip.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept

try:
    import numpy as np
    from percept.sprite import make_sprite
    _HAVE = True
except Exception:  # pragma: no cover - numpy is an extra, absent in the L1 stdlib lane
    _HAVE = False


@unittest.skipUnless(_HAVE, "needs numpy for the sprite montage math")
class MakeSpriteTests(unittest.TestCase):
    def test_montages_n_frames_into_one_grid_image(self):
        frames = [np.full((10, 8, 3), i * 20, dtype=np.uint8) for i in range(4)]  # 4 same-size frames
        sprite = make_sprite(frames, max_cols=2, pad=2)
        self.assertIsNotNone(sprite)
        # 4 tiles in a 2x2 grid of 10x8 tiles + 2px padding gutters
        self.assertEqual(sprite.shape, (2 * 10 + 3 * 2, 2 * 8 + 3 * 2, 3))
        self.assertEqual(sprite.dtype, np.uint8)

    def test_mismatched_sizes_resized_into_one_image(self):
        frames = [np.zeros((10, 10, 3), np.uint8), np.zeros((5, 20, 3), np.uint8)]
        sprite = make_sprite(frames, max_cols=2)
        self.assertIsNotNone(sprite)
        self.assertEqual(sprite.ndim, 3)  # a single regular grid despite mismatched inputs

    def test_single_frame_is_a_one_tile_sprite(self):
        sprite = make_sprite([np.zeros((6, 6, 3), np.uint8)])
        self.assertEqual(sprite.shape[2], 3)

    def test_non_image_frames_return_none(self):
        self.assertIsNone(make_sprite(["f0", "f1"]))  # strings -> not images -> caller falls back to strip
        self.assertIsNone(make_sprite([]))


class _RecordingVision:
    def __init__(self):
        self.calls: list[dict] = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None, frames=None, audio=None):
        self.calls.append({"frame": frame, "frames": frames})
        return Verdict(satisfied=False, confidence=0.5, evidence="stub")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _sprite_goal():
    return Goal(id="g", condition="c", say="s",
                state={"type": "watch", "count": 0, "perception": {"temporal": "sprite", "n_frames": 4}})


@unittest.skipUnless(_HAVE, "needs numpy for the sprite montage math")
class SpriteAssemblyTests(unittest.TestCase):
    def test_sprite_collapses_context_to_one_montage_image(self):
        rec = _RecordingVision()
        a = Percept.create(vision=rec, discover_plugins=False)
        a.add_goal(_sprite_goal())

        async def go():
            for i in range(4):  # push 4 numpy frames; the 4th is current, the first 3 are context
                await a.perceive_judged(np.full((8, 8, 3), i * 10, dtype=np.uint8), t=float(i))

        asyncio.run(go())
        last = rec.calls[-1]
        self.assertEqual(len(last["frames"]), 1)  # the N context frames folded into ONE montage image
        self.assertIsInstance(last["frames"][0], np.ndarray)
        self.assertEqual(last["frames"][0].ndim, 3)

    def test_sprite_degrades_to_strip_for_non_image_frames(self):
        rec = _RecordingVision()
        a = Percept.create(vision=rec, discover_plugins=False)
        a.add_goal(_sprite_goal())

        async def go():
            for tok in ("f0", "f1", "f2", "f3"):
                await a.perceive_judged(tok, t=0.0)

        asyncio.run(go())
        last = rec.calls[-1]
        self.assertTrue(all(isinstance(f, str) for f in last["frames"]))  # graceful fallback, no crash


if __name__ == "__main__":
    unittest.main()
