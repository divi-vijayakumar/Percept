"""Sprite montage primitive (Build 6) — L1 tests, plus its J2 DEPRECATION in the plan path.

make_sprite itself is still exercised on synthetic numpy frames (the montage math — the primitive is
untouched and importable). The agent-side assembly tests now assert the opposite of Build 6: an ordered
window beat the montage unconditionally (H0 0.68 -> 0.73), so ``temporal="sprite"`` is shimmed to
``"window"`` with a DeprecationWarning and NO montage is built on the judging path.
"""

from __future__ import annotations

import asyncio
import unittest
import warnings

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


class SpriteDeprecatedInThePlanPathTests(unittest.TestCase):
    """the implementation: the plan path no longer builds a montage for ANY concern. An authored
    ``temporal="sprite"`` is shimmed to an ordered window (strictly better: H0 0.68 -> 0.73)."""

    def _last_call(self, frames):
        rec = _RecordingVision()
        a = Percept.create(vision=rec, discover_plugins=False)
        a.add_goal(_sprite_goal())

        async def go():
            for i, f in enumerate(frames):
                await a.perceive_judged(f, t=float(i))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(go())
        return rec.calls[-1], caught

    @unittest.skipUnless(_HAVE, "needs numpy for real image frames")
    def test_sprite_no_longer_collapses_context_to_a_montage(self):
        frames = [np.full((8, 8, 3), i * 10, dtype=np.uint8) for i in range(4)]
        last, _ = self._last_call(frames)
        # the 3 earlier frames reach the judge AS FRAMES, ordered oldest→newest — not one grid image
        self.assertEqual(len(last["frames"]), 3)
        for sent, original in zip(last["frames"], frames[:3]):
            self.assertTrue(np.array_equal(sent, original))

    def test_sprite_warns_and_behaves_as_window(self):
        last, caught = self._last_call(["f0", "f1", "f2", "f3"])
        self.assertEqual(last["frames"], ["f0", "f1", "f2"])  # window = ALL earlier frames, ordered
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) and "sprite" in str(w.message)
                for w in caught)
        )

    def test_make_sprite_is_never_called_from_the_plan_path(self):
        import percept.sprite as sprite_mod

        original = sprite_mod.make_sprite
        sprite_mod.make_sprite = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the plan path must not montage frames (J2)")
        )
        try:
            last, _ = self._last_call(["f0", "f1", "f2", "f3"])
        finally:
            sprite_mod.make_sprite = original
        self.assertEqual(last["frames"], ["f0", "f1", "f2"])


if __name__ == "__main__":
    unittest.main()
