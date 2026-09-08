"""Public demo/docs drift guards for the governing episode context contract.

``shadow.fold.data.fold`` is intentionally the shadow observer's always-unbound answer.  The real
episode fold rides beside it as ``fold_episode_id`` / ``fold_match``.  These source-level guards keep
the static demos from presenting the shadow placeholder as the state that governed a notification,
and keep the configuration guide aligned with the unified vision/fused judgment pipeline.
"""

from __future__ import annotations

import unittest
from pathlib import Path


_REPO = Path(__file__).resolve().parents[3]
_DEMOS = (
    _REPO / "docs/demo/demo-01-video-watch.html",
    _REPO / "docs/demo/demo-02-screen-watch.html",
    _REPO / "docs/demo/demo-04-stream-uploaded-video.html",
)
_EPISODES_DOC = _REPO / "docs/configuration/episodes.md"
_DEMO_README = _REPO / "docs/demo/README.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise unittest.SkipTest(f"{path} is not present in this checkout")
    return path.read_text(encoding="utf-8")


class DemoContextLedgerUsesTheGoverningFold(unittest.TestCase):
    def test_every_demo_reads_the_governing_episode_siblings(self) -> None:
        for path in _DEMOS:
            with self.subTest(demo=path.name):
                source = _read(path)
                self.assertIn("fold_episode_id", source)
                self.assertIn("fold_match", source)

    def test_no_demo_reads_an_episode_identity_from_the_shadow_fold(self) -> None:
        forbidden = ("data.fold.episode_id", "fo.episode_id", "fold.episode_id")
        for path in _DEMOS:
            with self.subTest(demo=path.name):
                source = _read(path)
                for expression in forbidden:
                    self.assertNotIn(
                        expression,
                        source,
                        f"{path.name} must read governing fold_episode_id, not {expression}",
                    )

    def test_demo_guide_distinguishes_shadow_from_governing_fields(self) -> None:
        source = _read(_DEMO_README)
        self.assertIn("governing `fold_episode_id` / `fold_match` siblings", source)
        self.assertIn("The nested **shadow fold is advisory**", source)


class EpisodeConfigurationDocMatchesTheUnifiedPipeline(unittest.TestCase):
    def test_vision_and_fused_are_documented_as_episode_keyed(self) -> None:
        source = _read(_EPISODES_DOC)
        self.assertIn("| `percept.run()` | vision, fused | yes | yes |", source)
        self.assertIn(
            "Vision and fused Watches under `percept.run()` use that same episode fingerprint",
            source,
        )

    def test_the_retired_no_episode_key_claim_cannot_return(self) -> None:
        source = _read(_EPISODES_DOC).lower()
        self.assertNotIn("on the vision and fused paths there is no episode component", source)
        self.assertNotIn("one slot spans a close and a reopen", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
