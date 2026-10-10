"""Opt-in live smoke tests: real requests to each vendor judge, kept tiny.

Skipped unless ``PERCEPT_LIVE_TESTS=1`` and the vendor's key is set. The first sends one 64x64
solid red JPEG, one condition and at most 64 output tokens. The second runs the judge inside a
``Monitor`` on a short generated clip (a red block entering a zone), under a request limit and a
dollar cap. Model ids come from ``PERCEPT_LIVE_<VENDOR>_MODEL``; the defaults below are only
examples of small vision models.
"""

import asyncio
import os

import numpy as np
import pytest

from percept import MemoryStore, Monitor, PushSource, on_change
from percept.contracts import Condition, JudgeItem, JudgeRequest, PriceModel
from percept.judges import claude, gemini, openai_compatible

LIVE = os.environ.get("PERCEPT_LIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="live tests run only with PERCEPT_LIVE_TESTS=1")

VENDORS = {
    "gemini": (gemini, ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "google.genai", "gemini-2.5-flash-lite"),
    "claude": (claude, ("ANTHROPIC_API_KEY",), "anthropic", "claude-haiku-4-5"),
    "openai": (openai_compatible, ("OPENAI_API_KEY",), "openai", "gpt-4o-mini"),
}


@pytest.mark.parametrize("vendor", sorted(VENDORS))
def test_one_tiny_live_request(vendor):
    factory, keys, sdk, default_model = VENDORS[vendor]
    if not any(os.environ.get(k) for k in keys):
        pytest.skip(f"set {' or '.join(keys)} to run")
    pytest.importorskip(sdk)
    judge = factory(os.environ.get(f"PERCEPT_LIVE_{vendor.upper()}_MODEL", default_model))
    red = np.zeros((64, 64, 3), np.uint8)
    red[..., 2] = 255  # BGR
    request = JudgeRequest("live-1", "smoke", 0, (JudgeItem(Condition("red", "The image is mostly red")),),
                           (red,), (0.0, 0.0), deadline_s=30.0, max_output_tokens=64)
    result = asyncio.run(judge.judge(request))
    (ob,) = result.observations
    assert ob.failure is None, ob.reason
    assert ob.answer in ("yes", "no", "unknown")
    usage = result.usage[0].usage
    assert usage.tokens_in > 0 and usage.tokens_out > 0


# An assumed price, only so the monitor reserves and settles a non-zero cost. It is not any
# vendor's price; set ``price=`` from your own contract in real use.
LIVE_PRICE = PriceModel(usd_per_mtok_in=1.0, usd_per_mtok_out=4.0, tokens_per_image=300)
LIVE_CAP_USD = 0.01


def clip(n: int, enters_at: int) -> list[np.ndarray]:
    """A grey textured wall; a red block (BGR) enters the right half at frame ``enters_at`` and stays."""
    y, x = np.mgrid[0:96, 0:128]
    wall = ((x * 7 + y * 13) % 120 + 60).astype(np.uint8)
    frames = []
    for i in range(n):
        f = np.dstack([wall, wall, wall])
        if i >= enters_at:
            f[30:70, 76:116] = (20, 20, 230)
        frames.append(f)
    return frames


@pytest.mark.parametrize("vendor", sorted(VENDORS))
def test_a_real_judge_inside_a_monitor_reports_met_with_a_settled_cost(vendor):
    factory, keys, sdk, default_model = VENDORS[vendor]
    if not any(os.environ.get(k) for k in keys):
        pytest.skip(f"set {' or '.join(keys)} to run")
    pytest.importorskip(sdk)
    judge = factory(os.environ.get(f"PERCEPT_LIVE_{vendor.upper()}_MODEL", default_model), price=LIVE_PRICE)
    zone = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
    red = Condition("red", "a red block is in the frame", zones=(zone,), trigger=on_change(),
                    min_confidence=0.5)

    async def main():
        # Hard limits for an opt-in test: few attempts a minute and a dollar cap well under a cent.
        async with Monitor(judge, [red], store=MemoryStore(), max_requests_per_min=6,
                           max_cost_per_hour=LIVE_CAP_USD) as m:
            src = PushSource("live-clip")
            watch = m.watch(src, fps=None)
            for f in clip(12, enters_at=4):
                src.push(f)
                await asyncio.sleep(0.2)
            got = None
            async for delivery in watch:
                got = delivery.decision
                break
            return got, m.stats()

    decision, stats = asyncio.run(asyncio.wait_for(main(), 120))
    assert decision is not None and decision.type == "met" and decision.condition_id == "red"
    cam = stats["streams"]["live-clip"]
    assert 0 < cam["cost_usd"] < LIVE_CAP_USD
    assert cam["conditions"]["red"]["cost_usd"] == pytest.approx(cam["cost_usd"])
