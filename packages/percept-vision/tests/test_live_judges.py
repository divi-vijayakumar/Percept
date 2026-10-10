"""Opt-in live smoke tests: one tiny real request per vendor judge.

Skipped unless ``PERCEPT_LIVE_TESTS=1`` and the vendor's key is set. Each sends one 64x64
solid red JPEG, one condition and at most 64 output tokens. Model ids come from
``PERCEPT_LIVE_<VENDOR>_MODEL``; the defaults below are only examples of small vision models.
"""

import asyncio
import os

import numpy as np
import pytest

from percept.contracts import Condition, JudgeItem, JudgeRequest
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
