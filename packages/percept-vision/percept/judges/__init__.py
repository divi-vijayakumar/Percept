"""Judges decide what is true in a request's images.

``base`` holds the pure prompt, parsing and cost rules; ``policy`` sends a request with the
shared deadline, retry and failure handling; ``fake`` is a scripted judge for tests.

Vendor judges: ``gemini``, ``claude`` and ``openai_compatible`` (OpenAI and local
OpenAI-compatible servers). Each imports its SDK only when built, so these names are always
importable; install the matching extra to use one. ``cascade`` chains judges, cheapest first.
"""

from percept.judges import base, fake, policy
from percept.judges._cascade import cascade
from percept.judges._claude import claude
from percept.judges.fake import FakeJudge
from percept.judges._gemini import gemini
from percept.judges._openai_compat import openai_compatible

__all__ = ["FakeJudge", "base", "cascade", "claude", "fake", "gemini", "openai_compatible", "policy"]
