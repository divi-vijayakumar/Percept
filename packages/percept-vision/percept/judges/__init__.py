"""Judges decide what is true in a request's images.

``base`` holds the pure prompt, parsing and cost rules; ``policy`` sends a request with the
shared deadline, retry and failure handling; ``fake`` is a scripted judge for tests.
"""

from percept.judges import base, policy
from percept.judges.fake import FakeJudge

__all__ = ["FakeJudge", "base", "policy"]
