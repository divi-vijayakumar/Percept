"""Judges decide what is true in a request's images.

``base`` holds the pure prompt, parsing and cost rules; ``policy`` sends a request with the
shared deadline, retry and failure handling; ``fake`` is a scripted judge for tests.

Vendor judges: ``gemini``, ``claude`` and ``openai_compatible`` (OpenAI and local
OpenAI-compatible servers). Each imports its SDK only when built, so these names are always
importable; install the matching extra to use one::

    pip install "percept-vision[video,gemini]"     # or [video,claude], [video,openai]

Without it, building the judge raises ``ImportError: this judge needs the 'google.genai'
package: pip install 'percept-vision[gemini]'`` (``'anthropic'`` / ``[claude]``, ``'openai'`` /
``[openai]`` likewise). ``cascade`` chains judges, cheapest first.

Offline tests: ``FakeJudge(script)`` needs no key or network. ``script(request)`` is called once
per provider request and returns ``{condition_id: (answer, confidence, reason)}`` with
``answer`` one of ``"yes"``, ``"no"``, ``"unknown"``; ``request.images`` holds the same BGR
``uint8`` arrays a real judge would be sent (cropped to the zones), ``request.items`` the
conditions asked about and ``request.purpose`` is ``"detect"`` or ``"verify"``::

    def script(request):
        lit = (request.images[-1][..., 2] > 200).any()
        return {"lamp_on": ("yes", 0.9, "the lamp is lit") if lit else ("no", 0.9, "it is dark")}

    monitor = percept.Monitor(percept.judges.FakeJudge(script), conditions, store=percept.MemoryStore())

See :class:`~percept.judges.fake.FakeJudge` for every script form.
"""

from percept.judges import base, fake, policy
from percept.judges._cascade import cascade
from percept.judges._claude import claude
from percept.judges.fake import FakeJudge
from percept.judges._gemini import gemini
from percept.judges._openai_compat import openai_compatible

__all__ = ["FakeJudge", "base", "cascade", "claude", "fake", "gemini", "openai_compatible", "policy"]
