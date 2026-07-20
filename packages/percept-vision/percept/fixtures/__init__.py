"""Bundled deterministic OFFLINE FIXTURES — the keyless first run (card RB-OFFLINE).

Each ``*.json`` here is a self-contained, byte-stable replay: a goal, an ordered list of frame
tokens, the scripted verdict for each token, and the events the runtime is expected to produce.
They ship INSIDE the wheel (``[tool.setuptools.package-data]``) so ``pip install percept-vision``
followed by ``mode="offline"`` runs the real fovea -> judge -> gate path end to end with no API key.

A fixture demonstrates the RUNTIME. It does not, and must not, look like perception: every verdict
it returns is a value a human wrote into a JSON file. The loader/backend live in
:mod:`percept.offline`; this package is the data plus the resource-lookup root.
"""

from __future__ import annotations

__all__: list[str] = []
