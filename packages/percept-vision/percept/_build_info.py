"""Build provenance stub — checked into git as the fallback for an unbuilt source checkout.

A wheel or sdist produced via the packaging build step (``python -m build``, ``pip install .``,
...) regenerates this exact file with real values — see ``_build_backend.py`` at the
``percept-vision`` package root. Until that build step runs, every field here reports
``"unreleased"`` rather than a stale or fabricated commit/timestamp. Read this through
``percept.build_info()``, never by importing this module directly.
"""

PACKAGE_VERSION = "unreleased"
GIT_COMMIT = "unreleased"
CONTRACT_VERSION = "unreleased"
BUILD_TIMESTAMP = "unreleased"
