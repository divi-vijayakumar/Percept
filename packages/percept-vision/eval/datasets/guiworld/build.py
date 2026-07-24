"""Regenerate the committed GUI-World fixture tree + split + pre-registration from the sample.

Run from ``packages/percept-vision``::

    python3 -m eval.datasets.guiworld.build

Deterministic and offline: it reads ``testdata/guiworld_sample.json``, transforms it to fixtures,
writes ``testdata/fixtures/<fixture_id>/ground_truth.json`` + ``testdata/fixtures/splits.json``, and
freezes ``testdata/preregistration.json``. The tests regenerate into a temp dir and assert the
committed tree matches, so a stale commit is caught.
"""

from __future__ import annotations

import json
from pathlib import Path

from .split import build_preregistration, build_session_split
from .transform import load_sessions, sessions_to_fixtures, write_fixture_tree

HERE = Path(__file__).resolve().parent
SAMPLE_PATH = HERE / "testdata" / "guiworld_sample.json"
BARS_PATH = HERE / "bars.json"
FIXTURES_DIR = HERE / "testdata" / "fixtures"
PREREG_PATH = HERE / "testdata" / "preregistration.json"


def build(
    *,
    sample_path: Path = SAMPLE_PATH,
    fixtures_dir: Path = FIXTURES_DIR,
    bars_path: Path = BARS_PATH,
    prereg_path: Path | None = PREREG_PATH,
) -> Path:
    """Build the fixture tree (+ split) from ``sample_path`` into ``fixtures_dir``.

    Also freezes the pre-registration to ``prereg_path`` (pass ``None`` to skip). Returns
    ``fixtures_dir``.
    """
    sessions = load_sessions(sample_path)
    fixtures = sessions_to_fixtures(sessions)
    splits = build_session_split(fixtures)
    write_fixture_tree(fixtures, fixtures_dir, splits=splits)

    if prereg_path is not None:
        prereg = build_preregistration(fixtures_dir, bars_path=bars_path)
        prereg_path.write_text(json.dumps(prereg.to_dict(), indent=2, sort_keys=True) + "\n")
    return fixtures_dir


def main() -> int:
    out = build()
    print(f"wrote fixtures -> {out}")
    print(f"wrote pre-registration -> {PREREG_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
