"""Fresh-clone corpus selection for the stage-validation entry points."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import stage_metrics as SM  # noqa: E402


def test_default_corpus_is_the_tracked_current_revision(monkeypatch):
    monkeypatch.delenv("PERCEPT_SV_CORPUS", raising=False)
    path = SM.corpus_path()
    assert path.name == "corpus.v1.2.json"
    assert path.is_file()


def test_historical_corpus_override_remains_explicit_and_relative_to_stage_dir(monkeypatch):
    monkeypatch.setenv("PERCEPT_SV_CORPUS", "corpus.v1.1.json")
    assert SM.corpus_path() == SM.HERE / "corpus.v1.1.json"
