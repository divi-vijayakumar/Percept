"""Committed synthetic AMI NXT sample + the fixtures it produces (board card DS2).

- ``nxt/`` — a handful of NXT-format-representative annotation files (words, dialogue-acts,
  abstractive summary, summary links). NOT the real corpus; downloads nothing.
- ``fixtures/`` — the ``<meeting>/ground_truth.json`` (+ ``splits.json``) tree the transform emits
  from ``nxt/``, so the end-to-end path is unit-testable offline.
- ``_make_sample.py`` — regenerates both, deterministically (no network).
"""
