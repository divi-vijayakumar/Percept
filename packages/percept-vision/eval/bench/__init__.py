"""Percept Benchmark v1 harness — the Snapshot-Card emitter (BENCHMARK_PLAN.md, Phase A).

`types.TrackResult` is the contract every track scorer returns; `card.render_card` consumes a list of them
and emits a stamped §7 Snapshot Card; `run` is the CLI that scores the tracks and prints the card.
"""

from .types import CONFIGS, TrackResult

__all__ = ["TrackResult", "CONFIGS"]
