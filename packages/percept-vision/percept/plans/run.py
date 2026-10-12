"""Evaluation driver: run configurations you declare over your labelled clips, and score them.

Experimental: names and outputs may change before they are stable.

Every function takes a :class:`~percept.plans.Compiled` plan with one source, the
:class:`~percept.plans.labels.Labels`, the clip files to run (for example
``manifest.clips["dev"]``) and ``video_root``, the directory that holds them. Each clip is
opened with :func:`percept.open_source` at the plan source's ``max_side``. ``fps`` is the
processing rate (None: the Monitor's default). ``label_of`` maps a condition id to its label in
the labels (default: the id itself); a condition it leaves out is not scored.

* :func:`attention_only` runs only the attention step (:mod:`percept.attention`, the one the
  Monitor runs): no judge calls, no cost. Its :class:`~percept.plans.score.Fire` records are the
  looks made due, so :func:`~percept.plans.score.evaluate` reports the candidate stage of the
  funnel: how many occurrences any look would have seen.
* :func:`run` runs the full runtime, one :class:`~percept.Monitor` per clip, with the plan's judge
  wrapped in :class:`CachingJudge` (a hard dollar cap and a reply cache).
* :func:`sweep` runs each configuration you declare on the dev or validation split and returns
  their scores side by side, in your order. It does not rank or choose between them.
* :func:`register` freezes a plan before the test split is run, and :func:`score_split` scores
  the test split only for a registered plan whose hashes still match.

Shell module: it reads video and runs the Monitor.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

from percept import attention
from percept.contracts import JudgeRequest, JudgeResult, Observation
from percept.delivery import MemoryStore
from percept.judges import base
from percept.monitor import DEFAULT_FPS, Monitor
from percept.plans import EVENT_CLIP_S, Compiled
from percept.plans.labels import SPLITS, Labels, SplitManifest, _basename, _sha
from percept.plans.score import MIN_CONFIDENCE, ClipRun, Drop, Fire, Look, Prediction, Score, evaluate
from percept.sources import open_source
from percept.trace import read as read_trace
from percept.triggers import QualityGate, SceneMonitor, every, on_change

REGISTRATION_FORMAT = "percept.registration/v1alpha1"


def _setup(compiled: Compiled, labels: Labels, clips: Sequence[str], fps: Optional[float],
           label_of: Optional[Mapping[str, str]]) -> tuple[float, int, dict[str, str]]:
    """The processing rate, the source's max_side and condition id -> label; checks the clips."""
    if len(compiled.sources) != 1:
        raise ValueError(f"evaluation runs a plan with one source; this plan has {len(compiled.sources)}")
    if fps is not None and not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be a positive number or None, got {fps!r}")
    for file in clips:
        labels.clip(file)       # KeyError: a clip that is not in the labels
    ids = [c.id for c in compiled.conditions]
    names = {i: i for i in ids} if label_of is None else {i: label_of[i] for i in ids if i in label_of}
    return fps or DEFAULT_FPS, compiled.sources[0]["max_side"], names


def _cooldown(compiled: Compiled) -> float:
    """The shortest rearm of the plan's conditions: the recall ceiling is then an upper bound."""
    return min((c.rearm_s for c in compiled.conditions), default=0.0)


# --------------------------------------------------------------------------- attention only


def attention_only(compiled: Compiled, labels: Labels, clips: Sequence[str], *, video_root: Union[str, os.PathLike],
                   fps: Optional[float] = None, label_of: Optional[Mapping[str, str]] = None) -> list[ClipRun]:
    """The trigger stage alone: one :class:`ClipRun` per clip whose ``fires`` are the looks the
    attention step makes due (trigger looks and the settled look after a scene reset), at the clip
    time of their frame. No judge is called; ``cost_usd`` is 0 and ``compute_s`` is process CPU
    seconds. Score it with :func:`~percept.plans.score.evaluate`: ``funnel["candidate"]`` counts
    the occurrences a look would have seen."""
    rate, max_side, names = _setup(compiled, labels, clips, fps, label_of)
    root = os.fspath(video_root)

    async def attend(file: str) -> ClipRun:
        conds = {c.id: c for c in compiled.conditions}
        # The Monitor's defaults for a condition without a trigger, built the way it builds them.
        groups, tlabels = attention.build(conds.values(), default_change=on_change(), default_heartbeat=every(300))
        quality, scene, state = QualityGate(), SceneMonitor(), attention.State()
        fires: list[Fire] = []
        anchor, t, cpu = None, 0.0, time.process_time()
        async with open_source(os.path.join(root, file), fps=rate, max_side=max_side) as source:
            async for frame in source.frames():
                anchor = frame.t_wall if anchor is None else anchor
                t = frame.t_wall - anchor
                s, state = attention.step(frame, state, groups=groups, conditions=conds, labels=tlabels,
                                          quality=quality, scene=scene, period=1.0 / rate)
                due = [lk.condition_id for lk in s.looks] + (list(conds) if s.settled else [])
                fires += [Fire(names[cid], t) for cid in due if cid in names]
        return ClipRun(_basename(file), t + 1.0 / rate, fires=tuple(fires), compute_s=time.process_time() - cpu,
                       cooldown_s=_cooldown(compiled))

    async def main() -> list[ClipRun]:
        return [await attend(f) for f in clips]

    return asyncio.run(main())
