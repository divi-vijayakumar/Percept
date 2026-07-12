"""LabelStore + DispositionLog — the local, keyless flywheel rail (L-1, §4.2/§4.3).

The OSS side accumulates the raw material of the moat from day one, entirely on-box:

* :class:`LabelStore` — append-only JSONL of :class:`~percept.contracts.LabelRecord` at
  ``~/.percept/labels/{deployment_id}.jsonl``. Keyless, local, always; the file just grows. Written by
  three producers (manual ``incident.record_label()``, ASK answers, watch-calibration) into ONE schema.
  The stored row includes the local-only ``frame_ref`` — that is the whole point, the store is local.
* :class:`DispositionLog` — append-only JSONL of every gate ``FIRE``/``ASK`` at
  ``~/.percept/dispositions/{deployment_id}.jsonl`` as ``{incident_id, verdict_ref, disposition, ts}``.
  This is the *escalation meter's* raw feed — distinct from ``Verdict.usage`` (the *compute* meter) —
  so the paid side can price escalations "from data, not a guess." Opt-in; when not wired the runtime
  is byte-for-byte.

Mirrors the shape/spirit of :mod:`percept.persistence`: dumb, local, stdlib-only (``hashlib`` / ``json``
/ ``os`` / ``pathlib``). **No network code** — these sit on the keyless path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from percept.contracts import LabelRecord

__all__ = ["LabelStore", "DispositionLog", "verdict_ref", "default_root"]


def default_root() -> Path:
    """The on-box Percept home: ``~/.percept`` (honors ``$PERCEPT_HOME`` if set)."""
    env = os.environ.get("PERCEPT_HOME")
    return Path(env) if env else Path(os.path.expanduser("~")) / ".percept"


def verdict_ref(verdict: Any) -> str:
    """A deterministic short join-key for a Verdict — the key tying a label/disposition back to the fire
    (and, on the paid side, to the compute meter). A content hash of the verdict's decisive fields, so
    the same verdict always maps to the same ref. Duck-typed: any object with
    ``satisfied``/``confidence``/``evidence`` works (and a missing attr degrades to empty, never raises).
    """
    payload = "|".join(
        str(getattr(verdict, key, "")) for key in ("satisfied", "confidence", "evidence")
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class LabelStore:
    """Append-only JSONL of :class:`LabelRecord` at ``~/.percept/labels/{deployment_id}.jsonl``.

    Keyless, local, always — the file just grows. ``root`` overrides the base directory (tests point it
    at a temp dir); the default is ``~/.percept/labels``. The full record is stored, INCLUDING the
    local-only ``frame_ref`` (:meth:`LabelRecord.to_local_dict`) — that is the point: the store is local.
    """

    def __init__(self, deployment_id: str = "default", *, root: str | Path | None = None) -> None:
        base = Path(root) if root is not None else default_root() / "labels"
        self.deployment_id = deployment_id
        self.path = base / f"{deployment_id}.jsonl"

    def append(self, record: LabelRecord) -> LabelRecord:
        """Append one label as a JSONL line (creating the dir/file on first write). Returns the record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_local_dict(), ensure_ascii=False) + "\n")
        return record

    def iter(self, goal_id: str | None = None) -> Iterator[LabelRecord]:
        """Iterate stored records oldest→newest, optionally filtered to one ``goal_id``. Empty/absent
        file yields nothing; a malformed line is skipped (a bad row never crashes a review)."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = LabelRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if goal_id is None or rec.goal_id == goal_id:
                    yield rec

    def stats(self, goal_id: str | None = None) -> dict[str, Any]:
        """Aggregate counts over the store (optionally scoped to ``goal_id``): total, good/bad split,
        and per-goal / per-source tallies — the corpus's shape at a glance."""
        total = good = bad = 0
        by_goal: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for rec in self.iter(goal_id=goal_id):
            total += 1
            if rec.good:
                good += 1
            else:
                bad += 1
            by_goal[rec.goal_id] = by_goal.get(rec.goal_id, 0) + 1
            by_source[rec.source] = by_source.get(rec.source, 0) + 1
        return {"total": total, "good": good, "bad": bad, "by_goal": by_goal, "by_source": by_source}


class DispositionLog:
    """Append-only JSONL of gate dispositions at ``~/.percept/dispositions/{deployment_id}.jsonl``.

    One ``{incident_id, verdict_ref, disposition, ts}`` row per gate ``FIRE``/``ASK`` — the escalation
    meter's raw feed (distinct from ``Verdict.usage``, the compute meter). Recording it from day one is
    what lets the paid side price escalations from data. Opt-in: when the runtime is not given a
    disposition log, nothing is written and the path is byte-for-byte.
    """

    def __init__(self, deployment_id: str = "default", *, root: str | Path | None = None) -> None:
        base = Path(root) if root is not None else default_root() / "dispositions"
        self.deployment_id = deployment_id
        self.path = base / f"{deployment_id}.jsonl"

    def append(self, *, incident_id: str, verdict_ref: str, disposition: str, ts: float) -> dict[str, Any]:
        """Append one disposition row. ``incident_id`` is ``""`` for an ASK (asks are not incidents)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "incident_id": incident_id,
            "verdict_ref": verdict_ref,
            "disposition": disposition,
            "ts": float(ts),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def iter(self) -> Iterator[dict[str, Any]]:
        """Iterate disposition rows oldest→newest. Empty/absent file yields nothing; bad lines skip."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
