"""One restore contract: **validate -> coerce/skip per entry (logged) -> swap.**

Every ``restore()`` in this package rehydrates durable state a persistence backend handed back, and
a persistence backend can hand back anything: a truncated write, a type-confused record, a payload
written by a build two releases older. There is exactly one safe shape for that, and
:meth:`percept.goals.GoalRegistry.restore` was the only method in the tree that had it — build the
replacement into locals, validate it, and swap it in at the end.

The other four restores did not, and each failed in its own way:

* a bare sequence of assignments (``Gate.restore``) commits the first fields and then raises out of
  the last, leaving one half of a two-belt suppression in place and the other half empty. That is
  strictly worse than losing both belts: an armed-latch-without-refractory-memory re-notifies on the
  next confident-false blip, which is the exact duplicate notification the two belts exist to stop;
* ``self._entities = {}`` first, then build (``EntityGraph.restore``) turns a mid-loop ``KeyError``
  into a WIPED graph rather than a half-restored one;
* ``if not snap: return`` (``IdempotencyLedger.restore``) screens only FALSY input, so a truthy
  non-iterable reaches the ``for`` and propagates out of a method whose docstring promises a corrupt
  snapshot can never block a restart.

The policy this module implements
---------------------------------
1. **A missing field is legal.** ``None``/absent restores as empty — that is how a snapshot written
   before a field existed is read, and it must stay a no-op rather than an error.
2. **A bad ENTRY is skipped and logged**, never reinterpreted. The ledger already worked this way
   (per-item ``except ... : continue``); this generalises it and adds the log line, so a skipped
   entry is observable instead of silent.
3. **A bad TOP-LEVEL shape raises :class:`SnapshotError`** — the one case where there is nothing to
   salvage and the caller must decide. Callers on a restart path catch it and degrade to "start
   blank" rather than propagating; a caller that wants the failure loud can let it out.

:class:`SnapshotError` subclasses ``ValueError`` deliberately: the restore path in
``percept/agent.py`` already funnels legacy-payload failures through ``except (TypeError,
ValueError)``, and a typed error that escapes an existing handler would be a new crash rather than a
fix.

Never reinterpret a malformed entry
-----------------------------------
The sharpest lesson encoded here is :func:`key_tuple`. A gate key serializes as ``[goal_id, entity]``
and a pair serializes as ``[[goal_id, entity], armed]``. Lose one level of nesting and the pair
``['goal-a', 'e1']`` still unpacks cleanly as ``k='goal-a', v='e1'`` — and ``tuple('goal-a')`` is a
six-CHARACTER tuple while ``bool('e1')`` is ``True``. Nothing raises; the real key is simply absent,
and an absent gate key defaults to ARMED, so a spent edge silently re-opens and the watch
re-notifies. :func:`key_tuple` refuses a bare string outright, so the flattened entry is skipped with
a log instead of manufacturing a key nobody will ever evaluate.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

from .observability import get_logger, log

_LOGGER = "percept.snapshot"


class SnapshotError(ValueError):
    """A snapshot whose TOP-LEVEL shape is unusable — raised only when nothing can be salvaged.

    Subclasses ``ValueError`` on purpose (see the module docstring): existing restore paths catch
    ``ValueError`` and a sibling exception type would slip past them.

    :attr:`field` names the offending field (``"armed"``, ``"last_fire"``, ``"<root>"`` …) so the
    message a 2 a.m. operator reads says WHICH part of the payload was unusable, which is the one
    thing the untyped ``AttributeError``/``TypeError``/``ValueError`` this replaces never did.
    """

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field = field


def _skip(field: str, index: int, entry: Any, reason: str) -> None:
    """Record a per-entry skip. Skipping is a data loss, so it is never silent (ADR-001 §9)."""
    log(
        get_logger(_LOGGER),
        "snapshot.entry_skipped",
        field=field,
        index=index,
        reason=reason,
        entry=repr(entry)[:120],
    )


def _sequence(raw: Any, field: str) -> list:
    """The top-level shape check: a field is a JSON array or it is unusable.

    ``None``/absent is legal and reads as empty (rule 1). A ``str``/``bytes`` is rejected even though
    it is iterable — iterating one yields characters, which is how a mistyped field turns into a
    plausible-looking population of one-character records instead of an error.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, bytearray)):
        raise SnapshotError(f"snapshot field {field!r} is a string, expected a list", field=field)
    if isinstance(raw, dict):
        raise SnapshotError(f"snapshot field {field!r} is a mapping, expected a list", field=field)
    if not isinstance(raw, (list, tuple)):
        raise SnapshotError(
            f"snapshot field {field!r} is {type(raw).__name__}, expected a list", field=field
        )
    return list(raw)


def key_tuple(raw: Any) -> Hashable:
    """A serialized state key (``[goal_id, entity]``) -> the tuple the runtime keys on.

    Refuses a bare scalar. See the module docstring: accepting one is exactly how a flattened pair
    manufactures a per-character key and re-opens a spent edge.
    """
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    raise ValueError(f"snapshot key must be a list, got {type(raw).__name__}: {raw!r}")


def as_bool(raw: Any) -> bool:
    """A serialized boolean. Refuses a truthy non-boolean — ``bool('e1')`` is ``True`` and that is
    precisely the value a flattened pair supplies."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):  # JSON 0/1 from a hand-written or older payload
        return bool(raw)
    raise ValueError(f"snapshot boolean must be a bool, got {type(raw).__name__}: {raw!r}")


def as_float(raw: Any) -> float:
    """A serialized timestamp. Refuses a bool (``isinstance(True, int)``) and any string — a stamp
    that arrives as text is a corrupt record, not a number to be coerced."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"snapshot timestamp must be a number, got {type(raw).__name__}: {raw!r}")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf are not stamps
        raise ValueError(f"snapshot timestamp is not finite: {raw!r}")
    return value


def parse_pairs(
    raw: Any,
    field: str,
    key_fn: Callable[[Any], Any],
    val_fn: Callable[[Any], Any],
    *,
    on_skip: "Callable[[Any], None] | None" = None,
) -> dict:
    """``[[key, value], ...]`` -> ``{key: value}``, skipping (and logging) every entry that does not
    parse.

    Raises :class:`SnapshotError` only when ``raw`` itself is not a list — the one failure with
    nothing to salvage. Every other failure is per-entry and costs exactly that entry.

    ``on_skip(key_or_None)`` fires once per skipped entry, carrying the PARSED KEY when the key
    itself was readable and only its value was not. That is what lets a caller enforce a coupling
    rule across fields — the gate uses it to guarantee it never restores one half of a two-belt
    suppression, which is a strictly worse outcome than restoring neither half.
    """
    out: dict = {}
    for index, entry in enumerate(_sequence(raw, field)):
        if isinstance(entry, (str, bytes, bytearray)) or not isinstance(entry, (list, tuple)):
            _skip(field, index, entry, "entry is not a [key, value] pair")
            if on_skip is not None:
                on_skip(None)
            continue
        if len(entry) != 2:
            _skip(field, index, entry, f"entry has {len(entry)} elements, expected 2")
            if on_skip is not None:
                on_skip(None)
            continue
        try:
            key = key_fn(entry[0])
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            _skip(field, index, entry, str(exc))
            if on_skip is not None:
                on_skip(None)
            continue
        try:
            value = val_fn(entry[1])
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            _skip(field, index, entry, str(exc))
            if on_skip is not None:
                on_skip(key)
            continue
        out[key] = value
    return out


def parse_keys(
    raw: Any,
    field: str,
    key_fn: Callable[[Any], Any] = key_tuple,
    *,
    on_skip: "Callable[[Any], None] | None" = None,
) -> list:
    """``[key, ...]`` -> a list of parsed keys, skipping (and logging) every entry that does not
    parse. The flat-list twin of :func:`parse_pairs` (the gate's ``no_evict`` exemption set)."""
    out: list = []
    for index, entry in enumerate(_sequence(raw, field)):
        try:
            out.append(key_fn(entry))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            _skip(field, index, entry, str(exc))
            if on_skip is not None:
                on_skip(None)
    return out


def parse_records(raw: Any, field: str, record_fn: Callable[[dict], Any]) -> list:
    """``[{...}, ...]`` -> a list of whatever ``record_fn`` builds, skipping bad records.

    The self-describing form. A record names its own fields, so a lost nesting level is
    UNREPRESENTABLE rather than merely detectable — there is no positional slot for it to slide
    into.
    """
    out: list = []
    for index, entry in enumerate(_sequence(raw, field)):
        if not isinstance(entry, dict):
            _skip(field, index, entry, "record is not an object")
            continue
        try:
            out.append(record_fn(entry))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            _skip(field, index, entry, str(exc))
    return out


def reject(field: str, reason: str, snap: Any) -> None:
    """Log a whole-payload rejection. The caller then degrades to "start blank"/"leave untouched" —
    both of which are recoverable, unlike a raise on a restart path."""
    log(
        get_logger(_LOGGER),
        "snapshot.rejected",
        field=field,
        reason=reason,
        payload_type=type(snap).__name__,
    )


__all__ = [
    "SnapshotError",
    "key_tuple",
    "as_bool",
    "as_float",
    "parse_pairs",
    "parse_keys",
    "parse_records",
    "reject",
]
