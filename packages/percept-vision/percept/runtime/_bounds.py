"""One bounded, recency-ordered map — the ADR-001 §9 idiom, extracted so its callers agree.

ADR-001 §9 asks two things of every structure a 24/7 process keeps: it is **bounded**, and **a drop
forced by that bound is reported with a reason, never a silent counter bump**. The tree had grown
three hand-rolled answers to the same question, and only one of them satisfied both halves:

* :class:`percept.runtime.sinks.IdempotencyLedger` — correct AND traced. It refreshes a key by
  popping and re-inserting it (which moves it to the BACK of a dict's insertion order) and evicts
  from the FRONT via ``next(iter(...))``, calling ``on_evict(key, reason)`` on every drop. This
  module is that implementation, lifted verbatim;
* ``BehaviorExecutor._note_merge_anchor`` — correct only for a clock that MOVES, and silent. It
  re-sorted the whole map by timestamp and kept the top *N*. Python's sort is stable and its reverse
  sort does not reorder equal elements, so under the frozen clock the cap exists for — every stamp
  tying at one value — "keep the newest" kept the **earliest-inserted** *N* and threw away the stamp
  that had just been taken one line earlier. It then truncated with no log, no hook and no reason.

Recency, not the timestamp, is the honest ordering
--------------------------------------------------
The stamp a caller supplies is data; the ORDER writes arrived in is a fact this map observes for
itself. Keying eviction on insertion order therefore degrades gracefully where sorting by value does
not: for a clock that advances the two agree exactly, and for one that is frozen, replayed, or
walking backwards the insertion order is still a total order while the stamps are a heap of ties.

This is a ``dict`` subclass on purpose. Both callers expose their map to code (and to tests) that
reads it as a plain mapping — ``.get``, ``in``, ``len``, ``.items()``, ``.pop`` and equality against
a dict literal — and dicts have preserved insertion order since 3.7, so the recency order this needs
is the ordering the base type already maintains. Only :meth:`note`, :meth:`enforce` and
:meth:`expire` are new; plain ``m[k] = v`` writes without refreshing recency, exactly as a dict does.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

from percept.observability import get_logger, log

_LOGGER = "percept.runtime._bounds"


class BoundedRecencyMap(dict):
    """``key -> value`` with a hard count bound, evicting the LEAST-RECENTLY-NOTED key first.

    ``on_evict(key, reason)`` fires on every drop. When it is not wired the map logs
    ``bounded_map.evicted`` itself — a bound that drops silently is the failure ADR-001 §9 names, so
    there is no configuration in which a drop goes unreported.

    ``bound_reason`` is the ``reason`` the COUNT bound reports (``"max_keys"``; the idempotency
    ledger calls the same bound ``"max_records"``). Age/window sweeps pass their own reason to
    :meth:`expire`.

    Iteration is oldest-noted first — inherited straight from ``dict``, and every eviction below
    depends on it, so a subclass that reorders iteration breaks the bound.
    """

    def __init__(
        self,
        max_keys: int,
        *,
        on_evict: "Callable[[Any, str], None] | None" = None,
        bound_reason: str = "max_keys",
    ) -> None:
        super().__init__()
        self.max_keys = int(max_keys)
        self.on_evict = on_evict
        self.bound_reason = bound_reason

    # -- writes -----------------------------------------------------------------------------------
    def note(self, key: Hashable, value: Any) -> None:
        """Write ``key`` as the NEWEST entry, then enforce the count bound.

        Pop-then-reinsert is what moves an already-present key to the back; a bare ``self[key] =
        value`` would leave it where it was and let a key that is being refreshed every tick age out
        underneath a stale neighbour that has not been touched since it was created.
        """
        dict.pop(self, key, None)
        dict.__setitem__(self, key, value)
        self.enforce()

    def enforce(self) -> None:
        """Drop from the FRONT (least-recently-noted) until the count bound holds. Every drop traced."""
        while len(self) > self.max_keys:
            oldest = next(iter(self))
            dict.pop(self, oldest)
            self.evicted(oldest, self.bound_reason)

    def expire(self, is_stale: "Callable[[Any, Any], bool]", reason: str) -> None:
        """Drop from the FRONT while ``is_stale(key, value)``, stopping at the first survivor.

        The age sweep. It is front-anchored and short-circuiting because the caller that uses it
        (a TTL over a clock that advances) writes in stamp order, so the first entry that is young
        enough proves every entry behind it is too. A sweep that must inspect EVERY entry — one
        whose staleness is not monotone in insertion order — is not this method; see
        ``BehaviorExecutor._note_merge_anchor``, which retires window-expired stamps with a plain
        ``pop`` loop precisely because a future-dated stamp may sit in front of a stale one.
        """
        while self:
            key, value = next(iter(self.items()))
            if not is_stale(key, value):
                return
            dict.pop(self, key)
            self.evicted(key, reason)

    # -- the trace --------------------------------------------------------------------------------
    def evicted(self, key: Hashable, reason: str) -> None:
        """Report one drop (ADR-001 §9). The hook when wired, a structured log line when not."""
        if self.on_evict is not None:
            self.on_evict(key, reason)
        else:
            log(get_logger(_LOGGER), "bounded_map.evicted", reason=reason, key=repr(key)[:120])


__all__ = ["BoundedRecencyMap"]
