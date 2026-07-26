"""One uniform driver over `percept`'s senses, so the eval suite can talk to all of them at once.

The senses (`RhythmTracker`, `EntityGraph`, `SpanAssociator`, `EvidenceSequencer`,
`SerializedDecisionReducer`, `ContinuityIdentity`, `AppearanceIdentity`, `EvidenceExtractor`) share a
de facto shape — feed one observation, read the state that accumulated — under eight different verb
names, because each grew from its own concern and none of them needs to know the others exist. That
independence is correct for the runtime and wrong for a validation harness, which has to sweep every
sense with the same loop and compare their throughput on the same axis.

Unifying them inside `percept/` would mean breaking public API on a qualified 1.0.0 wheel to buy the
runtime nothing, so the seam lives out here instead. **Nothing under `percept/` may import this
module**: the dependency points one way, from the eval suite into the shipped package, and a sense
that ever needed the adapter to work would be a sense whose own API was incomplete.

The adapter is deliberately thin. It renames methods and it counts; it does not normalize argument
shapes, invent state a sense does not keep, or retry. Each sense's real signature stays the caller's
problem — `feed(*args, **kwargs)` forwards verbatim — because a harness that silently reshaped a
`Verdict` into an `Observation` would be measuring the adapter, not the sense.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from percept.entity import EntityGraph
from percept.extractors.ocr import ScriptedOCR
from percept.gate import Gate
from percept.identity import AppearanceIdentity, ContinuityIdentity
from percept.individuation import SpanAssociator
from percept.rhythm import RhythmTracker
from percept.sequencing import EvidenceSequencer, SerializedDecisionReducer


@dataclass(frozen=True)
class SenseSpec:
    """How one sense is driven: the constructor plus the verb names it happens to use.

    `read=None` marks a *pure* sense — one whose `feed` returns its whole answer and keeps no
    accumulating state to read back (the two identity matchers and the extractor). That is a real
    property of the sense, not a gap in this table, so `SenseProbe.read()` reports `None` for it
    rather than manufacturing a state view.

    `reset=None` means the sense exposes no reset verb and a probe reset re-runs `factory`. Every
    sense in `REGISTRY` is in that case today: `EvidenceSequencer.forget` / `EntityGraph.restore`
    are per-key/per-snapshot operations, not "become new again", and using them as resets would
    leave residue a validation run could not see.
    """

    name: str
    factory: Callable[..., Any]
    feed: str
    read: str | None
    reset: str | None = None
    read_args: tuple[Any, ...] = ()
    keyed: bool = False
    note: str = ""


class SenseProbe:
    """Drives one sense and records what it cost, without hiding what it did.

    Errors are counted **and re-raised**. A validation harness that saw a swallowed exception would
    score a sense as merely slow when it was actually broken, which is the one failure mode this
    suite exists to catch; the counter is for the report, the raise is for the truth.
    """

    def __init__(self, spec: SenseSpec, **factory_kwargs: Any) -> None:
        self._spec = spec
        self._factory_kwargs = dict(factory_kwargs)
        self._sense = spec.factory(**self._factory_kwargs)
        self._units_fed = 0
        self._reads = 0
        self._errors = 0
        self._wall_time_s = 0.0
        self._last_feed_args: tuple[Any, ...] = ()

    @property
    def spec(self) -> SenseSpec:
        return self._spec

    @property
    def sense(self) -> Any:
        """The live sense instance — the escape hatch for calls this adapter cannot express.

        `SerializedDecisionReducer.submit` is the standing example: it is a coroutine, so a
        synchronous `feed()` can never drive it and a harness that needs it must await it here.
        """
        return self._sense

    @property
    def readable(self) -> bool:
        """False for a pure sense, so a bare `None` from `read()` is never mistaken for state."""
        return self._spec.read is not None

    def feed(self, *args, **kwargs) -> Any:
        """Consume one observation, returning exactly what the underlying verb returned."""
        method = getattr(self._sense, self._spec.feed)
        started = time.perf_counter()
        try:
            result = method(*args, **kwargs)
        except BaseException:
            self._errors += 1
            raise
        finally:
            # Failed calls still cost wall time; excluding them would flatter a sense that raises.
            self._wall_time_s += time.perf_counter() - started
        self._units_fed += 1
        self._last_feed_args = args
        return result

    def read(self, *args, **kwargs) -> Any:
        """Read current state. Explicit arguments win; otherwise the spec's bound defaults apply.

        Several reads are per-key (`latest_admitted(key)`, `last_applied(key)`) or per-window
        (`windowed_count(now, window_s)`), so a no-argument `read()` needs something sensible to
        pass. A `keyed` sense replays the key from the most recent `feed()` — the same key the
        caller just wrote to is the only key a bare read could honestly mean.

        Before any feed (or after a `reset()`) there is no such key, and a keyed read is genuinely
        undefined: these senses answer `0` for a key they have never seen, so guessing one would
        report "no admissions yet" about a key the caller never named. That raises instead, and is
        not counted against the sense — the sense was never called.
        """
        self._reads += 1
        if self._spec.read is None:
            return None
        if not args and not kwargs:
            if self._spec.read_args:
                args = self._spec.read_args
            elif self._spec.keyed:
                if not self._last_feed_args:
                    raise TypeError(
                        f"{self._spec.name}.{self._spec.read}() is per-key; pass a key explicitly — "
                        f"read() can only replay a key after a feed()"
                    )
                args = self._last_feed_args[:1]
        method = getattr(self._sense, self._spec.read)
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        except BaseException:
            self._errors += 1
            raise
        finally:
            self._wall_time_s += time.perf_counter() - started

    def reset(self) -> None:
        """Begin a fresh trial: the sense and its counters both return to zero.

        Counters are cleared with the sense because a per-clip `stats` read that carried the
        previous clip's totals would silently misattribute cost across trials.
        """
        if self._spec.reset is None:
            self._sense = self._spec.factory(**self._factory_kwargs)
        else:
            getattr(self._sense, self._spec.reset)()
        self._units_fed = 0
        self._reads = 0
        self._errors = 0
        self._wall_time_s = 0.0
        self._last_feed_args = ()

    @property
    def stats(self) -> dict:
        return {
            "name": self._spec.name,
            "units_fed": self._units_fed,
            "reads": self._reads,
            "wall_time_s": self._wall_time_s,
            "errors": self._errors,
        }


def _reducer(gate: Gate | None = None, **kwargs: Any) -> SerializedDecisionReducer:
    """Supply the `Gate` the reducer requires, since it is the only sense with a hard dependency.

    A default `Gate()` is a usable one for probing — the reducer's own behaviour under test is
    sequence ordering, not the threshold the gate applies — but a caller measuring interaction with
    a specific gate policy passes their own.
    """
    return SerializedDecisionReducer(gate if gate is not None else Gate(), **kwargs)


REGISTRY: dict[str, SenseSpec] = {
    "rhythm": SenseSpec(
        name="rhythm",
        factory=RhythmTracker,
        feed="observe",
        read="windowed_count",
        # windowed_count(now, window_s) has no all-time form, so an unbounded window is the bound
        # default: every confirmed rep on the selected channel, independent of the clock.
        read_args=(0.0, float("inf")),
        note="feed is observe(t, {channel: value}); read is windowed_count(now, window_s).",
    ),
    "entity": SenseSpec(
        name="entity",
        factory=EntityGraph,
        feed="ingest",
        read="active",
        note="feed is ingest(observations: list[Observation], frame=None, source='camera', *, t=None) "
        "— a LIST per frame, not one Observation. snapshot() is the serializable peer of active().",
    ),
    "span": SenseSpec(
        name="span",
        factory=SpanAssociator,
        feed="add",
        read="count",
        note="feed is add(EventCandidate) -> (event, is_new); read is count(kind=None).",
    ),
    "sequencer": SenseSpec(
        name="sequencer",
        factory=EvidenceSequencer,
        feed="admit",
        read="latest_admitted",
        keyed=True,
        note="feed is admit(key, *, episode_id=None) -> int; read is latest_admitted(key).",
    ),
    "reducer": SenseSpec(
        name="reducer",
        factory=_reducer,
        feed="admit",
        read="last_applied",
        keyed=True,
        note="feed is the SYNCHRONOUS admit(key) reservation; submit(...) is a coroutine and must be "
        "awaited via .sense. Requires a Gate, defaulted by the factory.",
    ),
    "continuity_id": SenseSpec(
        name="continuity_id",
        factory=ContinuityIdentity,
        feed="match",
        read=None,
        note="pure: match(obs, known, *, signature=None) -> entity id | None; state lives in `known`.",
    ),
    "appearance_id": SenseSpec(
        name="appearance_id",
        factory=AppearanceIdentity,
        feed="match",
        read=None,
        note="pure: same signature as continuity_id; the appearance fallback needs signature= AND "
        "known entities carrying .appearance, else it degrades to continuity.",
    ),
    "extractor": SenseSpec(
        name="extractor",
        factory=ScriptedOCR,
        feed="extract",
        read=None,
        # percept.extraction.EvidenceExtractor is a typing.Protocol and cannot be instantiated, so
        # the registry names the shipped deterministic conformer instead of the abstract seam.
        note="pure: extract(condition, frame, roi=None) -> Enrichment. EvidenceExtractor itself is a "
        "Protocol; ScriptedOCR is the offline conformer, constructed with script={frame_key: facts}.",
    ),
}


def probe(name: str, **factory_kwargs: Any) -> SenseProbe:
    """Construct a probe for a registered sense; unknown names fail loudly with the valid set."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"unknown sense {name!r}; registered senses: {', '.join(sorted(REGISTRY))}")
    return SenseProbe(spec, **factory_kwargs)


__all__ = ["SenseSpec", "SenseProbe", "REGISTRY", "probe"]
