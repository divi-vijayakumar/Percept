### Changed — `watch()` accepts plain-string goals (additive, backward-compatible)

`watch(stream, goals, ...)` now accepts `goals: str | Goal | Iterable[str | Goal]` — a bare string, a
single `Goal`, or a mix — where it previously took `list[Goal]`. Existing `list[Goal]` callers are
unchanged; this only *widens* the accepted shape, so no call that worked before breaks.

`WATCH_VERSION` stays `"1.0.0"`: the function's streaming semantics and the `WatchEvent` it yields are
unchanged — only the ergonomics of specifying goals improved. The public seam and its version constant
remain the frozen contract.
