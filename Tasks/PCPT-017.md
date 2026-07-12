---
id: PCPT-017
title: R-0 — run(goals=...) convenience wrapper (make the one-line docs honest)
status: ready
priority: P0
owner: claude
wave: r-track
pr: overnight
human_test: pending
updated: 2026-07-11
---

## Why
`percept.run` requires a pre-built `core` (async def run(core, *, source, ...) at worker.py:182); there is NO `goals=` kwarg. Multiple docs show `percept.run(source="webcam:0", goals=[...])` which raises TypeError. OSS-SPEC §2 specifies the goals= surface as the intended one-liner. Close the gap so the docs become true.

## Scope
Add a thin convenience (wrapper or overload) `percept.run(source=..., goals=[...], vision=..., priors=..., exemplar=..., detectors=...)` that builds a `Percept` core internally (Percept.create + add_goal per goal) and delegates to the existing `run(core, ...)`. Keep the existing `run(core, ...)` positional form working (back-compat). Verify the goal-string->Goal/Concern compilation path (agent.add_goal at agent.py:227; goals.py) — reuse it, don't reinvent.

## Acceptance criteria
- [ ] `percept.run(source=..., goals=["..."], vision="fake")` builds a core with those goals and runs (unit test, no network)
- [ ] `percept.run(core, source=...)` (existing form) still works unchanged
- [ ] The exact quickstart/choose-your-path one-liner imports + constructs without error
- [ ] make test green

## Human test
Confirm the API surface matches OSS-SPEC §2; decide if `watch=` is a better name than `goals=`.
