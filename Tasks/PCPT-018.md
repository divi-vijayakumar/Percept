---
id: PCPT-018
title: R-2 — remove duplicate DetectorBank push block in worker.run
status: done
priority: P0
owner: claude
wave: r-track
pr: overnight
human_test: ready
updated: 2026-07-11
---

## Why
worker.py had a verbatim-duplicated boundary loop -> every edge boundary pushed+counted twice, corrupting stats.boundaries and the published inference count. Independent of everything else.

## Acceptance criteria
- [x] Duplicate `if banks and perceive_boundary is not None:` block deleted (one remains)
- [x] Full py suite green (917 passed, 1 pre-existing xfail)

## Feedback
- **2026-07-11 claude:** DONE, committed b53f6e3 on feat/overnight-oss-hardening. Verbatim dup at worker.py 508-516/522-530 removed. Boundary/detector/runtime tests (131) + full suite green.
