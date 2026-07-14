---
id: PCPT-027
title: Demo cleanup — delete examples demos, benchmark → eval/, doc links, skill-dir resync
status: done
priority: P1
owner: haiku
wave: boundary-cleanup-1
pr: ""
human_test: pending
updated: 2026-07-14
---

## Scope (plan Step 6)
- Delete `examples/{watch_agent,webcam_agent,fall_notification,loitering_clip,screenshare_assistant,realtime_watch_agent}.py`, `examples/web-demo/`, `examples/README.md`.
- `git mv examples/benchmark eval/benchmark_ci_gate`; update `Makefile` `MANIFEST ?=` + `docs/how-to/eval.md`; prove with `make eval-benchmark`.
- Update links in README.md, llms.txt, AGENTS.md, RFC-offline-scripted-mode.md, Tasks/PCPT-013.md, docs/how-to/{fall,runtime,eval,screenshare}.md, docs/tutorials/choose-your-path.md → `templates/realtime_voice_watch_agent/` + `percept.run`/`start`.
- Resync `.claude/skills/percept-agent/` and `.agents/skills/percept-agent/` (currently divergent) + add sync check.
- Add `templates/README.md` index.

## Acceptance
No dangling `examples/` references (`grep -rn "examples/" --include="*.md"` clean apart from historical Tasks cards); `make eval-benchmark` works; both skill dirs byte-identical.

Full spec: plan file Step 6.
