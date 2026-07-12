---
id: PCPT-016
title: Docs DX P0 — fix corrupted/stale .agents skill, de-dup, add live run() template
status: ready
priority: P0
owner: claude
wave: docs
pr: overnight
human_test: pending
updated: 2026-07-11
---

## Why
The DX gap survives in the skill scaffold: `.agents/skills/percept-agent/SKILL.md` is a STALE duplicate that teaches only `perceive_judged` and is CORRUPTED ("claude"->"Codex" bad find/replace, an invalid backend). The `.claude` skill's scaffold section still points at a `cv2` hand-roll template, and there is no `run()`-based live template.

## Scope
1. `.agents/skills/percept-agent/SKILL.md`: port the run()-first §2 + perceive_judged-as-§2b rewrite from the `.claude` copy; revert `Codex` -> `claude` (lines ~26,42,62-63,95).
2. De-duplicate `.claude` vs `.agents` skill (single source of truth; note the drift; don't re-corrupt).
3. Add `template_watch_agent_live.py` (webcam_agent.py-shaped: create + add_goal + `async for incident in percept.run(agent, source="webcam:0")`); fix the offline template docstring so "make it real" = point run() at source=, not a cv2 loop.
4. `.claude/skills/percept-agent/SKILL.md` scaffold section (~131-136): scaffold the run() path.

## Acceptance criteria
- [ ] No `Codex` backend string anywhere in the skills; backends are fake|gemini|claude
- [ ] `.agents` and `.claude` SKILL.md both lead with `percept.run(...)`; perceive_judged is the labeled BYO/offline path
- [ ] A live run()-based template exists and imports cleanly (py_compile)
- [ ] Snippets shown are runnable against the REAL signature (see PCPT-017 signature truth)

## Human test
Read the two SKILL.md files + the live template; confirm a fresh reader is taught `percept.run`, not a cv2 loop.
