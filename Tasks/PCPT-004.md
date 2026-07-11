---
id: PCPT-004
title: Gemini context caching for judge system prompt
status: backlog
priority: P1
owner: unassigned
wave: A
pr: PR-4
human_test: pending
---

## Why
Claude backend has cache_control prompt caching; Gemini (IRA's judge) resends _JUDGE_SYSTEM every call at 24/7 volume.

## Scope
cached_content for _JUDGE_SYSTEM + per-concern prefix, env-gated, off by default. Mirror the Anthropic pattern.

## Acceptance criteria
- [ ] Config-construction unit test with stubbed client (cache requested when enabled, absent when not)
- [ ] _resilience retry/breaker tests untouched and green
- [ ] L2 measurement with Verdict.usage shows reduced input tokens (reported in PR, not a CI gate)

## Human test
Check the measured token reduction on e2e-bench and confirm it justifies keeping the flag.
