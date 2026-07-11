---
id: PCPT-002
title: Populate Verdict.usage in all judge backends
status: backlog
priority: P0
owner: unassigned
wave: A
pr: PR-2
human_test: pending
---

## Why
The funnel/OTel token+cost hooks are wired end-to-end but dark: backends discard usage_metadata, so the VLM cost-audit story (IRA ask #13) reports latency only.

## Scope
Read resp.usage_metadata (Gemini), resp.usage (Anthropic), OpenRouter usage block; attach to Verdict.usage. FakeVision emits deterministic usage so the funnel is testable offline.

## Acceptance criteria
- [ ] Gemini/Anthropic/OpenRouter populate Verdict.usage; absent usage_metadata -> {} (never a crash)
- [ ] test_funnel.py: record_judge records tokens/cost from a FakeVision run; OTel counters non-empty
- [ ] Backend unit tests assert extraction from stubbed response objects
- [ ] make bench reports token deltas; fake-backend numbers captured as a baseline entry

## Human test
Run a short live session and confirm per-concern token/cost counters appear in the funnel output.
