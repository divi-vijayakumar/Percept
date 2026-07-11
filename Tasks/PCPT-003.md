---
id: PCPT-003
title: Gemini strip-parse hardening + IRA fenced-JSON bug repro
status: ready
priority: P0
owner: unassigned
wave: A
pr: PR-3
human_test: pending
---

## Why
IRA reported a true detection silently converted to a refusal on 0.6.0. Fence-stripping fix 3cfbf6c IS in the released wheel, so first reproduce (stale package? truncation variant?), then harden.

## Scope
Repro against the published 0.6.1 wheel (scratch script, not CI). Add response_mime_type=application/json (+ response_schema where supported) to the single GenerateContentConfig in gemini.py; keep _parse_verdict fence-stripping as fallback.

## Acceptance criteria
- [ ] Repro outcome documented on this card (stale package vs truncation vs other)
- [ ] New test_verdict_parse.py: IRA's exact fenced payload; truncated JSON -> unparsed -> ASK; prose-wrapped JSON; strip extra keys default cleanly
- [ ] JSON mode enabled on both single-frame and strip paths
- [ ] PERCEPT_LIVE strip ablation spot-run: verdict quality unchanged vs baselines

## Human test
Confirm the repro writeup answers why IRA saw the bug; sign off on the reply wording to IRA about it.
