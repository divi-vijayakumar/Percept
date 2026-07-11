---
id: PCPT-008
title: Sliding M-of-N confirmation primitive
status: backlog
priority: P1
owner: unassigned
wave: C
pr: PR-7b
human_test: pending
---

## Why
Only N-consecutive confirmation exists; one dropped frame resets it (IRA's exact complaint - naive consecutive is brittle on real video).

## Scope
Declared confirm:{m,n} field consumed in escalation.py/behavior.py. gate.py untouched.

## Acceptance criteria
- [ ] Dropped-frame sequences that break N-consecutive but pass M-of-N (as a named test case)
- [ ] Window expiry + interaction with cooldown covered
- [ ] gate.py byte-for-byte unchanged

## Human test
Replay a choppy fixture and confirm a real event confirms despite dropped frames.
