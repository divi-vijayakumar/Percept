---
id: PCPT-015
title: gate.py byte-for-byte hash guard in CI
status: ready
priority: P2
owner: unassigned
wave: foundation
pr: -
human_test: pending
---

## Why
The changelog promises gate.py is byte-for-byte stable across releases; nothing enforces it. Make regressions loud.

## Scope
test_gate_frozen.py asserting the file's SHA-256; comment explains how to intentionally rotate the hash with a changelog entry.

## Acceptance criteria
- [ ] Test fails on any gate.py edit with an actionable message
- [ ] Runs in default make test (L1, fast)

## Human test
None needed beyond CI green - self-evident.
