---
id: PCPT-001
title: Contract bump 1.2.0 -> 1.3.0 (additive fields, one PR)
status: in_progress
priority: P0
owner: claude
wave: foundation
pr: PR-1
human_test: ready
updated: 2026-07-07
---

## Why
Every later work item needs a new additive contract field. Batching them into ONE minor bump avoids version churn and keeps `negotiate()` semantics simple.

## Scope
Add to `percept.contracts`: `Verdict.usage`, `Verdict.events` (timestamps), `Concern` perception `temporal:"clip"`/`fps`/`verify_only`, `Tier0Signal.energy`, `Concern.version`. Regenerate JSON schemas. Mirror `Tier0Signal.energy` in percept-edge `contract.js`.

## Acceptance criteria
- [x] CONTRACT_VERSION == "1.3.0"; negotiate("1.2.0","1.3.0") passes, major mismatch still raises
- [x] All new fields round-trip in test_contracts.py; a 1.2.0-shaped payload still parses
- [x] JSON schemas regenerated and matched by test_contract_parity.py
- [x] contract.test.js: Tier0Signal with/without energy round-trips; cross-language fixtures pass
- [x] make test test-js parity audit-benchmark all green

## Human test
Review the new contract fields in docs/schema output; confirm the changelog entry describes the bump as additive-only.

## Feedback
- **2026-07-07 12:23 claude:** Done on branch feat/pcpt-001-contract-1.3.0 (uncommitted). CONTRACT_VERSION 1.3.0 both sides (JS was 1.1.0, now 1.3.0). New: ClipEvent DTO, Verdict.usage/.events, Concern.fps/verify_only/version, Tier0Signal.energy (null != 0.0 silence; 1.2-era payloads parse unchanged). judge_usage treats empty usage dict as unannotated so loose-attr shims still read. Gates: py 810 OK, js 86 OK, parity 12/12 byte-for-byte, audit-benchmark + eval-benchmark PASSED. Changelog: Unreleased section added. Human test: review CHANGELOG.md Unreleased + schema/ClipEvent.schema.json.
