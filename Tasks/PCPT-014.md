---
id: PCPT-014
title: Reply docs: IRA feedback response + PERCIA-031 correction
status: ready
priority: P1
owner: unassigned
wave: A
pr: -
human_test: pending
---

## Why
Half the IRA letter is answered by pointing at shipped 0.6.x APIs; their mirror is drifting against features that exist. PERCIA-031's 'nothing exists' premise is wrong (EnergyVad.pushSamples already consumes raw Float32 with hangover + turn-finalization).

## Scope
Ask-by-ask reply mapping each item to import paths (asks 1,2,4,5-schedule,6-OR,9,11,13 = shipped); upgrade guidance; PERCIA-031 note so web-core migrates to EnergyVad now, processAudio at 0.7. Per docs rule: only what ships today - no roadmap/pricing/enterprise talk.

## Acceptance criteria
- [ ] Reply covers all 16 IRA asks with status + import path or plan reference
- [ ] Bug-repro finding from PCPT-003 incorporated
- [ ] PERCIA-031 note corrects the premise with code pointers
- [ ] No roadmap/enterprise/pricing content

## Human test
Read both docs before they go out - external-facing.
