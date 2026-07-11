---
id: PCPT-010
title: Privacy: implement promised snippet redaction + Concern.privacy
status: backlog
priority: P1
owner: unassigned
wave: C
pr: PR-7d
human_test: pending
---

## Why
Docstrings promise 'privacy redaction nulls media' but NO code performs it - promised-but-absent is worse than absent (legal exposure for consumers like IRA).

## Scope
redact_snippet() nulls media/frames keeping hashes; Concern.privacy level triggers it in make_incident; document no-FR guidance for enrichers.

## Acceptance criteria
- [ ] Redacted incident serializes with zero media bytes anywhere (grep-the-JSON assertion)
- [ ] Privacy level plumbed from Concern to incident creation
- [ ] Docstrings and code now agree

## Human test
Create an incident under a privacy-marked concern and verify stored evidence is redacted end to end.
