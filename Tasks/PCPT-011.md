---
id: PCPT-011
title: feedback(incident_id, label) intake API
status: backlog
priority: P1
owner: unassigned
wave: C
pr: PR-7e
human_test: pending
---

## Why
Incident.label exists but nothing ingests it; alerts carry real/false-alarm labels with no contract path back. Persist labels now; adaptation loop stays deferred (threshold_for stays inert).

## Scope
feedback() persists label onto Incident + appends to ActionLog; per-concern label counts exported through the funnel.

## Acceptance criteria
- [ ] test_feedback.py: idempotent re-label, unknown id handled
- [ ] Funnel exposes per-concern label counts
- [ ] No adaptive thresholding introduced (threshold_for untouched)

## Human test
Label a test incident false-alarm and see the count in funnel output.
