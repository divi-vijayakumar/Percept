---
id: PCPT-020
title: L-1 — LabelRecord rail (schema + LabelStore + disposition log) [KEYSTONE]
status: ready
priority: P0
owner: claude
wave: keystone
pr: overnight
human_test: pending
updated: 2026-07-11
---

## Why
The flywheel keystone and the ONE seam OSS<->paid share (BUILD-BOTH rule 1: freeze schema week one). OSS-SPEC §4.

## Scope
- Schema: `@dataclass(frozen=True) LabelRecord` in contracts (fields per §4.1: incident_id, verdict_ref, goal_id, ts, good, note, tags, frame_ref, calibration_version, source: Literal["api","voice","ask_answer","watch_calibration"]). TREAT SCHEMA + 4 source values AS FROZEN — do not redesign.
- Media-free-head / local-tail: `frame_ref` is local-only; the egress serializer structurally CANNOT emit it (type-level).
- LabelStore: append-only JSONL at ~/.percept/labels/{deployment_id}.jsonl (mirror persistence.py protocol); append(record), iter(goal_id=None), stats().
- Surfacing: incident.label(good, note=None, tags=()) [resolve the Incident.label str field name collision]; RuntimeSession.label(incident_id, ...); reserved label_url field.
- Disposition log: append every gate FIRE/ASK to ~/.percept/dispositions/{deployment_id}.jsonl {incident_id, verdict_ref, disposition, ts} (gate.py).

## Contract-version note (DECISION for human)
Specs assume 1.3.0->1.4.0. Tree is now 1.0.0 (unreleased). Recommend: fold LabelRecord additively into 1.0.0 (no bump), consistent with the salience work. FLAG for human ratification; do not write "1.4.0".

## Acceptance criteria
- [ ] incident.label() writes locally with NO key present (zero-network e2e)
- [ ] egress serializer cannot emit frame_ref (schema/type test)
- [ ] LabelRecord golden-schema diff; JSON schema regenerated; round-trips
- [ ] disposition log appends on FIRE/ASK
- [ ] make test test-js parity all green

## Human test
Ratify the §4.1 schema + 4 source values (freeze). Confirm the 1.0.0-vs-bump call.
