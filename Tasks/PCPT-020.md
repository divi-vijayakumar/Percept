---
id: PCPT-020
title: L-1 — LabelRecord rail (schema + LabelStore + disposition log) [KEYSTONE]
status: done
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
- [x] incident.label() writes locally with NO key present (zero-network e2e)
- [x] egress serializer cannot emit frame_ref (schema/type test)
- [x] LabelRecord golden-schema diff; JSON schema regenerated; round-trips
- [x] disposition log appends on FIRE/ASK
- [x] make test test-js parity all green

## Human test
Ratify the §4.1 schema + 4 source values (freeze). Confirm the 1.0.0-vs-bump call.

## Feedback
- **2026-07-12 claude:** DONE + independently verified (commit cc54d2a). Gates: py 953 (+31), parity 21/21, js 135, CONTRACT_VERSION unchanged 1.0.0, percept-edge untouched. Egress guarantee verified (frame_ref absent from to_egress, present in local). Folded additively into 1.0.0 (no bump).
- **HUMAN RATIFICATION NEEDED (design calls the agent flagged):**
  1. `Incident.label` collision resolved by KEEPING the str field + adding a METHOD `record_label(good, note, tags, source=)` (not `label()`), because a dataclass field+method can't share a name. Vision-spec §4.3 wrote `incident.label(good=...)` — reconcile the doc or accept `record_label`.
  2. `verdict_ref` = 16-hex content hash of satisfied|confidence|evidence. Spec didn't pin the algorithm — confirm paid metering wants this scheme.
  3. `deployment_id` has no existing concept; store/log take it as a ctor arg defaulting to "default" (+ $PERCEPT_HOME root). Confirm the deployment-id source.
  4. `tags` shipped as field(default_factory=list) (frozen-safe) vs spec's literal `= ()`; semantically identical.
  5. SCHEMA FREEZE: the §4.1 LabelRecord schema + 4 source values are now in-tree — ratify the freeze (three producers depend on it).
