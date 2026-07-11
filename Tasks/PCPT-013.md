---
id: PCPT-013
title: percept-edge/worklet subpath (browser AudioWorklet glue)
status: backlog
priority: P2
owner: unassigned
wave: edge
pr: PR-9
human_test: pending
---

## Why
Browser capture glue belongs in a thin optional subpath, keeping the core environment-free (same philosophy as the Python vendor-free-core lint).

## Scope
AudioWorkletProcessor chunking mic samples into processAudio; chunking logic factored out and unit-tested; worklet file excluded from node test run; example under examples/.

## Acceptance criteria
- [ ] Chunking logic unit-tested in vitest (worklet shell excluded, documented why)
- [ ] Example runs in a browser against the mic
- [ ] Core src/ stays free of Web Audio imports

## Human test
Open the example page, grant mic, confirm energy signals stream to the console.
