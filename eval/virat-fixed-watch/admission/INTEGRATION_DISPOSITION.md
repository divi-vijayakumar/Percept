# Integration and merge disposition

This log is append-only for benchmark execution commits. The task branch is
`codex/percept-benchmarking-b0-b1`; it must descend from
`11d87a84a1dd44a021c54a21409505a31317b6db` and must not be pushed or merged by this task.

| UTC time | Milestone | Commit | Tests/evidence | Disposition |
|---|---|---|---|---|
| 2026-09-08T19:23:34Z | B0 | `456c69bd7a6cc2377925af60dbc7b8139532d725` | JSON parse, plan hash, clean ancestry | Atomic task-branch commit; eligible for later review only. |
| 2026-09-08T19:23:34Z | B1 | this atomic commit (resolve from history) | Receipt unit tests, JSON/JSONL parse, arithmetic, registry/partition invariants, ancestry, diff check | FAIL; retain evidence and stop before B2. Do not merge as a claim of benchmark qualification. |
| 2026-09-08T19:23:34Z | receipt hygiene | post-B1 commit (resolve from history) | Full branch `git diff --check` | Formatting-only cleanup; no gate or evidence change. |
| 2026-09-08T19:36:07Z | B1 rights correction | this atomic commit (resolve from history) | Founder confirmation, receipt unit tests, JSON parse, diff check | Signed VIRAT agreement admitted; B1 remains FAIL on annotation semantics and holdout capacity. |
| 2026-09-08 | B0 amendment 1 | this atomic commit (resolve from history) | Scope-amendment receipt tests, JSON parse, diff check | Founder-approved F2-Z1 replaces F1 as launch Watch; F1 becomes non-blocking stress evidence. |
| 2026-09-09T08:34:15Z | B1 F2-Z1 readmission | this atomic commit (resolve from history) | Official package/README, 138-file size/hash/clock manifest, aggregate rate survey, receipt tests | FAIL at 276/299 clean qualification seed events; stop before B2. Group 0502 exposed and removed immediately. |
| 2026-09-09T18:30:20Z | B1 PILOT-276 amendment | this atomic commit (resolve from history) | Founder decision, amendment hash, statistical-bound and receipt tests | ENGINEERING ADMIT for zero-paid B2–B4; publication qualification remains FAIL; qualification labels remain sealed. |
