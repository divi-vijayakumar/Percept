# B0 scope-freeze receipt

Status: **PASS**
Recorded: `2026-09-08T19:23:34Z`

This receipt binds execution to the founder-approved plan whose SHA-256 is
`2cf6790f7bba949063c3024bc8b7389156a3e42abda70c3bf465b1474aabc040`.

## Frozen launch claim

The only OSS 1.0 quality gate is fixed-camera Watch F1:

> Fire when the scene changes from no visible person to at least one visible person.

The relevant fixed-reference production-path runtime-reliability gates also block launch. Ego-motion
and BDD100K, raw-VLM/generic-Percept comparators, multi-goal scaling, and audio/fused A/V remain
separate non-blocking scorecards. Their results must never be pooled into F1 or silently promoted to
release gates.

## Frozen bars and measurement contract

- Event precision `>= 0.99` and event recall `>= 0.99`.
- Timely recall at durable receiver ACK `>= 0.99` within `2.0 s` of frozen source onset.
- Deliberate results, if enabled later, use a separate `120 s` deadline.
- Conservative all-in economy `<= $0.10/source-hour`.
- Zero duplicate logical fires and zero duplicate logical receiver deliveries after idempotency
  reconciliation.
- Exact episode, context-ledger, outbox, physical-attempt, logical-delivery, and ACK reconciliation.
- Maximum-cardinality one-to-one occurrence matching; unmatched fires remain false positives and
  missing, late, or unacknowledged events remain in the appropriate denominators.
- Only measured TP/FP/FN, precision, recall, timely recall, ASK, duplicates, misses, stage and
  onset-to-ACK p50/p95/p99/max latency, provider/all-in economy, and reliability reconciliation may
  be reported as results.

## Spend and execution controls

No paid provider call, paid benchmark, or 72-hour soak is authorized by this receipt. Each requires
separate explicit budget authorization and all plan admission gates. Execution stops at the earliest
failing joint gate.

## Base transition

The clean task branch had no substantive changes and moved from
`98bf9782c16ea529f61dc535fea966619b60e776` to the founder-preserved exact base
`11d87a84a1dd44a021c54a21409505a31317b6db` (tree
`765a620d6ca6883c4840724d8932daaf2b4c2d2c`). The primary worktree was not modified.

The machine-readable authority is [`b0-scope-freeze.json`](b0-scope-freeze.json).
