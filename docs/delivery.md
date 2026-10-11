# Delivery

## The decision contract

`Decision.to_dict()` is the wire form, described by `percept/decision.schema.json`
(`Decision.from_dict()` reads it back):

| Field | Type | Meaning |
|---|---|---|
| `contract_version` | string | `"1.0"` |
| `decision_id` | string | stable across redelivery: `{occurrence_id}/{k}` |
| `occurrence_id` | string | groups every decision about one occurrence: `{stream_id}/{condition_id}/{incarnation}-{n}` |
| `type` | string | `met`, `uncertain`, `cleared` or `expired` |
| `condition_id`, `stream_id`, `subject` | string | what, where, and about whom (`"scene"` unless poked with a subject) |
| `t_observed` | number | Unix time the evidence shows it |
| `t_decided` | number | Unix time of the decision |
| `confidence` | number | in `[0, 1]` |
| `reason` | string | human-readable; don't parse it |
| `cause` | string or null | machine-readable: `null` for a plain `met`; `deferred`, `timeout`, `unverified`, `restart`, `no`, `resolve`, `false_alarm`, `condition_changed`, `max_open` |
| `cost` | object | `attempts`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_s` charged since the previous decision of this occurrence |
| `trace_id` | string | the `occurrence_id`; finds the occurrence in a trace |
| `request_ids` | array | the judge requests behind it (the 16 most recent) |
| `evidence` | array | up to 8 frame references: `stream_id`, `stream_epoch`, `seq`, `t_wall`, `ephemeral` |
| `keyframe_jpeg_b64` | string or null | one JPEG of at most 40 KB, base64 |

Within contract 1.x, new top-level fields may be added and readers should ignore fields they
don't know; nested objects don't change shape. `from_dict` refuses another major version. The
rules that produce each field are in [decision-semantics.md](decision-semantics.md#14-decision-fields).

## At-least-once delivery

```python
async for delivery in monitor.watch(source, ack="manual", max_unacked=100):
    await agent.handle(delivery.decision)     # idempotent on decision_id
    await delivery.ack()
```

- **Commit before delivery.** A decision is written to the store, in the same atomic write as the
  decision state that produced it, before it is handed to you. It stays there until acknowledged.
- **Order.** Per stream, decisions arrive in the order they were made. There is no order across
  streams.
- **Redelivery.** After a restart, or when the stream is watched again, every unacknowledged
  decision is delivered again first, in order, with the **same `decision_id`**.
  `delivery.attempt` counts deliveries that may have happened and never repeats.
- **Idempotency.** You may see a decision more than once. Deduplicate on `decision_id`.

Acknowledgement:

- `ack="auto"` (the default) acknowledges a decision when you ask for the next one, that is after
  the loop body finished without raising. If the body raises, or you `break`, that decision is
  not acknowledged and is delivered again by the next `watch()` or after a restart.
- `ack="manual"` leaves it to you: `await delivery.ack()` once your agent has acted. At most
  `max_unacked` (100) decisions are held unacknowledged per stream; past that, delivery waits
  (capture and judging go on, and decisions keep being stored) and health `store_lag` is raised.
  Acknowledging twice is harmless.

A consumer that never reads loses nothing: decisions accumulate in the store, and memory stays
bounded.

## Stores

| Store | Durability | When |
|---|---|---|
| `SqliteStore("percept.db")` | survives a crash and power loss (WAL, full sync on every commit) | the default: `percept.db` in the working directory |
| `SqliteStore(path)` | as above | pass `store=percept.SqliteStore(path)` |
| `MemoryStore()` | survives a consumer error, not a crash; logs a warning once | opt in for demos, tests and ephemeral workloads |

- **One process per database.** The monitor locks `<path>.lock` when its first stream starts; a
  second process on the same file fails with `percept.delivery.StoreLocked` when it starts
  watching. The lock uses `fcntl`, which Windows doesn't have: there, keeping to one process per
  file is up to you.
- The database file is created readable only by the user that created it, since it holds keyframes.
- A slow disk delays delivery, never capture or judging; one writer thread makes every store call,
  and a failing call is retried (health `store_lag`).
- Any object with `load(key) -> bytes | None` and an atomic `commit(writes)` (a `None` value
  deletes) satisfies `percept.Store`.

## Health, stats and coverage

Nothing raises during operation. Configuration errors raise when you build the `Monitor`, call
`add()` or call `watch()`; everything else is reported:

- `Monitor.health()` is an async iterator of `percept.Health` events, kept in a bounded ring
  (repeats of an unread event raise its `count` instead of taking a slot). It ends at `aclose()`.

  | Kind | Meaning |
  |---|---|
  | `link_down` / `link_up` | a source's link went down or came back |
  | `judge_unavailable` | a judge request failed (outage, timeout, refusal, malformed reply) |
  | `degraded` | too many corrupt or blurry frames, or a lasting blur became the new normal |
  | `scene_reset` | most of the picture changed at once |
  | `budget_exhausted` | looks were due but the request or cost budget refused them |
  | `verify_deferred` | a verification is waiting for the budget, or its evidence left the pre-roll |
  | `unverified` | a candidate closed without verification (with `on_unverified="health"`) |
  | `capacity` | every decision slot on the stream is in use |
  | `store_lag` | the store is slow or failing, or delivery is paused at `max_unacked` |
  | `trace_dropped` | trace records were lost |

- `Monitor.stats()` returns exact counters at any time: per stream `frames`, `frames_capped`,
  `frames_vetoed`, `requests`, `attempts`, `evidence_unavailable`, fates, `decisions`,
  `delivered`, `redelivered`, `acked`, `unacked`, pending looks and verifications; the cost fields
  described in [judges.md](judges.md#reporting-cost); health counts; store and trace counters.
- `Monitor.coverage(stream_id, since_wall=0.0)` returns fates per condition (per minute) and the
  blind intervals overlapping that time. See [concepts.md](concepts.md#coverage-and-blind-intervals).

## Trace and replay

```python
monitor = percept.Monitor(judge, conditions, trace="percept-trace.jsonl")
...
result = percept.trace.replay("percept-trace.jsonl")
print(result.matches, result.recorded, result.inputs, result.differences)
```

- The trace is one JSON line per runtime event: every input given to the decision rules (with the
  time it was decided at and the condition versions), every decision, every judge request with its
  usage and latency, fates, dropped looks, health, blind intervals and pokes.
- `percept.trace.replay(path)` re-runs the recorded inputs through the decision rules, without the
  video or the judge, and compares the result with the recorded decisions. Because the rules are
  pure, a faithful trace reproduces every decision exactly; `differences` says where a replay
  first departs.
- `percept.trace.read(path)` yields the records, oldest first.
- Files rotate at 50 MB, keeping 4 older files; each new file starts with a checkpoint, so the
  newest file replays on its own.
- **Never written:** images, keyframes or other bytes, and raw source URIs. Strings are scrubbed of
  URL credentials and of values that look like keys, tokens or passwords; scrubbing is a pattern
  match, not a guarantee. What you write yourself is recorded in clear: condition text,
  `not_when`, poke reasons, feedback notes and the judge's reasons.
- Writing never blocks: when the writer's queue is full or the disk fails, records are dropped and
  counted (health `trace_dropped`). One process writes one trace path.

## Shutdown

`await monitor.aclose()` (or leaving `async with monitor:`) stops frames, cancels judge calls,
flushes the store (waiting up to 10 s), then closes the sources. Every `watch()` iterator ends.
