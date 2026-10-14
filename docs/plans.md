# Watch plans

> **Experimental.** The plan schema (`percept.watch/v1alpha1`) may change in a minor release. The
> runtime it compiles to (`Monitor`, `Condition`) is the stable 1.0 API.

A watch plan describes a whole watch in one plain-data document: its sources, its request budget,
the judges it may ask, and for each concern what to look at, how to judge it and when to decide.
`percept.plans.compile` checks the plan and returns the arguments of a `Monitor`. Load the plan
from JSON or YAML yourself: `compile` takes a `dict`.

```python
import percept

compiled = percept.plans.compile(plan, {"fast": percept.judges.gemini("gemini-2.5-flash-lite")})
async with percept.Monitor(**compiled.monitor_kwargs, store=percept.SqliteStore("percept.db")) as monitor:
    source = percept.open_source(url, stream_id="dock-cam", max_side=compiled.sources[0]["max_side"])
    async for delivery in monitor.watch(source, ack="manual"):
        ...
```

The second argument maps the plan's route names to judges. Nothing in `monitor_kwargs` sets a
source's resolution, so open each source with its `max_side` yourself, as above. A runnable,
offline version is [examples/watch_plan.py](../examples/watch_plan.py).

## The document

```json
{
  "schema_version": "percept.watch/v1alpha1",
  "plan_id": "dock-watch",
  "revision": 1,
  "sources": [{"id": "dock-cam", "kind": "video", "max_side": 640}],
  "budget": {"max_requests_per_min": 6},
  "concerns": [
    {"id": "person_enters", "source": "dock-cam",
     "description": "A person walks into view through the doorway.",
     "attention": {"any": [{"on_change": {"threshold": 0.1}}, {"every": 30}]},
     "judgment": {"route": "fast", "zones": [[[0.4, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.0]]]},
     "decision": {"kind": "event", "rearm_s": 30}}
  ],
  "metadata": {"owner": "free-form; not part of the plan hash"}
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `percept.watch/v1alpha1`; any other value is refused |
| `plan_id`, `revision` | yes | your identifiers; `revision` is an integer of 1 or more |
| `sources[]` | yes | `id`, `kind` (`video` only) and optional `max_side` (default 640): the longest side, in pixels, of the frames the judge sees |
| `budget` | no | `max_requests_per_min` (attempts per stream, default 6), `max_cost_per_hour`, `max_concurrent_requests` |
| `judges.order` | when routes map to more than one judge | the route names, cheapest first; they become a cascade, and verification uses the last |
| `concerns[]` | yes | one per condition (below) |
| `metadata` | no | free-form; not hashed |

Each concern becomes one `Condition`:

- **`attention`** (optional) decides when the concern is worth a look. `{"any": [...]}` lists
  `on_change` (a scene change inside the concern's zones; `threshold`, `exit_threshold`,
  `baseline_s`), `every` (a heartbeat in seconds) and `external_event` (a name; the concern is
  judged when your code calls `monitor.poke(stream_id, condition_id)`). Several entries combine
  into one combined trigger (a tuple in `Condition.trigger`): the concern is due when any fires. Without `attention`, an event is looked at on a scene change
  and a state on a scene change plus every 300 s.
- **`judgment`** (`route` required) says how to ask: `zones` and `exclude_zones` (normalised
  polygons; the judge sees the zone crop), `not_when`, `verification`, `min_confidence`,
  `fast_path_confidence`, and for events `evidence.pre_seconds` and `evidence.frames` (the clip the
  judge sees: `pre_seconds` above 0 and at most the 8 s pre-roll, default 4; `frames` from 1 to 16,
  default 4). A state with either is refused: it is judged on the newest frame.
- **`decision`** (`kind`: `event` or `state`, required) says when to tell your agent: `confirm`,
  `confirm_window_s`, `rearm_s`, `recheck_every_s`, `clear_after_no`, `max_open_s`,
  `cooldown_s`, `emit_cleared`, `on_unverified`, `candidate_deadline_s`. These map one-to-one onto
  the `Condition` fields in [conditions.md](conditions.md).

A malformed plan raises `percept.plans.PlanError`, whose message starts with the dotted path of the
field (`concerns[0].judgment.zones: ...`).

## What a compiled plan tells you

- **`explain()`**: every effective setting of every concern, and whether the plan set it, the SDK
  defaulted it, or the runtime fixes it.
- **`feasibility()`**: a static check of each source's periodic demand (heartbeats) against its
  request budget, with a verdict and notes. Change-driven looks depend on the scene and are not
  predicted.
- **`plan_hash`**: covers every effective setting, the runtime's fixed values and the judges'
  names, whatever the order of keys, concerns or sources. Two plans with the same hash watch the
  same way. `metadata` is not hashed.
- **`provenance()`**: the SDK version, the plan's identity and hash, the judges and the prompt
  hash a run used, without secrets. Nothing records it for you:
  `compiled.trace_record(t_wall, stream_id=None)` returns it as a trace record you may write
  yourself, to a file of your own, with `percept.trace.TraceWriter.write`. Replay ignores such records.
- **`events`**: each external event name mapped to the condition ids to poke.

## Types

- **`percept.plans.Compiled`**: `plan_hash`, `monitor_kwargs`, `conditions`, `sources`, `events`,
  `plan_id`, `revision`, and the methods `explain()`, `feasibility()`, `provenance()` and
  `trace_record(t_wall, stream_id=None)`.
- **`percept.plans.SCHEMA_VERSION`**: `"percept.watch/v1alpha1"`, the only accepted
  `schema_version`.
- **`percept.plans.PlanError`**: a malformed plan; the message starts with the field's dotted path.

## Choosing settings

A plan states a configuration; it does not find one. To compare configurations on your own footage
before relying on one, use the evaluation toolkit: [evaluation.md](evaluation.md).
