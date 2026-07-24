# Trace bundles & replay

Two things people mean by "replay". Percept ships one of them, and says so.

| | what it is | deterministic? | where it lives |
|---|---|---|---|
| **Cognition replay** | recorded normalized verdicts + timestamps fed back through the Gate and the incident logic | **yes** — same trace in, byte-identical decisions out | `percept.replay` (this page) |
| **Media reevaluation** | recorded media sent to live model calls again | **no** — sampling, model drift, provider changes | `percept eval` |

A trace that holds only a media URI and needs a mutable cloud model does **not** "replay the run" —
the normative definition of deterministic cognition replay lives in the `percept.replay` module
docstring (the design rationale is ADR-004).

## Record a run so it can be replayed

```python
from percept import Percept, RunTraceManifest, TracePolicy, TraceRecorder, gate_config_of

core = Percept.create(vision=..., discover_plugins=False)
manifest = RunTraceManifest.create(
    run_id="run-123", goals=goals,
    policy=TracePolicy.evaluation(),
    gate_config=gate_config_of(core.gate),   # the gate's knobs travel with the trace
)
rec = TraceRecorder(manifest)
core.on_trace = rec.tracer()
```

`gate_config` is what makes a replay reproduce *that run's* gate rather than today's defaults. Omit
it and the report says `gate_config_recorded=False` instead of guessing.

## Export one file for a bug report

```python
from percept import export_bug_report

path = export_bug_report(rec, "run-123")        # -> run-123.ptrace.zip
```

Media inclusion is **opt-in and off by default**. Any `evidence=[...]` offered to a default export
is dropped, and the drop is counted into the bundle manifest (`media_omitted`,
`media_omitted_reasons`) so the omission is visible. To pack pixels, someone has to type it:

```python
export_bug_report(rec, "run-123", include_media=True, evidence=[EvidenceItem(...)],
                  retention_days=7, redact_keys=("window_title",))
```

### Layout

```
run-123.ptrace/
|-- manifest.json      run manifest + mode, per-file sha256, retention/expiry, replay contract
|-- events.jsonl       one TraceRecord per line, ingest order
|-- summary.json       the derived funnel (percept.tracing.TraceSummary)
`-- evidence/          portable mode ONLY: frame-001.jpg, capsule-007.json, ...
```

`mode` is `local` (media stays outside, referenced by uri + hash) or `portable` (media bytes
inside). Retention defaults to 30 days; `retention_days=None` (keep forever) has to be typed.

## Verify and replay

```python
from percept import TraceBundle

problems = TraceBundle.verify_path("run-123.ptrace.zip")   # [] = intact
bundle = TraceBundle.read("run-123.ptrace.zip")
report = bundle.replay()

report.digest            # content hash of the whole decision sequence — the determinism claim
report.gate_outcomes     # {"fire": 2, "ask": 1, "silent": 2}
report.incidents         # a delivered FIRE is an incident; an ASK is not
report.divergences       # replayed decision != what the run recorded, reported not hidden
report.unreplayable      # gate records with no usable verdict — counted, never guessed
```

Authored behaviour a trace cannot carry (hard-quadrant `FIRE`→`ASK` escalation, `fire_when`,
`suppress_false`) is declared explicitly, otherwise treated as absent:

```python
from percept import ReplayGoalConfig
bundle.replay(goal_config={"spill": ReplayGoalConfig(escalates=True)})
```
