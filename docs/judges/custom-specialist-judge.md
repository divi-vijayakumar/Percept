---
title: Judging, and creating a custom judge or specialist
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Judging, and creating a custom judge or specialist

A **judge** is the model-as-evaluator step Percept escalates to when a goal cannot be resolved
cheaply on the local path. Concretely, a judge is a Vision adapter invoked via `Vision.judge(...)`.
This guide covers the judge lifecycle, the Protocol you implement, a custom specialist judge, and
bringing your own vision backend.

The one rule to keep in mind: a judge produces a `Verdict`; the same unmodified **Gate** — never the
judge — maps that verdict to FIRE / ASK / SILENT.

## The judge lifecycle is pre-arm and fail-closed

`WatchPlan.judge` names **which specialist judge you want** — an opaque reference like
`"audit_specialist_judge@0.1"`. It is a *request*, not an override. Before any watch arms, the
runtime runs a strictly ordered chain, entirely pre-arm:

1. **Authorize** (`resolve_authorized`, `percept.judge_registry`) — three hard checks, in order:
   1. **catalog membership** — the ref must be a key in the deployment-owned catalog. An empty
      catalog (the default) authorizes nothing.
   2. **installed factory** — the catalog entry's `factory_ref` must be a registered `vision`-seam
      factory (`registry.has("vision", …)`, a non-instantiating lookup).
   3. **context capability** — if the plan bound a static context but the spec declares
      `supports_static_context=False`, that is refused.
2. **Bind** (`JudgeContextStore`) — the static-context artifact (a rubric / policy / reference
   sheet) is read once from a local directory and proved **byte-for-byte** against a pinned SHA-256.
   Max 1 MiB, strict UTF-8, no network.
3. **Dispatch** — the verified text is injected into the judge call's `context` under the reserved
   key `judge_static_context`, and the adapter's `judge(...)` is called.

Any deviation — unknown ref, uninstalled factory, unsupported context, hash mismatch, oversize blob,
bad decode — **refuses the arm outright**. There is no silent fallback to the generalist vision
model. The deployment catalog is loaded from the env var `PERCEPT_JUDGE_CATALOG` (a JSON path);
unset means nothing is authorized.

## The Vision Protocol you implement

A judge and a vision backend share one seam: the `percept.contracts.Vision` Protocol. The methods,
verified against `percept/contracts/__init__.py`:

```python
async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                frames=None, audio=None, context=None) -> Verdict: ...
async def detect(self, frame, goals) -> list[Observation]: ...
async def detect_events(self, frames, goals) -> list: ...
```

`judge` is the firing path (returns a `Verdict(satisfied, confidence, evidence="")`). The optional
kwargs are escalations the brain assembles per concern: `frames` (an ordered temporal strip),
`audio` (first-class audio modality), `roi` (a crop `Box`), `exemplar`, `corrections` (user-asserted
facts), and `context` (caller-supplied structured facts, and where bound static context arrives). A
backend lacking a modality degrades gracefully; the `Verdict` shape is unchanged. `detect_events`
returns `[]` today (temporal events are a later phase).

### Specialist vs Conductor vs "specialist judge"

Two other model tiers exist behind their own Protocols — do not confuse them with a judge:

- **Specialist** — the reply tier: `def stream(self, prompt, *, frame=None, fast=False)` returning
  an `AsyncIterator[str]` (streams a spoken/text reply, optionally grounded on a frame).
- **Conductor** — authors goals: `async def run(self, event, utterance, context) -> Decision`.

A **"specialist judge"** (ADR-009 wording) is a Vision adapter — the thing this guide builds — and
is distinct from the Specialist reply Protocol despite the shared word.

## Recipe: a custom specialist judge

Implement the Vision Protocol, then register a factory on the `vision` seam.

```python
# my_pkg/audit_judge.py
from percept import registry
from percept.contracts import Verdict

class AuditSpecialistVision:                       # implements the Vision Protocol
    def __init__(self, settings=None): self._settings = settings
    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        rubric = (context or {}).get("judge_static_context")   # bound static context arrives here
        # ...call your model...
        return Verdict(satisfied=True, confidence=0.91, evidence="matches rubric §3")
    async def detect(self, frame, goals): return []
    async def detect_events(self, frames, goals): return []

registry.register("vision", "audit_specialist_factory", lambda s: AuditSpecialistVision(s))
```

The deployment catalog authorizes the opaque ref and maps it to your factory. `context_store_root`
is the directory the static-context binding resolves `ref` + `sha256` against:

```json
{
  "available_judges": {
    "audit_specialist_judge@0.1": {
      "factory_ref": "audit_specialist_factory",
      "provider_ref": "acme",
      "model_ref": "audit-vlm-v2",
      "supports_static_context": true
    }
  },
  "context_store_root": "/etc/percept/judge_context"
}
```

Point Percept at the catalog and author a plan that requests the judge:

```bash
export PERCEPT_JUDGE_CATALOG=/etc/percept/judges.json
```

```python
from percept.contracts import WatchPlan, ArtifactBinding

plan = WatchPlan(
    condition="an invoice total over the approval limit is on screen",
    judge="audit_specialist_judge@0.1",
    judge_context=ArtifactBinding(ref="rubric.txt", sha256="<64-hex>"),
)
```

`WatchPlan.condition` is required — it is the plain-language ask, carried verbatim. The
`ArtifactBinding.sha256` must be the real 64-hex digest of the artifact
(`sha256sum /etc/percept/judge_context/rubric.txt`); a half-specified binding (ref without hash) is
refused, and a wrong digest refuses the arm at bind time.

## Recipe: bring your own vision backend

A backend and a judge use the same Vision seam, so the same adapter can serve either. Register it by
name, then select it. Two ways to register:

```python
# in-process
from percept import registry
registry.register("vision", "myvlm", lambda settings: MyVision(settings))
```

```python
# from a third-party package: an entry point in the `percept.backends` group.
# The entry point value is a callable invoked with `register`:
def setup(register):
    register("vision", "myvlm", lambda s: MyVision(s))
```

```toml
[project.entry-points."percept.backends"]
myvlm = "my_pkg.backend:setup"
```

Select it three ways — by name, by instance, or via the provider door:

```python
from percept import Percept, resolve_vision

Percept.create(vision="myvlm")                 # by registered name
Percept.create(vision=MyVision(...))           # a constructed instance, used verbatim
adapter = resolve_vision("myvlm", model="...") # provider door (name + model, never a fused string)
```

### Verify your adapter with the conformance suite

The runnable conformance suite proves your adapter is **shaped right and fail-closed** — not that
the model is smart (accuracy is an eval question, never a conformance one). Run it from the CLI:

```bash
python -m percept.conformance mypkg.vision:MyVision   # exit 0 = conformant
```

or in your own tests:

```python
from percept.conformance import check_vision_adapter
report = check_vision_adapter(MyVision(), frame=...)
assert report.ok, report.format()
```

It checks the method set, the optional-kwarg escalations, the `Verdict` invariants the gate reads,
and that the adapter survives fail-closed injection into a live core (it is not silently replaced by
a scripted double, and a live core with no vision refuses to run at all).

See [../reference/backends.md](../reference/backends.md) for the shipped adapters and the
identity/configuration split, and [../reference/vision-adapter.md](../reference/vision-adapter.md)
for the full adapter contract.
