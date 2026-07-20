# The Vision adapter interface (public extension point)

Percept's cognition core — fovea, judge scheduling, gate, entity graph, escalation — **never imports a
vendor SDK**. Every model sits behind one frozen seam: `percept.contracts.Vision`. That makes the
Vision seam the supported way to bring your own model (a hosted VLM, a local weights server, a
fine-tune, a rules engine) without forking Percept.

This page is the contract. The *executable* version of this page is
[`percept.conformance`](#the-conformance-suite) — if the two ever disagree, the suite is right.

| artefact | version | what it versions |
|---|---|---|
| `percept.contracts.CONTRACT_VERSION` | `1.2.0` | the DTO wire shapes (`Verdict`, `Observation`, `Box`, …) |
| `percept.watch.WATCH_VERSION` | `1.0.0` | the `watch()` entrypoint signature + streaming semantics |
| `percept.conformance.CONFORMANCE_VERSION` | `1.0.0` | the set of conformance checks and their severities |

---

## The seam

```python
from typing import Any, Protocol
from percept.contracts import Box, Goal, Observation, Verdict


class Vision(Protocol):
    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: "AudioFrame | bytes | tuple[str, bytes] | None" = None,
        context: dict | None = None,
    ) -> Verdict: ...

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]: ...

    async def detect_events(self, frames: list, goals: list[Goal]) -> list: ...
```

`Vision` is `runtime_checkable`, so `isinstance(my_adapter, Vision)` is a real (if shallow) check.

### `judge` — the firing contract

`judge` answers **one plain-language `condition` about the current frame**. It does *not* compile the
condition into `attr_key=value` and exact-match — one firing path, one gate. Everything after the
`*` is an **optional escalation** the brain assembles per the concern's perception spec:

| kwarg | what it is | if you can't use it |
|---|---|---|
| `roi` | normalized fovea crop (`Box` in 0..1) | **required** — crop the frame yourself; the fovea is most of the win |
| `corrections` | user-asserted facts that beat your observation | **required** — put them in the prompt |
| `frames` | earlier context frames, oldest→newest, excluding `frame` | accept and ignore (judge the single frame) |
| `audio` | `AudioFrame`, `bytes`, or `(mime, bytes)` | accept and ignore |
| `context` | structured facts from an `Enricher` (e.g. `{"plate": "7XYZ123"}`) — authoritative | accept and ignore |
| `exemplar` | a reference image for "this specific thing" goals | accept and ignore |

Every escalation **must be accepted and must default to `None`**, so a bare single-frame call is the
base case. Raising `TypeError` on an unexpected kwarg takes the whole watch down; *degrading
gracefully* is the contract. A backend without ears is conformant. A backend that crashes when handed
audio is not.

### `Verdict` invariants (what the gate reads)

```python
Verdict(satisfied=True, confidence=0.82, evidence="steam above the pan rim",
        usage={"input_tokens": 1204, "output_tokens": 31, "cost_usd": 0.0009})
```

- `satisfied` — a real `bool`. Not `1`, not `"yes"`; the gate branches on it.
- `confidence` — a float in **`[0.0, 1.0]`**. The gate compares it against a probability threshold, so
  an unnormalized logit silently breaks firing.
- `evidence` — a short human-readable string; it lands on the incident card.
- `usage` — optional but strongly recommended: `input_tokens` / `output_tokens` / `cost_usd`. This is
  what `percept.observability.judge_usage` costs the stream from, and what the $/stream-hour SLO reads.
- `phase`, `action_present`, `running_answer`, `events` — optional structured fields for clip/temporal
  and process-QA judging. Leaving them at their defaults reproduces single-frame behaviour exactly.

### `detect` — the entity substrate

Returns `Observation(label, box, confidence, attrs)`. **`Box` is normalized 0..1 image coordinates**,
not pixels — the entity graph matches tracks by IoU across frames and assumes normalization. Pixel
boxes are the single most common adapter bug, and the conformance suite fails on them.

### `detect_events` — optional temporal detection

Return `[]` if your backend does no temporal detection. That is a capability you lack, not an error.

---

## Fail-closed injection

Two guarantees make "my adapter is the one running" *checkable*, not hopeful:

1. **Your instance is the instance.** `Percept.create(vision=my_adapter)` keeps your object on
   `core.vision`. Percept never wraps or substitutes it. Seam resolution is `explicit kwarg >
   settings/env > mode default`.
2. **A live core with no vision refuses to start.** `Percept.create(mode="live")` with no vision
   selected raises `ModeError` — it will *not* quietly fall back to the scripted in-core double. A
   selected real backend whose API key is missing also raises `ModeError` naming the exact env var.
   Scripted seams that *do* get used are announced loudly (the startup banner, the
   `scripted_backends` / `scripted_conductor` health signals).

That is the fail-closed half of the seam: you either get your model, or you get an error — never a
silent double that looks like it worked.

### Wiring your adapter in

Instance (tests, custom wiring):

```python
agent = percept.Percept.create(vision=MyVision(settings), discover_plugins=False)
```

By name, via the registry:

```python
from percept import registry

registry.register("vision", "myvlm", lambda settings: MyVision(settings))
agent = percept.Percept.create(vision="myvlm")
```

From your own package, auto-discovered — advertise a `percept.backends` entry point whose value is
called with `register`:

```toml
# pyproject.toml
[project.entry-points."percept.backends"]
myvlm = "myvlm.percept_plugin:setup"
```

```python
# myvlm/percept_plugin.py
def setup(register):
    register("vision", "myvlm", lambda settings: MyVision(settings))
```

A failing plugin never breaks resolution of the others (`registry.discover_entrypoints` swallows
per-plugin errors), so an adapter that raises at import time simply *does not appear* — check
`registry.available("vision")`.

---

## The conformance suite

```bash
python -m percept.conformance myvlm.vision:MyVision          # exits 0 when conformant, 1 when not
python -m percept.conformance --static myvlm.vision:MyVision # shape only, no calls, no key needed
```

In your own test suite:

```python
from percept.conformance import check_vision_adapter

def test_percept_conformance():
    report = check_vision_adapter(MyVision(), frame=load_test_frame())
    assert report.ok, report.format()
```

The target spec is `package.module:Attr`, where `Attr` is a class, a zero-arg factory, or an
already-constructed adapter instance. Pass a **real frame** (`frame=`) whenever you can — with none,
the suite uses an opaque sentinel, exercises plumbing only, and says so in the report.

### What it checks

| check | severity | why |
|---|---|---|
| `vision.methods_present` / `vision.runtime_protocol` | required | the three-method seam |
| `judge.is_coroutine_function` | required | the brain awaits it inside the async perceive loop |
| `judge.accepts_optional_escalations` | required | every escalation kwarg present and defaulting `None` (or `**kwargs`) |
| `judge.minimal_call_returns_verdict` | required | a real `Verdict`, bool `satisfied`, normalized `confidence` |
| `judge.escalation_roi`, `judge.escalation_corrections` | required | the fovea crop and assertion-beats-observation are core cognition |
| `judge.escalation_frames` / `_audio` / `_context` / `_exemplar` | advisory | modalities you may genuinely lack — accepted, may be ignored |
| `judge.reports_usage` | advisory | token/cost accounting for the $/stream-hour SLO |
| `detect.returns_normalised_observations` | required | `Observation` shape + boxes inside `[0,1]` |
| `detect.tolerates_empty_goals` | required | the brain calls `detect(frame, [])` |
| `detect_events.returns_list` | required | `[]` is a valid answer; `None` is not |
| `injection.instance_survives_create` | required | your instance reaches `core.vision` unwrapped |
| `injection.live_core_is_fail_closed` | required | a live core with no vision raises `ModeError` |
| `injection.not_a_scripted_double` | advisory | your adapter is not reported as a scripted seam |

A **required** failure means the report is not `ok` and the CLI exits non-zero. An **advisory**
failure is reported and never fails the run.

### What it does NOT check

**Accuracy.** No check asserts a particular `satisfied` for a particular frame. A conformant adapter
is free to be *wrong*; it is not free to be *shaped* wrong. Model quality is an L2/L3 eval question —
see [Benchmark methodology](benchmark-methodology.md) and the `percept.bench` scorecards.

### Reading the report

```python
report.ok         # no required check FAILED
report.complete   # every required check actually RAN (False after --static)
report.failures / report.blocking_failures / report.skipped
report.get("judge.escalation_roi").detail
report.format()   # the text the CLI prints — use it as your assert message
```

`ok and not complete` prints as `PARTIALLY CHECKED`: shape proved, behaviour unproved. A green
`--static` run is a real result, but it is not a pass.

---

## A conformant skeleton

```python
from percept.contracts import Box, Goal, Observation, Verdict


class MyVision:
    def __init__(self, settings=None):
        self._settings = settings          # build the client lazily — no vendor import at module top

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        image = crop(frame, roi) if roi is not None else frame
        answer = await my_model(condition, image, facts=corrections or {}, extra=context or {})
        return Verdict(
            satisfied=bool(answer.yes),
            confidence=float(answer.p),               # MUST be in [0, 1]
            evidence=answer.why[:200],
            usage={"input_tokens": answer.in_tok, "output_tokens": answer.out_tok},
        )

    async def detect(self, frame, goals: list[Goal]) -> list[Observation]:
        return [
            Observation(label=d.name, box=Box(d.x / W, d.y / H, d.w / W, d.h / H),
                        confidence=float(d.score))
            for d in await my_detector(frame)
        ]

    async def detect_events(self, frames, goals) -> list:
        return []
```

Keep the vendor import *inside* the method (or in a lazily-built client). Percept's core is
stdlib-pure and its L1 test lane installs no vendor SDK; an adapter that imports at module top level
cannot be statically conformance-checked without the dependency present.

---

## See also

- [Backends](backends.md) — the five seams, names, extras, env vars.
- [`watch()` contract](watch-contract.md) — the frozen entrypoint your adapter ends up serving.
- [Registration](registration.md) — detector / concern / enricher registries (the other extension points).
