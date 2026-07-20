# Configuration — what to set, and why

> **# TO BE UPDATED** — a working record of configuration learnings, not a finished reference.
> Written by hand on 2026-07-20, so it can drift from the code. Every claim below carries a
> `file:line` citation; check the citation before trusting the number. The intent is to fold this
> into a generated artifact (see [Open questions](#open-questions)).

## The core problem this page exists to record

**The best-known configuration is not the default configuration.** The tuned values live in the
`percept-watch` CLI ([`watch_cli.py:53`](../packages/percept-vision/percept/watch_cli.py#L53),
`_apply_good_defaults`), which sets them as env vars before importing `percept`. The library
defaults in [`config.py`](../packages/percept-vision/percept/config.py) are deliberately the
conservative, byte-for-byte-compatible ones.

So a developer who follows the embedding path — `percept.run(...)` / `Percept.create(...)`, which is
what [integration.md](integration.md) teaches — silently gets the *untuned* gate:

| Setting | Library default | `percept-watch` forces | Why it matters |
|---|---|---|---|
| `PERCEPT_SALIENCE_STAT` | `"mean"` ([config.py:175](../packages/percept-vision/percept/config.py#L175)) | `block_max` | recall roughly doubled on the H0 set (0.31 → 0.59) |
| `PERCEPT_GATE_REFRACTORY_S` | `0.0` — off ([config.py:109](../packages/percept-vision/percept/config.py#L109)) | `15.0` | without it, one event yields a stream of duplicate notifications |
| `PERCEPT_EGO_COMPENSATE` | `False` ([config.py:176](../packages/percept-vision/percept/config.py#L176)) | `--egomotion` flag | a moving camera escalates ~98% of frames; ~30% with it |

There is no warning when a live watch runs on the untuned path. It just underperforms quietly.
**If you are embedding Percept rather than running the CLI, set these yourself.**

## The tier-0 eye: which salience statistic

The tier-0 gate decides *whether to spend a cloud call*. The statistic it reduces the frame-diff
with is the single highest-leverage knob measured so far.

- `mean` (default) — whole-frame average motion. A small, localized action (holding a cup, sitting
  down) is averaged away by the still background and never escalates.
- `block_max` — the most active block. Localized, so low-motion actions survive.
- `p95` — 95th percentile; between the two.

**Thresholds are per-statistic and are not interchangeable.** `block_max`/`p95` sit on a higher value
scale than the mean, so the mean-tuned `0.10` over-fires them
([config.py:177-182](../packages/percept-vision/percept/config.py#L177)):

| Statistic | Env var | Default cutoff | Measured recall |
|---|---|---|---|
| `mean` | `PERCEPT_MOTION_GATE` | `0.10` | 0.06 |
| `block_max` | `PERCEPT_MOTION_GATE_BLOCK_MAX` | `0.20` | 0.85 |
| `p95` | `PERCEPT_MOTION_GATE_P95` | `0.15` | 0.79 |

Calibrated on Perception Test action events (N=80, 7.8k frames). If you change the statistic,
change the threshold with it — see [reference/edge-policy.md](reference/edge-policy.md).

## Physical events vs content events

The motion gate is the right instrument for **physical** events only. A screenshare or dashboard
changes *content* with ~zero pixel motion, so the motion gate never escalates and the agent sees
nothing. Use a fixed cadence instead ([how-to/screenshare.md](how-to/screenshare.md)):

```python
state={
    "perception": {"temporal": "strip", "n_frames": 3},
    "cadence": {"min_judge_interval": 0.5, "gate_on": "fixed"},
}
```

Diagnostic: if a dry run shows the judge was **never called**, that is a salience problem, not a
prompt problem — the gate never escalated, so no cloud call was ever spent. Compare peak motion
against the threshold before touching the goal wording.

## Moving cameras (glasses / body-cam)

Set `PERCEPT_EGO_COMPENSATE=true` to subtract global camera/head motion before diffing. Without it
a moving camera makes the whole frame "move" every tick: measured ~98% of frames escalating on real
first-person footage, dropping to ~30% with compensation.

Leave it **off** for a fixed camera — it is not free and there is nothing to subtract.

Honest limit: ego-compensation makes the *gate* economical, but detection is still frame-by-frame
VLM. Fast or blurry head motion can smear frames past the point of judging; raising `--fps` helps
catch a clean one. Detecting an object is solid; tracking a *specific instance* across occlusion is
the entity layer and is weaker.

## Models — unresolved

"What model should I use?" currently has more than one answer in-tree, depending on which surface
you read. Recording the divergence rather than pretending it is settled:

| Surface | Vision model |
|---|---|
| Library default ([config.py:137](../packages/percept-vision/percept/config.py#L137)) | `claude-opus-4-8` |
| `percept-watch`, openrouter ([watch_cli.py:34](../packages/percept-vision/percept/watch_cli.py#L34)) | `qwen/qwen3-vl-32b-instruct` |
| `percept-watch`, gemini | `gemini-3.1-flash-lite` |
| `percept-watch`, claude | `claude-sonnet-5` |
| [reference/backends.md](reference/backends.md) | `claude-opus-4-8`; says Gemini falls back to `gemini-2.5-flash` |

Both layers are internally defensible — the CLI's are *demo* defaults chosen for cost and latency,
per its own comment, while `config.py` holds the library default. Neither is labelled as the answer
to "what should I ship with", which is why the question resolves inconsistently.

**Not yet decided.** Do not treat any row above as the recommendation until this section says so.

## Gate semantics worth knowing

- `PERCEPT_GATE_THRESHOLD` — default `0.7` ([config.py:85](../packages/percept-vision/percept/config.py#L85)).
  A verdict below this is **not** a fire.
- An `ask` is a refusal to guess, not a miss. Treat it as "I don't know", never as a weak fire.
- The refractory suppresses repeats of the *same goal*; one fire per episode is the intended shape.

## Open questions

- **Should the tuned values become the library defaults?** That is a behaviour change — the
  `config.py` comments show byte-for-byte compatibility was a deliberate constraint — so it needs a
  version bump and a changelog entry. The alternative is keeping the defaults but making the gap
  visible, e.g. warning when a live watch runs on `mean` with no refractory.
- **Which model do we actually recommend?** See above; needs a decision, then one place to state it.
- **This page should not be hand-maintained.** Every value here is already declared in code —
  `config.py` dataclass fields carry their env var and description as inline comments, and
  [`tools/build_architecture.py`](../tools/build_architecture.py) already proves the pattern of
  generating a doc artifact from code with a CI gate that fails on staleness. Until that exists,
  this page is exactly the kind of prose that drifts.
