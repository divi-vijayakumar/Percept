# Long-horizon state run: ONE 81-minute source, many concurrent watches

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Source `eval/stage-validation/long_sources/long_run_smoke.mp4` — 3.3 min joined from 33 corpus videos, container-vs-sum drift 0.0 s. **6 distinct goals armed simultaneously** on one core, 1 ground-truth occurrences. Generated 2026-07-27T16:31:37Z.

Replay wall 60 s (realtime ratio 0.303), capture 6.0 fps, provider deadline 1.5 s, decision deadline **11.4 s** — scaled to the armed-goal count: a judgment batch judges goals sequentially (percept/agent.py:4090), so the shipped single-goal 1.9 s budget cancels a 73-goal batch wholesale (percept/runtime/worker.py:1120).

Replay integrity: 451 frames yielded, 0 dropped by the decoder without a receipt, per-frame pacing lag p50 0.1292 s / p95 0.2471 s / max 0.2602 s.

## 1. Entity tracks

Observation stream: harness motion-blob detector at frame rate (percept/rhythm_hero.py:67 call shape).

| Quantity | Value |
|---|---|
| `max_entities` setting | 64 |
| `iou_match` setting | 0.5 |
| predictive coast horizon `coast_s` | 1.0 |
| ingest batches | 81 |
| observations ingested | 592 |
| tracks created (total mints) | 325 |
| tracks live at end | 64 |
| evictions (`on_evict`) | 261 |
| re-identifications after the coast horizon | 26 |
| fragmentations (new id, no new observation) | 251 |
| label switches | 0 |
| resurrections of an evicted id | 0 |
| first eviction at media | 4.99 |

Track lifetime (s): {'n': 325, 'min': 0.0, 'p50': 0.0, 'p95': 2.3733, 'max': 47.4667, 'mean': 1.0219}
Touches per track: {'n': 325, 'min': 1.0, 'p50': 1.0, 'p95': 4.0, 'max': 14.0, 'mean': 1.8215}
Re-id gap (s): {'n': 26, 'min': 1.0667, 'p50': 1.6, 'p95': 43.7667, 'max': 44.4, 'mean': 9.6154}

## 2. Episode state

`Settings.episodes.window_s` = 3600.0 s, `max_episodes` = 64. Source is 198.0 s, so the run does not exceed the window.

**Episode logs the RUNTIME itself created for the 6 armed watches: 0.**

| Probe-log quantity | Value |
|---|---|
| probe logs | 6 |
| observations folded | 510 |
| episodes opened (`is_new`) | 60 |
| continuations | 450 |
| episodes minted, all logs | 60 |
| episodes LIVE at end, all logs | 60 |
| largest single log at end | 10 |
| bound-driven drops | {} |
| retained evidence bytes | 0 |

Runtime episode metrics: `{"episode.continuation": 450, "episode.new": 60}`

## 3. Event log — completeness and ordering

| Ledger line | Units |
|---|---|
| frames yielded by the paced source | 451 |
| frames emitted by the runtime | 451 |
| frames dropped WITH a receipt | 0 |
| decoder drops WITHOUT a receipt | 0 |
| `edge.decision` receipts | 451 |
| admitted (escalations) | 163 |
| suppressed WITH a receipt | 288 |
| perceive dispatches | 510 |
| superseded (latest-wins) | 78 |
| dropped: pending deadline | 0 |
| dropped: revision moved | 0 |
| expired: active decision deadline | 0 |
| scheduler receipts total | 78 |
| admissions LAUNCHED as batches | 85 |
| **admission residual** (admitted - launched - receipted) | 0 |
| dispatch units (frame x goal) | 510 |
| dispatches per launched batch | 6.0 |
| `judge.started` trace units | 510 |
| `judge.finished` trace units | 510 |
| `judge.degraded` trace units | 0 |
| started - finished - degraded | 0 |
| `gate.classified` trace units | 510 |
| finished - gate.classified | 0 |

Primary ledger balances (ADMISSION unit): **True** (admitted - launched - receipted = 0).
Funnel mirror: `funnel.judged`=510 vs `judge.started` trace =510 (delta 0); `funnel.refused`=980 vs `gate.ask`=490 (delta 490, double-count True).
Token cross-check adapter vs funnel: agree = True ({'funnel_input_tokens': 112506, 'funnel_output_tokens': 620, 'adapter_input_tokens': 112506, 'adapter_output_tokens': 620, 'agree': True})

## 4. Judge allocation across concurrent goals

6 goals armed; 6 were judged at least once, 0 never were. 510 judge dispatches, fair share 85.0 per goal, Gini of the per-goal judged count **0.0** (0 = perfectly even).

Per-goal judged: {'n': 6, 'min': 85.0, 'p50': 85.0, 'p95': 85.0, 'max': 85.0, 'mean': 85.0}
Gate actions: {'fire': 0, 'ask': 490, 'silent': 20}; RuntimeStats asks=490 fires=0.
Scheduler health: {'runtime_started': 1, 'judge_dispatch': 1, 'effective_config': 1, 'escalation': 163, 'ask': 490, 'judge_superseded': 78, 'judge_scheduler_backpressure': 5, 'judge_scheduler_recovered': 5, 'runtime_ended': 1}

**Spend cap bit at wall 31.6 s / media 31.55 s** — up to that instant the judge was a live provider (36 dispatches over 6 goals, 0 incidents); after it, every judge call is a harness abstention that never crossed the network. Allocation numbers spanning both regimes are marked.

PRE-CAP ONLY (the live-provider window): {'judged_total': 36, 'goals_ever_judged': 6, 'goals_never_judged': 0, 'judged_per_goal': {'n': 6, 'min': 6.0, 'p50': 6.0, 'p95': 6.0, 'max': 6.0, 'mean': 6.0}, 'min_over_max': 1.0, 'gate_actions': {'fire': 0, 'ask': 14, 'silent': 16}}

| # | goal | admitted | judged | finished | fire | ask | silent | incidents |
|---|---|---|---|---|---|---|---|---|
| 0 | Add salt to the chips. | 163 | 85 | 85 | 0 | 82 | 3 | 0 |
| 1 | Alert me when the video provides specific de | 163 | 85 | 85 | 0 | 82 | 3 | 0 |
| 2 | Checks the soup. | 163 | 85 | 85 | 0 | 81 | 4 | 0 |
| 3 | Count how many different toy vehicles are pr | 163 | 85 | 85 | 0 | 82 | 3 | 0 |
| 4 | Count how many legendary tribes are introduc | 163 | 85 | 85 | 0 | 82 | 3 | 0 |
| 5 | Count how many times the narrator is shown o | 163 | 85 | 85 | 0 | 81 | 4 | 0 |

## 5. Memory over the horizon

RSS start 235.4 MB -> end 284.0 MB, peak 284.0 MB over 12 samples. First-third mean 248.7 MB, last-third mean 282.1 MB (growth 33.4 MB); least-squares slope 3074.46 MB/h.

| wall s | RSS MB |
|---|---|
| 5.0 | 235.4 |
| 10.0 | 247.5 |
| 15.0 | 255.1 |
| 20.0 | 256.9 |
| 25.0 | 259.2 |
| 30.1 | 270.7 |
| 35.1 | 279.2 |
| 40.1 | 279.5 |
| 45.1 | 279.8 |
| 50.1 | 282.1 |
| 55.1 | 282.5 |
| 60.0 | 284.0 |

## Provider accounting and spend

Model `gemini-3.1-flash-lite`, 24 physical calls (476 refused by the harness spend cap), 4 cancelled at the deadline, 0 errors {}.
Usage rows 20 (83.3% of calls); input 112,506 tok, output 620 tok; **estimated spend $0.029056** at {'input': 0.25, 'output': 1.5, 'source': 'ai.google.dev/gemini-api/docs/pricing, page last updated 2026-07-21'}.

_Tokens are measured only on calls that RETURNED a usage row. A call the pool cancelled at the 1.5 s deadline may still have been billed by the provider; the fraction of such calls is reported so the figure can be read as a LOWER bound._

Terminal route outcomes: {'cancelled': 4, 'capacity': 4, 'deadline': 4, 'decisive': 20, 'uncertain': 482}
Provider latency, completed: {'n': 20, 'min': 1.0383, 'p50': 1.2226, 'p95': 1.3476, 'max': 1.3583, 'mean': 1.2244}
Provider latency, censored (cancelled/errored — a LOWER bound): {'n': 4, 'min': 1.4888, 'p50': 1.4913, 'p95': 1.4946, 'max': 1.495, 'mean': 1.4916}

## Limits

- Entity observations come from a harness-local motion-blob detector, not a semantic detector. Track counts are counts of MOVING REGIONS, not of people or objects. The association/eviction logic measured is the shipped EntityGraph, unmodified.
- Episode numbers come from probe logs driven through the public count_event API on unregistered goal ids. They measure EpisodeLog under the shipped Settings bounds; they do NOT show the armed watches opening episodes, because watch goals never do (see the episode section).
- The decision deadline is scaled to the goal count and is therefore NOT the shipped operating point. Every allocation number is a number for THAT deadline.
- Cost is measured tokens x a declared price, and only calls that returned a usage row carry tokens. Cancelled calls may still have been billed: the figure is a lower bound.
- One source, one provider, one model, one day, one machine.
