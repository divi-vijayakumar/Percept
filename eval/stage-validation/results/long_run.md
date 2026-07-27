# Long-horizon state run: ONE 81-minute source, many concurrent watches

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Source `eval/stage-validation/long_sources/long_run_all.mp4` — 81.6 min joined from 97 corpus videos, container-vs-sum drift 0.0 s. **73 distinct goals armed simultaneously** on one core, 315 ground-truth occurrences. Generated 2026-07-27T18:05:46Z.

Replay wall 5009 s (realtime ratio 1.023), capture 6.0 fps, provider deadline 1.5 s, decision deadline **138.7 s** — scaled to the armed-goal count: a judgment batch judges goals sequentially (percept/agent.py:4090), so the shipped single-goal 1.9 s budget cancels a 73-goal batch wholesale (percept/runtime/worker.py:1120).

Replay integrity: 24,484 frames yielded, 0 dropped by the decoder without a receipt, per-frame pacing lag p50 0.0401 s / p95 0.2107 s / max 0.2296 s.

## 1. Entity tracks

Observation stream: harness motion-blob detector at frame rate (percept/rhythm_hero.py:67 call shape).

| Quantity | Value |
|---|---|
| `max_entities` setting | 64 |
| `iou_match` setting | 0.5 |
| predictive coast horizon `coast_s` | 1.0 |
| ingest batches | 20184 |
| observations ingested | 102030 |
| tracks created (total mints) | 58101 |
| tracks live at end | 64 |
| evictions (`on_evict`) | 58037 |
| re-identifications after the coast horizon | 7934 |
| fragmentations (new id, no new observation) | 35460 |
| label switches | 0 |
| resurrections of an evicted id | 0 |
| first eviction at media | 14.99 |

Track lifetime (s): {'n': 58101, 'min': 0.0, 'p50': 0.0, 'p95': 3.6, 'max': 76.2, 'mean': 0.6466}
Touches per track: {'n': 58101, 'min': 1.0, 'p50': 1.0, 'p95': 4.0, 'max': 94.0, 'mean': 1.7561}
Re-id gap (s): {'n': 7934, 'min': 1.2, 'p50': 2.2, 'p95': 7.0, 'max': 47.0, 'mean': 3.0011}

## 2. Episode state

`Settings.episodes.window_s` = 3600.0 s, `max_episodes` = 64. Source is 4896.732 s, so the run EXCEEDS the window.

**Episode logs the RUNTIME itself created for the 73 armed watches: 0.**

| Probe-log quantity | Value |
|---|---|
| probe logs | 73 |
| observations folded | 4380 |
| episodes opened (`is_new`) | 4088 |
| continuations | 292 |
| episodes minted, all logs | 4088 |
| episodes LIVE at end, all logs | 2920 |
| largest single log at end | 40 |
| bound-driven drops | {'window_expired': 1168} |
| retained evidence bytes | 0 |

Runtime episode metrics: `{"episode.continuation": 292, "episode.new": 4088}`

## 3. Event log — completeness and ordering

| Ledger line | Units |
|---|---|
| frames yielded by the paced source | 24484 |
| frames emitted by the runtime | 24484 |
| frames dropped WITH a receipt | 0 |
| decoder drops WITHOUT a receipt | 0 |
| `edge.decision` receipts | 24484 |
| admitted (escalations) | 16631 |
| suppressed WITH a receipt | 7853 |
| perceive dispatches | 4380 |
| superseded (latest-wins) | 16571 |
| dropped: pending deadline | 0 |
| dropped: revision moved | 0 |
| expired: active decision deadline | 0 |
| scheduler receipts total | 16571 |
| admissions LAUNCHED as batches | 60 |
| **admission residual** (admitted - launched - receipted) | 0 |
| dispatch units (frame x goal) | 4380 |
| dispatches per launched batch | 73.0 |
| `judge.started` trace units | 4380 |
| `judge.finished` trace units | 4380 |
| `judge.degraded` trace units | 0 |
| started - finished - degraded | 0 |
| `gate.classified` trace units | 4380 |
| finished - gate.classified | 0 |

Primary ledger balances (ADMISSION unit): **True** (admitted - launched - receipted = 0).
Funnel mirror: `funnel.judged`=4380 vs `judge.started` trace =4380 (delta 0); `funnel.refused`=7480 vs `gate.ask`=3740 (delta 3740, double-count True).
Token cross-check adapter vs funnel: agree = True ({'funnel_input_tokens': 3680897, 'funnel_output_tokens': 22169, 'adapter_input_tokens': 3680897, 'adapter_output_tokens': 22169, 'agree': True})

## 4. Judge allocation across concurrent goals

73 goals armed; 73 were judged at least once, 0 never were. 4,380 judge dispatches, fair share 60.0 per goal, Gini of the per-goal judged count **0.0** (0 = perfectly even).

Per-goal judged: {'n': 73, 'min': 60.0, 'p50': 60.0, 'p95': 60.0, 'max': 60.0, 'mean': 60.0}
Gate actions: {'fire': 40, 'ask': 3740, 'silent': 600}; RuntimeStats asks=3740 fires=62.
Scheduler health: {'runtime_started': 1, 'judge_dispatch': 1, 'effective_config': 1, 'escalation': 16631, 'ask': 3740, 'judge_superseded': 16571, 'judge_scheduler_backpressure': 58, 'judge_scheduler_recovered': 58, 'incident': 62, 'runtime_ended': 1}

**Spend cap bit at wall 903.5 s / media 903.44 s** — up to that instant the judge was a live provider (949 dispatches over 73 goals, 62 incidents); after it, every judge call is a harness abstention that never crossed the network. Allocation numbers spanning both regimes are marked.

PRE-CAP ONLY (the live-provider window): {'judged_total': 949, 'goals_ever_judged': 73, 'goals_never_judged': 0, 'judged_per_goal': {'n': 73, 'min': 13.0, 'p50': 13.0, 'p95': 13.0, 'max': 13.0, 'mean': 13.0}, 'min_over_max': 1.0, 'gate_actions': {'fire': 40, 'ask': 260, 'silent': 576}}

| # | goal | admitted | judged | finished | fire | ask | silent | incidents |
|---|---|---|---|---|---|---|---|---|
| 0 | Add salt to the chips. | 16631 | 60 | 60 | 0 | 57 | 3 | 0 |
| 1 | Alert me when the video provides specific de | 16631 | 60 | 60 | 0 | 57 | 3 | 0 |
| 2 | Alert me whenever you see a group of people  | 16631 | 60 | 60 | 0 | 54 | 6 | 0 |
| 3 | As you watch, narrate the natural landscapes | 16631 | 60 | 60 | 0 | 53 | 7 | 0 |
| 4 | Carrot stew is being stir fried in the pan. | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 5 | Check the recipe in the manual. | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 6 | Checks the soup. | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 7 | Closing something | 16631 | 60 | 60 | 2 | 51 | 7 | 0 |
| 8 | Collect bottle of sauce from refrigerator, m | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 9 | Count how many different locations or themed | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 10 | Count how many different people are featured | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 11 | Count how many different toy vehicles are pr | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 12 | Count how many legendary tribes are introduc | 16631 | 60 | 60 | 0 | 54 | 6 | 0 |
| 13 | Count how many times the narrator is shown o | 16631 | 60 | 60 | 0 | 54 | 6 | 0 |
| 14 | Count how many times the narrator turns the  | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 15 | Cut and clean the yam. | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 16 | Cut tamarind pulp with a knife. | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 17 | Diced carrots are added to the tikka sauce. | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 18 | Disposing waste found on the countertop. | 16631 | 60 | 60 | 1 | 48 | 11 | 0 |
| 19 | Holding a bag | 16631 | 60 | 60 | 2 | 48 | 10 | 0 |
| 20 | Holding a blanket | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 21 | Holding a book | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 22 | Holding a phone/camera | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 23 | Holding a pillow | 16631 | 60 | 60 | 1 | 50 | 9 | 0 |
| 24 | Holding a shoe/shoes | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 25 | Holding a vacuum | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 26 | Holding something in a state | 16631 | 60 | 60 | 6 | 49 | 5 | 0 |
| 27 | I've just got this Multi Flask and want to t | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 28 | Indicating a region with the hand | 16631 | 60 | 60 | 2 | 51 | 7 | 0 |
| 29 | Launching something | 16631 | 60 | 60 | 2 | 51 | 7 | 0 |
| 30 | Let me know when the boy finally uncovers hi | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 31 | Let me know when the match ends with a victo | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 32 | Let me know when the reviewer starts the sou | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 33 | Monitor the charging case lid and update me  | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 34 | Monitor the narrator's presence and let me k | 16631 | 60 | 60 | 1 | 53 | 6 | 0 |
| 35 | Monitor the video and let me know whenever t | 16631 | 60 | 60 | 1 | 51 | 8 | 0 |
| 36 | Monitor whether Amanda is visible on screen  | 16631 | 60 | 60 | 0 | 53 | 7 | 0 |
| 37 | Moving object(s) around | 16631 | 60 | 60 | 7 | 52 | 1 | 0 |
| 38 | Obtain a plate as part of the preparation pr | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 39 | Peel onion. | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 40 | Please alert me whenever you see someone pro | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 41 | Pouring something from something into someth | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 42 | Pours nuts into the dough. | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 43 | Putting a blanket somewhere | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 44 | Putting a box somewhere | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 45 | Remove the outer skin from the onion. | 16631 | 60 | 60 | 1 | 50 | 9 | 0 |
| 46 | Select the beef. | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 47 | Showing something to the camera | 16631 | 60 | 60 | 0 | 53 | 7 | 0 |
| 48 | Shuffling objects | 16631 | 60 | 60 | 4 | 53 | 3 | 0 |
| 49 | Sitting in a chair | 16631 | 60 | 60 | 0 | 53 | 7 | 0 |
| 50 | Sitting on the floor | 16631 | 60 | 60 | 1 | 53 | 6 | 0 |
| 51 | Someone is cooking something | 16631 | 60 | 60 | 3 | 52 | 5 | 0 |
| 52 | Someone is laughing | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 53 | Someone is standing up from somewhere | 16631 | 60 | 60 | 1 | 53 | 6 | 0 |
| 54 | Someone is undressing | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 55 | Sprinkling shredded mozzarella cheese on the | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 56 | The cabbage is sliced. | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 57 | The gas cooker is turned on. | 16631 | 60 | 60 | 0 | 53 | 7 | 0 |
| 58 | The heat is regulated using firewood. | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 59 | The outer skin of the garlic is removed and  | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 60 | The potatoes are rinsed and then dried as pa | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 61 | When the boy shouts 'No Rio 2!', count how m | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 62 | When the first athlete group  celebrate on t | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 63 | Working at a table | 16631 | 60 | 60 | 3 | 52 | 5 | 0 |
| 64 | Writing or drawing something on a piece of p | 16631 | 60 | 60 | 0 | 52 | 8 | 0 |
| 65 | a person completes a bench_pressing | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 66 | a person completes a front raise | 16631 | 60 | 60 | 1 | 51 | 8 | 0 |
| 67 | a person completes a front_raise | 16631 | 60 | 60 | 1 | 51 | 8 | 0 |
| 68 | a person completes a jump_jack | 16631 | 60 | 60 | 0 | 50 | 10 | 0 |
| 69 | a person completes a pull_up | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 70 | a person completes a push_up | 16631 | 60 | 60 | 0 | 49 | 11 | 0 |
| 71 | a person completes a sit-up | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |
| 72 | a person completes a squat | 16631 | 60 | 60 | 0 | 51 | 9 | 0 |

## 5. Memory over the horizon

RSS start 219.4 MB -> end 132.5 MB, peak 272.5 MB over 334 samples. First-third mean 260.0 MB, last-third mean 215.5 MB (growth -44.6 MB); least-squares slope -43.33 MB/h.

| wall s | RSS MB |
|---|---|
| 15.0 | 219.4 |
| 135.1 | 234.1 |
| 255.2 | 243.1 |
| 375.2 | 251.9 |
| 495.3 | 260.8 |
| 615.5 | 265.1 |
| 735.6 | 265.3 |
| 855.7 | 266.5 |
| 975.8 | 267.8 |
| 1095.9 | 268.0 |
| 1216.0 | 269.8 |
| 1336.1 | 270.3 |
| 1456.2 | 270.6 |
| 1576.4 | 272.5 |
| 1696.5 | 223.8 |
| 1816.6 | 207.4 |
| 1936.6 | 211.6 |
| 2056.7 | 214.4 |
| 2176.9 | 209.5 |
| 2297.0 | 201.6 |
| 2417.1 | 216.2 |
| 2537.2 | 208.7 |
| 2657.2 | 207.8 |
| 2777.3 | 206.4 |
| 2897.4 | 136.1 |
| 3017.5 | 213.6 |
| 3137.6 | 212.1 |
| 3257.7 | 210.1 |
| 3377.8 | 213.2 |
| 3497.9 | 219.3 |
| 3618.0 | 226.5 |
| 3738.1 | 237.1 |
| 3858.2 | 229.8 |
| 3978.3 | 232.4 |
| 4098.4 | 231.4 |
| 4218.4 | 227.0 |
| 4338.5 | 219.8 |
| 4458.6 | 221.0 |
| 4578.7 | 220.6 |
| 4698.8 | 212.2 |
| 4818.9 | 224.4 |
| 4939.0 | 154.4 |

## Provider accounting and spend

Model `gemini-3.1-flash-lite`, 750 physical calls (3469 refused by the harness spend cap), 91 cancelled at the deadline, 0 errors {}.
Usage rows 659 (87.9% of calls); input 3,680,897 tok, output 22,169 tok; **estimated spend $0.953478** at {'input': 0.25, 'output': 1.5, 'source': 'ai.google.dev/gemini-api/docs/pricing, page last updated 2026-07-21'}.

_Tokens are measured only on calls that RETURNED a usage row. A call the pool cancelled at the 1.5 s deadline may still have been billed by the provider; the fraction of such calls is reported so the figure can be read as a LOWER bound._

Terminal route outcomes: {'cancelled': 91, 'capacity': 88, 'deadline': 91, 'decisive': 647, 'uncertain': 3554}
Provider latency, completed: {'n': 659, 'min': 0.6498, 'p50': 1.1717, 'p95': 1.4024, 'max': 1.4904, 'mean': 1.1528}
Provider latency, censored (cancelled/errored — a LOWER bound): {'n': 91, 'min': 1.4698, 'p50': 1.4955, 'p95': 1.5004, 'max': 1.521, 'mean': 1.4945}

## Limits

- Entity observations come from a harness-local motion-blob detector, not a semantic detector. Track counts are counts of MOVING REGIONS, not of people or objects. The association/eviction logic measured is the shipped EntityGraph, unmodified.
- Episode numbers come from probe logs driven through the public count_event API on unregistered goal ids. They measure EpisodeLog under the shipped Settings bounds; they do NOT show the armed watches opening episodes, because watch goals never do (see the episode section).
- The decision deadline is scaled to the goal count and is therefore NOT the shipped operating point. Every allocation number is a number for THAT deadline.
- Cost is measured tokens x a declared price, and only calls that returned a usage row carry tokens. Cancelled calls may still have been billed: the figure is a lower bound.
- One source, one provider, one model, one day, one machine.
