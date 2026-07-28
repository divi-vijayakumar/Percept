---
title: Tunables
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
generated_by: tools/build_tunables_doc.py
---

# Tunables — every behaviour-affecting knob

**This page is generated** from `percept.tunables.REGISTRY` by `tools/build_tunables_doc.py`. Do not hand-edit it — change the registry and regenerate (`make docs-tunables`). That is why it cannot drift from what ships.

Percept exposes **157 tunables**. Of those, **31 are `safe_to_optimize`** — the
only ones `percept-optimize` is permitted to search. The rest are declared, typed, and owned,
but changing them is a deliberate act, not a search-space dimension.

| | count |
|---|---|
| total tunables | **157** |
| safe to optimize | **31** |
| owned by `WatchPlan` | 54 |
| owned by `SourceProfile` | 39 |
| owned by `DeploymentConfig` | 29 |
| owned by `ExecutionPolicy` | 14 |
| owned by `unclassified` | 12 |
| owned by `Gate` | 9 |

## How to read a row

- **path** — the dotted registry path. This is the key an `OptimizationLock` or a
  `DomainPack.values` mapping writes to.
- **owner** — which configuration surface may legally set it. The resolver enforces this;
  a write from the wrong surface is refused, not silently applied.
- **env** — the `PERCEPT_*` variable, when one exists. Read once by `load_settings()`, so a
  run is reproducible from its environment.
- **risk** — blast radius if set wrong.

---

## The 31 safe-to-optimize tunables

These are the search space. `percept-optimize --space space.json` may propose values inside
`bounds`; the resulting `OptimizationLock` is provenance-stamped and validated before it
takes effect.

| path | owner | type | default | bounds | env | risk | note |
|---|---|---|---|---|---|---|---|
| `Settings.cache.audit_rate` | ExecutionPolicy | float | `0.05` | 0.0 – 1.0 | `PERCEPT_CACHE_AUDIT_RATE` | low | fraction of cache HITs re-perceived for audit. |
| `Settings.cache.tau_bits` | ExecutionPolicy | int | `6` | 0.0 – 64.0 | `PERCEPT_CACHE_TAU_BITS` | low | dHash Hamming near-match radius. |
| `Settings.cache.threshold` | ExecutionPolicy | float | `0.0` | 0.0 – 1.0 | `PERCEPT_CACHE_THRESHOLD` | medium | cascade short-circuit cutoff. |
| `Settings.cache.ttl_s` | ExecutionPolicy | float | `8.0` | — | `PERCEPT_CACHE_TTL_S` | low | hard ceiling on slow drift before a cache HIT is distrusted. |
| `Settings.routing.confidence_threshold` | ExecutionPolicy | float | `0.6` | 0.0 – 1.0 | `PERCEPT_ROUTE_THRESHOLD` | low | below this, the local router defers to the conductor (defer-on-miss is structural). |
| `WatchProfile.cadence.base_fps` | ExecutionPolicy | float | `4.0` | 0.0 – inf | — | low | fovea quiet floor; ExecutionPolicy row's own example ('cadence ceilings'). |
| `WatchProfile.cadence.escalate_threshold` | ExecutionPolicy | float | `0.15` | 0.0 – 1.0 | — | low | motion level that triggers a saccade toward max_fps. |
| `WatchProfile.cadence.hold_s` | ExecutionPolicy | float | `0.5` | 0.0 – inf | — | low | keeps an escalation alive briefly after a spike. |
| `WatchProfile.cadence.max_fps` | ExecutionPolicy | float | `15.0` | 0.0 – inf | — | low | fovea cost ceiling; must be >= base_fps (CadenceProfile.__post_init__). |
| `AcousticOnset.high` | SourceProfile | float | `0.3` | 0.0 – 1.0 | — | low | hysteresis-crossing high threshold (energy onset). |
| `AcousticOnset.low` | SourceProfile | float | `0.15` | 0.0 – 1.0 | — | low | hysteresis-crossing low threshold (re-arm). |
| `AcousticOnset.refractory_s` | SourceProfile | float | `0.0` | 0.0 – inf | — | low | detector-level debounce (distinct from Gate.refractory_s). |
| `MotionPeriodicity.min_height` | SourceProfile | float | `0.02` | 0.0 – 1.0 | — | low | ignores still-scene jitter below this motion magnitude. |
| `MotionPeriodicity.refractory_s` | SourceProfile | float | `0.0` | 0.0 – inf | — | low | detector-level debounce (distinct from Gate.refractory_s, the fire-suppression window). |
| `MotionPeriodicity.rel_drop` | SourceProfile | float | `0.5` | 0.0 – 1.0 | — | low | a bump confirms only once the signal falls to (1-rel_drop) of its peak. |
| `PoseOpenness.min_height` | SourceProfile | float | `0.6` | 0.0 – inf | — | low | inner MotionPeriodicity floor on the openness waveform. |
| `PoseOpenness.refractory_s` | SourceProfile | float | `0.4` | 0.0 – inf | — | low | detector-level debounce between counted reps. |
| `PoseOpenness.rel_drop` | SourceProfile | float | `0.4` | 0.0 – 1.0 | — | low | inner MotionPeriodicity descent-confirmation fraction. |
| `Settings.edge.motion_gate` | SourceProfile | float | `0.1` | 0.0 – 1.0 | `PERCEPT_MOTION_GATE` | medium | mean-stat motion escalation cutoff. |
| `Settings.edge.motion_gate_block_max` | SourceProfile | float | `0.2` | 0.0 – 1.0 | `PERCEPT_MOTION_GATE_BLOCK_MAX` | medium | block_max-scale escalation cutoff (calibrated: 0.20 -> 0.85 recall). |
| `Settings.edge.motion_gate_p95` | SourceProfile | float | `0.15` | 0.0 – 1.0 | `PERCEPT_MOTION_GATE_P95` | medium | p95-scale escalation cutoff (calibrated: 0.15 -> 0.79 recall). |
| `Settings.entity.iou_match` | SourceProfile | float | `0.5` | 0.0 – 1.0 | `PERCEPT_IOU_MATCH` | medium | frame-to-frame continuity threshold; identity_strategy calibration (SourceProfile row). |
| `Settings.episodes.window_s` | SourceProfile | float | `3600.0` | — | `PERCEPT_EPISODE_WINDOW_S` | medium | counting window ('today's cups'); episode shape. |
| `Settings.events.audio_window_s` | SourceProfile | float | `4.0` | — | `PERCEPT_AUDIO_WINDOW_S` | low | rolling seconds of audio per source. |
| `Settings.events.merge_gap_s` | SourceProfile | float | `1.5` | — | `PERCEPT_EVENT_MERGE_GAP_S` | low | same-type detections within this = 1 event. |
| `Settings.events.window` | SourceProfile | int | `12` | 1.0 – inf | `PERCEPT_FRAME_WINDOW` | low | rolling frames per source for temporal perception. |
| `Settings.identity.appearance_threshold` | SourceProfile | float | `0.8` | 0.0 – 1.0 | `PERCEPT_APPEARANCE_MATCH` | medium | cosine sim to re-id across a gap; identity_strategy calibration. |
| `SourceProfile.parameters.coast_interval_s` | SourceProfile | float | — | 0.0 – inf | — | low | fixed low-rate sampling floor when the change signal is absent/untuned. |
| `SourceProfile.parameters.unknown_continuity_gap_s` | SourceProfile | float | — | 0.0 – inf | — | low | default_episode_policy input: unknown-to-unknown temporal-proximity continuity window (percept.sourceprofile.SourceProfile.default_episode_policy); off by default. |
| `WakePolicy.floor` | SourceProfile | float | `0.0` | 0.0 – 1.0 | — | low | minimum fused score floor. |
| `WakePolicy.threshold` | SourceProfile | float | `0.1` | 0.0 – 1.0 | — | medium | fused wake score cutoff before spending a heavy vision call; admission-strategy shape. |

---

## Full registry, by owner

### `DeploymentConfig` — 29 tunables (0 safe to optimize)

| path | type | default | bounds / choices | env | safe? | risk | note |
|---|---|---|---|---|---|---|---|
| `Percept.create.allow_scripted_brain` | bool | `false` | — | — | — | high | safety guard for a LIVE run with a scripted conductor. |
| `Percept.create.conductor` | str | — | — | — | — | medium | explicit-kwarg mirror of Settings.models.conductor. |
| `Percept.create.discover_plugins` | bool | `true` | — | — | — | low | entry-point plugin discovery; deployment posture. |
| `Percept.create.embedder` | callable | — | — | — | — | medium | embedding provider wiring. |
| `Percept.create.identity` | object | — | — | — | — | medium | identity backend/provider wiring. |
| `Percept.create.mode` | enum | — | offline – live | `PERCEPT_MODE` | — | high | offline/live: selects scripted vs real backends; deployment/environment posture. |
| `Percept.create.specialist` | str | — | — | — | — | medium | explicit-kwarg mirror of Settings.models.specialist. |
| `Percept.create.stt` | str | — | — | — | — | medium | explicit-kwarg mirror of Settings.models.stt. |
| `Percept.create.tts` | str | — | — | — | — | medium | explicit-kwarg mirror of Settings.models.tts. |
| `Percept.create.vision` | str | — | — | — | — | high | explicit-kwarg mirror of Settings.models.vision; same DeploymentConfig ownership. |
| `PoseOpenness.provider_name` | str | — | rtmo – blazepose | `PERCEPT_POSE_PROVIDER` | — | medium | pose provider selection; lazy + opt-in (percept-vision[pose]). |
| `Settings.consent.actuation` | enum | `none` | none – screen – physical | `PERCEPT_ACTUATION` | — | high | 'physical' is a DECLARATION every act is refused (ADR-007; the gate cannot ask a loop). |
| `Settings.consent.allow_computer_use` | bool | `false` | — | `PERCEPT_ALLOW_COMPUTER_USE` | — | high | gates the `act` modality; deployment permission. |
| `Settings.consent.allowed_actions` | frozenset | `frozenset()` | — | `PERCEPT_ACT_ALLOWED_ACTIONS` | — | high | comma-separated action scope for the `act` modality. |
| `Settings.models.conductor` | str | — | — | `PERCEPT_CONDUCTOR_BACKEND` | — | medium | provider NAME for the conductor seam. |
| `Settings.models.conductor_model` | str | `claude-haiku-4-5` | — | `PERCEPT_CONDUCTOR_MODEL` | — | low | model id carried to the conductor adapter. |
| `Settings.models.specialist` | str | — | — | `PERCEPT_SPECIALIST_BACKEND` | — | medium | provider NAME for the specialist seam. |
| `Settings.models.specialist_model` | str | `claude-haiku-4-5` | — | `PERCEPT_SPECIALIST_MODEL` | — | low | model id carried to the specialist adapter. |
| `Settings.models.stt` | str | `fake` | — | `PERCEPT_STT_BACKEND` | — | medium | speech-to-text provider NAME. |
| `Settings.models.tts` | str | `fake` | — | `PERCEPT_TTS_BACKEND` | — | medium | text-to-speech provider NAME. |
| `Settings.models.vision` | str | — | — | `PERCEPT_VISION_BACKEND` | — | high | provider NAME (never a model id — percept.provider.ProviderError guards this). |
| `Settings.models.vision_model` | str | `claude-opus-4-8` | — | `PERCEPT_VISION_MODEL` | — | medium | model id CARRIED to the vision adapter, never a selector (Invariants). |
| `backend.percept_home` | str | — | — | `PERCEPT_HOME` | — | low | on-disk home directory for the label store; read at process/module init, needs a restart. |
| `backend.pose_channels` | str | — | — | `PERCEPT_POSE_CHANNELS` | — | low | channel-selection override for pose decomposition. |
| `backend.pose_device` | str | `cpu` | — | `PERCEPT_POSE_DEVICE` | — | low | inference device for the pose provider. |
| `backend.pose_model_path` | str | — | — | `PERCEPT_POSE_MODEL` | — | medium | on-disk/URI pose model override. |
| `backend.rtmo_model` | str | `rtmo` | — | `PERCEPT_RTMO_MODEL` | — | medium | RTMO pose model variant selection. |
| `backend.strip_prompt_variant` | enum | `change` | change – neutral | `PERCEPT_STRIP_PROMPT` | — | low | gemini backend strip-judging prompt variant switch. |
| `backend.vision_timeout_s` | float | — | 0.0 – inf | `PERCEPT_VISION_TIMEOUT_S` | — | medium | per-backend request timeout (anthropic/gemini_live/openrouter backends). |

### `ExecutionPolicy` — 14 tunables (9 safe to optimize)

| path | type | default | bounds / choices | env | safe? | risk | note |
|---|---|---|---|---|---|---|---|
| `Settings.cache.audit_rate` | float | `0.05` | 0.0 – 1.0 | `PERCEPT_CACHE_AUDIT_RATE` | ✅ | low | fraction of cache HITs re-perceived for audit. |
| `Settings.cache.enabled` | bool | `false` | — | `PERCEPT_CACHE_ENABLED` | — | medium | wires the FrameHashLayer cost-cascade short-circuit into the pipeline. |
| `Settings.cache.tau_bits` | int | `6` | 0.0 – 64.0 | `PERCEPT_CACHE_TAU_BITS` | ✅ | low | dHash Hamming near-match radius. |
| `Settings.cache.threshold` | float | `0.0` | 0.0 – 1.0 | `PERCEPT_CACHE_THRESHOLD` | ✅ | medium | cascade short-circuit cutoff. |
| `Settings.cache.ttl_s` | float | `8.0` | — | `PERCEPT_CACHE_TTL_S` | ✅ | low | hard ceiling on slow drift before a cache HIT is distrusted. |
| `Settings.edge.coast_enabled` | bool | `true` | — | `PERCEPT_COAST_ENABLED` | — | medium | whether a SourceProfile-declared coast floor is bound into the runtime; strict deployments disable via ExecutionPolicy. |
| `Settings.entity.max_entities` | int | `64` | 1.0 – inf | `PERCEPT_MAX_ENTITIES` | — | low | eviction bound; resource ceiling (ExecutionPolicy row: 'concurrency ceiling'-shaped). |
| `Settings.episodes.max_episodes` | int | `64` | 1.0 – inf | `PERCEPT_MAX_EPISODES` | — | low | bound on the per-goal episode log; resource ceiling. |
| `Settings.events.preroll_max_bytes` | int | `134217728` | 0.0 – inf | `PERCEPT_PREROLL_MAX_BYTES` | — | medium | byte budget for the incident pre-roll ring; the measured 'runaway memory' fix. |
| `Settings.routing.confidence_threshold` | float | `0.6` | 0.0 – 1.0 | `PERCEPT_ROUTE_THRESHOLD` | ✅ | low | below this, the local router defers to the conductor (defer-on-miss is structural). |
| `WatchProfile.cadence.base_fps` | float | `4.0` | 0.0 – inf | — | ✅ | low | fovea quiet floor; ExecutionPolicy row's own example ('cadence ceilings'). |
| `WatchProfile.cadence.escalate_threshold` | float | `0.15` | 0.0 – 1.0 | — | ✅ | low | motion level that triggers a saccade toward max_fps. |
| `WatchProfile.cadence.hold_s` | float | `0.5` | 0.0 – inf | — | ✅ | low | keeps an escalation alive briefly after a spike. |
| `WatchProfile.cadence.max_fps` | float | `15.0` | 0.0 – inf | — | ✅ | low | fovea cost ceiling; must be >= base_fps (CadenceProfile.__post_init__). |

### `Gate` — 9 tunables (0 safe to optimize)

| path | type | default | bounds / choices | env | safe? | risk | note |
|---|---|---|---|---|---|---|---|
| `Settings.gate.key_ttl_s` | float | `3600.0` | — | `PERCEPT_GATE_KEY_TTL_S` | — | medium | R-1 growth bound: evict a gate key untouched this long (0/neg = off). |
| `Settings.gate.max_keys` | int | `0` | 0.0 – inf | `PERCEPT_GATE_MAX_KEYS` | — | medium | hard LRU ceiling on gate keys (0 = no ceiling). |
| `Settings.gate.refractory_s` | float | `0.0` | — | `PERCEPT_GATE_REFRACTORY_S` | — | high | Refractory re-fire suppression window (precision-tuning finding's RE-FIRES source). |
| `Settings.gate.require_false_first` | bool | `false` | — | `PERCEPT_GATE_REQUIRE_FALSE_FIRST` | — | high | require a confident-false baseline before a key can fire (v0.1 hero matrix). |
| `Settings.gate.stale_ask` | bool | `false` | — | `PERCEPT_GATE_STALE_ASK` | — | high | A-3b: folds perception recency into freshness; the ONE switch that changes WHAT FIRES. |
| `Settings.gate.threshold` | float | `0.7` | 0.0 – 1.0 | `PERCEPT_GATE_THRESHOLD` | — | high | confident-yes cutoff; the Gate is the sole firing authority (ADR-001, ADR-008 Invariants). |
| `Settings.gate.validity_window_s` | float | `90.0` | — | `PERCEPT_GATE_VALIDITY_S` | — | medium | asserted-fact freshness window. |
| `WatchProfile.gate.refractory_s` | float | `0.5` | 0.0 – inf | — | — | high | typed handle onto the re-arm interval; same Gate-only ownership. |
| `WatchProfile.gate.require_false_first` | bool | `false` | — | — | — | high | typed handle onto Settings.gate.require_false_first; same Gate-only ownership. |

### `SourceProfile` — 39 tunables (22 safe to optimize)

| path | type | default | bounds / choices | env | safe? | risk | note |
|---|---|---|---|---|---|---|---|
| `AcousticOnset.high` | float | `0.3` | 0.0 – 1.0 | — | ✅ | low | hysteresis-crossing high threshold (energy onset). |
| `AcousticOnset.low` | float | `0.15` | 0.0 – 1.0 | — | ✅ | low | hysteresis-crossing low threshold (re-arm). |
| `AcousticOnset.refractory_s` | float | `0.0` | 0.0 – inf | — | ✅ | low | detector-level debounce (distinct from Gate.refractory_s). |
| `DetectorManifest.category` | enum | — | motion – audio – frame – presence – pose – audio_event – change – semantic_relevance | — | — | low | coarse sense CLASS; the one manifest field with a source-class-shaped taxonomy. |
| `MotionPeriodicity.min_height` | float | `0.02` | 0.0 – 1.0 | — | ✅ | low | ignores still-scene jitter below this motion magnitude. |
| `MotionPeriodicity.refractory_s` | float | `0.0` | 0.0 – inf | — | ✅ | low | detector-level debounce (distinct from Gate.refractory_s, the fire-suppression window). |
| `MotionPeriodicity.rel_drop` | float | `0.5` | 0.0 – 1.0 | — | ✅ | low | a bump confirms only once the signal falls to (1-rel_drop) of its peak. |
| `PoseOpenness.min_height` | float | `0.6` | 0.0 – inf | — | ✅ | low | inner MotionPeriodicity floor on the openness waveform. |
| `PoseOpenness.refractory_s` | float | `0.4` | 0.0 – inf | — | ✅ | low | detector-level debounce between counted reps. |
| `PoseOpenness.rel_drop` | float | `0.4` | 0.0 – 1.0 | — | ✅ | low | inner MotionPeriodicity descent-confirmation fraction. |
| `Settings.edge.ego_compensate` | bool | `false` | — | `PERCEPT_EGO_COMPENSATE` | — | medium | remove global camera motion before diffing. |
| `Settings.edge.motion_gate` | float | `0.1` | 0.0 – 1.0 | `PERCEPT_MOTION_GATE` | ✅ | medium | mean-stat motion escalation cutoff. |
| `Settings.edge.motion_gate_block_max` | float | `0.2` | 0.0 – 1.0 | `PERCEPT_MOTION_GATE_BLOCK_MAX` | ✅ | medium | block_max-scale escalation cutoff (calibrated: 0.20 -> 0.85 recall). |
| `Settings.edge.motion_gate_p95` | float | `0.15` | 0.0 – 1.0 | `PERCEPT_MOTION_GATE_P95` | ✅ | medium | p95-scale escalation cutoff (calibrated: 0.15 -> 0.79 recall). |
| `Settings.edge.salience_stat` | enum | `mean` | mean – block_max – p95 | `PERCEPT_SALIENCE_STAT` | — | medium | deployment-wide default localizing statistic (empirical tuning: block_max ~doubled recall). |
| `Settings.entity.iou_match` | float | `0.5` | 0.0 – 1.0 | `PERCEPT_IOU_MATCH` | ✅ | medium | frame-to-frame continuity threshold; identity_strategy calibration (SourceProfile row). |
| `Settings.episodes.window_s` | float | `3600.0` | — | `PERCEPT_EPISODE_WINDOW_S` | ✅ | medium | counting window ('today's cups'); episode shape. |
| `Settings.events.audio_window_s` | float | `4.0` | — | `PERCEPT_AUDIO_WINDOW_S` | ✅ | low | rolling seconds of audio per source. |
| `Settings.events.merge_gap_s` | float | `1.5` | — | `PERCEPT_EVENT_MERGE_GAP_S` | ✅ | low | same-type detections within this = 1 event. |
| `Settings.events.window` | int | `12` | 1.0 – inf | `PERCEPT_FRAME_WINDOW` | ✅ | low | rolling frames per source for temporal perception. |
| `Settings.identity.appearance_threshold` | float | `0.8` | 0.0 – 1.0 | `PERCEPT_APPEARANCE_MATCH` | ✅ | medium | cosine sim to re-id across a gap; identity_strategy calibration. |
| `SourceProfile.admission_strategy` | str | — | — | — | — | high | the ownership table's own 'source-class strategy' example. |
| `SourceProfile.default_timebase` | enum | `auto` | auto – media – wall | — | — | medium | the honest default clock for this source. |
| `SourceProfile.evidence_strategy` | str | — | — | — | — | medium | how evidence is packaged for the judge and the incident. |
| `SourceProfile.identity_strategy` | str | — | — | — | — | high | the ownership table's own 'source-class strategy' example. |
| `SourceProfile.modalities` | frozenset | — | — | — | — | medium | what the stream carries, e.g. {'video'}, {'audio','text'}. |
| `SourceProfile.name` | str | — | — | — | — | low | open-registry identity key. |
| `SourceProfile.notes` | str | — | — | — | — | low | provenance/honesty text (doctor quotes it). |
| `SourceProfile.parameters.change_signal` | str | — | — | — | — | medium | UNTUNED marker or a registered detector name; honesty-about-admission mechanism. |
| `SourceProfile.parameters.coast_interval_s` | float | — | 0.0 – inf | — | ✅ | low | fixed low-rate sampling floor when the change signal is absent/untuned. |
| `SourceProfile.parameters.coast_required` | bool | `false` | — | — | — | low | the profile's own assertion that its coast floor is not optional; read by the resolver and doctor so a strict_admission override of the coast is reported LOUDLY, never silently dropped. |
| `SourceProfile.parameters.coverage_floor` | float | — | — | — | — | medium | ADR-008 §Corrections(b)/§Open questions #1, resolved: a per-source-class minimum admission count over a run, declared explicitly and opt-in (default None = nothing declared). Must be explicit before a zero-admit run is ever reported 'degraded' rather than quiet (percept.doctor's own rule) -- an undeclared floor leaves RuntimeHealthEvent('zero_admission_coverage') exactly as diagnostic as it always was; a declared-and-unmet floor is what makes it read as degraded (percept.runtime.worker.run's coverage_floor= kwarg, sourced from this field via percept.runtime_compiler). |
| `SourceProfile.parameters.unknown_attaches_to_active_bound` | bool | `false` | — | — | — | low | default_episode_policy input: single-focus attach; off by default. |
| `SourceProfile.parameters.unknown_continuity_gap_s` | float | — | 0.0 – inf | — | ✅ | low | default_episode_policy input: unknown-to-unknown temporal-proximity continuity window (percept.sourceprofile.SourceProfile.default_episode_policy); off by default. |
| `SourceProfile.stability` | enum | `stable` | stable – experimental | — | — | low | experimental profiles may change strategy without a contract bump. |
| `SourceProfile.temporal_unit` | str | — | — | — | — | medium | what one unit of time IS on this source: 'frame'/'page'/'turn'. |
| `WakePolicy.floor` | float | `0.0` | 0.0 – 1.0 | — | ✅ | low | minimum fused score floor. |
| `WakePolicy.threshold` | float | `0.1` | 0.0 – 1.0 | — | ✅ | medium | fused wake score cutoff before spending a heavy vision call; admission-strategy shape. |
| `WakePolicy.weights` | dict | — | — | — | — | medium | per-channel weights in the calibrated weighted-OR fusion. |

### `WatchPlan` — 54 tunables (0 safe to optimize)

| path | type | default | bounds / choices | env | safe? | risk | note |
|---|---|---|---|---|---|---|---|
| `Goal.state.cadence_s` | float | — | 1.0 – inf | — | — | low | interval-rearm period; clamped to _MIN_CADENCE_S=1.0 so it can't busy-loop. |
| `Goal.state.deadline_after_s` | float | — | 0.0 – inf | — | — | low | deadline-rearm DURATION from arm time (never an absolute stamp — A-2). |
| `Goal.state.delivery.preempt` | enum | `queue` | interrupt – queue – drop | — | — | medium | requested attention priority; AttentionPolicy CLAMPS 'interrupt' absent earned authority (percept.incidents.AttentionPolicy.clamp) — the request alone never grants it. |
| `Goal.state.delivery.preferred_channel` | str | — | — | — | — | low | additive app hint, absent in the legacy dict -> None. |
| `Goal.state.delivery.ttl_s` | float | — | — | — | — | low | drop-if-stale bound on a queued delivery. |
| `Goal.state.delivery.verbatim` | bool | `false` | — | — | — | low | speak the authored text verbatim vs. paraphrase. |
| `Goal.state.episode.allow_reopen` | bool | — | — | — | — | low | whether a closed episode may reopen. |
| `Goal.state.episode.continuity_gap_s` | float | — | 0.0 – 86400.0 | — | — | low | clamped under a day; beyond that is not a 'gap'. |
| `Goal.state.episode.policy` | enum | — | — | — | — | medium | must name a shipped percept.episodes.EPISODE_POLICIES preset. |
| `Goal.state.episode.terminal_statuses` | list | — | — | — | — | low | bounded set (<=16, <=40 chars each) of statuses that close an episode. |
| `Goal.state.false_fire_cost` | enum | — | trivial – mild – severe – irreversible | — | — | high | consequence declaration; closed vocab, never optimizer-authored. |
| `Goal.state.fire_when` | str | — | — | — | — | high | nth-distinct predicate, AST-whitelisted (compile_fire_when); authored, never optimized. |
| `Goal.state.from` | str | — | — | — | — | low | transition-type from-predicate (plain language). |
| `Goal.state.hold_s` | float | — | 0.0 – inf | — | — | medium | R-1 sustained-duration dwell before firing. |
| `Goal.state.identity_policy` | str | — | — | — | — | medium | percept.identity_policy.identity_policy_of; overrides SourceProfile.identity_strategy for this concern only (percept.identity_policy.select_identity_policy). |
| `Goal.state.judge_mode` | enum | — | episode_memory | — | — | medium | percept.episode_memory.episode_memory_enabled; additive opt-in, no default-path change. |
| `Goal.state.miss_cost` | enum | — | trivial – mild – severe | — | — | high | consequence declaration; closed vocab, never optimizer-authored. |
| `Goal.state.perception.audio_s` | float | — | 0.0 – 5.0 | — | — | low | authored audio clip length, clamped to _MAX_AUDIO_S=5.0. |
| `Goal.state.perception.modality` | enum | — | vision – audio – fused | — | — | medium | which channel the judge reads. |
| `Goal.state.perception.n_frames` | int | — | 1.0 – 6.0 | — | — | low | strip frame count, clamped to _MAX_STRIP=6 (token-bomb ceiling). |
| `Goal.state.perception.roi` | enum | — | focus – hint – full | — | — | low | region-of-interest framing for the judge. |
| `Goal.state.perception.sampling` | enum | — | recent – uniform | — | — | low | which frames a strip/window samples. |
| `Goal.state.perception.temporal` | enum | — | single – strip – window – sprite | — | — | medium | evidence-gathering shape; Corrections(a): screen/document defaults must be 'window', never 'sprite' (percept.agent._normalize_plan_mode normalizes a stale 'sprite' with a warning). |
| `Goal.state.perception.window_s` | float | — | 0.0 – 30.0 | — | — | low | clip window seconds, clamped to _MAX_WINDOW_S=30.0; implies temporal='window'. |
| `Goal.state.protected_floor` | bool | — | — | — | — | high | can only be raised true, never authored false-down. |
| `Goal.state.rearm` | enum | `on_false` | on_false – never – interval – deadline – each_event | — | — | medium | re-arm policy after a fire. |
| `Goal.state.rhythm.amp_gate_frac` | float | — | 0.0 – 1.0 | — | — | low | fraction of expected_rom a rep must reach to count. |
| `Goal.state.rhythm.cadence_s` | float | — | 0.0 – 30.0 | — | — | low | rough expected rep period, clamped to _MAX_WINDOW_S=30.0. |
| `Goal.state.rhythm.canonical_select` | bool | `false` | — | — | — | low | opt-in canonical-grid channel scoring vs legacy medoid. |
| `Goal.state.rhythm.channel` | str | — | — | — | — | low | channel hint biasing RhythmTracker selection. |
| `Goal.state.rhythm.expected_rom` | float | — | 0.0 – 100.0 | — | — | low | expected range-of-motion, channel-native units; only honoured with a named channel. |
| `Goal.state.rhythm.expected_rom_space` | enum | `channel_native` | channel_native | — | — | low | unit-contract tag; only 'channel_native' is honoured today. |
| `Goal.state.rhythm.mode` | enum | — | discrete – continuous | — | — | low | rep regime; continuous is a recorded research seam, not yet counted. |
| `Goal.state.rhythm.smooth_s` | float | — | 0.0 – 1.0 | — | — | low | opt-in causal-smoothing span; rate-invariance knob. |
| `Goal.state.rhythm.whole_body` | bool | — | — | — | — | low | flags a multi-limb action. |
| `Goal.state.rhythm.window_s` | float | — | 0.0 – 30.0 | — | — | low | endurance window for windowed rep counts. |
| `Goal.state.sense.base_fps` | float | — | 0.0 – 60.0 | — | — | low | per-concern quiet-floor fps; explicit author override of the fovea's cadence. |
| `Goal.state.sense.escalate_threshold` | float | — | 0.0 – 1.0 | — | — | low | motion level that triggers the per-concern saccade to max_fps. |
| `Goal.state.sense.max_fps` | float | — | 0.0 – 60.0 | — | — | low | per-concern motion-escalated ceiling fps; never inverted below base_fps. |
| `Goal.state.to` | str | — | — | — | — | low | transition-type to-predicate (plain language). |
| `Goal.state.type` | enum | `watch` | watch – transition – count – rhythm | — | — | medium | which BehaviorProgram sense program_of() compiles to (behavior.py). |
| `Goal.state.window_s` | float | — | 0.0 – inf | — | — | low | accumulate window for a fire_when count. |
| `Percept.create.plan_mode` | enum | — | single – strip – window – sprite | — | — | medium | explicit temporal-shape override (Corrections(a): must never default to 'sprite'). |
| `Percept.create.sense_policy` | object | — | — | — | — | medium | explicit cadence override, bypassing WatchProfile resolution; explicit user intent. |
| `Percept.create.settings` | object | — | — | — | — | high | an explicit Settings override — ADR-008's own silent-last-writer-wins motivating example. |
| `WatchProfile.environment` | str | — | — | — | — | low | registry key segment (free-text, e.g. 'fixed_camera'/'handheld'/'wearable'). |
| `WatchProfile.goal_type` | str | — | — | — | — | low | registry key segment (free-text, e.g. 'stative'/'dynamic'/'periodic'). |
| `WatchProfile.name` | str | — | — | — | — | low | registry identity, not a tuned value. |
| `WatchProfile.plan_mode` | enum | `single` | single – strip – window – sprite | — | — | medium | explicit per-context evidence-gathering shape; Corrections(a) 'sprite' deprecation applies. |
| `WatchProfile.salience_stat` | enum | `mean` | mean – block_max – p95 | — | — | medium | explicit per-context override of Settings.edge.salience_stat; the ADR's own 'explicit salience_stat="block_max" override' WatchPlan example. |
| `WatchProfile.stat_thresholds` | dict | — | — | — | — | medium | per-stat escalation cutoffs (0..1 each); explicit companion to salience_stat. |
| `WatchProfile.state` | str | — | — | — | — | low | registry key segment (free-text, e.g. 'start_anchored'/'transition'). |
| `watch.profile` | object | — | — | — | — | medium | the WatchProfile/Settings selection a caller passes to watch(); WatchPlan's own example. |
| `watch.source` | str | `camera` | — | — | — | low | frame source key; WatchPlan ownership row's own example. |

### `unclassified` — 12 tunables (0 safe to optimize)

| path | type | default | bounds / choices | env | safe? | risk | note |
|---|---|---|---|---|---|---|---|
| `DetectorManifest.name` | str | — | — | — | — | low | self-describing registry metadata; no owner among the six exists yet (DomainPack-shaped). |
| `DetectorManifest.output` | str | — | — | — | — | low | the boundary shape it emits; descriptive metadata. |
| `DetectorManifest.senses` | str | — | — | — | — | low | the physical signal read; descriptive metadata. |
| `DetectorManifest.when_to_use` | str | — | — | — | — | low | conductor-authoring hint; DomainPack-shaped ('domain vocabulary') but DomainPack does not exist in code yet. |
| `Percept.create.audit` | callable | — | — | — | — | low | audit hook; wiring seam. |
| `Percept.create.clock` | callable | — | — | — | — | low | test/infra clock injection seam, not a tuning knob. |
| `Percept.create.on_health` | callable | — | — | — | — | low | observability hook; not an ownership-table concern. |
| `Percept.create.on_trace` | callable | — | — | — | — | low | observability hook; not an ownership-table concern. |
| `WatchProfile.goal_density_guidance.max_recommended_goals` | int | `3` | 1.0 – inf | — | — | low | advisory only, NOTHING enforces it (module docstring); empirical tuning held <=3. |
| `WatchProfile.goal_density_guidance.note` | str | — | — | — | — | low | free-text guidance carried alongside max_recommended_goals. |
| `WatchProfile.notes` | str | — | — | — | — | low | free-text provenance, not a runtime value. |
| `watch.percept` | object | — | — | — | — | low | pre-built Percept core reuse; a wiring seam, not a configuration value. |

---

## Related

- [Configuration reference](configuration.md) — the `Settings` tree and resolver precedence.
- [Run an optimization](../how-to/optimize.md) — searching the safe subset into a lock.
- [Create a custom configuration](../how-to/custom-configuration.md) — authoring a profile.
- [Tune admission](../how-to/tune-admission.md) — the knobs that decide what reaches a judge.
