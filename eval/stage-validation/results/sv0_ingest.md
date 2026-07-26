# Stage 0 — INGEST validation (SV-0.1 … SV-0.6)

**Result: 6 / 6 PASS.** Three findings recorded against the runtime; none of them breaks a stated
pass bar, but F-1 is a real live-capture defect and should be fixed before a live-source claim.

| | |
|---|---|
| Runtime under test | `percept 1.0.0` @ `/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept/` |
| Interpreter | CPython 3.12.8 (`/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python`) |
| Suite | `eval/stage-validation/sv0_ingest.py` (machine-readable: `results/sv0_ingest.json`) |
| Wall time | ~10 s |
| Network / provider calls | none — synthetic media, a scripted `cv2` double, `FakeVision` judge |
| Determinism | pass/fail stable across repeated runs; only the wall-clock timings in SV-0.2 vary (±0.5 ms/eviction) |

| ID | Test | Status |
|---|---|---|
| SV-0.1 | Live video continuity | **PASS** |
| SV-0.2 | Backpressure policy | **PASS** |
| SV-0.3 | Audio unit ordering | **PASS** |
| SV-0.4 | Fused A/V skew | **PASS** |
| SV-0.5 | Delivered-fps honesty | **PASS** |
| SV-0.6 | Clock authority | **PASS** |

Reproduce:

```
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv0_ingest.py
```

### Are these checks worth anything?

Every check was mutation-tested before this report was written. Two deliberate runtime breakages
were injected and both were caught:

| Mutation | Expected | Observed |
|---|---|---|
| `_FusedSynchronizer._skew` forced to always return `0.0` (silent pairing) | SV-0.4 FAIL | FAIL — `skew=-0.05: reported actual_skew_s=0.0 != oracle 0.05 at anchor 0.0` |
| `SourceTracker.inspect` blinded so gaps report `missing_sequences=0` | SV-0.1 FAIL | FAIL — `the injected gap did not emit exactly one source_gap event` |

The suite also carries in-test anti-vacuity guards: SV-0.2 requires the `latest` control arm to
actually drop on the same fixture (otherwise "block dropped nothing" proves nothing), and SV-0.4
requires the sweep to produce at least one over-tolerance and one within-tolerance pairing
(otherwise neither classification branch is exercised).

---

## SV-0.1 — Live video continuity — **PASS**

**Bar:** zero unaccounted gaps.

**What it actually measured.** Four parts, all on real runtime code paths.

*A. Real decoder, lossless replay.* A synthetic 60-frame 30 fps mp4 written with `cv2.VideoWriter`,
read back through `percept.runtime.source.open_source(..., queue_policy="block")`.

| Metric | Observed |
|---|---|
| Frames decoded / captured | 60 / 60 |
| Capture drops | 0 |
| Sequence numbers | contiguous `0…59` |
| Epochs | `[0]` (no spurious reconnect) |
| Distinct formats | 1 |
| `observed_time` regressions | 0 |
| `source_time` known / regressions | 60 of 60 / 0 |
| Malformed JPEG payloads | 0 |

*B. The ingress ledger agrees.* All 60 envelopes fed through a real `SourceTracker` with
`require_source_time=True`: `{observed: 60, excluded: 0, discontinuities: 0, missing_sequences: 0}`.
The tracker neither invented a gap nor excluded a continuous frame.

*C. Injected gap + reconnect through the public `percept.run()` front door.* A scripted
`SourceEnvelope` stream: sequences `0,1,2,3,4` then `8,9,10` (three sequences never arrive), then
epoch 1 restarting at sequence 0 (a reconnect).

| Metric | Observed |
|---|---|
| Injected missing sequences | 3 |
| **Reported `stats.source_missing_sequences`** | **3** |
| **Unaccounted gap sequences** | **0** |
| Typed discontinuities | 2 (`source_gap`, `source_reconnect`) |
| `source_gap` event payload | carried `missing_sequences: 3` — the exact count, not a flag |
| Envelopes observed / excluded | 11 / 0 |
| Downstream frame-time regressions | 0 |

*D. Out-of-order arrivals are excluded, not silently accepted.* Stream `0,1,2,3` then an exact
redelivery of `3`, then stale `2`, then stale `1`, then `4`.

| Metric | Observed |
|---|---|
| Delivered / observed / excluded | 8 / 5 / 3 |
| Typed events | `source_duplicate`, `source_reorder` |
| Mis-booked as missing sequences | 0 |
| Out-of-order envelopes reaching `on_frame` | 0 (`t = [0.0, 0.25, 0.5, 0.75, 1.25]`, strictly increasing) |

**Verdict: PASS.** Every injected gap is accounted for by a typed discontinuity carrying the exact
missing count; nothing arrives downstream out of order. See F-3 for a classification nuance.

---

## SV-0.2 — Backpressure policy — **PASS** (with defect F-1 recorded)

**Bar:** fixture (`block`) path drops exactly zero; live (`latest`) path bounded memory with every
drop counted.

**What it actually measured.**

*A. `queue_policy="latest"` under a fully stalled consumer.* The real `_capture` loop driven against
a scripted `cv2`, 40 frames produced into a `maxsize=4` queue that nothing reads.

| Metric | Observed |
|---|---|
| Produced | 40 |
| Dropped counter | 36 |
| Delivered | 4 |
| **Dropped + delivered** | **40** — no drop went uncounted |
| Retained sequences | `[36, 37, 38, 39]` — the **newest**, i.e. the oldest were evicted |
| Peak queue size | 4 (never exceeded its bound) |

*B. Same fixture, same stalled consumer, both policies* (`maxsize=1`, 150 ms/frame consumer, 20
decoded frames). The `latest` arm is the anti-vacuity control.

| Policy | Delivered | Dropped | Accounted | Contiguous |
|---|---|---|---|---|
| `latest` (control) | 11 | 9 | 20 = 20 ✓ | no — `[0,2,4,…,18,19]` |
| **`block` (fixture path)** | **20** | **0** | 20 ✓ | yes |

*C. Policy surface.* `open_source(queue_policy=…)` rejects `"drop"`, `""` and `None` with
`ValueError: unknown source queue policy: …` — no silent default.

**Verdict: PASS.** The fixture path drops exactly zero; the live path stays inside its bound and
every queue-level drop is counted. Two findings recorded below (F-1, F-2).

---

## SV-0.3 — Audio unit ordering — **PASS**

**Bar:** zero reordering, zero unaccounted sample loss.

**What it actually measured.** A synthetic 16 kHz mono s16le wav of **32 123 samples** —
deliberately *not* a whole number of 0.25 s chunks, so a truncated tail would surface as loss —
streamed through `percept.runtime.audio.open_audio_source`, then normalized via
`normalize_audio_envelope` and pushed into a real `percept.events.AudioWindow`.

| Metric | Observed |
|---|---|
| Units produced | 9 (8 full + a 246-byte tail) |
| Sequence numbers | contiguous `0…8` |
| Media-order reorderings | 0 |
| Interval discontinuities (`t_start[k] != t_end[k-1]`) | 0 — none |
| Intervals match cumulative PCM sample positions | yes |
| Envelope `source_time` vs frame `t_end` mismatches | 0 |
| **Samples delivered / source** | **32 123 / 32 123** |
| **Unaccounted samples** | **0** |
| **Concatenated PCM byte-identical to the source wav payload** | **yes** |
| Declared format preserved | `(16000, 1, 's16le')` for every unit |
| Energy in `[0,1]` and equal to `pcm_energy(pcm)` | yes (range `0.06705 … 0.06836`) |
| `canonicalize_audio_frame` alterations | 0 of 9 — lossless for the declared format |
| `AudioWindow` rejections / reported discontinuities | 0 / 0 |

*Control (the timeline must detect, not flatten).* Pushing units out of order returns
`reordered=True`, and re-pushing an already-seen unit returns `duplicate=True, accepted=False`.

**Verdict: PASS.** Units are contiguous, ordered, correctly stamped on the intrinsic PCM sample
position, and normalization is byte-exact.

---

## SV-0.4 — Fused A/V skew — **PASS**

**Bar:** skew always classified; no silent pairing beyond declared tolerance.

**What it actually measured.** `_AcceptedMediaUnit`s built field-for-field the way
`percept/runtime/worker.py:5648` (video) and `:4649` (audio) build them, fed to a real
`_FusedSynchronizer` and drained in the worker's exact order (`drain_ready()` → `bundle_for()` →
`close()`). Video anchors at 0.25 s cadence; the audio lane shifted by the injected skew.

Sweep: skews `0, ±50, ±100, ±250, ±500 ms` × declared tolerances `max_skew_s ∈ {250, 100} ms`,
`window_s = 1.0`.

The oracle is computed independently from the injected schedule (distance from the video anchor to
the nearest audio interval) and compared against the runtime's `actual_skew_s`.

| Metric | Observed |
|---|---|
| Bundles emitted | 144 |
| **`actual_skew_s` disagreements with the oracle** | **0** (tolerance 1e-9) |
| Over-tolerance pairings in the sweep | 5 |
| **Of those, typed `SKEW_EXCEEDED`** | **5 / 5** |
| Within-tolerance pairings | 139 |
| Bundles reported `complete` | 101 — every one with `actual_skew_s ≤ max_skew_s` and both modalities present |
| **Bundles with neither a measured skew nor a typed uncertainty** | **0** |

Typed uncertainties raised across the sweep: `missing_audio`, `skew_exceeded`, `eos_incomplete`.

*Mid-stream discontinuity.* After 4 anchors, `advance_generation(_FusedDiscontinuity.GAP)`; the
stream resumes 4 s later.

| Metric | Observed |
|---|---|
| Generation | 0 → 1 |
| Stale pre-boundary unit accepted into the new generation | **no** |
| `generation_start_event_time` | 5.0 (the resume coordinate) |
| Post-boundary bundles | 4 |
| **Pre-boundary media leaked into a post-boundary bundle** | **none** |

**Verdict: PASS.** Skew is always classified, the reported figure matches an independent oracle
exactly, no complete bundle pairs evidence beyond the declared tolerance, and the discontinuity
hard-partitions evidence. See F-4 for what `actual_skew_s` does and does not mean.

---

## SV-0.5 — Delivered-fps honesty — **PASS**

**Bar:** a source that reports 30 fps but delivers 15 is flagged within the 2.0 s window.

**What it actually measured.** The real `_capture` loop against a scripted capture that reports
`CAP_PROP_FPS = 30` while an injected monotonic clock advances at the true delivered rate.
Heuristic constants read from the runtime: `_FPS_WINDOW_S = 2.0`, `_FPS_MIN_EVAL_S = 1.0`,
`_FPS_TOL = 0.25`.

| Source | Reported | Delivered | `fps_misreported` | Measured | Fired at |
|---|---|---|---|---|---|
| **half-rate** | 30 | 15 | **yes (once)** | **15.0** | **1.067 s** (inside the 2.0 s window) |
| quarter-rate | 30 | 7 | yes (once) | 7.0 | 1.143 s |
| honest | 30 | 30 | no | — | — |
| within tolerance | 30 | 29 | no | — | — |
| instant offline fixture (span ≈ 0) | 30 | ∞ | no | — | — |

The event is emitted at most once per open (verified: exactly one event in each firing case), and
the worker drains the capture-thread health list into `on_health` on the frame cadence
(`percept/runtime/worker.py:5900-5903`), so this reaches an operator rather than sitting in an
unread list.

**Verdict: PASS.** The half-rate source is flagged with the correct measured rate well inside the
2.0 s window; three negative controls stay quiet, including the offline-fixture case that a naive
implementation would false-fire on.

---

## SV-0.6 — Clock authority — **PASS**

**Bar:** exactly one authoritative clock per envelope; decoder source-time present **only** when
the decoder truly provides it.

**What it actually measured.**

*A. Two independent clocks.* With an advancing decoder clock, every envelope carries both
`source_time` (`0.0, 0.0333, 0.0667, …`) and `observed_time` (`0.0333, 0.0667, 0.1, …`), and the two
are never equal — they are genuinely independent, not aliased.

*B. Exactly one authority, selected by timebase.*

| `timebase` | `legacy_timebase` | `event_time` resolves to | Other clock |
|---|---|---|---|
| `"media"` | `source` | `source_time` (0.0) | `observed_time` still present (0.0333) |
| `"wall"` | `observed` | `observed_time` (0.0333) | `source_time` still present (0.0) |

Selecting the wall timebase does not destroy the media clock, and vice versa.

*C. Source time is never manufactured.*

| Case | `source_time` observed |
|---|---|
| file, POS_MSEC advancing | `[0.0, 0.0333, 0.0667, 0.1, 0.1333, 0.1667]` |
| file, POS_MSEC stuck at 0.0 | `[0.0, None, None, None, None, None]` — 0.0 is valid for frame 0; a non-advancing clock afterwards is not provenance |
| file, POS_MSEC non-finite | all `None` |
| `rtsp://`, `rtsps://`, `udp://` | all `None` |
| `webcam:0` | all `None` |
| **Timestamps matching the reported-fps cadence (`i/30`) where the decoder gave none** | **none** |

*D. A missing media clock fails closed.* `envelope.event_time` on a source-timebase envelope with
no media clock raises `ValueError: source time is unknown` — it does not silently substitute
observation time. `SourceTracker` surfaces `source_time_unknown` as a typed event, and
`require_source_time=True` **excludes** the envelope (`coverage.kind = "excluded"`).

*E. Audio.* Every unit's `source_time` equals its cumulative PCM sample position exactly (0 errors
over 5 units), and observation time is present and distinct on all of them.

*F. Legacy tuples gain nothing.* `legacy_video_envelope(timebase="observed")` leaves `source_time`
as `None`; `timebase="source"` keeps `source_time=3.0` and `observed_time=99.0` separate.

**Verdict: PASS.**

---

# Findings

## F-1 — `queue_policy="latest"` stalls the capture thread ~100 ms per eviction — **defect, medium**

`percept/runtime/source.py:70-85` (`_put_latest`), specifically **`percept/runtime/source.py:77`**:

```python
try:
    q.put(item, timeout=0.1)      # <-- blocks the capture thread for a full 100 ms
except queue.Full:
    q.get_nowait()                 # only THEN evict the oldest
    dropped[0] += 1
    q.put(item, timeout=0.1)
```

The eviction is triggered by a `queue.Full` raised **after a 100 ms blocking put**, not by the
queue being full. The drop condition is therefore "full for 100 ms", not "full".

**Measured (SV-0.2, `latest`):** 40 frames into a `maxsize=4` queue with no consumer took
**3.72 s** for 36 evictions — **103.4 ms per eviction**, an effective capture rate of
**10.7 fps** against a 30 fps nominal source.

**Measured corollary (SV-0.2, `latest_inert_below_put_timeout`):** with a 50 ms/frame consumer —
slower than the producer but faster than the 100 ms timeout — `latest` evicted **nothing** and
delivered the entire 20-frame backlog **in arrival order** over 1.03 s. Below the timeout the
freshness policy is inert and the consumer receives the *oldest* queued frame.

**Why it matters.** The policy's documented purpose is "Live streams prefer fresh evidence over
stale backlog" (`percept/runtime/source.py:51-52`) and "Put without letting a slow consumer make
capture block forever" (`:71`). "Forever" is honoured, but on a live 30 fps camera a saturated
consumer throttles `cap.read()` to ~10 Hz — the exact Nyquist halving that SV-0.5's
`fps_misreported` heuristic exists to catch, except caused by our own transport rather than the
camera. Worse, frames the driver discards while the capture thread is parked inside `q.put` never
reach the queue at all, so they are invisible to the `dropped` counter: on a live source the
per-frame accounting SV-0.2 verifies is complete only for losses that happen *at the queue*.

The same pattern exists in the audio adapter: `percept/runtime/audio.py:264-275`.

**Not a bar failure.** Memory stayed inside `maxsize` and every queue-level drop was counted, so
SV-0.2's stated bar holds. This is a live-capture throughput/freshness defect, not an accounting
one.

**Suggested fix.** Try a non-blocking `put_nowait()` first and evict immediately on `queue.Full`,
falling back to the timed put only as a race backstop.

## F-2 — the lossless `block` policy is unreachable from `percept.run()` — **API gap, low/medium**

`percept/runtime/worker.py:4301-4308` builds the capture kwargs without `queue_policy`:

```python
source_kwargs = {
    "maxsize": source_queue_size,
    "reconnects": source_reconnects,
    "reconnect_backoff_s": source_reconnect_backoff_s,
    "rtsp_tcp": source_rtsp_tcp,
    "dropped": dropped,
    "health": source_health,
}
```

`_run_worker` has no `source_queue_policy` parameter (confirmed by signature introspection in the
suite: `public_run_exposes_queue_policy: false`), and `percept/runtime/source.py:302` defaults it to
`"latest"`. So a **deterministic finite-file replay through the public front door always uses
drop-oldest**, even though `percept/runtime/source.py:314` documents `block` as "a bounded, lossless
mode for deterministic finite-file evaluation". The benchmark harness sidesteps this by calling
`open_source(..., queue_policy="block")` directly (`eval/bench/track_perception_test_e2e.py:592`).

Not silent — drops land in `RuntimeStats.frames_dropped` — but the documented lossless fixture mode
has no public path, and combined with F-1 a slow `on_incident`/`on_frame` sink can silently change
which frames a `percept.run()` file replay ever sees.

## F-3 — `source_duplicate` vs `source_reorder` is a last-sequence distinction — **informational**

`percept/runtime/envelope.py:459-462`. `SourceTracker` retains only the last accepted
`(epoch, sequence)` per stream, so only an exact redelivery of **that** sequence is typed
`source_duplicate`; an earlier sequence arriving late is typed `source_reorder`. Verified in SV-0.1
part D: `3,3,2,1` produced one `source_duplicate` and two `source_reorder`.

Both are excluded without advancing state and neither is booked as a missing sequence, so the
accounting is exact. The note is only that a consumer wanting a true duplicate-vs-late-arrival split
cannot derive it from the event name.

## F-4 — `actual_skew_s` is a coverage distance, not a lip-sync offset — **informational**

`percept/fused_sync.py:611-626` (`_skew`) returns the minimum distance from the video anchor to the
nearest audio unit. A uniformly shifted but still-overlapping audio lane therefore reports
`0.0`: SV-0.4 observed `actual_skew_s = 0.0` for injected skews of ±50, ±100 and ±250 ms whenever
the shifted lane still covered the anchor.

This is by design and is not a defect — what the synchronizer polices is **coverage**, via
`_continuous_through(audio, required_audio_start, anchor.event_end - max_skew_s)` at
`percept/fused_sync.py:782-796`, which correctly raised `MISSING_AUDIO` in every case where the
shifted lane failed to cover the required range. Flagged only because the field name invites the
reading "the runtime will tell me my A/V is 100 ms out of sync", which it will not.
