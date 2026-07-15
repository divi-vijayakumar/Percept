# Problem Spec: Streaming Judge Session — continuous low-latency perception judgment

**Status:** research problem statement, 2026-07-14
**Audience:** research assistants / applied-research collaborators (no prior Percept knowledge assumed)
**Owner:** Divi
**One line:** Replace the per-frame request/response VLM judge with a *persistent streaming session* that watches a live video (± audio) stream continuously and emits a structured, calibrated yes/no/uncertain verdict the instant a watched condition becomes true — closing the latency-and-coverage gap that makes today's judge miss brief real-world events.

---

## 1. Background: what the system does (minimal)

Percept is a "watch agent" runtime. An application arms a plain-language **goal** — e.g. *"tell me the second I raise my hand"*, *"nudge me if you see me add sugar"* — and the runtime watches a video stream and fires a durable **incident** only on the **rising edge** of the condition (false→true), then stays quiet (refractory), so it never nags.

The pipeline today:

```
video frame ──► T0 (cheap local motion gate) ──► [admit?] ──► VLM JUDGE ──► GATE ──► INCIDENT
               ~sub-ms, no model              motion         ~3–5 s        rising-edge   durable event
```

- **T0** is a cheap, local, deterministic motion/salience gate. It admits only "interesting" frames to the expensive judge. No model call.
- **JUDGE** is a vision-language model answering one narrow yes/no question about the admitted frame(s): *"is `<condition>` true in this image?"* → `Verdict{satisfied: bool, confidence: float, evidence: str}`.
- **GATE** applies rising-edge / dwell / refractory semantics and emits the incident.

The judge is the only expensive, high-latency, cloud-dependent stage. **This spec is entirely about the JUDGE stage.**

---

## 2. The problem (with measured evidence)

The judge is a **request/response** call: one discrete API round-trip per admitted frame. Measured latency against Gemini (2026-07-14):

| Judge frame resolution | Mean latency | Detects the event? |
|---|---:|---|
| 1080p (full) | 4.51 s | yes (satisfied=True, conf 0.9) |
| 720p | 3.14 s | yes |
| 512p | 3.25 s | yes |
| 384p | 2.99 s | yes |

**Key finding: the latency floor is ~3 s and is model-reasoning-bound, not image-size-bound.** Downsizing 1080p→384p saves ~30% and never falls below ~3 s. The judgment itself is *accurate* — it correctly identifies the event frame and rejects all others.

Because perception must never block on a network call, the runtime runs the judge through a **non-blocking scheduler**: single-flight with *one active request + one latest-pending candidate*. When the active request finishes, the pending (newest) admitted frame is judged; older pending frames are **superseded (dropped)**. This is correct for a live stream — a judge that can't keep up should judge the *freshest* evidence, not stale evidence.

**But when judge latency (~3–5 s) exceeds the event duration (~1–2 s), this drops the event.** Controlled reproduction on a 35 s clip (person adds sugar to a drink at t≈26 s; goal = "the person adds sugar/sweetener/syrup"):

| Judge dispatch mode | escalations | judge calls | superseded | **fires** |
|---|---:|---:|---:|---:|
| `async` (single-flight, latest-pending) | 56 | 6 | 50 | **0** ❌ |
| `inline` (judge every admitted frame, blocking) | 56 | 56 | 0 | **3–5** ✅ |

In `async`, only 6 of 56 admitted frames were judged; the sugar-pour peak was among the 50 superseded. Mechanism: while judge *N* runs (t≈24→28 s), the pour at t=26 s escalates but is superseded by newer frames; when the judge frees at t≈28 s, "latest-pending" hands it the t≈28 s *"done"* frame, not the peak. The event falls into the blind window between discrete judge calls.

`inline` catches it — but it is not viable in production: it blocks perception, and total cost/time = Σ(all admitted-frame latencies) (415–443 s of wall-clock judging for 35 s of video).

**This same mechanism explains why live webcam tests never fired** on brief actions (a sip, a hand-raise): the human event is shorter than the judge round-trip, so it slips between judge calls.

### Why the cheap fixes don't close it
- **Downsize the frame:** ~30% faster, floor still ~3 s. Necessary, insufficient.
- **Bounded judge concurrency (N>1 in flight):** helps coverage but multiplies cost per goal and complicates gate-state serialization (the gate's rising-edge latch assumes ordered verdicts per goal). A partial mitigation, not the answer.
- **Just use `inline`:** correct for offline/recorded-clip accuracy work, unusable for live low-latency.
- **"Use a streaming model (Gemini Live / OpenAI Realtime)":** see §5 — not a drop-in; these are audio-conversation-first and do not natively emit structured visual verdicts.

---

## 3. The proposed direction: a Streaming Judge Session

Replace *discrete per-frame judgment* with a **persistent session that ingests the stream continuously and emits verdicts as events occur.** There is no per-frame round-trip and nothing to supersede: the judge is always watching, so a 1–2 s event cannot slip between calls.

Conceptually:

```
                 ┌──────────────── StreamingJudgeSession (persistent) ─────────────────┐
frames + audio ─►│  continuous multimodal ingest → running perception of the condition │─► verdict stream
plan (goals)   ─►│  emits Verdict{state, score, evidence, ts_window} on state change    │   (async, push)
                 └─────────────────────────────────────────────────────────────────────┘
```

Target properties:
1. **Continuous ingest:** the session consumes the admitted frame stream (and optionally world audio) without a per-frame request boundary.
2. **Emit-on-change:** it pushes a verdict when its belief about a watched condition changes (false→true, true→false, or uncertain), not on a fixed cadence.
3. **Low detection latency:** time from *event visible in frame* → *verdict emitted* should be a small multiple of the model's inference step, not a 3–5 s round-trip. **Target: sub-second to ~1.5 s.**
4. **No superseding blind window:** brief events (≥ ~0.5 s) are not dropped.
5. **Structured, calibrated output:** the emitted verdict must carry a discrete `state ∈ {yes, no, unknown}` + a `score`/confidence + an evidence-time-window, so the existing rising-edge Gate can consume it unchanged. `unknown` must be distinct from `no`.
6. **Multi-goal:** one session may watch several conditions concurrently over the same stream, with per-goal verdicts (or a documented fan-out if the backend can't).

---

## 4. Interface the research must target (so results drop into the runtime)

The runtime already has a stable downstream contract. A streaming judge must produce **the same normalized verdict** the Gate consumes today, plus session lifecycle. Sketch (Python-ish, illustrative — final names TBD):

```python
class StreamingJudgeSession(Protocol):
    async def update_plan(self, snapshot: PlanSnapshot) -> None: ...   # which conditions to watch (goals can change mid-run)
    def offer(self, evidence: EvidenceEvent) -> OfferResult: ...       # push a frame / audio chunk; non-blocking, bounded
    def verdicts(self) -> AsyncIterator[JudgmentReport]: ...           # verdicts pushed as conditions change
    async def close(self) -> None: ...

@dataclass(frozen=True)
class JudgmentReport:
    goal_id: str
    state: Literal["yes", "no", "unknown"]     # unknown ≠ no
    score: float | None                        # calibrated-ish confidence
    evidence_start_ts: float                   # the time WINDOW the verdict is about (not a single frame)
    evidence_end_ts: float
    emitted_ts: float                          # for latency measurement
    evidence: str                              # short human-readable justification
```

Design constraints from the runtime:
- **Perception never blocks on the judge.** `offer()` is non-blocking and bounded; verdicts arrive asynchronously.
- **Verdicts are time-*window* stamped**, not single-frame — a streaming judge's evidence is inherently an interval.
- **The Gate stays the only thing that fires.** The streaming judge outputs evidence; it does not itself decide to alert. Rising-edge/dwell/refractory remain downstream.
- **Plan changes mid-session:** goals can be armed/disarmed while the session is live; `update_plan` must re-target without tearing down the stream.
- A session is a **cache, not a store** — it can be killed/rotated without losing goals or incidents (those live in durable runtime state). Reconnect logic is expected.

---

## 5. Core research questions (the actual ask)

These are open. Investigate empirically; recommend with evidence, not vibes.

**A. Model choice & feasibility.** Which currently-available model families can act as a *continuous visual judge emitting structured verdicts*?
   - Gemini Live (`bidiGenerateContent`) is **audio-output-only** — it rejects a text response modality. Today the only way it "judges" is to *speak* a verdict which is then transcribed and parsed — adding TTS+STT latency and fragility. Can it be driven to emit low-latency structured signals another way (function-calling over the live channel? tool-call emission)? Measure real detection latency.
   - OpenAI Realtime and equivalents: same question — are they usable as continuous *visual* monitors, or are they fundamentally audio-conversation engines with vision bolted on? What is the true frame→verdict latency?
   - Open-weight / local streaming VLMs (continuous video understanding models): candidates, latency, accuracy, cost, on-device feasibility.
   - Is a purpose-built small streaming detector (continuously running, emit-on-change) more appropriate than repurposing a conversational S2S model?

**B. Latency vs. accuracy vs. cost frontier.** For each viable backend, characterize:
   - detection latency (event-visible → verdict-emitted), P50/P95;
   - false-positive / false-negative rate on brief events, vs. the request/response baseline;
   - cost model (session-seconds + ingress vs. per-call tokens) — a persistent session has *different economics* than per-call; when is it cheaper/more expensive?
   - behavior under multiple concurrent conditions on one stream.

**C. Emit-on-change semantics.** How should the session decide *when* to emit? Continuous frame-by-frame emission is wasteful and noisy; fixed cadence reintroduces blind windows. What triggers a verdict push — internal belief change, salience, a lightweight on-device change-detector gating the stream? How is `unknown` represented and when is it emitted?

**D. Calibration.** Streaming verdicts must be *calibrated* enough for a consequence-weighted gate (a high-false-negative-cost watch must not be silently suppressed by an over-confident `no`). How do we obtain a confidence/score from a streaming model that means something? How does calibration drift with model/prompt/domain?

**E. Coverage guarantee.** Can we *bound* the probability that an event of duration ≥ D is missed? The request/response + superseding path has no such bound. What guarantee can a streaming session offer, and how is it verified?

**F. Pre-roll & session lifecycle.** When a session opens (or reconnects), short events during the opening handshake are systematically missed unless recent frames are replayed. What pre-roll (last N frames / M ms of audio) is needed? How are reconnects/rotations made invisible to correctness?

**G. World-audio.** Some conditions are audio ("tell me when the baby cries"). Should the streaming judge fuse world audio, and how does that interact with a system that may also be *speaking* (its own output must never be treated as world evidence)?

---

## 6. Evaluation / success criteria

A candidate is successful if, on a labeled set of **brief-event** clips (events 0.5–3 s: a sip, a hand-raise, adding sugar, a door opening, a fall):

1. **Detection latency P95 ≤ ~1.5 s** (event-visible → verdict-emitted), vs. the ~3–5 s + blind-window baseline.
2. **Miss rate on brief events materially below the `async` request/response baseline** (which missed the reference sugar event entirely: 0/1), approaching the exhaustive `inline` baseline (which caught it) — *without* `inline`'s blocking cost.
3. **Verdicts drop into the existing Gate unchanged** (structured `state`/`score`/`window`; `unknown ≠ no`).
4. **Cost characterized** and, for at least one viable configuration, competitive with or better than request/response at the same coverage.
5. **Reproducible** on the harness in the appendix.

The reference regression to beat is exactly the table in §2: same clip, same goal, `async` fires=0 / `inline` fires=3–5. A streaming judge should fire on the event **at low latency without blocking and without exhaustive per-frame cost.**

---

## 7. Constraints & non-goals

- **Non-goal: replace the Gate.** Rising-edge/dwell/refractory/incident semantics are fixed and downstream. Deliver *evidence*, not firing decisions.
- **Non-goal: the delivery/voice path.** Getting a spoken alert to a speaker (incident→audio) is a *separate* latency problem (conversational turn-taking, historically ~20 s through a held wire) and is out of scope here. This spec ends at "verdict emitted."
- **Constraint: perception never blocks on the judge.** Any design that stalls frame ingest on model I/O is disqualified.
- **Constraint: a session is a cache.** Durable goals/incidents live elsewhere; the session must be killable/rotatable without state loss.
- **Constraint: honest uncertainty.** `unknown` must be first-class and distinct from `no`.
- **Privacy note:** opening a persistent remote session streams continuous video off-device; a local/edge streaming option is of high interest for privacy-sensitive deployments.

---

## Appendix A: reproduction harness

A headless harness reproduces §2 exactly (no browser/voice, isolates perception→judge→gate→incident):

- `templates/realtime_voice_watch_agent/scratch_video_smoke.py` — arms one structured goal at t0, plays a video at real-time pace through the runtime, prints the funnel heartbeat `frames / escalations / judge_calls / superseded / fires` and every incident. Toggle `JUDGE_DISPATCH=async|inline` and `PERCEPT_MOTION_GATE=<0..1>`.
- `scratch_judge_probe.py` — samples N frames across a clip and calls the judge directly, printing raw verdicts (used to confirm the judge is accurate and locate the event frame).
- `scratch_judge_latency.py` — times judge latency across resolutions (produced the §2 latency table).

Reference clip behavior: person makes tea, adds sugar at t≈26 s; goal "the person adds sugar/sweetener/syrup". Direct judge probe: `t=26.2 s satisfied=True conf=0.9`, all other sampled frames `False`.

## Appendix B: measured baselines (2026-07-14, Gemini request/response judge)

- Per-call judge latency: 1080p 4.5 s / 720p 3.1 s / 512p 3.3 s / 384p 3.0 s (floor ≈ 3 s, reasoning-bound).
- `async` single-flight, latest-pending: 56 escalations → 6 judged, 50 superseded, **0 fires** (event missed).
- `inline` exhaustive: 56 escalations → 56 judged, 0 superseded, **3–5 fires** (event caught), 415–443 s wall-clock for 35 s of video.
- Existing `GeminiLiveVision` backend (audio-out-only, spoken-verdict-transcribed, **per-call** session — not a persistent stream) is *not* a low-latency streaming judge and if anything adds TTS+STT latency; it is a realtime-seam stepping stone, not a solution.
