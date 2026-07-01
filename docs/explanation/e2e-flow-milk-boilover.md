# E2E flow — *"tell me when the milk is about to boil over"*

A concrete trace of one request through the whole system **as it exists today**, to locate the bottleneck.
TL;DR: the **brain (cognition) is ready and cheap; the bottleneck is entirely the eyes (perception)** — and
this example stresses *every* face of the perception ceiling at once.

---

## 0. The two halves

```
            ┌──────────────────────── COGNITION (the "brain") ─────────────────────────┐
            │  conductor · gate · executor · concern framework · entity graph          │
            │  general, LLM-authored, deterministic, cheap, stateful (has memory)       │
            └──────────────────────────────────────────────────────────────────────────┘
                                          ▲   feeds on
                                          │   Verdict{satisfied, confidence}
            ┌──────────────────────────── PERCEPTION (the "eyes") ─────────────────────┐
            │  vision.judge(condition, frame) -> Verdict      ★ ONE seam               │
            │  GeminiVision / AnthropicVision · stateless · one frame · ~1s/call cloud  │
            └──────────────────────────────────────────────────────────────────────────┘
```

Everything above the line we built/generalized this cycle. Everything the request actually *depends on* to
work lives below the line.

---

## 1. The full pipeline

```
USER 🎤  "tell me when the milk is about to boil over"
   │  speech → text  (agent.on_speech_endpoint)
   ▼
┌──────────────┐   high-confidence local rule?  no → it's goal-authoring
│  ROUTER      │ ───────────────────────────────────────────► DEFER
└──────┬───────┘
       ▼
┌──────────────┐   the ONLY tier that writes standing state
│ CONDUCTOR 🧠 │   authors a Concern/Goal:
│ (LLM, once)  │      condition = "the milk is about to boil over"
└──────┬───────┘      say       = "Heads up — the milk is about to boil over!"
       │              behavior  = watch · fire on the rising edge   (no anticipation primitive exists)
       ▼
  ═════════ a standing goal now lives in the brain; nothing fires yet ═════════

  then, for EVERY frame the camera streams:
       │
       ▼
┌──────────────┐  motion in the pot ROI?  → escalate, else skip      (on-device, optional in V0)
│ EDGE (tier0) │
└──────┬───────┘
       ▼
┌──────────────┐  ROI unchanged since last verdict? → reuse cached    (server; pure cost saver)
│ CACHE        │
└──────┬───────┘
       ▼
╔══════════════╗  vision.judge("the milk is about to boil over", frame)
║ VLM  "EYES"  ║       →  Verdict{satisfied, confidence, evidence}     ★★★  PERCEPTION
║ GeminiVision ║  one frame · stateless · ~1s round-trip
╚══════┬═══════╝
       ▼
┌──────────────┐  classify(verdict) → yes / no / unknown
│ GATE  🧠     │  rising-edge (fire only false→true) + refractory (no double-fire)
│ gate.py      │       →  FIRE · ASK · SILENT
└──────┬───────┘
       ▼
┌──────────────┐  the (sense·key·accumulate) program + consequence guards
│ EXECUTOR 🧠  │  (watch · on_false here); high-stakes → could escalate to ASK / verify
│ behavior.py  │
└──────┬───────┘
       ▼
🔊  "Heads up — the milk is about to boil over!"   (TTS / notification)
```

---

## 2. Stage-by-stage — what happens today

| # | stage | what happens for this request | healthy? |
|---|---|---|---|
| 1 | **Conductor authoring** | Parses the NL into a watch goal with `condition="the milk is about to boil over"`. **"about to" is baked into the *condition string*** — there is no anticipation/predictor primitive (deferred), so the system will judge *"is it about-to-boil-over **right now**?"* every frame. | ✅ brain fine; ⚠️ pushes a *predictive* judgment down onto a *present-tense* perceptual read |
| 2 | **Edge / Cache** | Rising foam = motion + changing ROI → frames are escalated and re-perceived (the cache won't wrongly serve a stale "calm" verdict). | ✅ |
| 3 | **VLM judge (perception)** | `GeminiVision.judge` sends **one frame** + the condition → `Verdict{satisfied, confidence}`. It can see a *snapshot* ("foam is near the rim") but not a *rate* ("foam is rising fast"), and it must *predict* the near future from that snapshot. | ❌ **the bottleneck** |
| 4 | **Gate** | Turns the verdict stream into exactly one alert on the rising edge; refuses to guess when confidence is low (→ ASK). Works perfectly *on whatever the eyes report*. | ✅ |
| 5 | **Executor / concern fw** | Runs the watch program; if this were marked high-consequence it would escalate/verify. Byte-for-byte deterministic. | ✅ |
| 6 | **Output** | Speaks the alert. | ✅ (if it fired in time) |

**Five of six stages are healthy and ready. The whole request lives or dies at stage 3.**

---

## 3. Where the bottleneck is — the boil-over timeline

Boil-over is a **fast, dynamic, multi-sensory** event. Here it is against what a stateless, ~1 fps,
~1s/call vision-only pipeline can actually do:

```
THE REAL EVENT  (physics — a few seconds of usable warning):
  t=0s            t=4s              t=7s         t=8s
  calm ░░░    foam forming ▒▒    foam AT RIM ▓▓   OVER ████  (milk hits the burner)
  │──────────────│──────────────────│─────────────│
                                    └──────────────┘
                                    the ~1–2s window where an alert is actually useful

THE SYSTEM TODAY  (vision-only, single-frame, cloud round-trip):
  frame@t=5 ──judge(~1s)──► "no"            (snapshot: foam present but it can't see it RISING)
  frame@t=6 ──judge(~1s)──► conf 0.4 → ASK  ("is that about to boil over?" — refuses to guess)
  frame@t=7 ──judge(~1s)──► "yes" → FIRE ───► reply reaches the user ≈ t=8s+   ❌ TOO LATE
                                              (sampling gap + per-call latency + TTS)
```

Four distinct perception limits collide here — **all "eyes", all in the deferred track:**

```
 ┌─ 1. STATELESS / no temporal context ───────────────────────────────────────────┐
 │  "about to boil over" is a RATE (foam rising fast). One frame = a snapshot.       │
 │  The model sees "foam high" but not "foam rising 2 cm/s". Needs frames[].          │
 │  FIX: windowed perception  vision.judge(condition, frames[])   (Track B1, deferred)│
 └────────────────────────────────────────────────────────────────────────────────┘
 ┌─ 2. ANTICIPATION ──────────────────────────────────────────────────────────────┐
 │  "about TO" = predict the near future. System judges the present tense only.      │
 │  FIX: an anticipation/predictor primitive  (deferred — false-fire-prone, unbuilt)  │
 └────────────────────────────────────────────────────────────────────────────────┘
 ┌─ 3. ACOUSTIC BLIND SPOT ────────────────────────────────────────────────────────┐
 │  Boil-over SOUNDS before it overflows: hiss intensifies, lid rattles, bubbling.   │
 │  Vision-only never hears the earliest, clearest cue — a DIFFERENT sensor.          │
 │  FIX: acoustic sense + fuse  (Track B2, NOT built — the strongest miss here)       │
 └────────────────────────────────────────────────────────────────────────────────┘
 ┌─ 4. WALL-CLOCK LATENCY ─────────────────────────────────────────────────────────┐
 │  The event is over in ~2s; ~1 fps sampling + ~1s/call + round-trip + TTS can blow │
 │  the window. This is "reaction latency" — NOT on any benchmark leaderboard,        │
 │  separate from temporal-accuracy. Today: UNMEASURED.                                │
 │  FIX: edge-side fast trigger + denser sampling on escalation; measure it.          │
 └────────────────────────────────────────────────────────────────────────────────┘
```

And underneath, the **confidence ceiling** we proved this session: even when it does fire, the model's
`confidence` doesn't reliably separate right from wrong (Claude pinned high → confident-wrong fires;
Gemini pinned low → over-asks; the gate threshold sweep on Gemini is **flat 0.5→1.0**). So the gate has no
trustworthy dial to lean on — by design it falls back to ASK rather than guess.

---

## 4. The honest verdict

```
   request: "tell me when the milk is about to boil over"
   ─────────────────────────────────────────────────────
   COGNITION (brain) :  ✅ authored, gated, deterministic, cheap, ready
   PERCEPTION (eyes) :  ❌ stateless · vision-only · ~1s/call
                           cannot reliably see "rising", predict "about to",
                           hear the boil, or react inside the ~2s window
   ─────────────────────────────────────────────────────
   BOTTLENECK = 100% perception. The brain is waiting on eyes it doesn't have yet.
```

This is the same wall as RepCount Δ0, one level more vivid: the cognition layer can only act on the
`(satisfied, confidence)` stream it's handed — it cannot perceive a rate, predict a future, hear a sound,
or beat the clock. **Moving this needle is a perception-track build** (windowed `frames[]`, an anticipation
primitive, **acoustic fusion**, and an edge-side fast trigger with reaction-latency measurement) — none of
which is cognition-layer work, and all of which we deliberately parked behind the perception milestone.

> The milk example is the perfect "buy the eyes" argument: a safety-shaped, fast, multi-sensory event where
> the brain is fully built and every remaining gap is a sensor or a clock.
```
