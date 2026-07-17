# Percept — pt-action-e2e N=10 per-video diagnostic

**Run:** `20260717T113933Z` · **model:** `gemini-3.1-flash-lite` (live) · **benchmark:** Perception Test action localization (CC-BY-4.0)  
**Headline:** notification recall **0.30** / precision **0.30** · edge (tier-0) recall **0.80** · median latency **0.49s** · coverage 10/10

**Scoring rule (important):** a fire is a HIT only if it lands inside the *exact* window `[onset, end]` — **no tolerance/lateness**. Precision = (fires inside any window) / (all fires).

**The decisive gap:** the judge *perceived* the action **in-window in 5/10** rows, but the pipeline *fired* **in-window in only 3/10** — the gate throws away real detections.

## Where the misses are

- ✅ HIT: **3**
- 🟠 NEVER SAMPLED: **2**
- 🔵 PERCEPTION MISS: **2**
- 🔴 GATE lost it: **2**
- 🟡 LATE: **1**

**The core mechanism (hits recall AND precision at once):** the rising edge fires on the *first* satisfied frame — which the model often reaches **slightly before** the human-labeled onset. That first fire (a) lands out-of-window → false fire, and (b) trips the refractory → the real **in-window fire is suppressed** → miss. Then the still-true condition re-fires periodically out-of-window → more false fires.

**Ceiling:** even with a perfect gate, recall ≤ **5/10 (0.50)** here — 5 rows had no in-window satisfied verdict at all (2 perception, 2 sub-second windows never sampled, 1 late). So 0.90 is unreachable on this brief-action set regardless of gate tuning. This is Percept's *adversarial* case (brief transient actions, tight windows), not its persistent-state sweet spot.

---
## Per-video

| video | action / goal | window (s) | edge saw it? | judge said YES in-window? | system fires (s) | in-window fire? | outcome |
|---|---|---|---|---|---|---|---|
| `video_10849` | Launching something | [0.333, 1.333] | no | no | [] | — | 🟠 NEVER SAMPLED (no frame judged inside the window) |
| `video_6760` | Hitting something against/with something | [0.068, 6.243] | yes | no | [] | — | 🔵 PERCEPTION MISS (judge said 'no' on every in-window frame) |
| `video_3391` | Indicating a region with the hand | [2.066, 6.232] | yes | yes @ [3.5, 4.5] | [2.0, 10.5, 18.49, 24.49, 29.49]… | — | 🔴 GATE lost it (judge saw it in-window, gate didn't fire) |
| `video_8222` | Showing something to the camera | [1.534, 4.87] | yes | yes @ [3.74] | [3.74, 6.54, 12.14] | ✅ [3.74] | ✅ HIT |
| `video_2314` | Putting something on top of something | [1.483, 4.291] | yes | yes @ [1.98, 2.37, 2.77, 3.17, 3.56] | [0.0, 1.19, 1.98, 5.14, 6.72]… | ✅ [1.98] | ✅ HIT |
| `video_5920` | Launching something | [1.232, 4.993] | yes | no | [] | — | 🔵 PERCEPTION MISS (judge said 'no' on every in-window frame) |
| `video_10283` | Showing something to the camera | [2.333, 5.166] | yes | yes @ [2.4, 2.8, 3.2, 4.4] | [0.4, 2.4, 6.0, 9.2] | ✅ [2.4] | ✅ HIT |
| `video_7201` | Indicating a region with the hand | [1.401, 4.602] | yes | yes @ [3.47, 3.9, 4.34] | [1.3, 6.07, 8.67, 10.41, 13.88]… | — | 🔴 GATE lost it (judge saw it in-window, gate didn't fire) |
| `video_482` | Lifting something and placing it back down | [1.993, 4.115] | yes | no | [4.86, 7.03, 16.75, 18.92] | — | 🟡 LATE (judge confirmed only after the window) |
| `video_1915` | Moving object(s) around | [2.699, 3.599] | no | no | [6.0, 33.59, 34.79] | — | 🟠 NEVER SAMPLED (no frame judged inside the window) |

## Row-by-row detail

### `video_10849` — Launching something  →  🟠 NEVER SAMPLED (no frame judged inside the window)
- **Goal watched:** "someone is launching something"
- **Labeled window:** [0.333, 1.333]s  ·  frames judged: 59 (in-window: 0)  ·  tier-0 edge admitted a frame in-window: no
- **Judge (gemini) said satisfied in-window:** NONE
- **All satisfied timestamps:** none
- **System fired at:** never  ·  in-window fires: none  ·  out-of-window fires (false): 0
- **Sample evidence:** _The person is holding a small cylindrical object, not launching anything._
- **Where the issue is:** **Tier-0 / cadence.** The window is sub-second; no frame was judged inside it. Fixable by sampling denser near motion — but sub-second localization is inherently hard.

### `video_6760` — Hitting something against/with something  →  🔵 PERCEPTION MISS (judge said 'no' on every in-window frame)
- **Goal watched:** "someone is hitting something against/with something"
- **Labeled window:** [0.068, 6.243]s  ·  frames judged: 10 (in-window: 2)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** NONE
- **All satisfied timestamps:** none
- **System fired at:** never  ·  in-window fires: none  ·  out-of-window fires (false): 0
- **Sample evidence:** _The person's hand is resting on or near a pen on the table; there is no indication of hitting or striking motion._
- **Where the issue is:** **Perception.** The judge said 'no' on every in-window frame — a genuine model miss on a brief/subtle action.

### `video_3391` — Indicating a region with the hand  →  🔴 GATE lost it (judge saw it in-window, gate didn't fire)
- **Goal watched:** "someone is indicating a region with the hand"
- **Labeled window:** [2.066, 6.232]s  ·  frames judged: 19 (in-window: 2)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** [(3.5, 1.0), (4.5, 1.0)]
- **All satisfied timestamps:** [2.0, 3.5, 4.5, 6.5, 10.5, 18.49, 20.49, 24.49, 29.49, 33.49]
- **System fired at:** [2.0, 10.5, 18.49, 24.49, 29.49, 33.49]  ·  in-window fires: none  ·  out-of-window fires (false): 6
- **Sample evidence:** _A person's hand is visible extending onto the black surface, with the index finger pointed downward, clearly indicating a specific_
- **Where the issue is:** **Gate defect.** The judge was RIGHT in-window (conf 1.0) but an earlier fire tripped the refractory, so the in-window fire was suppressed. Fixable: delay the rising-edge commit / allow in-window re-assert / add window tolerance.

### `video_8222` — Showing something to the camera  →  ✅ HIT
- **Goal watched:** "someone is showing something to the camera"
- **Labeled window:** [1.534, 4.87]s  ·  frames judged: 12 (in-window: 3)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** [(3.74, 1.0)]
- **All satisfied timestamps:** [3.74, 6.54, 7.47, 8.41, 12.14, 13.08]
- **System fired at:** [3.74, 6.54, 12.14]  ·  in-window fires: [3.74]  ·  out-of-window fires (false): 2
- **Sample evidence:** _A hand is holding a water bottle prominently in the foreground, effectively presenting it towards the camera lens._
- **Where the issue is:** Worked. (Precision still dinged by the extra out-of-window fires — needs refractory.)

### `video_2314` — Putting something on top of something  →  ✅ HIT
- **Goal watched:** "someone is putting something on top of something"
- **Labeled window:** [1.483, 4.291]s  ·  frames judged: 63 (in-window: 7)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** [(1.98, 1.0), (2.37, 1.0), (2.77, 1.0), (3.17, 1.0), (3.56, 1.0)]
- **All satisfied timestamps:** [0.0, 1.19, 1.98, 2.37, 2.77, 3.17, 3.56, 5.14, 6.72, 9.1, 9.49, 9.89, 10.29, 10.68, 12.66, 13.05, 13.45, 15.03, 15.43, 15.82, 16.22, 16.61, 17.01, 17.41, 18.2, 19.38, 19.78, 20.17, 20.57, 20.96, 21.36, 21.76, 22.15, 22.55, 23.73, 25.32]
- **System fired at:** [0.0, 1.19, 1.98, 5.14, 6.72, 9.1, 12.66, 15.03, 18.2, 19.38, 23.73, 25.32]  ·  in-window fires: [1.98]  ·  out-of-window fires (false): 11
- **Sample evidence:** _There are several items placed on top of the desk, such as a stack of books, a croquis book, and a box of tissues._
- **Where the issue is:** Worked. (Precision still dinged by the extra out-of-window fires — needs refractory.)

### `video_5920` — Launching something  →  🔵 PERCEPTION MISS (judge said 'no' on every in-window frame)
- **Goal watched:** "someone is launching something"
- **Labeled window:** [1.232, 4.993]s  ·  frames judged: 18 (in-window: 2)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** NONE
- **All satisfied timestamps:** none
- **System fired at:** never  ·  in-window fires: none  ·  out-of-window fires (false): 0
- **Sample evidence:** _There is no launching of any object visible; the person in the frame is standing still and the items on the table are stationary._
- **Where the issue is:** **Perception.** The judge said 'no' on every in-window frame — a genuine model miss on a brief/subtle action.

### `video_10283` — Showing something to the camera  →  ✅ HIT
- **Goal watched:** "someone is showing something to the camera"
- **Labeled window:** [2.333, 5.166]s  ·  frames judged: 29 (in-window: 5)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** [(2.4, 1.0), (2.8, 1.0), (3.2, 1.0), (4.4, 1.0)]
- **All satisfied timestamps:** [0.4, 2.4, 2.8, 3.2, 4.4, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 11.6]
- **System fired at:** [0.4, 2.4, 6.0, 9.2]  ·  in-window fires: [2.4]  ·  out-of-window fires (false): 3
- **Sample evidence:** _Two fists are positioned centrally and prominently in the foreground, directly facing the camera in a gesture._
- **Where the issue is:** Worked. (Precision still dinged by the extra out-of-window fires — needs refractory.)

### `video_7201` — Indicating a region with the hand  →  🔴 GATE lost it (judge saw it in-window, gate didn't fire)
- **Goal watched:** "someone is indicating a region with the hand"
- **Labeled window:** [1.401, 4.602]s  ·  frames judged: 42 (in-window: 3)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** [(3.47, 1.0), (3.9, 1.0), (4.34, 1.0)]
- **All satisfied timestamps:** [1.3, 3.47, 3.9, 4.34, 6.07, 8.67, 10.41, 10.84, 11.27, 13.88, 14.74, 17.34, 24.28, 25.15, 27.32]
- **System fired at:** [1.3, 6.07, 8.67, 10.41, 13.88, 17.34, 24.28, 27.32]  ·  in-window fires: none  ·  out-of-window fires (false): 8
- **Sample evidence:** _A human hand is visible in the frame, with the index finger extended and pointing toward the red fabric surface._
- **Where the issue is:** **Gate defect.** The judge was RIGHT in-window (conf 1.0) but an earlier fire tripped the refractory, so the in-window fire was suppressed. Fixable: delay the rising-edge commit / allow in-window re-assert / add window tolerance.

### `video_482` — Lifting something and placing it back down  →  🟡 LATE (judge confirmed only after the window)
- **Goal watched:** "someone is lifting something and placing it back down"
- **Labeled window:** [1.993, 4.115]s  ·  frames judged: 19 (in-window: 1)  ·  tier-0 edge admitted a frame in-window: yes
- **Judge (gemini) said satisfied in-window:** NONE
- **All satisfied timestamps:** [4.86, 7.03, 16.75, 18.92]
- **System fired at:** [4.86, 7.03, 16.75, 18.92]  ·  in-window fires: none  ·  out-of-window fires (false): 4
- **Sample evidence:** _The person in the image is holding a bowl in their right hand, clearly in the process of moving it relative to the table surface._
- **Where the issue is:** **Window/label tightness.** The full action completes just after the labeled window; judge confirmed late. Borderline label + no lateness tolerance in scoring.

### `video_1915` — Moving object(s) around  →  🟠 NEVER SAMPLED (no frame judged inside the window)
- **Goal watched:** "someone is moving object(s) around"
- **Labeled window:** [2.699, 3.599]s  ·  frames judged: 50 (in-window: 0)  ·  tier-0 edge admitted a frame in-window: no
- **Judge (gemini) said satisfied in-window:** NONE
- **All satisfied timestamps:** [6.0, 7.46, 8.93, 10.26, 12.0, 13.6, 14.93, 16.53, 18.26, 19.59, 21.46, 23.19, 24.79, 26.66, 26.93, 27.19, 27.46, 27.73, 27.99, 28.26, 28.52, 28.79, 29.06, 29.32, 29.59, 29.86, 30.12, 30.39, 30.66, 30.92, 31.19, 31.46, 31.72, 31.99, 32.26, 32.52, 33.59, 34.79]
- **System fired at:** [6.0, 33.59, 34.79]  ·  in-window fires: none  ·  out-of-window fires (false): 3
- **Sample evidence:** _The hands are shown in the act of manipulating and positioning pieces of leafy greens on a plate._
- **Where the issue is:** **Tier-0 / cadence.** The window is sub-second; no frame was judged inside it. Fixable by sampling denser near motion — but sub-second localization is inherently hard.
