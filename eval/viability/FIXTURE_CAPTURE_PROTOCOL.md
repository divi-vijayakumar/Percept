# Real-Environment Fixture Capture Protocol

**Purpose:** Define the minimum viable process for capturing and labeling video fixtures from a NEW deployment environment so the profile-derivation job (L1) can run the ablation ladder and emit an environment-tuned `WatchProfile`.

**This is the raw material for the continuous-learning loop.** New environment → fixtures → profile → runtime loads it. Same harness H0 used; this protocol ensures the fixtures match that harness' expectations.

---

## 1. Fixture Set Sizing & Composition

### Minimum Viable Set
- **8–12 clips per environment** (H0 baseline: 20 Charades videos for the launch quality gate; for a new environment without the label debt, 8–12 is viable per the ablation ladder's sample efficiency)
- **Total duration: 4–8 minutes** (H0: ~600 frames @ 1 fps across 20 videos; target ≥300 frames per environment to stabilize edge-gate and judge statistics)
- **Video codec & frame rate:** Match your deployment frame rate (typical: 24–30 fps capture; eval downsamples to 1 fps internally for cost; preserve original in fixtures)

### Composition by Goal Temporality
The ablation ladder (track_charades_e2e.py) tunes differently for stative vs dynamic-verb actions. Your fixture mix should reflect the **actual workload your environment will watch:**

| Goal Type | % of Fixture | Example Actions | Rationale |
|---|---|---|---|
| **Stative** (persistent state; holding, wearing, sitting) | ~40–50% | "someone is holding a tool" / "person is sitting at desk" | Salience mode (block_max on fixed camera) is the biggest lever for these; plan primitive stays single-frame |
| **Dynamic** (motion or transition; taking, running, opening) | ~40–50% | "someone is opening a door" / "person is walking" | Temporal plan (window/sprite) is critical; ablation tunes `plan_mode` here |
| **Expression/brief** (smiling, laughing, sneezing) | ~10–20% (optional) | "someone is laughing" | Tests sub-second cadence; deferred for v0.1 |

**Scene diversity:** Include 3–4 distinct camera views/mounts if possible:
- Fixed camera (desk, entry): → `block_max` salience; localized motion
- Handheld/moving camera: → `ego_compensate` (detects global shift; later refactor)
- Multi-person scenes vs single-actor

---

## 2. Labeling Format & Intervals

### JSON Structure
Mirror the Charades `selection.json` format exactly:

```json
[
  {
    "id": "env-site-001",                    // Unique clip ID: "{environment}-{scene}-{sequence}"
    "length": 45.3,                          // Total video duration in seconds (float)
    "quality": 6,                            // Label clarity/lighting: 1–7 (higher = better)
    "relevance": 7,                          // How well clip demonstrates the goal: 1–7
    "scene": "Office desk, side-lit",        // Free-text scene description
    "actions": [                             // Array of [label, start_time_s, end_time_s]
      ["Someone is holding a coffee cup", 2.1, 8.5],
      ["Person is typing", 8.7, 22.0],
      ["Someone is standing up from desk", 42.1, 45.0]
    ]
  }
]
```

### Action Interval Labeling Rules

1. **Boundaries are inclusive, in seconds** (float precision, 0.1s granularity):
   - `start`: Frame where the action first becomes observable (e.g., hand touches cup).
   - `end`: Frame where the action ceases (e.g., cup released or set down).
   - Do NOT label the "approach" or "aftermath"; label only the persistent state or the motion itself.

2. **Non-overlapping actions in the same clip are OK** (the track will deduplicate by object key):
   - `["holding a cup", 0.0, 5.0]` and `["sitting at desk", 0.0, 30.0]` can coexist.
   - `["holding a red cup", 0.0, 5.0]` and `["holding a cup", 0.0, 5.0]` will collide (deduplicate by object); keep only the most specific.

3. **Tolerance convention** (hardcoded in track_charades_e2e.py):
   - A fire is a HIT if it lands within `[start, end + WINDOW_PAD_S]` where `WINDOW_PAD_S = 1.0` second.
   - Label conservatively (the 1s pad absorbs labeling jitter and judge latency).
   - If unsure about an exact boundary, err on the start; the end has wiggle room.

4. **Completeness** (critical for precision):
   - Label **every action instance** of your watched goals in the clip. Unlabeled instances → false-fire penalty.
   - For stative goals (holding, sitting) that persist through the clip, label the entire span (`0.0` to video length).

---

## 3. Goal Authoring & Watch Plan Hints

When you label, you are implicitly authoring goals. The profile-derivation job will convert labels → `Goal` objects using the track's `_condition()` function (idempotent prefix logic). Phrase labels so they are:

### Falsifiable & Specific
- **BAD:** "doing something" / "action" (unverifiable)
- **GOOD:** "someone is holding a coffee cup" / "person is typing on keyboard"
- **BETTER:** "person in gray shirt is holding a *full* coffee cup" (adds discriminators; helps on crowded scenes)

### Temporally Anchored
- For **stative goals**, the condition is the persistent state (e.g., "holding a blanket").
  - **Watch plan hint:** Single-frame judge suffices; the track sets `plan_mode = 'frame'` for stative verbs (listed in `_STATIVE`).
- For **dynamic goals**, the condition is the motion/transition (e.g., "taking a cup from the shelf").
  - **Watch plan hint:** Temporal input required; the track sets `plan_mode = 'window'` for dynamic verbs and folds recent frames (ablation tunes this).

### Real-Environment Context
Document context in the **`scene` field** (free text):
```json
"scene": "Home office, fixed desk camera, overhead lighting, single occupant"
```
This informs:
- **Salience mode:** Fixed camera + localized actions → `block_max` (instead of `mean`).
- **Region hint:** If actions cluster in one area of frame (e.g., always at the desk), note it (later: `region_hint` in goal state).
- **Ego compensation:** If handheld, note "handheld/mobile camera" so track can set `ego_compensate` mode.

---

## 4. Minimum Viability Checklist

Before submitting fixtures to L1, verify:

- [ ] **Format & schema:** JSON matches `selection.json` structure; timestamps are floats in seconds.
- [ ] **Count:** ≥8 clips; ≥300 total frames @ 1 fps (check: sum of `length` / clips ≥ 25s average).
- [ ] **Diversity:**
  - [ ] ≥2 distinct scenes / camera angles.
  - [ ] ~40–50% stative, ~40–50% dynamic actions.
  - [ ] ≥2 clips per scene to stabilize environment learnings.
- [ ] **Label completeness:**
  - [ ] Every action instance of watched goals is labeled (no unlabeled frames = no false-fire penalty later).
  - [ ] For stative goals, spans cover the full persistent region (e.g., "holding" goes 0.0 → video length if held throughout).
  - [ ] Overlapping actions are disambiguated (different objects or clear object key).
- [ ] **Metadata:**
  - [ ] `quality` and `relevance` are 1–7 (1 = poor, 7 = excellent).
  - [ ] `scene` description captures camera type, lighting, occupancy (single/multi-person), mount (fixed/handheld).
- [ ] **Boundaries:**
  - [ ] Start times mark the first observable frame of the action.
  - [ ] End times mark cessation (within ±0.5s; the 1s pad absorbs jitter).

---

## 5. Environment-Specific Profile Derivation

Once fixtures are captured and labeled:

1. **Input to L1 (profile-derivation job):**
   - Fixture JSON (this format).
   - Target environment type (e.g., "fixed-desk", "mobile-handheld", "warehouse-overview").
   - Bars: recall ≥0.85 (internal), precision ≥0.75 (open axis; exact bar TBD by L3).

2. **Ablation ladder (same harness H0 ran):**
   - Rungs: `plan_mode` (frame / window / sprite / episode), `salience_stat` (mean / block_max / p95), goal density (k distinct objects).
   - Per-rung: recall, precision, call reduction, latency.
   - Outputs: **tuned `WatchProfile`** + scorecard.

3. **Profile usage in MVP runtime:**
   - Runtime loads profile by (environment type, goal temporality).
   - No knobs hardcoded; all behavior resolved from profile.
   - New learning from L1 = new versioned profile, loaded same day.

---

## 6. Authoring Best Practices (from docs/how-to/authoring-goals-and-watch-plans.md)

**One concern per goal; keep active set small:**
- If two goals share an object or actor (e.g., "holding a blanket" / "holding some clothes"), expect cross-talk on single-frame judges.
- Merge competing goals or disambiguate (e.g., "holding a *blue* blanket" vs "holding a *brown* blanket").

**Salience mode matching (fixed camera rule):**
- Static camera + small/localized action (hand, cup, person at desk) → `block_max` (not `mean`).
- Leaving it on `mean` is the classic reason low-motion actions (holding, watching) are silently missed.

**Cadence tuning:**
- Stative actions: `base_fps = 1.0`, `max_fps = 2.0` (slow, cheap).
- Dynamic actions: `base_fps = 2.0`, `max_fps = 30.0` (saccade on motion).

---

## 7. Example: Mini Fixture for Home Office Environment

```json
[
  {
    "id": "home-office-001",
    "length": 35.5,
    "quality": 6,
    "relevance": 7,
    "scene": "Home office, fixed webcam above monitor, typical lighting, single occupant",
    "actions": [
      ["Someone is holding a coffee cup", 2.1, 8.5],
      ["Person is typing on keyboard", 8.7, 22.0],
      ["Someone is standing up from desk", 32.1, 35.0]
    ]
  },
  {
    "id": "home-office-002",
    "length": 28.3,
    "quality": 7,
    "relevance": 6,
    "scene": "Home office, fixed webcam, seated view, low motion",
    "actions": [
      ["Person is sitting at desk", 0.0, 28.3],
      ["Someone is reading from screen", 3.2, 15.0],
      ["Person is taking notes on paper", 15.5, 25.1]
    ]
  }
]
```

**Rationale:**
- 2 clips, ~64s total (32s average; within 8–12 min budget).
- Mix of stative (sitting, holding) and dynamic (standing, typing).
- Consistent scene (fixed camera, single-person office).
- Labels are specific ("holding a coffee cup" not "holding something").
- For stative "sitting", span is full video (0.0 → 28.3).

---

## 8. Troubleshooting & Iteration

| Problem | Likely Cause | Fix |
|---|---|---|
| Recall low after L1 runs | Action labels missed; judge never saw condition. | Re-check clip for unlabeled instances; add them. Or condition too vague; make it more specific. |
| Precision low | Too many false fires; re-detections or over-detection. | Precision track (L3) will refactor `require_false_first` and refractory tuning. Fixtures are complete; this is a configuration problem, not labeling. |
| Edge gate never escalates | Motion too low or salience mode wrong. | Check scene description; if fixed camera + small gestures, push for `block_max`. Verify `threshold` is calibrated (H0 used 0.1; may differ per environment). |
| High label disagreement (if multi-person labeling) | Boundary ambiguity. | Use the 1s `WINDOW_PAD_S` pad; label conservatively on start time; end time has wiggle room. Document decision rule before labeling. |

---

## 9. Handoff to L1

**Deliverable:** A single JSON file at `eval/viability/{environment-name}/fixtures.json` following the format above.

**Metadata file** (optional but recommended): `eval/viability/{environment-name}/METADATA.md`
```markdown
# {Environment Name} Fixture Set

**Captured:** YYYY-MM-DD  
**Camera type:** fixed-desk / mobile-handheld / warehouse-overview  
**Expected watch goals:** list of 3–5 sample goals expected in deployment  
**Ablation tuning notes:** any known constraints (e.g., "low light expected; pre-tune block_max threshold")  
```

**Input command to L1 (example):**
```bash
PERCEPT_BENCH_DATA_DIR=eval/viability \
PERCEPT_BENCH_MAX_GOALS=3 \
python -m percept.bench.trace_profile_derivation \
  --fixture-set home-office \
  --recall-bar 0.85 \
  --precision-bar 0.75 \
  --output-profile eval/viability/home-office/PROFILE.json
```

---

## Summary

This protocol ensures:
1. **Consistency:** Fixtures match H0's harness; same ablation ladder runs on new environments.
2. **Efficiency:** 8–12 clips, ~4–8 min total, enough to stabilize per-environment profiles.
3. **Traceability:** Labels → goals → profile tuning; every knob is transparent and versioned.
4. **Iteration:** Real-environment learnings flow back in via L1 → L3 (precision track), closing the loop.

Next: **Card L1** consumes these fixtures and emits versioned, environment-tuned `WatchProfile` objects the runtime loads.

