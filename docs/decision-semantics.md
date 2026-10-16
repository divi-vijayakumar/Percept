# Decision semantics

This page states the rules Percept uses to decide whether your agent hears about something. They
are implemented by one pure function per stream (`percept/decide.py`): given the stream's state,
one input and the current wall-clock time, it returns the next state, the decisions to deliver
and the work the runtime must do (timers, verification and recheck requests, health events).
The same inputs always give the same decisions, which is what makes [trace replay](delivery.md#trace-and-replay)
exact.

Section names in brackets in the source code, such as `[Evidence floor]`, refer to the sections
below.

## 0. Terms

- **Key**: one `(condition_id, subject)` pair on one stream. The subject is `"scene"` unless you
  pass another one to `Monitor.poke(..., subject=...)`. Every rule below applies per key.
- **Occurrence**: one real-world instance of a condition becoming true. It has an
  `occurrence_id`, and every decision about it carries that id.
- **Phase**: where a key is in its lifecycle.
  - A state condition is `idle`, `confirming`, `verifying` or `holding`.
  - An event condition is `idle`, `candidate`, `verifying` or `reported`.
  - Any phase but `idle` is *open*.
- **Detect answer**: a judge answer to an ordinary look. **Verify answer**: an answer to a
  verification request (a second, stricter look; see [Events](#11-events)).

## 1. Time

- Every time the decision rules store is wall-clock (Unix seconds), so a saved state needs no
  conversion after a restart.
- Two clocks are read, both wall-clock:
  - **evidence time**, the end of the interval the judged images cover
    (`Observation.evidence_t[1]`). It measures the confirm window ([Confirm window](#9-confirm-window))
    and the evidence floor ([Evidence floor](#6-evidence-floor));
  - **arrival time**, when the answer or timer is processed. It measures
    `candidate_deadline_s`, `rearm_s`, `cooldown_s`, `max_open_s`, `recheck_every_s`, snoozes, and
    when an unfinished confirm window gives up.
- **Deadlines are data, timers are hints.** Every deadline is stored on the key. Each input that
  touches a key first applies every stored deadline that has passed. A timer that fires late, twice
  or never cannot change an outcome; one that fires early is rescheduled.
- **A wall clock that steps backwards** counts as no time elapsed for cooldown, and a zero
  cooldown never suppresses anything. Deadlines already stored simply apply later.

## 2. Answers

Every answer that is not stale ([Stale results](#5-stale-results)) falls in exactly one class:

| Class | Rule | Effect on state |
|---|---|---|
| failed | the judge timed out, was unavailable or refused, or its reply was malformed or left the condition out | **none, in every phase** |
| weak | `yes` or `no` with `confidence < min_confidence` | **none, in every phase** |
| unknown | the model answered `"unknown"`, at any confidence | **none, in every phase**, except a verify answer: see [Unverified candidates](#12-unverified-candidates) |
| yes / no | confident: `confidence >= min_confidence` | as described in [States](#10-states) and [Events](#11-events) |

- **Failures are never answers.** A failure is reported on the health channel
  (`judge_unavailable`) and counted in coverage (`failed`). It never counts as a "no", never
  clears a state and never confirms anything.
- **`min_confidence` is symmetric.** It applies to `yes` and `no` alike. A weak "no" can't clear a
  held state, and a weak "yes" can't open, confirm or hold one.
- Usage of a failed or weak answer about an open occurrence is still charged to that occurrence
  (see [Decision fields](#14-decision-fields)). That is bookkeeping, not state.

## 3. Identity

- `occurrence_id` is `{stream_id}/{condition_id}/{incarnation}-{n}`.
  - `n` comes from a per-stream counter that only grows and is saved with the state.
  - `incarnation` is minted from the wall clock when the stream's state is created. If a saved
    state is lost or unreadable, the stream starts over under a new incarnation, so no id is
    ever issued twice.
- `decision_id` is `{occurrence_id}/{k}`, with `k = 1, 2, …` within the occurrence. `trace_id` is
  the `occurrence_id`.
- Condition ids can't contain `/`, so ids can be read from the right.
- **Occurrence generation.** Each key carries a generation number, and every judge request is
  tagged with the generation of the key it was sent for. A new generation is minted when:
  - an occurrence opens;
  - the application resolves it or marks it a false alarm;
  - its condition is changed.

  Natural closes (a confirm window giving up, clearing by "no" answers, a rejected verification,
  an unverified candidate, the end of rearm) keep the generation; the
  [evidence floor](#6-evidence-floor) covers late answers there instead.

## 4. Order of processing

For an answer, in this order:

1. sync the condition's version ([Condition changes](#7-condition-changes));
2. apply every deadline that has passed ([Time](#1-time));
3. check staleness ([Stale results](#5-stale-results));
4. charge the usage to the open occurrence, if any;
5. if failed or weak, stop here (traced);
6. act on the phase.

For a timer: sync, apply passed deadlines, then reschedule an early timer, ignore one whose
deadline no longer exists, or start a due recheck. For an application control (resolve, false
alarm): find the open occurrence, sync, apply passed deadlines, then act only if it is still open.

## 5. Stale results

An answer is **stale** when, checked in this order:

1. **epoch**: it was judged on frames from an earlier stream epoch (before a reconnect or a
   restart). The exception: a verification of an event's original interval is checked against
   the epoch of the candidate's own evidence.
2. **version**: its condition has changed since the request was sent, or no longer exists.
3. **generation**: the key's occurrence generation has changed ([Identity](#3-identity)).
4. **purpose**: it is a verify answer and the key is no longer verifying.
5. **evidence**: it is a detect answer whose evidence ended at or before the key's
   [evidence floor](#6-evidence-floor).

A stale answer changes nothing. It is traced as `stale_result` and counted in coverage as
`stale`. In particular, an answer to a request sent before an outage, a condition change or a
resolve can never act on what came after it.

## 6. Evidence floor

When an occurrence closes because the evidence says it is over, the key records an evidence floor
(it only rises, and survives while the key exists):

| Close | Floor |
|---|---|
| a state cleared by `clear_after_no` confident "no" answers | the evidence end of the clearing "no" |
| an event's rearm ends | the end of the rearm period |
| a verification rejected (a confident "no") | the end of the verified interval (for `verify="fresh"`, the evidence end of the rejecting answer) |

A detect answer whose evidence ended at or before the floor is stale. A request that was in
flight when the occurrence closed can't reopen it; newer evidence always passes the floor.

## 7. Condition changes

`Monitor.add()` with an existing id replaces the condition and raises its version.

- **Updated**: every key of the condition closes and gets a new generation. A held state emits
  `cleared` with cause `condition_changed` (subject to [Emission rules](#13-emission-rules)).
  Answers to requests sent under the old version are stale.
- **Removed** (`Monitor.remove()`): its keys, its recorded version and its snoozes are deleted,
  with no decision.

Before handling any input that names a condition, the rules compare the recorded version with the
current condition and apply the change first if they differ.

## 8. Defaults that shape a decision

The `Condition` fields used below, with their defaults: `min_confidence=0.6`, `confirm=1`,
`confirm_window_s=20`, `verify="auto"`, `fast_path_confidence=None`, `candidate_deadline_s=30`,
`on_unverified="uncertain"`, `rearm_s=120` (events), `recheck_every_s=60`, `clear_after_no=3`,
`max_open_s=None` (states), `cooldown_s=0`, `emit_cleared=False`. Each is explained in
[conditions.md](conditions.md).

## 9. Confirm window

`confirm` is the number of confident "yes" answers needed; the first one counts. With
`confirm=1` (the default) the first confident "yes" confirms the occurrence in the same step.

With `confirm > 1`, the counted "yes" answers form a window. **The window is measured on evidence
time**: the evidence ends of every counted "yes" must lie within `confirm_window_s` of each other.
For a new confident "yes" with evidence end `e`:

| Case | Outcome |
|---|---|
| it fits within `confirm_window_s` of the counted ones | counted |
| it doesn't fit, and `e` is newer than the window | the window restarts at this "yes" (count 1), same occurrence |
| it doesn't fit, and `e` is older than the window | ignored |

When the count reaches `confirm` the occurrence is confirmed in that step. An unfinished window
gives up, by arrival time, at `window start + confirm_window_s + candidate_deadline_s` (the last
moment a "yes" from inside the window could still arrive). The key then returns to idle with no
decision and no evidence floor.

Because the window is measured on evidence time, two answers about frames a second apart count
together even if the second one arrived much later (for example after a retry).

## 10. States

A state (`kind="state"`) is something that stays true for a while: "the garage door is open".

**Confirming.** A confident "yes" while idle opens an occurrence and counts toward `confirm`. A
confident "no" whose evidence is not older than the window's start is a flicker: the key returns
to idle with no decision.

**Confirmed.** `verify="auto"` means no verification for states. When the occurrence is
confirmed:

- with `verify="off"`/`"auto"`, or when the confirming "yes" has confidence at or above
  `fast_path_confidence`, it is **accepted**;
- with `verify="fresh"`, it moves to `verifying` and one verification on a new frame is requested,
  due within `candidate_deadline_s`. A verify "yes" accepts it, a verify "no" closes it (with an
  evidence floor) and a verify "unknown" or the deadline ends it as
  [unverified](#12-unverified-candidates). Detect answers meanwhile are absorbed.

**Accepted: holding.** The key moves to `holding` and emits `met` (subject to
[Emission rules](#13-emission-rules)). While holding:

- **Hold.** A confident "yes" resets the count of "no" answers.
- **Recheck.** Every `recheck_every_s` the runtime looks again at a fresh frame. A state is also
  looked at by its trigger (by default a change trigger and a heartbeat every 300 s).
- **Clear on evidence only.** `clear_after_no` confident "no" answers with no confident "yes"
  between them clear the state: `cleared` (cause `no`) is emitted when `emit_cleared=True` and the
  occurrence's `met` was emitted, the evidence floor is set, and the key returns to idle.
  Failures, weak answers and "unknown" never count toward clearing: an outage can't clear a
  state.
- **Expired.** With `max_open_s` set, a state still held `max_open_s` after it was accepted emits
  `expired` (cause `max_open`) once, if its `met` was emitted. **It stays held**: it can still
  clear, and no new `met` is emitted for it.

## 11. Events

An event (`kind="event"`) happens at a moment: "a person falls". A later frame shows only its
aftermath, so events are judged and verified on the moment's own evidence.

**Candidate.** A confident "yes" while idle opens an occurrence. With `confirm > 1` it becomes a
candidate until the [confirm window](#9-confirm-window) fills. A confident "no" while a candidate
or verifying is **ignored**: a later frame without the event is not evidence that it didn't
happen.

**Verification on the original interval.** `verify="auto"` means `"original"` for events. When
the occurrence is confirmed, one verification is requested on **the same interval the counted
"yes" answers covered**, from the epoch of the first of them. The runtime re-judges the exact
frames of the request that produced that interval when it still holds them, otherwise up to the
condition's `clip_frames` (4 by default) frames held in its pre-roll inside the interval. The verifier is asked to be strict. Then:

- a verify "yes" accepts the event;
- a verify "no" rejects it: the key returns to idle with the evidence floor at the interval's end;
- a verify "unknown", or no answer by `candidate_deadline_s` after confirmation, ends it as
  [unverified](#12-unverified-candidates).

If the interval's frames have left the pre-roll, the verification can't run (health
`verify_deferred`) and the deadline ends it as unverified. Failed and weak verify answers change
nothing; the runtime asks again about once a second until the deadline. With `verify="off"`, or a
confirming "yes" at or above `fast_path_confidence`, the event is accepted without verification.
`verify="fresh"` is not allowed for events.

**Reported and rearm.** An accepted event emits `met` and stays `reported` for `rearm_s`, measured
on arrival time. Repeats meanwhile are absorbed and do **not** extend the rearm. Events never
emit `cleared`. When rearm ends, the key returns to idle with the evidence floor at the rearm's
end, so only evidence newer than that can open the next occurrence.

## 12. Unverified candidates

A candidate closes without a verified answer when the verify deadline passes (cause `timeout`),
the verifier answers "unknown" (cause `unverified`), or the process restarts during an event
verification (cause `restart`; see [Restart and reconnect](#16-restart-and-reconnect)).

Only a candidate supported by at least one confident "yes" is reported:

- with `on_unverified="uncertain"` (the default) it emits `uncertain`, under the candidate's own
  `occurrence_id`, with the confidence of its latest counted "yes";
- with `on_unverified="health"` it emits health `unverified` instead, and no decision.

The key returns to idle with no new generation and no evidence floor.

### Absorbed entries

While an event candidate verifies, a confident detect "yes" is absorbed (`awaiting_verification`).
If it is about evidence newer than the candidate's own interval, the key remembers it. When the
candidate then closes without `met` (verify "no", verify "unknown", or the deadline), the reducer
emits one `RequestRelook` effect, so that second entry gets a decision of its own, and raises the
evidence floor to the end of the candidate's interval so the look cannot reopen the closed entry.
The effect carries `observed_at` (the evidence end of the absorbed yes) and `after` (the closed
candidate's evidence end). At most one look is asked for per closed candidate; the runtime queues it
as an ordinary event detect (trigger label `relook`) under the stream's budget, anchored at
`observed_at` like a poke's `observed_at`, so its clip covers the absorbed entry however long the
verification took, and starting after `after`, so it does not show the rejected candidate again.
Recovery is guaranteed only while the verification completes within the pre-roll (8 s by default):
if `observed_at` has left the pre-roll, at the time the effect is handled or while the look waits,
no look is made (the newest frames cannot show the entry), and the trace record `relook` has
`anchored: false`. Likewise if the look is dropped (budget, deadline, shutdown), fails, or is stale
(its evidence ends at the floor, as on a stalled stream). In every such case the runtime records a
`relook_dropped` blind interval over `after`..`observed_at` with the reason in `detail`; it never
emits a decision for that.

A verified candidate, a close by an application control, state conditions, and a restart ask for no
look. On a restart the absorbed look's frames died with the process, so the candidate ends
`uncertain` (cause `restart`) and the new epoch's triggers carry on.

## 13. Emission rules

A decision is emitted only when no rule suppresses it. A suppressed decision is traced
(`suppressed`), never delivered later, except the deferred `met` below.

| Decision | Emitted only if |
|---|---|
| `met` | the key is not snoozed, and not within `cooldown_s` of this key's last `met` |
| `uncertain` | not snoozed, and not within `cooldown_s` of the later of the last `met` and the last `uncertain` |
| `cleared` | the condition is a state, `emit_cleared=True`, and this occurrence's `met` was emitted |
| `expired` | this occurrence's `met` was emitted |

- **Cooldown** for `met` counts from the last `met` only, so an `uncertain` never holds back a
  real `met`. The last `met` and `uncertain` times survive the key returning to idle, so cooldown
  spans occurrences.
- **Snooze** (`Monitor.snooze(condition_id, until_wall, subject=None)`) mutes `met` and
  `uncertain` until `until_wall`; judging continues. `subject=None` covers every subject. A new
  snooze for the same condition and subject replaces the old one; a time in the past cancels it.
- **Deferred delivery while holding.** When a state's `met` is suppressed (snooze or cooldown),
  the state is still held and the `met` is remembered. It is delivered, with cause `deferred`, at
  the first confident detect "yes" while the state is still held and no rule suppresses it. When
  a snooze ends, the runtime looks again at every held state it deferred a `met` for. A deferred
  `met` is dropped if the state clears, is resolved or marked a false alarm, or expires first.
  Snoozes survive a restart, but the look scheduled for a snooze's end does not: after a restart
  a deferred `met` is delivered at the state's next recheck or heartbeat instead.
- A suppressed event `met` stays silent for that occurrence; the event is still reported and
  rearms as usual.

## 14. Decision fields

| Field | `met` | `met` (deferred) | `uncertain` | `cleared` | `expired` |
|---|---|---|---|---|---|
| `cause` | `null` | `"deferred"` | `"timeout"`, `"unverified"` or `"restart"` | `"no"`, `"resolve"`, `"false_alarm"` or `"condition_changed"` | `"max_open"` |
| `reason` | the judge's reason (the verifier's for a verified event) | the reason of the "yes" that delivered it | a fixed sentence per cause | `no`: the clearing answer's reason; otherwise a fixed sentence | `"still true after max_open_s"` |
| `t_observed` | the window start: the evidence end of the earliest counted "yes" | the window start | the window start | `no`: the clearing answer's evidence end; otherwise the decision time | the decision time |
| `confidence` | of the deciding "yes" | of the delivering "yes" | of the latest counted "yes" | `no`: of that answer; otherwise 1.0 | of the `met` the agent received |

- `t_decided` is the wall-clock time of the decision.
- `cost` and `request_ids` cover everything charged to the occurrence since its previous decision:
  every answer about it that was not stale, failed and weak ones included, and the opening one.
  `request_ids` keeps the 16 most recent; `cost` sums them all.
- `evidence` (up to 8 frame references) and `keyframe` (one JPEG of at most 40 KB) are filled by
  the runtime from those requests.
- Agents should branch on `type` and `cause`. `reason` is for people.
- A `scene_return` cause is reserved for a future trigger and is not produced in this release.

## 15. Application controls

- `Monitor.resolve(occurrence_id)`: the application says the occurrence is over. A held state
  emits `cleared` (cause `resolve`, subject to the emission rules); any open occurrence closes and
  gets a new generation; a pending verification is cancelled. An event's rearm ends early, with
  no decision.
- `Monitor.feedback(occurrence_id, "false_alarm", note=...)`: as resolve (cause `false_alarm`),
  and the note is kept as a counter-example the judge is shown for that condition (the last 3
  are kept). It is kept even when the occurrence has already closed.
- `Monitor.pause(condition_id)` / `resume(condition_id)`: stop and restart judging a condition.
  Verifications already requested still finish. Every stream records a `paused` blind interval,
  and on resume the condition gets one look.
- A control for an occurrence that is not open is ignored (traced).

## 16. Restart and reconnect

**Restart** (a new process over the same store). The stream's saved state is loaded and a new
epoch starts before anything else:

- deadlines that passed while the process was down are applied first (confirm windows give up,
  verifications time out, rearm ends, states expire);
- a held state **stays held**, emits no new `met`, and is rechecked at once;
- an event candidate or event verification ends as unverified with cause `restart`, keeping its
  `occurrence_id`;
- a state's fresh verification is requested again for the new epoch, with the same deadline;
- every remaining deadline is rescheduled;
- every answer about the old epoch is stale;
- decisions committed before the restart and not acknowledged are delivered again, first, with
  the same `decision_id` (see [delivery.md](delivery.md)).

**Reconnect** (a camera comes back, or `set_link(True)` on a push source) is the same, except that
event candidates and verifications **stay**: the verify answer is checked against the candidate's
own epoch.

## 17. Capacity

Each stream keeps at most 64 keys. A key is created only by a confident "yes" while idle;
"no", weak, "unknown" and failed answers about a key that doesn't exist create nothing. A key is
**pinned** while it is open, or idle and still inside its cooldown.

When a new key is needed and the stream is full, the least recently touched unpinned key is
evicted (it loses its generation and evidence floor). When every key is pinned, the answer is
refused: health `capacity`, coverage fate `dropped_capacity`, and no state change.

## 18. Coverage fates

Every judged answer ends up in exactly one coverage fate, reported by `Monitor.coverage()`:

| Fate | Meaning |
|---|---|
| `judged` | a confident "yes" or "no" that moved, held, or was absorbed by the state |
| `judged_weak` | below `min_confidence`, or a model "unknown" |
| `failed` | timeout, outage, refusal, malformed or missing reply |
| `stale` | rejected by [Stale results](#5-stale-results) |
| `dropped_capacity` | refused by [Capacity](#17-capacity) |

The runtime adds the fates of looks that never reached a judge: `coalesced` (merged into a look
already waiting), `dropped_budget` (waited past its deadline), `dropped_for_verification`
(waited past its deadline while its attempts were kept for a waiting verification),
`dropped_quality` (the frame was vetoed as corrupt or blurry). See [concepts.md](concepts.md#coverage-and-blind-intervals).
