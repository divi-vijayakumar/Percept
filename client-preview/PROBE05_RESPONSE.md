# Response to Probe 05

Wheel under test: `percept_vision-0.8.1-py3-none-any.whl` (contract 1.9.0, up from the 0.8.0 build your
probe ran against). Every incident below was independently reproduced against our own code before we
scoped a fix — no root cause here is disputed, and two of your findings (SDK-001, SDK-002) shipped as a
combined contract change because they share one join.

Your validator's five evidence-link join rules (unique resolution, zero-match, multi-match, frame
resolvability, legacy tolerance) are good enough that we didn't write our own version — we adopted them
as our own regression tests. They live at
`packages/percept-vision/tests/test_inc1_evidence_links.py`, and the full suite runs them on every
change, not just this one.

**How to read this document:** each incident gets what we changed (the new API/field name + the exact
test that proves it), and — where the fix is only half of the story — what stays your action, because
`percept_audit_walkthrough` (your probe repo, your pack, your `plans.py`) is not something we can patch
from here.

---

## Incident-by-incident

| ID | Your finding | Status | Fixed in | Your action |
|---|---|---|---|---|
| **P05-SDK-001** | Approval fires correctly, but `evidence_seq=null` / no `judgment.event` on the transition path | **Fixed** | `evidence_origin`, real `JudgmentEvent` on transitions | none |
| **P05-SDK-002** | Two `judgment.event` rows share `dispatch_seq=1`; join ambiguous | **Fixed** | `judgment_event_id()` derived join key | none |
| **P05-SDK-003** | Blank screen admits 3×, spends 3 judge calls | **Fixed** | `EdgeSettings.coast_enabled` + `strict_admission@1` | recommended: arm `strict_admission@1` on your blank-input admission control (§3 below) |
| **P05-SDK-004** | 0/4 live correction chains | **Fixed (SDK half)** | claim-mode parsing in claude/gemini/openrouter backends | **required**: arm `identity_policy="claim_slot"` in your revision-suite plans — your probe currently arms nothing (§2 below) |
| **P05-DOM-001** | 3/8 exception positives fire; 12/12 negatives correct | **Not an SDK defect** | — | **required**: concrete condition wording for the 5 missed families + the historical/out-of-scope split (§1 below) |
| **P05-DOM-002** | Identical `25,000 dollars` values false-fired a mismatch @1.0 confidence | **Fixed (SDK half)** | `FieldNormalizer` / `extractors/fields.py` typed comparison extractor | recommended: chain the extractor + ASK-on-unnormalizable condition wording (§4 below) |
| **P05-DOM-003** | Keyboard-submit approval missed (5/6 fired) | **Fixed** | transition path now judges the resolved temporal strip | none |

---

### P05-SDK-001 — transition fires now carry a real, resolvable judgment

**Root cause (confirmed):** the A→B transition path spent two real judge calls per tick (`from`/`to`)
but built `FireEvent(judgment=None)` and emitted zero trace records — a designed gap, not a bug in the
judging itself.

**What changed:** `_perceive_transition` now builds a real `JudgmentEvent` on every transition fire,
stamped with a new closed-vocabulary field:

```python
EvidenceOrigin = Literal["judged", "transition"]   # percept.contracts

fire.judgment.evidence_origin   # "transition" for an A->B fire, "judged" for the watch path, None = legacy
fire.judgment.evidence_seq      # non-null — dereferences to the exact source frame
```

`Incident.evidence_origin` and `Incident.evidence_seq` carry the same values off the wire (contract
1.9.0, additive — a pre-1.9 consumer that doesn't read the new fields is unaffected). A `transition`
fire is never fabricated: if the judgment's timing stamp fails to freeze, the fire still delivers but
`judgment_event_id`/`evidence_origin`/`evidence_seq` are all `None` rather than invented values (proof:
`TestBuildFailedStillFires`).

New diagnosability surface: every transition tick (fire or not) now emits a `transition.decision` trace
record — `outcome ∈ {fire, armed, rearmed, suppressed_latched, idle}`, `armed`, `fired_latch` — so a
"why didn't this fire" question is answerable from the `.ptrace` without re-running anything (proof:
`TestBWithoutANeverFires`).

**Proof:** `test_inc1_evidence_links.py::TestTransitionE2E` (non-null `evidence_seq`, `evidence_origin`,
real timing stamps present, `verify_evidence_links` resolves the fire) and `TestBuildFailedStillFires`
(a broken stamp never changes what fires, links are absent not fabricated).

---

### P05-SDK-002 — one injective join key instead of a bare, colliding sequence number

**Root cause (confirmed):** `EvidenceSequencer` is per-`(goal_id, entity_id)`, so `dispatch_seq` alone
collides across entities/goals. The disambiguators were available at `make_incident` but unused.

**What changed:** a pure, exported helper —

```python
from percept.contracts import judgment_event_id

judgment_event_id(goal_id, entity_id, dispatch_seq) -> str   # "{goal}#{entity}#{seq}", # and % escaped
```

— derivable independently at both the trace-emit site and `make_incident` from fields already in
scope. `None`/empty entity normalizes consistently (`judgment_event_id("g", None, 3) ==
judgment_event_id("g", "", 3)`). It's opaque: treat it as an equality-comparable handle, never parse it.
`Incident.judgment_event_id` and the `judgment.event` / `incident.created` trace rows all carry it, so
the join validates from the `.ptrace` alone — no live re-run needed.

**Replay honesty:** the 1.9 stages are inert to replay — a pre-1.9 trace with no `judgment.event` row
replays to a byte-identical decision digest, because `replay_cognition` reads only `gate.classified`
inputs (proof: `TestPre19ReplayDigestUnchanged`).

**Verification tool, shipped:** `percept.replay.verify_evidence_links(records) -> EvidenceLinkReport`
implements your five join rules directly — `checked`, `resolved`, `legacy_unverifiable`, and typed
`violations` (`unresolved_evidence_ref`, `ambiguous_evidence_ref`, `missing_source_frame`). It's also a
`percept doctor` finding: `EVIDENCE_LINK_BROKEN`.

**Proof:** `test_inc1_evidence_links.py` —
`TestJudgmentEventIdInjective` (11 crafted triples with `#`/`%` in every component position, zero
collisions), `TestP0TwoEntitiesResolve` (your exact P0 repro: two entities, one goal, one trace, both
fire, distinct ids, both resolve), `TestVerifyEvidenceLinks` (zero-match/multi-match/legacy counting),
`TestGoldenVectorParity` (Python/JS share one golden-vector fixture — no drift between the two
implementations).

---

### P05-SDK-003 — signal-only admission is now a first-class, reported posture

**Root cause (confirmed):** `screen_document.coast_interval_s=2.0` forces a judge call every 2s
regardless of signal, with no first-class disable, an inert `coast_required` flag, and no counter
separating coast admits from real-signal admits.

**What changed:** `EdgeSettings.coast_enabled: bool = True` (default = today's behavior, byte-for-byte)
plus a fifth shipped `ExecutionPolicy`:

```python
from percept.execution_policy import get_execution_policy
strict = get_execution_policy("strict_admission@1")   # sets Settings.edge.coast_enabled = False
```

Under `strict_admission@1` the resolver skips binding the coast floor entirely — no
`Goal.state.sense.base_fps/max_fps`, no untuned `cadence.*` — while a real content change still admits
via the tuned/motion signal. This is a **gating read**, not a precedence write: an `ExecutionPolicy`
still cannot reach a `SourceProfile`-owned field (a policy that tries raises `ConfigError`, pinned as a
regression test so this can't quietly regress later).

**Reporting:** `RuntimeStats.coast_admits` / `signal_admits` now partition admissions —
`escalations == signal_admits + coast_admits + audio_wakes` — and the `runtime_ended` coverage summary
carries the same split. If a profile declares `coast_required=True` and `strict_admission@1` overrides
it anyway, that's never silent: `WatchReceipt.warnings` carries `coast_required_overridden`
(`applied_with_warnings` status), and `percept doctor` reports the same condition as a finding. An
untuned profile with no `coast_required` gets the softer `coast_disabled` warning instead — never an
error either way.

**Measured, on our blank-screen fixture:** default policy → 3 coast admits / 3 judge calls on 5s of
static content; `strict_admission@1` → 0 coast admits / 0 judge calls on the same static content, 2
signal admits when the content actually changes.

**Proof:** `test_inc2_coast_admission.py` — `BlankScreenAdmissionTests` (the exact before/after above),
`StrictResolutionTests` + `OwnershipGateTests` (no coast bindings emitted, ownership gate pinned),
`PartitionInvariantTests` + `CoverageSummaryTests` (the admission split), `UntunedProfileStrictTests`
(warns, never errors), `BriefCoastUnaffectedTests` (a goal-authored fovea coast is untouched by this
setting — it's a different mechanism), `ExplicitPrecedenceTests` (an explicit `WatchPlan` sense floor
still wins).

**Recommendation for your admission control — see §3 below.**

---

### P05-SDK-004 — the correction channel now exists in three real backends; you still need to arm it

**Root cause (confirmed):** the claim/supersession machinery (`Goal.state.identity_policy="claim_slot"`,
revision chains, `supersedes`) was fully built and reachable, but **no shipping judge backend ever
emitted `Verdict.operation`/`value`** — only scripted test judges did. A live verdict was always
`operation="assert"`, so nothing ever mutated a claim. Separately, your probe currently arms **no**
`identity_policy` at all — even with a parsing backend, your plan as written would not have activated
claim mode.

**What changed (the SDK half — done, no action needed):** when, and only when, the armed goal resolves
`identity_policy="claim_slot"`, the judge prompt requests and the parser reads the mutation channel —
`operation ∈ {assert, replace, retract, confirm}`, `value`, `normalized_value` — in `claude`, `gemini`,
and `openrouter`. This is claim-gated: a non-claim goal's prompt and parse are byte-for-byte identical
to before (proven, not asserted — see `TestPromptSnapshotByteForByte`). An unknown or missing operation
degrades to `assert` — the parser never invents a mutation. There is no diarization, so `asserted_by`
stays `None` (unrestricted authority) — we don't fabricate a speaker; a correction is applied on its
content, not an asserted role.

If you arm `claim_slot` on a backend without a mutation parser (a custom judge, for instance),
`apply_watch_plan` doesn't reject the plan — it arms and returns a receipt warning:

```python
receipt.warnings  # -> FieldDiagnostic(code="claim_mode_backend_unverified",
                  #                    path="Goal.state.identity_policy", goal_id=<your goal id>)
```

Check `receipt.warnings` before you rely on supersession forming — that's a decision your probe should
gate on today, since a warning here means the plan runs but never revises.

**What stays your action (the required plans.py change) — see §2 below.**

**Proof:** `test_inc3_claim_channel_parsing.py` (prompt-snapshot byte-for-byte proof across
anthropic/openrouter, claim-mode contract injection, parser strictness — unknown op → assert, missing op
→ assert, retract carries no value, malformed JSON takes the shared salvage path unchanged, receipt
warning present/absent correctly) and `test_inc3_live_revision_recipe.py` (a live-parser fake — reply
JSON, not a pre-built `Verdict` — through a real `run()`, proving the parse-to-revision-chain path end to
end, the way your probe will actually exercise it).

---

### P05-DOM-003 — the transition path now judges the same temporal strip the watch path does

**Root cause (confirmed):** `_perceive_transition` judged bare single frames — no strip, no window, no
extractor context. A status flip that lands between 4fps samples, or renders on a frame the sampler
just missed, is invisible to a single-frame judge even though the flip is unambiguous across two or
three frames.

**What changed:** the transition path now judges over the SAME resolved temporal strip
(`self._strip_frames`) the watch path already uses, honoring the compiled `plan_mode`. It also threads
extractor context into both the `from` and `to` judge calls exactly like the watch path, and emits
`transition.decision` on every tick (fire or not) with `strip_len` / `enriched` fields so a miss is now
diagnosable from the trace instead of invisible.

We pinned the regression as a strip-vs-single A/B **on the identical frames and identical judge**:

- `plan_mode="single"` → the judge sees only the anchor frame → the keyboard-submit sequence
  (`pending → gap → before → after`) MISSES, 0 fires (this reproduces your finding exactly).
- `plan_mode="strip"` → the judge sees the 3-frame clip → FIRES exactly once, confirmed by the latest
  frame given the earlier one in the strip.

10/10 of our negative sequences (armed-only, flip-before-arming, half-a-flip, pure noise, etc.) stay
quiet under the new strip path — the fix doesn't trade false negatives for false positives.

**Proof:** `test_inc4_transition_strip.py` — `TestKeyboardSubmitRegression` (the single→strip A/B above),
`TestNegativesQuiet` (10/10), `TestStripFireEvidenceLinks` (the strip fire still satisfies every SDK-001/
SDK-002 evidence rule — `verify_evidence_links` resolves it cleanly), `TestExtractorThreadedIntoBothJudgeCalls`
(extractor facts reach both the `from` and `to` judge calls and the trace, once per tick).

---

## §1. P05-DOM-001 — exception recall: concrete condition wording (your action)

This is not an SDK defect. Your pack and probe conditions live in `percept_audit_walkthrough`
(`plans.py`), and the five missed families need audit-domain *implication* your condition text doesn't
currently encode. Below is concrete replacement wording for each family — adapt the entity/field names
to your actual schema, but keep the causal structure (what evidence, what the absence of that evidence
means).

**1. Bypassed review.**

```
condition: "the record shows an approval status without a corresponding review event ever having
            occurred for it — i.e. the required review step was skipped entirely, not just late"
```

**2. Control not performed.**

```
condition: "a control that this record's workflow requires (e.g. a reconciliation, a sign-off, a
            secondary check) has no evidence of having been executed at all for this record"
```

**3. Late review → timeliness exception.**

```
condition: "a review event exists for this record, but its timestamp is after the record's approval
            timestamp, or after the SLA/deadline window defined for this workflow — this is a
            TIMELINESS exception, distinct from a bypassed review where no review exists at all"
```

**4. Same preparer + approver → segregation-of-duties exception.**

```
condition: "the 'prepared by' field and the 'approved by' field on this record name the same person —
            this is a SEGREGATION-OF-DUTIES exception regardless of whether the amount or content of
            the record is otherwise correct"
```

**5. System-failure spreadsheet workaround → unapproved manual workaround.**

```
condition: "this record's evidence trail shows the transaction was recorded or reconciled via a manual
            artifact (a spreadsheet, an email, a handwritten note) INSTEAD OF the system-of-record path,
            with no corresponding approved-exception/workaround authorization on file — flag this
            regardless of whether the manual figures match the eventual system entry"
```

Each of these should be its own goal/condition, not a single omnibus "is this compliant" condition —
your 12/12 clean negatives suggest your negative-family coverage is solid; the gap is specifically that
these five *positive* families were never independently encoded as their own conditions, so the judge
had nothing to check them against.

**The historical-vs-out-of-scope split.** Your current text appears to treat any past-tense statement
("last month...") as an out-of-scope example, which over-excludes real, in-scope occurrences that happen
to be phrased or discussed in the past tense. Split this into two distinct conditions instead of one:

```
# in-scope: a real occurrence dated within the audit period, described in any tense
condition: "this record's own approval/review/preparer timestamps fall within the audit period, and
            exhibit the exception pattern — regardless of whether the record or a person describes it
            using past tense ('last month we...')"

# out-of-scope: an illustrative/hypothetical mention, not a dated occurrence of THIS record
condition: "a person is describing a general pattern, a hypothetical, or a training example — not
            asserting that THIS record's specific dated fields exhibit the exception — e.g. 'we
            sometimes see cases like...' or 'for example, if someone...'"
```

The distinguishing signal is not tense, it's whether the statement is bound to this record's own dated
fields (in-scope, however phrased) versus describing a class of situation without asserting it happened
to this record on a specific date (out-of-scope). Encoding both explicitly, rather than a single
tense-keyed exclusion, is what recovers the "last month" occurrences your current wording drops.

---

## §2. P05-SDK-004 — arm `identity_policy="claim_slot"` in your revision suite (your action, required)

Your probe currently arms no `identity_policy` at all, so even with the parsing backends now live, your
revision-chain plans will not activate claim mode. The public recipe is one explicit key:

```python
from percept.contracts.watchplan import WatchPlan

plan = WatchPlan(
    condition="a buyer states the approved budget",
    explicit={"Goal.state.identity_policy": "claim_slot"},
)
```

If your `plans.py` (or equivalent) allow-lists which `explicit` fields a probe plan may set — which your
report's structure suggests it does, since your current plans only pass `modality`/`audio_s` — you'll
need to add this key to that allow-list. Illustrative shape (adapt to your actual file):

```python
# percept_audit_walkthrough/probe05/plans.py  (illustrative — match your actual structure)

ALLOWED_EXPLICIT_FIELDS = {
    "Goal.state.perception.modality",
    "Goal.state.perception.audio_s",
    "Goal.state.identity_policy",          # <-- add this line
}

def build_revision_plan(condition: str) -> WatchPlan:
    return WatchPlan(
        condition=condition,
        explicit={
            "Goal.state.perception.modality": "fused",
            "Goal.state.perception.audio_s": 4.0,
            "Goal.state.identity_policy": "claim_slot",   # <-- add this line
        },
    )
```

After arming, check `receipt.warnings` for `claim_mode_backend_unverified` before trusting supersession
in a given run — see the P05-SDK-004 section above for what that warning means and when it fires.

---

## §3. P05-SDK-003 — recommended: `strict_admission@1` for your blank-input admission control

Your blank-screen scenario is exactly the case `strict_admission@1` exists for: an admission control
where you never want to pay a judge call for "nothing changed", and you want a real content change to
still admit via the tuned signal. Recommended usage:

```python
from percept import Percept
from percept.execution_policy import get_execution_policy
from percept.source_profiles_default import SCREEN_DOCUMENT
from percept.watchplan_apply import apply_watch_plan
from percept.contracts.watchplan import WatchPlan

core = Percept.configure(vision="gemini", mode="live", source_profile=SCREEN_DOCUMENT)
plan = WatchPlan(condition="a red banner appears on the page")

receipt = apply_watch_plan(
    core, plan,
    source_profile=SCREEN_DOCUMENT,
    execution_policy=get_execution_policy("strict_admission@1"),
)
```

Watch for `coast_required_overridden` in `receipt.warnings` if the profile you're using declares
`coast_required=True` — it means `strict_admission@1` removed a floor the profile author considered
non-optional; that's an informed choice you're making, not a silent one, but confirm it's the one you
want for that source class.

---

## §4. P05-DOM-002 — recommended: chain the fields extractor + ASK-on-unnormalizable wording

The SDK half of this incident: `percept.extractors.fields.FieldNormalizer` turns OCR/ASR text into typed
`{field, raw, normalized, kind}` facts BEFORE the judge sees them, so a condition can compare
`normalized` values instead of raw surface text. Your exact false-fire case — `"25,000 dollars"` (spoken)
vs `"threshold: 25,000 dollars"` (visible) — now both normalize to `"USD 25000.00"` and read as equal
despite the differing surface form (label present/absent). Worked example (also in
`client-preview/ONBOARDING.md` §5):

```python
from percept.extractors.fields import chain_ocr_then_fields
from percept.extractors.ocr import TesseractOCR   # or your own OCR extractor

extractor = chain_ocr_then_fields(TesseractOCR())   # one extractor= for Percept.create

condition = (
    "compare context.extractor.facts.fields[0].normalized across the spoken and visible reads; "
    "fire only on a confirmed VALUE mismatch, never on differing raw text; "
    "ASK if either side's normalized value is missing"
)
```

The ASK rule has to be **in the condition text**, stated explicitly — the extractor gives the judge a
typed fact to compare, it doesn't itself decide what "cannot confirm" should do. `normalized=None` on
either side means the extractor recognized a currency/date/number *shape* but genuinely couldn't resolve
it unambiguously (e.g. an ambiguous currency abbreviation, an `MM/DD` vs `DD/MM` date) — that's a
"cannot confirm equal or different" case, and your condition should route it to ASK, never to a confident
mismatch. This mirrors the failure mode your report described: an unstructured compare treated a surface-
form difference as a semantic one, at confidence 1.0, with no ASK path at all.

---

## Pack-authoring guidance: pinning paraphrase families as contract tests

Your report's 12/12 clean negatives tells us your negative-family coverage is already solid; the pattern
worth generalizing to your positive families (the five in §1, and any future ones) is what we used to
pin the DOM-003 fix: **one fixture module per family, scripted frame-token sequences, asserted as a
single quiet/loud count** — see `test_inc4_transition_strip.py::TestNegativesQuiet` for the shape:

```python
_NEGATIVES = [
    ["gap"],                               # 1. nothing at all
    ["pending"],                           # 2. armed only, no flip
    ["before", "after"],                   # 4. flip but the precondition never witnessed
    # ... one entry per DISTINCT way the condition should NOT fire
]

def test_negatives_stay_quiet(self):
    quiet = 0
    for seq in _NEGATIVES:
        fires = run(seq)
        self.assertEqual(len(fires), 0, f"negative should stay quiet: {seq}")
        quiet += 1
    self.assertEqual(quiet, len(_NEGATIVES))   # N/N in the assertion, not just in a report
```

Apply the identical shape to a **positive** family — a list of scripted variants that all mean the same
underlying exception (different phrasing, different field ordering, different amount of surrounding
noise text) asserted to all fire, e.g.:

```python
_BYPASSED_REVIEW_PARAPHRASES = [
    "approved with no review on file",
    "review step was skipped before sign-off",
    "no reviewer ever recorded for this approval",
    # ... every phrasing your fixture corpus actually contains for this family
]

def test_bypassed_review_paraphrases_all_fire(self):
    fired = 0
    for text in _BYPASSED_REVIEW_PARAPHRASES:
        fires = run_condition(BYPASSED_REVIEW_CONDITION, text)
        self.assertEqual(len(fires), 1, f"paraphrase should fire: {text!r}")
        fired += 1
    self.assertEqual(fired, len(_BYPASSED_REVIEW_PARAPHRASES))
```

The pattern that matters: **the assertion count is derived from the list length, not hand-copied** — a
future paraphrase you add to the fixture automatically raises the bar, and a regression that silently
drops one variant fails immediately instead of quietly shrinking your reported recall.

---

## What to send back

- A rerun of Probe 05 against `percept_vision-0.8.1-py3-none-any.whl` — we expect SDK-001/002/003/004
  and DOM-003 clean, and DOM-001/DOM-002 substantially improved once the pack wording above lands on
  your side.
- Your `.ptrace` files if anything looks off — the evidence-link fields make a "why didn't this
  resolve" question answerable from the trace alone now, and we'll trace it to the field and layer that
  produced it.
