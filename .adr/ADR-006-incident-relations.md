# ADR-006: Incident relations — percept models the world, not its own findings

- **Status:** Proposed (design only — this ADR ships no code and changes no contract)
- **Depends on:** ADR-001 (the Gate/judgment contract — `Verdict -> GateDecision -> JudgmentEvent`,
  unchanged by this ADR); the frozen wire contract in
  `packages/percept-vision/percept/contracts/__init__.py` (`CONTRACT_VERSION = "1.10.0"`, `:33` —
  1.3 was the bump that shipped `Incident.entity_id`, see §7; 1.4 additively added `SubjectRef`; 1.5
  additively added the `Verdict` claim-mode channel + `ConcernAuthority`; 1.6 additively enriched the
  judgment envelope — `JudgmentEvent`/`Incident` gained `reason`/`watchplan_id`/`roi`/`modality`/`provenance`;
  1.7 additively gave `SubjectRef` a third `'provisional'` resolution state and added `Incident.evidence_seq`;
  1.8 additively shipped the ADR-008 configuration-provenance DTOs (`WatchPlan`/`SettingOrigin`/
  `ResolvedWatch`/`WatchReceipt`/`FieldDiagnostic`) in `percept.contracts.watchplan`;
  1.9 additively added the evidence-link correlation — `judgment_event_id()`/`EvidenceOrigin`,
  `JudgmentEvent.evidence_origin`, `Incident.judgment_event_id`/`Incident.evidence_origin`).
- **Related:** ADR-003 (Evidence artifacts — the proposed `evidence_ref` is an opaque handle that MAY name a
  bundle; this ADR is the analogous "pin the meaning before the capability ships" move, applied to
  the *relation* between findings rather than the evidence behind one); ADR-004 (Public delivery
  flow & replay — §1 pins the at-most-once-per-episode rising-edge guarantee this ADR shows is
  insufficient for claim-bearing subjects, and its Open Question 4 explicitly leaves the
  `Incident`/`DeliveryAttempt` chain unresolved; this ADR does not resolve it either).
- **Feeds:** the still-open **"public object" design decision**. This ADR deliberately stops at
  options and trade-offs. Anything that would pre-empt that decision is marked deferred.
- **Source (read for this ADR):** `percept/contracts/__init__.py` (`Observation` `:94-101`,
  `Subject` `:262-267`, `JudgmentEvent` `:514`, `IncidentSnippet` `:809`, `Incident` `:888`,
  `WatchAction`/`WatchEvent` `:1129-1156`), `percept/gate.py` (key shape `:13-16`, `GATE_REASONS`
  `:111-121`, `Gate.decide` `:488-632`, `reset_entity` `:676`), `percept/agent.py`
  (`perceive_judged` `:1023-1248`, `detect`+`ingest` `:1090-1091`, `base_key` `:1135`),
  `percept/entity.py` (`EntityGraph.ingest` `:104-138`, `_predict_match` `:140-153`, `_mint`
  `:179-191`), `percept/identity.py` (`ContinuityIdentity.match` `:47-55`),
  `percept/incidents.py` (`make_incident` `:21-53`), `percept/episodes.py` (`Episode` `:24-31`),
  `percept/runtime/sinks.py` (`incident_payload` `:181-191`),
  `percept/contracts/schema/Incident.schema.json`.
  (Every line cite in this ADR was re-derived against the tree at contract 1.3.0.)

## Context

Independent domain probes against percept 0.7.0 returned not-viable.
The failures were not perception failures — the judges scored recall 1.00 on documents
and handled retractions, counterfactuals and speaker attribution correctly. Four separate critical
findings emerged, and cross-probe analysis proposed that all four are one gap.
Reading the code confirms it:

**Percept models the WORLD well and does not model ITS OWN FINDINGS at all.**

The world side is real and well-built. `EntityGraph` (`entity.py:64`) is a genuine graph over
observations: entities are minted (`_mint`, `entity.py:179`), matched across frames by a pluggable
`Identity` strategy (`identity.py`), carry provenance-and-freshness-stamped `Attribute`s with the
"assertion beats observation" rule enforced in one place (`_record_observed`, `entity.py:193-200`),
get motion models for predictive continuity (`_predict_match`, `entity.py:140`), and die observably
(`_on_entity_evicted` → `gate.reset_entity` + cache invalidation, `agent.py:1706-1711`). That is a
real model of things-in-the-world, with identity, lifetime, and correction semantics.

The findings side has none of that. An `Incident` (`contracts/__init__.py:888`) is:

```python
id: str; concern_id: str; verdict: Verdict; ts: float
label: str | None = None; snippet: IncidentSnippet | None = None
message: str = ""; label_url: str | None = None
entity_id: str | None = None          # 1.3 — a correlation handle, not a relation
```

There is no field on `Incident` that references any other `Incident`. `entity_id` shipped in
contract 1.3 and points at the *subject* the fire was gated on, never at another finding — §7.
`make_incident` (`incidents.py:21-53`) mints a fresh `uuid4`-suffixed id per fire (`:46`) and reads
nothing about prior fires; the only cross-object thing it carries through is the gated subject's
`entity_id`, taken off the fire (`fire.entity_id`, `incidents.py:43-44`). The gate's memory is
per-`(goal_id, entity_or_episode_id)` key (`gate.py:13-16`,
`agent.py:1135`) and holds one armed bit, a pending dwell start plus its last-supported stamp, and
one last-fire stamp
(`gate.py:262-267`) — state *about whether to fire*, discarded into a boolean, never a durable edge
*between* the things it fired. So percept emits an unordered bag of point records and asks the
consumer to reconstruct the relations itself.

Each of the four probe findings is one missing relation. They are set out below as four problems,
because — critically — **they are not one fix**. The falsifiable prediction on the
settle/supersede pair (§2) is the clearest evidence of that, and this ADR takes it seriously rather
than collapsing the four into a single tidy feature.

### The generative failure: `watch()` was designed for state, and speech carries claims

ADR-004 §1 pins the delivery guarantee precisely: *at most one edge-consuming outward event per
`(goal, entity)` per continuous true-episode*. That is exactly right for a **state-bearing**
subject. "The door is open" becomes true, fires once, and stops being true when the door shuts —
`Gate` re-arms on a confident-`false` (`gate.py:569`).

Speech is not state-bearing. It is **claim-bearing**: a speaker asserts something, and may retract
or correct it later. "The invoice total is $4,200" does not become false the way a door closes; it
gets *superseded* by "sorry, $4,800". Percept's gate has no representation of that, because a
retraction is not a confident-`false` on the original condition — it is a *new claim about an old
claim*. Nothing in `classify()` (`gate.py:220-236`) or `Gate.decide` (`gate.py:488`) can see it,
because both take only `(verdict, freshness, threshold, fresh_min)` and a key; neither takes, or
could take, "the finding this one is about."

## Decision

This ADR decides **what the gap is, in what order the four sub-problems must be understood, and
which contract moves are additive versus breaking**. It deliberately does not decide the shape of
the public object (see Open Questions). Concretely, it pins four things:

### 1. There is no incident graph, and that is the gap — name it, do not paper over it

The named absence: percept has `EntityGraph` and has nothing of the same kind over `Incident`.
The four findings below are its four visible symptoms. Any future work that fixes one of them in
isolation, without acknowledging the other three, will produce four incompatible partial
mechanisms — which is the specific outcome this ADR exists to prevent. This mirrors ADR-003's
posture: pin the *meaning* before the capability ships, so the capability and the contract grow
together instead of the capability routing around the contract.

### 2. Settle window and supersede edge are two different fixes, not one

This is the substantive technical claim of the ADR, and it is falsifiable.

**Observed (probe):** three planted retractions all stood as findings at the end of the call. A
pure rising edge fired on the original claim; the correction either never fired, or fired as an
unrelated peer.

**Mechanism, in code.** Suppose the correction *would* fire. `Gate.decide` reaches the refractory
check only after the edge logic (`gate.py:594-600`):

```python
if effective_refractory is not None and effective_refractory > 0 and now is not None:
    last_fire = None if state is None else state.last_fire
    if last_fire is not None and (now - last_fire) < effective_refractory:
        self._state(key).armed = False  # already-fired; a later confident-false re-arms it
        self.clear_pending_dwell(key)
        return _d(SILENT, "refractory")
```

`refractory_s` cannot distinguish a *repeat* of a claim from a *correction* to it, because the key
is `(goal_id, entity_or_episode_id)` and carries nothing about claim content. So with
`refractory_s` on, the correction is suppressed as `"refractory"` — the guard designed to stop
duplicate noise silently eats the single most important event in the sequence. It is at least
observable (`"refractory"` is in `SUPPRESSION_REASONS`, `gate.py:143`, and invokes `on_suppress`
→ `RuntimeHealthEvent("gate_suppressed")`), but observable-suppression of a correction is not a
fix; it is a well-logged wrong answer.

**Fix A — settle window.** Delay committing a finding for a bounded window after the rising edge,
so a retraction arriving inside the window cancels the finding before it is ever emitted. Note this
is *not* `hold_s` (dwell): dwell requires the condition to stay continuously true before firing
(`gate.py` dwell path, `_dwell_since` `:263`), which is a **state** predicate. A settle window is a
*claim-revision* predicate: the claim was made, and we wait to see whether the speaker takes it
back. Their trigger conditions are different even where their timers look similar.

**Fix B — supersede edge.** When a correction arrives after the finding has been emitted, emit the
correction *linked to* the finding it replaces, so the consumer can resolve "which value is
current" without guessing.

**Why both are required — the falsifiable prediction, restated as this ADR's own claim:**
a settle window with any sane bound covers the two immediate retractions (at 21.6 s and 22.6 s in
the probe) and **cannot** cover the late one (34.0 s). Making the settle window long enough to
catch a 34-second-late retraction means every finding is withheld for >34 s, which destroys the
"proactive, fires on the edge" property that is percept's entire reason to exist. So the late
retraction is unreachable by settling *by construction*, and needs a supersede edge. Conversely, a
supersede edge alone leaves the two immediate cases emitting a wrong finding and then a correction
milliseconds later — technically recoverable, but noisy in exactly the way percept promises not to
be. **Neither fix subsumes the other. This prediction is testable against the probe transcript and
should be tested before either is built.**

### 3. An unlinked correction is strictly worse than a stale-only report

This is the sharpest of the four findings and deserves to be stated as a decision, not an
observation.

Corrections *do* fire today — as coequal, unlinked peers. Two `Incident`s come out with different
`id`s (`incidents.py:46`), the same `concern_id`, different `ts`, and nothing connecting them (a
shared `entity_id` says they are about the same *subject*, never that one revises the other).
`incident_payload` (`sinks.py:181-191`) serializes exactly that, so a webhook consumer receives two
JSON objects that read as **two facts** rather than **one fact and its correction**.

**Decision: percept treats "emitted a correction with no link to what it corrects" as a defect
class, not as partial credit.** A consumer that receives only the stale value has one wrong belief
and knows its provenance. A consumer that receives two contradictory coequal findings has one wrong
belief *plus* a false belief that both were separately observed — and no rule for choosing, except
"assume later `ts` wins", which the runtime never promised and which is wrong the moment two
genuinely distinct claims about the same subject legitimately coexist. Ordering-by-timestamp is an
assumption the consumer invents. This is the same discipline ADR-003 §3 applies to evidence
resolution (a typed failure, never a silent empty result standing in for "don't know"): never emit
a shape whose most natural reading is a fact the runtime did not establish.

### 4. `Incident` is a point; intervals are not derivable, and forward-filling is a consumer-invented assumption

`Incident.ts: float` (`contracts/__init__.py:916`) is a single instant. There is no `t_start`/
`t_end`, no duration, no "still true as of". Therefore a question of the form *"of these N minutes,
X are attributed to documented steps"* is **underivable from percept's output**, and this ADR says
so plainly rather than leaving integrators to discover it.

The tempting workaround — forward-fill each finding until the next fire on that goal — is an
assumption the *consumer* invents, not a measurement the *runtime* made. It is wrong in at least
three ways the runtime can see and the consumer cannot:

- the gate stays armed-`False` for the whole continuing-true episode (`already_fired` is the base
  contract, `gate.py:32-40` commentary), so "no further fire" means *nothing changed the edge* —
  which is indistinguishable from "the state ended and nothing new started";
- a key can be evicted by TTL/LRU (`_DEFAULT_MAX_KEYS`, `_DEFAULT_KEY_TTL_S`, `gate.py:77-78`) or
  reset on entity death (`reset_entity`, `gate.py:676`), silently ending an interval the consumer
  believes is still open;
- a degraded judge tick preserves armed/disarmed and last-fire/refractory state but clears pending
  positive-dwell and last-supported evidence (`agent.py:1995-2007,2561-2577`; ADR-004 §1,
  "fail-closed judging never fires") — either way, no supported interval is emitted for the
  stretch of *no information*, so forward-filling it as *continued truth* remains invented.

Two interval-shaped things already exist in the tree and are worth noting precisely because they
show the shape is not alien to percept — and equally that neither is on `Incident`:
`IncidentSnippet.t_start`/`t_end` (`contracts/__init__.py:831-832`) bound the **evidence media**,
not the finding; `Episode.started_at`/`last_seen` (`episodes.py:28-29`) bound a **counting
episode** inside `EpisodeLog`, which is internal runtime state that never reaches a consumer.
Whether an interval belongs on the finding, on a separate interval object, or nowhere is deferred
(§Open Questions 2).

### 5. Identity for non-physical subjects: percept's identity model is a *physical-object* model

`Observation` has no `id`:

```python
@dataclass
class Observation:
    """A single detected thing in one frame (the entity-attribute substrate)."""
    label: str
    box: Box
    confidence: float
    attrs: dict[str, str] = field(default_factory=dict)
```

(`contracts/__init__.py:94-101`.) Entities mint only from `Vision.detect()` output —
`observations = await self.vision.detect(frame, self.goals.active())` then
`self.graph.ingest(observations, frame, source, t=now)` (`agent.py:1090-1091`) — and matching is
IoU + label:

```python
for e in known:
    if e.label != obs.label:
        continue
    iou = e.box.iou(obs.box)
    if iou >= best_iou:
        best, best_iou = e, iou
```

(`ContinuityIdentity.match`, `identity.py:47-55`; the appearance and predicted-box paths,
`AppearanceIdentity` `identity.py:58` and `_predict_match` `entity.py:140-153`, are *also* purely
geometric/visual — they add a pixel signature and a motion model, not content.) `_mint`
(`entity.py:179`) assigns `f"e{counter}"` and copies `obs.label`/`obs.box`; nothing else.

**Consequence, stated exactly:** two invoices, INV-2291 and INV-2294, appearing in the same screen
region with the same detector label **are the same entity by construction**. Not by a tuning
failure — by the definition of the match function. Raising `iou_match` does not help (they overlap
fully); an appearance embedding does not help (two invoices look alike, and the differing pixels
are 8-point digits); the motion model does not help (documents on a screen do not move
ballistically).

**For documents, identity is in the CONTENT, not the geometry.** Percept's identity substrate has
no channel for content-derived identity: `Observation.attrs: dict[str, str]` is the only
content-ish field, and it is consumed as *attributes of an already-matched entity*
(`_record_observed`, `entity.py:193-200`) — it is never an input to `Identity.match`, which the
module docstring calls a frozen protocol `match(obs, known) -> id | None` (`identity.py:4-14`).
So a `ContentIdentity` cannot be written today purely as a pluggable strategy: the strategy seam
exists (`EntityGraph(identity=...)`, `entity.py:73-74`, and `_match_id` tolerates both arities,
`entity.py:170-177`), but the thing it would need to key on is not carried on the object it is
handed. That is the precise, minimal statement of the blocker — the seam is right, the payload is
missing.

Note this is *not only* an incident-graph problem: it is an entity-graph problem that **causes** an
incident-graph problem. If two invoices are one entity, then the gate key `(goal_id, entity_id)`
(`agent.py:1135`) is shared, and the second invoice's finding is suppressed as `already_fired`
before any incident-relation machinery could ever see it. **Ordering matters: no amount of relation
modelling on the findings side recovers a distinction the identity side already collapsed.**

### 6. Frozen-contract implications — what is additive, what is not

`CONTRACT_VERSION = "1.10.0"` (`contracts/__init__.py:33`); the cross-edge convention is documented
in `tests/test_contract_parity.py` — **same MAJOR == wire-compatible**, a minor bump is an additive
back-compatible field. JSON Schemas are generated from the dataclasses
(`percept/contracts/jsonschema.py`) and committed under `percept/contracts/schema/`, so any field
change is a regenerate-and-review step, not a silent one. `percept/watch.py`'s public signature and
`WATCH_VERSION = "1.0.0"` are separately frozen and **nothing in this ADR touches either**.

**Additive (minor bump, safe):**

- A new optional, defaulted field on `Incident` — e.g. a `supersedes: str | None = None` /
  `relations: list[...] = field(default_factory=list)`. **This route is already proven in-tree:**
  the implementation shipped exactly this shape as `Incident.entity_id: str | None = None`
  (`contracts/__init__.py:926`), which is what took `CONTRACT_VERSION` from `1.2.0` to `1.3.0`.
  `Incident` is a plain `@dataclass` (not frozen), all optional fields already carry defaults, and
  `incident_payload` serializes generically via `_jsonable` (`sinks.py:167-178`, called at
  `:187`) — a new field appears in the payload without touching the sink, which is why `entity_id`
  reached webhook consumers with no sink change. Existing readers that index known keys are
  unaffected.
- A new optional, defaulted field on `Observation` (e.g. a content/identity hint), for the same
  reasons — `Observation` is a plain `@dataclass` with a defaulted `attrs`.
- A **new DTO** (an incident-relation object, an interval object, a richer subject reference)
  exported alongside the existing ones and registered in `jsonschema.py`'s export list.
- Widening `Identity.match` with a further **optional keyword**. There is precedent:
  `signature=` was added as an optional kwarg and `_match_id` catches `TypeError` to keep
  two-arg custom implementations conforming (`entity.py:170-177`, `identity.py:11-14`).

**NOT additive (breaking; requires a major bump and a deliberate decision):**

- Changing the type or meaning of an existing field — e.g. re-typing `Incident.concern_id`, or
  redefining `Incident.ts` from "the instant of the fire" to "interval start". Even a
  same-type meaning change is breaking: every existing consumer's reading silently becomes wrong,
  which is worse than a type error.
- Making a currently-optional field required, or removing `label_url`/`message`.
- Changing the gate key tuple `(goal_id, entity_or_episode_id)` shape. This is not a wire contract
  but it is a *persistence* contract: `Gate.snapshot()`/`restore()` serialize keys as
  `[list(k), ...]` (`gate.py:710-861`) and `Percept.snapshot()`/`restore()` round-trip them
  (`agent.py:2031,2077`). A widened key silently invalidates every stored snapshot, and a
  restored-but-mismatched key re-opens a fire-once goal — the exact trap `_no_evict`
  (`gate.py:266`) exists to prevent (`gate.py:330-331` commentary).
- Adding a new value to `WatchAction` / `GateAction`. Both are closed `Literal`s
  (`contracts/__init__.py:1129`, `gate.py:80`) mirrored value-for-value by the JS edge; a fourth
  action is a major wire change. **A supersede relation must therefore be modelled as data on the
  event, never as a new action value** — this ADR states that constraint now, because it is the
  most likely accidental breakage in this area.
- Anything that changes `watch()`'s signature or `WATCH_VERSION`. If a future relation design
  appears to require it, that is a blocker to raise, not a change to make.

### 7. `Incident.entity_id` (SHIPPED in 1.3.0) does not preclude a richer subject reference later

**Status: shipped, not prospective.** `Incident.entity_id: str | None = None` exists today
(`contracts/__init__.py:926`), is populated by `make_incident` off the fire's own `entity_id`
(`incidents.py:43-44, :52`), and is the reason `CONTRACT_VERSION` reads `1.3.0` rather than
`1.2.0`. The rest of this section was written before it landed and is retained as the
compatibility argument that authorized it; it now reads as a record of a decision taken, not a
proposal.

Checked explicitly, as required. **It does not preclude a richer reference**, on three independent
grounds:

1. **Additive-over-additive is still additive.** `entity_id: str | None = None` is one optional
   defaulted field, and it was in fact appended last so every existing positional construction of
   `Incident(...)` kept its meaning (`contracts/__init__.py:924-926`). Adding a second optional
   defaulted field later (`subject: Subject | None = None`, or whatever the public-object decision
   names) is an independent minor bump; nothing about the first field constrains the second's
   shape. Dataclass field order is the only mechanical constraint, and appending after existing
   defaulted fields is always legal.
2. **The precedent already exists in-tree, in both directions.** `WatchEvent` already carries
   `entity_id: str | None = None` (`contracts/__init__.py:1155`) as a plain string, *and*
   `Subject` already exists in the same frozen module as the richer generic reference:
   `kind: Literal["entity","relation","expectation","event"]` + `ref: str`
   (`contracts/__init__.py:262-267`), used as `Concern.identity` (`:288`) and already exported to
   `schema/Subject.schema.json`. A plain string id and a typed subject reference coexist in this
   contract today without conflict.
3. **A plain `entity_id` is a strict subset of any richer reference.** Any future generic subject
   can define `Subject(kind="entity", ref=<that same id>)` as its migration reading, so the older
   field remains derivable and can be kept populated indefinitely for back-compat rather than
   removed. There is no lossy commitment being made.

The one real caution, now a live constraint rather than a forecast: `entity_id` inherits the
identity semantics of §5. For a physical-object domain it means what it says; for a document/claim
domain it currently means "the entity IoU+label happened to collapse this into". `entity_id` was
therefore safe to ship and **must not be documented as a stable subject identifier for non-physical
subjects** until §5 is resolved. The shipped docstring holds that line — it calls the field "a
correlation key ONLY" carrying "no claim about *who* or *which document* the subject is"
(`contracts/__init__.py:904-910`) — and that wording must not be relaxed while §5 stands open.

## Consequences

- The four probe findings become one named architectural gap with four distinguishable
  sub-problems, in a stated dependency order (§5 before §2/§3 — an identity collapse cannot be
  repaired downstream), rather than four independent bug reports that would attract four
  incompatible partial fixes.
- Percept gains a written, code-anchored answer to "can I get durations out of this?" — **no, not
  today, and forward-filling is your assumption not our measurement** (§4). A client debugging at
  2 am stops trying to derive an interval that does not exist.
- "Correction emitted as an unlinked peer" is now a named defect class (§3), so a future
  implementation cannot claim partial credit for firing the correction at all.
- A hard constraint is now written down before anyone trips it: a supersede relation must be data,
  never a fourth `GateAction`/`WatchAction` value (§6).
- This ADR authorizes **no implementation**. In particular it does not decide whether relations
  live on `Incident`, on a new object, or only in a store — that is the public-object decision.
  (`Incident.entity_id` shipped under not under this ADR; §7 only established that it
  does not foreclose the decision.)
- Nothing here weakens ADR-004 §1's delivery guarantees; §2 argues that guarantee is *correct for
  state-bearing subjects and insufficient for claim-bearing ones*, which is a scope statement about
  ADR-004, not a contradiction of it.

### Scope decisions taken for contract v1.4

These three decisions are **taken**. They are recorded here — and mirrored in the module docstrings a
reader actually opens (`percept/sourceprofile.py`, `percept/episodes.py`) and in
`docs/incidents/scope-and-persistence.md` — so that nobody reading the code can infer a stronger
promise than the one being made. They **change no behaviour**; they bound what the shipped behaviour
is allowed to be read as.

**(a) Scope is narrow. v1.4 covers two subject kinds and no others.**

1. **Screen documents with stable content keys** — the subject is identified by a key derived from
   the *content* (an invoice number, a ticket id, a file path), supplied by the integrator, not by
   pixel continuity.
2. **Conversational claim slots with explicit mutation semantics** — the subject is a named slot in a
   dialogue whose value can be *asserted*, *revised* and *retracted*, and the mutation is declared
   rather than inferred from timestamp order (§3).

**Explicitly EXCLUDED from v1.4, as scope and not as backlog:** biometric re-identification (deciding
that two observations are the same *person*), growth/trajectory prediction (inferring a future value
of a subject), and robotics control (closing a loop from a finding back onto an actuator). Percept
does not ship these, does not partially ship them, and a client must not build on the assumption that
the identity or relation machinery is a step toward them.

The **resolver interface stays open** — `SourceProfile.identity_strategy` is free text over an open
registry (`percept/sourceprofile.py`, §"An OPEN registry, never a closed enum"), so a future scheme is
added by *registering* one, never by editing this list. Openness of the seam is deliberate; it is
**not** a promise that v1.4 generalises. v1.4 does not generalise ahead of evidence: we have two
probes' worth of it, and the scope above is exactly what those two probes established.

**(b) Persistence is session-only. Episodes do not survive a restart.**

`Percept.snapshot()` (`agent.py:2010`) carries `goals`, `gate`, `entities` and `action_log`. It does
**not** carry `Percept.episodes` (the per-goal `EpisodeLog` map, `agent.py:616-623`), and
`Percept.restore()` therefore cannot rehydrate it. An episode log is **process-lifetime state**: on
restart, every counting goal restarts from zero episodes, and a subject last seen before the restart
is a *new* subject afterwards.

This **blocks any 24/7 viability claim** until a later phase gives episodes a persisted
representation. Nobody may market, document, or design against continuous multi-day operation on the
strength of the bounded-window machinery in `EpisodeLog` — boundedness is a memory story, not a
durability story.

**Critically, and stated wherever the field is:** if a `SubjectRef` carries `scope='durable'`, that
value describes **IDENTIFIER STABILITY** — the string `PO-4471` denotes the same real-world subject
next week as it does today — and **nothing else**. It is *not* a statement that episode state,
gate edges, or counts survive a process restart. The two readings are easy to conflate and the
consequences of conflating them are silent under-counting after a redeploy, so the caveat travels
with the field rather than living in a distant document.

**(c) Delivery is provisional incident + superseding revision, not a wide settle window.**

Of §2's two fixes, the shipped posture is: **emit the finding immediately** (the rising edge is
preserved — that is percept's reason to exist) as a *provisional* incident, and **supersede it when a
correction lands**. Any delay is **concern-specific** and **bounded by a declared latency SLO**
(`percept/slo.py`), never a global settle window applied to every finding. This resolves the
cross-cutting trade-off below in favour of Fix B as the default and Fix A as an opt-in, per-concern,
SLO-bounded knob — and it means Open Questions 1 and 5 are answered *for the delivery axis*: a settle
window, where used at all, is a delivery concept with a declared bound, not a gate concept and not a
default.

## Options and trade-offs — input to the "public object" design decision

Presented as options with trade-offs. **This ADR does not choose.**

### Option 1 — relations as optional fields on `Incident`

Add `supersedes: str | None`, and/or `relations: list[IncidentRelation]`, directly to `Incident`.

- **For:** smallest possible change; strictly additive (§6); flows through `incident_payload`
  automatically; a consumer reading one JSON object sees the link without a second lookup.
- **Against:** grows the one object every consumer already depends on, and bakes the relation
  vocabulary into the frozen contract with only a limited evidence base about what relations are
  needed. `supersedes` is easy; `corroborates`, `contradicts`,
  `refines`, `derived_from` are all plausible and none is validated. ADR-003's warning applies
  directly: a frozen field with an implicit narrow meaning is worse than no field.

### Option 2 — a separate relation object / incident store

A new DTO (`IncidentRelation(from_id, to_id, kind, ts)`) plus a queryable store, leaving `Incident`
untouched.

- **For:** the relation vocabulary can evolve without touching the object everyone reads; naturally
  supports many-to-many and later-arriving links (the 34 s retraction of §2 arrives long after the
  incident is emitted and delivered — a field on an already-delivered immutable-in-practice record
  cannot be filled in retrospectively, but a relation record can be appended); mirrors how
  `EntityGraph` is a separate structure from `Entity`.
- **Against:** two objects to fetch and join; a webhook consumer receiving one incident does not
  automatically see the relation; needs a delivery story of its own, which collides with ADR-004
  Open Question 4 (unresolved) and with idempotency/ledger questions.

### Option 3 — a first-class "finding" public object above `Incident`

The public object is a *finding* with subject, current value, interval, and revision history;
`Incident` stays the internal per-fire record.

- **For:** answers all four sub-problems in one coherent shape (§2 settle, §3 supersede, §4
  interval, §5 subject) instead of four bolt-ons; the consumer's mental model ("what is currently
  true, and how did we come to believe it") matches what clients actually asked for; the "one fact
  and its correction" reading of §3 becomes structural rather than conventional.
- **Against:** by far the largest surface, and the hardest to get right with one probe's worth of
  evidence; introduces a second public object beside `WatchEvent`, which ADR-004 Open Question 4
  already flags as unreconciled; risks the classic mistake of freezing a rich abstraction before
  the domain is understood.

### Cross-cutting trade-off, whichever option is chosen

Settle (§2 Fix A) trades **latency for correctness** on every finding; supersede (§2 Fix B) trades
**consumer complexity for correctness** on late corrections. They are independent knobs and should
be exposed and defaulted independently. A design that offers only one, or couples them behind one
setting, will fail the 21.6 s / 22.6 s / 34.0 s test in §2.

## Open questions — deferred to the "public object" design decision

> **Partially closed by the implementation** (see §Consequences → "Scope decisions taken for contract
> v1.4"): decision (c) settles the *delivery* half of Q1 and Q5 — provisional incident plus
> superseding revision, with any delay concern-specific and SLO-bounded, in delivery rather than in
> the gate. The *shape of the public object* (Q1's other half), Q2, Q3, Q4 and Q6 remain open.

1. **Which option above (or which hybrid) is the public object?** Not decided here.
2. **Does the finding carry an interval, or does an interval live on a separate object, or is
   "percept measures instants, not durations" the deliberate permanent answer?** All three are
   defensible. §4 establishes only that the interval is *not derivable today* and that
   forward-filling is unsound — not that percept must provide one.
3. **What is the relation vocabulary?** `supersedes` is evidenced by the probes.
   `corroborates`/`contradicts`/`refines`/`derived_from` are speculation. Freezing an unvalidated
   vocabulary is the failure mode ADR-003 warns about; deliberately shipping only `supersedes`
   first is a legitimate answer.
4. **Content-derived identity (§5): who supplies the content key?** The `Identity` strategy seam
   exists but `Observation` carries no content channel that reaches `match()`. Options: an optional
   defaulted field on `Observation`; routing `attrs` into `match()` (additive kwarg, per the
   `signature=` precedent); or an out-of-band identity resolver. Each has a different blast radius
   on the frozen contract and on every existing `Vision` backend's `detect()`. Not decided here.
5. **Should a settle window be a gate concept or a delivery concept?** Putting it in `Gate` reuses
   the one firing path (`gate.py:289`, invariant 3) but makes the gate hold state about
   *un-emitted* findings, which it has never done. Putting it in delivery keeps the gate pure but
   creates a second place where an event can be withheld — and ADR-002/ADR-001 §9's "a
   bound-driven drop is never silent" rule would need extending to cover a settle-cancelled
   finding. Both need a new reason code; note `GATE_REASONS` is a closed vocabulary
   (`gate.py:111-121`, with its `REASON_ACTION` map `:125-135` and `SUPPRESSION_REASONS` `:143`)
   and adding to it is a deliberate act.
6. **Does any of this reach `watch()`, or is it `run()`-only?** This is ADR-004 Open Question 4
   restated from the relations side and is explicitly *not* re-answered here. `watch()` yields
   `WatchEvent`, which has no `id` at all (`contracts/__init__.py:1135-1156`) — a relation needs
   something to point at, so any `watch()`-side answer implies giving `WatchEvent` an identity,
   which is additive but consequential.

---

Decision required on: all six open questions above, and specifically on the ordering claim
in §5 (identity before relations) — because if that ordering is accepted, the document-identity
work in §5/Q4 gates the relation work, and implementation should be sequenced accordingly.
