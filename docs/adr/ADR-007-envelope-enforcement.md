# ADR-007: Envelope enforcement in config, not prose — the three OUT verdicts

- **Status:** Accepted (the implementation; ships code, docs and tests — additive, no contract change)
- **Depends on:** ADR-005 (Source profiles — the `FORBIDDEN_FIELDS` precedent: structural
  enforcement over convention, and the `UNTUNED` pattern: a typed field a tool reads and reports,
  never a paragraph in a README); ADR-001 (the Gate/judgment contract — **ASK is a first-class
  verdict**, which is the whole mechanical argument in §2).
- **Related:** ADR-004 (Public delivery flow — `AttentionIntent` and the requested→effective clamp;
  §4 reports honestly what that surface can and cannot enforce); `percept/watch.py`
  (`WATCH_VERSION = "1.0.0"` — untouched, this ADR adds nothing to it).
- **Source (read for this ADR):** `percept/sourceprofile.py` (`UNTUNED` `:61`, `FORBIDDEN_FIELDS`
  `:73`), `percept/consent.py` (`ConsentProfile`, `permits`), `percept/agent.py` (`Percept.act` —
  the `modality == "act"` check), `percept/identity.py` (`ContinuityIdentity`,
  `AppearanceIdentity`), `percept/incidents.py` (`AttentionPolicy.clamp`, `RequestedDelivery`),
  `percept/doctor.py` (the findings vocabulary + exit codes), `percept/contracts/__init__.py`
  (`Goal.modality` `:195`, `AttentionIntent.preempt` `:1028`).
- **Ships:** `percept/envelope.py`, `ConsentProfile.actuation`, `ConsentSettings.actuation`
  (`PERCEPT_ACTUATION`), `percept doctor --actuation` + the ENVELOPE panel + exit code 6,
  `tests/test_envelope.py`.

## Context

The seven-domain evaluation returned **OUT** for three domains. Each OUT has a *narrowed survivor*
that is often one config change from the excluded form. Those refusals lived only in prose — and
prose does not stop a client wiring something at 2am.

| domain | why OUT | narrowed survivor |
|---|---|---|
| 24/7 security **identifying people** | biometric re-identification of non-consenting third parties, AND a safety mechanism on a clock percept cannot guarantee | scene-local, non-persisting individuation ("someone is at the door") with no cross-session person identity |
| **robotics in the control loop** | physical consequence on an unguaranteeable clock — and mechanically incoherent (see §2) | advisory acts (screen/software) and human-in-the-loop notification |
| **passive customer analytics** | the beneficiary is the business, not the observed person | an operator-facing watch on a condition the observed person also benefits from, with no per-person record |

ADR-005 already set the house rule for this class of problem. Gate semantics could have been
"documented as source-independent"; instead `SourceProfile` has *no gate field at all* and
`FORBIDDEN_FIELDS` makes a gate knob in `parameters` a construction error. This ADR applies the same
rule to the envelope.

## Decision

### 1. No person-identifying resolver ships, and one cannot be constructed

`percept/envelope.py` introduces `SubjectResolver` — a typed declaration of HOW a subject key is
obtained — plus `SubjectResolverRegistry` and the shipped `DEFAULT_SUBJECT_RESOLVERS`.

The refusal is in the constructor, not in a review checklist. A resolver whose `subject_class` is a
person class (`person`, `face`, `voice`, `biometric`) may not:

- `derives_from="pixels"` at all — that is biometric identification whatever it is called; or
- `persists=True` from anything percept derived — a persisting person key **is** a
  re-identification record however it was produced.

Both raise `EnvelopeViolation` at construction. `SubjectResolverRegistry.register` re-checks
`identifies_persons` independently, because the two guards fail independently (a duck-typed object
can reach `register` without `__post_init__`; a future constructor change could widen it).

The four shipped resolvers mirror `identity_strategy` values ADR-005's source profiles already name
— `track_iou_label`, `appearance_recent_set`, `content_key`, `speaker_label` — so this registry
describes reality rather than inventing a parallel one. None identifies a person:
`appearance_recent_set` is object-class and matches over the **recent set inside one run**;
`speaker_label` is a diarization **slot**, not a person, and is never carried across runs or matched
to a voice print.

**A client's own identity may enter — as an assertion, never as a derivation.** A badge scan, their
re-ID service or an SSO subject arrives via `client_asserted_subject(value=..., asserted_by=...)`,
which produces `binding="bound"`. `binding="bound"` is reachable ONLY from
`derives_from="client_asserted"` with a non-empty `asserted_by`. percept never promotes a handle it
computed into an identity, and shipping a *factory* rather than a registered resolver means no
release contains a person resolver anyone can simply switch on.

This one boundary carries two of the three verdicts: passive customer analytics becomes the excluded
form precisely when observations are keyed to a persisting person identity, and that key cannot be
constructed here.

### 2. `modality='act'` + physical actuation is a TYPED declaration

The robotics verdict is not only about consequence and clocks. It is **mechanically incoherent**:
the gate's honest failure mode is **ASK** (ADR-001), and a control loop cannot ask. A perception
system whose correct answer is sometimes "I need a human" cannot be the thing that decides whether
an actuator moves.

Following `UNTUNED`, this is a typed field a tool can read:

```python
ConsentProfile(allow_computer_use=True, allowed_actions=frozenset({"click"}),
               actuation="physical")   # none | screen | physical
```

- `none` (default — every existing config is byte-for-byte unchanged), `screen` (advisory
  software/computer-use acts — the narrowed survivor), `physical` (out of the envelope).
- An unknown value raises `EnvelopeViolation`; it is a declaration, not a preference.
- `actuation="physical"` makes `ConsentProfile.permits()` return `False` for **every** action, and
  `Percept.act` raises `ConsentDenied` quoting the verdict rather than an allowlist miss.
- `ConsentSettings.actuation` / `PERCEPT_ACTUATION` carry it end-to-end, so the declaration is
  reachable from deployment config and not only from Python.

The check is on the **pair**. `modality="act"` alone is fine (that is the survivor); `physical`
alone is fine (percept only speaks — though it is still reported, never silent). Together they are
`in_envelope=False`.

### 3. `percept doctor` says it

Every run now carries an `EnvelopeDiagnosis` (`report.envelope`, `to_dict()["envelope"]`, and an
ENVELOPE panel in `render()`), reporting the declared actuation class, which goals use
`modality='act'`, the shipped resolvers and how many of them identify persons (an invariant zero).

- `--actuation {none,screen,physical}` declares the class on the command line.
- `OUT_OF_ENVELOPE_ACTUATION` — **error** when the pair is declared, **warn** when `physical` is
  declared but nothing acts (the declaration is doing its job).
- `PERSON_IDENTIFYING_RESOLVER` — **error** if any registered resolver identifies persons. percept
  ships none, so this can only come from a third-party registration.
- New exit code **6** (`EXIT_OUT_OF_ENVELOPE`), additive beside 0/2/3/4/5.

An out-of-envelope declaration takes precedence over the coverage/source exit codes: what the client
needs to read first is that percept is being wired somewhere it refuses to go.

### 4. "Assistant-class only" is enforceable in DOCS, not in config — and here is what it would take

This is the card's reported item, and the honest answer is docs. It is recorded as data
(`ATTENTION_CLASS_ENFORCEMENT = "docs"`, `ATTENTION_CLASS_GAP`) and asserted in
`tests/test_envelope.py`, so the claim cannot drift away from the code.

`AttentionPolicy.clamp()` is real config enforcement — of **interrupt authority**. Its inputs are an
incident, a `GoalOrigin`, a `Consequence` and two allowlists (`allow_interrupt_consequences`,
`allow_interrupt_goal_ids`), and its only effect is reducing `preempt="interrupt"` to `"queue"`.
None of those inputs says *who listens*. `AttentionIntent.preferred_channel` is free text and
advisory: a sink that drives a factory horn is indistinguishable from one that speaks to a person.

To move this from `docs` to `declared`, three additive things are needed:

1. a typed consumer class on the delivery side — a sink declares
   `channel_class in ("assistant", "operator_console", "machine")` the way `SourceProfile` declares
   its strategies, instead of today's free-text `preferred_channel`;
2. a registry of delivery sinks keyed by that class, so the class is a property of the **wiring**
   and not of a string a caller passes;
3. an `AttentionPolicy` rule that refuses `machine` consumers outright and clamps them the way an
   unauthorized interrupt is clamped today.

Until (1) exists there is nothing for a policy to read. That work is not in this ADR: inventing a
`channel_class` without the sink registry would be a field nobody sets — which is the failure mode
this ADR exists to avoid.

## Consequences

- **Every existing configuration is unchanged.** `actuation` defaults to `"none"`; no shipped
  resolver changed behaviour; the identity implementations in `percept/identity.py` are untouched.
- **The registry is open but bounded.** Third parties register subject resolvers exactly as they
  register source profiles (ADR-005) — but the person refusal applies to them too, and doctor reports
  a violation as an error with exit 6.
- **`percept/watch.py` and `WATCH_VERSION` are untouched.** Nothing here reaches the frozen surface.
- **A future person resolver fails the suite if it violates this envelope.**
  `test_no_shipped_resolver_identifies_persons` asserts the live registry, and
  `assert_no_person_identifying_resolver()` is callable from any test or client process.
- **Open:** the delivery-side `channel_class` of §4, and whether an out-of-envelope declaration
  should also refuse to *start a run* rather than only refusing acts. Today the runtime refuses the
  act, which is where the physical consequence actually is.
