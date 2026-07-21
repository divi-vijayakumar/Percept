# ADR-003: Evidence artifacts, dereference & retention

- **Status:** Proposed
- **Depends on:** ADR-001 §1 (the judgment lifecycle `Verdict -> GateDecision -> JudgmentEvent`, which
  is the contract this ADR proposes to *extend*) and ADR-001 §5 (two time domains — fixes the clock
  the evidence timestamps below are stamped in). ADR-001 does **not** define any evidence pointer;
  see the correction note below.
- **Related:** ADR-002 (Runtime health & coverage — separate control path, not addressed here);
  ADR-004 (Public delivery flow — what `Incident` durably carries forward from a `JudgmentEvent`,
  not addressed here)
- **Source:** `PRD-RevisionB.md` §7, §14 gate 5 (internal; not part of the public tree)

> **Correction (2026-07, before Accepted).** Earlier revisions of this ADR declared *"Depends on:
> ADR-001 §4 (defines `evidence.frame_ref` as a field on the judgment contract)"* and built their
> Context on that premise. **That premise is false and the citation does not resolve.** ADR-001 §4 is
> *Dwell and refractory*; no section of ADR-001 mentions `frame_ref`, and there is no
> `evidence.frame_ref` anywhere on the judgment contract in the shipped tree. This ADR therefore
> proposes a **new** field rather than resolving an existing one. The verified starting position is
> stated in Context below. Nothing else about the ADR's intent changes, and its status stays
> **Proposed**.

## Context

### The true starting position (verified against the shipped tree)

All line numbers below are in `packages/percept-vision/percept/contracts/__init__.py` at the time of
writing.

- **The judgment contract carries no evidence pointer at all.** `Verdict.evidence` is a plain
  `str = ""` (`L-125`) — free-text rationale from the judge, not a handle to anything.
  `JudgmentEvent` (`L-514`) carries `key`, `evidence_seq`, `action`, `timing` and **no evidence
  payload or pointer whatsoever** — `evidence_seq` is a per-key *ordering* integer (ADR-001 §8), not
  a reference to an artifact. `WatchEvent` (`L-1136`) carries `goal_id`, `action`, `t`, `text`,
  `entity_id`, `verdict` — again **no evidence handle**. There is nothing today to redefine: this
  ADR is a *proposal to add* an evidence pointer, and it is the first place that pointer is named.
- **The only `frame_ref` in the tree belongs to a different type with the opposite semantics.**
  `LabelRecord.frame_ref` (`L-1269`) is annotated *"LOCAL filesystem path ONLY; the local-only tail
  — NEVER egressed"*. It is a ground-truth **label** record, not a judgment, and its `frame_ref` is
  deliberately and explicitly a filesystem path — precisely what §1 below forbids for the new
  handle. The two must not be conflated or merged, and this ADR does **not** change
  `LabelRecord.frame_ref`.

To keep those two apart, the field this ADR proposes is named **`evidence_ref`**, not `frame_ref`.
Any cross-reference elsewhere in the ADR set to "ADR-003's `frame_ref`" means this new
`evidence_ref`.

### Why the gap is worth closing now

Two things are true about Percept today that make adding an evidence pointer — and pinning its
meaning *before* it exists — urgent rather than cosmetic:

1. **Percept's strongest technical direction is capsule judging** — giving the judge more than one
   naked frame: a hi-res crop alongside the overview, a short ordered burst around a fast event, a
   filmstrip spanning a slower one, or synchronized audio + adjacent video for an acoustic-onset
   escalation (see `AudioFrame`, `Concern.clip`/`window_s`/`fps` in
   `packages/percept-vision/percept/contracts/__init__.py`). Multi-artifact evidence is not a future
   nice-to-have; it is already how the runtime is built to look.
2. **The contract is frozen, so the field gets exactly one chance.** `WatchEvent`/`JudgmentEvent` are
   the frozen public contract, and `percept.watch()`'s signature plus `WATCH_VERSION` are frozen
   outright. Adding `evidence_ref` is *additive* and therefore permissible — but once added, if it
   ships as (or is read by early adopters as) "the path to the one JPEG that proved this," every
   capsule-judging capability built afterward either breaks that assumption or has to invent a second,
   parallel evidence channel outside the frozen contract to avoid breaking it. A frozen field with an
   implicit narrow meaning is worse than no field at all, because it cannot be fixed without a
   contract bump.

This ADR pins `evidence_ref`'s meaning *before the field exists* and before any capsule-judging code
ships, so the contract and the capability grow together instead of the capability having to route
around the contract.

### The in-tree precedent this ADR generalizes

The codebase already has one working precedent for "a durable record with a media pointer that must
not leak," and this ADR is a generalization of it rather than a new invention: **`LabelRecord`'s
egress-safe-head + local-only-tail split**
(`packages/percept-vision/percept/contracts/__init__.py`, class at `L-1246`, invariant documented at
`L-1046..L-1053`). Every field declared above `frame_ref` is the *metadata head* that MAY leave the
machine; `frame_ref` (`L-1269`) is the *local-only tail*. `to_egress()` (`L-1289`) is built from the
explicit `EGRESS_FIELDS` whitelist (`L-1277`), which simply does not list `frame_ref` — so the
no-pixels rule is a **type-level fact, not a runtime filter someone can forget to apply**. That
head/tail-by-whitelist construction is exactly the property §4 below carries over to `evidence_ref`.

Note the deliberate divergence: `LabelRecord.frame_ref` *is* a filesystem path (that is why it can
never egress), whereas the new `evidence_ref` is an opaque store key (§1) that also does not egress
by default (§4). Same structural guarantee, different reason — do not merge the two fields.

`WebhookSink` similarly defaults
`include_media=False` and strips `snippet.media`/`snippet.frames` from the JSON body unless a caller
opts in (`packages/percept-vision/percept/runtime/sinks.py`, `incident_payload`). Both are evidence
this repo already treats "a media pointer/blob crossing the network by default" as a bug class to
design out, not a policy to remember. This ADR generalizes that pattern to the new opaque-handle shape.

## Decision

### 1. Scope: a NEW field, `evidence_ref` — a name, not a description — and it may name a bundle

**Scope statement.** This section defines a **new, not-yet-existing** field, proposed for the
judgment contract and named **`evidence_ref`**. It does **not** describe, constrain, or redefine any
field in the shipped tree. In particular it says nothing about `LabelRecord.frame_ref`
(`packages/percept-vision/percept/contracts/__init__.py` `L-1269`), which is a documented LOCAL
filesystem path on a *label* record and stays exactly as it is; the "MUST NOT be treated as a path"
rule below applies to `evidence_ref` alone.

`evidence_ref` is an **opaque handle** into an evidence store. It carries no information about
what the referenced artifact *is* — not its media type, not whether it is one image or several, not
its size. A caller MUST NOT parse it, construct one, assume it is a filesystem path, or infer capture
shape from its format. The only operation defined on it is resolution (§2).

Concretely, this means: **the resolved artifact behind an `evidence_ref` MAY be a bundle** — an overview
frame, a hi-res crop, four ordered frames, a filmstrip, a short clip, or audio plus adjacent video —
determined by what the concern's `PerceptionSpec` captured (single frame vs. `clip`/`window_s` vs.
audio-fused), not by anything the name itself encodes. A judge or app that reads `evidence_ref` as
"the one image" and later receives a bundle-backed handle must not silently misbehave; `kind` (§2)
exists precisely so a caller can branch on shape before treating the artifact as a single image.

This is the entire point of the ADR: without it stated here, `evidence_ref` would pin the implementation to
"one image forever," and every capsule-judging feature has to fight the contract instead of using it.

### 2. The `EvidenceArtifact` shape and the resolver

```python
@dataclass(frozen=True)
class EvidenceArtifact:
    ref: str                                    # the same opaque handle as the judgment's evidence_ref
    kind: Literal["frame", "crop", "clip", "bundle"]
    media_type: str                             # e.g. "image/jpeg", "video/mp4", "application/x-percept-bundle"
    capture_start_ts: float
    capture_end_ts: float                       # == capture_start_ts for an instantaneous single frame
    sha256: str                                  # integrity + dedup; also the tamper-evidence anchor for audit
    expires_at: float | None                     # None = does not expire under the active retention class

artifact = w.resolve_evidence(event.evidence_ref)
```

`kind` distinguishes a single still (`"frame"`), a region-of-interest still (`"crop"`), a temporal
window (`"clip"` — covers both an ordered frame burst and an encoded short video, since both are
"more than one instant, one track"), and a heterogeneous multi-artifact capture (`"bundle"` — e.g. an
overview frame plus a hi-res crop, or a clip plus adjacent audio). A `"bundle"` resolves to an ordered
collection of sub-artifacts of the same frozen shape; this ADR does not further nest the type beyond
that — bundle contents are `list[EvidenceArtifact]` sharing one `ref`'s namespace, not a new type.

`capture_start_ts`/`capture_end_ts` are in the same monotonic production domain as `JudgmentTiming`
(ADR-001 §5) — never the untrusted `source_media_ts` — so evidence age and judgment latency are
directly comparable without a clock-domain conversion.

`w.resolve_evidence(...)` is the only sanctioned path from a handle to bytes. It is synchronous from
the caller's point of view (the store may be local disk, an embedded DB, or a future remote blob
store; that is an implementation detail behind the resolver, not a caller concern).

### 3. Typed failures — resolution never returns a silent empty result

```python
class EvidenceExpired(Exception): ...      # retention window closed; artifact was deleted on schedule
class EvidenceUnavailable(Exception): ...  # not (yet) written, store unreachable, or transient I/O failure
class EvidenceAccessDenied(Exception): ...  # caller lacks the authority to resolve this class of evidence
class EvidenceCorrupt(Exception): ...       # sha256 mismatch — bytes on disk do not match what was recorded
```

`resolve_evidence()` raises one of these four; it never returns `None` or an empty artifact to mean
"couldn't get it." A `None`/empty return is indistinguishable from "there genuinely was no evidence,"
which is a different fact a caller must not confuse with a resolution failure — this is the same
failure-typing discipline ADR-001 already applies to the judgment funnel, extended to the evidence
side of the contract. `EvidenceExpired` and `EvidenceUnavailable` are the two callers should expect
in normal operation (retention did its job; store hiccuped); `EvidenceAccessDenied` and
`EvidenceCorrupt` are exceptional and MUST be surfaced, not swallowed into a generic failure.

### 4. Handles are not filesystem paths, and are excluded from egress by default

An `evidence_ref` string carries no filesystem, URL, or storage-backend semantics a caller may rely on —
treating it as a path (opening it directly, constructing one by string concatenation) is a contract
violation even if it happens to work against today's local-disk implementation.

By default, `evidence_ref` is **excluded from any egress payload** (webhook bodies, external sinks) —
the same posture `LabelRecord.to_egress()` already enforces structurally for its own `frame_ref`
tail via an explicit whitelist, not a runtime filter. The handle is meaningless to a third-party
receiver that cannot call `resolve_evidence()` against this runtime, and an opaque store key
crossing the network by default is exactly the class of leak `LabelRecord`/`WebhookSink` already
guard against for the current media pointers. A sink MAY opt in to carrying the bare handle string
(for a trusted, co-located, or authenticated downstream resolver — mirroring today's
`WebhookSink(include_media=True)` opt-in shape) — that is a separate, narrower decision than whether
the sink may **dereference** the handle to embedded media bytes, addressed in §5.

## 5. Retention matrix — priced for the founder

Four evidence classes, priced against public cloud object-storage list pricing (S3-class tiers,
USD/GB-month, order-of-magnitude and re-checkable at build time — not a quote): **Standard** ≈
$0.023, **Infrequent-Access-class** ≈ $0.0125, **Archive/cold-instant-retrieval-class** ≈ $0.004.
Illustrative sizing: a single frame/crop ≈ 50–200 KB (JPEG); a short clip or 4-frame filmstrip bundle
≈ 0.5–5 MB. These are shape assumptions for comparing the options below, not measured production
numbers — rescale to the actual capsule size and fleet volume at build time.

| Class | Option A | Option B | Option C | Recommendation |
|---|---|---|---|---|
| **FIRE evidence** | Full bundle, hot tier, 30 days flat. Simplest; costs the most per incident over the window (≈30× a single day's bundle size, all at Standard-tier pricing). | Full bundle hot for 7 days, then degrade to the single best frame (drop clip/filmstrip) at IA-class pricing for the remaining 23 days. Cuts steady-state bytes ~5–10× after day 7 while keeping *some* audit trail through day 30. | Single best frame only, no clip persisted past the immediate delivery window (~24h); 30-day retention of that one frame. Cheapest; loses the capsule for later dispute/calibration review. | **B.** FIRE is the rare, consequential class the "trace replay to debug every miss/false fire" claim (§3.6) depends on — losing the full capsule after a week is acceptable, losing it entirely after 24h is not. |
| **ASK evidence** | Full bundle retained only until answered + 24h grace, then deleted. Cheapest; matches "ASK expires" (§7). | Full bundle retained a fixed 72h regardless of answer state, to cover a slow human reviewer. Costs more at high ASK volume — ASK is gameable/high-frequency by construction (§3.1), so flat retention scales with how often the gate abstains, not with anything consequential. | Drop the clip/bundle immediately on answer; keep the single frame + the label for 7 days as an audit trail of ASK→label decisions. | **A** as the default (bounded by ASK volume, not a flat clock); **C** available as an opt-in for teams running active ASK-calibration review — never B, since ASK is the class most exposed to volume-driven storage blowup. |
| **Sampled debug/benchmark (explicit opt-in only)** | Fixed-rate sampling (e.g. 1% of judged frames) behind an explicit `debug_capture=True` flag, retained 14 days hot, under a hard storage budget cap. | Time-boxed full-fidelity capture for an explicit benchmark run window only (e.g. "capture everything for 2 hours"), auto-expiring at run end + 48h — bounded blast radius by construction, good fit for `eval`/benchmark harness runs (§10). | No media by default; only structured `Verdict` logs unless a `--sample-media` flag is passed per invocation. | **C** as the absolute default (nothing captured absent an explicit flag), **B** for eval/benchmark harness sessions, **A** for standing drift-debugging — all three require the opt-in; none is ever on by default. |
| **Internal SILENT (debug-only, no default)** | Zero retention — SILENT never captures media. This IS the default; not a tradeoff against cost, since SILENT fires on every non-firing tick and is the highest-volume class by definition. | Under an explicit `debug_capture_silent=True` flag only: single frame, ring-buffer bounded (last *N* captures, hard byte cap), 1-hour TTL — for live-debugging a threshold/gate issue, never for standing retention. | (Rejected) full bundle retention for SILENT under any condition — SILENT's volume makes any persistence here a storage and privacy liability disproportionate to its debug value. | **A** always in shipped defaults; **B** only reachable through the explicit debug flag, and MUST NOT be a config a deployment can leave on unintentionally (short TTL, hard cap, ring-buffer — not an accumulating store). |

**Cross-cutting retention parameters** (apply across the table above; not a fifth row):

- **Maximum bytes.** One global `evidence_store_max_bytes` ceiling, with per-class sub-budgets so a
  debug-capture storm cannot starve FIRE evidence of its share. FIRE gets first claim on the budget;
  SILENT-debug gets the smallest, hardest-capped share.
- **Eviction order.** Within a class: oldest-past-expiry first. Across classes, when the global
  ceiling is hit before per-class TTLs would otherwise evict: SILENT-debug → sampled debug/benchmark
  → ASK → FIRE, i.e. FIRE is evicted last. A byte-pressure event must never be the reason a FIRE
  capsule disappears before an ASK or debug one does.
- **Privacy policy.** Reuse the existing `IncidentSnippet` precedent: redaction clears `media`/
  `frames` in place while keeping `kind`/`mime`/timestamps for the health dashboard. Evidence-store
  redaction (e.g. on an explicit operator/legal request) MUST be able to run ahead of a class's TTL,
  not only at TTL expiry — redaction is a distinct trigger from retention-window closure and both
  must be able to fire independently.
- **Persistence across restarts.** The `ref → artifact metadata + backing location` index MUST
  persist across process restarts (a crash must not silently orphan or leak an uncollectible
  artifact). Actual bytes may live on local disk or object storage keyed by `ref`, independent of
  process lifetime. On restart, a reconciliation sweep clears anything past its expiry that a crash
  prevented from being evicted on schedule.
- **Does labelling extend retention?** **Yes, bounded.** A `record_label()`'d incident (the flywheel's
  raw material, §14 gate elsewhere) has its evidence retention extended past its class default, since
  a human already spent attention confirming it — but to a fixed labelled-retention ceiling (not
  indefinitely), so an active labelling program cannot turn "labelled" into unbounded accumulation.
- **May webhook delivery dereference media?** **No, by default** — a webhook sink resolves nothing;
  it may carry the bare `evidence_ref` handle only if explicitly configured to (§4), and even then MUST
  NOT embed the resolved bytes in the payload without a second, explicit opt-in analogous to today's
  `WebhookSink(include_media=True)`. This mirrors the sink's existing default (`include_media=False`,
  `incident_payload()` strips `snippet.media`/`snippet.frames` unless asked) rather than introducing a
  new posture — the new opaque-handle system inherits the same default the current direct-media-embed
  path already has.

## Consequences

- Capsule-judging work (multi-frame, crop+overview, clip, audio-fused evidence) can proceed without
  a second, parallel "real evidence" channel outside the frozen contract — `evidence_ref` is specified
  to mean the right thing from the day it is added.
- Every consumer of `event.evidence_ref` must go through `resolve_evidence()` and handle four
  typed failures; no consumer may treat a resolution failure as "no evidence."
- Sinks (webhook or otherwise) need no new redaction logic to stay safe by default — exclusion is the
  default, matching the existing `LabelRecord`/`WebhookSink` posture, so this ADR closes a gap rather
  than opening one.
- Retention becomes a per-class, budget-aware system rather than a single flat TTL, which is more
  operational surface (four classes × eviction order × byte ceiling) but is required to keep FIRE
  evidence durable while SILENT/debug capture stays bounded by construction.

## Decision questions (§14 gate 5) — owner call required

Verbatim from `PRD-RevisionB.md` §14, gate 5: *"Evidence retention per class + byte ceilings (§7),
including whether webhook delivery may dereference media."* Specifically, before this ADR can move
from Proposed to Accepted, the owner must set:

1. **FIRE evidence retention window and degrade point.** Recommended default: full bundle hot 7
   days, degrade to single frame through day 30, then expire.
2. **ASK evidence retention window.** Recommended default: full bundle until answered + 24h grace,
   then delete; no flat-clock fallback.
3. **Sampled debug/benchmark opt-in mechanism and default sampling rate**, if the debug-capture path
   ships in v0.1 at all. Recommended default: opt-in only, no rate enabled out of the box (Option C
   in §5's row).
4. **SILENT debug-capture flag name, TTL, and ring-buffer cap**, if shipped. Recommended default:
   1-hour TTL, small fixed ring-buffer, off unless explicitly set.
5. **Global `evidence_store_max_bytes` ceiling and per-class sub-budget split.** No default
   recommended here — this is fleet-size- and deployment-dependent; the eviction *order*
   (SILENT-debug → sampled → ASK → FIRE) is the part this ADR fixes regardless of the number chosen.
6. **Labelled-incident retention ceiling** — the bounded extension a `record_label()`'d incident gets
   beyond its class default. No default number recommended; the "yes, bounded" policy is fixed, the
   ceiling is not.
7. **Whether webhook delivery may dereference media.** Recommended default: **no** — a sink may carry
   the bare handle only if explicitly configured to, and may embed resolved bytes only behind a
   second, separate explicit opt-in. This matches the existing `WebhookSink(include_media=False)`
   default already shipped in `percept/runtime/sinks.py`.
