# Scope and persistence (percept 1.0.0)

**Read this before you design around percept.** It states, plainly, what percept 1.0.0 covers, what it
explicitly does not cover, and what happens to state when the process restarts. Everything here is a
**decision already taken**, not a roadmap. It describes no new behaviour — it bounds how the shipped
behaviour may be read. (The wire contract at this release is `CONTRACT_VERSION = "1.10.0"`; the scope
below is a product decision, independent of that number.)

Normative source: the scope decisions recorded in ADR-006, mirrored in the module docstrings of
`percept/sourceprofile.py`, `percept/episodes.py`, and `percept/incidents.py` (the public code is the
source of truth).

---

## 1. Scope is narrow

percept 1.0.0 handles exactly two subject kinds:

| Subject kind | What identifies it | What you supply |
| --- | --- | --- |
| **Screen document with a stable content key** | a key derived from the *content* — invoice number, ticket id, file path | the key. Percept does not infer document identity from pixels. |
| **Conversational claim slot with explicit mutation semantics** | a named slot in a dialogue whose value is asserted / revised / retracted | the slot name and the mutation. Percept does not infer "this revises that" from timestamp order. |

### Explicitly excluded

These are **out of scope**, not on a backlog. Do not design against them:

- **Biometric re-identification** — deciding that two observations are the same *person*.
- **Growth / trajectory prediction** — inferring a subject's future value.
- **Robotics control** — closing a loop from a finding back onto an actuator.

Percept does not ship these and does not partially ship them.

### The seam is open; the scope is not

`SourceProfile.identity_strategy` is free text over an open registry, so a future resolution scheme
is added by *registering* one rather than by editing percept
([registration](registration.md), [derive a profile](../how-to/derive-a-profile.md)). That openness
is a seam for you to extend — it is **not** a claim that percept generalises. percept does not
generalise ahead of evidence.

---

## 2. Persistence is session-only

**Episodes are not persisted across restart.**

For a plain-goal core, `Percept.snapshot()` carries `goals`, `gate`, `entities` and `action_log`. It
does **not** carry `Percept.episodes` — the per-goal `EpisodeLog` map — so `Percept.restore()` cannot
bring it back. A core with an applied compiled Watch refuses `snapshot()`/`save_state()` with
`CompiledWatchSnapshotUnsupportedError`; the OSS runtime does not create a partial recovery artifact
for session-local runtime, profile, judge, or episode bindings.

What that means concretely:

- after a restart, every counting goal starts again at **zero episodes**;
- a subject last seen before the restart is a **new** episode afterwards;
- pending `hold_s` dwell starts again; `Gate.restore()` drops its pre-restart clock stamp;
- `EpisodeLog`'s `window_s` / `max_episodes` bounds are a **memory** story (a long-running process
  does not grow without limit). They are **not** a durability story.

**This blocks any 24/7 continuous-operation guarantee** until a later release gives episodes a
persisted representation. If your deployment plan assumes continuous multi-day counting across
redeploys, percept 1.0.0 does not support it, and it will fail quietly — as under-counting, not as an
error.

### `scope='durable'` means identifier stability, never restart-safety

If a subject reference carries `scope='durable'`, that describes **the identifier**: the string
`PO-4471` denotes the same real-world subject next week as it does today.

It says **nothing** about whether episode state, gate edges or counts survive a process restart.
They do not. The two readings are easy to conflate and the cost of conflating them is silent
under-counting after a redeploy.

---

## 3. Delivery is provisional incident + superseding revision, not a wide settle window

Percept does **not** hold findings behind a wide settle window waiting for the world to stop
changing. The posture is:

1. **Emit immediately** on the rising edge, as a *provisional* incident. Firing on the edge is what
   percept is for; a global settle window would destroy it.
2. **Supersede** it when a correction lands, so the consumer can resolve "which value is current"
   without inventing an ordering rule.

Any delay is **concern-specific** and **bounded by a declared latency SLO** (`percept/slo.py`) —
never a global default applied to every finding.

Consumer consequence: treat a finding as provisional until superseded, and resolve currency by the
supersede link rather than by "latest timestamp wins", which the runtime has never promised. The
incident-relations design (ADR-006) is why an unlinked correction is treated as a defect class rather
than as partial credit; the behaviour lives in `percept/incidents.py`.
