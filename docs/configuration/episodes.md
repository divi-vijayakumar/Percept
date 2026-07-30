---
title: Episodes — one occurrence, one notification
updated: 2026-07-29
applies_to: percept-vision 1.0.0
status: stable
---
# Episodes — one occurrence, one notification

A Watch that fires every time a condition is *still* true is unusable. A parked car is still parked
a thousand frames later; a package is still on the porch all afternoon. **An episode is the object
that says "this is the same happening as before"**, so the second, third and thousandth observation
of one occurrence produce no second notification.

This page covers what an episode does, the one thing you must configure to get any at all, and the
cost interaction you need to know about before you enable the verdict cache.

## What an episode gives you

Without episodes, suppression is the Gate's rising-edge rule alone: fire when the condition becomes
true, stay silent while it stays true, re-arm on a confident false. That is correct, but it is
per-`(goal, entity)` gate state with no notion of *the occurrence itself* — nothing survives an
entity id being re-minted, and nothing records that this is the 400th confirmation rather than the
first.

With an episode:

- the occurrence has an **identity** that outlives entity-tracking churn
- repeat observations fold **in place** — a `confirmed` heartbeat updates the episode rather than
  appending a row, so a 24/7 stream does not grow the log
- an already-emitted episode **does not consult the Gate again**, so a TTL sweep or a re-minted
  entity id cannot manufacture a duplicate notification
- the open episode is serialised into the judge's context, so the model answering "is the bag still
  there?" knows there is an *it* to still be there

## You must bind a SourceProfile

**This is the part that surprises people.** Episode semantics resolve from the source's identity
strategy, which lives on the `SourceProfile`. Configure a core without one and **no episode log is
created at all** — not an empty one, none. Every judgment falls back to gate-only behaviour, and
nothing in the incident stream tells you the layer is missing.

```python
from percept import Percept
from percept.source_profiles_default import PHYSICAL_SCENE

core = Percept.configure(
    vision="gemini",
    source_profile=PHYSICAL_SCENE,      # ← without this there are no episodes
    source_profile_explicit=True,
)
```

A camera watch folds a **scene-scoped** occurrence: one open occurrence per concern. The forfeit is
explicit — a scene singleton **cannot count subjects**. "A second person enters" is what the Gate's
per-entity key is for; per-entity episodes are deliberately not offered on camera sources, because
entity-id churn on real footage would turn every re-mint into a new occurrence.

A goal may also declare the scene scope directly, which is what the shipped goal templates do:

```python
goal.state["identity_policy"] = "scene"     # what DomainPack templates declare
```

A declared policy beats the profile default. Either route reaches the same scene fold and produces the
same occurrence semantics.

## What a long occurrence looks like

A condition that holds continuously produces **one** episode that stays `open`, **one** notification,
and a suppressed repeat for every subsequent observation. If you are seeing one notification per
observation, the episode layer is not engaged — check the profile first.

A condition that genuinely stops and starts again produces one episode per occurrence, each with its
own notification. That is not over-delivery; that is the number of things that happened.

## How episodes and the verdict cache compose

The verdict cache short-circuits a judge call when the current frame perceptually matches one already
judged for the same concern, subject and condition.

An open episode is serialised into the judge context, so a verdict formed under one episode state must
never be re-served under a different one. The cache handles that by **putting the episode state in the
key**, not by refusing to cache: the key carries a hash of the episode's *answer-affecting* state —
its identity, status, subject, value and the changes recorded against it — while the parts that move
on every tick without changing the answer stay out.

That split rests on a structural property rather than a guess. A `confirmed` heartbeat **coalesces in
place**: it advances "last seen" and appends nothing. So an occurrence that is simply continuing does
not disturb the key, and repeat observations of it can be served from cache. A genuine change — a
close, a reopen, an update, an annotation — changes the key, and the next judgment is a real one.

The practical consequence:

> **A long-running watch pays a model call when the occurrence changes, not on every frame that
> observes it continuing.**

Both mechanisms are off unless you turn them on. Episodes need a `SourceProfile`; the cache needs
`Settings.cache.enabled`, which defaults to `False`.

## Enabling the cache safely

Two settings need attention the first time you enable it on a new source.

**`Settings.cache.threshold` defaults to `0.0`, which means every hit is served.** The cache computes a
confidence for each hit — how closely the frame matched, decayed by how stale the entry is — and that
number is only ever compared against this threshold. At zero the comparison passes unconditionally, so
the similarity radius and the age decay gate nothing. Raise it, and both start doing their job.

**Raise `Settings.cache.audit_rate` too.** Re-perception is the **only** mechanism that detects a false
serve — a cached verdict a fresh judgment would have contradicted. At the `0.05` default a systematic
error is caught only after roughly nineteen of every twenty have already been delivered.

Note also that `Settings.cache.ttl_s` is not the main lever on hit rate. The layer holds at most one
entry per `(goal, entity, condition)` and overwrites on every store, so under dense admission the
slot already holds the immediately preceding judged frame whatever the TTL says. Size the TTL to how
long your scene stays decidable from a single frame.

## Related

- [Configuration overview](overview.md) — the layering rules and who owns each knob
- [Tunables](tunables.md) — every cache and episode setting, generated from the code
- [Profiles](profiles.md) — choosing and customising a `SourceProfile`
- [The gate](../watches/the-gate.md) — rising-edge, dwell and refractory, which episodes sit above
