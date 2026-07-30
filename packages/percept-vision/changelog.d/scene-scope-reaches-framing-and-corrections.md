### Scene scope now reaches evidence framing and the correction API

**What was wrong.** A concern that declares `identity_policy="scene"` has said there is no
per-entity subject: it reduces onto the non-entity gate key, and its occurrence is one scene
occurrence rather than one per entity. Two consumers of the entity graph never learned that, because
both predate `identity_policy` and neither was revisited when scene scope landed —
`_resolve_roi`'s focus branch by a month (2026-06-22 against 2026-07-21), `assert_attribute` by
slightly more.

- **The crop.** `PerceptionSpec.roi` defaults to `"focus"`, and **nothing in the package ever sets
  it**, so every concern inherited an entity-derived crop rather than choosing one. For a scene
  concern that crop came from `in_focus()` — *"the largest entity in the most recent frame"* — so
  the judge was handed whichever object won an election the concern had explicitly opted out of.
- **The correction.** `Percept.assert_attribute` bound to that same focus and returned `None` when
  there was none. A caller could not tell that apart from "nothing in focus yet", so on a deployment
  with no entity scope at all the user's word silently never reached the next judgment.

**What changed.**

- `_resolve_roi` now declines the **identity-derived** crop for a scene-declared concern and judges
  it on the whole frame. Only that crop is refused: an authored `region_hint` with `roi="hint"` is
  the non-identity where-channel and is still honoured, ahead of this branch, so a scene concern
  that genuinely wants a crop can still say so.
- `assert_attribute` raises the new **`UnsupportedCorrectionScopeError`** when *no* active concern
  carries entity scope — a binding that can never succeed, not now and not on a later frame. It
  still returns `None` when scope EXISTS but nothing is in focus on this frame; that is a timing
  condition, not a contradiction. Raise on what can never work, return `None` on what has not
  worked yet. An explicit `entity_id=` supplies the scope itself and always binds.

Both decisions read the **declared** policy, never the resolved one, for the same reason the detect
guard does: a concern that merely inherits a scene-like source-profile default still reduces on
focus identity, and widening either to the resolved policy turns a normal startup into an error.

**Why this is a correction, not a new policy.** The previous behaviour was residue rather than a
decision — no test and no doc coupled `roi` to identity scope, and the generated architecture page
showed `roi=focus` only because it is rendered from the code. Naming the decision in `_resolve_roi`
also removes it as a *silent side effect* of eliding the detect leg: with no observation leg there
is no focus, so the crop vanished either way and nothing said so.

**Not yet measured.** Whether the whole frame beats an arbitrary crop on real scene concerns is an
open question that needs a live provider. The argument here is that the crop was *arbitrary*, not
that it was *worse*. `tests/test_scene_scope_reaches_framing.py` pins the contract; the quality
comparison is a separate, budgeted decision.
