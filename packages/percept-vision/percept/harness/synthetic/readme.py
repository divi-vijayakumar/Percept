"""The text of ``eval/synthetic/README.md``.

It lives here, in the tree, because the directory it is written into is **gitignored** — a README
committed to ``eval/synthetic/`` would vanish on a fresh clone. The generator emits it alongside the
data, so the explanation and the corpus can never drift apart.
"""

from __future__ import annotations

from .corpora import episode_counting as _ec
from .corpora import idempotency as _idem
from .corpora import longhorizon as _lh
from .corpora._common import CONTINUITY_GAP_S as _GAP
from .schema import SCHEMA_VERSION, SEED

__all__ = ["render"]

_TEMPLATE = """\
# Synthetic capability corpora (UNIT layer)

**GENERATED — do not edit, do not commit.** This whole directory is gitignored. Regenerate with:

```bash
cd packages/percept-vision && python3 -m percept.harness.synthetic.generate
```

Seed `{seed}`, schema version `{schema}`. Generation is a pure function of the seed and module
constants — no wall clock, no unseeded randomness — so regeneration is byte-identical everywhere.

## The split these corpora sit inside

> **Capability is deterministic and belongs in unit tests.**
> **Quality is statistical and belongs in an end-to-end benchmark at release.**

This is the **unit** layer. Every fixture is synthetic, deterministic, cheap, and drivable by
`percept.harness.scripted.ScriptedJudge` with **no API key, no spend, no real footage**.

A green corpus means: *the mechanism can express the right answer.*
A green corpus never means: *a real model would produce that answer on real input.*

## What is here

Each corpus directory holds two files:

- `stream.json` — the INPUT. Cases of ordered `steps`; a frames step carries `{{t, frame,
  observations, verdict}}`, a turns step carries `{{t, speaker, text}}`. The `verdict` block is what
  a scripted judge returns for that step, and may carry fields that do not exist on `Verdict` yet
  (`subject_key`, `binding`, `operation`) — `ScriptedJudge` attaches them additively today and will
  pass them to the constructor the day P0 promotes them.
- `labels.json` — the GROUND TRUTH: per case an `expect` block (the expected ANSWER, machine
  readable), plus `proves` and `does_not_prove` strings. Most cases also carry a `wrong_answer`
  block naming the specific failure mode the case exists to catch.

Load them with:

```python
from percept.harness.synthetic import load_corpus, load_all, ScriptedJudge

corpus = load_corpus("continuation")
case = corpus.case("same_subject_after_gap_is_two_episodes")
judge = ScriptedJudge.from_steps(case.steps)   # returns case.steps[i]["verdict"] on call i
case.expect["episode_count"]                    # -> 2
```

---

## 1. `subject_identity/` — what makes two sightings the same thing

| case | expects |
|---|---|
| `same_region_same_label_different_keys` | 2 distinct subjects (same region, same label, keys INV-2291 vs INV-2294) |
| `unreadable_key_binds_unknown` | 1 subject, `binding="unknown"`, `key=None` |
| `reappearance_after_unrelated_document` | 2 subjects; INV-2291 resolves to the SAME key at steps 0 and 2 |

**Proves:** identity must be expressible from a durable content key — including the honest third
answer `unknown`, which is neither a fabricated key nor a silently dropped subject. The first case
is exactly the input today's IoU+label identity model cannot express.
**Does NOT prove:** that any real OCR/VLM reads a key correctly, or that keys survive restarts.

## 2. `continuation/` — continuation vs new episode

Labelled against a continuity gap of **{gap:g}s**.

| case | expects |
|---|---|
| `same_subject_inside_gap_is_one_episode` | 1 episode, actions `fire, silent` |
| `same_subject_after_gap_is_two_episodes` | 2 episodes, actions `fire, fire` |
| `interleaved_a_b_a_continues_a` | 2 episodes, 2 fires — A's return CONTINUES A; **no spurious third fire** |
| `b_stuck_in_ask_must_not_refire_a` | 2 episodes, **1 fire** — B's unresolved ASK must not let A refire |

**Proves:** a durable subject key must not imply permanent suppression (past the gap the same
subject fires again), episode state is per-subject rather than per-recency, and one subject's
evaluation state never governs another's duplicate suppression.
**Does NOT prove:** that {gap:g}s is the right gap for any domain, or that a real judge attributes
each sighting to the right subject in the first place.

## 3. `mutation/` — containing a value is not setting it

Turns, against the goal condition *"a BUYER-side speaker states the approved budget"*.

| utterance | act | value after |
|---|---|---|
| "The budget is $40K" | assert | $40K |
| "Could we do $25K?" | mention | $40K |
| "Another customer paid $25K" | mention | $40K |
| "The budget is not $25K" | negation | $40K |
| "Actually, the approved budget is $25K" | **replace** | **$25K** |

Plus `valueless_retraction_clears_the_value` ("Scratch what I said about the budget" -> value
becomes `None`) and `seller_lacks_authority_to_replace_buyer_value` (a textbook replace from the
wrong speaker leaves the buyer's $40K standing).

**Proves:** the state machine downstream of act classification — four utterances contain `$25K` and
only one changes anything; retraction is a first-class valueless operation; authority is checked
separately from the act.
**Does NOT prove:** that a model classifies these speech acts correctly, or that speaker roles are
reliably diarized. Act-classification accuracy is a release-benchmark question.

## 4. `episode_counting/` — the cup problem

Boundary gap **{ecgap:g}s**, epsilon **{eceps:g}s**. Rule: a separation `>=` the gap starts a NEW
episode.

| case | expects |
|---|---|
| `three_separated_occurrences` | count 3 |
| `one_continuous_occurrence_sampled_fifty_times` | **count 1**, not 50 |
| `boundary_separation_exactly_gap` | count 2 (the `>=` is inclusive) |
| `boundary_separation_gap_minus_epsilon` | count 1 |
| `boundary_separation_gap_plus_epsilon` | count 2 |

**Proves:** counting is over episodes, not over satisfied frames, and the boundary rule itself is
pinned — it cannot drift without a fixture going red. Note this needed **no real footage**: the
input that matters is the pattern of presence and absence over time.
**Does NOT prove:** that a real judge detects each occurrence, or that epsilon-scale timing survives
real jitter, dropped frames and clock skew.

## 5. `intervals/` — coverage comes from observation, never interpolation

| case | expects |
|---|---|
| `two_sparse_observations_are_not_a_span` | two point observations + one `unobserved` gap — **not** a 19s interval |
| `continuous_observation_is_the_span` | one merged `observed` interval 0..10 |
| `degraded_judge_gap_is_not_continued_truth` | intervals 0..3 and 9..10, plus a `judge_degraded` gap |

**Proves:** forward-filling an unobserved gap is rejected; an outage is reported as a gap with a
reason, not as continued truth and not as a confident negative.
**Does NOT prove:** what actually happened during the gap — that is unknowable from the stream,
which is the point — nor how the runtime should recover afterwards.

## 6. `idempotency/` — one happening, one incident

| case | expects |
|---|---|
| `unchanged_confirmation_does_not_reincident` | 1 incident from 3 satisfied ticks |
| `key_ttl_expiry_while_episode_open_does_not_refire` | 1 incident across a {ttl:g}s gate-key TTL sweep |

**Proves:** re-confirmation is not news, and a memory-bound eviction is bookkeeping that must never
surface as a user-visible event while the episode is still open.
**Does NOT prove:** that the TTL/max-keys bounds are correctly sized for a 24/7 stream — that is a
soak question.

## 7. `longhorizon/` — READ THIS ONE CAREFULLY

A **{mins:g}-minute SYNTHETIC stream** ({steps} steps at {sample:g}s cadence) in which the same
subject recurs at roughly minutes 2, 7 and 14. Expected: **3 episodes**, boundaries
{bounds}, same durable subject key throughout.

**This is a CAPABILITY fixture, NOT a quality benchmark.** It proves the MECHANISM links occurrence
1 to occurrence 3 across a >=10-minute horizon — that memory survives the horizon and the key is
still the same key twelve minutes later.

**It proves NOTHING about real-world accuracy.** The stream is generated and the judge is perfect:
no appearance drift, no occlusion, no lighting change, no look-alikes, no missed detections. It must
never be cited as long-horizon re-identification performance. That number can only come from the
end-to-end benchmark on real footage.

---

## What none of this covers

- **Model behaviour.** Every verdict here is scripted. Nothing in this directory constrains what a
  real judge returns.
- **Perception quality.** No recall, precision, latency or cost figure can be derived from these
  corpora.
- **Real-world timing.** Steps are exact synthetic timestamps; real streams jitter, drop and skew.
- **Long-horizon accuracy.** See corpus 7 above.

If you want any of those numbers, run the release benchmark on real footage. These fixtures exist so
that the *logic* underneath it is already known-correct before you spend a cent on it.
"""


def render() -> str:
    return _TEMPLATE.format(
        seed=SEED,
        schema=SCHEMA_VERSION,
        gap=_GAP,
        ecgap=_ec.BOUNDARY_GAP_S,
        eceps=_ec.EPSILON_S,
        ttl=_idem.KEY_TTL_S,
        mins=_lh.DURATION_S / 60.0,
        steps=int(_lh.DURATION_S / _lh.SAMPLE_S) + 1,
        sample=_lh.SAMPLE_S,
        bounds=", ".join(f"[{a:g}, {b:g}]s" for a, b in _lh.OCCURRENCES),
    )
