# RFC: `offline` mode + `scripted` backends — make faking an explicit, announced choice

**Status:** proposed (revises the earlier `mode=` RFC)
**Affects:** `agent.py` (`Percept.create`, `health`), `config.py` (`ModelSettings`), `backends/fake.py` (selector aliases), `tests/conftest.py`, examples/docs/skill
**Supersedes:** the `demo`/`fake`/`offline`/`stub` terminology in the prior draft
**Reconciles with:** the E3 stub-conductor warning already shipped in `9f3d6b9` (`agent.py`)
**Size:** ~60 lines SDK + one `conftest` fixture + a mechanical docs/examples sweep

---

## 1. The problem (one sentence)

Today a user gets **scripted backends by *not* typing something** — `ModelSettings` defaults every seam to a scripted double, so `Percept.create()` and even `Percept.create(vision="gemini")` return a fluent agent that authors no goals and looks completely healthy. The failure is silent because the scripted conductor's "do nothing" (`Decision(route="ignore")`) is a *legitimate value* — indistinguishable from a real conductor correctly declining.

Scripted backends are **good and necessary** — they're how 895 tests run keyless and how you test the template wiring. The doubles are not the bug. **Faking-by-default is the bug.** Faking must be a choice you *make and see*, never one you *fall into*.

---

## 2. Consistent vocabulary — two words, used everywhere

The single biggest source of confusion today is that this one idea is called `fake`, `offline`, `demo`, `stub`, `mock`, and `null` in different places. This RFC collapses it to **two** terms, used in config, logs, `health()`, and docs — and nowhere else.

| Concept | The word | Retired synonyms |
|---|---|---|
| the **mode** (the switch) | **`offline`** | `demo` (you demo *live* — a landmine), `test` |
| the **backends** it selects | **`scripted`** | `fake`, `stub`, `mock`, `null` |

> **`mode="offline"` selects `scripted` backends.** That is the entire vocabulary.

Class names (`FakeVision`, `FakeConductor`) may stay as internal implementation, but the **user-facing selector, logs, `health()`, and docs say `scripted`**. Add `"scripted"` as the canonical backend selector with `"fake"` kept as a deprecated alias:

```python
Percept.create(vision="scripted")     # canonical
Percept.create(vision="fake")         # still works, deprecation note
```

---

## 3. The switch (config)

You cannot land on scripted backends without naming the mode or explicitly selecting a scripted seam. Asking for scripted is a **required, visible act.**

```python
Percept.create(mode="offline")                    # explicit → all seams scripted
PERCEPT_MODE=offline                              # env, identical. kwarg wins on conflict.
Percept.create(mode="offline", conductor="gemini")# mixed, declared — still announced (§4)
```

Precedence: **explicit kwarg → `PERCEPT_MODE` → live.** The default is **not** offline — a bare `Percept.create()` targets real backends and errors on a missing credential rather than silently falling back to scripted (see §7 for the migration this forces and how the one `conftest` line absorbs it).

`config.py` stops defaulting the seams to scripted; the value comes from the mode, not a silent fallback:

```python
class ModelSettings:
    vision: str | None = None          # was: "fake"
    conductor: str | None = None       # was: "fake"
    specialist: str | None = None      # was: "fake"
    # env vars still override, as today
```

---

## 4. The logs — same word, every run, impossible to miss

Any run with **any** scripted backend announces it loudly, at construction, using the consistent vocabulary:

```
[percept] ⚠ OFFLINE — scripted backends: vision, conductor.
          This proves WIRING, not intelligence. A scripted conductor authors NO goals.
```

And `health()` lets a running system be *asked*, not guessed at:

```python
{"mode": "offline",
 "scripted": ["vision", "conductor"],
 "backends": {"vision": "FakeVision", "conductor": "FakeConductor"}}
```

*(Reconciliation: this replaces E3's `{"conductor":"fake","degraded":true,...}` shape from `9f3d6b9` — one payload, consistent keys. See §6.)*

**Banner discipline:** emit **once per process**, and suppress under the test `conftest` fixture — 895 banners would train people to scroll past banners, which is the exact failure this RFC exists to prevent.

---

## 5. The rule — scripted is always loud; a mute brain behind a live voice is loud *and* fatal

The earlier draft's "real vision + scripted conductor → always raise" **over-fires on the most common Percept pattern.** The conductor is the *language-authoring* seam: it turns an *utterance* into `goal_ops`. A **code-armed watch agent** — `run(source=…, on_incident=…)` with goals added via `add_goal()` and no voice — legitimately never invokes the conductor. There, a scripted conductor is correct and harmless.

So the trigger is not "real vision." It is **"a language path is wired to a scripted brain."** Two behaviors:

| Situation | Behavior |
|---|---|
| Any scripted backend | **Loud log** (§4) + `health().scripted` — always |
| Scripted conductor, **no** language path (code-armed watch) | **Loud log**, non-fatal: *"conductor is SCRIPTED — language goal-authoring is off; goals must be code-armed. Voice 'tell me when…' will be ignored."* |
| Scripted conductor **+ a live language path** (`on_utterance`, an attached transport, `arm_from_language`) | **`ModeError` at construction** — the user wired a voice and the brain silently can't author. This is the demo bug. |

The escape hatch stays ugly enough that nobody types it by accident:

```python
Percept.create(vision="gemini", on_utterance=h, allow_scripted_brain=True)  # constructs, loudly
```

### Check the object, not the string

Keep E3's existing instinct (already in `9f3d6b9`): decide on the **instantiated class**, not the requested string. A registry miss, a plugin, or a stray `PERCEPT_CONDUCTOR_BACKEND` can land you on a scripted backend *after* you asked for real — the stray-`.env` vector that "saved" the demo lived exactly there. The string is the request; the class is the fact.

---

## 6. Reconciliation with E3 (shipped in `9f3d6b9`)

`agent.py` already ships, from the Phase-1b loud-failure audit:

- `_is_fake_backend(obj)` — the object-check (KEEP; rename → `_is_scripted_backend`).
- `_degraded_conductor` field + `health()` degraded reporting (KEEP; re-key to the §4 `scripted` payload).
- `logger.warning("degraded_stub_conductor")` + `on_health("degraded_stub_conductor")`, fired **unconditionally** on real-vision+scripted-conductor.

**This RFC modifies, not replaces:**
1. Rename `degraded_stub_conductor` → the consistent `scripted` vocabulary (`on_health("scripted_conductor")`).
2. **Condition it** per §5: the unconditional warning becomes (a) an always-on log when scripted, plus (b) the code-armed vs. live-voice-path split — warning for the former, `ModeError` for the latter.
3. Fold the E3 `health()` degraded block into the single §4 payload.

Net: E3's object-check and health surface survive; the trigger gets correct, and the words get consistent.

---

## 7. Migration — one fixture absorbs the tests; the rest is a mechanical sweep

Flipping the default off "silently scripted" breaks every bare `Percept.create()`/`create(vision="gemini")` that relied on the scripted fallback — **loudly, at construction.** That's the correct failure. Absorb it, don't spread it:

**The 895 tests — one line, zero test edits:**
```python
# tests/conftest.py
@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """The suite is offline BY CONSTRUCTION — one declared property, not 895 implicit fallbacks."""
    monkeypatch.setenv("PERCEPT_MODE", "offline")
```

**Examples / docs / skill — a bounded, known list** (grepped) that must gain `mode="offline"` or a real key. Doing this sweep is itself the point: each edit converts a silent assumption into a stated one.
- `docs/explanation/architecture.md:118` (`Percept.create()` documented as "fakes, offline, no keys" → `mode="offline"`)
- `docs/tutorials/quickstart.md:51` plus the since-removed `examples/*.py` demos (passed a scripted `vision` object → gained `mode="offline"` for clarity before `examples/` was consolidated into `templates/`)
- `quickstart.md:113`, `docs/how-to/runtime.md:10` and the since-removed `examples/webcam_agent.py`/`realtime_watch_agent.py` (`vision="gemini"` + default conductor → exactly the §5 cases; each declared its intent: code-armed watch, or wire a real conductor)

**Positioning note (needs a one-line ruling, not code):** the OSS story is "free = keyless agent," and `architecture.md` sells keyless `create()` as a feature. Under this RFC, **keyless first-run is `Percept.create(mode="offline")`** — a keyless agent that is *honestly labelled scripted* rather than one that looks intelligent and isn't. Lead the README/quickstart with `mode="offline"` and the offline→scripted vocabulary. This is a strictly more honest keyless story; confirm it's the one we want before the OSS cut.

---

## 8. Sibling: `ScriptedConductor` — because a double must never return the value that looks like success

`mode` stops you *reaching* a scripted brain by accident. It does not make the scripted conductor a *useful* double — and today the **goal-authoring path has no keyless test at all**, because `FakeConductor.run` returns `route="ignore"` for every input. There is no offline test that asserts *"'tell me when the kettle boils' → a goal exists."* That gap is plausibly why the watch-tool bug shipped: the authoring path had nothing to fail.

`ScriptedConductor` maps utterance → `Decision` with real `goal_ops`, exactly as `ScriptedVision` (today's `FakeVision`) maps token → `Verdict`:

```python
conductor = ScriptedConductor({
    "caffeine": Goal(id="caffeine", condition="the person is drinking coffee or tea",
                     say="Hey — that's caffeine."),
})
# "tell me when I'm drinking caffeine" → GoalOp(op="create", goal=…)
# no match → Decision(route="respond_direct")   ← an HONEST no-author, NOT a silent "ignore"
```

The last line is the whole point, and it generalizes to every double in the SDK: **a scripted backend must never return the one value that is indistinguishable from success.** `FakeConductor`'s default `ignore` violates that; `respond_direct` doesn't. Sequence this **first** — it's what makes the §5 antibody testable.

---

## 9. Acceptance

1. `Percept.create()` with no key → **`ModeError`** naming the fix (set a key, or `mode="offline"`). *(Today: silently scripted.)*
2. `Percept.create(vision="gemini", on_utterance=h)` (a live language path + default scripted conductor) → **`ModeError`: scripted brain behind a live voice.** *(The test that matters — the demo bug, refused at construction.)*
3. `Percept.create(vision="gemini")` **code-armed, no language path** → constructs, **loud log** that language-authoring is off. *(Not a crash — the common watch-agent pattern.)*
4. `Percept.create(mode="offline")` → all scripted, banner once on stderr, `health()["mode"] == "offline"`, `health()["scripted"] == ["vision", "conductor", …]`.
5. `PERCEPT_MODE=offline` ≡ the kwarg; kwarg wins on conflict.
6. `vision="scripted"` is canonical; `vision="fake"` still works with a deprecation note.
7. `allow_scripted_brain=True` → case 2 constructs, loudly. The hatch exists and is ugly.
8. **Full suite green with the single `conftest` fixture and zero test edits.** If any test needs editing, the default is wrong.
9. `ScriptedConductor` authors a real goal from a scripted utterance; its no-match fallback is `respond_direct`, never `ignore`. A keyless test asserts *"utterance → goal exists."*

---

## 10. One line

Scripted backends are legitimate — for testing the template. They must be **switched on explicitly and announced loudly**, in **one consistent vocabulary** (`offline` mode, `scripted` backends), and the only configuration that silently *looks* like success — a scripted brain behind a live voice — must be **impossible to reach by accident.**
