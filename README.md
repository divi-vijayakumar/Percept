# percept

The open-source **cognition layer** for goal-driven, proactive vision agents.

You state a goal in plain language — *"nudge me when I drink a coffee"*, *"tell me when the kettle
boils"* — and percept turns a live audio-visual stream into an agent that:

- **reasons over entities across time** — tracked things with stable ids, attributes carrying
  provenance (*observed* vs *you-told-me*) and freshness;
- **fires proactively**, but only on the **rising edge** of a condition becoming true;
- **refuses to guess** — a three-state gate maps *known → act · not → silent · unknown → ask*, so it
  never delivers a confidently-wrong nag.

The wedge is **temporal cognition** (entity memory + the three-state gate + reasoning over events),
which a raw VLM-in-a-loop and a per-frame pipeline both lack. percept builds on frontier models for
perception behind vendor-neutral seams.

## The envelope (refusals, stated proudly)

- **Assistant-class, never the safety mechanism.** percept informs a human who stays responsible. It
  is never the thing standing between a person and harm on a clock it can't guarantee. *Out:*
  driver-drowsiness intervention, turn detection. *In:* a post-hoc driving debrief.
- **Watches the user's own world, with the user as beneficiary** — never a non-consenting third party
  to the wearer's advantage. *Out:* covert analysis of someone you're negotiating with; card-counting
  against a live casino. *In:* a play-money practice trainer.

These refusals are a feature. **Do not lead with surveillance demos.**

## Status

**Phase 0 — contracts, skeleton, fakes, CI.** The frozen membrane (`adapters.py`), the backend
registry, deterministic fakes, the cache cascade, and the three CI lanes are in place. The system
*runs* offline with fakes but does nothing intelligent yet — the gate and entity graph arrive in
Phase 1. See [`Spec/`](Spec/) for the architecture and the phase-by-phase plan.

## Quickstart (no keys)

```bash
make test                       # L1 unit lane — fakes only, <2s, no keys
python3 -c "import sys; sys.path.insert(0,'packages/percept-vision'); \
  import percept; a = percept.Percept.create(vision='fake'); print(a.health())"
```

## Layout

```
packages/percept-vision/   the SDK (pip install percept-vision; import percept)
eval/                       golden corpus — two tracks (see eval/README.md)
Spec/                       the spec + implementation plan
Makefile                    test · eval-live · e2e · bench · check · reproduce
```

## License

Apache-2.0.
