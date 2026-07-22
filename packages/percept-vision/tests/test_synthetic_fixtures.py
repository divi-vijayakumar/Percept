"""Smoke test for the synthetic UNIT-layer capability corpora.

**What this asserts: the FIXTURES are valid.** Every corpus generates, writes, loads back, and its
ground truth is well-formed and internally consistent — so the corpus is provably ready before the
implementation cards that consume it arrive.

**What this deliberately does NOT assert: the capabilities themselves.** Subject keys, continuation
vs new episode, speech-act mutation, episode counting, interval gaps and idempotency are NOT BUILT
YET. A test here that asserted them would be asserting a plan, and would have to be deleted or
rewritten the moment the real behaviour landed. The capability assertions belong in the cards.

The generated data lives in the repo's gitignored ``eval/synthetic/``, so this test generates into a
tmp dir of its own and never depends on that directory existing. When it *does* exist (a developer
has run the generator), the last test additionally validates it in place.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from percept.contracts import Verdict, Vision
from percept.harness.scripted import ScriptedExhausted, ScriptedJudge, verdict_from_spec
from percept.harness.synthetic import (
    ACTIONS,
    BINDINGS,
    CORPUS_KIND,
    CORPUS_NAMES,
    REQUIRED_EXPECT_KEYS,
    SPEECH_ACTS,
    ValidationError,
    generate_all,
    load_all,
    load_corpus,
    synthetic_root,
    validate_corpus,
)


class _GeneratedCorpora(unittest.TestCase):
    """Generates the whole set ONCE into a tmp dir, then loads it back through the public loader."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        generate_all(cls.root)
        cls.corpora = load_all(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


class CorpusLoadTests(_GeneratedCorpora):
    def test_every_corpus_generates_and_loads(self):
        self.assertEqual(sorted(self.corpora), sorted(CORPUS_NAMES))
        for name, corpus in self.corpora.items():
            with self.subTest(corpus=name):
                self.assertEqual(corpus.kind, CORPUS_KIND[name])
                self.assertTrue(len(corpus), f"{name} has no cases")
                for f in ("stream.json", "labels.json"):
                    self.assertTrue((self.root / name / f).is_file(), f"{name}/{f} missing")

    def test_generation_is_deterministic(self):
        """Same seed, same bytes — the whole reason the data can be gitignored and regenerated."""
        with tempfile.TemporaryDirectory() as second:
            generate_all(Path(second))
            for name in CORPUS_NAMES:
                for f in ("stream.json", "labels.json"):
                    with self.subTest(corpus=name, file=f):
                        self.assertEqual(
                            (self.root / name / f).read_bytes(),
                            (Path(second) / name / f).read_bytes(),
                            f"{name}/{f} is not reproducible — a wall clock or unseeded RNG crept in",
                        )

    def test_manifest_and_readme_are_written(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(sorted(manifest["corpora"]), sorted(CORPUS_NAMES))
        self.assertTrue(manifest["synthetic"] and manifest["gitignored"])
        readme = (self.root / "README.md").read_text()
        # The README must state its limits, not just its contents.
        self.assertIn("does NOT", readme.replace("Does NOT", "does NOT"))
        self.assertIn("CAPABILITY fixture, NOT a quality benchmark", readme)


class GroundTruthWellFormedTests(_GeneratedCorpora):
    def test_every_case_states_an_expected_answer(self):
        for name, corpus in self.corpora.items():
            for case in corpus:
                with self.subTest(corpus=name, case=case.id):
                    self.assertTrue(case.description)
                    self.assertTrue(case.steps, "a case with no steps proves nothing")
                    self.assertTrue(case.expect, "ground truth must state the EXPECTED ANSWER")
                    for key in REQUIRED_EXPECT_KEYS[name]:
                        self.assertIn(key, case.expect)

    def test_every_case_declares_what_it_does_not_prove(self):
        """A fixture that does not state its limits will eventually be over-claimed."""
        for name, corpus in self.corpora.items():
            for case in corpus:
                with self.subTest(corpus=name, case=case.id):
                    self.assertTrue(case.proves.strip())
                    self.assertTrue(case.does_not_prove.strip())

    def test_case_ids_are_unique_within_a_corpus(self):
        for name, corpus in self.corpora.items():
            ids = [c.id for c in corpus]
            self.assertEqual(len(ids), len(set(ids)), f"{name} has duplicate case ids")

    def test_steps_are_monotonic_in_time(self):
        for name, corpus in self.corpora.items():
            for case in corpus:
                times = [s["t"] for s in case.steps]
                self.assertEqual(times, sorted(times), f"{name}/{case.id} goes backwards in time")

    def test_labelled_vocabularies_are_closed(self):
        for name, corpus in self.corpora.items():
            for case in corpus:
                with self.subTest(corpus=name, case=case.id):
                    for a in case.expect.get("actions", []):
                        self.assertIn(a, ACTIONS)
                    for a in case.expect.get("acts", []):
                        self.assertIn(a, SPEECH_ACTS)
                    for s in case.expect.get("subjects", []):
                        self.assertIn(s["binding"], BINDINGS)

    def test_validator_rejects_a_broken_corpus(self):
        """The validator has to be able to FAIL, or its passing means nothing."""
        corpus = self.corpora["continuation"]
        broken = json.loads(json.dumps(corpus.labels))
        broken["cases"][0]["expect"]["episode_count"] = 99  # disagrees with len(episodes)
        with self.assertRaises(ValidationError):
            validate_corpus("continuation", corpus.stream, broken)

        no_limits = json.loads(json.dumps(corpus.labels))
        no_limits["cases"][0]["does_not_prove"] = ""
        with self.assertRaises(ValidationError):
            validate_corpus("continuation", corpus.stream, no_limits)

    def test_missing_corpus_raises_with_the_fix(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_corpus("intervals", empty)
        self.assertIn("percept.harness.synthetic.generate", str(ctx.exception))


class ScriptedJudgeTests(unittest.TestCase):
    """The enabling piece: a judge that can be told exactly what to return, per step."""

    def test_satisfies_the_vision_protocol(self):
        self.assertIsInstance(ScriptedJudge([]), Vision)

    def test_returns_the_script_in_order(self):
        j = ScriptedJudge([
            {"satisfied": True, "confidence": 0.9},
            {"satisfied": False, "confidence": 0.5},
        ])

        async def go():
            return [await j.judge("c", "f"), await j.judge("c", "f")]

        first, second = asyncio.run(go())
        self.assertTrue(first.satisfied)
        self.assertFalse(second.satisfied)
        self.assertEqual(j.cursor, 2)
        self.assertEqual(j.remaining, 0)

    def test_exhaustion_is_loud(self):
        j = ScriptedJudge([{"satisfied": True, "confidence": 1.0}])

        async def go():
            await j.judge("c", "f")
            await j.judge("c", "f")

        with self.assertRaises(ScriptedExhausted):
            asyncio.run(go())

    def test_tolerates_verdict_fields_that_do_not_exist_yet(self):
        """Forward-compat: a fixture may label a field before it becomes a real ``Verdict`` field.

        ``operation`` has now landed (card Y2) — so it flows through the constructor to ``v.operation``
        and is NO LONGER a scripted extra — while ``subject_key``/``binding`` are still labelled ahead
        of any real field and remain extras. This proves both halves of the seam."""
        v = verdict_from_spec(
            {"satisfied": True, "confidence": 0.9, "subject_key": "document:INV-2291",
             "binding": "resolved", "operation": "replace"}
        )
        self.assertIsInstance(v, Verdict)
        # subject_key/binding are not real Verdict fields -> still carried as extras.
        self.assertEqual(v.subject_key, "document:INV-2291")
        self.assertEqual(v.scripted_extras["subject_key"], "document:INV-2291")
        self.assertEqual(v.scripted_extras["binding"], "resolved")
        # operation is a REAL field now -> promoted out of the extras and onto the dataclass.
        self.assertEqual(v.operation, "replace")
        self.assertNotIn("operation", v.scripted_extras)
        # ...and the real fields still behave like a plain Verdict.
        self.assertTrue(v.satisfied)
        self.assertEqual(v.evidence, "")

    def test_records_what_the_runtime_asked(self):
        j = ScriptedJudge([{"satisfied": True, "confidence": 1.0}])
        asyncio.run(j.judge("is the door open", "frame-1", context={"invoice_number": "INV-2291"}))
        self.assertEqual(j.calls[0]["condition"], "is the door open")
        self.assertEqual(j.calls[0]["context"], {"invoice_number": "INV-2291"})

    def test_by_token_and_hook_take_precedence(self):
        j = ScriptedJudge(
            [{"satisfied": False, "confidence": 0.1}],
            by_token={"special": {"satisfied": True, "confidence": 0.99}},
        )
        v = asyncio.run(j.judge("c", "special"))
        self.assertTrue(v.satisfied)
        self.assertEqual(j.cursor, 0, "a by_token hit must not consume the sequence")


class CorpusDrivesTheScriptedJudgeTests(_GeneratedCorpora):
    def test_every_frame_case_drives_one_call_per_step(self):
        """The fixtures are only useful if they actually drive a judge — prove the wiring, not the logic."""
        for name, corpus in self.corpora.items():
            if corpus.kind != "frames":
                continue
            for case in corpus:
                with self.subTest(corpus=name, case=case.id):
                    judge = ScriptedJudge.from_steps(case.steps)

                    async def go():
                        return [await judge.judge("scripted condition", s["frame"]) for s in case.steps]

                    verdicts = asyncio.run(go())
                    self.assertEqual(len(verdicts), len(case.steps))
                    self.assertEqual(judge.remaining, 0)
                    for v in verdicts:
                        self.assertIsInstance(v, Verdict)

    def test_turn_cases_carry_speaker_and_text(self):
        for case in self.corpora["mutation"]:
            for step in case.steps:
                self.assertTrue(step["speaker"])
                self.assertTrue(step["text"])


class OnDiskCorpusTests(unittest.TestCase):
    """If a developer has generated the real corpus, the on-disk copy must validate too."""

    def test_repo_corpus_validates_when_present(self):
        root = synthetic_root()
        if not (root / "manifest.json").is_file():
            self.skipTest(f"no generated corpus at {root} (gitignored by design — run the generator)")
        corpora = load_all(root)  # loading validates
        self.assertEqual(sorted(corpora), sorted(CORPUS_NAMES))


if __name__ == "__main__":
    unittest.main()
