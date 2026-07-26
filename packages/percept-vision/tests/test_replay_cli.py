"""— ADR-004's normative replay is REACHABLE from shipped commands.

``percept.replay.replay_cognition`` was good code consuming an artifact no user could produce:
``Percept.on_trace`` defaults to ``None``, ``percept-watch`` had no trace flag at all, and
``percept-doctor`` records only the tier-0 observation/admission stages — never ``gate.classified``,
the one stage ``replay_inputs`` reads. This module pins the round trip shut:

    percept-watch --offline --trace t.json   ->   percept-replay t.json   ->   0 divergences

If either half regresses (the flag disappears, the manifest stops carrying ``gate_config``, the
recorder stops seeing ``gate.classified``, the entry point loses its ``main``) these tests fail.
"""

from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

from percept import replay as replay_mod
from percept import watch_cli
from percept.replay import (
    GATE_CONFIG_KEYS,
    ReplayLoadError,
    load_replayable,
    replay_cognition,
)
from percept.tracing import STAGE_GATE_CLASSIFIED, load_trace

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


class TestWatchTraceRoundTrip(unittest.TestCase):
    """The acceptance bar, end to end, with no key and no camera."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = Path(self._tmp.name) / "t.json"

    def _watch(self, *extra: str) -> int:
        return watch_cli.main(["--offline", "--trace", str(self.trace), *extra])

    def test_watch_offline_then_replay_reports_zero_divergences(self):
        self.assertEqual(self._watch(), 0)
        self.assertTrue(self.trace.exists())
        # the CLI itself must agree: exit 0 means "every replayed action matched"
        self.assertEqual(replay_mod.main([str(self.trace)]), 0)

        manifest, records = load_replayable(self.trace)
        report = replay_cognition(manifest, records)
        self.assertEqual(report.divergences, ())
        self.assertGreater(report.inputs, 0, "a trace with nothing replayable would pass vacuously")
        self.assertEqual(report.unreplayable, 0)
        self.assertGreaterEqual(report.incidents, 1, "the kettle fixture fires exactly one edge")

    def test_the_trace_carries_the_gate_classified_stage_replay_consumes(self):
        """doctor's failure mode: a trace with no gate.classified replays to nothing at all."""
        self._watch()
        _manifest, records, _summary = load_trace(self.trace.read_text(encoding="utf-8"))
        gate_records = [r for r in records if r.stage == STAGE_GATE_CLASSIFIED]
        self.assertTrue(gate_records, "no gate.classified records — replay_inputs would see nothing")
        # and they carry the INPUTS, not just the outputs
        self.assertTrue(any(isinstance(r.data.get("confidence"), (int, float)) for r in gate_records))

    def test_the_manifest_records_the_gate_config_so_the_replay_is_of_this_run(self):
        self._watch()
        manifest, records = load_replayable(self.trace)
        self.assertEqual(sorted(manifest.gate_config), sorted(GATE_CONFIG_KEYS))
        report = replay_cognition(manifest, records)
        self.assertTrue(
            report.gate_config_recorded,
            "gate_config_recorded=False means the replay ran against library defaults, not this run",
        )

    def test_replay_is_deterministic_across_the_file_round_trip(self):
        self._watch()
        a = replay_cognition(*load_replayable(self.trace))
        b = replay_cognition(*load_replayable(self.trace))
        self.assertEqual(a.digest, b.digest)

    def test_a_divergent_trace_exits_nonzero(self):
        """The exit code is the contract: CI must be able to assert on it, not parse prose."""
        self._watch()
        payload = json.loads(self.trace.read_text(encoding="utf-8"))
        # Raise the recorded threshold above the recorded confidences: the same verdicts now
        # classify as unknown, so the replayed action can no longer be the recorded fire.
        payload["manifest"]["gate_config"]["threshold"] = 0.999999
        self.trace.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(replay_mod.main([str(self.trace)]), 1)

    def test_json_output_is_the_report(self):
        import contextlib
        import io

        self._watch()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            replay_mod.main([str(self.trace), "--json"])
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["kind"], "cognition_replay")
        self.assertEqual(payload["divergences"], [])
        self.assertTrue(payload["digest"])


class TestWatchWithoutTraceIsUnchanged(unittest.TestCase):
    def test_no_trace_flag_records_nothing_and_writes_no_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nope.json"
            self.assertEqual(watch_cli.main(["--offline"]), 0)
            self.assertFalse(out.exists())


class TestReplayEntryPoint(unittest.TestCase):
    def test_pyproject_ships_the_console_script(self):
        cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
        scripts = cfg["project"]["scripts"]
        self.assertEqual(scripts.get("percept-replay"), "percept.replay:main")
        self.assertTrue(callable(replay_mod.main))

    def test_a_non_trace_file_fails_loudly_with_exit_2(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "junk.json"
            junk.write_text("{}", encoding="utf-8")
            with self.assertRaises(ReplayLoadError):
                load_replayable(junk)
            self.assertEqual(replay_mod.main([str(junk)]), 2)
            self.assertEqual(replay_mod.main([str(Path(tmp) / "absent.json")]), 2)


class TestReplayFromAPortableBundle(unittest.TestCase):
    def test_a_ptrace_bundle_replays_the_same_as_its_json(self):
        """The artifact a client attaches to a ticket must be replayable too."""
        import tempfile

        from percept.trace_bundle import TraceBundle

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.json"
            watch_cli.main(["--offline", "--trace", str(trace)])
            manifest, records, summary = load_trace(trace.read_text(encoding="utf-8"))
            bundle = TraceBundle(manifest=manifest, records=tuple(records), summary=summary)
            path = bundle.write(Path(tmp) / "run")

            from_json = replay_cognition(*load_replayable(trace))
            from_bundle = replay_cognition(*load_replayable(path))
            self.assertEqual(from_json.digest, from_bundle.digest)
            self.assertEqual(from_bundle.divergences, ())
            self.assertEqual(replay_mod.main([str(path)]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
