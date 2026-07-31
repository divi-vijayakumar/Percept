"""Self-echo hardening for the ``on_utterance`` seam.

The seam opens an un-AEC'd mic and hands the app raw PCM. If the handler blocks the audio pump (as it
did originally) a slow voice turn creates a capture backlog that is replayed through the endpointer the
instant it returns — Percept endpoints its own spoken reply and loops. These tests pin the fix:

  * R1 — the handler runs OFF the pump, so capture never stalls (and ``muted()`` is evaluated live).
  * R2/R3 — ``deaf_until`` suppresses endpointing for a window and RESETS the VAD on exit, so the tail
    of our own speech can't be endpointed and a half-built utterance can't straddle the window.
  * ``mute()`` set during a slow handler suppresses muted-window chunks (only true
    once the pump is live, i.e. post-R1).
  * A raising handler no longer kills the pump.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame
from percept.edge.detectors import WakePolicy
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run

STILL = gray_frame(8, 8, 0)  # identical frames => zero motion => the visual gate never escalates


def _core(*, gate_on: str | None = None) -> Percept:
    vision = FakeVision(script={"quiet": Verdict(satisfied=False, confidence=0.9, evidence="nothing")})
    core = Percept.create(vision=vision, discover_plugins=False)
    state = {"cadence": {"gate_on": gate_on}} if gate_on else {}
    core.add_goal(Goal(id="listen", condition="someone is speaking", say="heard you", state=state))
    return core


def _frame(byte: int) -> AudioFrame:
    return AudioFrame(pcm=bytes([byte, byte]) * 1600, sample_rate=16000, channels=1)


async def _audio_gen(energies, *, dt=0.1, offset=0.05, clock=None, done=None):
    """Async audio source. If ``clock`` (a 1-list) is given, stamp it with each chunk's t before yield so
    an injected ``wall_clock`` reads wall==stream. Sets ``done`` (an Event) after the last chunk."""
    for i, e in enumerate(energies):
        at = i * dt + offset
        if clock is not None:
            clock[0] = at
        yield (at, _frame(0x40 if e > 0 else 0x00), e)
        await asyncio.sleep(0)
    if done is not None:
        done.set()


async def _held_video(gate: "asyncio.Event", *, warmup=5):
    """A video source that yields a few frames then blocks on ``gate`` — keeps ``run()`` alive (the main
    loop is parked awaiting the next frame) while the audio pump runs, without ending the run early."""
    for i in range(warmup):
        yield (float(i), STILL, "quiet")
        await asyncio.sleep(0)
    await gate.wait()
    yield (999.0, STILL, "quiet")


async def _collect(core, video, **kw):
    return [inc async for inc in run(core, source=video, **kw)]


class OffPumpTests(unittest.TestCase):
    def test_slow_handler_does_not_block_the_pump(self):
        """R1: while a slow handler runs, the pump keeps draining capture (pre-R1 it froze here)."""
        core = _core()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls: list = []
        entry_chunks: list = [None]

        async def handler(clip, t):
            calls.append(t)
            entry_chunks[0] = stats.audio_chunks
            entered.set()
            await release.wait()  # a long turn — must NOT stall capture

        # one utterance (endpoints ~frame 13), then a long silent tail the pump should keep consuming.
        energies = [0.0] * 2 + [0.9] * 4 + [0.0] * 20
        stats = RuntimeStats()

        async def go():
            task = asyncio.ensure_future(_collect(
                core, _held_video(release), audio_source=_audio_gen(energies),
                on_utterance=handler, stats=stats))
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            for _ in range(400):  # let the (off-pump) pump drain the tail while the handler is blocked
                await asyncio.sleep(0)
            drained_while_blocked = stats.audio_chunks
            release.set()
            await asyncio.wait_for(task, timeout=2.0)
            return drained_while_blocked

        drained = asyncio.run(go())
        self.assertEqual(len(calls), 1)
        # The pump advanced well past the endpoint chunk while the handler was still blocked.
        self.assertGreater(drained, entry_chunks[0])
        self.assertEqual(drained, len(energies))  # it drained EVERYTHING off-pump

    def test_raising_handler_leaves_the_pump_running(self):
        """R1 must not regress the sink-error policy: an off-pump handler that raises is reported but the
        pump survives and later utterances still fire (inline+raise used to kill the run)."""
        core = _core()
        calls: list = []
        health: list = []
        done = asyncio.Event()

        async def handler(clip, t):
            calls.append(t)
            raise RuntimeError("boom")

        # two separate utterances
        energies = [0.0] * 2 + [0.9] * 4 + [0.0] * 10 + [0.9] * 4 + [0.0] * 10
        stats = RuntimeStats()

        async def go():
            await asyncio.wait_for(_collect(
                core, _held_video(done), audio_source=_audio_gen(energies, done=done),
                on_utterance=handler, stats=stats, on_health=lambda e: health.append(e.kind)), timeout=3.0)

        asyncio.run(go())  # must not raise
        self.assertEqual(len(calls), 2)                    # both utterances reached the handler
        self.assertEqual(stats.sink_errors, 2)             # both errors were counted
        self.assertIn("utterance_sink_error", health)      # and surfaced to on_health


class MuteDuringHandlerTests(unittest.TestCase):
    def test_mute_during_slow_handler_suppresses_endpointing(self):
        """Mute set while a slow handler runs suppresses muted-window chunks — only true
        because the pump keeps running (R1). Pre-R1 the backlog drained after unmute -> a 2nd endpoint."""
        core = _core()
        muted = [False]
        release = asyncio.Event()
        entered = asyncio.Event()
        fed = asyncio.Event()
        calls: list = []

        async def handler(clip, t):
            calls.append(t)
            muted[0] = True          # the app mutes the moment it starts responding
            entered.set()
            await release.wait()
            muted[0] = False         # ...and unmutes when the turn is done

        # utterance 1 (endpoints), then a 2nd speech burst that arrives DURING the handler.
        energies = [0.0] * 2 + [0.9] * 4 + [0.0] * 10 + [0.9] * 5 + [0.0] * 10
        stats = RuntimeStats()

        async def go():
            task = asyncio.ensure_future(_collect(
                core, _held_video(release), audio_source=_audio_gen(energies, done=fed),
                on_utterance=handler, muted=lambda: muted[0], stats=stats))
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            await asyncio.wait_for(fed.wait(), timeout=2.0)  # 2nd burst consumed LIVE while muted -> skipped
            release.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(go())
        self.assertEqual(len(calls), 1, "the muted 2nd burst must not endpoint")


class DeafenTests(unittest.TestCase):
    def _run_stream(self, energies, deadline):
        core = _core()
        clock = [0.0]
        heard: list = []
        durs: list = []

        video = [(float(i), STILL, "quiet") for i in range(len(energies) + 12)]

        async def go():
            await _collect(
                core, video, audio_source=_audio_gen(energies, clock=clock),
                on_utterance=lambda a, t: heard.append(t),
                deaf_until=lambda: deadline, wall_clock=lambda: clock[0],
                on_health=lambda e: durs.append(e.data["dur_s"]) if e.kind == "utterance" else None,
            )

        asyncio.run(go())
        return heard, durs

    def test_fully_deafened_utterance_never_endpoints(self):
        """R2: with the window covering the whole utterance, nothing is endpointed."""
        energies = [0.0] * 2 + [0.9] * 5 + [0.0] * 12
        heard, _ = self._run_stream(energies, deadline=10.0)   # deaf for the whole stream
        self.assertEqual(heard, [])
        # control: same stream, never deafened -> it DOES endpoint exactly once
        heard2, _ = self._run_stream(energies, deadline=0.0)
        self.assertEqual(len(heard2), 1)

    def test_deafen_resets_a_straddling_utterance(self):
        """R2/R3: speech that starts inside the window and continues past it does not survive — the VAD
        is reset on exit, so the endpointed clip covers only the POST-window speech (shorter duration)."""
        # frames: 2 sil, 5 speech (in window), 6 speech (out of window), 13 sil.
        energies = [0.0] * 2 + [0.9] * 5 + [0.9] * 6 + [0.0] * 13
        # window ends at t=0.65 -> frame 7 (t=0.75) is the first post-window speech frame.
        heard, durs = self._run_stream(energies, deadline=0.65)
        self.assertEqual(len(heard), 1)
        self.assertTrue(durs)
        # post-window start (t~0.75) -> dur ~1.3s; had the pre-window speech (t~0.25) survived it'd be ~1.8s.
        self.assertLess(durs[0], 1.55)

        # control: never deafened -> the whole run counts, longer duration.
        _, durs_ctrl = self._run_stream(energies, deadline=0.0)
        self.assertGreater(durs_ctrl[0], 1.55)


class EchoDuringHandlerTests(unittest.TestCase):
    def test_echo_during_handler_fires_once(self):
        """The composed demo path runs the handler OFF the pump and DEAFENS
        (R2/R3) around its own playback, so the assistant's speech captured WHILE the handler runs is
        suppressed instead of endpointed as a second turn. Exactly one invocation."""
        core = _core()
        clock = [0.0]
        deadline = [0.0]
        deafened = asyncio.Event()
        release = asyncio.Event()
        calls: list = []

        async def handler(clip, t):
            calls.append(t)
            deadline[0] = clock[0] + 100.0   # deafen for the whole playback window (as the demo does)
            deafened.set()
            await release.wait()             # a real turn takes time; the pump keeps draining meanwhile

        async def audio_gen():
            # utterance 1 -> endpoints; then (after the handler deafens) the assistant's own echo speech.
            energies = [0.0] * 2 + [0.9] * 4 + [0.0] * 10
            for i, e in enumerate(energies):
                clock[0] = i * 0.1 + 0.05
                yield (clock[0], _frame(0x40 if e > 0 else 0x00), e)
                await asyncio.sleep(0)
            await deafened.wait()            # echo only exists once we've started speaking (deafened)
            for j, e in enumerate([0.9] * 6 + [0.0] * 10):
                clock[0] = 5.0 + j * 0.1
                yield (clock[0], _frame(0x40 if e > 0 else 0x00), e)
                await asyncio.sleep(0)

        async def go():
            task = asyncio.ensure_future(_collect(
                core, _held_video(release), audio_source=audio_gen(),
                on_utterance=handler, deaf_until=lambda: deadline[0], wall_clock=lambda: clock[0]))
            await asyncio.wait_for(deafened.wait(), timeout=2.0)
            for _ in range(400):
                await asyncio.sleep(0)       # let the echo drain through the (live, deafened) pump
            release.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(go())
        self.assertEqual(len(calls), 1, "the assistant's own echo must not endpoint a second turn")


class StaleAudioTests(unittest.TestCase):
    def test_stale_backlog_dropped_from_vad_path(self):
        """R4: when the pump lags real time past stale_audio_s, the chunk is dropped from the VAD path
        (no endpoint) but still fills the AudioWindow; an 'audio_stale' health event is emitted."""
        core = _core()
        wall = [0.0]
        heard: list = []
        health: list = []
        # A real utterance, but we jump the wall clock forward mid-stream (simulating a stalled pump that
        # is now replaying old capture) so the speech frames read as stale.
        energies = [0.0] * 2 + [0.9] * 5 + [0.0] * 10

        async def audio_gen():
            for i, e in enumerate(energies):
                at = i * 0.1 + 0.05
                # stream advances 0.1/chunk; wall jumps 5s at the first speech frame -> lag >> 2.0s.
                wall[0] = at + (5.0 if e > 0 else 0.0)
                yield (at, _frame(0x40 if e > 0 else 0x00), e)
                await asyncio.sleep(0)

        stats = RuntimeStats()

        video = [(float(i), STILL, "quiet") for i in range(30)]

        async def go():
            await _collect(
                core, video, audio_source=audio_gen(),
                on_utterance=lambda a, t: heard.append(t),
                stale_audio_s=2.0, wall_clock=lambda: wall[0], stats=stats,
                on_health=lambda e: health.append(e.kind))

        asyncio.run(go())
        self.assertEqual(heard, [], "stale (replayed) speech must not endpoint")
        self.assertIn("audio_stale", health)
        self.assertGreater(stats.audio_chunks, 0, "but the AudioWindow was still filled")


class UtteranceTruncationTests(unittest.TestCase):
    """A long turn is handed WHOLE from the endpointer's OWN buffer, not sliced from the shared
    (fused-judge) AudioWindow whose size is the small fused default. Slicing that rolling window
    silently truncated any utterance longer than ``audio_window_s`` to its last N seconds."""

    def test_long_utterance_handed_whole_not_truncated_to_window(self):
        core = _core()
        heard: list = []
        health: list = []
        done = asyncio.Event()
        # 2 silence, 120 speech (=12s @ dt=0.1), 12 trailing silence -> one endpoint. The shared fused
        # AudioWindow default is ~4s; a window-sliced clip carries only its last ~4s (~40 chunks, ~80 B).
        energies = [0.0] * 2 + [0.9] * 120 + [0.0] * 12

        async def handler(clip, t):
            heard.append(clip)

        async def go():
            await asyncio.wait_for(_collect(
                core, _held_video(done), audio_source=_audio_gen(energies, done=done),
                on_utterance=handler, on_health=lambda e: health.append(e.kind)), timeout=5.0)

        asyncio.run(go())
        self.assertEqual(len(heard), 1)
        clip = heard[0]
        self.assertIsInstance(clip, AudioFrame)
        # ~128 100ms chunks survive; a 4s window slice would retain only ~40 chunks.
        self.assertGreater(len(clip.pcm), 3200 * 100)
        self.assertNotIn("utterance_truncated", health)

    def test_over_long_utterance_trips_truncation_health(self):
        """The OWN buffer is bounded by ``utterance_max_s`` (a stuck-open mic must not OOM). When a turn
        exceeds the cap the handed clip is shorter than the utterance's known duration — and that is
        surfaced LOUD via on_health('utterance_truncated'), never silently dropped."""
        core = _core()
        heard: list = []
        health: list = []
        done = asyncio.Event()
        energies = [0.0] * 2 + [0.9] * 40 + [0.0] * 12  # ~4s turn, capped to 1s

        async def go():
            await asyncio.wait_for(_collect(
                core, _held_video(done), audio_source=_audio_gen(energies, done=done),
                on_utterance=lambda a, t: heard.append(a), utterance_max_s=1.0,
                on_health=lambda e: health.append(e.kind)), timeout=5.0)

        asyncio.run(go())
        self.assertEqual(len(heard), 1)
        self.assertIn("utterance_truncated", health)


class PumpWatchdogTests(unittest.TestCase):
    """B3-residual: a raise at the pump COROUTINE level (not inside the on_utterance handler) died
    silently — only the per-utterance task had a done-callback. The pump/autosave tasks now carry
    watchdogs that surface the death through on_health."""

    def test_pump_coroutine_death_surfaces_health(self):
        core = _core()
        health: list = []
        pump_dead = asyncio.Event()
        gate = asyncio.Event()

        async def audio_gen():
            yield (0.05, _frame(0x00), 0.0)
            yield (0.15, _frame(0x00), 0.0)
            raise RuntimeError("pump boom")  # a coroutine-level failure, NOT a handler failure

        def on_health(e):
            health.append(e.kind)
            if e.kind == "audio_pump_died":
                pump_dead.set()

        async def go():
            task = asyncio.ensure_future(_collect(
                core, _held_video(gate), audio_source=audio_gen(), on_health=on_health))
            await asyncio.wait_for(pump_dead.wait(), timeout=2.0)  # times out today: silent death
            gate.set()  # release the video so the run ends cleanly
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(go())
        self.assertIn("audio_pump_died", health)


class EnergyFreshnessTests(unittest.TestCase):
    """A dead/stalled mic leaves ``audio['energy']`` pinned at its last (speech-level) value. A
    WakePolicy that fuses that channel would then keep escalating on phantom audio forever. The frame
    loop must read energy only while FRESH; a stale reading degrades the gate to VIDEO-ONLY."""

    def _wake(self):
        return WakePolicy(threshold=0.5, weights={"audio": 1.0})

    def _fused_core(self):
        # The goal OPTS IN to the fusion. A configured wake_policy no longer overrides every goal's
        # own gate_on (BUG-A-01-04), so a Watch that wants to wake on the fused score says so.
        return _core(gate_on="fused")

    def test_stale_energy_degrades_gate_to_video_only(self):
        # Baseline: no audio at all, STILL video, wake_policy armed -> zero escalations.
        base = RuntimeStats()
        video0 = [(2.0 + i, STILL, "quiet") for i in range(10)]
        asyncio.run(_collect(self._fused_core(), video0, wake_policy=self._wake(), stats=base))

        # Dead mic: two early LOUD chunks then silence-of-death; the video runs seconds later, so the
        # pinned 0.9 energy is stale for every video frame. Without the freshness guard those pinned
        # readings escalate the gate every frame -> judge_calls climbs well above the no-audio baseline.
        stats = RuntimeStats()
        video = [(2.0 + i, STILL, "quiet") for i in range(10)]
        audio = [(0.05, _frame(0x40), 0.9), (0.15, _frame(0x40), 0.9)]  # loud, then the mic dies
        # audio_gate=None isolates the ENERGY→WakePolicy path from the separate acoustic-onset wake.
        asyncio.run(_collect(self._fused_core(), video, audio_source=audio, audio_gate=None,
                             wake_policy=self._wake(), stats=stats))
        self.assertEqual(stats.judge_calls, base.judge_calls,
                         "a dead mic's pinned energy must not keep escalating the gate")

    def test_fresh_energy_still_admits(self):
        """The guard must not be over-aggressive: a live mic whose energy tracks the frame clock stays
        FRESH and still wakes the gate (otherwise the 'fix' silently disables audio — the same trap)."""
        stats = RuntimeStats()
        video = [(i * 0.1, STILL, "quiet") for i in range(1, 12)]
        audio = [(i * 0.1 + 0.02, _frame(0x40), 0.9) for i in range(1, 12)]  # in-step, always fresh
        # audio_gate=None -> only the ENERGY→WakePolicy path can escalate (no acoustic-onset wake).
        asyncio.run(_collect(self._fused_core(), video, audio_source=audio, audio_gate=None,
                             wake_policy=self._wake(), stats=stats))
        self.assertGreater(stats.judge_calls, 0, "fresh loud energy must still wake the gate")

    def test_a_motion_goal_is_no_longer_woken_by_a_wake_policy_it_did_not_ask_for(self):
        # The A-01-04 regression, on the live runtime: the SAME loud, fresh audio and the SAME policy,
        # on the default gate_on="motion" goal. Before, the policy replaced the motion branch for
        # every Watch in the spec and this escalated on every still frame.
        stats = RuntimeStats()
        video = [(i * 0.1, STILL, "quiet") for i in range(1, 12)]
        audio = [(i * 0.1 + 0.02, _frame(0x40), 0.9) for i in range(1, 12)]
        asyncio.run(_collect(_core(), video, audio_source=audio, audio_gate=None,
                             wake_policy=self._wake(), stats=stats))
        self.assertEqual(stats.judge_calls, 0)


class StaleGuardRevalidationTests(unittest.TestCase):
    """After monotonic-clock unification, the ``stale_audio_s`` VAD guard compares a delta-of-deltas
    ((wall-elapsed) - (stream-elapsed) since the first chunk), which is magnitude-robust to the absolute
    value of the chunk timestamp. It must NOT false-fire on a healthy real-time pump."""

    def test_healthy_pump_does_not_false_fire_audio_stale(self):
        core = _core()
        clock = [0.0]
        heard: list = []
        health: list = []
        energies = [0.0] * 2 + [0.9] * 5 + [0.0] * 12
        video = [(float(i), STILL, "quiet") for i in range(len(energies) + 12)]

        async def go():
            await _collect(
                core, video, audio_source=_audio_gen(energies, clock=clock),
                on_utterance=lambda a, t: heard.append(t),
                stale_audio_s=2.0, wall_clock=lambda: clock[0],
                on_health=lambda e: health.append(e.kind))

        asyncio.run(go())
        self.assertEqual(len(heard), 1, "a healthy utterance still endpoints after clock unification")
        self.assertNotIn("audio_stale", health, "a healthy real-time pump must not false-fire audio_stale")


if __name__ == "__main__":
    unittest.main()
