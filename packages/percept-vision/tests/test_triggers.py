import json

import numpy as np
import pytest

import percept
from percept.contracts import Condition, Frame, Trigger
from percept.triggers import (
    GATE_H,
    GATE_W,
    Every,
    OnChange,
    QualityGate,
    SceneMonitor,
    block_max,
    gate_frame,
    sharpness,
    smear_fraction,
)

SCALE = 10  # test frames are 480x360, ten times the gate frame
RIGHT_HALF = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
gy, gx = np.mgrid[0:GATE_H, 0:GATE_W]


def texture(k: int = 0) -> np.ndarray:
    """A deterministic textured scene at gate resolution, values 60..160."""
    return (60 + (gx * 7 + gy * 13 + k * 31 + (gx * gy) % 17) % 101).astype(np.float64)


def upscale(g: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.clip(g, 0, 255).astype(np.uint8), SCALE, 0), SCALE, 1)


def frame(g: np.ndarray, t: float = 0.0, seq: int = 0) -> Frame:
    return Frame("cam", 0, seq, t, 1_000.0 + t, upscale(g))


def with_patch(g, y0, y1, x0, x1, delta):
    out = g.copy()
    out[y0:y1, x0:x1] += delta
    return np.clip(out, 0, 255)


def cond(cid="c", **kw) -> Condition:
    return Condition(cid, "something happens", **kw)


def run(trigger, frames, conditions, state=None, t0=0.0, dt=0.2):
    """Step ``trigger`` over ``frames``; returns the list of Due per frame and the final state."""
    dues = []
    for i, g in enumerate(frames):
        now = t0 + i * dt
        due, state = trigger.step(frame(g, now, i), conditions, state, now)
        dues.append(dict(due))
    return dues, state


def fired(dues, cid="c"):
    return [i for i, d in enumerate(dues) if cid in d]


# --------------------------------------------------------------------------- gate frame


def test_gate_frame_area_averages_any_size():
    g = texture()
    out = gate_frame(frame(g))
    assert out.shape == (GATE_H, GATE_W) and out.dtype == np.float32
    assert np.array_equal(out, g.astype(np.float32))
    odd = np.full((101, 77), 90, dtype=np.uint8)
    assert np.allclose(gate_frame(odd), 90.0, atol=1e-3)
    tiny = np.full((5, 7), 200, dtype=np.uint8)
    assert np.allclose(gate_frame(tiny), 200.0, atol=1e-3)


def test_gate_frame_bgr_to_gray():
    bgr = np.zeros((36, 48, 3), dtype=np.uint8)
    bgr[..., 2] = 100  # red
    assert np.allclose(gate_frame(bgr), 29.9, atol=1e-3)


# --------------------------------------------------------------------------- on_change


def test_factories_and_protocol():
    assert isinstance(percept.on_change(), OnChange) and isinstance(percept.on_change(), Trigger)
    assert isinstance(percept.every(30), Every) and isinstance(percept.every(30), Trigger)
    Condition("x", "a door is open", trigger=percept.on_change(threshold=0.3))
    with pytest.raises(ValueError):
        percept.on_change(threshold=0.2, exit_threshold=0.3)
    with pytest.raises(ValueError):
        percept.every(0)


def test_static_scene_never_fires():
    dues, _ = run(OnChange(), [texture()] * 30, (cond(),))
    assert fired(dues) == []


def test_localized_change_in_zone_fires_once():
    c = cond(zones=(RIGHT_HALF,))
    changed = with_patch(texture(), 12, 18, 30, 36, 120)
    frames = [texture()] * 5 + [changed] * 15
    dues, state = run(OnChange(), frames, (c,))
    assert fired(dues) == [5]
    assert dues[5]["c"].startswith("change 0.")
    assert any(state["conditions"]["c"]["active"])  # still settling, so those blocks can't refire yet


def test_refires_only_after_the_change_settles():
    c = cond(zones=(RIGHT_HALF,))
    first = with_patch(texture(), 12, 18, 30, 36, 120)
    second = with_patch(first, 24, 30, 40, 46, -120)
    frames = [texture()] * 5 + [first] * 80 + [second] * 5
    dues, _ = run(OnChange(), frames, (c,))
    assert fired(dues) == [5, 85]


def test_change_outside_zone_or_in_exclude_zone_does_not_fire():
    left_change = with_patch(texture(), 12, 18, 6, 12, 120)
    right_change = with_patch(texture(), 12, 18, 30, 36, 120)
    in_zone = cond("zoned", zones=(RIGHT_HALF,))
    excluded = cond("excluded", exclude_zones=(RIGHT_HALF,))
    whole = cond("whole")
    conditions = (in_zone, excluded, whole)
    dues, _ = run(OnChange(), [texture()] * 3 + [left_change] * 3, conditions)
    assert fired(dues, "zoned") == [] and fired(dues, "excluded") == [3] and fired(dues, "whole") == [3]
    dues, _ = run(OnChange(), [texture()] * 3 + [right_change] * 3, conditions)
    assert fired(dues, "zoned") == [3] and fired(dues, "excluded") == [] and fired(dues, "whole") == [3]


def test_block_max_fires_where_the_whole_frame_mean_barely_moves():
    base = texture()
    changed = with_patch(base, 0, 6, 0, 6, 90)  # exactly one block, no clipping
    mean = float(np.abs(changed - base).mean()) / 255
    assert mean < 0.01
    assert block_max(base, changed) == pytest.approx(90 / 255)
    dues, _ = run(OnChange(), [base] * 3 + [changed], (cond(),))
    assert fired(dues) == [3]


def test_slow_lighting_drift_is_absorbed():
    frames = [texture() + 0.4 * i for i in range(100)]  # +40 grey levels over 20 s
    dues, state = run(OnChange(), frames, (cond(),))
    assert fired(dues) == []
    assert state["conditions"]["c"]["score"] < 0.05
    # The same total shift in one step is a change.
    dues, _ = run(OnChange(), [texture()] * 3 + [texture() + 40], (cond(),))
    assert fired(dues) == [3]


def test_conditions_added_and_removed():
    trig = OnChange()
    changed = with_patch(texture(), 12, 18, 30, 36, 120)
    _, state = run(trig, [texture()] * 3, (cond("a"),))
    due, state = trig.step(frame(changed, 1.0), (cond("b"),), state, 1.0)
    assert set(due) == {"b"} and set(state["conditions"]) == {"b"}


# --------------------------------------------------------------------------- every


def test_every_fires_on_schedule():
    trig = Every(10)
    a, b = cond("a"), cond("b")
    times = [0.0, 5.0, 9.99, 10.0, 15.0, 20.5, 30.0]
    state, log = None, []
    for i, t in enumerate(times):
        conditions = (a,) if t < 15 else (a, b)
        due, state = trig.step(frame(texture(), t, i), conditions, state, t)
        log.append(dict(due))
    assert log[0] == {"a": "heartbeat"}
    assert log[1] == {} and log[2] == {}
    assert log[3] == {"a": "every 10s"}
    assert log[4] == {"b": "heartbeat"}
    assert set(log[5]) == {"a"} and set(log[6]) == {"b"}  # each on its own schedule


def test_every_clock_going_backwards_is_due():
    trig = Every(10)
    _, state = trig.step(frame(texture(), 100.0), (cond(),), None, 100.0)
    due, _ = trig.step(frame(texture(), 5.0), (cond(),), state, 5.0)
    assert "c" in due


# --------------------------------------------------------------------------- quality


def sharp_image(k: int = 0) -> np.ndarray:
    y, x = np.mgrid[0:64, 0:64]
    return ((x * x * 31 + y * 17 + x * y * 7 + k * 13) % 256).astype(np.uint8)


def smeared_image() -> np.ndarray:
    img = sharp_image()
    for top in range(0, 64, 8):
        img[top:top + 8] = img[top]  # every row of the block copies the block's first row
    return img


def blurred_image() -> np.ndarray:
    y, x = np.mgrid[0:64, 0:64]
    return (20 + 0.5 * x + 1.0 * y).astype(np.uint8)  # smooth: no edges, not a smear


def qframe(img, i=0):
    return Frame("cam", 0, i, float(i), float(i), img)


def test_quality_primitives():
    assert smear_fraction(sharp_image().astype(np.float32)) < 0.05
    assert smear_fraction(smeared_image().astype(np.float32)) == 1.0
    assert smear_fraction(blurred_image().astype(np.float32)) < 0.05
    assert sharpness(blurred_image()) < 0.25 * sharpness(sharp_image())


def test_quality_vetoes_smeared_and_blurred_frames():
    gate = QualityGate()
    _, state = gate.check(qframe(sharp_image()), None, 0.0)
    verdict, state = gate.check(qframe(smeared_image()), state, 0.0)
    assert verdict == percept.triggers.QualityVerdict(ok=False, reason="corrupt", degraded=False)
    # Blur is relative to this stream, so it's only judged after 30 good frames.
    verdict, state = gate.check(qframe(blurred_image()), state, 0.0)
    assert verdict.ok
    for i in range(30):
        verdict, state = gate.check(qframe(sharp_image(i)), state, float(i))
        assert verdict.ok
    verdict, state = gate.check(qframe(blurred_image()), state, 31.0)
    assert not verdict.ok and verdict.reason == "blurry"


def test_degraded_hysteresis():
    gate = QualityGate()
    state, flags = None, []
    # Corruption is counted when it appears, so bad frames alternate with good ones.
    seq = [sharp_image(i) for i in range(10)] + [smeared_image(), sharp_image()] * 7 + [
        sharp_image(i) for i in range(60)]
    bad = set(range(10, 24, 2))
    for i, img in enumerate(seq):
        verdict, state = gate.check(qframe(img, i), state, float(i))
        assert verdict.ok == (i not in bad), i
        flags.append(verdict.degraded)
    # 7 bad out of 23 (30.4%) turns it on; it stays on until bad <= 10% of the last 60.
    on = flags.index(True)
    assert on == 22
    off = flags.index(False, on)
    assert off == 70  # the first bad frame leaves the window: 6 of 60
    assert all(flags[on:off]) and not any(flags[off:])


# --------------------------------------------------------------------------- scene reset


def drive(scene, trig, frames, conditions, dt=0.25):
    """What the runtime does: a reset suppresses triggers and drops their state; when it ends,
    every condition gets one look at the settled scene."""
    s_state = t_state = None
    log = []
    for i, g in enumerate(frames):
        now = i * dt
        f = frame(g, now, i)
        signal, s_state = scene.step(f, s_state, now)
        if signal.suppressed:
            t_state = None
            log.append((signal, None))
            continue
        due, t_state = trig.step(f, conditions, t_state, now)
        due = dict(due)
        if signal.ended:
            due.update({c.id: "scene settled" for c in conditions})
        log.append((signal, due))
    return log


def test_scene_reset_suppresses_then_rebuilds():
    inverted = 255 - texture()
    later = with_patch(inverted, 12, 18, 30, 36, -120)
    frames = [texture()] * 10 + [inverted] * 20 + [later] * 3
    log = drive(SceneMonitor(), OnChange(), frames, (cond(),))
    signals = [s for s, _ in log]
    assert [i for i, s in enumerate(signals) if s.reset] == [10]
    assert signals[10].changed > 0.6
    suppressed = [i for i, s in enumerate(signals) if s.suppressed]
    assert suppressed == list(range(10, 18))  # 2 s at 4 fps
    assert [i for i, s in enumerate(signals) if s.ended] == [18]
    dues = [d for _, d in log]
    assert dues[18] == {"c": "scene settled"}  # one look at the settled scene
    assert all(not d for i, d in enumerate(dues[:30]) if d is not None and i != 18)  # nothing else
    assert "c" in dues[30]  # the rebuilt baseline is the new scene; a real change still fires


def test_without_scene_monitor_the_flip_would_fire():
    dues, _ = run(OnChange(), [texture()] * 10 + [255 - texture()] * 2, (cond(),))
    assert fired(dues) == [10]


def test_shifted_image_is_a_reset_and_moving_camera_extends():
    frames = [texture()] * 5 + [np.roll(texture(), 3 * (i + 1), axis=1) for i in range(15)] + [
        np.roll(texture(), 45, axis=1)] * 15
    log = drive(SceneMonitor(), OnChange(), frames, (cond(),))
    signals = [s for s, _ in log]
    assert signals[5].reset
    assert all(s.suppressed for s in signals[5:20])  # still panning: the window keeps extending
    assert [i for i, s in enumerate(signals) if s.ended] == [27]  # 2 s after the last wholesale change
    assert sum(s.reset for s in signals) == 1


def test_local_change_is_not_a_reset():
    frames = [texture()] * 5 + [with_patch(texture(), 0, 18, 0, 24, 120)] * 3  # a quarter of the frame
    signals = [s for s, _ in drive(SceneMonitor(), OnChange(), frames, (cond(),))]
    assert not any(s.reset or s.suppressed for s in signals)


# --------------------------------------------------------------------------- state and determinism


def plain(state):
    """A JSON round trip: what a Store would persist."""
    return json.loads(json.dumps(state, default=lambda o: o.tolist()))


def test_state_round_trips_through_plain_data():
    c = cond(zones=(RIGHT_HALF,))
    changed = with_patch(texture(), 12, 18, 30, 36, 120)
    head = [texture() + 0.3 * i for i in range(8)]
    tail = [changed] * 5 + [texture(1)] * 5

    for trig in (OnChange(), Every(0.5)):
        _, state = run(trig, head, (c,))
        a, _ = run(trig, tail, (c,), state=state, t0=8 * 0.2)
        b, _ = run(trig, tail, (c,), state=plain(state), t0=8 * 0.2)
        assert a == b and any(a)

    scene, quality = SceneMonitor(), QualityGate()
    s_state = q_state = None
    for i, g in enumerate(head):
        _, s_state = scene.step(frame(g, i * 0.2), s_state, i * 0.2)
        _, q_state = quality.check(qframe(sharp_image(i)), q_state, float(i))
    s_copy, q_copy = plain(s_state), plain(q_state)
    for i, g in enumerate([255 - texture()] * 3):
        now = (8 + i) * 0.2
        assert scene.step(frame(g, now), s_state, now)[0] == scene.step(frame(g, now), s_copy, now)[0]
    imgs = [smeared_image(), sharp_image(3)]
    for img in imgs:
        va, q_state = quality.check(qframe(img), q_state, 9.0)
        vb, q_copy = quality.check(qframe(img), q_copy, 9.0)
        assert va == vb


def test_step_does_not_mutate_its_input_state():
    _, state = run(OnChange(), [texture()] * 3, (cond(),))
    snapshot = plain(state)
    OnChange().step(frame(texture(2), 1.0), (cond(),), state, 1.0)
    assert plain(state) == snapshot


def test_deterministic():
    frames = [texture(i % 3) for i in range(20)]
    assert run(OnChange(), frames, (cond(),))[0] == run(OnChange(), frames, (cond(),))[0]


# --------------------------------------------------------------------------- review regressions


def test_persistent_blur_is_not_vetoed_forever():
    gate, state = QualityGate(), None
    for i in range(30):
        _, state = gate.check(qframe(sharp_image(i), i), state, float(i))
    verdicts = []
    for i in range(300):
        v, state = gate.check(qframe(blurred_image(), 30 + i), state, 30.0 + i)
        verdicts.append(v)
    assert sum(not v.ok for v in verdicts) <= 31  # a streak past 30 frames becomes the new normal
    assert all(v.ok for v in verdicts[-200:]) and not verdicts[-1].degraded


def test_dimmer_but_focused_scene_is_not_blurry():
    gate, state = QualityGate(), None
    for i in range(30):
        _, state = gate.check(qframe(sharp_image(i), i), state, float(i))
    dim = (sharp_image(5) // 8).astype(np.uint8)  # lights dimmed: same structure, an eighth of the contrast
    v, _ = gate.check(qframe(dim, 31), state, 31.0)
    assert v.ok


def test_fence_or_blinds_are_not_corrupt():
    y, x = np.mgrid[0:64, 0:64]
    fence = np.where(x % 4 < 2, 200, 50).astype(np.uint8)  # rows identical, columns vary: smear-shaped
    gate, state = QualityGate(), None
    for i in range(20):
        v, state = gate.check(qframe(fence, i), state, float(i))
        assert v.ok, i
    # Sudden concealment still counts.
    _, state = gate.check(qframe(sharp_image(), 20), state, 20.0)
    v, state = gate.check(qframe(smeared_image(), 21), state, 21.0)
    assert v.reason == "corrupt"


def test_close_subject_is_a_trigger_not_a_reset():
    person = with_patch(texture(), 0, GATE_H, 6, 42, 90)  # 75% of the frame, clear of the side borders
    frames = [texture()] * 10 + [person] * 10 + [texture()] * 3
    log = drive(SceneMonitor(), OnChange(), frames, (cond(),))
    assert not any(s.reset or s.suppressed for s, _ in log)
    assert [i for i, (_, d) in enumerate(log) if d and "c" in d][0] == 10


def test_reset_ends_with_one_look_for_every_condition():
    conditions = (cond("a"), cond("b", zones=(RIGHT_HALF,)))
    frames = [texture()] * 10 + [255 - texture()] * 20
    log = drive(SceneMonitor(), OnChange(), frames, conditions)
    looks = [(i, set(d)) for i, (_, d) in enumerate(log) if d]
    assert looks == [(18, {"a", "b"})]


def test_slow_pan_keeps_the_window_open():
    # After the reset the picture keeps changing over about 40% of its blocks each step: less than
    # the reset bar, more than half of it.
    flipped = 255 - texture()
    frames = [texture()] * 5 + [flipped] + [with_patch(flipped, 0, GATE_H, 0, 20, 60 * (i % 2 == 0))
                                            for i in range(12)]
    signals = [s for s, _ in drive(SceneMonitor(), OnChange(), frames, (cond(),))]
    assert signals[5].reset
    assert all(0.3 < s.changed < 0.6 for s in signals[6:13])
    assert all(s.suppressed for s in signals[5:18])


def test_second_event_in_another_block_fires_during_absorption():
    door = with_patch(texture(), 0, 12, 0, 6, 90)
    person = with_patch(door, 24, 30, 36, 42, 90)
    frames = [texture()] * 3 + [door] * 5 + [person] * 3  # the person arrives 1 s after the door opens
    dues, _ = run(OnChange(), frames, (cond(),))
    assert fired(dues) == [3, 8]


def test_low_light_scene_still_fires():
    dark = 5 + (texture() - 60) * 15 / 100  # grey 5..20
    frames = [dark] * 5 + [with_patch(dark, 12, 18, 12, 18, 30)] * 2
    dues, _ = run(OnChange(), frames, (cond(),))
    assert fired(dues) == [5]
