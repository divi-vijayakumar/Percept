"""Edge↔harness selected-contract parity + edge latency.

Compares the explicitly listed detector boundaries, motion scores, audio energy, and ordered-audio
result fields between the JS percept-edge route and Python percept-harness route on identical inputs.
Passing means those selected fields conform for these frozen vectors; it is not a byte-for-byte
serialization proof or a claim about untested downstream precision/recall. Also measures edge frame
latency (JS Edge.processFrame throughput).

Run (node auto-discovered via PATH or ~/.nvm; or set PERCEPT_NODE):
    PYTHONPATH=packages/percept-vision \
        python3.12 packages/percept-edge/parity/compare.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from percept.adapters import AudioFrame, Box
from percept.events import AudioBoundary, AudioUnit, AudioWindow
from percept.harness.detectors import AcousticOnset, MotionPeriodicity
from percept.harness.tier0 import ego_residual_score, motion_score
from percept.runtime.audio import pcm_energy

HERE = Path(__file__).resolve().parent
AUDIO_FIXTURE = json.loads(
    (HERE.parent / "test" / "fixtures" / "audio_stream.golden.json").read_text()
)

# --- deterministic test cases (must produce identical output in both languages) ---
MOTION_SIGNALS = [
    [0, 1, 0, 1, 0, 1, 0],
    [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0],
    [0, 0.01, 0, 0.015, 0, 0.005, 0],                       # jitter below min_height -> no peaks
    [0, 0.3, 0.9, 0.4, 0.1, 0.7, 0.2, 0.05, 0.6, 0.1],      # noisy / irregular
]
ENERGY_SIGNALS = [
    [0.0, 0.5, 0.5, 0.0, 0.5],
    [0.0, 0.5, 0.2, 0.5, 0.2],                              # hysteresis: no re-fire while above low
    [0.0, 0.9, 0.9, 0.9, 0.9],                              # sustained -> single onset
    [0.0, 0.4, 0.0, 0.31, 0.0, 0.29, 0.0],                  # straddles the high threshold
]
W = H = 8
FRAME_PAIRS = [
    (W, H, [0] * 64, [0] * 64),                             # identical -> 0
    (W, H, [0] * 64, [255] * 64),                           # inverted -> 1
    (W, H, [0] * 64, [128] * 64),                           # half
    (W, H, list(range(64)), [(i * 3) % 256 for i in range(64)]),  # arbitrary
]


def _patch(base, w, px, py, pw, ph, val):
    """A copy of `base` (flat row-major) with a `pw x ph` block at `(px, py)` set to `val`."""
    out = list(base)
    for r in range(py, py + ph):
        for c in range(px, px + pw):
            out[r * w + c] = val
    return out


def _grad(w, h):
    return [(x * 5 + y * 3) % 256 for y in range(h) for x in range(w)]


def _pan(base, w, h, dx):
    """`base` shifted left by `dx` px (edge column clamped) — a pure horizontal camera pan."""
    out = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            sx = x + dx
            sx = 0 if sx < 0 else (w - 1 if sx >= w else sx)
            out[y * w + x] = base[y * w + sx]
    return out


# Mode A (localized statistic): a 6x6 event on a still 48x36 background — block_max/p95 catch it,
# whole-frame mean does not. Plus a non-divisible 13x11 case to lock the block-boundary math, and a
# region+block compose. Every case must be byte-for-byte identical Python<->JS.
_BG = [20] * (48 * 36)
_LOC = _patch(_BG, 48, 0, 0, 6, 6, 200)
_NDA = [(i * 7) % 256 for i in range(13 * 11)]
_NDB = [(i * 5 + 9) % 256 for i in range(13 * 11)]
SALIENCE_PAIRS = [
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "mean"},        # same input, mean stays low...
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "block_max"},   # ...block_max lights the event
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "p95"},
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "stat": "block_max"},  # non-divisible boundary lock
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "stat": "p95"},
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "block_max",    # ROI + block compose
     "region": {"x": 0.0, "y": 0.0, "w": 0.25, "h": 0.25}},
]

# Mode B (ego compensation): a pure pan residual ~0 while its raw mean is high; a localized subject the
# shift cannot explain away (residual preserved); a non-divisible noise pair.
_GRAD = _grad(48, 36)
EGO_PAIRS = [
    {"w": 48, "h": 36, "a": _GRAD, "b": _pan(_GRAD, 48, 36, 2), "k": 4},       # pan -> ~0
    {"w": 48, "h": 36, "a": _GRAD, "b": _patch(_GRAD, 48, 10, 10, 6, 6, 255), "k": 4},  # subject preserved
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "k": 3},                          # noise, non-divisible
]
LATENCY = {"width": 32, "height": 32, "frames": 5000}


def py_detector(cls, signal):
    det = cls()
    out = []
    for t, v in enumerate(signal):
        for b in det.push(float(t), v):
            out.append({"detector": b.detector, "kind": b.kind, "t": b.t, "value": b.value})
    return out


def run_js(cases):
    node = os.environ.get("PERCEPT_NODE") or shutil.which("node")
    if node is None:
        nvm = Path.home() / ".nvm/versions/node"
        cands = sorted(nvm.glob("*/bin/node")) if nvm.exists() else []
        node = str(cands[-1]) if cands else None
    if node is None:
        sys.exit("node not found (set PERCEPT_NODE or add node to PATH)")
    p = subprocess.run(
        [node, str(HERE / "edge_runtime.mjs")], input=json.dumps(cases),
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"node failed:\n{p.stderr}")
    return json.loads(p.stdout)


def boundaries_equal(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x["detector"] != y["detector"] or x["kind"] != y["kind"]:
            return False
        if abs(x["t"] - y["t"]) > tol or abs(x["value"] - y["value"]) > tol:
            return False
    return True


def main():
    cases = {
        "motionSignals": MOTION_SIGNALS,
        "energySignals": ENERGY_SIGNALS,
        "framePairs": [{"w": w, "h": h, "a": a, "b": b} for (w, h, a, b) in FRAME_PAIRS],
        "saliencePairs": SALIENCE_PAIRS,
        "egoPairs": EGO_PAIRS,
        "audioEnergyVectors": AUDIO_FIXTURE["energy_vectors"],
        "audioTimelineVectors": AUDIO_FIXTURE["timeline_vectors"],
        "latency": LATENCY,
    }
    js = run_js(cases)

    py_motion = [py_detector(MotionPeriodicity, s) for s in MOTION_SIGNALS]
    py_onset = [py_detector(AcousticOnset, s) for s in ENERGY_SIGNALS]
    py_scores = [motion_score((w, h, tuple(a)), (w, h, tuple(b))) for (w, h, a, b) in FRAME_PAIRS]
    py_salience = [
        motion_score(
            (c["w"], c["h"], tuple(c["a"])), (c["w"], c["h"], tuple(c["b"])),
            region=Box(**c["region"]) if c.get("region") else None,
            stat=c.get("stat", "mean"),
        )
        for c in SALIENCE_PAIRS
    ]
    py_ego = [
        ego_residual_score((c["w"], c["h"], tuple(c["a"])), (c["w"], c["h"], tuple(c["b"])), k=c["k"])
        for c in EGO_PAIRS
    ]
    py_audio_energy = [
        pcm_energy(struct.pack(f"<{len(c['samples'])}h", *c["samples"]))
        for c in AUDIO_FIXTURE["energy_vectors"]
    ]
    py_audio_timelines = []
    for case in AUDIO_FIXTURE["timeline_vectors"]:
        timeline = AudioWindow(seconds=10)
        results = []
        for row in case["units"]:
            boundaries = tuple(
                AudioBoundary(boundary["kind"], boundary["t"])
                for boundary in row.get("boundaries", ())
            )
            unit = AudioUnit(
                seq=row["seq"],
                t_start=row["t_start"],
                t_end=row["t_end"],
                chunk=AudioFrame(
                    pcm=b"\x00\x00",
                    sample_rate=row.get("sample_rate", 16_000),
                    channels=row.get("channels", 1),
                ),
                sample_rate=row.get("sample_rate"),
                channels=row.get("channels"),
                direction=row.get("direction", "unknown"),
                participant_id=row.get("participant_id"),
                boundaries=boundaries,
            )
            results.append(timeline.push_unit(unit).as_dict())
        py_audio_timelines.append({
            "results": results,
            "order": [unit.seq for unit in timeline.units()],
            "directions": [unit.direction for unit in timeline.units()],
            "participants": [unit.participant_id for unit in timeline.units()],
            "boundaryKinds": [boundary.kind for boundary in timeline.boundaries()],
        })

    checks = []
    for i, (pj, jj) in enumerate(zip(py_motion, js["motionBoundaries"])):
        checks.append((f"motion-periodicity[{i}]  ({len(pj)} peaks)", boundaries_equal(pj, jj)))
    for i, (pj, jj) in enumerate(zip(py_onset, js["onsetBoundaries"])):
        checks.append((f"acoustic-onset[{i}]     ({len(pj)} onsets)", boundaries_equal(pj, jj)))
    for i, (ps, jsc) in enumerate(zip(py_scores, js["motionScores"])):
        checks.append((f"motion-gate score[{i}]   = {ps:.6f}", abs(ps - jsc) <= 1e-9))
    for i, (ps, jsc) in enumerate(zip(py_salience, js["saliences"])):
        stat = SALIENCE_PAIRS[i].get("stat", "mean")
        roi = "+roi" if SALIENCE_PAIRS[i].get("region") else ""
        checks.append((f"salience[{i}] {stat}{roi} = {ps:.6f}", abs(ps - jsc) <= 1e-9))
    for i, (ps, jsc) in enumerate(zip(py_ego, js["egoScores"])):
        checks.append((f"ego-residual[{i}] (k={EGO_PAIRS[i]['k']}) = {ps:.6f}", abs(ps - jsc) <= 1e-9))
    for i, (ps, jsc) in enumerate(zip(py_audio_energy, js["audioEnergies"])):
        name = AUDIO_FIXTURE["energy_vectors"][i]["name"]
        checks.append((f"audio-energy[{i}] {name} = {ps:.6f}", abs(ps - jsc) <= 1e-12))
    for i, (py_timeline, js_timeline) in enumerate(
        zip(py_audio_timelines, js["audioTimelines"])
    ):
        name = AUDIO_FIXTURE["timeline_vectors"][i]["name"]
        same = (
            py_timeline["order"] == js_timeline["order"]
            and py_timeline["directions"] == js_timeline["directions"]
            and py_timeline["participants"] == js_timeline["participants"]
            and py_timeline["boundaryKinds"] == js_timeline["boundaryKinds"]
        )
        if same:
            keys = (
                "accepted", "duplicate", "reordered", "missing", "gap_s", "overlap_s",
                "format_changed", "expired", "expired_bytes", "evicted", "evicted_bytes",
            )
            for py_result, js_result in zip(py_timeline["results"], js_timeline["results"]):
                for key in keys:
                    left, right = py_result[key], js_result[key]
                    if isinstance(left, float):
                        same = same and abs(left - right) <= 1e-12
                    else:
                        same = same and left == right
        checks.append((f"audio-timeline[{i}] {name}", same))

    print("=" * 72)
    print("  EDGE <-> HARNESS SELECTED-CONTRACT PARITY   (JS vs Python)")
    print("=" * 72)
    allok = True
    for name, ok in checks:
        allok = allok and ok
        print(f"   [{'MATCH' if ok else 'DIFF!'}]  {name}")
    print("-" * 72)
    print(
        f"   {len(checks)} checks - "
        f"{'ALL SELECTED FIELDS MATCH ON FROZEN VECTORS' if allok else 'MISMATCH FOUND'}"
    )
    lat = js.get("latency")
    if lat:
        print("-" * 72)
        print(f"  EDGE LATENCY  (JS Edge.processFrame, {lat['width']}x{lat['height']} grayscale, "
              f"motion-periodicity + gate active):")
        print(f"   {lat['frames']:,} frames in {lat['totalMs']:.1f} ms  ->  "
              f"{lat['perFrameUs']:.2f} us/frame  ~  {lat['fps']:,} fps")
    print("=" * 72)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
