"""Edge↔harness parity + edge latency.

Proves the JS @percept/edge route ≡ the Python percept-harness route on identical inputs (same frozen
contract, same detector algorithms) — the rigorous form of "do the edge benchmarks match the harness
route": if every gate decision and boundary is identical, downstream precision / recall / proactivity
are identical *by construction*, because the only difference between the two routes is WHERE the gate
runs, not WHAT it decides. Also measures edge frame latency (JS Edge.processFrame throughput).

Run (node auto-discovered via PATH or ~/.nvm; or set PERCEPT_NODE):
    PYTHONPATH=packages/percept-harness:packages/percept-vision \
        python3.12 packages/percept-edge/parity/compare.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from percept_harness.detectors import AcousticOnset, MotionPeriodicity
from percept_harness.tier0 import motion_score

HERE = Path(__file__).resolve().parent

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
        "latency": LATENCY,
    }
    js = run_js(cases)

    py_motion = [py_detector(MotionPeriodicity, s) for s in MOTION_SIGNALS]
    py_onset = [py_detector(AcousticOnset, s) for s in ENERGY_SIGNALS]
    py_scores = [motion_score((w, h, tuple(a)), (w, h, tuple(b))) for (w, h, a, b) in FRAME_PAIRS]

    checks = []
    for i, (pj, jj) in enumerate(zip(py_motion, js["motionBoundaries"])):
        checks.append((f"motion-periodicity[{i}]  ({len(pj)} peaks)", boundaries_equal(pj, jj)))
    for i, (pj, jj) in enumerate(zip(py_onset, js["onsetBoundaries"])):
        checks.append((f"acoustic-onset[{i}]     ({len(pj)} onsets)", boundaries_equal(pj, jj)))
    for i, (ps, jsc) in enumerate(zip(py_scores, js["motionScores"])):
        checks.append((f"motion-gate score[{i}]   = {ps:.6f}", abs(ps - jsc) <= 1e-9))

    print("=" * 72)
    print("  EDGE <-> HARNESS PARITY   (@percept/edge JS  vs  percept-harness Python)")
    print("=" * 72)
    allok = True
    for name, ok in checks:
        allok = allok and ok
        print(f"   [{'MATCH' if ok else 'DIFF!'}]  {name}")
    print("-" * 72)
    print(f"   {len(checks)} checks - "
          f"{'ALL MATCH: the two routes are byte-for-byte equivalent' if allok else 'MISMATCH FOUND'}")
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
