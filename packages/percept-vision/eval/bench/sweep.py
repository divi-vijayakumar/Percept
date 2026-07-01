"""Phase-0 sweep harness — measure a metric curve across a parameter grid, so constants are picked from a
set, never tuned against the N=1 scorecard.

First instance: the rhythm **amplitude-gate fraction** (``amp_gate_frac``) over the local N=8 repcount hero
set (cached RTMO poses — no video decode, fast & deterministic). It runs the committed single-subject path
(Arm A) at each gate fraction and reports OBO/MAE, plus a prior-less baseline (the scorecard's regime).

    PYTHONPATH=packages/percept-vision python -m eval.bench.pose_cache    # once, if cache is missing
    PYTHONPATH=packages/percept-vision python -m eval.bench.sweep         # amp_gate_frac sweep
    PYTHONPATH=packages/percept-vision python -m eval.bench.sweep 0 0.15 0.2 0.25 0.3 0.5
"""
from __future__ import annotations

import sys

from eval.bench.hero_ab import EXPECTED_ROM, FPS, MIN_CONF, MANIFEST, VLM_PRIOR
from eval.bench.pose_cache import load_detections, reconstruct


def arm_a_count(cache_frames, prior, erom, frac) -> int:
    """Committed single-subject count off cached poses with an injectable ``amp_gate_frac``.

    ``prior is None`` reproduces the prior-less (unsupervised) scorecard regime. ``frac == 0`` turns the
    absolute amplitude gate OFF (min_amp = 0). Mirrors :func:`eval.bench.hero_ab._arm_a`."""
    from percept import Goal, Percept
    from percept.harness.pose import hip_centre, pose_channels, select_nearest_hip

    state: dict = {"type": "rhythm", "count": 0}
    if prior is not None:
        rhythm: dict = {"channel": prior}
        if erom is not None:
            rhythm["expected_rom"] = erom
            rhythm["expected_rom_space"] = "channel_native"
        if frac is not None:
            rhythm["amp_gate_frac"] = frac
        state["rhythm"] = rhythm

    core = Percept.create(discover_plugins=False)
    core.add_goal(Goal(id="g", condition="a completed repetition", say="rep", state=state))
    anchor = last_ch = None
    for fr in cache_frames:
        chosen = select_nearest_hip(reconstruct(fr["dets"]), anchor, MIN_CONF)
        if chosen is not None:
            anchor = hip_centre(chosen)
            last_ch = pose_channels(chosen.keypoints)
        if last_ch:
            core.perceive_rhythm("g", t=fr["t"], channels=last_ch)
    tr = core.rhythm.get("g")
    return tr.count if tr is not None else 0


def _obo(errs):
    return sum(e <= 1 for e in errs) / len(errs) if errs else 0.0


def _nmae(rows):  # normalized |pred-gt|/max(gt,1), matching the repcount harness
    return sum(abs(c - gt) / max(gt, 1) for c, gt in rows) / len(rows) if rows else 0.0


def main(argv: list[str]) -> int:
    fracs = [float(a) for a in argv] if argv else [0.0, 0.1, 0.2, 0.3, 0.5]
    caches = {vid: load_detections(vid, FPS, "rtmo") for vid, _a, _g in MANIFEST}
    missing = [v for v, c in caches.items() if c is None]
    if missing:
        print(f"missing pose cache for {missing} — run `python -m eval.bench.pose_cache` first")
        return 1

    print(f"\n  amp_gate_frac sweep · N={len(MANIFEST)} · primed (per-action channel prior) · fps={FPS}")
    print("  " + "=" * 78)
    header = "  {:10s}{:13s}{:>4}".format("clip", "action", "gt")
    header += "".join(f"{('f=' + str(f)):>7}" for f in fracs) + f"{'unsup':>7}"
    print(header)
    print("  " + "-" * 78)

    cols: dict = {f: [] for f in fracs}
    unsup: list = []
    for vid, action, gt in MANIFEST:
        prior = VLM_PRIOR.get(action)
        erom = EXPECTED_ROM.get(prior) if prior else None
        row = f"  {vid:10s}{action:13s}{gt:>4}"
        for f in fracs:
            c = arm_a_count(caches[vid], prior, erom, f)
            cols[f].append((c, gt))
            row += f"{c:>7}"
        u = arm_a_count(caches[vid], None, None, None)  # prior-less (scorecard regime)
        unsup.append((u, gt))
        row += f"{u:>7}"
        print(row)

    print("  " + "-" * 78)
    print(f"  {'OBO / NMAE (all N=8)':<27}", end="")
    for f in fracs:
        print(f"  f={f}: {_obo([abs(c - g) for c, g in cols[f]]):.2f}/{_nmae(cols[f]):.2f}", end="")
    print(f"   unsup: {_obo([abs(c - g) for c, g in unsup]):.2f}/{_nmae(unsup):.2f}")
    print("  " + "=" * 78)
    print("  (lower NMAE / higher OBO is better; current default amp_gate_frac=0.5)\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
