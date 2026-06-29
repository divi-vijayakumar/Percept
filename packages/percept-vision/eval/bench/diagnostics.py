"""Health trace — funnel the pipeline's SILENT failures into a structured, logged trace.

The benchmark floored repeatedly on things that produced a number instead of an error: a degraded pose
provider that quietly emitted 5 reps, an amplitude gate that was a no-op because its threshold was 10-80x
below the real signal, a foreshortened pose whose normalizer hit its floor and inflated every amplitude, a
VLM that answered "not enough evidence" on every frame. None raised; all corrupted the score.

`HealthTrace` collects typed `Issue`s (and logs them), so a run SAYS when it's degraded instead of
silently scoring wrong. `counting_health()` is the rule set for the rhythm/counting path; the same
collector is meant to carry pose / VLM / decode issues too.

    PYTHONPATH=packages/percept-vision \
        python -m eval.bench.diagnostics       # demo: the trace over the local N=8 counting set
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

log = logging.getLogger("percept.health")

Severity = str  # "error" | "warn" | "info"


@dataclass
class Issue:
    code: str          # stable machine code, e.g. "pose.degraded"
    severity: Severity
    message: str       # human-readable, with the numbers that triggered it
    data: dict = field(default_factory=dict)


@dataclass
class HealthTrace:
    """A run's accumulated health signals. Push issues as you find them; serialize into the result trace."""

    scope: str = ""                       # what this trace is about, e.g. "counting:stu10_5"
    issues: list = field(default_factory=list)

    def note(self, code: str, severity: Severity, message: str, **data) -> None:
        issue = Issue(code=code, severity=severity, message=message, data=data)
        self.issues.append(issue)
        lvl = {"error": logging.ERROR, "warn": logging.WARNING, "info": logging.INFO}.get(severity, logging.INFO)
        log.log(lvl, "[%s] %s: %s", self.scope or "health", code, message)

    @property
    def degraded(self) -> bool:
        """True if any error/warn fired — the run's number should not be trusted at face value."""
        return any(i.severity in ("error", "warn") for i in self.issues)

    def as_dict(self) -> dict:
        return {"scope": self.scope, "degraded": self.degraded,
                "issues": [asdict(i) for i in self.issues]}


# --- thresholds (named, swept offline — not magic numbers buried in branches) --------------------------
TAU_KP_CONF = 0.5         # mean keypoint confidence below this => degraded provider
FRAC_NO_DETECTION = 0.30  # >this fraction of frames with no person => degraded provider / wrong subject
FRAC_SCALE_FLOORED = 0.20 # >this fraction of frames where body_scale hit its floor => inflated amplitudes
GATE_INERT_RATIO = 0.10   # min_amp below this fraction of the median rep amplitude => the gate is a no-op
TAU_REGULARITY = 0.55     # selected-channel periodicity below this => low-confidence (phantom / continuous)


def counting_health(
    trace: HealthTrace,
    *,
    n_frames: int,
    frac_no_detection: float,
    mean_kp_conf: float,
    frac_scale_floored: float,
    selected: str | None,
    selected_regularity: float,
    rep_amplitudes: list[float],
    min_amp: float,
    per_channel_counts: dict[str, int],
    expected_rom: float | None,
) -> HealthTrace:
    """Turn one counting run's measured signals into flagged silent-failures."""
    if n_frames == 0:
        trace.note("decode.empty", "error", "0 frames decoded — wrong video path or unreadable stream")
        return trace

    if frac_no_detection > FRAC_NO_DETECTION:
        trace.note("pose.no_detection", "warn",
                   f"{frac_no_detection:.0%} of frames had no person (subject lost / provider degraded)",
                   frac_no_detection=round(frac_no_detection, 3))
    if mean_kp_conf < TAU_KP_CONF:
        trace.note("pose.low_confidence", "warn",
                   f"mean keypoint confidence {mean_kp_conf:.2f} < {TAU_KP_CONF} — degraded keypoints "
                   f"(the n1-fresh '5' signature); count is untrustworthy",
                   mean_kp_conf=round(mean_kp_conf, 3))
    if frac_scale_floored > FRAC_SCALE_FLOORED:
        trace.note("pose.scale_floored", "warn",
                   f"{frac_scale_floored:.0%} of frames hit the body-scale floor — torso normalizer is "
                   f"degenerate, channel amplitudes are INFLATED (phantom large reps)",
                   frac_scale_floored=round(frac_scale_floored, 3))

    # the amplitude-gate no-op: expected_rom is in a unit far below the observed amplitudes
    if expected_rom is not None and rep_amplitudes:
        med = sorted(rep_amplitudes)[len(rep_amplitudes) // 2]
        if med > 0 and min_amp < GATE_INERT_RATIO * med:
            trace.note("gate.inert", "warn",
                       f"amplitude gate is a NO-OP: min_amp={min_amp:.3f} is <{GATE_INERT_RATIO:.0%} of the "
                       f"median rep amplitude {med:.2f} (expected_rom unit mismatch) — it rejects nothing",
                       min_amp=round(min_amp, 4), median_rep_amplitude=round(med, 3))

    if selected is not None and selected_regularity < TAU_REGULARITY:
        trace.note("count.low_regularity", "warn",
                   f"selected channel '{selected}' regularity {selected_regularity:.2f} < {TAU_REGULARITY} — "
                   f"weak periodicity (phantom reps or continuous/occluded motion); should escalate or abstain",
                   selected=selected, regularity=round(selected_regularity, 3))

    if per_channel_counts and selected is not None:
        sel_c = per_channel_counts.get(selected, 0)
        top = max(per_channel_counts.values())
        if sel_c and top > 2 * sel_c:
            trace.note("count.channel_disagreement", "info",
                       f"channels disagree {dict(per_channel_counts)} — selection ambiguous (selected "
                       f"'{selected}'={sel_c}, max={top}); count is selection-sensitive",
                       per_channel=dict(per_channel_counts))
    return trace


def from_run(
    scope: str,
    *,
    tracker,
    n_frames: int,
    frac_no_detection: float = 0.0,
    mean_kp_conf: float = 1.0,
    frac_scale_floored: float = 0.0,
    expected_rom: float | None = None,
    amp_gate_frac: float = 0.5,
) -> HealthTrace:
    """Build a counting HealthTrace from a finished ``RhythmTracker`` + accumulated pose stats.

    The tracker-side signals (selected channel, regularity, per-channel counts, amplitudes) are always
    available; the pose-side stats (no-detection / kp-confidence / scale-floored fractions) are optional —
    pass them from the loop when the eye exposes them (``PoseChannels._last_*``)."""
    sel = tracker.selected if tracker else None
    channels = dict(tracker._channels) if tracker else {}
    per_ch = {name: len(s.reps) for name, s in channels.items()}
    sel_st = channels.get(sel) if sel else None
    amps = [r.amplitude for r in sel_st.reps] if sel_st else []
    reg = sel_st.regularity() if sel_st else 0.0
    min_amp = (expected_rom * amp_gate_frac) if expected_rom else 0.0
    ht = HealthTrace(scope=scope)
    counting_health(ht, n_frames=n_frames, frac_no_detection=frac_no_detection, mean_kp_conf=mean_kp_conf,
                    frac_scale_floored=frac_scale_floored, selected=sel, selected_regularity=reg,
                    rep_amplitudes=amps, min_amp=min_amp, per_channel_counts=per_ch, expected_rom=expected_rom)
    return ht


# --- demo over the local N=8 (cached poses) -----------------------------------------------------------
def _run_one(vid, action, gt):
    from eval.bench.hero_ab import EXPECTED_ROM, FPS, MIN_CONF, VLM_PRIOR
    from eval.bench.pose_cache import load_detections, reconstruct
    from percept import Goal, Percept
    from percept.harness.pose import (_MIN_BODY_SCALE, body_scale, channel_coverage, hip_centre,
                                       pose_channels, select_nearest_hip)

    prior = VLM_PRIOR.get(action)
    erom = EXPECTED_ROM.get(prior) if prior else None
    cache = load_detections(vid, FPS, "rtmo") or []
    n = len(cache)
    no_det = kpconf = floored = 0
    kpc_n = 0
    core = Percept.create(discover_plugins=False)
    st = {"type": "rhythm", "count": 0}
    if prior:
        st["rhythm"] = {"channel": prior, "expected_rom": erom, "expected_rom_space": "channel_native"}
    core.add_goal(Goal(id="g", condition="rep", say="rep", state=st))
    anchor = last = None
    for fr in cache:
        dets = [d for d in reconstruct(fr["dets"]) if d.confidence >= MIN_CONF]
        if not dets:
            no_det += 1
        ch = select_nearest_hip(reconstruct(fr["dets"]), anchor, MIN_CONF)
        if ch is not None:
            anchor = hip_centre(ch)
            last = pose_channels(ch.keypoints)
            if body_scale(ch.keypoints) <= _MIN_BODY_SCALE * 1.01:
                floored += 1
            cov = channel_coverage(ch.keypoints)
            kpconf += cov.get(prior, sum(cov.values()) / len(cov)) if prior else sum(cov.values()) / len(cov)
            kpc_n += 1
        if last:
            core.perceive_rhythm("g", t=fr["t"], channels=last)
    tr = core.rhythm.get("g")
    sel = tr.selected if tr else None
    per_ch = {name: len(s.reps) for name, s in (tr._channels.items() if tr else [])}
    sel_st = tr._channels.get(sel) if (tr and sel) else None
    amps = [r.amplitude for r in sel_st.reps] if sel_st else []
    reg = sel_st.regularity() if sel_st else 0.0
    min_amp = (erom * 0.5) if (erom and prior) else 0.0  # default amp_gate_frac=0.5
    ht = HealthTrace(scope=f"counting:{vid}")
    counting_health(ht, n_frames=n, frac_no_detection=(no_det / n if n else 0.0),
                    mean_kp_conf=(kpconf / kpc_n if kpc_n else 0.0),
                    frac_scale_floored=(floored / kpc_n if kpc_n else 0.0),
                    selected=sel, selected_regularity=reg, rep_amplitudes=amps, min_amp=min_amp,
                    per_channel_counts=per_ch, expected_rom=erom)
    return tr.count if tr else 0, ht


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from eval.bench.hero_ab import MANIFEST

    print("\n  HEALTH TRACE over the N=8 counting set (cached poses)\n  " + "=" * 70)
    for vid, action, gt in MANIFEST:
        count, ht = _run_one(vid, action, gt)
        verdict = "OK" if abs(count - gt) <= 1 else ("DEGRADED" if ht.degraded else "WRONG")
        print(f"\n  {vid} {action} GT={gt} -> count={count}  [{verdict}]")
        for i in ht.issues:
            print(f"      • {i.severity.upper():5s} {i.code}: {i.message}")
        if not ht.issues:
            print("      (no issues flagged)")


if __name__ == "__main__":  # pragma: no cover
    main()
