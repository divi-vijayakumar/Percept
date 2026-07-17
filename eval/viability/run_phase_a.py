"""Phase-A for-record viability run — architecture isolation (Viability Probe Spec §4, §7).

FAKE JUDGE ONLY. No provider, no API keys, no network, no cost, no real wall-clock. Everything
here is a deterministic architecture measurement under ``FakeClock`` — it exercises Percept's
queue/concurrency/latency/soak machinery isolated from any model. **Nothing in this report is a
product or provider number.** Real recall/precision, real provider latency, and real $/stream-hour
are Phase-B (need the Gemini adapter + key) and are listed as DEFERRED, never fabricated.

Run:  python3 eval/viability/run_phase_a.py
Emits: eval/viability/results/phase_a/<run_id>/{report.md, metrics.json, run_stamp.json}
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

# --- repo wiring: the package + the eval/viability fixtures dir both need to import ---
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "packages" / "percept-vision"))
sys.path.insert(0, str(_REPO / "eval" / "viability"))

import yaml  # noqa: E402

from percept.backends.fake_judge import ErrorInjectionConfig, LatencyProfile, FakeJudge  # noqa: E402
from percept.clock import FakeClock  # noqa: E402
from percept.contracts import JudgmentTiming  # noqa: E402
from percept.harness.load_driver import LoadDriver, LoadProfile, RampSchedule  # noqa: E402
from percept.harness.metrics import CostSample, MetricsCollector  # noqa: E402
from percept.harness.soak_runner import (  # noqa: E402
    SoakConfig,
    SoakRunner,
    assert_clean_recovery,
    compute_trend,
)
from fixtures.core_archetypes import build_core_archetype_fixtures  # noqa: E402

import asyncio  # noqa: E402

CONFIG_PATH = _REPO / "eval" / "viability" / "reference_config.yaml"
RESULTS_ROOT = _REPO / "eval" / "viability" / "results" / "phase_a"
SEED = 20260717  # fixed → replayable


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_locked_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if cfg.get("bars_locked") is not True:
        raise SystemExit(
            "REFUSING to run for-record: reference_config.yaml bars_locked is not true. "
            "The run-validity gate requires the §2 bars locked + committed before results are inspected."
        )
    return cfg


def _pick_first(values, default):
    """Config N/M lists give [min, typical, peak]; use the typical (middle) for the run point."""
    if isinstance(values, list) and values:
        return values[len(values) // 2]
    return default if values in (None, "TODO") else values


def _band(p95: float) -> str:
    return "GREEN" if p95 < 2.0 else ("YELLOW" if p95 < 3.0 else "RED")


def run() -> dict:
    cfg = _load_locked_config()
    lp = cfg["load_profile"]
    nb = cfg["numeric_bars"]
    N = int(_pick_first(lp["concurrent_streams"], 4))
    M = int(_pick_first(lp["watches_per_stream"], 2))
    E = float(lp["admitted_episodes_per_stream_minute"])
    F = int(lp["fan_out_judge_calls_per_episode"])
    baseline_fps = float(nb["naive_vlm_baseline_fps"])
    tol_queue = float(nb["queue_age_slope_tolerance_percent_per_hour"])
    tol_mem = float(nb["memory_slope_tolerance_mb_per_hour"])
    tol_drift = float(nb["latency_drift_tolerance_ms_p95"])
    soak_cap_h = int(nb["soak_endurance_max_hours"])

    profile = LoadProfile(n=N, m=M, e_per_stream_min=E, f=F)
    fixtures = build_core_archetype_fixtures()
    # a positive transition fixture's timeline drives the streams (motion → admission)
    timeline = next(f.timeline for f in fixtures if f.positive)

    results: dict = {"phase": "A", "substrate": "fake_judge (architecture isolation)",
                     "n_m_e_f": {"N": N, "M": M, "E": E, "F": F}, "gates": {}}

    # ---- G3-A latency FLOOR (fake latency dist; NOT the product number) ----------------
    clk = FakeClock()
    judge = FakeJudge(seed=SEED, latency=LatencyProfile(kind="uniform", low_s=0.10, high_s=0.50),
                      simulate_sleep=False)
    driver = LoadDriver(profile, RampSchedule(ramp_s=30.0, steady_s=120.0), timeline,
                        clock=clk, judge=judge)
    stats = asyncio.run(driver.run_to_completion(step_s=1.0))
    mc = MetricsCollector(clock=FakeClock())
    for lat in judge.latencies:
        mc.record_judgment(JudgmentTiming(
            ingest_observed_ts=0.0, admitted_ts=0.0, judge_started_ts=0.0,
            judge_completed_ts=lat, decision_emitted_ts=lat + 0.05))
    lat_report = mc.latency_summary()
    p95 = lat_report.admitted_to_decision.p95
    results["gates"]["G3_latency_floor"] = {
        "admitted_to_decision_p95_s": round(p95, 4),
        "band": _band(p95),
        "label": "ARCHITECTURE FLOOR on a FAKE uniform(0.10,0.50)s latency dist — NOT the product/provider number",
        "note": "real p95 (G3-A) and actionable p95 (G3-B) require the provider — Phase B",
    }

    # ---- G2 econ STRUCTURE: VID vs naive-VLM judgment VOLUME (‌$/call is Phase B) --------
    vid_judgments_per_stream_hour = E * 60.0 * F
    naive_judgments_per_stream_hour = baseline_fps * 3600.0
    reduction = (naive_judgments_per_stream_hour / vid_judgments_per_stream_hour
                 if vid_judgments_per_stream_hour else float("inf"))
    results["gates"]["G2_econ_structure"] = {
        "vid_judgments_per_stream_hour": round(vid_judgments_per_stream_hour, 2),
        "naive_baseline_judgments_per_stream_hour": round(naive_judgments_per_stream_hour, 2),
        "volume_reduction_factor": round(reduction, 2),
        "observed_fan_out_F": round(stats.observed_f, 3),
        "cost_ceiling_usd_per_stream_hour": nb["cost_ceiling_usd_per_stream_hour"],
        "label": "judgment-VOLUME economics (real; VID cuts judge calls vs the 1.0fps baseline). "
                 "$/stream-hour needs real per-call pricing — DEFERRED to Phase B.",
    }

    # ---- G1 boundedness + stability (SoakRunner, simulated hours ≤ soak cap) ------------
    soak_hours = min(6, soak_cap_h)  # simulated; deterministic; well under the 24h cap
    stable = SoakConfig(hours=soak_hours, profile=profile, timeline=timeline, seed=SEED)
    stable_snaps = asyncio.run(SoakRunner(stable, clock=FakeClock()).run())
    trend = compute_trend(stable_snaps)
    max_depth = max(s.max_queue_depth for s in stable_snaps)
    max_age = max(s.max_oldest_pending_age_s for s in stable_snaps)
    # slopes: queue-age as % of hour-1 baseline; memory as MB/hour; latency drift as ms
    h1_age = stable_snaps[0].max_oldest_pending_age_s or 1.0
    queue_pct_per_hr = 100.0 * trend.queue_age_slope_s_per_hour / h1_age
    mem_mb_per_hr = trend.memory_slope_bytes_per_hour / 1e6
    drift_ms = getattr(trend, "latency_p95_drift_s", 0.0) * 1000.0
    g1_stable_pass = (queue_pct_per_hr <= tol_queue and mem_mb_per_hr <= tol_mem and drift_ms <= tol_drift)

    # failure-hour injection → clean recovery
    def _fail_hour(h):  # hour 3 fails 100%
        return ErrorInjectionConfig(rate=1.0, modes=("provider_error",)) if h == 3 else ErrorInjectionConfig(rate=0.0)
    recov_cfg = SoakConfig(hours=min(4, soak_cap_h), profile=profile, timeline=timeline,
                           seed=SEED, errors_fn=_fail_hour)
    recov_snaps = asyncio.run(SoakRunner(recov_cfg, clock=FakeClock()).run())
    clean_recovery = assert_clean_recovery(recov_snaps, failure_hours={3})

    results["gates"]["G1_boundedness"] = {
        "simulated_hours": soak_hours,
        "max_queue_depth": max_depth,
        "max_oldest_pending_age_s": round(max_age, 3),
        "queue_age_slope_pct_per_hour": round(queue_pct_per_hr, 4),
        "memory_slope_mb_per_hour": round(mem_mb_per_hr, 4),
        "latency_p95_drift_ms": round(drift_ms, 4),
        "tolerances": {"queue_pct_per_hr": tol_queue, "mem_mb_per_hr": tol_mem, "drift_ms": tol_drift},
        "stable_within_tolerance": bool(g1_stable_pass),
        "clean_recovery_after_injected_failure_hour": bool(clean_recovery),
        "verdict": "PASS" if (g1_stable_pass and clean_recovery) else "FAIL",
    }

    results["deferred_to_phase_b"] = [
        "real recall/precision per archetype (needs a real judge over the fixtures + golden clips)",
        "real provider latency p95 (G3-A/G3-B product numbers)",
        "real $/stream-hour vs cost ceiling (needs real per-call pricing)",
        "P-JUDGE-SAFE-1 / P-WRAP-1 against real adapters",
    ]
    return results


def _write_artifacts(results: dict) -> Path:
    now = _dt.datetime.now(_dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    run_stamp = {
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "phase": "A",
        "git_sha": "e0eb3df57482b93fc78202b992a48c23ea829c31",  # bars-lock commit
        "config_sha256": _sha256(CONFIG_PATH),
        "fixtures_sha256": _sha256(_REPO / "eval" / "viability" / "fixtures" / "core_archetypes.py"),
        "random_seed": SEED,
        "judge": "FakeJudge (uniform(0.10,0.50)s latency) — architecture isolation, no provider",
        "n_m_e_f": results["n_m_e_f"],
    }
    (out / "run_stamp.json").write_text(json.dumps(run_stamp, indent=2))

    g1 = results["gates"]["G1_boundedness"]
    g3 = results["gates"]["G3_latency_floor"]
    g2 = results["gates"]["G2_econ_structure"]
    md = f"""# Percept Viability — Phase-A For-Record Run

**Run:** `{run_id}` · **git:** `e0eb3df` (bars-lock commit) · **seed:** `{SEED}`
**Substrate:** {results['substrate']} · **Load N/M/E/F:** {results['n_m_e_f']}

> ⚠️ **PHASE A = architecture isolation (fake judge).** Every number below measures Percept's
> queue/concurrency/soak/latency machinery *isolated from any model*. **None of these are product
> or provider numbers.** Real quality, provider latency, and $/stream-hour are Phase-B (need the
> Gemini key) — see *Deferred*.

## G1 — Runtime boundedness  →  **{g1['verdict']}**
- max queue depth: **{g1['max_queue_depth']}** · max oldest-pending age: **{g1['max_oldest_pending_age_s']}s**
- queue-age slope: **{g1['queue_age_slope_pct_per_hour']}%/hr** (tol {g1['tolerances']['queue_pct_per_hr']})
- memory slope: **{g1['memory_slope_mb_per_hour']} MB/hr** (tol {g1['tolerances']['mem_mb_per_hr']})
- latency p95 drift: **{g1['latency_p95_drift_ms']} ms** (tol {g1['tolerances']['drift_ms']})
- stable within tolerance: **{g1['stable_within_tolerance']}** · clean recovery after injected failure hour: **{g1['clean_recovery_after_injected_failure_hour']}**
- simulated hours: {g1['simulated_hours']} (soak cap 24h; simulated via FakeClock)

## G2 — Watch economics (VOLUME structure; $/call deferred)
- VID judgments/stream-hour: **{g2['vid_judgments_per_stream_hour']}** vs naive baseline **{g2['naive_baseline_judgments_per_stream_hour']}**
- **volume reduction: {g2['volume_reduction_factor']}×** · observed fan-out F: {g2['observed_fan_out_F']}
- cost ceiling: ${g2['cost_ceiling_usd_per_stream_hour']}/stream-hour — *real $ needs Phase B*

## G3 — Latency FLOOR (fake dist; NOT the product number)
- admitted→decision p95: **{g3['admitted_to_decision_p95_s']}s** → band **{g3['band']}**
- {g3['label']}

## Deferred to Phase B (needs the provider — NOT faked)
""" + "\n".join(f"- {d}" for d in results["deferred_to_phase_b"]) + "\n"
    (out / "report.md").write_text(md)
    return out


if __name__ == "__main__":
    res = run()
    out = _write_artifacts(res)
    print(f"Phase-A run complete → {out}")
    print(f"  G1 boundedness: {res['gates']['G1_boundedness']['verdict']}")
    print(f"  G3 latency floor: {res['gates']['G3_latency_floor']['band']} "
          f"(p95={res['gates']['G3_latency_floor']['admitted_to_decision_p95_s']}s, fake)")
    print(f"  G2 volume reduction: {res['gates']['G2_econ_structure']['volume_reduction_factor']}× vs naive")
