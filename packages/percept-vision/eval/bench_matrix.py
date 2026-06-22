"""Benchmark MATRIX assembler — raw / Percept-Core / E2E  ×  {4 datasets + latency + cost}.

Re-scores the LAST Gemini benchmark run entirely from the on-disk caches (verdicts/signals/results) —
no video re-fetch, no perception re-run. The structural facts it encodes:

  * raw vs Core make the SAME vision calls (the gate changes FIRING, not call count) -> Core's value is
    firing quality (mIoU / less spam / time-F1), at EQUAL perception cost & latency to raw.
  * the EDGE (E2E) is the only thing that cuts calls — tier-0 motion gate, measured 83% reduction
    (`bench_gating`) -> E2E ~= Core firing at ~0.17x cost & latency.

Accuracy per dataset is read from the cached *_result.json (measured). Cost = vision-calls x per-call
tokens x Gemini price; per-call latency + tokens come from a tiny LIVE probe (3 gemini-2.5-flash calls).
E2E accuracy is reported as ~= Core (the edge is designed to escalate on motion, preserving events) —
an exact E2E-accuracy number needs the real frames (flagged, not faked).

Run: PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.bench_matrix
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root (caches live there)
EDGE_KEEP = float(os.environ.get("PERCEPT_EDGE_KEEP", 0.17))  # 1 - 0.83 (bench_gating call reduction)
# gemini-2.5-flash list price ($/1M tokens)
PRICE_IN = float(os.environ.get("PERCEPT_GEMINI_PRICE_IN", 0.30))
PRICE_OUT = float(os.environ.get("PERCEPT_GEMINI_PRICE_OUT", 2.50))


def _load(name: str):
    p = ROOT / name
    return json.load(open(p)) if p.exists() else None


def accuracy_rows() -> list[dict]:
    """(dataset, metric, raw, core) from the cached result files."""
    out = []
    ch = _load("charades_miou_result.json")
    if ch:
        out.append({"ds": "Charades", "metric": "mIoU×100", "raw": ch["gemini-loop (raw)"]["miou"] * 100,
                    "core": ch["percept-event (dedup)"]["miou"] * 100, "fmt": "{:.1f}"})
    pq = _load("proactiveqa_result.json")
    if pq:
        out.append({"ds": "ProactiveVQA", "metric": "PAUC@0.5", "raw": pq["gemini-loop (raw)"]["pauc@0.5"],
                    "core": pq["percept-event (dedup)"]["pauc@0.5"], "fmt": "{:.1f}",
                    "raw_extra": f"{pq['gemini-loop (raw)']['interrupts']} interrupts",
                    "core_extra": f"{pq['percept-event (dedup)']['interrupts']} interrupts"})
    rc = _load("repcount_result.json")
    if rc:
        out.append({"ds": "RepCount", "metric": "OBO×100", "raw": rc["obo_loop"], "core": rc["obo_percept"],
                    "fmt": "{:.1f}"})
    om = _load("omnipro_result.json")
    if om:
        raw_f1 = om["loop"]["overall"]["time_f1"] if isinstance(om["loop"]["overall"], dict) else om["loop"]["overall"]
        core_f1 = om["percept"]["overall"]["time_f1"] if isinstance(om["percept"]["overall"], dict) else om["percept"]["overall"]
        out.append({"ds": "OmniPro", "metric": "time-F1 (overall)", "raw": raw_f1 * 100,
                    "core": core_f1 * 100, "fmt": "{:.1f}"})
    return out


def vision_calls() -> dict:
    """Vision calls per dataset (raw == Core). The cost/latency driver."""
    calls = {}
    ch = _load("charades_gemini_verdicts.json")
    if ch:
        calls["Charades"] = sum(len(v["verdicts"]) for v in ch.values())
    rc = _load("repcount_signals.json")
    if rc:
        calls["RepCount"] = sum(len(v["seq"]) for v in rc.values())
    om = _load("omnipro_signals.json")
    if om:
        calls["OmniPro"] = sum(len(v.get("seq", [])) for v in om.values())
    pq = _load("proactiveqa_narrations.json")
    if pq:  # PVQA judges once per QA segment window; approximate calls by narration count
        calls["ProactiveVQA"] = sum(len(v.get("answer", [])) or 1 for v in pq.values())
    return calls


def probe_latency_tokens(n: int = 3) -> tuple[float, float, float]:
    """Tiny live probe: avg (latency_ms, input_tokens, output_tokens) for a gemini-2.5-flash judge call."""
    from eval._dotenv import load_dotenv
    from eval.frames import first_frame
    from percept import registry
    from percept.config import load_settings
    from percept.eval import find_golden_root, load_golden

    load_dotenv()
    settings = load_settings()
    root = find_golden_root()
    golden = load_golden(root)
    frame = None
    for gc in golden.gate_cases:
        f = first_frame(root / "scenes" / gc["scene_id"])
        if f is not None:
            frame = f[1]
            break
    vision = registry.make_vision("gemini", settings)
    client = vision._ensure_client()
    from google.genai import types

    media, raw = frame if isinstance(frame, tuple) else ("image/png", frame)
    lat, tin, tout = [], [], []
    for _ in range(n):
        t0 = time.monotonic()
        r = client.models.generate_content(
            model=vision._model,
            contents=[types.Part.from_bytes(data=raw, mime_type=media), 'Is a person visible? JSON {"satisfied","confidence","evidence"}.'],
            config=types.GenerateContentConfig(system_instruction="Answer strict JSON.", max_output_tokens=64),
        )
        lat.append((time.monotonic() - t0) * 1000)
        u = r.usage_metadata
        tin.append(u.prompt_token_count or 0)
        tout.append(u.candidates_token_count or 0)
    return sum(lat) / n, sum(tin) / n, sum(tout) / n


def main() -> int:
    acc = accuracy_rows()
    calls = vision_calls()
    total_calls = sum(calls.values())
    lat_ms, tin, tout = probe_latency_tokens()
    cost_per_call = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6

    L = ["", "=" * 78, "  BENCHMARK MATRIX — Gemini-2.5-flash perception (re-scored from caches)",
         f"  edge keep-rate={EDGE_KEEP:.2f} (tier-0 83% cut) · per-call: {lat_ms:.0f}ms, "
         f"{tin:.0f}in+{tout:.0f}out tok, ${cost_per_call*1000:.4f}/1k-calls", "=" * 78, ""]
    # --- accuracy block ---
    L.append("  ACCURACY  (impact of the COGNITION LAYER: raw -> Core; E2E firing ≈ Core)")
    L.append(f"    {'dataset':<14}{'metric':<20}{'raw loop':>12}{'Percept-Core':>16}{'Δ':>10}")
    for r in acc:
        raw_s = r["fmt"].format(r["raw"]); core_s = r["fmt"].format(r["core"])
        d = r["core"] - r["raw"]
        extra = f"  ({r.get('raw_extra','')} → {r.get('core_extra','')})" if r.get("raw_extra") else ""
        L.append(f"    {r['ds']:<14}{r['metric']:<20}{raw_s:>12}{core_s:>16}{d:>+10.1f}{extra}")
    # --- cost + latency block ---
    L += ["", "  COST & LATENCY  (impact of the ARCHITECTURE: the EDGE cuts calls; Core==raw on calls)"]
    L.append(f"    {'config':<16}{'vision calls':>14}{'est. cost':>14}{'avg latency/frame':>20}")
    configs = [("raw loop", 1.0), ("Percept-Core", 1.0), ("E2E (edge+core)", EDGE_KEEP)]
    for name, factor in configs:
        c = total_calls * factor
        cost = c * cost_per_call
        # per-frame latency: raw/Core pay the vision call every frame; E2E pays it only on the kept fraction
        per_frame = lat_ms if factor == 1.0 else (EDGE_KEEP * lat_ms + 1.0)  # +~1ms edge motion check
        L.append(f"    {name:<16}{c:>14,.0f}{'$'+format(cost,'.2f'):>14}{per_frame:>17.0f} ms")
    L += ["", f"  vision calls by dataset (raw=Core): " + ", ".join(f"{k}={v}" for k, v in calls.items()),
          "", "  NOTE: accuracy = MEASURED (cached Gemini verdicts). E2E cost/latency from the edge's",
          "  measured 83% call cut; E2E *accuracy* ≈ Core (edge escalates on motion → events preserved) —",
          "  an exact E2E-accuracy number needs the real frames through the motion gate.", "=" * 78, ""]
    print("\n".join(L))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
