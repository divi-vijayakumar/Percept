#!/usr/bin/env python3
"""Cross-model comparison: detailed scorecard + cost/latency-vs-quality curves.

Directional sweep (N=8 cross-task + N=10 pt-action) — same edge gate, cadence and firing contract on
every arm; only the model swaps. Reads the committed snapshot ``_data.json`` so it reproduces from
tracked files alone; pass ``--refresh`` to rebuild that snapshot from a local
``eval/local-benchmark/runs/`` directory (git-ignored). Writes scorecard.md + 2 PNGs next to this file.

Cost axis = MEASURED candidate_calls; estimated tokens derived with the documented assumptions below.
These are directional numbers, NOT a release claim — see README.md.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SNAPSHOT = os.path.join(HERE, "_data.json")
RUNS = "eval/local-benchmark/runs"  # only used by --refresh; git-ignored / local-only
MODELS = [("gem35", "gemini-3.5-flash", "REST / Gemini", "#1f77b4", "g3.5-flash"),
          ("qwen", "qwen3-vl-32b", "OpenRouter / open", "#2ca02c", "qwen3-vl"),
          ("gemlive", "gemini-3.1-flash-live", "realtime", "#d62728", "g3.1-live")]

# ---- token estimate assumptions (DOCUMENTED — no measured usage available) ----
IMG_TOK = 258           # tokens per image frame (Gemini-documented ~258; used uniformly as a proxy)
PROMPT_TOK = 250        # system+condition prompt
OUT_JUDGE, OUT_PVQA = 80, 220
FRAMES = {"pt-action-e2e": 5, "charades-e2e": 5, "proactiveqa-e2e": 13}  # frame + strip
def tok_per_call(track):
    out = OUT_PVQA if track == "proactiveqa-e2e" else OUT_JUDGE
    return FRAMES.get(track, 5) * IMG_TOK + PROMPT_TOK + out

def load_run(run):
    """Build one run's {track: {cells, cov}} from the git-ignored runs/ dir (refresh path only)."""
    out = {}
    for f in glob.glob(f"{RUNS}/{run}/results/*.json"):
        t = os.path.basename(f)[:-5]
        with open(f) as fh:
            d = json.load(fh)
        c = {(":".join(k) if isinstance(k, tuple) else k): v for k, v in (d.get("cells") or {}).items()}
        out[t] = {"cells": c, "cov": (d.get("coverage") or [{}])[0]}
    return out

def refresh_snapshot():
    data = {}
    for s, *_ in MODELS:
        data[f"sweep2-{s}"] = load_run(f"sweep2-{s}")
        data[f"sweep-{s}"] = load_run(f"sweep-{s}")
    with open(DATA_SNAPSHOT, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"refreshed {DATA_SNAPSHOT} from {RUNS}")
    return data

def cell(store, s, track, key):
    return store.get(s, {}).get(track, {}).get("cells", {}).get(key)

# ================= scorecard.md =================
ALL_METRICS = [
    ("edge:candidate_recall", "edge recall"), ("edge:candidate_precision_proxy", "edge precision (proxy)"),
    ("e2e:notification_recall", "E2E notification recall"), ("e2e:notification_precision", "E2E notification precision"),
    ("e2e:recall", "recall"), ("e2e:precision", "precision"),
    ("e2e:median_latency_s", "median fire latency (s)"), ("e2e:latency", "latency (s)"),
    ("e2e:first_answer_latency_p50", "first-answer latency p50 (s)"),
    ("e2e:candidate_calls", "candidate calls"), ("e2e:calls_per_min", "calls/min"), ("e2e:call_reduction", "call reduction"),
    ("e2e:fires/ep", "fires/episode"), ("e2e:duplicate_rate", "duplicate rate"),
    ("e2e:pauc@0.5_proxy", "pvqa PAUC@0.5"), ("e2e:pauc@0_proxy", "pvqa PAUC@0"), ("e2e:pauc@1_proxy", "pvqa PAUC@1"),
    ("e2e:p-pauc", "p-PAUC"), ("e2e:unresolved_rate", "unresolved rate"),
    ("e2e:early_fire_rate", "early-fire rate"), ("e2e:late_fire_rate", "late-fire rate"),
    ("e2e:timing_accuracy", "timing accuracy"), ("e2e:sparse_span_miou_diagnostic", "sparse-span mIoU (diag)"),
    ("e2e:interruptions", "interruptions"),
]
TRACKS = [("pt-action-e2e", "PT-action"), ("charades-e2e", "Charades"),
          ("proactiveqa-e2e", "ProactiveQA"), ("cuespeak", "CueSpeak"), ("timespeak", "TimeSpeak")]
def fmt(v):
    if isinstance(v, bool): return str(v)
    if isinstance(v, (int, float)): return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:.1f}"
    return "—"

def build_scorecard(N8, N10):
    lines = ["# Full model comparison — detailed scorecard (directional)\n",
             "> **Directional, small sample (N=8 cross-task; N=10 pt-action). Not a release claim.** Same edge gate,",
             "> cadence and firing contract on every arm — only the model swaps. Cost axis = measured `candidate_calls`;",
             f"> estimated tokens use a documented proxy (img~{IMG_TOK} tok, prompt~{PROMPT_TOK}, frames/call "
             f"{FRAMES}, out {OUT_JUDGE}/{OUT_PVQA}).\n"]
    for track, tlabel in TRACKS:
        present = [(s, name) for s, name, *_ in MODELS if track in N8.get(s, {})]
        if not present: continue
        lines.append(f"\n## {tlabel} (N=8) — all metrics\n")
        lines.append("| metric | " + " | ".join(name for _, name in present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for key, mlabel in ALL_METRICS:
            vals = [cell(N8, s, track, key) for s, _ in present]
            if all(v is None for v in vals): continue
            lines.append(f"| {mlabel} | " + " | ".join(fmt(v) for v in vals) + " |")
        covs = [N8[s][track]["cov"] for s, _ in present]
        lines.append("| **coverage (obs/exp, fail)** | " +
                     " | ".join(f"{c.get('observed')}/{c.get('expected')}, {c.get('failed')}" for c in covs) + " |")

    lines.append("\n## pt-action — N=10 (earlier directional run)\n")
    lines.append("| metric | " + " | ".join(name for _, name, *_ in MODELS) + " |")
    lines.append("|---|---|---|---|")
    for key, mlabel in [("edge:candidate_recall", "edge recall"), ("e2e:notification_recall", "E2E recall"),
                        ("e2e:notification_precision", "E2E precision"), ("e2e:median_latency_s", "median latency (s)"),
                        ("e2e:candidate_calls", "candidate calls")]:
        lines.append(f"| {mlabel} | " + " | ".join(fmt(cell(N10, s, "pt-action-e2e", key)) for s, *_ in MODELS) + " |")

    lines.append("\n## Estimated token cost per track (N=8) — total **and per completed clip**\n")
    lines.append("> ⚠️ Compare **per-clip**, not total. g3.1-live's lower *totals* are a coverage artifact (it"
                 " completed 6/8 pt-action, 7/8 charades — fewer clips = less total work, NOT efficiency). Per"
                 " completed clip it is *more* expensive on pt-action; its lower charades cost reflects lower edge"
                 " recall (under-firing), not savings.\n")
    lines.append("| track | tokens/call | " + " | ".join(name for _, name, *_ in MODELS) + " |")
    lines.append("|---|---|---|---|---|")
    for track, tlabel in [("pt-action-e2e", "PT-action"), ("charades-e2e", "Charades"), ("proactiveqa-e2e", "ProactiveQA")]:
        row = [f"| {tlabel} | {tok_per_call(track)} |"]
        for s, *_ in MODELS:
            calls = cell(N8, s, track, "e2e:candidate_calls"); obs = N8.get(s, {}).get(track, {}).get("cov", {}).get("observed")
            if calls is not None and obs:  # a measured 0 calls is a real datum, not a missing cell
                tot = int(calls) * tok_per_call(track) / 1000
                row.append(f" {tot:.0f}k ({tot/obs:.0f}k/clip) |")
            else:
                row.append(" — |")
        lines.append("".join(row))
    with open(f"{HERE}/scorecard.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote scorecard.md")

# ================= curves =================
QKEY = {"pt-action-e2e": ("e2e:notification_recall", "E2E recall"),
        "charades-e2e": ("e2e:notification_recall", "E2E recall"),
        "proactiveqa-e2e": ("e2e:pauc@0.5_proxy", "PAUC@0.5")}
LKEY = {"pt-action-e2e": "e2e:median_latency_s", "charades-e2e": "e2e:median_latency_s",
        "proactiveqa-e2e": "e2e:first_answer_latency_p50"}

def build_curves(N8):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping PNG curves (scorecard.md still written)")
        return

    def scatter(fig_path, xfn, xlabel, title):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6)); fig.suptitle(title, fontsize=13, fontweight="bold")
        for ax, (track, tl) in zip(axs, [("pt-action-e2e", "PT-action"), ("charades-e2e", "Charades"), ("proactiveqa-e2e", "ProactiveQA")]):
            qk, ql = QKEY[track]
            for i, (s, name, _, color, short) in enumerate(MODELS):
                y = cell(N8, s, track, qk); x = xfn(s, track)
                if y is None or x is None:
                    ax.scatter([], [], color=color, label=f"{name} (n/a)"); continue
                ax.scatter([x], [y], s=140, color=color, zorder=3, edgecolor="k", linewidth=0.6, label=name)
                ax.annotate(short, (x, y), textcoords="offset points", xytext=(8, 4 + (i - 1) * 11),
                            fontsize=8, color=color, fontweight="bold")
            ax.set_title(tl); ax.set_xlabel(xlabel); ax.set_ylabel(ql); ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8, loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fig_path, dpi=130); plt.close(fig); print("wrote", os.path.basename(fig_path))

    def cost_per_clip(s, t):  # coverage-normalized: removes the "fewer-clips-completed" artifact
        calls = cell(N8, s, t, "e2e:candidate_calls"); obs = N8.get(s, {}).get(t, {}).get("cov", {}).get("observed")
        return (calls * tok_per_call(t) / obs / 1000) if (calls is not None and obs) else None
    scatter(f"{HERE}/cost-quality.png", cost_per_clip,
            "est tokens per COMPLETED clip (k)  [better = up-left]",
            "Quality vs Cost (tokens per completed clip, coverage-normalized) - higher & cheaper is better")
    scatter(f"{HERE}/latency-quality.png",
            lambda s, t: cell(N8, s, t, LKEY[t]),
            "latency (s)  [better = up-left]", "Quality vs Latency - higher & faster is better")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help=f"rebuild _data.json from {RUNS}/ (git-ignored) before building")
    args = ap.parse_args()
    if args.refresh:
        data = refresh_snapshot()
    else:
        with open(DATA_SNAPSHOT) as fh:
            data = json.load(fh)
    N8 = {s: data[f"sweep2-{s}"] for s, *_ in MODELS if f"sweep2-{s}" in data}
    N10 = {s: data[f"sweep-{s}"] for s, *_ in MODELS if f"sweep-{s}" in data}
    build_scorecard(N8, N10)
    build_curves(N8)
    print("done")

if __name__ == "__main__":
    main()
