"""Run arms over a dataset and write a manifest, predictions, metrics and a report.

    python -m benchmarks.run --dataset synthetic --judge fake --out benchmarks/results/dry-run
    python -m benchmarks.run --config benchmarks/configs/e1.json --judge gemini:<model> \\
        --price <in>,<out>,<tokens_per_image> --max-cost-usd 10 --out benchmarks/results/e1-meva

Every arm sees the same frames (checked by digest) and the same condition text. A run that
reaches ``--max-cost-usd`` stops cleanly, writes what it has, marks itself aborted and exits 3.
A case that fails is recorded as skipped with its reason; it never loses the run. Results
directories are ignored by git: footage, labels and results are never committed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import percept
from percept import Condition, Monitor
from percept.judges import base

from benchmarks import arms as A
from benchmarks import datasets, judging, metrics

ARMS = ("percept", "percept-matched", "chunk5", "chunk10", "chunk30", "detector_debounce", "detector_then_vlm")
MAX_LAG_S = A.MAX_LAG_S     # a Percept feed further behind real time than this is not publishable
PRODUCT_REARM_S = Condition("x", "x", kind="event").rearm_s
PRODUCT_BUDGET = inspect.signature(Monitor.__init__).parameters["max_requests_per_min"].default


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=datasets.REPO).stdout.strip()
    except OSError:
        return ""


def _package_hash() -> str:
    """sha256 over the imported percept package's source files: what actually ran."""
    h = hashlib.sha256()
    root = Path(percept.__file__).parent
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def arm_label(arm: str, args) -> str:
    """How an arm is named in tables: a Percept arm says what it changed from the product defaults."""
    if not arm.startswith("percept"):
        return arm
    changed = [] if arm == "percept" else [f"max_requests_per_min {args.percept_budget}"]
    if args.percept_rearm is not None and args.percept_rearm != PRODUCT_REARM_S:
        changed.append(f"rearm standardised to {args.percept_rearm:g} s")
    if arm == "percept":
        return "percept (product defaults" + ("; " + "; ".join(changed) if changed else "") + ")"
    return f"{arm} ({'; '.join(changed)})"


def monitor_config(arm: str, args, judge) -> dict:
    """Everything that configures a Percept arm's ``Monitor`` and conditions."""
    defaults = {k: v.default for k, v in inspect.signature(Monitor.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty and k not in ("store", "trace")}
    if arm == "percept-matched":
        defaults["max_requests_per_min"] = args.percept_budget
    cond = Condition("x", "x", kind="event", **({} if args.percept_rearm is None else {"rearm_s": args.percept_rearm}))
    keep = ("min_confidence", "confirm", "confirm_window_s", "verify", "candidate_deadline_s", "rearm_s",
            "cooldown_s", "on_unverified")
    changed = {}
    if arm == "percept-matched":
        changed["max_requests_per_min"] = {"product_default": PRODUCT_BUDGET, "used": args.percept_budget}
    if cond.rearm_s != PRODUCT_REARM_S:
        changed["condition.rearm_s"] = {"product_default": PRODUCT_REARM_S, "used": cond.rearm_s,
                                        "why": "standardised to the baselines' cooldown for comparability"}
    return dict(defaults, label=arm_label(arm, args), changed_from_product_defaults=changed,
                attempts_reserved_per_request=base.max_attempts(judge), store="MemoryStore",
                watch_fps=None, trigger="default for events (on_change)", verify_mode=cond.verify_mode,
                condition={k: getattr(cond, k) for k in keep})


async def run_arm(arm: str, cases: list, judge, timelines, out: Path, args, foot: dict) -> list:
    """Every case of one arm, ``--parallel`` at a time (Percept in real time, baselines as fast as
    decoding and the judge allow). A case that raises becomes a result with ``error`` set."""
    sem, traces, realtime = asyncio.Semaphore(args.parallel), out / "traces" / arm, arm.startswith("percept")
    if isinstance(judge.inner, judging.OracleJudge):
        judge.inner.realtime = realtime
    if realtime:
        traces.mkdir(parents=True, exist_ok=True)
    missing = sorted({s.path for c in cases for s in c.segments if not Path(s.path).is_file()})
    if missing:     # the media went away (e.g. a drive unmounted): stop loudly, keep finished arms
        raise SystemExit(f"{arm}: {len(missing)} media files are missing, first {missing[0]}")

    async def one(case):
        async with sem:
            if judge.tripped.is_set():
                return A.ArmResult(arm, case.case_id, aborted=True)
            if realtime:
                budget = args.percept_budget if arm == "percept-matched" else None
                return await A.run_percept(case, judge, timelines, traces / f"{case.case_id}.jsonl",
                                           speed=args.speed, fps=args.fps, max_requests_per_min=budget,
                                           rearm_s=args.percept_rearm, arm=arm, stats=foot)
            if arm.startswith("chunk"):
                return await A.run_chunk(case, judge, timelines, float(arm[5:]), cooldown_s=args.cooldown,
                                         fps=args.fps)
            return await A.run_detector(case, judge, timelines, then_vlm=arm == "detector_then_vlm",
                                        cooldown_s=args.cooldown, fps=args.fps)
    try:
        done = await asyncio.gather(*(one(c) for c in cases), return_exceptions=True)
    finally:
        if isinstance(judge.inner, judging.OracleJudge):
            judge.inner.realtime = False
    return [r if isinstance(r, A.ArmResult) else
            A.ArmResult(arm, c.case_id, error=f"{type(r).__name__}: {r}"[:300]) for c, r in zip(cases, done)]


def _fmt(v, pct=False, money=False) -> str:
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    return f"${v:.4f}" if money else (f"{v:.2f}" if isinstance(v, float) else str(v))


def not_publishable(m: dict, table: dict) -> list[str]:
    """Why this run's numbers can't be published (empty when they can)."""
    why = []
    if m["judge"]["spec"] == "fake":
        why.append("the judge is the offline oracle")
    if m["dataset"] != "meva":
        why.append(f"dataset {m['dataset']} is diagnostic only (E1 publishes MEVA)")
    if m["speed"] != 1.0:
        why.append(f"Percept ran at {m['speed']}x real time")
    if m["aborted"]:
        why.append("the spend cap stopped the run")
    if any(s["skipped"] for s in table.values()):
        why.append("some cases were skipped")
    lost = {arm: s["unintended_loss_total"] for arm, s in table.items() if s["unintended_loss_total"]}
    if lost:        # intentional sampling (the fps policies) never disqualifies; unintended loss does
        why.append("frames were lost to decoding, queues or lag (unintended_loss: "
                   + ", ".join(f"{a} {n}" for a, n in lost.items()) + ")")
    lag = max((s["footprint"].get("max_push_lag_s", 0.0) for s in table.values()), default=0.0)
    if lag > MAX_LAG_S:
        why.append(f"the Percept feed fell {lag:.1f} s behind real time")
    return why


def report(manifest: dict, table: dict, digests: dict) -> str:
    m, lines = manifest, []
    lines += [f"# Percept benchmark run `{m['run_id']}`", ""]
    if m["judge"]["spec"] == "fake":
        lines += ["> **Dry run.** The judge is an oracle that answers from ground truth with injected noise. "
                  "These numbers test the harness; they are not results for any model.", ""]
    if m["dataset"] != "meva":
        lines += [f"> **Diagnostic, not publishable.** Dataset `{m['dataset']}` is not the published benchmark "
                  "(MEVA, fixed cameras). Results here locate where Percept wins and loses; they are not a "
                  "published comparison.", ""]
    if m["aborted"]:
        lines += [f"> **Aborted: spend cap reached** (${m['spent_usd']:.4f} of ${m['max_cost_usd']}). "
                  "Cases after the cap are skipped and excluded from every figure; see Coverage.", ""]
    if m["speed"] != 1.0:
        lines += [f"> Percept ran at {m['speed']}x real time. Its budget and timers use the real clock, "
                  "so only 1.0x results are publishable.", ""]
    lines += [f"**Publishable: {'yes' if not m['not_publishable'] else 'no'}**"
              + ("" if not m["not_publishable"] else " (" + "; ".join(m["not_publishable"]) + ")"), ""]
    synth = sorted({c["dataset"] for c in m["cases"] if c["synthetic"]})
    lines += [f"Dataset `{m['dataset']}`: {len(m['cases'])} cases, {m['source_hours']:.3f} source-hours"
              + (f" (declared synthetic: {', '.join(synth)})" if synth else "") + f". Judge `{m['judge']['name']}`"
              f", same model in every arm. Matching tolerance ±{m['tolerance_s']} s; baseline cooldown "
              f"{m['cooldown_s']} s. Git `{m['git_sha'][:10]}`{' (dirty)' if m['git_dirty'] else ''}, "
              f"percept {m['percept']['version']}. Actual spend ${m['spent_usd']:.4f}, from the usage the judge "
              f"reported ({m['requests']} requests).", ""]
    for arm, c in m["monitor_config"].items():
        changed = c["changed_from_product_defaults"]
        lines.append(f"- {c['label']}: {c['max_requests_per_min']} attempts a minute per stream, "
                     f"{c['attempts_reserved_per_request']} reserved per request; rearm {c['condition']['rearm_s']:g} s; "
                     f"verify `{c['verify_mode']}`; min confidence {c['condition']['min_confidence']}. "
                     + ("Changed from the product defaults: " + "; ".join(
                         f"`{k}` {v['used']:g} (product default {v['product_default']:g})" for k, v in changed.items())
                        + "." if changed else "Every setting is the product default."))
    cols = ("arm", "coverage", "recall", "recall ceiling", "precision", "false/h", "dup/occ", "latency p50/p95 s",
            "missed", "cost", "$/monitored h", "$/true event", "requests")
    lines += ["", "## Every arm", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for arm, s in table.items():
        cover = f"{s['cases']} cases, {s['source_hours']:.3f} h, {s['conditions']} conds" \
            + (f", {s['cases_skipped']} skipped" if s["cases_skipped"] else "") + (" (aborted)" if s["aborted"] else "")
        lat = f"{_fmt(s['latency_p50_s'])} / {_fmt(s['latency_p95_s'])}"
        lines.append("| " + " | ".join([
            s["label"], cover, _fmt(s["recall"], True), _fmt(s["recall_ceiling"], True), _fmt(s["precision"], True),
            _fmt(s["false_per_hour"]), _fmt(s["duplicates_per_occurrence"]), lat,
            f"{s['missed']} of {s['occurrences']}", _fmt(s["cost_usd"], money=True),
            _fmt(s["cost_per_hour"], money=True), _fmt(s["cost_per_true_event"], money=True),
            str(s["requests"])]) + " |")
    lines += ["", "The recall ceiling is what a perfect judge could reach given the arm's cooldown or rearm. "
              "Cost is the judge-reported usage of the footage actually watched; $/monitored h divides it by the "
              "hours watched.", "", "## Losses and verification", "",
              "| arm | watched s | budget drops | trigger never fired | verify rejected | unverified | uncertain |",
              "|---|---|---|---|---|---|---|"]
    for arm, s in table.items():
        lines.append(f"| {s['label']} | {s['watched_s']:g} | {s['budget_drops']} | {s['trigger_never_fired']} | "
                     f"{s['verify_rejected']} | {s['unverified']} | {s['uncertain']} |")
    lines += ["", "Budget drops count every look the request budget dropped (the Monitor's counter); "
              "trigger never fired and verify rejected count missed occurrences.", "", "## Misses by stage", "", "| arm | " + " | ".join(metrics.STAGES) + " |",
              "|" + "---|" * (len(metrics.STAGES) + 1)]
    for arm, s in table.items():
        lines.append(f"| {s['label']} | " + " | ".join(str(s["misses_by_stage"][k]) for k in metrics.STAGES) + " |")
    lines += ["", "`budget_dropped` counts misses whose only looks the request budget refused or that expired "
              "waiting for it (Percept traces each dropped look).", "", "## Event survival funnel", "",
              "Each annotated event, followed until the first stage it did not pass. "
              "Counts (share of annotated); – marks a stage the arm does not have.", "",
              "| arm | " + " | ".join(metrics.FUNNEL) + " | matched outside the funnel |",
              "|" + "---|" * (len(metrics.FUNNEL) + 2)]
    for arm, s in table.items():
        fn, total = s["funnel"], s["funnel"]["annotated"] or 1
        cells = ["–" if k in s["funnel_not_applicable"] else f"{fn[k]} ({100 * fn[k] / total:.1f}%)"
                 for k in metrics.FUNNEL]
        lines.append(f"| {s['label']} | " + " | ".join(cells) + f" | {s['funnel_matched_outside']} |")
    lines += ["", "candidate: a look, a dropped look or a detector gate overlapped the event (± tolerance). "
              "admitted: a look was sent. evaluated: the judge answered. positive: a detect `yes` at "
              f"confidence {metrics.MIN_CONFIDENCE} or more. verified: a verification said `yes`. delivered: a "
              "decision matched the event. A look only has to overlap the event, so sparse looks (chunk30's 4 "
              "frames span 30 s) can count as evaluated without showing it. Matched outside the funnel: a "
              "decision matched although no earlier stage overlapped (e.g. a decision whose evidence reached "
              "the event only within the matching tolerance).", "", "## Footprint", "",
              "| arm | CPU s per source-hour | process peak RSS MB | pre-roll frames | intentional sampling | "
              "unintended loss | real-time lag s |", "|---|---|---|---|---|---|---|"]
    detail = lambda d: ", ".join(f"{k} {v}" for k, v in d.items()) or "0"    # noqa: E731
    for arm, s in table.items():
        f = s["footprint"]
        lines.append(f"| {s['label']} | {_fmt(f['cpu_s_per_source_hour'])} | {_fmt(f['peak_rss_mb'])} | "
                     f"{f.get('preroll_frames', '–')} | {detail(s['intentional_sampling'])} | "
                     f"{detail(s['unintended_loss'])} | {f.get('max_push_lag_s', '–')} |")
    lines += ["", "CPU is the harness process while the arm ran (decoding included, "
              f"{m['parallel']} cases at a time), not one stream in isolation. Peak RSS is the process high-water "
              "mark so far; run one arm per invocation for a per-arm peak. Real-time lag is how far the Percept "
              "feed fell behind the clock. Intentional sampling is frames the configured fps policies skipped "
              "(expected); unintended loss is frames lost to decode gaps, the push handoff or lag beyond "
              f"{MAX_LAG_S:g} s, and any of it makes the run not publishable.", ""]
    for name in [a for a in table if a.startswith("percept")]:
        p, worse = table[name], []
        for arm, s in table.items():
            for key, better, kw in (("recall", max, {"pct": True}), ("precision", max, {"pct": True}),
                                    ("false_per_hour", min, {}), ("cost_per_true_event", min, {"money": True}),
                                    ("latency_p50_s", min, {})):
                if not arm.startswith("percept") and None not in (s[key], p[key]) and better(s[key], p[key]) != p[key]:
                    worse.append(f"{arm} beats {name} on {key} ({_fmt(s[key], **kw)} vs {_fmt(p[key], **kw)})")
        lines += [f"## Where {p['label']} loses", ""] + ([f"- {w}" for w in worse] or ["- nowhere in this run"]) + [""]
    lines += ["## Coverage", ""]
    for arm, s in table.items():
        items = s["unsupported"]
        if items:
            lines.append(f"- `{arm}` cannot express {len(items)} case-conditions (not scored for it): "
                         + ", ".join(items[:8]) + (" …" if len(items) > 8 else ""))
    same = all(len(set(d.values())) <= 1 for d in digests.values())
    lines.append("- Identical evidence: yes: every arm decoded the same frames (per-case frame digests are in "
                 "`manifest.json`)." if same else
                 "- Identical evidence: NO; see `manifest.json` (an aborted or failed case stops early).")
    for c in m["skipped"]:
        lines.append(f"- skipped `{c['arm']}` / `{c['case_id']}`: {c['reason']}")
    return "\n".join(lines) + "\n"


async def main_async(args) -> int:
    if not args.dataset:
        raise SystemExit("--dataset is required (or a --config that sets it)")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or f"benchmarks/results/{run_id}")
    out.mkdir(parents=True, exist_ok=True)
    cases = datasets.load(args.dataset, limit=args.limit, root=args.root, negatives=args.negatives,
                          seed=args.seed, workdir=out)
    if not cases:
        raise SystemExit(f"no cases for {args.dataset}: is the data where the adapter looks?")
    by_id = {c.case_id: c for c in cases}
    timelines = judging.Timelines()
    oracle = dict(fn_rate=args.oracle_fn, fp_rate=args.oracle_fp, latency_s=args.oracle_latency, seed=args.seed)
    inner = judging.build(args.judge, cases=cases, timelines=timelines,
                          price=judging.parse_price(args.price) if args.price else None, oracle=oracle)
    judge = judging.CostGuard(inner, args.max_cost_usd)
    arm_names = [a for a in args.arms.split(",") if a]
    bad = [a for a in arm_names if a not in ARMS and not (a.startswith("chunk") and a[5:].replace(".", "").isdigit())]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {ARMS}")
    table, digests, skipped, preds_dir = {}, {}, [], out / "predictions"
    preds_dir.mkdir(exist_ok=True)
    for arm in arm_names:
        foot: dict = {}
        with A.Footprint() as fp:
            results = await run_arm(arm, cases, judge, timelines, out, args, foot)
        s = dict(label=arm_label(arm, args), **metrics.score(results, by_id, args.tolerance))
        hours = s["source_hours"] or 1.0
        s["footprint"] = dict(cpu_s=round(fp.cpu_s, 2), cpu_s_per_source_hour=fp.cpu_s / hours,
                              peak_rss_mb=fp.peak_rss_mb, **foot)
        table[arm] = s
        skipped += [dict(sk, arm=arm) for sk in s["skipped"]]
        for r in results:
            if r.digest:
                digests.setdefault(r.case_id, {})[arm] = r.digest
        with open(preds_dir / f"{arm}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps({"case_id": r.case_id, "aborted": r.aborted, "error": r.error,
                                    "seconds": r.seconds, "uncertain": r.uncertain,
                                    "unverified": r.unverified, "budget_dropped": r.budget_dropped,
                                    "intentional_sampling": r.intentional_sampling,
                                    "unintended_loss": r.unintended_loss,
                                    "preds": [dataclasses.asdict(p) for p in r.preds],
                                    "looks": len(r.looks), "cost_usd": r.cost_usd}) + "\n")
        print(f"{arm}: recall {_fmt(s['recall'], True)} precision {_fmt(s['precision'], True)} "
              f"cost ${s['cost_usd']:.4f}" + (" (aborted)" if s["aborted"] else ""), flush=True)
    import cv2
    manifest = {
        "run_id": run_id, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "percept": {"version": percept.__version__, "path": str(Path(percept.__file__).parent),
                    "package_sha256": _package_hash()},
        "python": sys.version.split()[0], "platform": platform.platform(), "opencv": cv2.__version__,
        "detectors_available": sorted(A.Detectors.available()),
        "dataset": args.dataset, "source_hours": sum(c.duration for c in cases) / 3600,
        "cases": [c.summary() for c in cases], "skipped": skipped, "frame_digests": digests,
        "arms": arm_names, "judge": {"spec": args.judge, "name": judge.name,
                                     "price": dataclasses.asdict(judge.price) if judge.price else None,
                                     "oracle": oracle if args.judge == "fake" else None},
        "monitor_config": {a: monitor_config(a, args, judge) for a in arm_names if a.startswith("percept")},
        "harness": {"clip_frames": A.CLIP_FRAMES, "clip_s": A.CLIP_S, "min_confidence": A.MIN_CONFIDENCE,
                    "onset_before_s": judging.ONSET_BEFORE_S, "onset_after_s": judging.ONSET_AFTER_S,
                    "precog_s": metrics.PRECOG_S, "max_lag_s": MAX_LAG_S},
        "seeds": {"seed": args.seed}, "config": args.config, "tolerance_s": args.tolerance, "cooldown_s": args.cooldown,
        "fps": args.fps, "speed": args.speed, "parallel": args.parallel, "max_cost_usd": args.max_cost_usd,
        "spent_usd": judge.spent, "requests": judge.requests, "refused_by_cap": judge.refused,
        "aborted": judge.tripped.is_set(),
    }
    manifest["not_publishable"] = not_publishable(manifest, table)
    manifest["publishable"] = not manifest["not_publishable"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (out / "metrics.json").write_text(json.dumps(table, indent=1))
    (out / "report.md").write_text(report(manifest, table, digests))
    print(f"wrote {out}/report.md" + (" (ABORTED: spend cap reached)" if manifest["aborted"] else ""))
    return 3 if manifest["aborted"] else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m benchmarks.run", description=__doc__.splitlines()[0])
    p.add_argument("--config", help="a JSON file of defaults, e.g. benchmarks/configs/e1.json")
    p.add_argument("--dataset",
                   help="perception-test | perception-test-sparse | meva | local:<dir> | synthetic")
    p.add_argument("--root", help="dataset root (default: the adapter's staging path)")
    p.add_argument("--limit", type=int, help="at most this many cases")
    p.add_argument("--negatives", type=int, default=2, help="absent labels watched per case")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--judge", default="fake", help="fake | gemini:<model>")
    p.add_argument("--price", help="in,out,tokens_per_image: USD per million tokens, and tokens per image")
    p.add_argument("--max-cost-usd", type=float, help="hard cap for the whole run (reserved before each request)")
    p.add_argument("--out")
    p.add_argument("--run-id")
    p.add_argument("--tolerance", type=float, default=2.0, help="matching tolerance, seconds")
    p.add_argument("--cooldown", type=float, default=A.COOLDOWN_S, help="baseline cooldown, seconds")
    p.add_argument("--percept-budget", type=int, default=12,
                   help="max_requests_per_min of the percept-matched arm (12: chunk5's request rate)")
    p.add_argument("--percept-rearm", type=float, default=None,
                   help="rearm_s for the Percept arms (default: the product default)")
    p.add_argument("--fps", type=float, default=5.0, help="frames sampled per second, every arm")
    p.add_argument("--speed", type=float, default=1.0, help="Percept arm pacing; only 1.0 is publishable")
    p.add_argument("--parallel", type=int, default=8, help="cases run at once, per arm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oracle-fn", type=float, default=0.1)
    p.add_argument("--oracle-fp", type=float, default=0.02)
    p.add_argument("--oracle-latency", type=float, default=1.0)
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.config:                     # the config sets defaults; flags on the command line still win
        config = json.loads(Path(args.config).read_text())
        if "arm_judges" in config:
            raise SystemExit(f"{args.config}: {config.get('status', 'per-arm judges are not supported')}")
        p.set_defaults(**{k: v for k, v in config.items() if k in vars(args)})
        args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
