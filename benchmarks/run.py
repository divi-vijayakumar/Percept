"""Run arms over a dataset and write a manifest, raw results, metrics and a report.

    python -m benchmarks.run --dataset synthetic --judge fake --out benchmarks/results/dry-run
    python -m benchmarks.run --config benchmarks/configs/e1.json --judge gemini:<model> \\
        --price <in>,<out>,<tokens_per_image> --max-cost-usd 10 --out benchmarks/results/e1-meva
    python -m benchmarks.run --rescore benchmarks/results/e1-meva [--root <dataset root>]

``raw/<arm>.jsonl`` keeps everything scoring needs; ``--rescore`` rebuilds ``metrics.json`` and
``report.md`` from it without calling any judge.

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
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import percept
from percept import Condition, Monitor
from percept.clips import PreRoll
from percept.judges import base

from benchmarks import arms as A
from benchmarks import datasets, judging, metrics

ARMS = ("percept", "percept-matched", "chunk5", "chunk10", "chunk30", "detector_debounce", "detector_then_vlm")
MAX_LAG_S = A.MAX_LAG_S     # a Percept feed further behind real time than this is not publishable
PRODUCT_REARM_S = Condition("x", "x", kind="event").rearm_s
PRODUCT_BUDGET = inspect.signature(Monitor.__init__).parameters["max_requests_per_min"].default
PRODUCT_CONCURRENCY = inspect.signature(Monitor.__init__).parameters["max_concurrent_requests"].default


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


def worker_threads(args) -> int:
    """The shared worker pool (JPEG encoding, keyframes, trace flushes): enough for every
    request slot of every case run at once, so no case waits on another's encoding."""
    return args.parallel * PRODUCT_CONCURRENCY + 4


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
    if arm == "percept-matched" and args.percept_budget != PRODUCT_BUDGET:
        changed["max_requests_per_min"] = {"product_default": PRODUCT_BUDGET, "used": args.percept_budget}
    if defaults["max_concurrent_requests"] != PRODUCT_CONCURRENCY:
        changed["max_concurrent_requests"] = {"product_default": PRODUCT_CONCURRENCY,
                                              "used": defaults["max_concurrent_requests"]}
    if cond.rearm_s != PRODUCT_REARM_S:
        changed["condition.rearm_s"] = {"product_default": PRODUCT_REARM_S, "used": cond.rearm_s,
                                        "why": "standardised to the baselines' cooldown for comparability"}
    preroll = PreRoll()
    return dict(defaults, label=arm_label(arm, args), changed_from_product_defaults=changed,
                attempts_reserved_per_request=base.max_attempts(judge), store="MemoryStore",
                watch_fps=None, trigger="default for events (on_change)", verify_mode=cond.verify_mode,
                condition={k: getattr(cond, k) for k in keep},
                preroll={"seconds": preroll.seconds, "fps": preroll.fps},
                concurrency={"monitor_per_case": True, "cases_at_once": args.parallel,
                             "max_concurrent_requests_per_case": defaults["max_concurrent_requests"],
                             "decode": "one thread per case", "worker_threads": worker_threads(args)})


async def run_arm(arm: str, cases: list, judge, timelines, out: Path, args, foot: dict) -> list:
    """Every case of one arm, ``--parallel`` at a time (Percept in real time, baselines as fast as
    decoding and the judge allow). A case that raises becomes a result with ``error`` set. Each
    Percept case has its own ``Monitor``, so its request slots are never shared with another case."""
    sem, traces, realtime = asyncio.Semaphore(args.parallel), out / "traces" / arm, arm.startswith("percept")
    if isinstance(judge.inner, judging.OracleJudge):
        judge.inner.realtime = realtime
    if realtime:
        traces.mkdir(parents=True, exist_ok=True)
    missing = sorted({s.path for c in cases for s in c.segments if not Path(s.path).is_file()})
    if missing:     # the media went away (e.g. a drive unmounted): stop loudly, keep finished arms
        raise SystemExit(f"{arm}: {len(missing)} media files are missing, first {missing[0]}")

    pace = args.speed if getattr(args, "pace_baselines", False) else None

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
                                         fps=args.fps, pace=pace)
            return await A.run_detector(case, judge, timelines, then_vlm=arm == "detector_then_vlm",
                                        cooldown_s=args.cooldown, fps=args.fps, pace=pace)
    try:
        done = await asyncio.gather(*(one(c) for c in cases), return_exceptions=True)
    finally:
        if isinstance(judge.inner, judging.OracleJudge):
            judge.inner.realtime = False
    return [r if isinstance(r, A.ArmResult) else
            A.ArmResult(arm, c.case_id, error=f"{type(r).__name__}: {r}"[:300]) for c, r in zip(cases, done)]


# --------------------------------------------------------------------------- raw results


def write_raw(out: Path, arm: str, results: list, footprint: dict) -> None:
    """Everything scoring needs, so a scoring fix never needs a new (paid) run."""
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    with open(raw / f"{arm}.jsonl", "w") as f:
        f.write(json.dumps({"kind": "arm", "arm": arm, "footprint": footprint}) + "\n")
        for r in results:
            f.write(json.dumps(dict(dataclasses.asdict(r), kind="result")) + "\n")


def read_raw(path: Path) -> tuple[str, list, dict]:
    """(arm, results, footprint) from one ``raw/<arm>.jsonl``."""
    arm, results, footprint = path.stem, [], {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.pop("kind") == "arm":
            arm, footprint = rec["arm"], rec["footprint"]
            continue
        rec["preds"] = [A.Pred(**dict(p, frames=tuple(p["frames"]), evidence_frames=tuple(p["evidence_frames"]),
                                      request_ids=tuple(p["request_ids"]))) for p in rec["preds"]]
        rec["looks"] = [A.Look(**dict(lk, frames=tuple(lk["frames"]))) for lk in rec["looks"]]
        rec["budget_drops"] = [tuple(d) for d in rec["budget_drops"]]
        rec["fires"] = [tuple(x) for x in rec["fires"]]
        results.append(A.ArmResult(**rec))
    return arm, results, footprint


# --------------------------------------------------------------------------- scoring


LOWER_IS_BETTER = {"false_per_hour", "cost_per_true_event", "latency_p50_s"}
COMPARED = ("recall", "onset_recall", "precision", "false_per_hour", "cost_per_true_event", "latency_p50_s")


def _cost_comparable(common: set, *arms_pairs: set) -> bool:
    """Request cost covers every condition of a case: it compares only when, in each case, both
    arms score exactly the common conditions."""
    for pairs in arms_pairs:
        for case_id in {k[0] for k in common}:
            if {k for k in pairs if k[0] == case_id} != {k for k in common if k[0] == case_id}:
                return False
    return True


def compare(p_res: list, x_res: list, p_score: dict, x_score: dict, cases: dict, args) -> dict:
    """A Percept arm and a baseline on like-for-like case-conditions: their full scores when they
    score the same pairs, otherwise both rescored on the pairs both score."""
    pp, px = metrics.scored_pairs(p_res, cases), metrics.scored_pairs(x_res, cases)
    common = pp & px
    out = {"pairs_percept": len(pp), "pairs_baseline": len(px), "pairs_common": len(common),
           "subset": pp != px, "comparable": bool(common), "cost_comparable": _cost_comparable(common, pp, px)}
    if not common:
        return out
    if pp != px:
        p_score = metrics.score(p_res, cases, args.tolerance, only=common, seed=args.seed)
        x_score = metrics.score(x_res, cases, args.tolerance, only=common, seed=args.seed)
    out["percept"] = {k: p_score[k] for k in COMPARED}
    out["baseline"] = {k: x_score[k] for k in COMPARED}
    out["percept_ci95"] = {k: p_score["ci95"].get(k) for k in COMPARED}
    out["baseline_ci95"] = {k: x_score["ci95"].get(k) for k in COMPARED}
    return out


def build_table(results: dict, cases: dict, args, footprints: dict) -> dict:
    """Per-arm metrics (and Percept-versus-baseline comparisons) from per-arm results."""
    table = {}
    for arm, res in results.items():
        s = dict(label=arm_label(arm, args), **metrics.score(res, cases, args.tolerance, seed=args.seed))
        hours = s["source_hours"] or 1.0
        fp = dict(footprints.get(arm, {}))
        if "cpu_s" in fp:
            fp["cpu_s_per_source_hour"] = fp["cpu_s"] / hours
        s["footprint"] = fp
        table[arm] = s
    for p in [a for a in results if a.startswith("percept")]:
        table[p]["vs"] = {x: compare(results[p], results[x], table[p], table[x], cases, args)
                          for x in results if not x.startswith("percept")}
    return table


# --------------------------------------------------------------------------- the report


def _fmt(v, pct=False, money=False) -> str:
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    return f"${v:.4f}" if money else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _ci(v, ci, **kw) -> str:
    """A value with its 95% interval."""
    if v is None:
        return "–"
    if not ci or ci[0] is None:
        return _fmt(v, **kw)
    return f"{_fmt(v, **kw)} ({_fmt(ci[0], **kw)}–{_fmt(ci[1], **kw)})"


def not_publishable(m: dict, table: dict) -> list[str]:
    """Why this run's numbers can't be published (empty when they can)."""
    why = []
    if m["judge"]["spec"] == "fake":
        why.append("the judge is the offline oracle")
    if m["dataset"] != "meva":
        why.append(f"dataset {m['dataset']} is diagnostic only (E1 publishes MEVA)")
    if not m.get("pace_baselines"):
        why.append("the baseline arms ran faster than real time (pass --pace-baselines)")
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
    failing = {arm: s["failed_look_rate"] for arm, s in table.items()
               if (s.get("failed_look_rate") or 0.0) > metrics.MAX_FAILED_LOOK_RATE}
    if failing:     # a judge that failed often measures the outage, not the pipeline
        why.append(f"more than {metrics.MAX_FAILED_LOOK_RATE:.0%} of an arm's looks failed ("
                   + ", ".join(f"{a} {r:.1%}" for a, r in failing.items()) + ")")
    untraced = {arm: s["preds_without_yes_frames"] for arm, s in table.items() if s.get("preds_without_yes_frames")}
    if untraced:    # scored at their evidence's ends, which may credit frames the judge said no to
        why.append("some decisions could not be traced to the yes-looks behind them ("
                   + ", ".join(f"{a} {n}" for a, n in untraced.items()) + ")")
    return why


def first_try_rate(c: dict) -> int:
    """Requests a minute a budget admits when each succeeds first time: a request starts while
    what was spent plus its reservation fits, and is then charged one attempt."""
    limit, reserve = c["max_requests_per_min"], c["attempts_reserved_per_request"]
    return max(0, limit - reserve + 1)


def same_frames(digests: dict, case_ids: list, arms: list) -> bool:
    """Every arm has a digest for every case, and they agree."""
    return all(len({digests.get(c, {}).get(a) for a in arms}) == 1 and None not in
               {digests.get(c, {}).get(a) for a in arms} for c in case_ids)


def _comparison_lines(name: str, table: dict) -> tuple[list, list]:
    """(losses, no clear difference) for one Percept arm, by non-overlapping 95% intervals."""
    loses, unclear = [], []
    kws = {"recall": {"pct": True}, "onset_recall": {"pct": True}, "precision": {"pct": True},
           "cost_per_true_event": {"money": True}}
    for arm, c in table[name]["vs"].items():
        x = table[arm]
        if not c["comparable"]:
            unclear.append(f"{arm}: not comparable (no case-condition is scored by both arms)")
            continue
        where = ""
        if c["subset"]:
            where = f" on the {c['pairs_common']} case-conditions both arms score"
            if x["unsupported"]:
                where += f" ({arm} cannot express {len(x['unsupported'])} case-conditions)"
        for key in COMPARED:
            pv, xv = c["percept"][key], c["baseline"][key]
            if None in (pv, xv):
                continue
            if key == "cost_per_true_event" and not c["cost_comparable"]:
                continue    # each request covers every condition of a case: not divisible by condition
            kw = kws.get(key, {})
            pci, xci = c["percept_ci95"].get(key), c["baseline_ci95"].get(key)
            note = " (makes no judge calls)" if key == "cost_per_true_event" and not x["requests"] else ""
            text = (f"{arm} vs {name} on {key}{where}: {_ci(xv, xci, **kw)} vs {_ci(pv, pci, **kw)}{note}")
            low = key in LOWER_IS_BETTER
            if pci and xci and pci[0] is not None and xci[0] is not None:
                clear_x = xci[1] < pci[0] if low else xci[0] > pci[1]
                clear_p = pci[1] < xci[0] if low else pci[0] > xci[1]
            else:
                clear_x = clear_p = False
            if clear_x:
                loses.append(text)
            elif not clear_p:
                unclear.append(text)
    return loses, unclear


def report(manifest: dict, table: dict, digests: dict) -> str:
    m, lines = manifest, []
    lines += [f"# Percept benchmark run `{m['run_id']}`", ""]
    if m.get("merged_from"):
        lines += [f"> Merged from {len(m['merged_from'])} parts run at the same time with the same code, data "
                  "and settings: " + "; ".join(f"`{p['run_id']}` ({', '.join(p['arms'])})" for p in m["merged_from"])
                  + ".", ""]
    if m.get("rescored_utc"):
        lines += [f"> Rescored offline at {m['rescored_utc']} (git `{m['rescore_git_sha'][:10]}`) from the "
                  "run's raw results; no judge was called.", ""]
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
        lines += [f"> Percept ran at {m['speed']}x real time. Its budget and timers use the real clock, and its "
                  "pre-roll samples frames by wall clock, so any speed other than 1.0 also thins the clips it "
                  "judges. Such runs are not evidence; only 1.0x results are publishable.", ""]
    lines += [f"**Publishable: {'yes' if not m['not_publishable'] else 'no'}**"
              + ("" if not m["not_publishable"] else " (" + "; ".join(m["not_publishable"]) + ")"), ""]
    synth = sorted({c["dataset"] for c in m["cases"] if c["synthetic"]})
    lines += [f"Dataset `{m['dataset']}`: {len(m['cases'])} cases, {m['source_hours']:.3f} source-hours"
              + (f" (declared synthetic: {', '.join(synth)})" if synth else "") + f". Judge `{m['judge']['name']}`"
              f", same model in every arm. Matching tolerance ±{m['tolerance_s']} s; baseline cooldown "
              f"{m['cooldown_s']} s. Git `{m['git_sha'][:10]}`{' (dirty)' if m['git_dirty'] else ''}, "
              f"percept {m['percept']['version']}. Actual spend ${m['spent_usd']:.4f}, from the usage the judge "
              f"reported ({m['requests']} requests).", ""]
    for _arm, c in m["monitor_config"].items():
        changed = c["changed_from_product_defaults"]
        conc = c.get("concurrency", {})
        lines.append(f"- {c['label']}: {c['max_requests_per_min']} attempts a minute per stream, "
                     f"{c['attempts_reserved_per_request']} reserved per request and the attempts used charged "
                     f"(at most {first_try_rate(c)} requests a minute when each succeeds first time; retries "
                     f"and verification draw on the same budget); rearm {c['condition']['rearm_s']:g} s; "
                     f"verify `{c['verify_mode']}`; min confidence {c['condition']['min_confidence']}; one Monitor "
                     f"per case, {conc.get('max_concurrent_requests_per_case', c['max_concurrent_requests'])} "
                     f"requests in flight per case, {conc.get('cases_at_once', m['parallel'])} cases at once. "
                     + ("Changed from the product defaults: " + "; ".join(
                         f"`{k}` {v['used']:g} (product default {v['product_default']:g})" for k, v in changed.items())
                        + "." if changed else "Every setting is the product default."))
    cols = ("arm", "coverage", "recall", "onset recall", "recall ceiling", "precision", "false/h", "dup/occ",
            "latency p50/p95 s", "missed", "cost", "$/monitored h", "$/true event", "requests")
    lines += ["", "## Every arm", "", "Values with a 95% case-bootstrap interval in brackets.", "",
              "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _arm, s in table.items():
        ci = s["ci95"]
        cover = f"{s['cases']} cases, {s['source_hours']:.3f} h, {s['conditions']} conds" \
            + (f", {s['cases_skipped']} skipped" if s["cases_skipped"] else "") + (" (aborted)" if s["aborted"] else "")
        lat = f"{_fmt(s['latency_p50_s'])} / {_fmt(s['latency_p95_s'])}"
        lines.append("| " + " | ".join([
            s["label"], cover, _ci(s["recall"], ci["recall"], pct=True),
            _ci(s["onset_recall"], ci["onset_recall"], pct=True), _fmt(s["recall_ceiling"], True),
            _ci(s["precision"], ci["precision"], pct=True), _ci(s["false_per_hour"], ci["false_per_hour"]),
            _fmt(s["duplicates_per_occurrence"]), lat, f"{s['missed']} of {s['occurrences']}",
            _fmt(s["cost_usd"], money=True), _fmt(s["cost_per_hour"], money=True),
            _ci(s["cost_per_true_event"], ci["cost_per_true_event"], money=True), str(s["requests"])]) + " |")
    lines += ["", "A decision matches an event only if one of the frames shown to the judge lies within the event "
              "(± tolerance); the interval those frames span earns nothing by itself. Onset recall counts only "
              "matches whose decision showed a frame within ± tolerance of the onset. The recall ceiling is what "
              "a perfect judge could reach given the arm's cooldown or rearm. Cost is the judge-reported usage "
              "of the footage actually watched; $/monitored h divides it by the hours watched.", "",
              "## Judge failures", "",
              "| arm | looks | failed (rate) | failures by type | refused by the spend cap | unknown answers (rate) "
              "| empty clips |", "|---|---|---|---|---|---|---|"]
    for _arm, s in table.items():
        failed = sum(s["looks_failed_by_type"].values())
        by_type = ", ".join(f"{k} {v}" for k, v in s["looks_failed_by_type"].items()) or "–"
        lines.append(f"| {s['label']} | {s['looks_total']} | {failed} ({_fmt(s['failed_look_rate'], True)}) | "
                     f"{by_type} | {s['looks_refused_by_spend_cap']} | {s['unknown_answers']} "
                     f"({_fmt(s['unknown_rate'], True)}) | {s['empty_clips']} |")
    lines += ["", f"A run in which any arm's failed-look rate exceeds {metrics.MAX_FAILED_LOOK_RATE:.0%} is not "
              "publishable. The rate leaves out looks the harness spend cap refused. An `unknown` answer is not a "
              "failure: it counts as answered and not `yes`. Empty clips are judge calls a baseline skipped "
              "because no frame was held in the window.", "", "## Losses and verification", "",
              "| arm | watched s | dropped looks | of which: budget / deadline | trigger never fired | "
              "verify rejected | unverified | uncertain |", "|---|---|---|---|---|---|---|---|"]
    for _arm, s in table.items():
        lines.append(f"| {s['label']} | {s['watched_s']:g} | {s['budget_drops']} | {s['drops_budget']} / "
                     f"{s['drops_deadline']} | {s['trigger_never_fired']} | {s['verify_rejected']} | "
                     f"{s['unverified']} | {s['uncertain']} |")
    lines += ["", "Dropped looks count every look the Monitor gave up unjudged (its counter). From its trace: "
              "`budget`, the request budget refused it; `deadline`, it waited past its deadline for a request "
              "slot or for evidence (a look ended by the stream ending is neither). Trigger never fired and "
              "verify rejected count missed occurrences.", "", "## Misses by stage", "",
              "| arm | " + " | ".join(metrics.STAGES) + " |", "|" + "---|" * (len(metrics.STAGES) + 1)]
    for _arm, s in table.items():
        lines.append(f"| {s['label']} | " + " | ".join(str(s["misses_by_stage"][k]) for k in metrics.STAGES) + " |")
    lines += ["", "`budget_dropped` counts misses whose only looks the request budget refused or that expired "
              "waiting for it (Percept traces each dropped look).", "", "## Event survival funnel", "",
              "Each annotated event, followed until the first stage it did not pass. "
              "Counts (share of annotated); – marks a stage the arm does not have.", "",
              "| arm | " + " | ".join(metrics.FUNNEL) + " | matched outside the funnel |",
              "|" + "---|" * (len(metrics.FUNNEL) + 2)]
    for _arm, s in table.items():
        fn, total = s["funnel"], s["funnel"]["annotated"] or 1
        cells = ["–" if k in s["funnel_not_applicable"] else f"{fn[k]} ({100 * fn[k] / total:.1f}%)"
                 for k in metrics.FUNNEL]
        lines.append(f"| {s['label']} | " + " | ".join(cells) + f" | {s['funnel_matched_outside']} |")
    lines += ["", "candidate: a look showed a frame within the event (± tolerance), a dropped look's clip "
              "overlapped it, or a detector gate opened in it. admitted: a look was sent. evaluated: the judge "
              f"answered. positive: a detect `yes` at confidence {metrics.MIN_CONFIDENCE} or more. verified: a "
              "verification said `yes`. delivered: a decision matched the event. Matched outside the funnel: a "
              "decision matched although an earlier stage shows no pass (e.g. a delivered decision's evidence "
              "differs from the looks the trace recorded).", "", "## Footprint", "",
              "| arm | CPU s per source-hour | process peak RSS MB | pre-roll frames | intentional sampling | "
              "unintended loss | real-time lag s |", "|---|---|---|---|---|---|---|"]
    detail = lambda d: ", ".join(f"{k} {v}" for k, v in d.items()) or "0"    # noqa: E731
    for _arm, s in table.items():
        f = s["footprint"]
        lines.append(f"| {s['label']} | {_fmt(f.get('cpu_s_per_source_hour'))} | {_fmt(f.get('peak_rss_mb'))} | "
                     f"{f.get('preroll_frames', '–')} | {detail(s['intentional_sampling'])} | "
                     f"{detail(s['unintended_loss'])} | {f.get('max_push_lag_s', '–')} |")
    lines += ["", "CPU is the harness process while the arm ran (decoding included, "
              f"{m['parallel']} cases at a time), not one stream in isolation. Peak RSS is the process high-water "
              "mark so far; run one arm per invocation for a per-arm peak. Real-time lag is how far the Percept "
              "feed fell behind the clock. Intentional sampling is frames the configured fps policies skipped "
              "(expected); unintended loss is frames lost to decode gaps, the push handoff or lag beyond "
              f"{MAX_LAG_S:g} s, and any of it makes the run not publishable.", ""]
    h = m["harness"]
    pre = next((c.get("preroll") for c in m["monitor_config"].values() if c.get("preroll")), None)
    lines += ["## Method notes", "",
              ("- Pacing. Every arm played the footage at the same rate, "
               f"{m['speed']:g}x real time, so each called the provider at the rate it would live."
               if m.get("pace_baselines") else
               "- Pacing. Percept played the footage in real time; the baselines ran as fast as decoding and the "
               "judge allowed, so provider load differed between arms."),
              f"- Frames judged. Baselines keep a history of at most {1 / h.get('baseline_history_spacing_s', 0.5):g} "
              f"frames a second and judge up to {h['clip_frames']} of them spread over the window: `chunk<T>` "
              f"shows {h['clip_frames']} frames about T/{h['clip_frames'] - 1} s apart; detector-gated calls show "
              f"the last {h['clip_s']:g} s."
              + (f" Percept's pre-roll keeps {pre['fps']:g} frames a second for {pre['seconds']:g} s and builds "
                 "each clip from it." if pre else ""),
              "- Suppression. A baseline's cooldown runs from the end of the evidence of the last decision it "
              "emitted for the condition; Percept's rearm runs from its decision time.",
              "- `unknown` answers count as answered and not `yes` in every arm (judge_said_no, funnel "
              "`evaluated`).",
              "- Comparisons below are made on the case-conditions both arms score; cost per true event is "
              "compared only where both arms score every condition of the same cases (a request covers every "
              "condition of a case). A difference is listed as a loss only when the 95% intervals do not "
              "overlap.", ""]
    for name in [a for a in table if a.startswith("percept")]:
        loses, unclear = _comparison_lines(name, table)
        lines += [f"## Where {table[name]['label']} loses", ""] + ([f"- {w}" for w in loses]
                                                                  or ["- nowhere in this run"]) + [""]
        lines += ["No clear difference (95% intervals overlap, or a side has none):", ""] \
            + ([f"- {w}" for w in unclear] or ["- none"]) + [""]
    lines += ["## Coverage", ""]
    for arm, s in table.items():
        items = s["unsupported"]
        if items:
            lines.append(f"- `{arm}` cannot express {len(items)} case-conditions (not scored for it): "
                         + ", ".join(items[:8]) + (" …" if len(items) > 8 else ""))
    same = same_frames(digests, [c["case_id"] for c in m["cases"]], m["arms"])
    lines.append("- Same decoded frames: yes. Every arm decoded the same frames (per-case frame digests are in "
                 "`manifest.json`); the frames each arm judged differ by its sampling (see Method notes)."
                 if same else
                 "- Same decoded frames: NO. An arm is missing a frame digest for a case, or the digests differ; "
                 "see `manifest.json` (an aborted or failed case stops early).")
    for c in m["skipped"]:
        lines.append(f"- skipped `{c['arm']}` / `{c['case_id']}`: {c['reason']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- running


def finish(out: Path, manifest: dict, results: dict, cases: dict, args, footprints: dict) -> dict:
    """Score, then write metrics.json, report.md and manifest.json."""
    table = build_table(results, cases, args, footprints)
    digests = {}
    for arm, res in results.items():
        for r in res:
            if r.digest:
                digests.setdefault(r.case_id, {})[arm] = r.digest
    manifest["frame_digests"] = digests
    manifest["skipped"] = [dict(sk, arm=arm) for arm, s in table.items() for sk in s["skipped"]]
    manifest["not_publishable"] = not_publishable(manifest, table)
    manifest["publishable"] = not manifest["not_publishable"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (out / "metrics.json").write_text(json.dumps(table, indent=1))
    (out / "report.md").write_text(report(manifest, table, digests))
    return table


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
    judge = judging.CostGuard(inner, args.max_cost_usd, spend_log=out / "spend.jsonl")
    arm_names = [a for a in args.arms.split(",") if a]
    bad = [a for a in arm_names if a not in ARMS and not (a.startswith("chunk") and a[5:].replace(".", "").isdigit())]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {ARMS}")
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(worker_threads(args)))
    results, footprints = {}, {}
    for arm in arm_names:
        foot: dict = {}
        with A.Footprint() as fp:
            res = await run_arm(arm, cases, judge, timelines, out, args, foot)
        results[arm], footprints[arm] = res, dict(cpu_s=round(fp.cpu_s, 2), peak_rss_mb=fp.peak_rss_mb, **foot)
        write_raw(out, arm, res, footprints[arm])
        s = metrics.score(res, by_id, args.tolerance, seed=args.seed)
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
        "cases": [c.summary() for c in cases],
        "arms": arm_names, "judge": {"spec": args.judge, "name": judge.name,
                                     "price": dataclasses.asdict(judge.price) if judge.price else None,
                                     "oracle": oracle if args.judge == "fake" else None},
        "monitor_config": {a: monitor_config(a, args, judge) for a in arm_names if a.startswith("percept")},
        "harness": {"clip_frames": A.CLIP_FRAMES, "clip_s": A.CLIP_S, "min_confidence": A.MIN_CONFIDENCE,
                    "baseline_history_spacing_s": A.HISTORY_SPACING_S,
                    "onset_before_s": judging.ONSET_BEFORE_S, "onset_after_s": judging.ONSET_AFTER_S,
                    "precog_s": metrics.PRECOG_S, "max_lag_s": MAX_LAG_S,
                    "max_failed_look_rate": metrics.MAX_FAILED_LOOK_RATE, "bootstrap_resamples": metrics.BOOTSTRAP_N,
                    "worker_threads": worker_threads(args)},
        "seeds": {"seed": args.seed}, "config": args.config, "tolerance_s": args.tolerance, "cooldown_s": args.cooldown,
        "fps": args.fps, "speed": args.speed, "parallel": args.parallel, "max_cost_usd": args.max_cost_usd,
        "pace_baselines": bool(getattr(args, "pace_baselines", False)),
        "spent_usd": judge.spent, "requests": judge.requests, "refused_by_cap": judge.refused,
        "aborted": judge.tripped.is_set(), "args": vars(args),
    }
    finish(out, manifest, results, by_id, args, footprints)
    print(f"wrote {out}/report.md" + (" (ABORTED: spend cap reached)" if manifest["aborted"] else ""))
    return 3 if manifest["aborted"] else 0


def rescore(out: Path, root: str | None = None) -> int:
    """Rebuild metrics.json and report.md from ``<out>/raw`` without calling any judge. Cases are
    reloaded with the run's dataset, root, limit, negatives and seed (``root`` overrides the root)."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    if "args" not in manifest or not (out / "raw").is_dir():
        raise SystemExit(f"{out} has no raw results to rescore (runs before raw/ was written cannot be rescored)")
    args = argparse.Namespace(**dict(parser().parse_args([]).__dict__, **manifest["args"]))
    if root is not None:
        args.root = root
    cases = datasets.load(args.dataset, limit=args.limit, root=args.root, negatives=args.negatives,
                          seed=args.seed, workdir=out)
    by_id = {c.case_id: c for c in cases}
    results, footprints = {}, {}
    for arm in manifest["arms"]:
        _, res, footprints[arm] = read_raw(out / "raw" / f"{arm}.jsonl")
        missing = sorted({r.case_id for r in res} - set(by_id))
        if missing:
            raise SystemExit(f"{arm}: cases {missing[:3]} are not in the reloaded dataset (check --root)")
        results[arm] = res
    manifest.update(rescored_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    rescore_git_sha=_git("rev-parse", "HEAD"))
    finish(out, manifest, results, by_id, args, footprints)
    print(f"rescored {out}/report.md")
    return 0


# What may differ between the parts of one run: which arms each ran, its cap, where it wrote.
PART_ARGS = {"arms", "max_cost_usd", "out", "run_id", "rescore", "merge", "root"}
SAME_KEYS = ("git_sha", "git_dirty", "percept", "python", "platform", "opencv", "detectors_available", "dataset",
             "cases", "judge", "harness", "seeds", "config", "tolerance_s", "cooldown_s", "fps", "speed",
             "parallel", "pace_baselines")


def merge(parts: list, out: Path, root: str | None = None) -> int:
    """One run from parts that ran different arms at the same time (separate processes, same
    code, data and settings), then rescored as one: the parts must agree on everything but their
    arms, cap and output, and no arm may appear twice. Spend, requests and aborts add up."""
    out = Path(out)
    ms = [json.loads((Path(p) / "manifest.json").read_text()) for p in parts]
    first = ms[0]
    for p, m in zip(parts, ms):
        bad = [k for k in SAME_KEYS if m.get(k) != first.get(k)]
        bad += [f"args.{k}" for k in set(m["args"]) | set(first["args"])
                if k not in PART_ARGS and m["args"].get(k) != first["args"].get(k)]
        if bad:
            raise SystemExit(f"{p} differs from {parts[0]} in {sorted(bad)}: not parts of one run")
    arms = [a for m in ms for a in m["arms"]]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"an arm appears in more than one part: {arms}")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for p, m in zip(parts, ms):
        for arm in m["arms"]:
            shutil.copyfile(Path(p) / "raw" / f"{arm}.jsonl", out / "raw" / f"{arm}.jsonl")
            if (Path(p) / "traces" / arm).is_dir():
                shutil.copytree(Path(p) / "traces" / arm, out / "traces" / arm, dirs_exist_ok=True)
    caps = [m["max_cost_usd"] for m in ms]
    cap = None if None in caps else sum(caps)     # one uncapped part leaves the run uncapped
    manifest = dict(first)
    manifest.update(
        run_id=out.name, arms=arms,
        monitor_config={k: v for m in ms for k, v in m.get("monitor_config", {}).items()},
        spent_usd=sum(m["spent_usd"] for m in ms), requests=sum(m["requests"] for m in ms),
        refused_by_cap=sum(m["refused_by_cap"] for m in ms), aborted=any(m["aborted"] for m in ms),
        max_cost_usd=cap,
        args=dict(first["args"], arms=",".join(arms), max_cost_usd=cap,
                  out=str(out), run_id=out.name),
        merged_from=[{"run_id": m["run_id"], "arms": m["arms"], "created_utc": m["created_utc"],
                      "max_cost_usd": m["max_cost_usd"], "spent_usd": m["spent_usd"], "aborted": m["aborted"]}
                     for m in ms])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return rescore(out, root)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m benchmarks.run", description=__doc__.splitlines()[0])
    p.add_argument("--config", help="a JSON file of defaults, e.g. benchmarks/configs/e1.json")
    p.add_argument("--merge", nargs="+", metavar="PART_DIR",
                   help="merge parts that ran different arms at the same time into --out, then rescore")
    p.add_argument("--rescore", metavar="OUT_DIR",
                   help="rescore a finished run from its raw/ results (no judge calls); --root may override")
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
    p.add_argument("--percept-budget", type=int, default=13,
                   help="max_requests_per_min (attempts) of the percept-matched arm; 13 admits at most 12 "
                        "requests a minute when each succeeds first time (chunk5's rate)")
    p.add_argument("--percept-rearm", type=float, default=None,
                   help="rearm_s for the Percept arms (default: the product default)")
    p.add_argument("--fps", type=float, default=5.0, help="frames sampled per second, every arm")
    p.add_argument("--speed", type=float, default=1.0, help="Percept arm pacing; only 1.0 is publishable")
    p.add_argument("--parallel", type=int, default=8, help="cases run at once, per arm")
    p.add_argument("--pace-baselines", action="store_true",
                   help="play baseline arms at --speed times real time, as Percept is (publishable runs need it)")
    p.add_argument("--seed", type=int, default=0, help="dataset negatives, oracle noise and bootstrap resampling")
    p.add_argument("--oracle-fn", type=float, default=0.1)
    p.add_argument("--oracle-fp", type=float, default=0.02)
    p.add_argument("--oracle-latency", type=float, default=1.0)
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.merge:
        if not args.out:
            raise SystemExit("--merge needs --out")
        return merge(args.merge, Path(args.out), args.root)
    if args.rescore:
        return rescore(Path(args.rescore), args.root)
    if args.config:                     # the config sets defaults; flags on the command line still win
        config = json.loads(Path(args.config).read_text())
        if "arm_judges" in config:
            raise SystemExit(f"{args.config}: {config.get('status', 'per-arm judges are not supported')}")
        p.set_defaults(**{k: v for k, v in config.items() if k in vars(args)})
        args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
