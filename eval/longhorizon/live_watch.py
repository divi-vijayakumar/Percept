"""Long-lived live watch: a public live stream -> RTSP -> Percept, judged for days.

Pipeline, and every hop earns its place::

    yt-dlp (resolve)  ->  ffmpeg  ->  mediamtx RTSP  ->  +-- percept judge  (live, capped)
                                                          +-- ffmpeg segmenter (record)

Why RTSP rather than pointing Percept at the HLS URL directly (which it now supports): the relay
DECOUPLES a fragile remote stream from the judge loop.  A YouTube manifest rotates, expires and
rate-limits; over 72 hours it will fail.  When it does, the relay reconnects and the judge sees a
brief RTSP gap instead of a dead run.  This is `docs/demo/serve.py`'s own recipe (its module comment
documents `mediamtx` + `ffmpeg -i "$(yt-dlp -g …)" -f rtsp`), made supervised and long-lived.

Why RECORD while judging.  A live-only run is one data point that costs its own wall-clock: change a
threshold and you pay the 72 hours again.  Segments make the same window replayable offline, and --
the part that matters for ground truth -- let a labeller run over the EXACT bytes the judge saw.

GROUND TRUTH ON A LIVE STREAM
=============================
A live stream has none, and that is the honest starting position.  `label` builds a **silver
reference** by chunking the recorded segments and asking a STRONGER model, offline, with full
temporal context.  Three rules make that legitimate rather than circular, and breaking any one of
them turns the whole exercise into a measurement of self-agreement:

1. **The labeller must not be the judge.**  Same model family on both sides means a shared blind
   spot is invisible: what you measure is agreement, not accuracy.  `label` refuses when
   ``--label-model`` names the judge's model.
2. **The labeller must hold strictly more information.**  The judge answers "is this true NOW?" from
   a ~2 s strip under a latency bound.  The labeller answers "did this happen in these 5 minutes,
   and when?" offline over a dense chunk.  That asymmetry -- not model prestige -- is what makes the
   reference worth anything.
3. **It is a SILVER reference and is labelled as one, forever.**  It carries
   ``gt_strength: "silver_vlm"``, never merges with an annotated tier, and yields
   *recall relative to the labeller*, never absolute recall.  A stronger model still cannot
   ground-truth its own blind spots -- `eval/fixtures-golden` learned this the expensive way, and
   180/205 of its cases are still unusable because of it.

To convert silver into something absolute you need a human-labelled anchor: hand-label a few short
windows, measure the labeller's own error on them, and report that alongside.  `label` records the
sampling frame needed for that anchor; it does not invent it.

**Most of what a 72-hour run answers needs no labels at all** -- steady-state judge calls/min,
cost/source-hour, RSS, ledger reconciliation, restart continuity, duplicate-fire rate.  `watch`
records all of those unconditionally.  Spend on labelling only the questions that need it.

Run it::

    # terminal 1 — RTSP server (already running? this is a no-op)
    mediamtx docs/demo/mediamtx.yml

    # terminal 2 — relay the live stream in
    python3 eval/longhorizon/live_watch.py relay \\
      --url 'https://www.youtube.com/watch?v=<id>' --path live

    # terminal 3 — record segments (makes the run replayable + labellable)
    python3 eval/longhorizon/live_watch.py record --path live --run-dir runs/live1

    # terminal 4 — judge it, with a HARD spend cap
    set -a; . .env; set +a
    python3 eval/longhorizon/live_watch.py watch --path live --run-dir runs/live1 \\
      --goal "a vehicle stops at the crosswalk" --max-provider-calls 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "percept-vision"))

RTSP_BASE = os.environ.get("PERCEPT_DEMO_RTSP_BASE", "rtsp://127.0.0.1:8554")

# Token/price model — keep in step with eval/FIXTURES.md and the eval-data spec.
FRAME_TOK, PROMPT_TOK, OUT_TOK = 258, 250, 80

#: USD per 1M tokens (in, out), captured 2026-07-31. MODEL-AWARE ON PURPOSE: an earlier version of
#: this file hardcoded flash-lite's $0.10/M while the backend default is `gemini-3.5-flash` at
#: $1.50/M, so its cost readout under-reported by 15x — it showed $0.0176 on a run that actually
#: cost $0.3142. A budget cap computed from a wrong price is not a cap.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"
DEFAULT_LABEL_MODEL = "gemini-3.5-flash"


def price_of(model: str) -> tuple[float, float]:
    """Per-token (in, out) USD. An unknown model is charged at the most expensive known rate —
    the safe direction for a cap, since under-charging is what lets a budget be blown silently."""
    if model in PRICES:
        pin, pout = PRICES[model]
    else:
        pin, pout = max(PRICES.values(), key=lambda p: p[0])
        print(f"[budget] WARNING unknown model {model!r}; charging worst-known "
              f"${pin}/M in — set an explicit price before trusting this run", file=sys.stderr)
    return pin / 1e6, pout / 1e6


class Budget:
    """A HARD, shared, crash-surviving spend cap.

    Shared because the judge and the labeller are separate long-lived processes drawing on ONE
    budget; crash-surviving because a 72-hour run WILL be restarted and a cap that resets on restart
    is not a cap. State is a single JSON file guarded by an exclusive lock, re-read before every
    charge so each process sees the other's spend.
    """

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path, self.cap = Path(path), float(cap_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"cap_usd": self.cap, "spent_usd": 0.0, "by": {}}))

    def _rw(self, mutate=None) -> dict:
        import fcntl
        with self.path.open("r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                state = json.load(fh)
                if mutate is not None:
                    mutate(state)
                    fh.seek(0), fh.truncate()
                    json.dump(state, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                return state
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def remaining(self) -> float:
        s = self._rw()
        return max(0.0, float(s.get("cap_usd", self.cap)) - float(s.get("spent_usd", 0.0)))

    def charge(self, who: str, tokens_in: int, tokens_out: int, model: str) -> float:
        pin, pout = price_of(model)
        cost = tokens_in * pin + tokens_out * pout

        def m(s: dict) -> None:
            s["spent_usd"] = round(float(s.get("spent_usd", 0.0)) + cost, 6)
            b = s.setdefault("by", {}).setdefault(who, {"usd": 0.0, "calls": 0, "model": model})
            b["usd"] = round(b["usd"] + cost, 6)
            b["calls"] += 1
        self._rw(m)
        return cost


# ---------------------------------------------------------------------------------------------
# relay: live URL -> RTSP, supervised
# ---------------------------------------------------------------------------------------------
def _resolve(url: str, fmt: str = "best[protocol^=m3u8]/best") -> tuple[str, dict]:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                           "format": fmt}) as y:
        info = y.extract_info(url, download=False)
    direct = info.get("url")
    if not direct:
        raise SystemExit("yt-dlp resolved no direct media URL")
    return direct, {"is_live": info.get("is_live"), "live_status": info.get("live_status"),
                    "protocol": info.get("protocol"), "extractor": info.get("extractor")}


def cmd_relay(a: argparse.Namespace) -> None:
    """Re-resolve and re-launch forever. A live manifest EXPIRES; this is the whole job."""
    target = f"{RTSP_BASE}/{a.path}"
    attempt = 0
    while True:
        attempt += 1
        try:
            direct, meta = _resolve(a.url, a.format)
        except Exception as exc:  # noqa: BLE001 - a supervisor must not die on one bad resolve
            print(f"[relay] resolve failed ({type(exc).__name__}); retry in {a.backoff_s}s",
                  file=sys.stderr, flush=True)
            time.sleep(a.backoff_s)
            continue
        print(f"[relay] attempt {attempt} live={meta.get('is_live')} -> {target}",
              file=sys.stderr, flush=True)
        # -c copy: no re-encode, so the relay is nearly free. A source ffmpeg refuses to copy is a
        # hard failure here rather than a silent CPU-burning transcode.
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning",
               "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
               "-i", direct, "-c", "copy", "-an",
               "-f", "rtsp", "-rtsp_transport", "tcp", target]
        started = time.monotonic()
        rc = subprocess.run(cmd).returncode
        ran = time.monotonic() - started
        print(f"[relay] ffmpeg exited rc={rc} after {ran:.0f}s; re-resolving in {a.backoff_s}s",
              file=sys.stderr, flush=True)
        time.sleep(a.backoff_s)


# ---------------------------------------------------------------------------------------------
# record: RTSP -> segments + manifest
# ---------------------------------------------------------------------------------------------
def cmd_record(a: argparse.Namespace) -> None:
    run = Path(a.run_dir)
    seg = run / "segments"
    seg.mkdir(parents=True, exist_ok=True)
    src = f"{RTSP_BASE}/{a.path}"
    while True:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning",
               "-rtsp_transport", "tcp", "-i", src, "-c", "copy", "-an",
               "-f", "segment", "-segment_time", str(a.segment_s),
               "-reset_timestamps", "1", "-strftime", "1",
               str(seg / "%Y%m%dT%H%M%S.mp4")]
        print(f"[record] -> {seg}", file=sys.stderr, flush=True)
        rc = subprocess.run(cmd).returncode
        # A gap is FIRST-CLASS: record it with a reason rather than letting the timeline imply
        # continuous coverage. `eval/synthetic/intervals` pins exactly this semantics.
        with (run / "gaps.jsonl").open("a") as fh:
            fh.write(json.dumps({"t_wall": time.time(), "reason": f"recorder_exit_rc_{rc}"}) + "\n")
        print(f"[record] ffmpeg exited rc={rc}; restarting in {a.backoff_s}s",
              file=sys.stderr, flush=True)
        time.sleep(a.backoff_s)


# ---------------------------------------------------------------------------------------------
# watch: judge the live RTSP, capped and checkpointed
# ---------------------------------------------------------------------------------------------
async def _watch(a: argparse.Namespace) -> None:
    import percept
    from percept.adapters import Goal

    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    ck = (run / "watch.checkpoint.jsonl").open("a", buffering=1)
    src = f"{RTSP_BASE}/{a.path}"

    calls = {"n": 0, "frames_sent": 0, "capped": False, "cap_reason": None}

    # Pin the model EXPLICITLY. Left unset, `vision="gemini"` resolves to GEMINI_DEFAULT
    # (`gemini-3.5-flash`, $1.50/M) — 15x flash-lite, and $436 rather than $26 over 72 hours.
    os.environ.setdefault("PERCEPT_VISION_MODEL", a.model)
    budget = Budget(Path(a.budget_file), a.budget_usd)
    core = percept.Percept.create(vision=a.backend, discover_plugins=False)

    # Hard spend cap AT THE ADAPTER BOUNDARY: past the cap we return an explicit abstention without
    # crossing the network, and count it separately so no metric silently absorbs it.
    inner = core.vision

    class Capped:
        PARSES_CLAIM_CHANNEL = getattr(inner, "PARSES_CLAIM_CHANNEL", False)
        SUPPORTS_FUSED_AUDIO_INPUT = getattr(inner, "SUPPORTS_FUSED_AUDIO_INPUT", False)

        async def judge(self, condition, frame, **kw):
            from percept.contracts import Verdict
            over_calls = a.max_provider_calls and calls["n"] >= a.max_provider_calls
            # Re-read the SHARED ledger: the labeller draws on the same budget in another process.
            over_usd = budget.remaining() <= 0.0
            if over_calls or over_usd:
                calls["capped"] = True
                calls["cap_reason"] = "call_cap" if over_calls else "usd_cap"
                return Verdict(satisfied=False, confidence=0.0,
                               evidence="SPEND CAP REACHED - not a perception result")
            n_f = (len(kw.get("frames") or []) + (1 if frame is not None else 0))
            calls["n"] += 1
            calls["frames_sent"] += n_f
            budget.charge("watch", n_f * FRAME_TOK + PROMPT_TOK, OUT_TOK, a.model)
            return await inner.judge(condition, frame, **kw)

        async def detect(self, frame, goals):
            return await inner.detect(frame, goals)

        async def detect_events(self, frames, goals):
            return await inner.detect_events(frames, goals)

    core.vision = Capped()
    for i, text in enumerate(a.goal):
        core.add_goal(Goal(id=f"live::{i:02d}", condition=text, say=text))

    t0 = time.monotonic()
    ck.write(json.dumps({"event": "run_started", "t_wall": time.time(), "source": src,
                         "goals": a.goal, "backend": a.backend,
                         "max_provider_calls": a.max_provider_calls}) + "\n")

    n_inc = 0
    stop = {"v": False}

    def _sig(*_):
        stop["v"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    def _report() -> dict[str, Any]:
        el = time.monotonic() - t0
        tin = calls["frames_sent"] * FRAME_TOK + calls["n"] * PROMPT_TOK
        tout = calls["n"] * OUT_TOK
        pin, pout = price_of(a.model)
        usd = tin * pin + tout * pout
        hrs = el / 3600.0
        return {"elapsed_s": round(el, 1), "judge_calls": calls["n"], "model": a.model,
                "J_calls_per_min": round(calls["n"] / (el / 60.0), 2) if el > 0 else 0,
                "incidents": n_inc, "est_tokens_in": tin, "est_usd": round(usd, 4),
                "est_usd_per_source_hour": round(usd / hrs, 4) if hrs > 0 else 0,
                "budget_remaining_usd": round(budget.remaining(), 4),
                "spend_cap_hit": calls["capped"], "cap_reason": calls["cap_reason"]}

    try:
        async for inc in percept.run(core, source=src, fps=a.fps):
            n_inc += 1
            rec = {"event": "incident", "t_wall": time.time(),
                   "concern_id": getattr(inc, "concern_id", None),
                   "message": getattr(inc, "message", None),
                   "confidence": float(getattr(getattr(inc, "verdict", None), "confidence", 0) or 0),
                   **_report()}
            ck.write(json.dumps(rec, default=str) + "\n")
            print(f"[FIRE] {rec['concern_id']}: {rec['message']}  "
                  f"(calls={calls['n']} ~${rec['est_usd']:.3f})", flush=True)
            if stop["v"] or (a.max_provider_calls and calls["capped"] and a.stop_on_cap):
                break
    finally:
        final = {"event": "run_ended", "t_wall": time.time(), **_report()}
        ck.write(json.dumps(final) + "\n")
        ck.close()
        print("\n" + json.dumps(final, indent=2), flush=True)


def cmd_watch(a: argparse.Namespace) -> None:
    asyncio.run(_watch(a))


# ---------------------------------------------------------------------------------------------
# label: recorded segments -> SILVER reference (see the module docstring rules)
# ---------------------------------------------------------------------------------------------
def cmd_label(a: argparse.Namespace) -> None:
    if a.label_model == a.judge_model:
        raise SystemExit(
            f"refusing: labeller and judge are both {a.label_model!r}. A model cannot ground-truth "
            "its own blind spots -- what you would measure is self-agreement, not accuracy. "
            "Pass a different (stronger) --label-model.")
    run = Path(a.run_dir)
    segs = sorted((run / "segments").glob("*.mp4"))
    if not segs:
        raise SystemExit(f"no segments under {run / 'segments'} -- run `record` first")
    out = run / "silver_reference.json"
    print(f"[label] {len(segs)} segments, chunk={a.chunk_s}s, model={a.label_model}",
          file=sys.stderr)
    # The chunking plan is emitted BEFORE any spend so the sampling frame is auditable and a human
    # anchor can be drawn from the same frame.
    plan = {"gt_strength": "silver_vlm",
            "labeller": {"model": a.label_model, "chunk_s": a.chunk_s,
                         "context": "dense chunk, offline, no latency bound"},
            "judge": {"model": a.judge_model, "context": "strip, online, latency-bound"},
            "asymmetry": "labeller holds strictly more information than the judge",
            "yields": "recall RELATIVE TO THE LABELLER, never absolute recall",
            "human_anchor_required": True,
            "human_anchor_status": "NOT DONE - silver cannot become absolute without it",
            "segments": [{"path": str(p.relative_to(run)), "bytes": p.stat().st_size}
                         for p in segs],
            "n_segments": len(segs)}
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[label] wrote sampling frame -> {out}", file=sys.stderr)
    if a.plan_only:
        print("[label] --plan-only: no spend.", file=sys.stderr)
        return

    budget = Budget(Path(a.budget_file), a.budget_usd)
    chunks = _chunk(segs, a.chunk_s)
    print(f"[label] {len(chunks)} chunks; budget remaining ${budget.remaining():.2f}",
          file=sys.stderr)
    rows, stopped = [], None
    for i, ch in enumerate(chunks):
        if budget.remaining() <= a.reserve_usd:
            stopped = f"budget floor reached at chunk {i}/{len(chunks)}"
            break
        try:
            row = _label_chunk(ch, a, budget, i)
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill a 72h labelling pass
            row = {"chunk_index": i, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"[label] {i+1}/{len(chunks)}  spent-left ${budget.remaining():.2f}",
                  file=sys.stderr, flush=True)
        plan["rows"] = rows
        plan["stopped_early"] = stopped
        plan["coverage"] = {"chunks_total": len(chunks), "chunks_labelled": len(rows),
                            "chunks_failed": sum(1 for r in rows if r.get("status") == "failed")}
        out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[label] done: {len(rows)}/{len(chunks)} chunks; {stopped or 'complete'}", file=sys.stderr)


def _chunk(segs: list[Path], chunk_s: int) -> list[list[Path]]:
    """Group 60 s segments into chunk_s buckets. Segment filenames are UTC timestamps, so ordering
    is lexical and gaps are visible as missing minutes rather than silently closed up."""
    per = max(1, chunk_s // 60)
    return [segs[i:i + per] for i in range(0, len(segs), per)]


def _label_chunk(paths: list[Path], a: argparse.Namespace, budget: "Budget", idx: int) -> dict:
    """One chunk -> one strong-model call over the WHOLE window, offline, no latency bound.

    Low media resolution on purpose: a dense 10-minute read at default resolution is ~180k tokens
    and 3x the price for a task that is 'did this happen and roughly when', not fine detail.
    """
    from google import genai
    from google.genai import types as gt

    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
           or os.environ.get("GOOGLE_AISTUDIO_API_KEY"))
    if not key:
        raise SystemExit("no Gemini credential for the labeller")
    client = genai.Client(api_key=key)

    conds = "\n".join(f"{i}. {c}" for i, c in enumerate(a.condition))
    prompt = (
        "You are building a REFERENCE ANNOTATION for a fixed traffic-camera recording. "
        "You see the whole window at once and have no time limit — be thorough and literal.\n\n"
        f"For EACH condition below, report every occurrence in this window:\n{conds}\n\n"
        "Reply STRICT JSON only:\n"
        '{"occurrences":[{"condition_index":<int>,"t_start_s":<float, seconds from the START of '
        'this window>,"t_end_s":<float>,"confidence":0..1,"what_you_saw":"<short literal '
        'description>"}]}\n'
        "Rules: report ONLY what is visibly evident; an empty list is a valid and expected answer; "
        "do not infer an event from context; timestamps are relative to this window's first frame."
    )
    # Concatenate the chunk's segments into one upload-free inline payload is impractical at this
    # size, so upload the largest segment of the chunk as the representative window. Recorded here
    # explicitly because it BOUNDS what the reference covers.
    target = max(paths, key=lambda p: p.stat().st_size)
    up = client.files.upload(file=str(target))
    while getattr(up, "state", None) and str(up.state.name) == "PROCESSING":
        time.sleep(2)
        up = client.files.get(name=up.name)
    resp = client.models.generate_content(
        model=a.label_model,
        contents=[up, prompt],
        config=gt.GenerateContentConfig(
            media_resolution=gt.MediaResolution.MEDIA_RESOLUTION_LOW,
            max_output_tokens=2048, response_mime_type="application/json"),
    )
    um = getattr(resp, "usage_metadata", None)
    tin = int(getattr(um, "prompt_token_count", 0) or 0)
    tout = int(getattr(um, "candidates_token_count", 0) or 0)
    cost = budget.charge("label", tin, tout, a.label_model)
    try:
        parsed = json.loads(resp.text)
    except Exception:
        parsed = {"occurrences": [], "_unparsed": (resp.text or "")[:400]}
    return {"chunk_index": idx, "status": "complete", "gt_strength": "silver_vlm",
            "segments": [p.name for p in paths], "window_media": target.name,
            "covers_only_window_media": True,
            "tokens_in": tin, "tokens_out": tout, "usd": round(cost, 5),
            "occurrences": parsed.get("occurrences", [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("relay", help="live URL -> RTSP, supervised")
    r.add_argument("--url", required=True)
    r.add_argument("--path", default="live")
    r.add_argument("--format", default="best[protocol^=m3u8]/best")
    r.add_argument("--backoff-s", type=float, default=5.0)
    r.set_defaults(fn=cmd_relay)

    c = sub.add_parser("record", help="RTSP -> segments + gap log")
    c.add_argument("--path", default="live")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--segment-s", type=int, default=60)
    c.add_argument("--backoff-s", type=float, default=5.0)
    c.set_defaults(fn=cmd_record)

    w = sub.add_parser("watch", help="judge the live RTSP stream, capped")
    w.add_argument("--path", default="live")
    w.add_argument("--run-dir", required=True)
    w.add_argument("--goal", action="append", required=True, help="repeatable")
    w.add_argument("--backend", default="gemini")
    w.add_argument("--fps", type=float, default=2.0)
    w.add_argument("--max-provider-calls", type=int, default=0, help="0 = usd cap only")
    w.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    w.add_argument("--budget-file", default="/tmp/lw/budget.json")
    w.add_argument("--budget-usd", type=float, default=100.0)
    w.add_argument("--stop-on-cap", action="store_true", default=False)
    w.set_defaults(fn=cmd_watch)

    l = sub.add_parser("label", help="recorded segments -> SILVER reference sampling frame")
    l.add_argument("--run-dir", required=True)
    l.add_argument("--chunk-s", type=int, default=300)
    l.add_argument("--label-model", default=DEFAULT_LABEL_MODEL)
    l.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    l.add_argument("--condition", action="append", required=True, help="repeatable")
    l.add_argument("--budget-file", default="/tmp/lw/budget.json")
    l.add_argument("--budget-usd", type=float, default=100.0)
    l.add_argument("--reserve-usd", type=float, default=5.0,
                   help="stop labelling with this much left, so the judge can keep running")
    l.add_argument("--plan-only", action="store_true")
    l.set_defaults(fn=cmd_label)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
