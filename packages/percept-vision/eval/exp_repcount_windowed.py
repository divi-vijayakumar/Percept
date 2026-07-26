"""RepCount-A off-by-one — WINDOWED perception (S1 gating spike) vs the single-frame baseline.

The single-frame counter (``exp_repcount.py``) asked Gemini per-frame "is the body in the up/peak
phase of one <action> rep?" and counted rising edges. It FAILED (OBO=15.0, MAE=4.0): a single frame
mid-rep is ambiguous (going up or down?) and short clips are undersampled (~2 frames/rep), so the
cognition layer (refractory dedup) had no clean edge to dedup.

This spike tests the hypothesis that feeding a WINDOW of consecutive frames + asking a COMPLETION /
COUNT question removes the single-frame ambiguity. Two windowed variants, same 20 videos, same GT:

  (a) windowed-direct  — split each video into ~6-8s windows, show ~6 frames evenly spanning each
                         window in temporal order, ask "how many full reps COMPLETED in this window?"
                         and SUM across windows.
  (b) windowed-sliding — a short K=6-frame window over ~2s slid at ~1s stride; ask "did one full rep
                         COMPLETE within this window? true|false"; count the trues with a small
                         refractory (one completion can span adjacent windows).

Metric: OBO (off-by-one accuracy = fraction of videos with |pred - gt| <= 1, x100) + MAE. Printed next
to the single-frame baseline (loop OBO 15.0 / MAE 4.0). Perception held at Gemini 2.5 Flash. Live:
PERCEPT_LIVE=1 + a Gemini key. Run:
  PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.exp_repcount_windowed
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import cv2

from ._dotenv import load_dotenv
from .exp_e2e import gemini_client

ROOT = Path(__file__).resolve().parents[3]
LANCE = ROOT / "eval/bench-staging/repcount/raw/data/test.lance"
N_VIDEOS = int(os.environ.get("PERCEPT_REPCOUNT_N", 20))

# ---- windowed-direct (variant a) ----
DIRECT_WINDOW_S = float(os.environ.get("PERCEPT_RC_DIRECT_WIN_S", 7.0))   # ~6-8s windows
DIRECT_FRAMES = int(os.environ.get("PERCEPT_RC_DIRECT_FRAMES", 6))        # frames per window

# ---- windowed-sliding (variant b) ----
SLIDE_WINDOW_S = float(os.environ.get("PERCEPT_RC_SLIDE_WIN_S", 2.0))     # ~2s window
SLIDE_STRIDE_S = float(os.environ.get("PERCEPT_RC_SLIDE_STRIDE_S", 1.0))  # ~1s stride
SLIDE_FRAMES = int(os.environ.get("PERCEPT_RC_SLIDE_FRAMES", 6))          # K=6 frames per window
SLIDE_REFRACTORY_S = float(os.environ.get("PERCEPT_RC_SLIDE_REFRACTORY_S", 0.8))

# single-frame baseline (exp_repcount.py: gemini-loop / percept refractory)
BASE_OBO = 15.0
BASE_MAE = 4.0


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def _to_jpeg(fr, max_dim=512):
    h, w = fr.shape[:2]
    m = max(h, w)
    if m > max_dim:
        sc = max_dim / m
        fr = cv2.resize(fr, (int(w * sc), int(h * sc)))
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def grab_frames(path, fids):
    """Read a list of frame ids from a video, returning [(fid, jpeg_bytes), ...] (skips unreadable)."""
    out = []
    c = cv2.VideoCapture(path)
    for fid in fids:
        c.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, fr = c.read()
        if not ok:
            continue
        jpg = _to_jpeg(fr)
        if jpg is not None:
            out.append((fid, jpg))
    c.release()
    return out


def gemini_window_count(client, model, action, frames):
    """(a) Across a temporally-ordered window of frames, how many full reps COMPLETED?"""
    from google.genai import types
    parts = []
    for _fid, jpg in frames:
        parts.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
    prompt = (
        f"These {len(frames)} frames are consecutive moments from a video, in TEMPORAL ORDER, of a "
        f"person repeatedly performing: {action}. Counting only repetitions whose full cycle "
        f"(down->up->down, or the action's natural start->finish) COMPLETES within this span of "
        f"frames, how many full {action} repetitions completed? Do not count a rep that is only "
        f"partially shown. Return STRICT JSON {{\"reps\": <int>}}."
    )
    parts.append(types.Part.from_text(text=prompt))
    r = client.models.generate_content(model=model, contents=parts)
    d = _json(r.text or "")
    try:
        return max(0, int(d.get("reps", 0)))
    except Exception:
        return 0


def gemini_window_completed(client, model, action, frames):
    """(b) Did exactly one full rep COMPLETE within this short window? -> bool."""
    from google.genai import types
    parts = []
    for _fid, jpg in frames:
        parts.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
    prompt = (
        f"These {len(frames)} frames are consecutive moments from a short (~2 second) clip, in "
        f"TEMPORAL ORDER, of a person performing: {action}. Within JUST this short clip, does one "
        f"full {action} repetition COMPLETE its cycle (reaches the end/return of one rep)? Answer "
        f"true only if a rep finishes inside this clip, not if it is merely in progress. Return "
        f"STRICT JSON {{\"completed\": true|false}}."
    )
    parts.append(types.Part.from_text(text=prompt))
    r = client.models.generate_content(model=model, contents=parts)
    return bool(_json(r.text or "").get("completed"))


def load_subset():
    """Same subset as exp_repcount.py: counts 4..20, first N. video_blob read lazily via take_blobs."""
    import lance
    ds = lance.dataset(str(LANCE))
    meta = ds.to_table(columns=["video_id", "action_type", "count", "video_ext"]).to_pylist()
    idx = [i for i, m in enumerate(meta) if str(m["count"]).isdigit() and 4 <= int(m["count"]) <= 20]
    idx = idx[:N_VIDEOS]
    return ds, idx, [meta[i] for i in idx]


def _window_fids(nfr, fps, start_s, win_s, n_frames):
    """n_frames evenly spanning [start_s, start_s+win_s), clamped to [0, nfr)."""
    a = start_s * fps
    b = min((start_s + win_s) * fps, nfr)
    if b <= a:
        return []
    if n_frames <= 1:
        return [int(a)]
    span = b - a
    return sorted({min(nfr - 1, int(a + span * k / (n_frames - 1))) for k in range(n_frames)})


async def capture(ds, idx, meta):
    """For each video: run windowed-direct (sum of window counts) and windowed-sliding (refractory
    count of completion edges). Returns {vid: {action, gt, direct, sliding}}."""
    client = gemini_client()
    sem = asyncio.Semaphore(4)
    loop = asyncio.get_event_loop()
    caps: dict = {}

    for i, r in zip(idx, meta):
        vid = r["video_id"]
        gt = int(r["count"])
        action = r["action_type"]
        blob = ds.take_blobs("video_blob", indices=[i])[0].read()
        tmp = tempfile.NamedTemporaryFile(
            suffix="." + (r.get("video_ext") or "mp4").lstrip("."), delete=False)
        tmp.write(blob)
        tmp.close()
        c = cv2.VideoCapture(tmp.name)
        fps = c.get(cv2.CAP_PROP_FPS) or 30.0
        nfr = int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        c.release()
        if nfr == 0:
            os.unlink(tmp.name)
            continue
        dur = nfr / fps

        # ---- (a) windowed-direct: non-overlapping ~win_s windows, sum the per-window counts ----
        direct_starts = []
        t = 0.0
        while t < dur:
            direct_starts.append(t)
            t += DIRECT_WINDOW_S

        async def direct_count(start_s, action=action, path=tmp.name, nfr=nfr, fps=fps):
            fids = _window_fids(nfr, fps, start_s, DIRECT_WINDOW_S, DIRECT_FRAMES)
            frames = await loop.run_in_executor(None, grab_frames, path, fids)
            if len(frames) < 2:
                return 0
            try:
                async with sem:
                    return await loop.run_in_executor(
                        None, gemini_window_count, client, model, action, frames)
            except Exception:
                return 0

        # ---- (b) windowed-sliding: short window slid by stride, completion bool per window ----
        slide_starts = []
        t = 0.0
        while t + SLIDE_WINDOW_S <= dur + SLIDE_STRIDE_S:
            slide_starts.append(round(t, 2))
            t += SLIDE_STRIDE_S
        if not slide_starts:
            slide_starts = [0.0]

        async def slide_completed(start_s, action=action, path=tmp.name, nfr=nfr, fps=fps):
            fids = _window_fids(nfr, fps, start_s, SLIDE_WINDOW_S, SLIDE_FRAMES)
            frames = await loop.run_in_executor(None, grab_frames, path, fids)
            if len(frames) < 2:
                return [start_s, False]
            try:
                async with sem:
                    done = await loop.run_in_executor(
                        None, gemini_window_completed, client, model, action, frames)
            except Exception:
                done = False
            return [start_s, bool(done)]

        direct_counts, slide_seq = await asyncio.gather(
            asyncio.gather(*(direct_count(s) for s in direct_starts)),
            asyncio.gather(*(slide_completed(s) for s in slide_starts)),
        )
        direct_total = int(sum(direct_counts))
        slide_seq = sorted(slide_seq, key=lambda x: x[0])

        caps[vid] = {
            "action": action, "gt": gt,
            "direct": direct_total,
            "direct_windows": direct_counts,
            "slide_seq": slide_seq,
        }
        os.unlink(tmp.name)
        print(f"  {vid} ({action}) gt={gt}  direct={direct_total}  "
              f"slide_trues={sum(1 for _, d in slide_seq if d)}/{len(slide_seq)}")
    return caps


def count_completions(seq, refractory_s):
    """Count true completion windows, ignoring trues within refractory_s of the last counted one."""
    n, last = 0, None
    for ts, done in seq:
        if done and (last is None or ts - last >= refractory_s):
            n += 1
            last = ts
    return n


model = os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")


async def main():
    load_dotenv()
    cap_path = ROOT / "repcount_windowed_signals.json"

    if cap_path.exists() and os.environ.get("PERCEPT_RECAPTURE") != "1":
        caps = json.loads(cap_path.read_text())
        print(f"loaded {len(caps)} videos' windowed signals (offline re-score)")
    else:
        ds, idx, meta = load_subset()
        print(f"subset: {len(meta)} videos · actions={sorted({r['action_type'] for r in meta})}")
        caps = await capture(ds, idx, meta)
        cap_path.write_text(json.dumps(caps))
        print(f"captured {len(caps)} videos (Gemini {model}) -> {cap_path}")

    # ---- score both windowed variants ----
    direct_err, slide_err, rows = [], [], []
    for vid, c in caps.items():
        gt = c["gt"]
        pdirect = int(c["direct"])
        pslide = count_completions(c["slide_seq"], SLIDE_REFRACTORY_S)
        direct_err.append(abs(pdirect - gt))
        slide_err.append(abs(pslide - gt))
        rows.append((vid, c["action"], gt, pdirect, pslide))

    def obo(errs):
        return 100.0 * sum(e <= 1 for e in errs) / len(errs) if errs else 0.0

    def mae(errs):
        return sum(errs) / len(errs) if errs else 0.0

    print("\n" + "=" * 74)
    print("  RepCount-A off-by-one — WINDOWED perception vs single-frame baseline")
    print(f"  {len(caps)} videos · OBO = % within +/-1 of GT count · model={model}")
    print("=" * 74)
    print(f"  {'system':<28}{'OBO x100':>10}{'MAE':>9}")
    print(f"  {'single-frame baseline':<28}{BASE_OBO:>10.1f}{BASE_MAE:>9.2f}")
    print(f"  {'windowed-direct (sum)':<28}{obo(direct_err):>10.1f}{mae(direct_err):>9.2f}")
    print(f"  {'windowed-sliding (edges)':<28}{obo(slide_err):>10.1f}{mae(slide_err):>9.2f}")
    print("=" * 74)
    best_obo = max(obo(direct_err), obo(slide_err))
    best_mae = min(mae(direct_err), mae(slide_err))
    verdict = ("GO" if (best_obo >= 35.0 or best_mae <= 2.5) else "NO-GO")
    print(f"  baseline OBO {BASE_OBO}/MAE {BASE_MAE}  ->  best windowed OBO {best_obo:.1f}/MAE {best_mae:.2f}")
    print(f"  VERDICT: {verdict}")
    print("  per-video (vid · action · gt · direct · sliding):")
    for vid, act, gt, pd, ps in rows:
        print(f"    {vid:<16}{act:<16} gt={gt:<3} direct={pd:<3} sliding={ps}")

    Path("repcount_windowed_result.json").write_text(json.dumps({
        "obo_direct": obo(direct_err), "mae_direct": mae(direct_err),
        "obo_sliding": obo(slide_err), "mae_sliding": mae(slide_err),
        "baseline_obo": BASE_OBO, "baseline_mae": BASE_MAE,
        "verdict": verdict, "rows": rows,
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + a Gemini key.")
        raise SystemExit(0)
    asyncio.run(main())
