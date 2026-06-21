"""RepCount-A off-by-one — GEMINI FLASH perception, Percept-counter vs raw-Gemini-loop vs TML (35.4).

Repetition counting is the PUREST test of the cognition layer as an event individuator: the same noisy
per-frame phase signal (Gemini: "is the body in the up/peak phase of one <action> rep?") is turned into
a rep COUNT two ways —
  * gemini-loop  — raw: count every false->true edge of the per-frame signal (perception flicker = over-count)
  * percept      — the event track: count edges with a refractory window (one rep per oscillation, robust)
Metric: OBO (off-by-one accuracy = fraction of videos with |pred - gt| <= 1, x100) + MAE, vs TML's 35.4
(streaming RepCount-A; competitors 1.3). Data: ungated lmms-lab-eval/repcounta-lance (embeds video_blob +
count + cycle_bounds). Perception held constant at Gemini Flash. Live: PERCEPT_LIVE=1 + a Gemini key.
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
TARGET_FPS = float(os.environ.get("PERCEPT_REPCOUNT_FPS", 3.0))
FRAME_CAP = int(os.environ.get("PERCEPT_REPCOUNT_CAP", 150))
N_VIDEOS = int(os.environ.get("PERCEPT_REPCOUNT_N", 20))
REFRACTORY_S = float(os.environ.get("PERCEPT_REPCOUNT_REFRACTORY_S", 0.6))
TML_OBO = 35.4  # streaming RepCount-A; competitors 1.3


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def gemini_phase(client, model, action, frame):
    """Per-frame binary phase: True if the body is in the up/peak phase of the action's rep."""
    from google.genai import types
    media, raw = frame
    prompt = (f"The person is repeatedly performing: {action}. Looking at THIS single frame, is the "
              f"body in the 'up'/raised/open/peak phase of the repetition (yes) rather than the "
              f"'down'/lowered/closed/rest phase (no)? Return STRICT JSON {{\"up\":true|false}}.")
    r = client.models.generate_content(model=model,
        contents=[types.Part.from_bytes(data=raw, mime_type=media), prompt])
    return bool(_json(r.text or "").get("up"))


def count_edges(seq, refractory_s=0.0):
    """Count false->true rising edges; with refractory, ignore edges within refractory_s of the last."""
    n, last, prev = 0, None, False
    for ts, up in seq:
        if up and not prev and (last is None or ts - last >= refractory_s):
            n += 1; last = ts
        prev = up
    return n


def load_subset():
    """video_blob is a Lance blob-ref (struct<position,size>) — read bytes lazily via take_blobs."""
    import lance
    ds = lance.dataset(str(LANCE))
    meta = ds.to_table(columns=["video_id", "action_type", "count", "video_ext"]).to_pylist()
    # prefer moderate counts (4..20) for a clean, bounded run
    idx = [i for i, m in enumerate(meta) if str(m["count"]).isdigit() and 4 <= int(m["count"]) <= 20]
    idx = idx[:N_VIDEOS]
    return ds, idx, [meta[i] for i in idx]


async def main():
    load_dotenv()
    model = os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")
    cap_path = ROOT / "repcount_signals.json"

    if cap_path.exists() and os.environ.get("PERCEPT_RECAPTURE") != "1":
        caps = json.loads(cap_path.read_text())
        print(f"loaded {len(caps)} videos' phase signals (offline re-score)")
    else:
        ds, idx, meta = load_subset()
        print(f"subset: {len(meta)} videos · actions={sorted({r['action_type'] for r in meta})}")
        client = gemini_client(); sem = asyncio.Semaphore(4); loop = asyncio.get_event_loop()
        caps = {}
        for i, r in zip(idx, meta):
            vid = r["video_id"]; gt = int(r["count"]); action = r["action_type"]
            blob = ds.take_blobs("video_blob", indices=[i])[0].read()
            tmp = tempfile.NamedTemporaryFile(suffix="." + (r.get("video_ext") or "mp4").lstrip("."), delete=False)
            tmp.write(blob); tmp.close()
            cap = cv2.VideoCapture(tmp.name); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release()
            if nfr == 0:
                os.unlink(tmp.name); continue
            step = max(1, round(fps / TARGET_FPS))
            fids = list(range(0, nfr, step))[:FRAME_CAP]

            def grab(fid, path=tmp.name):
                c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, fr = c.read(); c.release()
                if not ok:
                    return None
                h, w = fr.shape[:2]; m = max(h, w)
                if m > 512:
                    sc = 512 / m; fr = cv2.resize(fr, (int(w * sc), int(h * sc)))
                ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return ("image/jpeg", buf.tobytes()) if ok else None

            async def phase_at(fid, action=action, fps=fps):
                frame = grab(fid)
                if frame is None:
                    return None
                try:
                    async with sem:
                        up = await loop.run_in_executor(None, gemini_phase, client, model, action, frame)
                except Exception:
                    return None
                return [round(fid / fps, 2), up]

            seq = [x for x in await asyncio.gather(*(phase_at(f) for f in fids)) if x]
            seq.sort(key=lambda x: x[0])
            caps[vid] = {"action": action, "gt": gt, "seq": seq}
            os.unlink(tmp.name)
            print(f"  {vid} ({action}) gt={gt}  frames={len(seq)}")
        cap_path.write_text(json.dumps(caps))
        print(f"captured {len(caps)} videos (Gemini {model}) -> {cap_path}")

    # ---- score: raw edges vs refractory-deduped edges ----
    loop_err, perc_err, rows_out = [], [], []
    for vid, c in caps.items():
        gt = c["gt"]; seq = c["seq"]
        ploop = count_edges(seq, 0.0)
        pperc = count_edges(seq, REFRACTORY_S)
        loop_err.append(abs(ploop - gt)); perc_err.append(abs(pperc - gt))
        rows_out.append((vid, c["action"], gt, ploop, pperc))

    def obo(errs):
        return 100.0 * sum(e <= 1 for e in errs) / len(errs) if errs else 0.0

    def mae(errs):
        return sum(errs) / len(errs) if errs else 0.0

    print("\n" + "=" * 74)
    print("  RepCount-A off-by-one — GEMINI FLASH perception · Percept-counter vs raw-loop vs TML")
    print(f"  {len(caps)} videos · OBO = % within +/-1 of GT count · model={model}")
    print("=" * 74)
    print(f"  {'system':<24}{'OBO x100':>10}{'MAE':>9}")
    print(f"  {'gemini-loop (raw edges)':<24}{obo(loop_err):>10.1f}{mae(loop_err):>9.2f}")
    print(f"  {'percept (refractory)':<24}{obo(perc_err):>10.1f}{mae(perc_err):>9.2f}")
    print(f"  {'TML (native model)':<24}{TML_OBO:>10.1f}{'—':>9}")
    print(f"  {'competitors (TML-reported)':<24}{1.3:>10.1f}{'—':>9}")
    print("=" * 74)
    print(f"  cognition delta: OBO {obo(loop_err):.1f} -> {obo(perc_err):.1f}, "
          f"MAE {mae(loop_err):.2f} -> {mae(perc_err):.2f}; TML bar {TML_OBO}")
    print("  per-video (vid · action · gt · loop · percept):")
    for vid, act, gt, pl, pp in rows_out:
        print(f"    {vid:<14}{act:<14} gt={gt:<3} loop={pl:<3} percept={pp}")
    Path("repcount_result.json").write_text(json.dumps(
        {"obo_loop": obo(loop_err), "obo_percept": obo(perc_err),
         "mae_loop": mae(loop_err), "mae_percept": mae(perc_err),
         "rows": rows_out}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + a Gemini key."); raise SystemExit(0)
    asyncio.run(main())
