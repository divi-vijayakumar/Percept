"""Visualize the rhythm-hero A/B — draw the per-frame hero tracking so you can SEE what each count did.

For each of the N=8 clips it renders:
  * an annotated MP4 (best-effort H.264 so it plays in a browser; falls back to mp4v), and
  * a contact-sheet JPG (always viewable — a grid of annotated frames), and
  * an index.html tying them together with the GT / A / B / C verdicts.

Boxes are coloured BY ENTITY ID, and the SELECTED performer is drawn thick green ("HERO"). That makes the
failure modes legible at a glance: a single jumper whose id keeps changing colour = fragmentation (the jack
regression); frames with no green box = the performer wasn't held that frame (low hero_track_coverage).

  PYTHONPATH=packages/percept-vision python -m eval.bench.pose_cache   # once
  PYTHONPATH=packages/percept-vision python -m eval.bench.hero_viz     # writes bench-staging/repcount/hero_viz/
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from eval.bench.hero_ab import (EXPECTED_ROM, FPS, MIN_CONF, VLM_PRIOR, _arm_a, _goal_state)
from eval.bench.pose_cache import load_detections, reconstruct
from eval.bench.repcount import _STAGING

OUT = _STAGING.parent / "hero_viz"

# The measured A/B/C results (from `eval.bench.hero_ab` edge-only + live runs) — captions only; the per-frame
# hero boxes are recomputed deterministically below. (vid: action, gt, A, B, C_live, src, conf, cov, htc, ents)
RESULTS = {
    "stu10_5":   ("bench_pressing", 16, 15, 17, 7,  "pose",       0.28, 0.82, 0.94, 84),
    "train143":  ("bench_pressing", 0,  4,  4,  4,  "pose",       0.41, 0.99, 1.00, 1),
    "stu8_23":   ("front_raise",    3,  4,  4,  2,  "pose",       0.01, 0.99, 0.76, 5),
    "test538":   ("front_raise",    5,  6,  6,  6,  "pose",       0.82, 0.97, 1.00, 4),
    "stu8_34":   ("jump_jack",      5,  7,  2,  1,  "pose",      -0.11, 1.00, 0.57, 3),
    "train1665": ("jump_jack",      3,  6,  6,  3,  "pose",       0.37, 0.99, 0.95, 2),
    "stu4_28":   ("others",         10, 9,  6,  1,  "pose",       0.14, 1.00, 0.44, 6),
    "stu13_33":  ("pommelhorse",    33, 9,  33, 33, "silhouette", 0.70, 0.00, 0.98, 10),
}
ORDER = ["stu8_34", "stu4_28", "stu10_5", "stu13_33", "train1665", "stu8_23", "train143", "test538"]


def _color(eid: str):
    h = int(hashlib.md5(eid.encode()).hexdigest(), 16)
    return (60 + int((h & 0xFF) * 0.75), 60 + int(((h >> 8) & 0xFF) * 0.75), 60 + int(((h >> 16) & 0xFF) * 0.75))


def _annotate(frame, boxes, subject, hud):
    import cv2

    img = frame.copy()
    h, w = img.shape[:2]
    for eid, (bx, by, bw, bh) in boxes:
        x0, y0, x1, y1 = int(bx * w), int(by * h), int((bx + bw) * w), int((by + bh) * h)
        is_hero = eid == subject
        col = (0, 255, 0) if is_hero else _color(eid)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3 if is_hero else 1)
        tag = f"HERO {eid}" if is_hero else eid
        cv2.putText(img, tag, (x0 + 2, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 if is_hero else 0.4, col, 2 if is_hero else 1, cv2.LINE_AA)
    # top banner
    band = max(22, int(0.06 * h))
    cv2.rectangle(img, (0, 0), (w, band * len(hud)), (0, 0, 0), -1)
    for i, line in enumerate(hud):
        cv2.putText(img, line, (6, band * i + band - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _hero_pass(vid, action):
    """Recompute Arm B deterministically, recording per-frame entity boxes. Returns (subject, frames, full)."""
    from eval.count_video import decode
    from percept import Goal, Percept
    from percept.identity import AppearanceIdentity
    from percept.rhythm_hero import RhythmHero
    from percept_harness.appearance import appearance_signature
    from percept_harness.detectors import person_measures

    cache = load_detections(vid, FPS, "rtmo")
    _small, full = decode(str(_STAGING / f"{vid}.mp4"), FPS)
    n = min(len(cache), len(full))
    cache, full = cache[:n], full[:n]
    prior = VLM_PRIOR.get(action)
    erom = EXPECTED_ROM.get(prior) if prior else None

    core = Percept.create(discover_plugins=False, identity=AppearanceIdentity(), embedder=appearance_signature)
    core.add_goal(Goal(id="g", condition="a completed rep", say="rep", state=_goal_state(prior, erom)))
    hero = RhythmHero(core, "g", prior_channel=prior, record=True)
    for fr, (_t, bgr) in zip(cache, full):
        hero.observe(fr["t"], person_measures(reconstruct(fr["dets"]), min_conf=MIN_CONF), frame=bgr)
    return hero.result().subject, hero.frames, full


def _write_clip(vid):
    import cv2

    action, gt, A, B, C, src, conf, cov, htc, ents = RESULTS[vid]
    subject, frames, full = _hero_pass(vid, action)
    by_t = {round(t, 4): boxes for t, boxes in frames}
    hud = [
        f"{vid}  {action}  GT={gt}",
        f"A={A}  B={B}  C(live)={C}   src={src} hero={subject} htc={htc} ents={ents} conf={conf}",
    ]
    h, w = full[0][1].shape[:2]

    # --- annotated MP4 (H.264 if available, else mp4v) ---
    mp4 = OUT / f"{vid}.mp4"
    writer, used = None, None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*fourcc), FPS, (w, h))
        if writer.isOpened():
            used = fourcc
            break
        writer.release()
    annotated = []
    for t, bgr in full:
        a = _annotate(bgr, by_t.get(round(t, 4), []), subject, hud)
        annotated.append(a)
        if writer is not None:
            writer.write(a)
    if writer is not None:
        writer.release()

    # --- contact sheet (always viewable): 4x4 grid of evenly-spaced annotated frames ---
    cols, rows = 4, 4
    k = cols * rows
    idx = [round(i * (len(annotated) - 1) / (k - 1)) for i in range(k)] if len(annotated) > 1 else [0]
    tile_w = 320
    tiles = []
    for j in idx:
        a = annotated[min(j, len(annotated) - 1)]
        th = int(a.shape[0] * tile_w / a.shape[1])
        t = cv2.resize(a, (tile_w, th))
        cv2.putText(t, f"t={full[min(j, len(full) - 1)][0]:.1f}s", (4, th - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    import numpy as np
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in tiles]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    sheet = OUT / f"{vid}.jpg"
    cv2.imwrite(str(sheet), grid)
    return {"vid": vid, "action": action, "gt": gt, "A": A, "B": B, "C": C, "src": src,
            "htc": htc, "ents": ents, "subject": subject, "codec": used}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for vid in ORDER:
        print(f"  rendering {vid} ...", flush=True)
        cards.append(_write_clip(vid))

    def verdict(p, gt):
        return "ok" if abs(p - gt) <= 1 else ("over" if p > gt else "under")

    rowshtml = []
    for c in cards:
        gt = c["gt"]
        rowshtml.append(f"""
        <section>
          <h2>{c['vid']} &middot; {c['action']} &middot; GT {gt}</h2>
          <p class="m">
            A(committed)=<b class="{verdict(c['A'],gt)}">{c['A']}</b> &nbsp;
            B(connected edge)=<b class="{verdict(c['B'],gt)}">{c['B']}</b> &nbsp;
            C(connected+live VLM)=<b class="{verdict(c['C'],gt)}">{c['C']}</b> &nbsp; | &nbsp;
            src={c['src']} &middot; hero={c['subject']} &middot; htc={c['htc']} &middot; entities={c['ents']}
          </p>
          <video controls preload="metadata" width="640" src="{c['vid']}.mp4"></video>
          <img src="{c['vid']}.jpg" alt="{c['vid']} contact sheet"/>
        </section>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>rhythm hero A/B viz</title>
<style>
 body{{font:14px -apple-system,system-ui,sans-serif;margin:24px;background:#111;color:#ddd}}
 h1{{font-size:18px}} h2{{font-size:15px;margin:18px 0 4px}} .m{{color:#aaa;margin:0 0 8px}}
 b.ok{{color:#6f6}} b.over{{color:#fc6}} b.under{{color:#f88}}
 section{{border-top:1px solid #333;padding-top:10px}}
 video,img{{display:block;max-width:100%;margin:8px 0;border:1px solid #333}}
 img{{width:880px}}
</style>
<h1>Rhythm hero A/B — per-clip tracking (green = selected performer; colour = entity id)</h1>
<p class="m">Watch the green HERO box. A single body that keeps changing id colour = fragmentation (undercount).
Frames with no green box = the performer wasn't held that frame (low htc). Video needs H.264; if it won't play,
the contact sheet below each video always shows the annotated frames.</p>
{''.join(rowshtml)}"""
    (OUT / "index.html").write_text(html)
    print(f"\n  wrote {len(cards)} clips -> {OUT}/index.html")
    for c in cards:
        print(f"    {c['vid']:10s} codec={c['codec']}  (A={c['A']} B={c['B']} C={c['C']} gt={c['gt']})")


if __name__ == "__main__":  # pragma: no cover
    main()
