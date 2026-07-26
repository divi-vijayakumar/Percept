"""Phase-0 de-risk probe — does a strip / audio actually change a verdict a single frame gets wrong?

This is the GO/NO-GO gate before trusting strip/audio (naive windowing
was a NO-GO for RepCount; nothing is assumed to help). For each hand-recorded clip it issues the SAME
condition to Gemini four ways at the ambiguous moment:

    single        — one frame (today's path)
    +strip        — the current frame + N earlier frames (temporal context, one multimodal call)
    +audio        — the current frame + an audio clip
    +strip+audio  — both

and prints (satisfied, confidence, latency) per config. GO iff an escalation moves the verdict toward the
clip's ground truth on the continuous/discrete clips WHILE the single-state CONTROL stays flat (no false
lift). If only the null result repeats, ship the control plane (PerceptionSpec ROI/cadence) and drop
strip/audio.

CLIP FORMAT (so this is runnable the moment you record): put each clip at
``eval/perception_probe/<name>/clip.json`` with:
    {
      "condition":  "the milk is about to boil over",
      "shape":      "continuous" | "discrete" | "control",
      "expected":   true,                  # ground truth AT the ambiguous moment
      "ambiguous_t": 6.5,                  # seconds; the frame a single look gets wrong
      "frames":     [[t_seconds, "<base64 jpeg>"], ...],   # same shape as the golden corpus
      "audio":      {"mime": "audio/wav", "data": "<base64>"}   # optional; needed for +audio
    }
Record ~3 to start: a continuous+audio clip (milk boil), a discrete clip (bench press at a cooperative
frame rate), and a single-state control (stove on / door open).

Run: PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.exp_perception_spec_probe
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent / "perception_probe"
STRIP_N = 4  # current + up to (STRIP_N-1) earlier frames


def _media(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def _load_clip(path: Path) -> dict:
    spec = json.loads(path.read_text())
    frames = []
    for t, b64 in spec.get("frames", []):
        raw = base64.b64decode(b64)
        frames.append((float(t), (_media(raw), raw)))
    frames.sort(key=lambda x: x[0])
    spec["_frames"] = frames
    aud = spec.get("audio")
    spec["_audio"] = (aud["mime"], base64.b64decode(aud["data"])) if aud else None
    return spec


def _assemble(spec: dict):
    """Pick the current frame nearest ``ambiguous_t`` and the preceding strip (oldest→newest)."""
    frames = spec["_frames"]
    if not frames:
        return None, []
    t0 = float(spec.get("ambiguous_t", frames[-1][0]))
    cur_i = min(range(len(frames)), key=lambda i: abs(frames[i][0] - t0))
    current = frames[cur_i][1]
    strip = [f for _, f in frames[max(0, cur_i - (STRIP_N - 1)) : cur_i]]  # earlier context
    return current, strip


async def _judge(vision, condition, frame, *, frames=None, audio=None):
    t0 = time.monotonic()
    v = await vision.judge(condition, frame, frames=frames, audio=audio)
    return v, (time.monotonic() - t0) * 1000.0


def _toward(expected: bool, single, esc) -> str:
    """Did the escalation move the verdict the RIGHT way vs the single-frame read?"""
    if esc.satisfied == expected and single.satisfied != expected:
        return "FLIP→correct"
    if esc.satisfied == single.satisfied == expected and esc.confidence > single.confidence + 0.1:
        return "conf↑"
    if esc.satisfied == single.satisfied:
        return "flat"
    return "FLIP→wrong"


async def _run() -> int:
    import asyncio  # noqa: F401  (imported for symmetry; caller uses asyncio.run)
    from eval._dotenv import load_dotenv
    from percept import registry
    from percept.config import load_settings

    load_dotenv()
    clips = sorted(PROBE_DIR.glob("*/clip.json"))
    if not clips:
        print(f"\n  No probe clips found under {PROBE_DIR}/<name>/clip.json")
        print("  Record ~3 (see the CLIP FORMAT block in this file's module docstring):")
        print("    • continuous+audio  — milk boiling over (the canonical 'rising' single-frame miss)")
        print("    • discrete          — bench press at a cooperative frame rate (phase, not sub-Nyquist)")
        print("    • control           — a single-state watch (stove on / door open) — must stay FLAT")
        print("  Then: PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python3.12 -m eval.exp_perception_spec_probe")
        return 0

    vision = registry.make_vision("gemini", load_settings())
    rows: list[tuple] = []
    print("=" * 92)
    print("  PERCEPTION-SPEC PROBE — Gemini-2.5-flash · single vs +strip vs +audio vs +both")
    print("=" * 92)
    for cp in clips:
        spec = _load_clip(cp)
        cond, shape, expected = spec["condition"], spec.get("shape", "?"), bool(spec.get("expected"))
        current, strip = _assemble(spec)
        audio = spec["_audio"]
        configs = [("single", None, None), ("+strip", strip, None)]
        if audio is not None:
            configs += [("+audio", None, audio), ("+strip+audio", strip, audio)]
        print(f"\n  [{shape}] {cp.parent.name}: \"{cond}\"  (expected={expected}, strip={len(strip)} earlier)")
        verdicts: dict[str, object] = {}
        for name, fr, au in configs:
            v, ms = await _judge(vision, cond, current, frames=fr, audio=au)
            verdicts[name] = v
            mark = "" if name == "single" else f"  [{_toward(expected, verdicts['single'], v)}]"
            print(f"      {name:<13} satisfied={str(v.satisfied):<5} conf={v.confidence:.2f}  {ms:6.0f}ms{mark}")
        rows.append((shape, {k: verdicts[k] for k in verdicts}, expected))

    # GO/NO-GO: escalation must help non-control shapes and leave the control flat.
    helped = any(
        shape != "control"
        and any(_toward(exp, vs["single"], v) in ("FLIP→correct", "conf↑") for k, v in vs.items() if k != "single")
        for shape, vs, exp in rows
    )
    control_false = any(
        shape == "control"
        and any(_toward(exp, vs["single"], v) == "FLIP→wrong" for k, v in vs.items() if k != "single")
        for shape, vs, exp in rows
    )
    print("\n" + "=" * 92)
    verdict = "GO" if (helped and not control_false) else "NO-GO"
    print(f"  VERDICT: {verdict}   (escalation helped a non-control shape: {helped}; "
          f"control false-lift: {control_false})")
    print("  GO  -> build strip/audio + the edge path (Phases 4-6).  NO-GO -> ship control plane only.")
    print("=" * 92)
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover — live probe
    raise SystemExit(main())
