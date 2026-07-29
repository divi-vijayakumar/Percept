"""Bring your own admission detector — the OCR-driven "look closely NOW" path.

Runs OFFLINE and deterministically: no camera, no keys, no model. `python examples/byo_admission_detector.py`

WHAT THIS SHOWS — the two supported ways to put YOUR OWN cheap model in front of Percept's expensive
judge, and the one thing neither of them does:

  seam 1  percept.run(..., semantic_adapter=f)   f(frame) -> 0..1   fills the Tier0Signal.semantic
          channel; a goal that gates on "semantic" spends a cloud judge when your score clears its
          threshold. This is ADMISSION: your model decides WHEN the judge looks.

  seam 2  register_detector("document-changed", ...)   emits timed Boundary events from the cheap
          per-frame signal; the brain counts them as distinct occurrences (goal.accumulate.fire_when)
          with NO vision call at all. This is OCCURRENCE COUNTING at native fps.

  NEITHER delivers `invoice_number = "INV-2291"` to the judge. A detector boundary says
  "the document changed at t=8.2"; it never says WHAT it now says. If you need extracted FACTS to
  reach the model, call the judge yourself with `vision.judge(condition, frame, context={...})` —
  `context=` is a live Vision parameter, but the watch loop has no channel that fills it for you.
  See docs/admission/detectors.md, "Admission is not extraction".
"""

from __future__ import annotations

from percept.edge.detectors import Boundary, DetectorBank, register_detector, unregister
from percept.edge.tier0 import Frame, MotionGate, compile_watchspec


# ---------------------------------------------------------------------------
# Your cheap "OCR" — stand-in for pytesseract / PaddleOCR / a small VLM on-device.
# ---------------------------------------------------------------------------


def ink_density(frame: Frame) -> float:
    """Fraction of dark pixels — a deterministic stand-in for 'how much text is on screen'."""
    return sum(1 for p in frame.pixels if p < 128) / float(len(frame.pixels))


# ---------------------------------------------------------------------------
# SEAM 1 — the tier-0 adapter: your score becomes the gate's `semantic` channel.
# ---------------------------------------------------------------------------


class DocumentChangeScore:
    """`(frame) -> 0..1`: how much the on-screen text changed since the previous frame."""

    def __init__(self) -> None:
        self._prev: float | None = None

    def __call__(self, frame: Frame) -> float:
        now = ink_density(frame)
        prev, self._prev = self._prev, now
        if prev is None:
            return 0.0
        return min(1.0, abs(now - prev) * 10.0)


# ---------------------------------------------------------------------------
# SEAM 2 — a registered detector: the same score, emitted as timed boundaries.
# ---------------------------------------------------------------------------


class DocumentChanged:
    """A frame-channel detector: one `threshold` boundary each time the page's text changes.

    The detector protocol is three things: a `signal` attribute naming the channel it reads
    ("motion" | "energy" | "frame"), `push(t, value) -> [Boundary]`, and `reset()`.
    """

    signal = "frame"

    def __init__(self, delta: float = 0.02) -> None:
        self._delta = delta
        self._prev: float | None = None

    def push(self, t: float, frame: Frame) -> list[Boundary]:
        now = ink_density(frame)
        prev, self._prev = self._prev, now
        if prev is None or abs(now - prev) < self._delta:
            return []
        return [Boundary(detector="document-changed", kind="threshold", t=t, value=abs(now - prev))]

    def reset(self) -> None:
        self._prev = None


def register() -> None:
    """Register the detector — the WHOLE integration cost. Percept core is never edited."""
    register_detector(
        "document-changed",
        DocumentChanged,
        senses="raw frames (a cheap OCR/ink read)",
        when_to_use="a screen/document workload where the text, not the pixels, is the event",
        output="one 'threshold' boundary each time the on-screen text changes",
        category="frame",
    )


# ---------------------------------------------------------------------------
# A deterministic fixture stream: a still page, then a page turn, then still again.
# ---------------------------------------------------------------------------


def _page(text_rows: int, w: int = 8, h: int = 8) -> Frame:
    """A frame with `text_rows` rows of dark 'text' on a light page."""
    px = [30 if (i // w) < text_rows else 220 for i in range(w * h)]
    return Frame(w=w, h=h, pixels=tuple(px))


FIXTURE = [(0.0, _page(2)), (1.0, _page(2)), (2.0, _page(5)), (3.0, _page(5)), (4.0, _page(1))]


def demo() -> dict:
    """Drive both seams over the fixture; return what each produced (asserted by the test suite)."""
    from percept.adapters import Goal

    register()
    try:
        # --- seam 1: admission. A goal that gates on the `semantic` channel. ------------------
        goal = Goal(id="doc", condition="the invoice on screen changed", say="the invoice changed")
        goal.state["cadence"] = {"gate_on": "semantic"}  # gate this goal on the semantic channel
        spec = compile_watchspec([goal], threshold=0.15)
        gate = MotionGate(spec, semantic_adapter=DocumentChangeScore())

        admitted = [t for t, f in FIXTURE if gate.evaluate(f, ts=t).escalate]

        # In a real watch this is one keyword — no GatedWatchLoop surgery required:
        #     async for incident in percept.run(
        #         core, source="screen:0", semantic_adapter=DocumentChangeScore()
        #     ):
        #         ...

        # --- seam 2: occurrence counting, zero vision calls. ----------------------------------
        bank = DetectorBank(["document-changed"])
        boundaries = [b for t, f in FIXTURE for b in bank.push(t, frame=f)]

        # --- what neither seam did: extraction. ------------------------------------------------
        # `admitted` / `boundaries` carry a TIME, not a READING. To hand the judge a fact, you
        # supply it yourself at the judge call:
        #     await vision.judge(goal.condition, frame, context={"invoice_number": "INV-2291"})
        return {"admitted_ts": admitted, "boundary_ts": [b.t for b in boundaries]}
    finally:
        unregister("document-changed")


if __name__ == "__main__":  # pragma: no cover - demo output
    out = demo()
    print("seam 1 — frames admitted to the judge by the semantic adapter:", out["admitted_ts"])
    print("seam 2 — detector boundaries (no vision call spent):        ", out["boundary_ts"])
    print("neither carries an extracted fact; pass context={...} to judge yourself.")
