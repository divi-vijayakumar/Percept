"""Claude (Anthropic) vision adapter — narrow grounded VQA (spec §6.1).

The single biggest quality/cost lever (spec §6.1): stop calling the cloud VLM as "describe the
scene." Call it as a **narrow grounded question about a cropped ROI** — *"is the user drinking
from THIS mug <exemplar>?"* — and attach the deictic exemplar image when the user demonstrated
("this mug"). The judge returns the firing-contract ``Verdict{satisfied, confidence, evidence}``;
it never compiles the goal to ``attr=value`` (one firing path, one gate).

**Lane discipline.** This is an **L2-only** backend: it needs a key and the ``anthropic`` SDK.
The module imports cleanly with neither present (the core never imports a vendor SDK at module
top level — invariant 7; the SDK is **lazy-imported inside methods**). Calling a method without
the key/SDK raises a clear, actionable error. The L1 lane uses ``FakeVision``, never this.

The model id is taken from ``settings.models.vision_model`` — no literal in logic (rule 5).
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ..adapters import Box, Goal, Observation, Verdict

# A tight system prompt that pins the contract: narrow grounded yes/no + calibrated confidence
# + a one-line evidence string, as strict JSON. Kept here (a prompt is data for this adapter,
# not a tunable of the cognition logic) — the firing decision still lives in the gate.
_JUDGE_SYSTEM = (
    "You are a perception module answering ONE narrow grounded question about a specific "
    "region of an image. Answer only about what is visible in the cropped region (and the "
    "reference exemplar if one is attached). Do NOT describe the whole scene. Return STRICT "
    'JSON: {"satisfied": <true|false>, "confidence": <0..1>, "evidence": "<short reason>"}. '
    "Set confidence low when the region is ambiguous, occluded, or you are unsure — a wrong "
    "confident yes is the worst outcome."
)


def _require_anthropic():
    """Lazy-import the Anthropic SDK with a clear error if it is absent (L2 boundary)."""
    try:
        import anthropic  # noqa: PLC0415 — lazy by design (invariant 7)
    except ModuleNotFoundError as exc:  # pragma: no cover — exercised only without the extra
        raise RuntimeError(
            "AnthropicVision needs the 'anthropic' SDK — install percept-vision[claude]. "
            "(L2 only; the L1 lane uses the fake vision backend.)"
        ) from exc
    return anthropic


def _settings_of(arg: Any):
    """Accept either a full Settings tree or a ModelSettings; tolerate None for late binding."""
    if arg is None:
        from ..config import load_settings

        return load_settings()
    return arg


class AnthropicVision:
    """``Vision`` adapter backed by Claude. Narrow grounded VQA on a cropped ROI.

    Constructed by the registry as ``AnthropicVision(settings)``. The client is built lazily on
    first call so the module imports with no key present (L2 boundary).
    """

    def __init__(self, settings: Any = None, *, client: Any = None) -> None:
        self._settings = _settings_of(settings)
        self._client = client  # injectable for L2 fixtures; built lazily otherwise
        self._model = getattr(
            getattr(self._settings, "models", self._settings), "vision_model", "claude-opus-4-8"
        )

    # -- client ------------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            anthropic = _require_anthropic()
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
        return self._client

    # -- the firing-contract judge ----------------------------------------
    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None = None,
        exemplar: Any = None,
        corrections: dict | None = None,
    ) -> Verdict:
        """Ask Claude a narrow grounded yes/no about ``condition`` over the cropped ROI.

        The prompt grounds on the ROI crop; when an ``exemplar`` is present it is attached as a
        second image and the question is phrased deictically ("...from THIS one <exemplar>?").
        Any user ``corrections`` are surfaced as authoritative context (asserted beats observed).
        """
        client = self._ensure_client()
        content = self._build_content(condition, frame, roi=roi, exemplar=exemplar,
                                      corrections=corrections)
        resp = client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_verdict(_text_of(resp))

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        """Detect goal-relevant things in a frame (the entity-attribute substrate).

        Kept deliberately thin: the entity graph's continuity needs boxes + labels, not a scene
        description. Returns ``[]`` on a parse miss rather than fabricating detections.
        """
        client = self._ensure_client()
        labels = ", ".join(sorted({g.condition for g in goals})) or "any salient object"
        prompt = (
            "Detect goal-relevant objects. For each, return JSON list items "
            '{"label": str, "box": [x,y,w,h] normalized 0..1, "confidence": 0..1}. '
            f"Goals of interest: {labels}. Return STRICT JSON list (possibly empty)."
        )
        content = [_image_block(frame), {"type": "text", "text": prompt}]
        resp = client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_observations(_text_of(resp))

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        """Temporal events land in Phase 5; this adapter defers (returns ``[]``)."""
        return []

    # -- prompt construction ----------------------------------------------
    def _build_content(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None,
        exemplar: Any,
        corrections: dict | None,
    ) -> list:
        """Build the multimodal user turn: the ROI crop, the exemplar, and the narrow question."""
        content: list = [_image_block(frame, roi=roi, label="region of interest")]
        if exemplar is not None:
            content.append(_image_block(exemplar, label="reference exemplar"))
            question = (
                f'Looking only at the region of interest, and comparing to the reference '
                f'exemplar: is this true — "{condition}" — about THIS specific object? '
            )
        else:
            question = (
                f'Looking only at the region of interest: is this true right now — '
                f'"{condition}"? '
            )
        if corrections:
            facts = "; ".join(f"{k}={v}" for k, v in corrections.items())
            question += (
                f"The user has told you these authoritative facts (trust them over your own "
                f"read): {facts}. "
            )
        question += 'Answer as STRICT JSON {"satisfied", "confidence", "evidence"}.'
        content.append({"type": "text", "text": question})
        return content


# ---------------------------------------------------------------------------
# Image + response helpers (stdlib only; no vendor types leak into the core)
# ---------------------------------------------------------------------------


def _image_block(frame: Any, *, roi: Box | None = None, label: str | None = None) -> dict:
    """Render an opaque frame to an Anthropic image content block.

    Accepts already-encoded ``{"type": "image", ...}`` blocks (pass-through), raw bytes (assumed
    PNG/JPEG; base64-encoded), or a ``(media_type, bytes)`` pair. ROI cropping of real pixels is
    the caller's job (the narrow-VQA win is *what we ask*, grounded by the crop); we pass the
    ``roi`` through as a hint only when the frame is a plain descriptor.
    """
    if isinstance(frame, dict) and frame.get("type") == "image":
        return frame
    media_type = "image/png"
    raw: bytes
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        media_type, raw = str(frame[0]), bytes(frame[1])
    elif isinstance(frame, (bytes, bytearray)):
        raw = bytes(frame)
    else:
        # A non-image descriptor (e.g. a path or token) — let the SDK caller pre-encode in L2.
        raise TypeError(
            "AnthropicVision needs image bytes / an Anthropic image block as the frame; got "
            f"{type(frame).__name__}. Pre-encode the frame (and crop the ROI) before judging."
        )
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }


def _text_of(resp: Any) -> str:
    """Pull the concatenated text out of an Anthropic Messages response."""
    blocks = getattr(resp, "content", None)
    if blocks is None and isinstance(resp, dict):
        blocks = resp.get("content")
    parts: list[str] = []
    for b in blocks or []:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _coerce_json(text: str) -> Any:
    """Extract the first JSON object/array from a possibly-fenced model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = min(
        [i for i in (text.find("{"), text.find("[")) if i != -1] or [-1]
    )
    if start == -1:
        raise ValueError("no JSON found in model reply")
    return json.loads(text[start:])


def _parse_verdict(text: str) -> Verdict:
    """Parse the strict-JSON judge reply into a ``Verdict`` (defensive defaults on a bad read).

    A parse failure is treated as *low-confidence unknown* (satisfied=False, confidence=0.0) so
    the gate defers to ASK rather than risking a confident-wrong fire (invariant 1).
    """
    try:
        data = _coerce_json(text)
        return Verdict(
            satisfied=bool(data.get("satisfied", False)),
            confidence=float(data.get("confidence", 0.0)),
            evidence=str(data.get("evidence", "")),
        )
    except (ValueError, TypeError, AttributeError, KeyError):
        return Verdict(satisfied=False, confidence=0.0, evidence=f"unparsed: {text[:120]}")


def _parse_observations(text: str) -> list[Observation]:
    try:
        data = _coerce_json(text)
    except ValueError:
        return []
    out: list[Observation] = []
    for item in data if isinstance(data, list) else []:
        try:
            box = item["box"]
            out.append(
                Observation(
                    label=str(item["label"]),
                    box=Box(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


__all__ = ["AnthropicVision"]
