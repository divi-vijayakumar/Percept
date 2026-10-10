"""The shell shared by the vendor adapters: the ``Judge`` wrapper, the reply schema and the
mapping from provider errors to the three kinds ``policy.call`` understands.

An adapter subclasses ``Adapter`` and implements ``_request``: one provider request, returning a
``RawReply`` or raising. Everything else (deadline, retry, parsing, usage) is ``policy.call``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from percept.contracts import ANSWERS, JudgeRequest, JudgeResult, PriceModel
from percept.judges import policy
from percept.judges.base import Prompt, RawReply, RefusalError, RetryableError

RETRYABLE_STATUS = frozenset({408, 429})          # plus every 5xx (529 is "overloaded")
CONNECTION_ERRORS = frozenset({"APIConnectionError", "TransportError", "NetworkError",
                               "RemoteProtocolError", "ClientConnectionError"})


def require(module: str, extra: str) -> ModuleType:
    """Import a vendor SDK, or fail with the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"this judge needs the {module!r} package: "
                          f"pip install 'percept-vision[{extra}]'") from exc


def reply_schema(request: JudgeRequest, *, closed: bool) -> dict:
    """JSON Schema of the reply ``base.SYSTEM`` asks for, limited to the requested ids.

    ``closed`` adds ``additionalProperties: false``, which strict structured-output modes need.
    """
    entry: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": [item.condition.id for item in request.items]},
            "answer": {"type": "string", "enum": list(ANSWERS)},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["id", "answer", "confidence", "reason"],
    }
    schema: dict[str, Any] = {"type": "object", "properties": {"results": {"type": "array", "items": entry}},
                              "required": ["results"]}
    if closed:
        entry["additionalProperties"] = False
        schema["additionalProperties"] = False
    return schema


def _class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):  # anthropic/openai: status_code; google-genai: code
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _seconds(value: Any, scale: float = 1.0) -> float | None:
    try:
        seconds = float(str(value).strip().rstrip("s")) * scale
    except (TypeError, ValueError):
        return None  # an HTTP date or garbage: let the policy pick the backoff
    return seconds if seconds >= 0 else None


def retry_after(exc: BaseException) -> float | None:
    """The provider's retry hint in seconds: ``retry-after-ms`` / ``retry-after`` headers, or a
    Google ``RetryInfo.retryDelay`` in the error body."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
                value = headers.get(key)
                if value is not None and _seconds(value, scale) is not None:
                    return _seconds(value, scale)
        except Exception:  # noqa: BLE001 - a header object we don't understand is no hint
            pass
    details = getattr(exc, "details", None)
    error = details.get("error", details) if isinstance(details, dict) else None
    for item in (error.get("details") or []) if isinstance(error, dict) else []:
        if isinstance(item, dict) and "retryDelay" in item:
            return _seconds(item["retryDelay"])
    return None


def map_error(exc: Exception) -> Exception:
    """What ``send`` raises for a provider exception.

    Transport timeouts become ``TimeoutError``; 408, 429 and 5xx (including 529 overloaded) and
    dropped connections become ``RetryableError``; anything else (400, 401, 403, 404) is
    returned unchanged and fails the request as ``unavailable``.
    """
    names = _class_names(exc)
    if isinstance(exc, TimeoutError) or any("Timeout" in name for name in names):
        return TimeoutError(str(exc) or type(exc).__name__)
    status = _status(exc)
    if status is not None and (status in RETRYABLE_STATUS or 500 <= status <= 599):
        return RetryableError(f"HTTP {status}: {exc}"[:200], retry_after_s=retry_after(exc))
    if status is None and names & CONNECTION_ERRORS:
        return RetryableError(f"{type(exc).__name__}: {exc}"[:200])
    return exc


def usage_int(obj: Any, *names: str) -> int:
    """Sum of the integer usage fields ``names`` on ``obj`` (missing or None count as 0)."""
    total = 0
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            total += max(0, value)
    return total


class Adapter:
    """A ``Judge`` that sends each provider request through ``_request``.

    ``sent`` is a test hook: it is None in normal use, and when a test sets it to a list every
    provider request is appended to it as ``(prompt, images, max_output_tokens, request)``, the
    shape the conformance suite checks.
    """

    sent: list | None = None

    def __init__(self, name: str, *, price: PriceModel | None, can_batch: bool, client: Any) -> None:
        self.name = name
        self.price = price
        self.can_batch = can_batch
        self.client = client

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        return await policy.call(self._send, request, judge_name=self.name, price=self.price,
                                 batch=self.can_batch)

    async def _send(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                    request: JudgeRequest) -> RawReply:
        if self.sent is not None:
            self.sent.append((prompt, images, max_output_tokens, request))
        try:
            return await self._request(prompt, images, max_output_tokens, request)
        except (RetryableError, RefusalError, TimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 - mapped, then re-raised for the policy
            raise map_error(exc) from exc

    async def _request(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                       request: JudgeRequest) -> RawReply:
        raise NotImplementedError
