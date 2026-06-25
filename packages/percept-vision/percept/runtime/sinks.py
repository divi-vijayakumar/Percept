"""Notification sinks for the runtime worker."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.request
from dataclasses import asdict, is_dataclass
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def incident_payload(incident: Any, *, include_media: bool = False) -> dict[str, Any]:
    """Return a webhook-safe JSON payload for an Incident.

    Media is omitted by default; webhook notifications should carry metadata and evidence, while callers
    can opt into inline media for local/dev sinks.
    """
    payload = _jsonable(incident)
    if not include_media and payload.get("snippet"):
        payload["snippet"]["media"] = None
        payload["snippet"]["frames"] = None
    return payload


class WebhookSink:
    """POST runtime incidents as JSON to a webhook URL using stdlib urllib."""

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 5.0,
        headers: dict[str, str] | None = None,
        include_media: bool = False,
        retries: int = 0,
        retry_backoff_s: float = 0.25,
        signing_secret: str | bytes | None = None,
        dead_letter: list[dict[str, Any]] | None = None,
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.include_media = include_media
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.signing_secret = signing_secret.encode("utf-8") if isinstance(signing_secret, str) else signing_secret
        self.dead_letter = dead_letter

    def _headers(self, body: bytes) -> dict[str, str]:
        headers = dict(self.headers)
        if self.signing_secret:
            digest = hmac.new(self.signing_secret, body, hashlib.sha256).hexdigest()
            headers["X-Percept-Signature"] = f"sha256={digest}"
        return headers

    def __call__(self, incident: Any) -> None:
        body = json.dumps(incident_payload(incident, include_media=self.include_media)).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body, headers=self._headers(body), method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    status = getattr(resp, "status", 200)
                    if status >= 400:
                        raise RuntimeError(f"webhook sink failed with HTTP {status}")
                    return
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_s)
        if self.dead_letter is not None:
            self.dead_letter.append(
                {
                    "url": self.url,
                    "body": body,
                    "headers": self._headers(body),
                    "error": repr(last_exc),
                }
            )
        if last_exc is not None:
            raise last_exc


def webhook(url: str, **kwargs) -> WebhookSink:
    """Convenience factory for ``run(..., on_incident=webhook(url))``."""
    return WebhookSink(url, **kwargs)
