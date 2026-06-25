"""Notification sinks for the runtime worker."""

from __future__ import annotations

import base64
import json
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
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.include_media = include_media

    def __call__(self, incident: Any) -> None:
        body = json.dumps(incident_payload(incident, include_media=self.include_media)).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise RuntimeError(f"webhook sink failed with HTTP {status}")


def webhook(url: str, **kwargs) -> WebhookSink:
    """Convenience factory for ``run(..., on_incident=webhook(url))``."""
    return WebhookSink(url, **kwargs)
