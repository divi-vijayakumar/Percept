"""Notification sinks for the runtime worker."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import socket
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


class UrlPolicyError(ValueError):
    """Raised when a sink/source URL is rejected by its :class:`UrlPolicy`."""


# CGNAT (RFC 6598) is internal but is_global only returns False for it on CPython >= 3.12.4; check it
# explicitly so the SSRF block holds on every supported interpreter (>=3.10).
_CGNAT4 = ipaddress.ip_network("100.64.0.0/10")


def _is_internal_ip(ip: "ipaddress._BaseAddress") -> bool:
    """True for any address that is NOT a publicly routable target (SSRF-blocked)."""
    return (not ip.is_global) or (ip.version == 4 and ip in _CGNAT4)


@dataclass(frozen=True)
class UrlPolicy:
    """SSRF-safe URL policy for outbound sinks.

    Defaults are strict: **https only**, and any **non-globally-routable** target (private, loopback,
    link-local incl. the metadata IP ``169.254.169.254``, CGNAT ``100.64/10``, reserved …) is blocked,
    so a ``WebhookSink`` URL — which an app may take from user input — can't be aimed at an internal
    service. ``file://`` fails the scheme check. Use :meth:`dev` for a permissive local-dev policy.

    Checks are split: :meth:`check_static` is DNS-free (scheme + IP literals) and runs at construction;
    :meth:`check_resolved` does the DNS lookup and runs at **send** time, so the constructor never
    blocks on DNS and the host is re-resolved close to use. (Residual DNS-rebinding TOCTOU remains —
    urllib re-resolves when it connects — so this narrows, not closes, the window.)
    """

    allowed_schemes: frozenset = field(default_factory=lambda: frozenset({"https"}))
    block_private: bool = True
    allowed_hosts: frozenset | None = None
    resolve: bool = True  # resolve hostnames so a name pointing at an internal IP is also blocked

    @classmethod
    def dev(cls) -> "UrlPolicy":
        """Permissive policy for local development: allow http + loopback/private targets."""
        return cls(allowed_schemes=frozenset({"https", "http"}), block_private=False)

    def check_static(self, url: str) -> None:
        """Cheap, DNS-free checks: scheme, host present, allowlist, and IP-literal not-global."""
        try:
            parts = urllib.parse.urlsplit(url)
        except ValueError as exc:
            raise UrlPolicyError(f"unparseable URL: {url!r}") from exc
        if parts.scheme not in self.allowed_schemes:
            raise UrlPolicyError(
                f"scheme {parts.scheme!r} not allowed (allowed: {sorted(self.allowed_schemes)}) — "
                "pass url_policy=UrlPolicy.dev() for local http/loopback sinks"
            )
        host = parts.hostname
        if not host:
            raise UrlPolicyError(f"URL has no host: {url!r}")
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            raise UrlPolicyError(f"host {host!r} not in the allowlist")
        if self.block_private:
            lowered = host.lower()
            if lowered == "localhost" or lowered.endswith(".local") or lowered == "metadata.google.internal":
                raise UrlPolicyError(
                    f"host {host!r} is an internal/loopback name (SSRF-blocked) — use UrlPolicy.dev() locally"
                )
            try:
                ip = ipaddress.ip_address(host)  # IP literal
            except ValueError:
                return  # a DNS name — resolved at send time by check_resolved()
            if _is_internal_ip(ip):
                raise UrlPolicyError(f"host {host!r} -> {ip} is not globally routable (SSRF-blocked)")

    def check_resolved(self, url: str) -> None:
        """DNS-resolving check, run at SEND time: block when a host resolves to an internal address.

        Catches encoded-IP literals (``0x7f.0.0.1``, ``2130706433``) the OS resolver decodes, and names
        that point at internal IPs. ``block_private=False`` or ``resolve=False`` skips it.
        """
        if not (self.block_private and self.resolve):
            return
        host = urllib.parse.urlsplit(url).hostname
        if not host:
            return
        try:
            ipaddress.ip_address(host)
            return  # already covered by check_static
        except ValueError:
            pass
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return  # unresolvable — the request fails naturally
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if _is_internal_ip(ip):
                raise UrlPolicyError(f"host {host!r} -> {ip} is not globally routable (SSRF-blocked)")

    def validate(self, url: str) -> None:
        """Full validation (static + DNS) — used by tests and explicit pre-checks."""
        self.check_static(url)
        self.check_resolved(url)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow 3xx redirects — a redirect could bounce a validated URL to a blocked host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise UrlPolicyError(f"webhook target redirected to {newurl!r}; redirects are not followed")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _send(req: urllib.request.Request, timeout: float):
    """Single outbound seam (no-redirect opener) — patched in tests, and the redirect guardrail."""
    return _OPENER.open(req, timeout=timeout)


def redact_url(url: str) -> str:
    """Reduce a URL to ``scheme://host[:port]`` for logs/DLQ records.

    Drops userinfo **and the path/query/fragment** — the most common webhook-secret locations
    (Slack/Discord ``/services/T/B/SECRET`` path tokens, ``?token=`` query). Re-brackets IPv6 hosts.
    """
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable-url>"
    host = p.hostname or ""
    if ":" in host:  # IPv6 literal — re-bracket
        host = f"[{host}]"
    if p.port is not None:
        host = f"{host}:{p.port}"
    if not host:
        return "<no-host>"
    return f"{p.scheme}://{host}" if p.scheme else host


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
        url_policy: UrlPolicy | None = None,
    ) -> None:
        # Strict SSRF policy by default. Cheap, DNS-free checks run now (a bad scheme/IP fails loudly at
        # construction without blocking on DNS); the resolving check runs per send (see __call__), so a
        # name is re-resolved close to use. Pass UrlPolicy.dev() for local http/loopback sinks.
        self.url_policy = url_policy if url_policy is not None else UrlPolicy()
        self.url_policy.check_static(url)
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
        # Re-resolve the host against the policy immediately before sending (narrows DNS-rebinding TOCTOU;
        # a name public at construction but flipped to an internal IP is blocked here).
        self.url_policy.check_resolved(self.url)
        body = json.dumps(incident_payload(incident, include_media=self.include_media)).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body, headers=self._headers(body), method="POST")
                with _send(req, self.timeout_s) as resp:
                    status = getattr(resp, "status", 200)
                    if status >= 400:
                        raise RuntimeError(f"webhook sink failed with HTTP {status}")
                    return
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_s)
        if self.dead_letter is not None:
            # REDACTED: never persist the raw body, the signing HMAC, or auth headers in a DLQ record.
            # A content hash lets you correlate/dedupe without storing the payload or its credentials.
            self.dead_letter.append(
                {
                    "url": redact_url(self.url),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "error": repr(last_exc)[:500],
                    "attempts": self.retries + 1,
                }
            )
        if last_exc is not None:
            raise last_exc


class FileDeadLetterQueue:
    """Append failed webhook deliveries to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Persist only the already-redacted fields — no raw body, signature, or auth headers.
        line = {
            "url": record.get("url"),
            "body_sha256": record.get("body_sha256"),
            "error": record.get("error"),
            "attempts": record.get("attempts"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def webhook(url: str, **kwargs) -> WebhookSink:
    """Convenience factory for ``run(..., on_incident=webhook(url))``.

    By default the URL must be **https to a public host** (SSRF-safe — see :class:`UrlPolicy`). For a
    local/dev sink (``http://localhost:8080`` …) pass ``url_policy=UrlPolicy.dev()``::

        webhook("http://localhost:8080/hook", url_policy=UrlPolicy.dev())
    """
    return WebhookSink(url, **kwargs)
