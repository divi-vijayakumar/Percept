"""Notification sinks for the runtime worker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import ipaddress
import json
import socket
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

from percept.incidents import incident_revision, incident_supersedes
from percept.observability import get_logger, log


class UrlPolicyError(ValueError):
    """Raised when an outbound sink URL is rejected by its :class:`UrlPolicy`.

    The policy guards **outbound sinks** (e.g. :class:`WebhookSink`). Capture *source* URIs
    (RTSP / ``file://`` / ``webcam:``) are operator config and are NOT policy-validated here.
    """


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
        # Redact the redirect target: its path/query can carry a secret (a redirect Location is
        # attacker-influenced) and this message lands in the DLQ 'error' field.
        raise UrlPolicyError(f"webhook target redirected to {redact_url(newurl)}; redirects are not followed")


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
    try:
        port = p.port  # SplitResult.port raises ValueError on a malformed/out-of-range port
    except ValueError:
        port = None  # omit an unparseable port rather than crash the log/DLQ path
    if port is not None:
        host = f"{host}:{port}"
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
        max_attempt_age_s: float | None = None,
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
        # ADR-001 §9: "delivery attempts" bound. `retries` already bounds the ATTEMPT COUNT
        # (finite by construction — a plain int). `max_attempt_age_s` adds an orthogonal wall-clock
        # bound so a long backoff schedule can't keep retrying indefinitely in elapsed time either;
        # either bound reaching its limit is a TERMINAL failure (dead-lettered with a reason). None
        # (default) disables the age bound — byte-for-byte the attempt-count-only behavior.
        self.max_attempt_age_s = max_attempt_age_s
        self.signing_secret = signing_secret.encode("utf-8") if isinstance(signing_secret, str) else signing_secret
        self.dead_letter = dead_letter

    def _headers(self, body: bytes) -> dict[str, str]:
        headers = dict(self.headers)
        if self.signing_secret:
            digest = hmac.new(self.signing_secret, body, hashlib.sha256).hexdigest()
            headers["X-Percept-Signature"] = f"sha256={digest}"
        return headers

    def __call__(self, incident: Any) -> Any:
        """Deliver ``incident`` — **never on the event-loop thread**.

        A webhook POST is blocking by construction: stdlib ``urllib`` has no async form, and the
        retry backoff is a ``time.sleep``. Run on the event loop that made delivery part of the
        perception critical path — one unreachable subscriber stalled gate evaluation, the audio
        pump and decode consumption, and the bounded capture queue then dropped frames for it.

        So the blocking work lives in :meth:`_deliver_blocking` and this method only chooses how to
        get there:

        * **inside a running event loop** — returns an AWAITABLE that runs ``_deliver_blocking`` on a
          worker thread (``asyncio.to_thread``). The loop thread never encodes the payload, never
          waits on the socket, and never sleeps for the backoff. The runtime's ``_call_sink`` awaits
          any awaitable a sink returns, so this is drop-in there; so are
          :class:`IdempotentSink`/:class:`SupersessionSink`, which both already forward an awaitable
          result. The awaited result raises exactly what the blocking call would have raised, so
          retry/DLQ/error semantics are byte-for-byte unchanged.
        * **no running loop** (a plain thread, a script, a synchronous test) — calls
          ``_deliver_blocking`` directly and returns ``None``: byte-for-byte today's behavior,
          because blocking the caller's own thread is exactly what that caller asked for.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._deliver_blocking(incident)
        return asyncio.to_thread(self._deliver_blocking, incident)

    def _deliver_blocking(self, incident: Any) -> None:
        """The whole blocking delivery: encode, POST, retry-with-backoff, dead-letter.

        MUST NOT run on the event-loop thread — it makes a synchronous socket call and calls
        ``time.sleep`` for the retry backoff. :meth:`__call__` is the only caller and is responsible
        for putting it on a worker thread whenever a loop is running.
        """
        body = json.dumps(incident_payload(incident, include_media=self.include_media)).encode("utf-8")
        last_exc: Exception | None = None
        start = time.monotonic()
        attempts = 0
        # ADR-001 §9: how delivery attempts terminate — either bound below is a TERMINAL failure,
        # dead-lettered with an explicit reason (never a silent give-up).
        reason = "retries_exhausted"
        for attempt in range(self.retries + 1):
            if (
                attempt > 0
                and self.max_attempt_age_s is not None
                and (time.monotonic() - start) > self.max_attempt_age_s
            ):
                reason = "max_attempt_age_exceeded"
                break  # the wall-clock bound tripped before the attempt budget did
            attempts += 1
            try:
                # Re-resolve the host against the policy immediately before sending (narrows DNS-rebinding
                # TOCTOU; a name public at construction but flipped to an internal IP is blocked here). A
                # block raises UrlPolicyError, which flows through the failure path below so it is
                # dead-lettered (with a redacted URL) instead of escaping uncaught before the DLQ record.
                self.url_policy.check_resolved(self.url)
                req = urllib.request.Request(self.url, data=body, headers=self._headers(body), method="POST")
                with _send(req, self.timeout_s) as resp:
                    status = getattr(resp, "status", 200)
                    if status >= 400:
                        raise RuntimeError(f"webhook sink failed with HTTP {status}")
                    return
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    # The ONLY time.sleep in this module, and it is unreachable from the event-loop
                    # thread: __call__ hands this whole method to a worker thread whenever a loop is
                    # running (tests/test_delivery_off_critical_path.py asserts both halves).
                    time.sleep(self.retry_backoff_s)
        if self.dead_letter is not None:
            # REDACTED: never persist the raw body, the signing HMAC, or auth headers in a DLQ record.
            # A content hash lets you correlate/dedupe without storing the payload or its credentials.
            self.dead_letter.append(
                {
                    "url": redact_url(self.url),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "error": repr(last_exc)[:500],
                    "attempts": attempts,
                    "reason": reason,
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
            "reason": record.get("reason"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


class BoundedDeadLetterQueue:
    """A bounded, in-memory dead-letter sink (ADR-001 §9: "delivery queue" bound).

    A bare ``list`` (the type ``WebhookSink(dead_letter=...)`` has always accepted) grows without
    limit — a sustained outage turns a 24/7 watch's DLQ into an unbounded leak. This wraps the same
    ``.append(record)`` seam with a hard ceiling on both COUNT (``max_records``) and estimated BYTES
    (``max_bytes``, the JSON size of each record): when either would be exceeded, the OLDEST queued
    record is evicted to make room. Every eviction is TRACED — never silent — through
    ``on_drop(record, reason)`` (``reason`` is ``"max_records"`` or ``"max_bytes"``); the default logs
    a structured line. Drop-in for the ``list`` a ``WebhookSink``/``FileDeadLetterQueue`` already write
    to (``append`` + iteration + :meth:`read_all`).
    """

    def __init__(
        self, *, max_records: int = 1000, max_bytes: int = 10_000_000,
        on_drop: "Callable[[dict[str, Any], str], None] | None" = None,
    ) -> None:
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.on_drop = on_drop
        self._items: deque[tuple[dict[str, Any], int]] = deque()
        self._bytes = 0

    @staticmethod
    def _size(record: dict[str, Any]) -> int:
        try:
            return len(json.dumps(record, default=str).encode("utf-8"))
        except Exception:  # noqa: BLE001 - size estimation must never crash a DLQ write
            return 0

    def append(self, record: dict[str, Any]) -> None:
        size = self._size(record)
        self._items.append((record, size))
        self._bytes += size
        self._evict()

    def _evict(self) -> None:
        while self._items and (len(self._items) > self.max_records or self._bytes > self.max_bytes):
            reason = "max_records" if len(self._items) > self.max_records else "max_bytes"
            dropped, size = self._items.popleft()
            self._bytes -= size
            if self.on_drop is not None:
                self.on_drop(dropped, reason)
            else:
                log(
                    get_logger("percept.runtime.sinks"), "dlq.dropped",
                    reason=reason, url=dropped.get("url"),
                )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return (rec for rec, _ in self._items)

    def read_all(self) -> list[dict[str, Any]]:
        return [rec for rec, _ in self._items]


class IdempotencyLedger:
    """Bounded, restart-surviving idempotency ledger keyed by ``event_id`` (= ``Incident.id`` — see
    ``percept/incidents.py``: "the stable, durable handle the rest of the system references").

    ADR-001 §6: delivery is at-least-once with a stable ``event_id`` that retries reuse; a duplicate
    delivery ATTEMPT for the SAME event_id (e.g. a post-restart redelivery of a persisted/replayed
    Incident) must not create a duplicate logical Incident downstream. This ledger is the Percept-side
    memory that makes that dedupe real: it remembers only event_ids that reached a TERMINAL-OK
    delivery. A retry of an in-flight/failed attempt (not yet terminal-OK) is deliberately NOT
    remembered here — it reuses the same event_id and is left to retry/redeliver normally (the sink's
    own ``retries``/``max_attempt_age_s`` + dead-letter queue already own that path; see
    :class:`WebhookSink`).

    BOUNDED (ADR-001 §9) with the same sizing philosophy as :class:`BoundedDeadLetterQueue`:
    capped by COUNT (``max_records``) and AGE (``ttl_s``) — drop-oldest-with-trace via
    ``on_evict(event_id, reason)`` (``reason`` is ``"max_records"`` or ``"ttl"``). A 24/7 watch must not
    let this grow without limit; an event_id that ages out of the ledger simply becomes eligible for
    (re)delivery again, same as one that was never delivered.

    SURVIVES RESTART: :meth:`snapshot`/:meth:`restore` round-trip only ``(event_id, ts)`` pairs — no
    secrets, no raw media, no incident payload (ADR-001 §1 lifecycle) — through the SAME
    ``Persistence`` backend (``percept.persistence``) the runtime already uses for goals/entities. The
    persistence boundary is deliberately narrow: this ledger is a durability BACKSTOP for the dedupe
    decision, not a store of the incidents themselves.
    """

    def __init__(
        self, *, max_records: int = 10_000, ttl_s: float | None = 3600.0,
        on_evict: "Callable[[str, str], None] | None" = None,
    ) -> None:
        self.max_records = max_records
        self.ttl_s = ttl_s
        self.on_evict = on_evict
        self._delivered: "dict[str, float]" = {}  # event_id -> ts of terminal-OK delivery, insertion-ordered

    def is_delivered(self, event_id: str, *, now: float | None = None) -> bool:
        """True iff ``event_id`` already reached a terminal-OK delivery and that record hasn't aged
        out. A record older than ``ttl_s`` (when both ``ttl_s`` and ``now`` are given) reads as
        NOT-delivered — a stale memory must never wedge a legitimately-retried event open forever."""
        ts = self._delivered.get(event_id)
        if ts is None:
            return False
        if self.ttl_s is not None and now is not None and (now - ts) > self.ttl_s:
            return False
        return True

    def mark_delivered(self, event_id: str, *, ts: float) -> None:
        """Record a TERMINAL-OK delivery for ``event_id``. Idempotent — marking an already-delivered
        id just refreshes its recency (moves it to the back of the LRU eviction order)."""
        self._delivered.pop(event_id, None)
        self._delivered[event_id] = ts
        self._evict(now=ts)

    def _evict(self, *, now: float) -> None:
        while self.ttl_s is not None and self._delivered:
            oldest_id, oldest_ts = next(iter(self._delivered.items()))
            if (now - oldest_ts) <= self.ttl_s:
                break
            self._delivered.pop(oldest_id)
            self._trace(oldest_id, "ttl")
        while len(self._delivered) > self.max_records:
            oldest_id, _oldest_ts = next(iter(self._delivered.items()))
            self._delivered.pop(oldest_id)
            self._trace(oldest_id, "max_records")

    def _trace(self, event_id: str, reason: str) -> None:
        if self.on_evict is not None:
            self.on_evict(event_id, reason)
        else:
            log(get_logger("percept.runtime.sinks"), "idempotency_ledger.evicted", reason=reason, event_id=event_id)

    def __len__(self) -> int:
        return len(self._delivered)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._delivered

    def snapshot(self) -> list[list]:
        """JSON-able snapshot: ``[[event_id, ts], ...]``, insertion-ordered. No secrets/media."""
        return [[k, v] for k, v in self._delivered.items()]

    def restore(self, snap: Any) -> None:
        """Rehydrate from :meth:`snapshot`. Missing/malformed input is a no-op — a corrupt or absent
        ledger snapshot must never block a restart, it just means the ledger starts blank."""
        if not snap:
            return
        for item in snap:
            try:
                event_id, ts = item[0], float(item[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            self._delivered[str(event_id)] = ts
        if self._delivered:
            self._evict(now=max(self._delivered.values()))


class IdempotentSink:
    """Wrap a notification sink (``on_incident``) with an :class:`IdempotencyLedger` — the production
    dedupe-on-redelivery guard in the delivery path (ADR-001 §6).

    Consults the ledger by ``incident.id`` (the event_id) BEFORE delivering: if this event_id already
    reached a terminal-OK delivery, the call is a no-op — the wrapped sink is NOT invoked again, so a
    post-restart redelivery of the same event never reaches a downstream consumer a second time. A
    retry of an in-flight/failed attempt (event_id not yet marked delivered) passes straight through,
    unchanged from today. The terminal-OK outcome is recorded ONLY after the wrapped sink returns
    without raising — a failed/raising attempt leaves the ledger untouched, so it stays eligible for
    retry (matching :class:`WebhookSink`'s own retry/dead-letter semantics).

    Handles both sync and async sinks (``on_incident`` may be either — see ``worker._call_sink``):
    a coroutine result is marked delivered only once it actually completes without raising.
    """

    def __init__(
        self, sink: Callable, ledger: "IdempotencyLedger", *, clock: "Callable[[], float] | None" = None,
    ) -> None:
        self._sink = sink
        self._ledger = ledger
        self._clock = clock or time.monotonic

    def __call__(self, incident: Any) -> Any:
        event_id = getattr(incident, "id", None)
        now = self._clock()
        if event_id is not None and self._ledger.is_delivered(event_id, now=now):
            return None  # already terminal-OK — dedupe: no re-delivery, no duplicate logical incident
        result = self._sink(incident)
        if inspect.isawaitable(result):
            return self._await_and_mark(result, event_id)
        if event_id is not None:
            self._ledger.mark_delivered(event_id, ts=now)
        return result

    async def _await_and_mark(self, awaitable: Any, event_id: str | None) -> Any:
        out = await awaitable
        if event_id is not None:
            self._ledger.mark_delivered(event_id, ts=self._clock())
        return out


class SupersessionLedger:
    """Bounded ``revision_id -> delivered incident.id`` map for the provisional/superseding delivery
    path.

    The episode fold names the revision a correction supersedes as a stable revision handle
    (``<episode_id>#r<n>``, :func:`percept.behavior.revision_id`), NOT the prior INCIDENT's id — the
    incident id is minted only here, at the delivery boundary. This ledger is the small memory that
    lets a superseding revision be resolved back to the actual prior incident id, so a consumer sees
    ``supersedes=<prior incident id>`` and not just an opaque revision string.

    BOUNDED with the same sizing philosophy as :class:`IdempotencyLedger` / :class:`BoundedDeadLetterQueue`
    (ADR-001 §9): capped by COUNT (``max_records``), drop-oldest-with-trace via
    ``on_evict(revision_id, reason)`` (``reason`` is always ``"max_records"``). A 24/7 watch must not
    let this grow without limit; a revision that ages out simply resolves to ``None`` (an
    unresolved-but-still-TRACED supersession — never a silent drop, see :class:`SupersessionSink`).
    """

    def __init__(
        self, *, max_records: int = 10_000, on_evict: "Callable[[str, str], None] | None" = None,
    ) -> None:
        self.max_records = max_records
        self.on_evict = on_evict
        self._by_revision: "dict[str, str]" = {}  # revision_id -> incident.id, insertion-ordered (LRU)

    def record(self, revision_id: str, incident_id: str) -> None:
        """Remember that ``revision_id`` was delivered as incident ``incident_id``."""
        self._by_revision.pop(revision_id, None)
        self._by_revision[revision_id] = incident_id
        while len(self._by_revision) > self.max_records:
            oldest = next(iter(self._by_revision))
            self._by_revision.pop(oldest)
            if self.on_evict is not None:
                self.on_evict(oldest, "max_records")
            else:
                log(get_logger("percept.runtime.sinks"), "supersession_ledger.evicted",
                    reason="max_records", revision_id=oldest)

    def resolve(self, revision_id: str) -> "str | None":
        """The incident id a prior ``revision_id`` was delivered as, or ``None`` if unknown/aged out."""
        return self._by_revision.get(revision_id)

    def __len__(self) -> int:
        return len(self._by_revision)

    def __contains__(self, revision_id: str) -> bool:
        return revision_id in self._by_revision


class SupersessionSink:
    """Wrap a notification sink (``on_incident``) to relate a SUPERSEDING revision to the incident it
    corrects, and TRACE that relation — never silent.

    The delivery model: a finding is emitted PROVISIONALLY on the rising edge; when the mutation
    channel later classifies a turn as replace/retract on the same episode, the fold fires a superseding
    revision carrying ``supersedes=<prior revision>`` (surfaced on the fire, then copied onto the
    incident as a runtime-local annotation by :func:`percept.incidents.make_incident`). This wrapper is
    the delivery-boundary piece that:

    * resolves the fold's revision-level ``supersedes`` to the PRIOR INCIDENT's id via a bounded
      :class:`SupersessionLedger`, and stamps it onto the outgoing incident as
      ``incident._supersedes_incident_id`` (read via :func:`superseded_incident_id`) — so an in-process
      ``async for incident in run(...)`` consumer sees ``supersedes=<prior incident id>`` exactly as
      needed, WITHOUT a wire-contract change;
    * TRACES every supersession (a structured log line by default, or ``on_supersede(record)`` if
      supplied) so a client debugging the stream can SEE the original→correction link even when the
      downstream sink itself carries no supersession field — "trace the supersession, never silent".

    It then delegates to the wrapped sink UNCHANGED (sync or async — the coroutine is returned as-is for
    the caller to await, exactly like ``worker._call_sink`` already does). An incident with no
    supersession annotation (the common provisional case) passes straight through, its revision recorded
    for a possible future correction; a sink with nothing to relate is byte-for-byte the bare sink.
    """

    def __init__(
        self,
        sink: Callable,
        *,
        ledger: "SupersessionLedger | None" = None,
        on_supersede: "Callable[[dict[str, Any]], None] | None" = None,
    ) -> None:
        self._sink = sink
        self._ledger = ledger if ledger is not None else SupersessionLedger()
        self._on_supersede = on_supersede

    def __call__(self, incident: Any) -> Any:
        revision = incident_revision(incident)
        supersedes = incident_supersedes(incident)
        if supersedes is not None:
            prior_incident_id = self._ledger.resolve(supersedes)
            # Stamp the resolved prior INCIDENT id (the consumer-facing "supersedes=<prior incident
            # id>"); None when the prior revision aged out of the bounded ledger — still traced below.
            incident._supersedes_incident_id = prior_incident_id  # type: ignore[attr-defined]
            self._trace(incident, prior_revision=supersedes, prior_incident_id=prior_incident_id)
        if revision is not None:
            self._ledger.record(revision, getattr(incident, "id", None) or "")
        return self._sink(incident)

    def _trace(self, incident: Any, *, prior_revision: str, prior_incident_id: "str | None") -> None:
        record = {
            "incident_id": getattr(incident, "id", None),
            "concern_id": getattr(incident, "concern_id", None),
            "revision_id": incident_revision(incident),
            "supersedes_revision": prior_revision,
            "supersedes_incident_id": prior_incident_id,
        }
        if self._on_supersede is not None:
            self._on_supersede(record)
        else:
            log(get_logger("percept.runtime.sinks"), "incident.superseded", **record)


def superseded_incident_id(incident: Any) -> "str | None":
    """The PRIOR incident id a superseding revision replaces, once :class:`SupersessionSink` has
    resolved it. ``None`` for an original incident, or when the prior revision aged out of
    the bounded ledger before this correction arrived."""
    return getattr(incident, "_supersedes_incident_id", None)


def webhook(url: str, **kwargs) -> WebhookSink:
    """Convenience factory for ``run(..., on_incident=webhook(url))``.

    By default the URL must be **https to a public host** (SSRF-safe — see :class:`UrlPolicy`). For a
    local/dev sink (``http://localhost:8080`` …) pass ``url_policy=UrlPolicy.dev()``::

        webhook("http://localhost:8080/hook", url_policy=UrlPolicy.dev())
    """
    return WebhookSink(url, **kwargs)
