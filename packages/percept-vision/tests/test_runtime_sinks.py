"""Runtime sink tests — webhook serialization, signing, retry, and dead-letter behavior."""

from __future__ import annotations

import hmac
import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from percept.runtime.sinks import (
    BoundedDeadLetterQueue,
    FileDeadLetterQueue,
    UrlPolicy,
    UrlPolicyError,
    WebhookSink,
    _NoRedirect,
    incident_payload,
)
from percept.contracts import Incident, IncidentSnippet, Verdict


def _incident():
    return Incident(
        id="inc-1",
        concern_id="door",
        verdict=Verdict(True, 0.91, "person at the door"),
        ts=1.25,
        message="Door alert",
        snippet=IncidentSnippet(kind="clip", media=b"secret", mime="video/mp4", t_start=0.5, t_end=1.25),
    )


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _gai(ip: str):
    """A minimal socket.getaddrinfo() return value resolving to a single ``ip``."""
    return [(2, 1, 6, "", (ip, 0))]


class WebhookSinkTests(unittest.TestCase):
    def setUp(self):
        # Send-time SSRF re-check calls socket.getaddrinfo; stub it so unit tests never touch the network
        # (an unresolvable host -> check_resolved skips, exactly as offline). Tests needing a specific
        # resolution override this patch locally.
        p = patch("percept.runtime.sinks.socket.getaddrinfo", side_effect=OSError("no DNS in tests"))
        p.start()
        self.addCleanup(p.stop)

    def test_payload_redacts_media_by_default(self):
        payload = incident_payload(_incident())
        self.assertIsNone(payload["snippet"]["media"])
        self.assertEqual(payload["snippet"]["mime"], "video/mp4")

    def test_payload_can_include_media(self):
        payload = incident_payload(_incident(), include_media=True)
        self.assertEqual(payload["snippet"]["media"]["base64"], "c2VjcmV0")

    def test_posts_signed_json(self):
        seen = []

        def fake_urlopen(req, timeout):
            seen.append((req, timeout))
            return _Resp()

        sink = WebhookSink("https://example.test/hook", signing_secret="shh", timeout_s=3.0)
        with patch("percept.runtime.sinks._send", fake_urlopen):
            sink(_incident())
        req, timeout = seen[0]
        body = req.data
        expected = hmac.new(b"shh", body, sha256).hexdigest()
        self.assertEqual(timeout, 3.0)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers["X-percept-signature"], f"sha256={expected}")
        self.assertEqual(json.loads(body)["message"], "Door alert")

    def test_retries_then_succeeds(self):
        attempts = []

        def fake_urlopen(req, timeout):
            attempts.append(req)
            if len(attempts) == 1:
                raise TimeoutError("slow")
            return _Resp()

        sink = WebhookSink("https://example.test/hook", retries=1, retry_backoff_s=0)
        with patch("percept.runtime.sinks._send", fake_urlopen):
            sink(_incident())
        self.assertEqual(len(attempts), 2)

    def test_dead_letter_records_are_redacted(self):
        dead = []
        # a Slack-style URL with the secret in the PATH + userinfo + signing secret: NONE may appear.
        sink = WebhookSink(
            "https://user:pass@hooks.example.com/services/T000/B000/SECRETTOKEN?token=QUERYSECRET",
            retries=1, retry_backoff_s=0, dead_letter=dead, signing_secret="shh",
        )
        with patch("percept.runtime.sinks._send", side_effect=TimeoutError("slow")):
            with self.assertRaises(TimeoutError):
                sink(_incident())
        self.assertEqual(len(dead), 1)
        rec = dead[0]
        self.assertEqual(set(rec), {"url", "body_sha256", "error", "attempts", "reason"})
        self.assertEqual(rec["url"], "https://hooks.example.com")  # userinfo + path + query stripped
        self.assertEqual(len(rec["body_sha256"]), 64)
        self.assertEqual(rec["attempts"], 2)
        self.assertEqual(rec["reason"], "retries_exhausted")  # generic cause, never the URL/a secret
        # no raw payload, no signing secret, no path/query/userinfo secret anywhere in the record —
        # including the new `reason` field specifically (ADR-001 §9: a drop reason must be TRACED but
        # must never itself become a secret-leak vector).
        blob = json.dumps(rec)
        for leaked in ("Door alert", "shh", "user:pass", "SECRETTOKEN", "QUERYSECRET"):
            self.assertNotIn(leaked, blob)
            self.assertNotIn(leaked, rec["reason"])

    def test_max_attempt_age_terminates_delivery_before_retries_exhausted(self):
        # ADR-001 §9 / card A9: "delivery attempts" bound. A wall-clock cap independent of the retry
        # COUNT — even with retries left in the budget, exceeding max_attempt_age_s is a TERMINAL
        # failure (dead-lettered with an explicit, distinct reason), not silently swallowed.
        dead = []
        calls = []
        clock = [0.0]

        def fake_urlopen(req, timeout):
            calls.append(req)
            clock[0] += 10.0  # each attempt "takes" 10s of wall-clock time
            raise TimeoutError("slow")

        sink = WebhookSink(
            "https://example.test/hook", retries=5, retry_backoff_s=0,
            max_attempt_age_s=15.0, dead_letter=dead,
        )
        with patch("percept.runtime.sinks._send", fake_urlopen), \
             patch("percept.runtime.sinks.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                sink(_incident())
        self.assertLess(len(calls), 6, "the age bound must cut the run short of the full retry budget")
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["reason"], "max_attempt_age_exceeded")

    def test_bounded_dead_letter_queue_drops_oldest_with_traced_reason_on_count_overflow(self):
        # ADR-001 §9 / card A9: "delivery queue (count + bytes) -> dead-letter/drop-with-reason". A
        # plain list dead_letter grows without bound; BoundedDeadLetterQueue caps it and TRACES every
        # eviction — never silent.
        dropped = []
        dlq = BoundedDeadLetterQueue(max_records=3, on_drop=lambda rec, reason: dropped.append((rec["i"], reason)))
        for i in range(5):
            dlq.append({"i": i})
        self.assertEqual(len(dlq), 3, "the queue never exceeds max_records")
        self.assertEqual([r["i"] for r in dlq.read_all()], [2, 3, 4], "oldest records fall out first")
        self.assertEqual(dropped, [(0, "max_records"), (1, "max_records")])

    def test_bounded_dead_letter_queue_drops_oldest_with_traced_reason_on_byte_overflow(self):
        dropped = []
        # Sized to hold exactly ONE ~120-byte record but not two — the byte bound must evict the
        # oldest to make room for the second, well short of max_records (1000).
        dlq = BoundedDeadLetterQueue(
            max_records=1000, max_bytes=150, on_drop=lambda rec, reason: dropped.append(reason))
        dlq.append({"i": 0, "payload": "x" * 100})
        dlq.append({"i": 1, "payload": "y" * 100})
        self.assertEqual(len(dlq), 1, "the byte bound evicts even though max_records is nowhere near hit")
        self.assertEqual(dlq.read_all()[0]["i"], 1, "the newest record survives")
        self.assertEqual(dropped, ["max_bytes"])

    def test_bounded_dead_letter_queue_default_drop_is_traced_via_log_not_silent(self):
        # No on_drop supplied: the default path must still observably log the drop (never a no-op).
        dlq = BoundedDeadLetterQueue(max_records=1)
        with self.assertLogs("percept.runtime.sinks", level="INFO") as cm:
            dlq.append({"url": "https://a.example.com"})
            dlq.append({"url": "https://b.example.com"})
        self.assertTrue(any("dlq.dropped" in line for line in cm.output))

    def test_send_time_rebind_to_internal_is_blocked_and_dead_lettered(self):
        # A name public at construction (passes check_static) that RESOLVES to an internal IP at send is
        # blocked by check_resolved — and because the check now runs inside the retry/DLQ path, the block
        # is dead-lettered (redacted) rather than escaping uncaught before any DLQ record is written.
        dead = []
        sink = WebhookSink("https://flips.example.com/hook", dead_letter=dead)
        with patch("percept.runtime.sinks.socket.getaddrinfo", return_value=_gai("169.254.169.254")):
            with self.assertRaises(UrlPolicyError):
                sink(_incident())
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["url"], "https://flips.example.com")

    def test_file_dead_letter_records_jsonl_redacted(self):
        with TemporaryDirectory() as d:
            queue = FileDeadLetterQueue(Path(d) / "dead" / "webhooks.jsonl")
            sink = WebhookSink(
                "https://example.test/hook",
                retries=0,
                dead_letter=queue,
                signing_secret="shh",
            )
            with patch("percept.runtime.sinks._send", side_effect=TimeoutError("slow")):
                with self.assertRaises(TimeoutError):
                    sink(_incident())
            rows = queue.read_all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["url"], "https://example.test")
            self.assertEqual(len(rows[0]["body_sha256"]), 64)
            self.assertNotIn("headers", rows[0])
            self.assertNotIn("body", rows[0])
            self.assertNotIn("shh", "\n".join(json.dumps(r) for r in rows))


class UrlPolicyTests(unittest.TestCase):
    def setUp(self):
        # Default: names are unresolvable (offline) so hostname tests are hermetic. Resolution-specific
        # tests override this patch.
        p = patch("percept.runtime.sinks.socket.getaddrinfo", side_effect=OSError("no DNS in tests"))
        p.start()
        self.addCleanup(p.stop)

    def test_default_blocks_non_https_and_file(self):
        for bad in ("http://example.test/hook", "file:///etc/passwd", "ftp://example.test/x"):
            with self.assertRaises(UrlPolicyError):
                UrlPolicy().validate(bad)

    def test_default_blocks_metadata_loopback_and_private(self):
        for bad in (
            "https://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
            "https://127.0.0.1/hook",                      # loopback
            "https://localhost/hook",                      # loopback name
            "https://10.0.0.5/hook",                       # private
            "https://[::1]/hook",                          # IPv6 loopback
        ):
            with self.assertRaises(UrlPolicyError):
                UrlPolicy().validate(bad)

    def test_default_blocks_cgnat_literal(self):
        # RFC 6598 CGNAT — internal, but ip.is_global only catches it on CPython >= 3.12.4
        with self.assertRaises(UrlPolicyError):
            UrlPolicy().validate("https://100.64.0.1/hook")

    def test_default_allows_public_https(self):
        UrlPolicy().validate("https://hooks.example.com/abc")  # unresolved public name -> allowed

    def test_resolved_name_to_private_is_blocked(self):
        # a name that RESOLVES to an internal IP is blocked at send-time check_resolved (not just literals)
        with patch("percept.runtime.sinks.socket.getaddrinfo", return_value=_gai("10.0.0.5")):
            with self.assertRaises(UrlPolicyError):
                UrlPolicy().validate("https://sneaky.example.com/hook")

    def test_resolved_name_to_public_is_allowed(self):
        with patch("percept.runtime.sinks.socket.getaddrinfo", return_value=_gai("93.184.216.34")):
            UrlPolicy().validate("https://ok.example.com/hook")  # resolves public -> allowed

    def test_redact_url_drops_path_query_and_brackets_ipv6(self):
        from percept.runtime.sinks import redact_url
        self.assertEqual(
            redact_url("https://u:p@hooks.example.com/services/T/B/SECRET?token=x"),
            "https://hooks.example.com",
        )
        self.assertEqual(redact_url("https://[2001:db8::1]:8443/hook"), "https://[2001:db8::1]:8443")

    def test_redact_url_tolerates_malformed_port(self):
        from percept.runtime.sinks import redact_url
        # SplitResult.port raises ValueError on out-of-range/non-numeric ports; redact_url must NOT crash
        # (it runs on the DLQ path when a delivery has already failed) — it drops the bad port.
        self.assertEqual(redact_url("https://hooks.example.com:99999/hook?token=x"), "https://hooks.example.com")
        self.assertEqual(redact_url("https://hooks.example.com:notaport/x"), "https://hooks.example.com")

    def test_blocked_redirect_error_is_redacted(self):
        # _NoRedirect embeds the redirect target in the raised error, which lands in the DLQ 'error' field;
        # its path/query (a secret Location) must be redacted.
        with self.assertRaises(UrlPolicyError) as cm:
            _NoRedirect().redirect_request(
                None, None, 302, "Found", {}, "https://internal.svc/callback?token=SUPERSECRET"
            )
        self.assertNotIn("SUPERSECRET", str(cm.exception))
        self.assertIn("https://internal.svc", str(cm.exception))

    def test_dev_policy_allows_http_and_loopback(self):
        UrlPolicy.dev().validate("http://localhost:8080/hook")
        UrlPolicy.dev().validate("http://127.0.0.1/hook")

    def test_allowlist_restricts_hosts(self):
        pol = UrlPolicy(allowed_hosts=frozenset({"hooks.example.com"}))
        pol.validate("https://hooks.example.com/x")
        with self.assertRaises(UrlPolicyError):
            pol.validate("https://evil.example.com/x")

    def test_webhook_constructor_rejects_unsafe_url(self):
        with self.assertRaises(UrlPolicyError):
            WebhookSink("http://169.254.169.254/")  # default policy blocks at construction

    def test_redirect_handler_refuses_to_follow(self):
        with self.assertRaises(UrlPolicyError):
            _NoRedirect().redirect_request(None, None, 302, "Found", {}, "http://169.254.169.254/")


if __name__ == "__main__":
    unittest.main()
