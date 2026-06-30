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


class WebhookSinkTests(unittest.TestCase):
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
        # credentialed URL + signing secret: NEITHER may appear in the DLQ record.
        sink = WebhookSink(
            "https://user:pass@example.test/hook",
            retries=1, retry_backoff_s=0, dead_letter=dead, signing_secret="shh",
        )
        with patch("percept.runtime.sinks._send", side_effect=TimeoutError("slow")):
            with self.assertRaises(TimeoutError):
                sink(_incident())
        self.assertEqual(len(dead), 1)
        rec = dead[0]
        self.assertEqual(set(rec), {"url", "body_sha256", "error", "attempts"})
        self.assertEqual(rec["url"], "https://example.test/hook")  # user:pass@ stripped
        self.assertEqual(len(rec["body_sha256"]), 64)
        self.assertEqual(rec["attempts"], 2)
        # no raw payload, no signing secret, no credentials anywhere in the serialized record
        blob = json.dumps(rec)
        for leaked in ("Door alert", "shh", "user:pass"):
            self.assertNotIn(leaked, blob)

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
            self.assertEqual(rows[0]["url"], "https://example.test/hook")
            self.assertEqual(len(rows[0]["body_sha256"]), 64)
            self.assertNotIn("headers", rows[0])
            self.assertNotIn("body", rows[0])
            self.assertNotIn("shh", "\n".join(json.dumps(r) for r in rows))


class UrlPolicyTests(unittest.TestCase):
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

    def test_default_allows_public_https(self):
        UrlPolicy().validate("https://hooks.example.com/abc")  # unresolved public name -> allowed

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
