"""Runtime sink tests — webhook serialization, signing, retry, and dead-letter behavior."""

from __future__ import annotations

import hmac
import json
import unittest
from hashlib import sha256
from unittest.mock import patch

from percept.runtime.sinks import WebhookSink, incident_payload
from percept_contracts import Incident, IncidentSnippet, Verdict


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
        with patch("urllib.request.urlopen", fake_urlopen):
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
        with patch("urllib.request.urlopen", fake_urlopen):
            sink(_incident())
        self.assertEqual(len(attempts), 2)

    def test_dead_letter_records_failed_delivery(self):
        dead = []
        sink = WebhookSink("https://example.test/hook", retries=1, retry_backoff_s=0, dead_letter=dead)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
            with self.assertRaises(TimeoutError):
                sink(_incident())
        self.assertEqual(len(dead), 1)
        self.assertIn(b"Door alert", dead[0]["body"])


if __name__ == "__main__":
    unittest.main()
