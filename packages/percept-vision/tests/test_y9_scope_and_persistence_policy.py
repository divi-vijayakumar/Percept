"""Card Y9 — the v1.4 scope and persistence decisions must stay stated, and stay TRUE.

Two paying client teams read percept 0.7.0 as promising more than it does. Card Y9 wrote the founder
decisions down in the three places a reader actually looks: ADR-006's Consequences, the module
docstrings of ``percept.sourceprofile`` / ``percept.episodes``, and a client-facing reference doc.

These are DRIFT GUARDS, not prose review. Two kinds of check:

1. **The statement is present** where a reader will hit it — deleting or softening it fails here.
2. **The statement is true against the code** — specifically that ``Percept.snapshot()`` really does
   not carry episodes, so claim (b) cannot silently become a lie if snapshot() grows a key.

If a later phase genuinely persists episodes, the failing test in :class:`PersistenceIsSessionOnly`
is the signal to go rewrite the policy text — not to loosen the assertion.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import percept.episodes as episodes_mod
import percept.sourceprofile as sourceprofile_mod
from percept import Percept
from percept.backends.fake import FakeVision
from percept.episodes import EpisodeLog

_PKG = Path(inspect.getfile(episodes_mod)).resolve().parents[1]  # packages/percept-vision
_REPO = _PKG.parents[1]
_ADR = _REPO / ".adr" / "ADR-006-incident-relations.md"
_SCOPE_DOC = _REPO / "docs" / "reference" / "v1-4-scope-and-persistence.md"

#: The three exclusions are scope, not backlog. Phrased as substrings so wording can breathe but the
#: named exclusions cannot quietly disappear.
_EXCLUSIONS = ("biometric re-identification", "prediction", "robotics control")


def _doc(path: Path) -> str:
    if not path.exists():  # docs/ live outside the installed package
        raise unittest.SkipTest(f"{path} not present in this checkout")
    return path.read_text(encoding="utf-8").lower()


class ScopeIsStatedNarrow(unittest.TestCase):
    """(a) v1.4 covers two subject kinds; three things are EXPLICITLY excluded."""

    def _assert_scope_text(self, text: str, where: str) -> None:
        low = text.lower()
        self.assertIn("stable content key", low, f"{where}: screen-document scope missing")
        self.assertIn("claim slot", low, f"{where}: conversational claim-slot scope missing")
        for excluded in _EXCLUSIONS:
            self.assertIn(excluded, low, f"{where}: exclusion {excluded!r} missing")

    def test_sourceprofile_docstring_states_the_narrow_scope(self):
        self._assert_scope_text(sourceprofile_mod.__doc__ or "", "percept.sourceprofile")

    def test_adr006_consequences_state_the_narrow_scope(self):
        self._assert_scope_text(_doc(_ADR), "ADR-006")

    def test_client_facing_doc_states_the_narrow_scope(self):
        self._assert_scope_text(_doc(_SCOPE_DOC), "docs/reference/v1-4-scope-and-persistence.md")

    def test_the_resolver_seam_is_still_open(self):
        """The scope is narrow BECAUSE the seam is open — a closed enum would be a different claim."""
        self.assertIsInstance(sourceprofile_mod.SourceProfile.__annotations__["identity_strategy"], str)
        doc = (sourceprofile_mod.__doc__ or "").lower()
        self.assertIn("open registry", doc)
        self.assertIn("does not generalise ahead of evidence", doc)


class PersistenceIsSessionOnly(unittest.TestCase):
    """(b) episodes are process-lifetime state — asserted against the code, not only the prose."""

    @staticmethod
    def _agent() -> Percept:
        return Percept.create(vision=FakeVision(script={}), discover_plugins=False)

    def test_snapshot_does_not_carry_episodes(self):
        p = self._agent()
        p.episodes["g1"] = EpisodeLog("g1")
        snap = p.snapshot()
        self.assertNotIn("episodes", snap)
        self.assertNotIn("episodes", repr(snap), "no episode state may ride along inside snapshot()")

    def test_restore_cannot_rehydrate_episodes(self):
        source = self._agent()
        source.episodes["g1"] = EpisodeLog("g1")
        restored = self._agent()  # a fresh process = a restart
        restored.restore(source.snapshot())
        self.assertEqual(restored.episodes, {}, "a restarted process starts at zero episodes")

    def test_episodes_docstring_says_session_only_and_blocks_24_7(self):
        doc = (episodes_mod.__doc__ or "").lower()
        self.assertIn("session-only", doc)
        self.assertIn("do not survive a restart", doc)
        self.assertIn("24/7", doc)

    def test_durable_scope_caveat_travels_with_the_field_not_a_distant_doc(self):
        """``scope='durable'`` is IDENTIFIER stability. Both modules a reader opens must say so."""
        for mod in (episodes_mod, sourceprofile_mod):
            doc = (mod.__doc__ or "").lower()
            with self.subTest(module=mod.__name__):
                self.assertIn("durable", doc)
                self.assertIn("identifier", doc)
                self.assertIn("restart", doc)

    def test_the_docs_say_it_too(self):
        for text, where in ((_doc(_ADR), "ADR-006"), (_doc(_SCOPE_DOC), "scope doc")):
            with self.subTest(where=where):
                self.assertIn("session-only", text)
                self.assertIn("24/7", text)
                self.assertIn("scope='durable'", text)


class DeliveryIsProvisionalPlusSupersede(unittest.TestCase):
    """(c) emit immediately and supersede; delay is concern-specific and SLO-bounded."""

    def _assert_delivery_text(self, text: str, where: str) -> None:
        self.assertIn("supersede", text, f"{where}: supersede posture missing")
        self.assertIn("provisional", text, f"{where}: provisional-incident posture missing")
        self.assertIn("concern-specific", text, f"{where}: delay must be concern-specific")
        self.assertIn("slo", text, f"{where}: delay must be bounded by a declared latency SLO")

    def test_adr006_states_the_delivery_posture(self):
        self._assert_delivery_text(_doc(_ADR), "ADR-006")

    def test_client_facing_doc_states_the_delivery_posture(self):
        self._assert_delivery_text(_doc(_SCOPE_DOC), "scope doc")

    def test_no_wide_settle_window_is_promised(self):
        """A global settle window would destroy the rising edge; both docs must reject it by name."""
        for text, where in ((_doc(_ADR), "ADR-006"), (_doc(_SCOPE_DOC), "scope doc")):
            with self.subTest(where=where):
                self.assertIn("settle window", text)
                self.assertIn("not a wide settle window", text.replace("**", ""))


class TheDocIsReachable(unittest.TestCase):
    """A policy nobody can find is a policy nobody reads."""

    def test_docs_index_and_integration_guide_link_the_scope_doc(self):
        target = "reference/v1-4-scope-and-persistence.md"
        for rel in ("docs/README.md", "docs/how-to/integrate-into-your-app.md"):
            with self.subTest(doc=rel):
                self.assertIn(target, _doc(_REPO / rel))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
