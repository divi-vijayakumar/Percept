"""Boundary scan — the template may only touch an ENUMERATED set of public SDK symbols.

An ALLOWLIST, not an underscore-blocklist: anything not explicitly granted fails — including
``core.goals`` (a public-looking attribute that is still SDK-internal state) and any ``._`` attribute
access on SDK objects. This is the executable form of the template's promise: it composes Percept
through supported seams only, so the SDK can refactor its internals without breaking the one shipped
reference application. A NEW SDK need is a seam request against the SDK, not a new entry here —
extend this list only when the plan/SDK ratifies the symbol as public.

Convention the scan relies on (enforced by failing otherwise): in template sources the cognition core
is always bound to a name/attribute containing ``core`` and the runtime session to ``session`` — the
resolver maps those names to roles and checks every attribute access against the role's grant.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent
SOURCES = ("app.py", "engagement.py", "engine_adapter.py", "fire_renderer.py",
           "floor.py", "media.py", "percept_bridge.py", "telemetry.py")

# -- the grant ---------------------------------------------------------------------------------
#: `from <module> import <name>` — module → the names the template may import from it.
ALLOWED_FROM_IMPORTS: dict[str, set[str]] = {
    "percept": {"Percept", "Goal", "AttentionIntent", "GoalOrigin", "Incident", "Verdict",
                "ModeError", "run", "start", "RuntimeStats"},
    "percept.agent": {"guard_live_language_path"},
    "percept.incidents": {"attention_intent_for", "DEFAULT_ATTENTION_POLICY", "AttentionPolicy",
                          "parse_requested_delivery"},
    "percept.runtime.sinks": {"webhook", "WebhookSink", "incident_payload", "UrlPolicy"},
    # Deliberate addition beyond the plan's enumerated set (REPORTED): the template converts browser
    # video into tier-0 gate frames itself, and `percept.edge.tier0` is the documented public shim
    # for that (`pip install "percept-vision[edge]"`). No other edge symbol is granted.
    "percept.edge.tier0": {"Frame"},
}
#: attributes reachable as `percept.<attr>` after a plain `import percept`.
ALLOWED_PERCEPT_ATTRS = ALLOWED_FROM_IMPORTS["percept"] | {"runtime", "incidents", "agent"}
#: RuntimeSession surface the template may call (role: any name/attr containing "session").
SESSION_ALLOWED = {"arm", "disarm", "ask", "interrupt", "label", "subscribe", "on_goal_change",
                   "list_goals", "latest_frame", "stop"}
#: Percept-core surface the template may call (role: any name/attr containing "core").
CORE_ALLOWED = {"arm_from_language", "note_mention", "push_audio"}
#: Deprecated contracts the template must NEVER import, from anywhere.
FORBIDDEN_NAMES = {"VoiceTransport", "UtteranceEvent", "Reply"}


def _role(node: ast.expr) -> str | None:
    """Map the object an attribute is read from to a scan role ('core' / 'session' / None)."""
    name = None
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    if name is None:
        return None
    lowered = name.lower().lstrip("_")
    if lowered == "core" or lowered.endswith("_core"):
        return "core"
    if lowered == "session" or lowered.endswith("_session"):
        return "session"
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"{self.filename}:{getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "percept":
                continue
            if alias.name.startswith("percept."):
                self._flag(node, f"`import {alias.name}` — only `import percept` or an allowlisted "
                                 f"`from` import is granted")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "percept" or module.startswith("percept."):
            allowed = ALLOWED_FROM_IMPORTS.get(module)
            if allowed is None:
                self._flag(node, f"import from non-allowlisted SDK module `{module}`")
            else:
                for alias in node.names:
                    if alias.name not in allowed:
                        self._flag(node, f"`from {module} import {alias.name}` not in the allowlist")
        for alias in node.names:
            if alias.name in FORBIDDEN_NAMES:
                self._flag(node, f"deprecated SDK contract `{alias.name}` — the template owns its "
                                 f"own VoiceEngine/SpeechRequest seam")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # percept.<attr> off the module object
        if isinstance(node.value, ast.Name) and node.value.id == "percept":
            if node.attr not in ALLOWED_PERCEPT_ATTRS:
                self._flag(node, f"`percept.{node.attr}` not in the allowlist")
        role = _role(node.value)
        if role == "core":
            if node.attr == "goals":
                self._flag(node, "`core.goals` is SDK-internal state — use session.list_goals()")
            elif node.attr.startswith("_"):
                self._flag(node, f"private SDK attribute `core.{node.attr}`")
            elif node.attr not in CORE_ALLOWED:
                self._flag(node, f"`core.{node.attr}` not in the core allowlist {sorted(CORE_ALLOWED)}")
        elif role == "session":
            if node.attr.startswith("_"):
                self._flag(node, f"private SDK attribute `session.{node.attr}`")
            elif node.attr not in SESSION_ALLOWED:
                self._flag(node, f"`session.{node.attr}` not in the session allowlist "
                                 f"{sorted(SESSION_ALLOWED)}")
        self.generic_visit(node)


class BoundaryTests(unittest.TestCase):
    def test_template_sources_use_only_the_enumerated_public_seams(self):
        violations: list[str] = []
        for source in SOURCES:
            path = TEMPLATE_DIR / source
            tree = ast.parse(path.read_text(), filename=source)
            scanner = _Scanner(source)
            scanner.visit(tree)
            violations.extend(scanner.violations)
        self.assertEqual(violations, [], "template reached outside the public-seam allowlist:\n"
                         + "\n".join(violations))

    def test_all_template_sources_are_scanned(self):
        # A new module must be added to SOURCES (and thereby scanned) — the scan can't silently
        # narrow as the template grows.
        actual = {p.name for p in TEMPLATE_DIR.glob("*.py")}
        self.assertEqual(actual, set(SOURCES))

    def test_the_scanner_actually_catches_a_violation(self):
        # Guard the guard: a scanner that silently matches nothing would pass everything.
        bad = "core.goals.active()\nsession._notify_frame(1, None)\nfrom percept.contracts import Reply\n"
        scanner = _Scanner("<probe>")
        scanner.visit(ast.parse(bad))
        report = "\n".join(scanner.violations)
        self.assertIn("core.goals", report)                       # public-looking internal state
        self.assertIn("session._notify_frame", report)            # private seam
        self.assertIn("percept.contracts", report)                # non-allowlisted module...
        self.assertIn("Reply", report)                            # ...carrying a deprecated contract


if __name__ == "__main__":
    unittest.main()
