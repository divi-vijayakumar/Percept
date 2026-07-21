"""Schema + validator for the synthetic UNIT-layer capability corpora.

The founder's split, which shapes every line here:

    CAPABILITY IS DETERMINISTIC AND BELONGS IN UNIT TESTS.
    QUALITY IS STATISTICAL AND BELONGS IN AN END-TO-END BENCHMARK AT RELEASE.

These corpora are the **unit** layer. Every case is synthetic, seeded, cheap, and drivable by
``percept.harness.scripted.ScriptedJudge`` with no API key and no spend. A corpus asserts that a
mechanism *can express* an answer; it never asserts that a real model *would produce* it.

Wire shape — each corpus directory holds exactly two JSON files:

``stream.json``  the INPUT (what the runtime is fed)::

    {
      "schema_version": "1", "corpus": "<name>", "kind": "frames" | "turns",
      "synthetic": true, "seed": 20260721, "generator": "<module>",
      "cases": [
        {"id": "<snake_case>", "description": "...", "steps": [ <step>, ... ]}
      ]
    }

  A ``frames`` step: ``{"t": <float s>, "frame": "<token>", "observations": [...],
  "verdict": {<Verdict field or additive field> ...}}``.
  A ``turns`` step:  ``{"t": <float s>, "speaker": "<role>", "text": "..."}``.

``labels.json``  the GROUND TRUTH — the EXPECTED ANSWER, not a restatement of the input::

    {
      "schema_version": "1", "corpus": "<name>",
      "cases": [
        {"id": "<same id, same order>", "proves": "...", "does_not_prove": "...",
         "expect": { <corpus-specific, machine-readable> }}
      ]
    }

``expect`` keys are corpus-specific and declared in :data:`REQUIRED_EXPECT_KEYS`; the validator
enforces them so a malformed fixture fails loudly *before* an implementation card ever reads it.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "SCHEMA_VERSION",
    "SEED",
    "CORPUS_NAMES",
    "CORPUS_KIND",
    "REQUIRED_EXPECT_KEYS",
    "ValidationError",
    "validate_corpus",
]

#: Bump when the on-disk shape changes incompatibly. The loader refuses a foreign version.
SCHEMA_VERSION = "1"

#: The single seed every generator derives from. No wall clock, no unseeded randomness anywhere.
SEED = 20260721

#: Corpus name -> the step kind its cases carry.
CORPUS_KIND: dict[str, str] = {
    "subject_identity": "frames",
    "continuation": "frames",
    "mutation": "turns",
    "episode_counting": "frames",
    "intervals": "frames",
    "idempotency": "frames",
    "longhorizon": "frames",
}

CORPUS_NAMES: tuple[str, ...] = tuple(CORPUS_KIND)

#: Corpus name -> keys every case's ``expect`` block MUST carry. This is the contract an
#: implementation card reads: the expected ANSWER, machine-readable, never just the input.
REQUIRED_EXPECT_KEYS: dict[str, tuple[str, ...]] = {
    # How many distinct subjects, what durable key each carries, and whether the binding resolved.
    "subject_identity": ("distinct_subject_count", "subjects"),
    # Episode segmentation + what the gate should surface, per step.
    "continuation": ("episode_count", "episodes", "actions"),
    # The speech act per turn, and the goal value AFTER each turn (the mutation, or the non-mutation).
    "mutation": ("acts", "value_after", "final_value"),
    # The cup problem: how many episodes, and where their boundaries fall.
    "episode_counting": ("count", "boundaries"),
    # Which spans are genuinely covered, and which are GAPS that must not be forward-filled.
    "intervals": ("intervals", "gaps", "forward_fill_forbidden"),
    # How many user-visible incidents the stream may produce.
    "idempotency": ("incident_count", "actions"),
    # Long-horizon episode linkage, on a synthetic stream.
    "longhorizon": ("episode_count", "boundaries", "duration_s"),
}

#: Speech acts the mutation corpus labels. Only ``assert``/``replace``/``retract`` may change a value.
SPEECH_ACTS = ("assert", "replace", "mention", "negation", "retract")
MUTATING_ACTS = ("assert", "replace", "retract")

#: Gate actions a labelled step may expect (mirrors ``percept.gate``'s FIRE/ASK/SILENT).
ACTIONS = ("fire", "ask", "silent")

#: Subject binding states. ``unknown`` is a FIRST-CLASS answer — not a fabricated key, and not
#: silently "no subject". A fixture that expects ``unknown`` is asserting the honest refusal.
BINDINGS = ("resolved", "unknown")


class ValidationError(AssertionError):
    """A corpus or label file is malformed. Loud, and fatal to the loader."""


def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def validate_corpus(name: str, stream: Mapping[str, Any], labels: Mapping[str, Any]) -> None:
    """Assert a ``(stream, labels)`` pair is well-formed for corpus ``name``.

    Structural only — it validates that the FIXTURE is valid, never that the capability works. The
    capabilities are not built yet; asserting them here would be asserting a plan.
    """
    _req(name in CORPUS_KIND, f"unknown corpus {name!r}")
    kind = CORPUS_KIND[name]

    for label, doc in (("stream", stream), ("labels", labels)):
        _req(doc.get("schema_version") == SCHEMA_VERSION,
             f"{name}/{label}: schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION!r}")
        _req(doc.get("corpus") == name, f"{name}/{label}: corpus field is {doc.get('corpus')!r}")
        _req(isinstance(doc.get("cases"), list) and doc["cases"],
             f"{name}/{label}: 'cases' must be a non-empty list")

    _req(stream.get("kind") == kind, f"{name}/stream: kind {stream.get('kind')!r} != {kind!r}")
    _req(stream.get("synthetic") is True, f"{name}/stream: must be flagged synthetic: true")
    _req(isinstance(stream.get("seed"), int), f"{name}/stream: 'seed' must be an int (reproducibility)")
    _req(isinstance(stream.get("generator"), str) and stream["generator"],
         f"{name}/stream: 'generator' must name the module that produced it")

    scases, lcases = stream["cases"], labels["cases"]
    _req(len(scases) == len(lcases),
         f"{name}: {len(scases)} stream case(s) vs {len(lcases)} label case(s)")

    seen: set[str] = set()
    for case, lab in zip(scases, lcases):
        cid = case.get("id")
        _req(isinstance(cid, str) and cid, f"{name}: a case has no id")
        _req(cid not in seen, f"{name}: duplicate case id {cid!r}")
        seen.add(cid)
        _req(lab.get("id") == cid, f"{name}: label order mismatch ({lab.get('id')!r} != {cid!r})")
        _req(isinstance(case.get("description"), str) and case["description"],
             f"{name}/{cid}: needs a human-readable description")

        steps = case.get("steps")
        _req(isinstance(steps, list) and steps, f"{name}/{cid}: 'steps' must be a non-empty list")
        last_t = None
        for i, step in enumerate(steps):
            _req(isinstance(step.get("t"), (int, float)), f"{name}/{cid}: step {i} has no numeric 't'")
            t = float(step["t"])
            _req(last_t is None or t >= last_t, f"{name}/{cid}: step {i} goes backwards in time")
            last_t = t
            if kind == "frames":
                _req(isinstance(step.get("frame"), str) and step["frame"],
                     f"{name}/{cid}: step {i} needs a 'frame' token")
                if "verdict" in step:
                    _req(isinstance(step["verdict"], dict), f"{name}/{cid}: step {i} verdict must be a dict")
                for j, obs in enumerate(step.get("observations") or []):
                    _req(isinstance(obs.get("label"), str), f"{name}/{cid}: step {i} obs {j} needs a label")
                    box = obs.get("box")
                    _req(isinstance(box, list) and len(box) == 4,
                         f"{name}/{cid}: step {i} obs {j} box must be [x, y, w, h]")
            else:
                _req(isinstance(step.get("speaker"), str) and step["speaker"],
                     f"{name}/{cid}: step {i} needs a 'speaker'")
                _req(isinstance(step.get("text"), str) and step["text"],
                     f"{name}/{cid}: step {i} needs 'text'")

        # -- the ground truth itself -----------------------------------------------------------
        for key in ("proves", "does_not_prove"):
            _req(isinstance(lab.get(key), str) and lab[key],
                 f"{name}/{cid}: label needs a non-empty {key!r} (a corpus must state its own limits)")
        expect = lab.get("expect")
        _req(isinstance(expect, dict) and expect, f"{name}/{cid}: 'expect' must be a non-empty dict")
        for key in REQUIRED_EXPECT_KEYS[name]:
            _req(key in expect, f"{name}/{cid}: expect is missing required key {key!r}")

        _validate_expect(name, cid, steps, expect)


def _validate_expect(name: str, cid: str, steps: list, expect: Mapping[str, Any]) -> None:
    """Per-corpus expectation checks — shape and internal consistency with the step count."""
    n = len(steps)

    if "actions" in expect:
        acts = expect["actions"]
        _req(isinstance(acts, list) and len(acts) == n,
             f"{name}/{cid}: expect.actions must align 1:1 with {n} step(s)")
        for a in acts:
            _req(a in ACTIONS, f"{name}/{cid}: unknown action {a!r} (want one of {ACTIONS})")

    if name == "subject_identity":
        subjects = expect["subjects"]
        _req(isinstance(subjects, list) and subjects, f"{name}/{cid}: expect.subjects must be non-empty")
        resolved = 0
        for s in subjects:
            _req(s.get("binding") in BINDINGS, f"{name}/{cid}: binding must be one of {BINDINGS}")
            if s["binding"] == "resolved":
                resolved += 1
                _req(isinstance(s.get("key"), str) and s["key"],
                     f"{name}/{cid}: a resolved subject needs a durable key")
            else:
                _req(s.get("key") is None,
                     f"{name}/{cid}: an unknown binding MUST NOT carry a fabricated key")
            idx = s.get("step_indices")
            _req(isinstance(idx, list) and idx, f"{name}/{cid}: subject needs step_indices")
            _req(all(isinstance(i, int) and 0 <= i < n for i in idx),
                 f"{name}/{cid}: step_indices out of range")
        _req(expect["distinct_subject_count"] == len(subjects),
             f"{name}/{cid}: distinct_subject_count disagrees with len(subjects)")
        _req(resolved <= len(subjects), f"{name}/{cid}: impossible resolved count")

    elif name in ("continuation", "longhorizon"):
        eps = expect["episodes"] if name == "continuation" else None
        if eps is not None:
            _req(isinstance(eps, list), f"{name}/{cid}: expect.episodes must be a list")
            _req(expect["episode_count"] == len(eps),
                 f"{name}/{cid}: episode_count disagrees with len(episodes)")
            for e in eps:
                _req(isinstance(e.get("subject"), str) and e["subject"],
                     f"{name}/{cid}: an episode needs a subject key")
                _req(float(e["end_t"]) >= float(e["start_t"]),
                     f"{name}/{cid}: episode ends before it starts")
        if name == "longhorizon":
            _req(float(expect["duration_s"]) >= 600.0,
                 f"{name}/{cid}: a long-horizon fixture must span >= 600s")
            _req(expect["episode_count"] == len(expect["boundaries"]),
                 f"{name}/{cid}: episode_count disagrees with len(boundaries)")
            _req(expect.get("capability_fixture") is True,
                 f"{name}/{cid}: must be flagged capability_fixture: true (it is NOT a quality benchmark)")

    elif name == "mutation":
        acts, values = expect["acts"], expect["value_after"]
        _req(isinstance(acts, list) and len(acts) == n,
             f"{name}/{cid}: expect.acts must align 1:1 with {n} turn(s)")
        _req(isinstance(values, list) and len(values) == n,
             f"{name}/{cid}: expect.value_after must align 1:1 with {n} turn(s)")
        for a in acts:
            _req(a in SPEECH_ACTS, f"{name}/{cid}: unknown speech act {a!r}")
        prev = expect.get("value_before")
        blocked = expect.get("blocked_by_authority") or [False] * n
        _req(len(blocked) == n, f"{name}/{cid}: blocked_by_authority must align 1:1 with {n} turn(s)")
        for i, (a, v) in enumerate(zip(acts, values)):
            if a not in MUTATING_ACTS:
                _req(v == prev,
                     f"{name}/{cid}: turn {i} is a {a!r} but the value changed {prev!r} -> {v!r}; "
                     "a non-mutating act MUST leave the value untouched")
            elif v == prev:
                # A mutating act that changed nothing is only legal when something BLOCKED it —
                # otherwise the label is silently claiming a no-op mutation.
                _req(bool(blocked[i]),
                     f"{name}/{cid}: turn {i} is a mutating {a!r} that left the value at {v!r} but is not "
                     "marked blocked_by_authority — an unexplained no-op mutation")
            prev = v
        _req(expect["final_value"] == (values[-1] if values else None),
             f"{name}/{cid}: final_value disagrees with the last value_after")

    elif name == "episode_counting":
        b = expect["boundaries"]
        _req(isinstance(b, list) and expect["count"] == len(b),
             f"{name}/{cid}: count disagrees with len(boundaries)")
        for span in b:
            _req(isinstance(span, list) and len(span) == 2 and span[1] >= span[0],
                 f"{name}/{cid}: a boundary must be [start_t, end_t] with end >= start")

    elif name == "intervals":
        _req(expect["forward_fill_forbidden"] is True,
             f"{name}/{cid}: forward_fill_forbidden must be true — the design rejects filling a gap")
        for span in expect["intervals"]:
            _req(span.get("kind") in ("observed", "validity_horizon"),
                 f"{name}/{cid}: an interval must be 'observed' or 'validity_horizon'")
            _req(float(span["end"]) >= float(span["start"]), f"{name}/{cid}: interval ends before it starts")
        for gap in expect["gaps"]:
            _req(isinstance(gap.get("reason"), str) and gap["reason"],
                 f"{name}/{cid}: a gap must state its reason")
            _req(float(gap["end"]) >= float(gap["start"]), f"{name}/{cid}: gap ends before it starts")

    elif name == "idempotency":
        _req(isinstance(expect["incident_count"], int) and expect["incident_count"] >= 0,
             f"{name}/{cid}: incident_count must be a non-negative int")
        fires = sum(1 for a in expect["actions"] if a == "fire")
        _req(fires == expect["incident_count"],
             f"{name}/{cid}: {fires} fire(s) but incident_count={expect['incident_count']}")
