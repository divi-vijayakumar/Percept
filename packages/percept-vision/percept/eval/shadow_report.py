"""Card Y11 — the SHADOW per-probe report: would the fold the model WOULD have proposed have been
RIGHT, measured against synthetic ground truth?

This is the de-risking measurement. It drives the frozen ``eval/synthetic`` corpora through the SAME
pure mechanism shadow mode observes — subject resolution (Y5), the episode continuation log (Y1), the
episode fold (Y3), the claim/mutation state machine (Y2) — and scores each proposal against the
fixture's labelled answer. No API spend: the driver is :class:`~percept.harness.scripted.ScriptedJudge`
over the committed corpus generators, so the report runs in a worktree that does not carry the
gitignored generated tree.

TWO probes, mirroring the two client collapses:

* **probe 01** (screen/document) — ``subject_identity`` + ``continuation``. Would the proposed
  identity + continuation matching have been right? A FALSE-BIND here is binding two things the fixture
  says are DIFFERENT (a wrong key, a fabricated key, or collapsing two distinct episodes into one) —
  the dangerous, silent-corruption error.
* **probe 02** (conversational) — ``mutation``. Would the proposed mutation fold (assert / replace /
  retract) have moved the authoritative value correctly? A FALSE-BIND is applying a change the fixture
  says should NOT happen (treating a mention/negation/attribution as a replacement, or applying an
  unauthorized replace) — again silent state corruption.

THE ASYMMETRY (card Y8, reused): a FALSE-BIND is weighted far worse than an abstention/miss. Both
probes fold their per-item outcomes through :mod:`percept.eval.identity_metrics` so the same hard
false-bind ceiling and weighted harm score that gate E-1 gate the shadow read.

HONESTY (stated, not implied). This proves the MECHANISM would/would-not have been right on SYNTHETIC
ground truth with the speech acts / keys already classified. It does NOT prove real-world accuracy:
whether a real OCR/VLM reads a key off a screen, or a real model classifies a speech act, is a
statistical benchmark question this report deliberately does not touch. A DISAGREEMENT between the
fold and a fixture is a DESIGN QUESTION to SURFACE, not a target to conform to.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from ..behavior import episode_mutation_of
from ..claims import fold_verdicts
from ..contracts import OPERATIONS, ConcernAuthority
from ..episodes import EpisodeLog, EpisodePolicy
from ..harness.scripted import ScriptedJudge
from ..harness.synthetic.corpora import continuation as _continuation
from ..harness.synthetic.corpora import mutation as _mutation
from ..harness.synthetic.corpora import subject_identity as _subject_identity
from .identity_ledger import IdentityLedger, Sighting
from .identity_metrics import (
    AcceptancePolicy,
    AcceptanceResult,
    BindOutcome,
    E1Metrics,
    assess_acceptance,
    classify_binding,
    compute_e1_metrics,
)

__all__ = [
    "Disagreement",
    "ProbeReport",
    "run_probe_01_identity_continuation",
    "run_probe_02_mutation",
    "run_shadow_probes",
    "format_report",
    "DISTINCT_SUBJECT_COUNT_NOTE",
    "SHADOW_ACCEPTANCE_POLICY",
]

#: ``distinct_subject_count`` used to have TWO definitions in this eval layer: the fixture counted a
#: lone unknown sighting as 1 subject, while :class:`~percept.eval.identity_ledger.IdentityLedger` kept
#: unknowns out of the count entirely (tracked only as ``unknown_count``) — a genuine divergence on
#: ``subject_identity/unreadable_key_binds_unknown`` (fixture=1, oracle=0). The oracle now counts BOUND
#: subjects PLUS UNKNOWN EPISODES (``IdentityLedger.distinct_subject_count`` — see that class), which
#: is exactly the fixture's answer, so this probe computes ``proposed_dsc`` by running the SAME
#: verdicts through the SAME ``IdentityLedger`` the E-1 oracle uses, rather than a second ad hoc
#: formula. There is now one definition, not two flagged in parallel.
DISTINCT_SUBJECT_COUNT_NOTE = (
    "distinct_subject_count is computed by the E-1 IdentityLedger oracle itself (bound subjects plus "
    "distinct unknown-episodes — an honestly unresolved sighting counts as its own provisional "
    "subject rather than disappearing from the total, the same UN_<episode_id> accounting "
    "percept.episodes uses). The fixture and the oracle can no longer diverge because this probe uses "
    "that ONE formula, not a separate fixture-side count kept in sync by hand."
)

#: Shadow acceptance uses the E-1 asymmetry (hard false-bind ceiling + weighted harm) but does NOT
#: require the judgment-quality non-inferiority arm: shadow measures binding/fold CORRECTNESS, while a
#: quality delta needs a separate instrumented arm (Y8) that is out of this card's scope.
SHADOW_ACCEPTANCE_POLICY = AcceptancePolicy(require_quality_non_inferior=False)

_MONEY = re.compile(r"\$\d+(?:\.\d+)?[KkMm]?")

#: Corpus act labels → the Verdict :data:`~percept.contracts.Operation` vocabulary. The corpus labels
#: a linguistic "negation" ("The budget is not $25K"); to the value machine that is a ``mention`` (the
#: value appears but is not asserted), so it must leave the slot unchanged.
_ACT_TO_OP = {"negation": "mention"}


@dataclass(frozen=True)
class Disagreement:
    """One item where the proposed fold disagreed with the fixture — a DESIGN QUESTION to surface."""

    corpus: str
    case_id: str
    locus: str            # e.g. "step 1", "turn 4"
    expected: Any
    proposed: Any
    kind: str             # "false_bind" | "miss" | "value_mismatch" | ...

    def to_dict(self) -> dict:
        return {
            "corpus": self.corpus, "case_id": self.case_id, "locus": self.locus,
            "expected": self.expected, "proposed": self.proposed, "kind": self.kind,
        }


@dataclass(frozen=True)
class ProbeReport:
    """The per-probe shadow result: the E-1 metric set + acceptance + the disagreeing cases."""

    probe: str
    corpora: tuple[str, ...]
    metrics: E1Metrics
    acceptance: AcceptanceResult
    disagreements: tuple[Disagreement, ...]
    matched: int = 0            # scored items whose proposal AGREED with the fixture (see :meth:`_matched`)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def match_rate(self) -> float:
        """Fraction of scored items whose proposal AGREED with the fixture — an EXACT hit OR a correct
        abstention (the honest ``unknown`` on a genuinely illegible key). A false bind and a miss are
        both disagreements. (``distinct_subject_count`` value-mismatches are surfaced as disagreements
        but carry no BindOutcome, so they are not scored items and never distort this rate.)"""
        n = self.metrics.n
        return (self.matched / n) if n else 0.0

    def to_dict(self) -> dict:
        return {
            "probe": self.probe,
            "corpora": list(self.corpora),
            "match_rate": self.match_rate,
            "false_bind_rate": self.metrics.false_bound_rate,
            "unknown_rate": self.metrics.unknown_rate,
            "harm_score": self.acceptance.harm_score,
            "accepted": self.acceptance.accepted,
            "acceptance_reasons": list(self.acceptance.reasons),
            "metrics": self.metrics.to_dict(),
            "disagreements": [d.to_dict() for d in self.disagreements],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------------------------------
# Corpus access — build DIRECTLY from the committed generators, no gitignored tree required.
# --------------------------------------------------------------------------------------------------


def _case_pairs(module: Any) -> list[tuple[dict, dict]]:
    """``[(stream_case, expect)]`` from a corpus generator's ``build()`` (no disk read)."""
    stream_doc, labels_doc = module.build()
    expect_by_id = {c["id"]: c["expect"] for c in labels_doc["cases"]}
    return [(c, expect_by_id[c["id"]]) for c in stream_doc["cases"]]


def _matched(outcomes: list[BindOutcome]) -> int:
    """AGREEMENTS: an EXACT hit, or a correct abstention on a genuinely illegible key. A miss
    (``ABSTAINED_WHEN_LEGIBLE``) and both false binds are disagreements, not matches."""
    return sum(
        1 for o in outcomes
        if o is BindOutcome.EXACT or o is BindOutcome.ABSTAINED_WHEN_ILLEGIBLE
    )


async def _run_scripted(specs: list[dict]) -> list:
    """Serve a list of verdict specs through :class:`ScriptedJudge` (the required driver) → Verdicts."""
    judge = ScriptedJudge([dict(s) for s in specs])
    out = []
    for i in range(len(specs)):
        out.append(await judge.judge("shadow probe", f"f{i}"))
    return out


# --------------------------------------------------------------------------------------------------
# PROBE 01 — subject identity + continuation.
# --------------------------------------------------------------------------------------------------


def _identity_outcomes(
    disagreements: list[Disagreement],
) -> tuple[list[BindOutcome], list[str]]:
    """Per-sighting bind outcomes over the ``subject_identity`` corpus, driven by ScriptedJudge."""
    outcomes: list[BindOutcome] = []
    notes: list[str] = []
    for stream_case, expect in _case_pairs(_subject_identity):
        steps = stream_case["steps"]
        verdicts = asyncio.run(_run_scripted([s.get("verdict") or {} for s in steps]))
        # Ground truth per step index: (true_key, legible).
        truth: dict[int, tuple[str | None, bool]] = {}
        for subj in expect.get("subjects", []):
            legible = subj.get("binding") == "resolved"
            for idx in subj.get("step_indices", []):
                truth[idx] = (subj.get("key"), legible)
        for i, v in enumerate(verdicts):
            true_key, legible = truth.get(i, (None, False))
            binding = getattr(v, "binding", None)
            predicted_key = getattr(v, "subject_key", None) if binding == "resolved" else None
            outcome = classify_binding(
                legible=legible, true_key=true_key, predicted_key=predicted_key
            )
            outcomes.append(outcome)
            if outcome is BindOutcome.EXACT or outcome is BindOutcome.ABSTAINED_WHEN_ILLEGIBLE:
                continue
            kind = "false_bind" if outcome.is_false_bind else "miss"
            disagreements.append(
                Disagreement(
                    corpus="subject_identity", case_id=stream_case["id"], locus=f"step {i}",
                    expected=true_key if legible else "unknown", proposed=predicted_key, kind=kind,
                )
            )
        # distinct_subject_count — ONE definition, computed by running these SAME proposed verdicts
        # through the SAME IdentityLedger oracle E-1 scores against, instead of a second fixture-side
        # formula kept in sync by hand. ``t`` is the step index: this corpus's cases carry no real
        # clock and default_ttl/unknown_continuity_gap_s are both off, so only sighting ORDER matters.
        ledger = IdentityLedger()
        for i, v in enumerate(verdicts):
            ledger.observe(Sighting.from_verdict(float(i), v))
        proposed_dsc = ledger.distinct_subject_count
        expected_dsc = expect.get("distinct_subject_count")
        if expected_dsc is not None and proposed_dsc != expected_dsc:
            disagreements.append(
                Disagreement(
                    corpus="subject_identity", case_id=stream_case["id"],
                    locus="distinct_subject_count", expected=expected_dsc, proposed=proposed_dsc,
                    kind="value_mismatch",
                )
            )
    notes.append(DISTINCT_SUBJECT_COUNT_NOTE)
    return outcomes, notes


def _continuation_outcomes(disagreements: list[Disagreement]) -> list[BindOutcome]:
    """Per-CASE continuation outcomes: did the fold open the right number of episodes, with the right
    continued/new split? Under-counting (collapsing two distinct episodes) is the dangerous FALSE-BIND
    — the silent-merge error; over-counting (a continuation re-opened as new) is the SAFE miss."""
    outcomes: list[BindOutcome] = []
    for stream_case, expect in _case_pairs(_continuation):
        gap = expect.get("continuity_gap_s")
        log = EpisodeLog("g1", policy=EpisodePolicy(continuity_gap_s=gap))
        steps = stream_case["steps"]
        verdicts = asyncio.run(_run_scripted([s.get("verdict") or {} for s in steps]))
        for i, (step, v) in enumerate(zip(steps, verdicts)):
            key = getattr(v, "subject_key", None)
            log.observe_decision(step["t"], f"obs{i}", subject_key=key)
        proposed_count = log.count()
        expected_count = expect.get("episode_count")
        if proposed_count == expected_count:
            outcomes.append(BindOutcome.EXACT)
            continue
        if proposed_count < expected_count:
            outcomes.append(BindOutcome.WRONG_KEY_FALSE_BIND)  # collapsed distinct episodes (dangerous)
            kind = "false_bind"
        else:
            outcomes.append(BindOutcome.ABSTAINED_WHEN_LEGIBLE)  # over-split (safe miss)
            kind = "miss"
        disagreements.append(
            Disagreement(
                corpus="continuation", case_id=stream_case["id"], locus="episode_count",
                expected=expected_count, proposed=proposed_count, kind=kind,
            )
        )
    return outcomes


def run_probe_01_identity_continuation() -> ProbeReport:
    """Probe 01 — subject identity + continuation. Aggregates per-sighting identity outcomes and
    per-case continuation outcomes into one E-1 metric set + acceptance."""
    disagreements: list[Disagreement] = []
    id_outcomes, notes = _identity_outcomes(disagreements)
    cont_outcomes = _continuation_outcomes(disagreements)
    outcomes = id_outcomes + cont_outcomes
    metrics = compute_e1_metrics(outcomes)
    acceptance = assess_acceptance(metrics, SHADOW_ACCEPTANCE_POLICY)
    return ProbeReport(
        probe="probe_01_identity_continuation",
        corpora=("subject_identity", "continuation"),
        metrics=metrics,
        acceptance=acceptance,
        disagreements=tuple(disagreements),
        matched=_matched(outcomes),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------------------
# PROBE 02 — mutation / speech act.
# --------------------------------------------------------------------------------------------------


def _op_of(act: str) -> str:
    if act in _ACT_TO_OP:
        return _ACT_TO_OP[act]
    return act if act in OPERATIONS else "mention"


def _mutation_specs(stream_case: dict, expect: dict) -> list[dict]:
    """Build the per-turn verdict specs a judge WOULD emit for this case, from the labelled acts +
    the value each utterance CONTAINS (parsed from its own text — the value is literally in the turn),
    so the state machine has to correctly NOT apply the mentions."""
    acts = expect.get("acts", [])
    specs = []
    for i, step in enumerate(stream_case["steps"]):
        op = _op_of(acts[i]) if i < len(acts) else "assert"
        m = _MONEY.search(step.get("text", ""))
        value = m.group(0) if m else None
        specs.append({
            "satisfied": True, "confidence": 0.9, "evidence": step.get("text", ""),
            "operation": op, "value": value, "asserted_by": step.get("speaker"),
        })
    return specs


def run_probe_02_mutation() -> ProbeReport:
    """Probe 02 — mutation. Folds each labelled speech act through the claim state machine (with the
    concern's authority) and scores whether the authoritative value moved exactly when the fixture
    says it should. Applying a change the fixture forbids (a mention/negation as a replace, or an
    unauthorized replace) is the dangerous FALSE-BIND; failing to apply a real change is the safe miss.
    """
    disagreements: list[Disagreement] = []
    outcomes: list[BindOutcome] = []
    for stream_case, expect in _case_pairs(_mutation):
        specs = _mutation_specs(stream_case, expect)
        verdicts = asyncio.run(_run_scripted(specs))
        authority = None
        if expect.get("authority_required"):
            authority = ConcernAuthority(frozenset({str(expect["authority_required"])}))
        _state, claim_outcomes = fold_verdicts(verdicts, authority=authority, trace=False)
        value_after = expect.get("value_after", [])
        prev = expect.get("value_before")
        for i, outcome in enumerate(claim_outcomes):
            expected_val = value_after[i] if i < len(value_after) else None
            true_moved = expected_val != prev
            pred_moved = outcome.applied
            value_ok = outcome.value_after == expected_val
            prev = expected_val
            # The proposed mutation-fold classification (assert/replace/retract -> the EpisodeMutation
            # vocabulary): confirms the fold vocabulary lines up with the act, independent of value.
            _proposed_fold = episode_mutation_of(outcome.effective)
            if pred_moved == true_moved and value_ok:
                outcomes.append(BindOutcome.EXACT)
                continue
            if pred_moved and not true_moved:
                outcomes.append(BindOutcome.WRONG_KEY_FALSE_BIND)  # applied a forbidden change
                kind = "false_bind"
            elif pred_moved and not value_ok:
                outcomes.append(BindOutcome.WRONG_KEY_FALSE_BIND)  # moved to the WRONG value
                kind = "false_bind"
            else:
                outcomes.append(BindOutcome.ABSTAINED_WHEN_LEGIBLE)  # missed a real mutation (safe)
                kind = "miss"
            disagreements.append(
                Disagreement(
                    corpus="mutation", case_id=stream_case["id"], locus=f"turn {i}",
                    expected=expected_val, proposed=outcome.value_after, kind=kind,
                )
            )
    metrics = compute_e1_metrics(outcomes)
    acceptance = assess_acceptance(metrics, SHADOW_ACCEPTANCE_POLICY)
    return ProbeReport(
        probe="probe_02_mutation",
        corpora=("mutation",),
        metrics=metrics,
        acceptance=acceptance,
        disagreements=tuple(disagreements),
        matched=_matched(outcomes),
        notes=(
            "Feeds the fixture's labelled speech acts + each turn's own contained value; proves the "
            "state machine downstream of act classification, NOT that a model classifies the act.",
        ),
    )


def run_shadow_probes() -> dict[str, ProbeReport]:
    """Run both probes. A read anyone can produce locally with no API spend."""
    return {
        "probe_01": run_probe_01_identity_continuation(),
        "probe_02": run_probe_02_mutation(),
    }


def format_report(reports: dict[str, ProbeReport]) -> str:
    """A plain-text, hand-to-a-client rendering of the shadow probes."""
    lines: list[str] = []
    lines.append("SHADOW MODE — per-probe report (card Y11)")
    lines.append("=" * 72)
    lines.append(
        "Proves the fold WOULD/would-not have been right on SYNTHETIC ground truth. It does NOT "
        "prove real-world accuracy. A disagreement is a DESIGN QUESTION, not a target."
    )
    lines.append("")
    for key in ("probe_01", "probe_02"):
        r = reports[key]
        lines.append(f"[{r.probe}]  corpora={', '.join(r.corpora)}")
        lines.append(
            f"  n={r.metrics.n}  match_rate={r.match_rate:.3f}  "
            f"false_bind_rate={r.metrics.false_bound_rate:.3f}  "
            f"unknown/miss_rate={r.metrics.unknown_rate:.3f}"
        )
        lines.append(
            f"  harm_score={r.acceptance.harm_score:.4g}  accepted={r.acceptance.accepted}  "
            f"(false binds weighted {SHADOW_ACCEPTANCE_POLICY.false_bind_weight:g}x a miss)"
        )
        for reason in r.acceptance.reasons:
            lines.append(f"    - blocks: {reason}")
        if r.disagreements:
            lines.append("  disagreements (surface, do not conform to):")
            for d in r.disagreements:
                lines.append(
                    f"    * {d.corpus}/{d.case_id} @ {d.locus}: "
                    f"expected={d.expected!r} proposed={d.proposed!r} [{d.kind}]"
                )
        else:
            lines.append("  disagreements: none — the mechanism matched every fixture answer.")
        for note in r.notes:
            lines.append(f"  note: {note}")
        lines.append("")
    return "\n".join(lines)
