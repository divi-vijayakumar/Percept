"""Session-disjoint held-out split + a pre-registration for the GUI-World fixtures.

The one rule that protects the trust metric on this dataset: **the split is by SESSION (video),
never by keyframe/clip.** Two fixtures derived from the same recording share a session, and a
session is placed wholly on one side of the held-out boundary — otherwise a modal event and a page
event from the SAME recording could land on opposite sides and leak. :func:`build_session_split`
guarantees this by construction; the tests assert no session id appears on both sides after the
split round-trips through :func:`percept.eval.differential.load_held_out_split`.

The pre-registration freezes everything that must be fixed BEFORE any number exists — the ladder
(what eye tunings will be compared), the exact held-out split, the operator's bars, and the fixture
set's own content hash — into one content-addressed :class:`percept.eval.derivation.DerivationPreRegistration`.
Its ``registration_hash`` is stamped so a later reader can prove the split and bars were not moved
after seeing the scores.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from percept.eval.derivation import (
    DerivationPreRegistration,
    Ladder,
    LadderRung,
    fixture_set_hash,
)
from percept.eval.differential import (
    BarsDocument,
    Fixture,
    HeldOutSplit,
    load_bars,
    load_fixtures,
    load_held_out_split,
)
from percept.source_profiles_default import SCREEN_COAST_INTERVAL_S, SCREEN_DOCUMENT
from percept.watchprofile import CadenceProfile, GateProfile, WatchProfile

from .transform import session_of

#: Deterministic salt for the session ordering — the split must not depend on wall time or dict
#: iteration order, so two builds over the same sessions produce the same split.
_SPLIT_SEED = "guiworld-ds3-v1"


# ---------------------------------------------------------------------------
# 1. Session-disjoint split
# ---------------------------------------------------------------------------


def build_session_split(
    fixtures: Sequence[Fixture],
    *,
    held_out_fraction: float = 0.5,
    seed: str = _SPLIT_SEED,
) -> dict[str, list[str]]:
    """Assign whole SESSIONS to ``test`` (held out) vs ``train`` (exploratory).

    Deterministic: sessions are ordered by a stable hash of ``(seed, session_id)`` and the first
    ``held_out_fraction`` of them (rounded) become the held-out ``test`` split. Because the unit is
    the session, every fixture of a given recording lands on the same side — the split is
    session-disjoint by construction. ``dev`` is left empty (this dataset is a two-way held-out
    split, not a three-way tuning split).
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError(f"held_out_fraction must be in (0, 1), got {held_out_fraction!r}")
    sessions = sorted({session_of(f.fixture_id) for f in fixtures})
    ordered = sorted(
        sessions,
        key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(),
    )
    n_test = round(len(ordered) * held_out_fraction)
    n_test = max(1, min(len(ordered) - 1, n_test))  # never all-or-nothing
    held_out_sessions = set(ordered[:n_test])

    test: list[str] = []
    train: list[str] = []
    for fixture in fixtures:
        target = test if session_of(fixture.fixture_id) in held_out_sessions else train
        target.append(fixture.fixture_id)
    return {"train": sorted(train), "dev": [], "test": sorted(test)}


# ---------------------------------------------------------------------------
# 2. The ladder the derivation would compare (coast-only vs client change signal)
# ---------------------------------------------------------------------------


def _screen_watch_profile(
    name: str, *, base_fps: float, max_fps: float, escalate_threshold: float, notes: str
) -> WatchProfile:
    """A ``screen_document`` eye tuning. ``environment`` is a camera mount ("screen_capture"), NOT a
    source-class name — a source class rides on the SourceProfile axis (ADR-005), and naming it here
    would trip :func:`percept.eval.derivation.assert_source_class_not_overloaded`.
    ``plan_mode="window"`` is the screen rule: an ordered frame window, never a sprite montage.
    """
    return WatchProfile(
        name=name,
        environment="screen_capture",
        goal_type="dynamic",
        state="transition",
        salience_stat="block_max",
        stat_thresholds={"block_max": 0.20},
        plan_mode="window",
        cadence=CadenceProfile(
            base_fps=base_fps, max_fps=max_fps, escalate_threshold=escalate_threshold, hold_s=0.5
        ),
        gate=GateProfile(require_false_first=True, refractory_s=0.5),
        notes=notes,
    )


def build_screen_ladder() -> Ladder:
    """A 2-rung ladder over the ``screen_document`` source class.

    - ``coast_only`` (baseline) — the shipped default: ``block_max``/0.20 plus the fixed low-rate
      coast floor, with no client change signal (the eye never escalates above the coast).
    - ``client_change_signal`` (candidate) — the same eye once the client registers a page-change /
      phash / OCR signal that escalates above the coast (docs/admission/tuning-admission.md). This is the
      A/B the ``screen_document`` finding says a page-scale-edit client must run on THEIR fixtures.
    """
    coast_fps = 1.0 / SCREEN_COAST_INTERVAL_S
    baseline = LadderRung(
        name="coast_only",
        watch_profile=_screen_watch_profile(
            "screen-coast-only",
            base_fps=coast_fps,
            max_fps=coast_fps,
            escalate_threshold=1.0,  # never escalates: coast floor alone
            notes=(
                "Shipped default: block_max/0.20 + the mandatory coast floor, no client change "
                "signal. A page-scale text edit dilutes below every calibrated cutoff, so only the "
                "coast keeps it observed."
            ),
        ),
        source_profile=SCREEN_DOCUMENT,
        description="block_max/0.20 default + coast floor, no client change signal",
    )
    candidate = LadderRung(
        name="client_change_signal",
        watch_profile=_screen_watch_profile(
            "screen-client-change-signal",
            base_fps=coast_fps,
            max_fps=4.0,
            escalate_threshold=0.20,  # a registered change signal escalates above the coast
            notes=(
                "Client-registered page-change / phash / OCR change signal escalating above the "
                "coast floor. The coast floor is retained underneath it, not replaced."
            ),
        ),
        source_profile=SCREEN_DOCUMENT,
        description="client change signal escalating above the retained coast floor",
    )
    return Ladder(rungs=(baseline, candidate), source="eval.datasets.guiworld", declared_at="2026-07-23")


# ---------------------------------------------------------------------------
# 3. Pre-registration
# ---------------------------------------------------------------------------


def build_preregistration(
    fixtures_root: str | Path,
    *,
    bars_path: str | Path | None = None,
    split_name: str = "test",
    declared_at: str = "2026-07-23",
) -> DerivationPreRegistration:
    """Assemble the frozen pre-registration for the committed GUI-World fixtures.

    Loads the fixtures and the split through Percept's OWN loaders (so the split round-trips through
    :func:`percept.eval.differential.load_held_out_split`, which enforces disjointness), computes
    :func:`percept.eval.derivation.fixture_set_hash`, and freezes the ladder + split + bars +
    fixture-set hash into a :class:`~percept.eval.derivation.DerivationPreRegistration`. Nothing is
    scored here.
    """
    root = Path(fixtures_root)
    fixtures = load_fixtures(root)
    split: HeldOutSplit = load_held_out_split(root, fixtures, split_name=split_name)
    bars: BarsDocument = load_bars(bars_path)
    # Freeze PATH-INDEPENDENTLY: load_held_out_split / load_bars stamp ``source`` with the absolute
    # checkout path (``str(path)``), and DerivationPreRegistration folds both into its
    # registration_hash — so the SAME labeled corpus in two checkouts would hash differently (a
    # worktree path leaked into the committed hash and broke on merge). Normalize ``source`` to its
    # basename so the frozen hash is portable, per fixture_set_hash's own contract: "two checkouts
    # of the same labeled corpus must hash identically".
    split = replace(split, source=Path(split.source).name)
    if bars.source and bars.source != "(none supplied)":
        bars = replace(bars, source=Path(bars.source).name)
    return DerivationPreRegistration(
        ladder=build_screen_ladder(),
        split=split,
        bars=bars,
        fixture_set_hash=fixture_set_hash(fixtures),
        split_name=split_name,
        declared_at=declared_at,
    )
