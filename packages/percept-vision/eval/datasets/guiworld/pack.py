"""``screen_observation@1`` — a self-contained :class:`~percept.domain_pack.DomainPack` for screen
content (dev-tooling SCAFFOLD; not yet wired into the shipped registry).

Three templates, one per screen watch condition, built ONLY from the existing ``transition`` sense
(``percept.behavior.coerce_authored_behavior``) — no new sense, nothing invented:

- ``modal_appeared`` — no modal -> a modal dialog is shown.
- ``error_shown``    — no error -> an error message is shown.
- ``page_changed``   — the current page -> a different page.

This module has NO import-time side effect: it defines ``SCREEN_OBSERVATION_V1`` and a
``register_entrypoint_pack`` hook but does NOT register itself. Registration is the documented
one-line follow-up (see :func:`register_entrypoint_pack`) — kept out of this scaffold so it never
races the sibling cards that own ``percept/packs_default.py`` and
``percept/source_profiles_default.py``.

The pack ``notes`` preserve the shipped ``screen_document`` finding verbatim in substance: no
salience statistic admits a full-page, text-scale edit on the 48x36 tier-0 gate frame; the
``block_max``/0.20 default only clears on a LOCALIZED change; the fixed low-rate coast floor stays
MANDATORY underneath it; and a client dominated by page-scale edits must register their own
page-change / phash / OCR change signal and A/B it with ``percept eval``. Identity is a content key,
never geometry; ``plan_mode`` is an ordered frame window, never a sprite montage.
"""

from __future__ import annotations

from percept.direct_judge import DirectRouting
from percept.domain_pack import ComponentRecommendation, DomainPack, GoalTemplate

# ---------------------------------------------------------------------------
# Templates — all the transition sense (no modal -> modal, no error -> error, page A -> page B)
# ---------------------------------------------------------------------------

_MODAL_APPEARED = GoalTemplate(
    name="modal_appeared",
    condition="a modal dialog appeared",
    target="modal dialog",
    behavior={
        "type": "transition",
        "from": "no modal dialog is shown",
        "to": "a modal dialog is shown",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    source="screen",
    notes="fires once, on the rising edge of a modal/dialog/popup appearing over the screen.",
)

_ERROR_SHOWN = GoalTemplate(
    name="error_shown",
    condition="an error message is shown",
    target="error surface",
    behavior={
        "type": "transition",
        "from": "no error message is shown",
        "to": "an error message is shown",
    },
    # An unnoticed error surface (a failed save, a rejected payment) is costlier to miss than a
    # spurious fire is to receive.
    consequence={"miss_cost": "severe", "false_fire_cost": "mild"},
    source="screen",
    notes="fires once, on the rising edge of an error dialog / toast / banner appearing.",
)

_PAGE_CHANGED = GoalTemplate(
    name="page_changed",
    condition="the page changed",
    target="page",
    behavior={
        "type": "transition",
        "from": "the current page",
        "to": "a different page",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    source="screen",
    notes="fires once, on a navigation / page change to a different page identity.",
)

_TEMPLATES = {t.name: t for t in (_MODAL_APPEARED, _ERROR_SHOWN, _PAGE_CHANGED)}

# ---------------------------------------------------------------------------
# Component recommendations — all advisory ("preferred"): this pack recommends, never requires.
# ---------------------------------------------------------------------------

_COMPONENTS = (
    ComponentRecommendation(
        ref="sense:screen_document",
        strength="preferred",
        reason=(
            "screen recordings and document/page content are the screen_document source class, not "
            "a physical camera scene: block_max/0.20 admission plus the MANDATORY coast floor, an "
            "ordered frame window and content-key identity."
        ),
    ),
    ComponentRecommendation(
        ref="identity_resolver:content_key",
        strength="preferred",
        reason=(
            "screen identity must key on CONTENT (document id / page key), never geometry — two "
            "invoices rendered in the same region with the same label are NOT the same entity."
        ),
    ),
    ComponentRecommendation(
        ref="execution_policy:low_cost@1",
        strength="preferred",
        reason=(
            "screen observation is typically sparse-event; a quieter cadence saves spend — but the "
            "execution policy MUST keep the coast floor bound (do NOT use a strict_admission policy "
            "that unbinds coast_enabled, or a page-scale edit goes unobserved)."
        ),
    ),
)

# ---------------------------------------------------------------------------
# DirectJudge routes — a RESERVED namespace only; no rule implementation ships here.
# ---------------------------------------------------------------------------

_DIRECT_JUDGE_ROUTES = (DirectRouting(condition_namespace="screen_observation."),)

# ---------------------------------------------------------------------------
# Fixtures + quality bars + search space — plain data declarations (later cards consume them).
# The fixture names resolve to eval/datasets/guiworld/testdata/fixtures/ (dev tooling).
# ---------------------------------------------------------------------------

_FIXTURES = (
    "guiworld/modal_appeared",
    "guiworld/error_shown",
    "guiworld/page_changed",
)

_QUALITY_BARS = {
    "modal_appeared.recall_min": 0.60,
    "error_shown.recall_min": 0.60,
    "page_changed.recall_min": 0.55,
}

_SEARCH_SPACE = {
    "WatchProfile.cadence.base_fps": {"min": 0.5, "max": 4.0},
    "WatchProfile.cadence.escalate_threshold": {"min": 0.05, "max": 0.40},
    # The coast interval is a FLOOR, not a free knob — bounded, never allowed to reach 0 (which
    # would drop the mandatory floor the screen_document finding says must remain).
    "SourceProfile.screen_document.coast_interval_s": {"min": 1.0, "max": 2.0},
}

SCREEN_OBSERVATION_V1 = DomainPack(
    id="screen_observation",
    version="1",
    templates=_TEMPLATES,
    components=_COMPONENTS,
    policy_clamps={"false_negative_cost_min": "mild", "retention_max_days": 30.0},
    direct_judge_routes=_DIRECT_JUDGE_ROUTES,
    fixtures=_FIXTURES,
    quality_bars=_QUALITY_BARS,
    search_space=_SEARCH_SPACE,
    notes=(
        "Screen recordings, document and page content. Built entirely from the existing transition "
        "sense (modal appeared, error shown, page changed) — no new sense. Recommends the "
        "screen_document source profile. SHIPPED FINDING preserved: no salience statistic admits a "
        "full-PAGE, text-scale edit on the 48x36 tier-0 gate frame; the block_max/0.20 default only "
        "clears on a LOCALIZED change (a modal, a highlighted cell, a UI element); the fixed "
        "low-rate coast floor stays MANDATORY underneath it and must NOT be dropped; a client whose "
        "content is dominated by page-scale edits must register their own page-change / phash / OCR "
        "change signal (register_detector) and A/B it with `percept eval` on THEIR fixtures against "
        "THEIR bars (docs/admission/tuning-admission.md). Identity is a content key, never geometry; "
        "plan_mode is an ordered frame window, never a sprite montage."
    ),
)


def register_entrypoint_pack(register) -> None:
    """The ``percept.packs`` entry-point TARGET a promoting change would advertise (mirrors
    :func:`percept.packs_default.register_entrypoint_pack`).

    FOLLOW-UP WIRING (one line, owned by the human / Fable, NOT done here): to ship this pack, add
    ``register_domain_pack(SCREEN_OBSERVATION_V1)`` to ``percept/packs_default.py`` (or advertise
    this callable under the ``percept.packs`` entry-point group in ``pyproject.toml``). Left undone
    in this scaffold so it never races the sibling cards editing that registry block.
    """
    register(SCREEN_OBSERVATION_V1, overwrite=True)


__all__ = ["SCREEN_OBSERVATION_V1", "register_entrypoint_pack"]
