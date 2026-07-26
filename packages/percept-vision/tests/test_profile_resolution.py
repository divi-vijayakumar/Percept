"""Tests for percept.profile_resolution: situation -> WatchProfile registry key + overrides.

Covers each axis independently, the full 2x4x2 combination matrix, wildcard/unknown-axis fallbacks, purity
(determinism + no env-var leakage), the "never sprite" invariant, and resolve_profile's interaction with a
real WatchProfileRegistry (including the already-shipped percept.profiles_default defaults).
"""

from __future__ import annotations

import itertools
import os

import pytest

import percept.profiles_default as profiles_default
from percept.profile_resolution import (
    CAMERA_MOUNTS,
    GOAL_TEMPORALITIES,
    START_STATES,
    SituationResolution,
    resolve_profile,
    resolve_situation,
)
from percept.watchprofile import (
    WILDCARD,
    GateProfile,
    ProfileNotFoundError,
    WatchProfile,
    WatchProfileRegistry,
)


# ---------------------------------------------------------------------------
# Goal temporality axis (independent of the other two)
# ---------------------------------------------------------------------------


def test_stative_maps_to_single_frame_plan() -> None:
    res = resolve_situation("stative", "fixed", "start_anchored")
    assert res.key[1] == "stative"
    assert res.overrides["plan_mode"] == "single"


def test_dynamic_maps_to_ordered_window_plan() -> None:
    res = resolve_situation("dynamic", "fixed", "start_anchored")
    assert res.key[1] == "dynamic"
    assert res.overrides["plan_mode"] == "window"


def test_unknown_goal_temporality_falls_back_to_wildcard_key_and_window_plan() -> None:
    res = resolve_situation("unknown-goal-shape", "fixed", "start_anchored")
    assert res.key[1] == WILDCARD
    # Safest known default (matches percept.profiles_default.WILDCARD_DEFAULT's own choice), not "single".
    assert res.overrides["plan_mode"] == "window"


@pytest.mark.parametrize("garbage", [None, "", "STATIVE ", " Dynamic", 42, object(), ["stative"]])
def test_goal_temporality_case_and_type_variants(garbage: object) -> None:
    # Non-canonical spellings/types never raise; they either normalize correctly or fall back to wildcard.
    res = resolve_situation(garbage, "fixed", "start_anchored")
    assert res.overrides["plan_mode"] in ("single", "window")
    assert res.overrides["plan_mode"] != "sprite"


def test_goal_temporality_is_case_insensitive_and_trims_whitespace() -> None:
    assert resolve_situation("STATIVE ", "fixed", "start_anchored").overrides["plan_mode"] == "single"
    assert resolve_situation(" Dynamic", "fixed", "start_anchored").overrides["plan_mode"] == "window"


# ---------------------------------------------------------------------------
# Camera mount axis (independent of the other two)
# ---------------------------------------------------------------------------


def test_fixed_mount_maps_to_fixed_camera_environment_and_no_ego_compensation() -> None:
    res = resolve_situation("stative", "fixed", "start_anchored")
    assert res.key[0] == "fixed_camera"
    assert res.overrides["ego_compensate"] is False


@pytest.mark.parametrize("mount", ["wearable", "glasses", "handheld"])
def test_moving_mounts_map_to_handheld_environment_and_ego_compensation_on(mount: str) -> None:
    res = resolve_situation("stative", mount, "start_anchored")
    assert res.key[0] == "handheld"
    assert res.overrides["ego_compensate"] is True


def test_unknown_camera_mount_falls_back_to_wildcard_env_with_ego_compensation_on() -> None:
    # Guessing "moving" and being wrong is a no-op; guessing "fixed" and being wrong is a ~98% escalation
    # blowup (see module docstring) -- so the unknown-mount default is ego_compensate=True.
    res = resolve_situation("stative", "drone-mounted-thing", "start_anchored")
    assert res.key[0] == WILDCARD
    assert res.overrides["ego_compensate"] is True


def test_all_recognized_camera_mounts_are_covered_by_the_vocabulary_constant() -> None:
    assert set(CAMERA_MOUNTS) == {"fixed", "wearable", "glasses", "handheld"}


# ---------------------------------------------------------------------------
# Start-state axis (independent of the other two)
# ---------------------------------------------------------------------------


def test_start_anchored_relaxes_gate_to_require_false_first_false() -> None:
    res = resolve_situation("stative", "fixed", "start_anchored")
    assert res.key[2] == "start_anchored"
    assert res.overrides["gate"]["require_false_first"] is False


def test_transition_only_requires_false_first() -> None:
    res = resolve_situation("stative", "fixed", "transition")
    assert res.key[2] == "transition"
    assert res.overrides["gate"]["require_false_first"] is True


def test_unknown_start_state_falls_back_to_wildcard_and_conservative_gate() -> None:
    res = resolve_situation("stative", "fixed", "mystery-state")
    assert res.key[2] == WILDCARD
    # Conservative default: don't assume "already true" without evidence.
    assert res.overrides["gate"]["require_false_first"] is True


# ---------------------------------------------------------------------------
# salience_stat: NOT a situation axis (ADR-008 ownership transfer) — it belongs to the
# profile/source layer, so resolve_situation() never emits it, for any input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "goal_temporality,camera_mount,start_state",
    list(itertools.product(GOAL_TEMPORALITIES + (None, "bogus"), CAMERA_MOUNTS + (None, "bogus"), START_STATES + (None, "bogus"))),
)
def test_salience_stat_is_not_a_situation_override(goal_temporality, camera_mount, start_state) -> None:
    res = resolve_situation(goal_temporality, camera_mount, start_state)
    assert "salience_stat" not in res.overrides
    assert "stat_thresholds" not in res.overrides


# ---------------------------------------------------------------------------
# Full combination matrix over the recognized vocabulary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "goal_temporality,camera_mount,start_state",
    list(itertools.product(GOAL_TEMPORALITIES, CAMERA_MOUNTS, START_STATES)),
)
def test_full_recognized_combination_matrix(goal_temporality, camera_mount, start_state) -> None:
    res = resolve_situation(goal_temporality, camera_mount, start_state)

    expected_env = "fixed_camera" if camera_mount == "fixed" else "handheld"
    expected_goal_type = goal_temporality
    expected_state = start_state
    assert res.key == (expected_env, expected_goal_type, expected_state)

    assert res.overrides["plan_mode"] == ("single" if goal_temporality == "stative" else "window")
    assert res.overrides["ego_compensate"] == (camera_mount != "fixed")
    assert res.overrides["gate"]["require_false_first"] == (start_state == "transition")
    assert "salience_stat" not in res.overrides


# ---------------------------------------------------------------------------
# Wildcard / fully-unknown situation.
# ---------------------------------------------------------------------------


def test_fully_unknown_situation_falls_back_safely_on_every_axis() -> None:
    res = resolve_situation(None, None, None)
    assert res.key == (WILDCARD, WILDCARD, WILDCARD)
    assert res.overrides["plan_mode"] == "window"
    assert "salience_stat" not in res.overrides
    assert res.overrides["gate"]["require_false_first"] is True
    assert res.overrides["ego_compensate"] is True


# ---------------------------------------------------------------------------
# "Never sprite" invariant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "goal_temporality",
    ["stative", "dynamic", "sprite", "single", "window", "strip", None, "", 123, object()],
)
@pytest.mark.parametrize("camera_mount", ["fixed", "wearable", "glasses", "handheld", None, "orbital"])
@pytest.mark.parametrize("start_state", ["start_anchored", "transition", None, "unknown"])
def test_resolve_situation_never_selects_sprite(goal_temporality, camera_mount, start_state) -> None:
    res = resolve_situation(goal_temporality, camera_mount, start_state)
    assert res.overrides["plan_mode"] in ("single", "window")
    assert res.overrides["plan_mode"] != "sprite"


def test_resolve_profile_never_selects_sprite_across_the_full_default_registry() -> None:
    registry = WatchProfileRegistry()
    profiles_default.seed(registry, overwrite=True)
    for goal_temporality in GOAL_TEMPORALITIES + (None,):
        for camera_mount in CAMERA_MOUNTS + (None,):
            for start_state in START_STATES + (None,):
                profile, _overrides = resolve_profile(
                    goal_temporality, camera_mount, start_state, registry=registry
                )
                assert profile.plan_mode != "sprite"


# ---------------------------------------------------------------------------
# Purity: deterministic, no I/O, no env reads.
# ---------------------------------------------------------------------------


def test_resolve_situation_is_deterministic() -> None:
    a = resolve_situation("dynamic", "wearable", "transition")
    b = resolve_situation("dynamic", "wearable", "transition")
    assert a == b
    assert a is not b  # distinct objects, not memoized identity


def test_resolve_situation_result_mutation_does_not_leak_between_calls() -> None:
    first = resolve_situation("dynamic", "wearable", "transition")
    first.overrides["plan_mode"] = "sprite"  # mutate the caller's copy
    first.overrides["gate"]["require_false_first"] = False
    second = resolve_situation("dynamic", "wearable", "transition")
    assert second.overrides["plan_mode"] == "window"
    assert second.overrides["gate"]["require_false_first"] is True


def test_resolve_situation_ignores_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = resolve_situation("dynamic", "wearable", "transition")
    monkeypatch.setenv("PERCEPT_EGO_COMPENSATE", "0")
    monkeypatch.setenv("PERCEPT_SALIENCE_STAT", "mean")
    for key in list(os.environ):
        if key.startswith("PERCEPT_"):
            monkeypatch.setenv(key, "definitely-not-a-real-value")
    after = resolve_situation("dynamic", "wearable", "transition")
    assert after == baseline


# ---------------------------------------------------------------------------
# resolve_profile: interaction with a real WatchProfileRegistry.
# ---------------------------------------------------------------------------


def test_resolve_profile_overrides_a_custom_registered_profile() -> None:
    registry = WatchProfileRegistry()
    registry.register(
        WatchProfile(
            name="test-fixed-stative-start",
            environment="fixed_camera",
            goal_type="stative",
            state="start_anchored",
            salience_stat="mean",
            plan_mode="window",
            gate=GateProfile(require_false_first=True, refractory_s=3.0),
        )
    )
    profile, overrides = resolve_profile("stative", "fixed", "start_anchored", registry=registry)

    assert profile.plan_mode == "single"  # overridden from the registered "window"
    # salience_stat/stat_thresholds are NOT a situation override (ADR-008 ownership transfer,
    # C2) -- the registered profile's own declared "mean" passes through untouched.
    assert profile.salience_stat == "mean"
    assert profile.gate.require_false_first is False  # overridden from the registered True
    assert profile.gate.refractory_s == 3.0  # NOT overridden; rule axes do not touch refractory_s
    assert overrides["ego_compensate"] is False
    # name/environment/goal_type/state are passed through from whichever profile the registry matched.
    assert profile.name == "test-fixed-stative-start"


def test_resolve_profile_falls_back_through_registry_wildcards() -> None:
    registry = WatchProfileRegistry()
    broad = WatchProfile(
        name="broad-default",
        environment=WILDCARD,
        goal_type=WILDCARD,
        state=WILDCARD,
        salience_stat="mean",
        plan_mode="single",
    )
    registry.register(broad)

    # No exact/partial match exists for this key -- only the fully-wildcard profile is registered.
    profile, _overrides = resolve_profile("dynamic", "wearable", "transition", registry=registry)
    assert profile.name == "broad-default"
    assert profile.plan_mode == "window"  # overridden per the dynamic-goal rule
    assert profile.salience_stat == "mean"  # NOT overridden -- passes the registered value through


def test_resolve_profile_raises_when_registry_has_no_match_at_all() -> None:
    empty_registry = WatchProfileRegistry()
    with pytest.raises(ProfileNotFoundError):
        resolve_profile("stative", "fixed", "start_anchored", registry=empty_registry)


def test_resolve_profile_against_the_shipped_h0_defaults() -> None:
    registry = WatchProfileRegistry()
    profiles_default.seed(registry, overwrite=True)

    # Exact match onto percept.profiles_default.DYNAMIC_ACTION (fixed_camera, dynamic, transition). Its
    # shipped gate.require_false_first is False (schema default); the transition
    # rule overrides it to True, proving the override is applied rather than passed through untouched.
    profile, overrides = resolve_profile("dynamic", "fixed", "transition", registry=registry)
    assert profile.name == "dynamic-action"
    assert profile.gate.require_false_first is True
    assert overrides["gate"]["require_false_first"] is True

    # Exact match onto FIXED_CAM_LOCALIZED (fixed_camera, stative, start_anchored). Its shipped plan_mode
    # is "window"; the stative rule overrides it to "single".
    profile2, _ = resolve_profile("stative", "fixed", "start_anchored", registry=registry)
    assert profile2.name == "fixed-cam-localized"
    assert profile2.plan_mode == "single"
    assert profile2.gate.require_false_first is False  # start-anchored rule agrees with the shipped value

    # An unregistered exact combination on a moving mount still resolves via HANDHELD's wildcard
    # goal_type/state registration.
    profile3, overrides3 = resolve_profile("stative", "glasses", "transition", registry=registry)
    assert profile3.name == "handheld"
    assert overrides3["ego_compensate"] is True


def test_resolve_profile_ego_compensate_is_not_a_watchprofile_field() -> None:
    # Locks in the documented design choice: ego_compensate has no schema field yet (see
    # percept.profiles_default.EGO_COMPENSATION_NOTE), so it only ever appears in the overrides dict.
    registry = WatchProfileRegistry()
    profiles_default.seed(registry, overwrite=True)
    profile, overrides = resolve_profile("stative", "wearable", "start_anchored", registry=registry)
    assert not hasattr(profile, "ego_compensate")
    assert "ego_compensate" in overrides


def test_situation_resolution_shape() -> None:
    res = resolve_situation("dynamic", "handheld", "transition")
    assert isinstance(res, SituationResolution)
    assert isinstance(res.key, tuple) and len(res.key) == 3
    assert set(res.overrides) == {
        "plan_mode",
        "gate",
        "ego_compensate",
    }
