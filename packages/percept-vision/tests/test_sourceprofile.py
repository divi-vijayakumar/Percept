"""the implementation / ADR-005 — SourceProfile: an OPEN registry that must not touch gate semantics.

The three things these tests exist to protect:

1. The registry is OPEN — a third party adds ``industrial_line`` without editing a percept literal.
2. A source profile CANNOT change FIRE/ASK/SILENT, the rising edge, refractory or dwell. There is
   one Gate and one incident truth on every source.
3. ``screen_document`` ships a TUNED @1 localized admission default — the SAME
   ``block_max``/0.20 cutoff ``physical_scene`` carries, never lowered — with a mandatory coast
   floor underneath it for the page-scale content that eye still cannot resolve, and doctor
   reports which regime is active.
"""

from __future__ import annotations

import dataclasses

import pytest

from percept.doctor import (
    ADMISSION_SIGNAL_UNTUNED,
    SOURCE_PROFILE_EXPERIMENTAL,
    DoctorConfigError,
    build_parser,
    run_doctor,
)
from percept.harness.tier0 import Frame
from percept.source_profiles_default import (
    ALL_SOURCE_PROFILES,
    CONVERSATION,
    DEFAULT_SOURCE_PROFILE_NAME,
    PHYSICAL_SCENE,
    SCREEN_COAST_INTERVAL_S,
    SCREEN_DOCUMENT,
    seed,
)
from percept.sourceprofile import (
    DEFAULT_SOURCE_REGISTRY,
    FORBIDDEN_FIELDS,
    UNTUNED,
    SourceProfile,
    SourceProfileError,
    SourceProfileNotFoundError,
    SourceProfileRegistry,
    coast_sense_prior,
    get_source_profile,
    register_source_profile,
)


def _quiet_frames(n: int = 6):
    """A still stream — the doctor tests here are about the PROFILE panel, not about salience."""
    blank = Frame(w=48, h=36, pixels=tuple([0] * (48 * 36)))
    return [(float(i), blank) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. the registry is OPEN
# ---------------------------------------------------------------------------


def _industrial() -> SourceProfile:
    """A profile percept has never heard of — built entirely from public API."""
    return SourceProfile(
        name="industrial_line",
        modalities=frozenset({"video", "telemetry"}),
        temporal_unit="cycle",
        admission_strategy="plc_cycle_boundary",
        evidence_strategy="cycle_window_plus_telemetry",
        identity_strategy="part_serial",
        default_timebase="wall",
        parameters={"coast_interval_s": 5.0, "episode_unit": "shift"},
    )


@pytest.mark.parametrize(
    "name",
    ["industrial_line", "medical_imaging", "satellite_change", "code_editor", "financial_dashboard"],
)
def test_a_third_party_source_class_registers_without_editing_percept(name):
    registry = SourceProfileRegistry()
    profile = dataclasses.replace(_industrial(), name=name)
    registry.register(profile)
    assert registry.resolve(name) is profile
    assert name in registry.names()


def test_registry_refuses_to_silently_replace_a_different_profile():
    registry = SourceProfileRegistry()
    registry.register(_industrial())
    other = dataclasses.replace(_industrial(), temporal_unit="frame")
    with pytest.raises(SourceProfileError, match="already registered"):
        registry.register(other)
    registry.register(other, overwrite=True)      # explicit is fine
    assert registry.resolve("industrial_line").temporal_unit == "frame"
    registry.register(_industrial(), overwrite=True)  # equal-by-value re-register is a no-op path
    assert len(registry) == 1


def test_unknown_name_raises_with_the_registered_names_listed():
    registry = SourceProfileRegistry()
    registry.register(_industrial())
    with pytest.raises(SourceProfileNotFoundError, match="industrial_line"):
        registry.resolve("nope")


def test_process_registry_extension_point_round_trips():
    profile = dataclasses.replace(_industrial(), name="d7_test_only_profile")
    try:
        register_source_profile(profile)
        assert get_source_profile("d7_test_only_profile") is profile
    finally:
        DEFAULT_SOURCE_REGISTRY.unregister("d7_test_only_profile")


def test_seed_is_idempotent_on_an_isolated_registry():
    registry = SourceProfileRegistry()
    seed(registry)
    seed(registry)
    assert len(registry) == len(ALL_SOURCE_PROFILES)


# ---------------------------------------------------------------------------
# 2. a source profile MUST NOT change gate semantics
# ---------------------------------------------------------------------------


def test_sourceprofile_has_no_gate_field_at_all():
    fields = {f.name for f in dataclasses.fields(SourceProfile)}
    assert not (fields & FORBIDDEN_FIELDS), (
        "a SourceProfile field naming gate semantics would be a per-source gate in disguise"
    )
    assert fields == {
        "name", "modalities", "temporal_unit", "admission_strategy", "evidence_strategy",
        "identity_strategy", "default_timebase", "parameters", "stability", "notes",
    }


@pytest.mark.parametrize("bad", sorted(FORBIDDEN_FIELDS))
def test_gate_semantics_cannot_sneak_in_through_parameters(bad):
    with pytest.raises(SourceProfileError, match="gate semantics"):
        SourceProfile(
            name="sneaky",
            modalities=frozenset({"video"}),
            temporal_unit="frame",
            admission_strategy="x",
            evidence_strategy="y",
            identity_strategy="z",
            parameters={bad: 1},
        )


def test_no_shipped_profile_carries_gate_semantics():
    for profile in ALL_SOURCE_PROFILES:
        assert not (FORBIDDEN_FIELDS & set(profile.parameters))


def test_parameters_are_immutable():
    profile = _industrial()
    with pytest.raises(TypeError):
        profile.parameters["coast_interval_s"] = 0.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# 3. validation is about SHAPE, never a closed vocabulary
# ---------------------------------------------------------------------------


def test_modalities_must_be_a_non_empty_set_of_strings():
    with pytest.raises(SourceProfileError, match="modalities"):
        SourceProfile(
            name="n", modalities=frozenset(), temporal_unit="frame",
            admission_strategy="a", evidence_strategy="e", identity_strategy="i",
        )


def test_timebase_and_stability_are_closed_but_the_strategies_are_not():
    with pytest.raises(SourceProfileError, match="default_timebase"):
        SourceProfile(
            name="n", modalities=frozenset({"video"}), temporal_unit="frame",
            admission_strategy="a", evidence_strategy="e", identity_strategy="i",
            default_timebase="sundial",
        )
    with pytest.raises(SourceProfileError, match="stability"):
        SourceProfile(
            name="n", modalities=frozenset({"video"}), temporal_unit="frame",
            admission_strategy="a", evidence_strategy="e", identity_strategy="i",
            stability="pretty-good",
        )
    # …but a strategy nobody has ever heard of is accepted: the registry is open.
    ok = SourceProfile(
        name="n", modalities=frozenset({"lidar"}), temporal_unit="sweep",
        admission_strategy="point_cloud_delta", evidence_strategy="voxel_diff",
        identity_strategy="track_id",
    )
    assert ok.admission_strategy == "point_cloud_delta"


# ---------------------------------------------------------------------------
# 4. screen_document — tuned @1, coast underneath
# ---------------------------------------------------------------------------


def test_screen_document_is_tuned_at_the_same_cutoff_as_physical_scene():
    """the implementation promotion: a real, named admission signal — never UNTUNED — and never a lowered cutoff."""
    assert SCREEN_DOCUMENT.parameters["change_signal"] != UNTUNED
    assert SCREEN_DOCUMENT.admission_strategy != UNTUNED
    assert SCREEN_DOCUMENT.admission_tuned is True
    assert SCREEN_DOCUMENT.parameters["salience_stat"] == "block_max"
    assert SCREEN_DOCUMENT.parameters["salience_threshold"] == 0.20
    note = SCREEN_DOCUMENT.admission_note()
    assert "UNTUNED" not in note
    assert "admission via" in note


def test_screen_document_coast_is_mandatory_and_fixed():
    # The coast floor stays MANDATORY alongside the tuned default — it covers page-scale content the
    # localized statistic still cannot resolve (see the module's measured evidence).
    assert SCREEN_DOCUMENT.coast_interval_s == SCREEN_COAST_INTERVAL_S
    assert SCREEN_DOCUMENT.parameters["coast_required"] is True
    assert 1.0 <= SCREEN_DOCUMENT.coast_interval_s <= 2.0


def test_screen_document_identity_is_content_never_geometry():
    # IoU+label cannot separate INV-2291 from INV-2294: two invoices in the same screen region with
    # the same label are the SAME entity by construction under geometric identity.
    assert SCREEN_DOCUMENT.identity_strategy == "content_key"
    assert "iou" not in SCREEN_DOCUMENT.identity_strategy.lower()


def test_screen_document_shape():
    assert SCREEN_DOCUMENT.default_timebase == "media"        # files
    assert SCREEN_DOCUMENT.temporal_unit == "page"
    assert SCREEN_DOCUMENT.parameters["episode_unit"] == "page_interval"
    assert SCREEN_DOCUMENT.parameters["plan_mode"] == "window"    # never sprite (ADR-008 Corrections(a))
    parts = SCREEN_DOCUMENT.parameters["evidence_parts"]
    assert parts[:2] == ("full_screenshot", "changed_region_crop")
    assert "previous_screenshot" in SCREEN_DOCUMENT.parameters["evidence_optional_parts"]


def test_screen_document_notes_carry_the_measured_evidence_not_a_guess():
    for number in ("0.0919", "0.200", "48x36", "clean.mp4"):
        assert number in SCREEN_DOCUMENT.notes


def test_the_measured_thresholds_were_not_lowered_to_make_screen_admit():
    # The one thing this card must never do. screen_document's tuned default is the SAME calibrated
    # cutoff physical_scene carries, never a lowered one invented to force this fixture to admit.
    assert PHYSICAL_SCENE.parameters["salience_threshold"] == 0.20
    assert SCREEN_DOCUMENT.parameters["salience_threshold"] == 0.20
    assert SCREEN_DOCUMENT.parameters["salience_stat"] == PHYSICAL_SCENE.parameters["salience_stat"]


def test_client_tunes_the_shipped_profile_without_editing_percept():
    tuned = SCREEN_DOCUMENT.with_parameters(change_signal="acme-phash", coast_interval_s=1.0)
    assert tuned.admission_tuned is True
    assert tuned.coast_interval_s == 1.0
    assert tuned.parameters["change_signal"] == "acme-phash"
    # the shipped profile is untouched by a client's copy-on-write tune
    assert SCREEN_DOCUMENT.parameters["change_signal"] == "tier0_block_max_salience"
    assert SCREEN_DOCUMENT.coast_interval_s == SCREEN_COAST_INTERVAL_S


def test_the_coast_arms_a_real_sampling_controller_on_the_existing_seam():
    """The coast is not a documented intention — it lands on the runtime seam that already exists."""
    from percept.adapters import Goal
    from percept.runtime.worker import _capture_plan

    prior = coast_sense_prior(SCREEN_DOCUMENT)
    assert prior == {"base_fps": 0.5, "max_fps": 0.5}      # 1 / 2.0s
    goal = Goal(id="g0", condition="the document being edited changes", say="it changed",
                state={"sense": prior})
    _capture_fps, coast, required_fps = _capture_plan([goal], 4.0)
    assert coast is not None, "a screen_document goal must coast even when nothing moves"
    assert required_fps == 0.5
    # …and a profile with a real change signal declares no coast, so nothing changes for cameras.
    assert coast_sense_prior(PHYSICAL_SCENE) is None
    assert _capture_plan([Goal(id="g1", condition="someone walks in", say="hi")], 4.0)[1] is None


def test_conversation_is_experimental_and_says_why():
    assert CONVERSATION.stability == "experimental"
    reasons = " ".join(CONVERSATION.parameters["experimental_reasons"])
    assert "speaker identity" in reasons.lower()
    assert "24s" in reasons


def test_physical_scene_is_the_only_calibrated_class_and_the_default():
    assert DEFAULT_SOURCE_PROFILE_NAME == PHYSICAL_SCENE.name
    assert PHYSICAL_SCENE.admission_tuned is True
    assert PHYSICAL_SCENE.stability == "stable"


# ---------------------------------------------------------------------------
# 5. doctor reports the active source profile (the implementation item 6)
# ---------------------------------------------------------------------------


def test_doctor_defaults_to_physical_scene_and_labels_it_defaulted():
    report = run_doctor(frames=_quiet_frames(), goals=["someone walks in"])
    assert report.plan.source_profile == "physical_scene"
    assert report.plan.source_profile_origin == "default"
    assert not report.has(ADMISSION_SIGNAL_UNTUNED)


def test_doctor_says_the_screen_profile_carries_a_tuned_localized_default():
    """the implementation: screen_document no longer reports ADMISSION_SIGNAL_UNTUNED — it ships a tuned
    localized default (block_max@0.20) with a coast floor underneath it, not coast alone."""
    report = run_doctor(
        frames=_quiet_frames(), goals=["the document being edited changes"],
        source_profile="screen_document",
    )
    assert report.plan.source_profile == "screen_document"
    assert report.plan.source_profile_origin == "explicit"
    assert not report.has(ADMISSION_SIGNAL_UNTUNED)
    assert "screen_document" in report.render()
    assert report.to_dict()["plan"]["source_profile"]["profile"]["admission_tuned"] is True


def test_doctor_flags_an_experimental_source_profile():
    report = run_doctor(
        frames=_quiet_frames(), goals=["the caller asks for a refund"],
        source_profile="conversation",
    )
    assert report.has(SOURCE_PROFILE_EXPERIMENTAL)


def test_doctor_accepts_a_profile_object_and_rejects_a_typo():
    report = run_doctor(
        frames=_quiet_frames(), goals=["a part is misaligned"], source_profile=_industrial(),
    )
    assert report.plan.source_profile == "industrial_line"
    with pytest.raises(DoctorConfigError, match="unknown source profile"):
        run_doctor(frames=_quiet_frames(), goals=["x"], source_profile="screen-document")


def test_doctor_cli_exposes_source_profile_as_open_text_not_a_closed_choice():
    args = build_parser().parse_args(
        ["--source", "x.mp4", "--goal", "g", "--source-profile", "industrial_line"]
    )
    assert args.source_profile == "industrial_line"


def test_untuned_admission_does_not_change_the_exit_code_contract():
    # A source profile must never invent an exit code: exit 2 here is zero coverage, as always.
    quiet = run_doctor(frames=_quiet_frames(), goals=["the page changes"],
                       source_profile="screen_document")
    assert quiet.exit_code == 2


# ---------------------------------------------------------------------------
# 6. select_source_profile — explicit wins, strong metadata infers, ambiguous never auto-switches
# ---------------------------------------------------------------------------


def test_select_source_profile_explicit_wins_over_any_metadata():
    from percept.sourceprofile import select_source_profile

    sel = select_source_profile(explicit="physical_scene", uri="screen://display0")
    assert sel.name == "physical_scene"
    assert sel.reason == "explicit"
    assert sel.recommend is False

    sel2 = select_source_profile(explicit=SCREEN_DOCUMENT, uri="webcam:0")
    assert sel2.name == "screen_document"
    assert sel2.reason == "explicit"


def test_select_source_profile_strong_metadata_infers():
    from percept.sourceprofile import select_source_profile

    for kwargs in (
        {"uri": "screen://display0"},
        {"source_kind": "screen"},
        {"capture_type": "screen_recording"},
        {"capture_type": "screenshot"},
    ):
        sel = select_source_profile(**kwargs)
        assert sel.name == "screen_document"
        assert sel.reason == "inferred"
        assert sel.recommend is False


def test_select_source_profile_ambiguous_never_auto_switches():
    from percept.sourceprofile import select_source_profile

    # A plain file path -- could be surveillance, handheld, or a recorded meeting.
    sel = select_source_profile(uri="clip.mp4")
    assert sel.name == "physical_scene"
    assert sel.reason == "default"
    assert sel.recommend is True   # something to flag, but never silently switched


def test_select_source_profile_unambiguous_camera_uris_get_no_recommendation():
    from percept.sourceprofile import select_source_profile

    for uri in ("webcam:0", "rtsp://camera.local/stream"):
        sel = select_source_profile(uri=uri)
        assert sel.name == "physical_scene"
        assert sel.reason == "default"
        assert sel.recommend is False   # unambiguously camera-shaped -- nothing to flag


def test_select_source_profile_no_metadata_at_all_gets_no_recommendation():
    from percept.sourceprofile import select_source_profile

    sel = select_source_profile()
    assert sel.name == "physical_scene"
    assert sel.reason == "default"
    assert sel.recommend is False


def test_doctor_resolve_source_profile_wires_the_selection_rule():
    """percept.doctor._resolve_source_profile is the doctor-facing wrapper around
    select_source_profile — proves the three-tier rule reaches doctor's own resolution."""
    from percept.doctor import _resolve_source_profile

    # explicit
    profile, origin, recommend = _resolve_source_profile("screen_document", uri="clip.mp4")
    assert profile.name == "screen_document"
    assert origin == "explicit"
    assert recommend is False

    # strong metadata -> inferred
    profile2, origin2, recommend2 = _resolve_source_profile(None, uri="screen://display0")
    assert profile2.name == "screen_document"
    assert origin2 == "inferred"
    assert recommend2 is False

    # ambiguous plain file path -> default, with a recommendation
    profile3, origin3, recommend3 = _resolve_source_profile(None, uri="meeting.mp4")
    assert profile3.name == "physical_scene"
    assert origin3 == "default"
    assert recommend3 is True

    # no uri at all (the frames= path) -> default, no recommendation
    profile4, origin4, recommend4 = _resolve_source_profile(None, uri=None)
    assert profile4.name == "physical_scene"
    assert origin4 == "default"
    assert recommend4 is False
