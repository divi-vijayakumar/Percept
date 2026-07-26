"""the implementation / ADR-007 — the envelope is enforced in CONFIG, so these tests check the CONFIG.

Three domains came back OUT of the seven-domain evaluation. Prose does not stop a client wiring
something, so each refusal has to fail here rather than in review:

(a) percept ships NO person-identifying subject resolver, and one cannot be constructed;
(b) ``modality='act'`` + physical actuation is a TYPED declaration a tool reads (the ADR-005
    ``UNTUNED`` pattern), and the declaration is enforced by ``ConsentProfile.permits``;
(c) "assistant-class delivery only" is honestly reported as enforced in DOCS, with the gap named.
"""

from __future__ import annotations

import pytest

from percept.consent import ConsentProfile
from percept.contracts import Goal, GoalOrigin
from percept.doctor import (
    EXIT_OUT_OF_ENVELOPE,
    OUT_OF_ENVELOPE_ACTUATION,
    DoctorConfigError,
    run_doctor,
)
from percept.envelope import (
    ACTUATION_CLASSES,
    ATTENTION_CLASS_ENFORCEMENT,
    ATTENTION_CLASS_GAP,
    DEFAULT_SUBJECT_RESOLVERS,
    ENFORCEMENTS,
    OUT_OF_ENVELOPE,
    PHYSICAL_ACTUATION,
    EnvelopeViolation,
    SubjectResolver,
    SubjectResolverRegistry,
    actuation_check,
    assert_no_person_identifying_resolver,
    client_asserted_subject,
    envelope_summary,
)
from percept.harness.tier0 import Frame
from percept.incidents import AttentionPolicy, RequestedDelivery


# --------------------------------------------------------------------------- the verdicts as data


def test_all_three_out_verdicts_are_typed_and_carry_a_survivor():
    """The three OUT domains exist as data, not as a paragraph, and each names its survivor."""
    assert set(OUT_OF_ENVELOPE) == {
        "security_person_id",
        "robotics_control_loop",
        "passive_analytics",
    }
    for v in OUT_OF_ENVELOPE.values():
        assert v.verdict == "out"
        assert v.enforcement in ENFORCEMENTS
        assert v.narrowed_survivor.strip(), f"{v.domain} states no narrowed survivor"
        assert v.boundary.strip(), f"{v.domain} states no boundary"
        assert v.domain in v.sentence()


def test_the_two_person_domains_are_structurally_enforced_not_merely_documented():
    """The two refusals that hinge on a person key must be 'structural' — a config change alone
    cannot cross them, because the excluded object does not exist."""
    assert OUT_OF_ENVELOPE["security_person_id"].enforcement == "structural"
    assert OUT_OF_ENVELOPE["passive_analytics"].enforcement == "structural"
    assert OUT_OF_ENVELOPE["robotics_control_loop"].enforcement == "declared"


# --------------------------------------------------------------------------- (a) no person resolver


def test_no_shipped_resolver_identifies_persons():
    """THE deliverable of Z7(a): the shipped registry contains no person-identifying resolver."""
    assert DEFAULT_SUBJECT_RESOLVERS.person_identifying() == []
    assert_no_person_identifying_resolver()
    assert len(DEFAULT_SUBJECT_RESOLVERS) >= 4, "the shipped resolvers should be enumerated, not empty"
    for r in DEFAULT_SUBJECT_RESOLVERS:
        assert not r.identifies_persons
        assert r.derives_from != "pixels"


def test_a_person_resolver_derived_from_pixels_cannot_be_constructed():
    """Not 'is not registered' — cannot exist. This is the config-not-prose line."""
    for subject_class in ("person", "face", "voice", "biometric"):
        with pytest.raises(EnvelopeViolation) as exc:
            SubjectResolver(
                name="reid", subject_class=subject_class, derives_from="pixels"
            )
        assert "security_person_id" in str(exc.value)


def test_a_scene_local_person_key_may_not_persist():
    """Scene-local individuation is the survivor ONLY while it dies with the run; persisting it is
    a re-identification record however it was produced."""
    ok = SubjectResolver(name="who_is_at_the_door", subject_class="person",
                         derives_from="scene_local", persists=False)
    assert not ok.identifies_persons
    with pytest.raises(EnvelopeViolation) as exc:
        SubjectResolver(name="who_is_at_the_door", subject_class="person",
                        derives_from="scene_local", persists=True)
    assert "PERSIST" in str(exc.value)


def test_bound_binding_requires_a_client_assertion():
    """binding='bound' names a real subject — reachable only when THEIR system asserts it."""
    with pytest.raises(EnvelopeViolation):
        SubjectResolver(name="x", subject_class="object", derives_from="scene_local",
                        binding="bound", asserted_by="acme")
    with pytest.raises(EnvelopeViolation):
        SubjectResolver(name="x", subject_class="person", derives_from="client_asserted",
                        binding="bound", asserted_by="")


def test_client_asserted_person_key_is_the_only_door_in():
    """A badge scan / the client's own re-ID service may enter as a bound key — and percept still
    derived nothing from pixels."""
    key = client_asserted_subject(value="emp-4417", asserted_by="acme-badge-reader")
    assert key.binding == "bound"
    assert key.derives_from == "client_asserted"
    assert key.asserted_by == "acme-badge-reader"
    assert key.persists is True
    assert key.identifies_persons is False  # THEIR assertion, not percept's derivation
    # ...and it may be registered, because it is not percept identifying anyone.
    reg = SubjectResolverRegistry()
    reg.register(key)
    assert reg.person_identifying() == []


def test_registry_refuses_a_person_identifying_resolver_independently_of_the_constructor():
    """Second, independent guard: a duck-typed object that dodged __post_init__ still cannot land
    in the registry."""

    class Sneaky(SubjectResolver):
        def __init__(self):  # bypass __post_init__ entirely
            object.__setattr__(self, "name", "sneaky_reid")
            object.__setattr__(self, "subject_class", "person")
            object.__setattr__(self, "derives_from", "pixels")
            object.__setattr__(self, "binding", "bound")
            object.__setattr__(self, "asserted_by", "")
            object.__setattr__(self, "persists", True)
            object.__setattr__(self, "notes", "")

    reg = SubjectResolverRegistry()
    with pytest.raises(EnvelopeViolation) as exc:
        reg.register(Sneaky())
    assert "identifies persons" in str(exc.value)
    assert reg.names() == []


def test_assert_helper_catches_a_third_party_registration():
    """The invariant is checked against the LIVE registry, so a resolver forced past both guards
    (as a careless future change might) still fails loudly."""
    reg = SubjectResolverRegistry()
    forced = SubjectResolver(name="bad", subject_class="object", derives_from="scene_local")
    object.__setattr__(forced, "subject_class", "person")
    object.__setattr__(forced, "persists", True)
    reg._resolvers["bad"] = forced
    with pytest.raises(EnvelopeViolation):
        assert_no_person_identifying_resolver(reg)


# --------------------------------------------------------------------------- (b) typed actuation


def test_actuation_is_a_typed_field_on_consentprofile_with_a_closed_vocabulary():
    assert "actuation" in ConsentProfile.__dataclass_fields__
    assert ConsentProfile().actuation == "none"
    assert ACTUATION_CLASSES == ("none", "screen", "physical")
    with pytest.raises(EnvelopeViolation):
        ConsentProfile(actuation="robot_arm")


def test_declaring_physical_actuation_refuses_every_act():
    """The declaration is enforcement, not a label: permits() goes to False for everything."""
    permissive = ConsentProfile(allow_computer_use=True, allowed_actions=frozenset({"click"}))
    assert permissive.permits("click") is True
    declared = ConsentProfile(allow_computer_use=True, allowed_actions=frozenset({"click"}),
                              actuation=PHYSICAL_ACTUATION)
    assert declared.permits("click") is False
    assert declared.envelope().in_envelope is False
    assert declared.envelope().verdict is OUT_OF_ENVELOPE["robotics_control_loop"]


def test_screen_actuation_is_the_narrowed_survivor_and_still_works():
    p = ConsentProfile(allow_computer_use=True, allowed_actions=frozenset({"click"}),
                       actuation="screen")
    assert p.permits("click") is True
    assert p.envelope().in_envelope is True


def test_actuation_check_needs_the_PAIR_to_be_out():
    """modality='act' alone is fine (advisory acts); physical alone is fine (percept only speaks).
    The pair is the OUT."""
    assert actuation_check(modality="act", actuation="screen").in_envelope is True
    speak_physical = actuation_check(modality="speak", actuation="physical")
    assert speak_physical.in_envelope is True
    assert speak_physical.verdict is not None  # still reported, never silent
    assert actuation_check(modality="act", actuation="physical").in_envelope is False


def test_percept_act_reports_the_envelope_refusal_not_a_permission_miss():
    """A client debugging at 2am must read WHY, not hunt an allowlist."""
    from percept.agent import Percept
    from percept.config import ConsentSettings, Settings
    from percept.consent import ConsentDenied

    core = Percept.create(
        settings=Settings(
            consent=ConsentSettings(
                allow_computer_use=True,
                allowed_actions=frozenset({"click"}),
                actuation=PHYSICAL_ACTUATION,
            )
        ),
        discover_plugins=False,
    )
    assert core.consent.actuation == PHYSICAL_ACTUATION  # the declaration reached the runtime
    core.add_goal(Goal(id="g1", condition="the arm is clear", say="clear", modality="act"))
    with pytest.raises(ConsentDenied) as exc:
        core.act("g1", "click")
    assert "actuation='physical'" in str(exc.value)
    assert "robotics_control_loop" in str(exc.value)
    assert core.action_log.entries("act_blocked"), "a blocked act is still audited"


# --------------------------------------------------------------------------- doctor says it


def _frames(n: int = 4) -> list[tuple[float, Frame]]:
    return [(float(i) * 0.5, Frame(w=8, h=6, pixels=bytes([10 + i] * 48))) for i in range(n)]


def test_doctor_reports_the_envelope_on_every_run():
    report = run_doctor(frames=_frames(), goals=["the door opens"])
    env = report.envelope
    assert env is not None
    assert env.check.actuation == "none"
    assert env.person_identifying_resolvers == ()
    assert set(env.subject_resolvers) == set(DEFAULT_SUBJECT_RESOLVERS.names())
    blob = report.to_dict()["envelope"]
    assert blob["schema"] == "percept.envelope/1"
    assert set(blob["out_of_envelope"]) == set(OUT_OF_ENVELOPE)
    assert "ENVELOPE" in report.render()


def test_doctor_exits_6_and_names_the_verdict_when_the_pair_is_declared():
    goal = Goal(id="arm", condition="the part is aligned", say="aligned", modality="act")
    report = run_doctor(frames=_frames(), goals=[goal], actuation="physical")
    assert report.exit_code == EXIT_OUT_OF_ENVELOPE
    assert report.has(OUT_OF_ENVELOPE_ACTUATION)
    finding = next(f for f in report.findings if f.code == OUT_OF_ENVELOPE_ACTUATION)
    assert finding.severity == "error"
    assert "control loop" in finding.detail
    assert report.envelope.act_goal_ids == ("arm",)


def test_doctor_warns_when_physical_is_declared_but_nothing_acts():
    report = run_doctor(frames=_frames(), goals=["the door opens"], actuation="physical")
    finding = next(f for f in report.findings if f.code == OUT_OF_ENVELOPE_ACTUATION)
    assert finding.severity == "warn"
    assert report.exit_code != EXIT_OUT_OF_ENVELOPE


def test_doctor_rejects_an_unknown_actuation_as_a_config_error():
    with pytest.raises(DoctorConfigError):
        run_doctor(frames=_frames(), goals=["the door opens"], actuation="robot_arm")


def test_doctor_cli_exposes_the_actuation_flag():
    from percept.doctor import build_parser

    ns = build_parser().parse_args(["--source", "x.mp4", "--goal", "g", "--actuation", "physical"])
    assert ns.actuation == "physical"


# --------------------------------------------------------------------------- (c) the honest report


def test_assistant_class_only_is_reported_as_docs_enforced_with_the_gap_named():
    """The contract accepts 'only in docs, here is what it would take' — but it must be CHECKED, so the
    claim cannot drift away from the code."""
    assert ATTENTION_CLASS_ENFORCEMENT == "docs"
    assert "channel_class" in ATTENTION_CLASS_GAP
    assert "AttentionPolicy" in ATTENTION_CLASS_GAP


def test_the_policy_surface_really_only_clamps_interrupt_authority():
    """The evidence behind the 'docs' verdict: AttentionPolicy's inputs contain nothing that says
    WHO listens, and its only knobs are interrupt allowlists."""
    fields = set(AttentionPolicy.__dataclass_fields__)
    assert fields == {"allow_interrupt_consequences", "allow_interrupt_goal_ids"}
    assert "channel" not in " ".join(fields)
    assert "preferred_channel" in RequestedDelivery.__dataclass_fields__

    class _Inc:
        id, concern_id, message, ts = "i1", "g1", "hi", 0.0

    intent = AttentionPolicy().clamp(
        requested=RequestedDelivery(preempt="interrupt", preferred_channel="factory-horn"),
        origin=GoalOrigin.STRUCTURED_TRUSTED,
        consequence=None,
        incident=_Inc(),
    )
    # Trusted origin earns the interrupt; the CONSUMER CLASS ('factory-horn') passes through
    # untouched, which is exactly why the assistant-class claim is docs-only today.
    assert intent.preempt == "interrupt"
    assert intent.preferred_channel == "factory-horn"


def test_adr007_citations_match_the_code():
    """ADR-007 is a map of this code (the ADR-005 / the implementation rule): its load-bearing line cites and
    its actuation vocabulary must be checkable, not trusted."""
    import inspect
    from pathlib import Path

    import percept.contracts as contracts_mod
    import percept.sourceprofile as sp_mod

    adr = Path(inspect.getfile(sp_mod)).resolve().parents[3] / "docs" / "adr" / (
        "ADR-007-envelope-enforcement.md"
    )
    if not adr.exists():  # docs/ live outside the installed package
        pytest.skip(f"{adr} not present in this checkout")
    text = adr.read_text()

    # Assert the SYMBOL EXISTS, not the line it sits on. A line-number citation is broken by any
    # edit above it: this test failed the moment the implementation added a scope paragraph to
    # sourceprofile.py's module docstring, even though both references were individually correct.
    # A citation test should pin what is cited, not where it happens to be today.
    def defines(mod, symbol: str) -> bool:
        return symbol in Path(inspect.getfile(mod)).read_text()

    assert defines(sp_mod, "UNTUNED")
    assert defines(sp_mod, "FORBIDDEN_FIELDS")
    assert defines(contracts_mod, "modality")
    assert defines(contracts_mod, "preempt")
    for cls in ACTUATION_CLASSES:
        assert f"`{cls}`" in text or f"actuation=\"{cls}\"" in text or cls in text
    assert "EXIT_OUT_OF_ENVELOPE" in text
    assert 'ATTENTION_CLASS_ENFORCEMENT = "docs"' in text


def test_envelope_summary_is_one_dict_a_tool_can_print():
    s = envelope_summary()
    assert s["schema"] == "percept.envelope/1"
    assert s["person_identifying_resolvers"] == []
    assert s["attention_class_enforcement"] == "docs"
    assert set(s["out_of_envelope"]) == set(OUT_OF_ENVELOPE)
