"""— profile versioning + provenance: content-addressed, auditable, reversible.

Three properties are load-bearing and each is asserted directly here:

1. **Lossless round-trip.** ``profile_from_dict(profile_to_dict(p)) == p`` for every shipped default and
   for hand-built edge cases; a payload with an unknown or missing field raises rather than silently
   loading a *different* profile under the same name.
2. **Content addressing.** The hash depends on the knobs and nothing else — not on dict insertion order,
   not on the provenance stamp — and any single knob flip changes it.
3. **Reversible diffs.** ``diff_registries`` reports field-level deltas, and ``inverse()`` composed with
   the original is the identity on the registry content.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from dataclasses import replace
from typing import get_type_hints

from percept import profile_versioning, profiles_default
from percept.profile_versioning import (
    ABSENT,
    PROFILE_SCHEMA_VERSION,
    FieldDelta,
    ProfileProvenance,
    VersionedProfile,
    canonical_json,
    content_hash,
    diff_profiles,
    diff_registries,
    load_registry,
    profile_from_dict,
    profile_to_dict,
    registry_hash,
    registry_to_dicts,
    short_hash,
    stamp,
)
from percept.watchprofile import (
    CadenceProfile,
    GateProfile,
    GoalDensityGuidance,
    WatchProfile,
    WatchProfileError,
    WatchProfileRegistry,
)


def _profile(**overrides) -> WatchProfile:
    base = dict(
        name="p",
        environment="fixed_camera",
        goal_type="stative",
        state="start_anchored",
        salience_stat="block_max",
        stat_thresholds={"block_max": 0.20},
        plan_mode="window",
        cadence=CadenceProfile(base_fps=2.0, max_fps=10.0, escalate_threshold=0.3, hold_s=1.0),
        gate=GateProfile(require_false_first=True, refractory_s=15.0),
        goal_density_guidance=GoalDensityGuidance(max_recommended_goals=2, note="n"),
        notes="hello",
    )
    base.update(overrides)
    return WatchProfile(**base)


def _seeded() -> WatchProfileRegistry:
    reg = WatchProfileRegistry()
    profiles_default.seed(reg)
    return reg


class SerializationTests(unittest.TestCase):
    def test_round_trip_is_exact_for_a_fully_populated_profile(self):
        p = _profile()
        self.assertEqual(profile_from_dict(profile_to_dict(p)), p)

    def test_round_trip_is_exact_for_every_shipped_default(self):
        for p in profiles_default.ALL_PROFILES:
            with self.subTest(p.name):
                self.assertEqual(profile_from_dict(profile_to_dict(p)), p)

    def test_serialized_form_is_json_safe(self):
        payload = profile_to_dict(_profile())
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_every_watchprofile_field_is_serialized(self):
        payload = profile_to_dict(_profile())
        self.assertEqual(set(payload), set(WatchProfile.__dataclass_fields__))

    def test_nested_subprofiles_are_fully_serialized(self):
        payload = profile_to_dict(_profile())
        self.assertEqual(set(payload["cadence"]), set(CadenceProfile.__dataclass_fields__))
        self.assertEqual(set(payload["gate"]), set(GateProfile.__dataclass_fields__))
        self.assertEqual(
            set(payload["goal_density_guidance"]), set(GoalDensityGuidance.__dataclass_fields__)
        )

    def test_defaults_round_trip_when_optional_sections_are_absent(self):
        minimal = {
            "name": "m",
            "environment": "e",
            "goal_type": "g",
            "state": "s",
        }
        p = profile_from_dict(minimal)
        self.assertEqual(p.cadence, CadenceProfile())
        self.assertEqual(p.gate, GateProfile())
        self.assertEqual(p.stat_thresholds, {})

    def test_unknown_top_level_field_raises(self):
        payload = profile_to_dict(_profile())
        payload["ego_compensate"] = True
        with self.assertRaises(WatchProfileError):
            profile_from_dict(payload)

    def test_unknown_nested_field_raises(self):
        payload = profile_to_dict(_profile())
        payload["cadence"]["burst_fps"] = 30.0
        with self.assertRaises(WatchProfileError):
            profile_from_dict(payload)

    def test_missing_required_field_raises(self):
        payload = profile_to_dict(_profile())
        del payload["environment"]
        with self.assertRaises(WatchProfileError):
            profile_from_dict(payload)

    def test_value_validation_is_still_the_dataclasses_job(self):
        payload = profile_to_dict(_profile())
        payload["salience_stat"] = "nonsense"
        with self.assertRaises(WatchProfileError):
            profile_from_dict(payload)

    def test_non_mapping_payload_raises(self):
        with self.assertRaises(WatchProfileError):
            profile_from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_profile_to_dict_rejects_non_profile(self):
        with self.assertRaises(WatchProfileError):
            profile_to_dict({"name": "not a profile"})  # type: ignore[arg-type]


class ContentHashTests(unittest.TestCase):
    def test_hash_is_stable_and_sha256_shaped(self):
        h = content_hash(_profile())
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))
        self.assertEqual(h, content_hash(_profile()))

    def test_hash_is_independent_of_stat_threshold_insertion_order(self):
        a = _profile(stat_thresholds={"block_max": 0.2, "p95": 0.4})
        b = _profile(stat_thresholds={"p95": 0.4, "block_max": 0.2})
        self.assertEqual(content_hash(a), content_hash(b))

    def test_any_knob_change_changes_the_hash(self):
        base = _profile()
        variants = [
            _profile(salience_stat="p95"),
            _profile(plan_mode="single"),
            _profile(stat_thresholds={"block_max": 0.21}),
            _profile(gate=GateProfile(require_false_first=False, refractory_s=15.0)),
            _profile(gate=GateProfile(require_false_first=True, refractory_s=14.0)),
            _profile(cadence=replace(base.cadence, base_fps=1.0)),
            _profile(goal_density_guidance=GoalDensityGuidance(max_recommended_goals=1, note="n")),
            _profile(notes="different"),
            _profile(name="other"),
        ]
        for v in variants:
            with self.subTest(v.notes or v.name):
                self.assertNotEqual(content_hash(base), content_hash(v))

    def test_hash_accepts_a_serialized_dict_and_agrees_with_the_object(self):
        p = _profile()
        self.assertEqual(content_hash(p), content_hash(profile_to_dict(p)))

    def test_short_hash_is_a_prefix(self):
        p = _profile()
        self.assertEqual(short_hash(p, 12), content_hash(p)[:12])
        with self.assertRaises(WatchProfileError):
            short_hash(p, 0)

    def test_canonical_json_sorts_keys(self):
        self.assertEqual(canonical_json({"b": 1, "a": 2}), '{"a":2,"b":1}')


class ProvenanceTests(unittest.TestCase):
    def test_stamp_records_the_eval_run(self):
        vp = stamp(
            _profile(),
            eval_run_id="h0-2026-07-19",
            harness_version="0.7.0",
            source="charades-e2e",
            notes="recall 0.68 -> 0.90",
        )
        self.assertEqual(vp.provenance.eval_run_id, "h0-2026-07-19")
        self.assertEqual(vp.provenance.harness_version, "0.7.0")
        self.assertTrue(vp.provenance.created_at)
        self.assertEqual(vp.schema_version, PROFILE_SCHEMA_VERSION)

    def test_provenance_does_not_participate_in_the_content_hash(self):
        p = _profile()
        a = stamp(p, eval_run_id="run-a", harness_version="0.7.0")
        b = stamp(p, eval_run_id="run-b", harness_version="9.9.9", created_at="2020-01-01T00:00:00+00:00")
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertEqual(a.content_hash, content_hash(p))

    def test_restamp_keeps_the_identity_and_swaps_the_stamp(self):
        vp = stamp(_profile(), eval_run_id="run-a", harness_version="0.7.0")
        later = vp.restamp(ProfileProvenance(eval_run_id="run-b", harness_version="0.8.0"))
        self.assertEqual(later.content_hash, vp.content_hash)
        self.assertEqual(later.provenance.eval_run_id, "run-b")

    def test_empty_provenance_fields_are_rejected(self):
        with self.assertRaises(WatchProfileError):
            ProfileProvenance(eval_run_id="", harness_version="0.7.0")
        with self.assertRaises(WatchProfileError):
            ProfileProvenance(eval_run_id="r", harness_version="   ")

    def test_versioned_profile_round_trips_through_json(self):
        vp = stamp(_profile(), eval_run_id="run-a", harness_version="0.7.0", notes="n")
        back = VersionedProfile.from_json(vp.to_json())
        self.assertEqual(back.profile, vp.profile)
        self.assertEqual(back.provenance, vp.provenance)
        self.assertEqual(back.content_hash, vp.content_hash)
        self.assertEqual(back.to_json(), vp.to_json())

    def test_envelope_carries_the_hash_as_a_checksum(self):
        vp = stamp(_profile(), eval_run_id="run-a", harness_version="0.7.0")
        payload = vp.to_dict()
        self.assertEqual(payload["content_hash"], content_hash(vp.profile))

    def test_hand_edited_body_under_an_old_hash_is_rejected(self):
        vp = stamp(_profile(), eval_run_id="run-a", harness_version="0.7.0")
        payload = vp.to_dict()
        payload["profile"]["stat_thresholds"]["block_max"] = 0.9  # tamper, keep the old address
        with self.assertRaises(WatchProfileError) as ctx:
            VersionedProfile.from_dict(payload)
        self.assertIn("content_hash mismatch", str(ctx.exception))

    def test_unsupported_schema_version_is_rejected(self):
        vp = stamp(_profile(), eval_run_id="run-a", harness_version="0.7.0")
        payload = vp.to_dict()
        payload["schema_version"] = PROFILE_SCHEMA_VERSION + 1
        with self.assertRaises(WatchProfileError):
            VersionedProfile.from_dict(payload)

    def test_missing_envelope_sections_are_rejected(self):
        vp = stamp(_profile(), eval_run_id="run-a", harness_version="0.7.0")
        for missing in ("profile", "provenance"):
            payload = vp.to_dict()
            del payload[missing]
            payload.pop("content_hash", None)
            with self.subTest(missing):
                with self.assertRaises(WatchProfileError):
                    VersionedProfile.from_dict(payload)


class RegistrySerializationTests(unittest.TestCase):
    def test_seeded_registry_round_trips(self):
        reg = _seeded()
        rebuilt = load_registry(registry_to_dicts(reg))
        self.assertEqual(len(rebuilt), len(reg))
        self.assertEqual(
            sorted(p.key for p in rebuilt), sorted(p.key for p in reg)
        )
        for p in reg:
            self.assertEqual(rebuilt.get(*p.key), p)

    def test_load_registry_defaults_to_a_fresh_registry(self):
        from percept.watchprofile import DEFAULT_REGISTRY

        before = len(DEFAULT_REGISTRY)
        rebuilt = load_registry(registry_to_dicts(_seeded()))
        self.assertIsNot(rebuilt, DEFAULT_REGISTRY)
        self.assertEqual(len(DEFAULT_REGISTRY), before)

    def test_registry_hash_is_order_independent(self):
        a = WatchProfileRegistry()
        b = WatchProfileRegistry()
        for p in profiles_default.ALL_PROFILES:
            a.register(p)
        for p in reversed(profiles_default.ALL_PROFILES):
            b.register(p)
        self.assertEqual(registry_hash(a), registry_hash(b))

    def test_registry_hash_changes_when_a_knob_changes(self):
        a = _seeded()
        b = _seeded()
        tweaked = replace(profiles_default.FIXED_CAM_LOCALIZED, stat_thresholds={"block_max": 0.25})
        b.register(tweaked, overwrite=True)
        self.assertNotEqual(registry_hash(a), registry_hash(b))


class DiffTests(unittest.TestCase):
    def test_identical_registries_diff_empty(self):
        diff = diff_registries(_seeded(), _seeded())
        self.assertTrue(diff.is_empty)
        self.assertFalse(bool(diff))
        self.assertEqual(len(diff.unchanged), len(profiles_default.ALL_PROFILES))
        self.assertIn("no profile changes", diff.summary())

    def test_changed_knob_is_reported_as_a_field_delta(self):
        before = _seeded()
        after = _seeded()
        after.register(
            replace(profiles_default.FIXED_CAM_LOCALIZED, stat_thresholds={"block_max": 0.25}),
            overwrite=True,
        )
        diff = diff_registries(before, after)
        self.assertTrue(bool(diff))
        self.assertEqual(len(diff.changed), 1)
        delta = diff.changed[0]
        self.assertEqual(delta.key, profiles_default.FIXED_CAM_LOCALIZED.key)
        self.assertEqual(
            delta.fields,
            (FieldDelta(path="stat_thresholds.block_max", before=0.20, after=0.25),),
        )
        self.assertNotEqual(delta.before_hash, delta.after_hash)
        self.assertIn("stat_thresholds.block_max", diff.summary())

    def test_added_and_removed_profiles(self):
        before = _seeded()
        after = _seeded()
        newcomer = _profile(name="new", environment="drone", goal_type="dynamic", state="transition")
        after.register(newcomer)
        before.unregister(*profiles_default.HANDHELD.key)
        diff = diff_registries(before, after)
        self.assertEqual(sorted(diff.added), sorted([newcomer.key, profiles_default.HANDHELD.key]))
        self.assertEqual(diff.removed, {})
        summary = diff.summary()
        self.assertIn("+ drone/dynamic/transition", summary)

    def test_field_delta_marks_one_sided_paths_absent(self):
        a = _profile(stat_thresholds={})
        b = _profile(stat_thresholds={"block_max": 0.2})
        deltas = diff_profiles(a, b)
        self.assertEqual(
            deltas, (FieldDelta(path="stat_thresholds.block_max", before=ABSENT, after=0.2),)
        )

    def test_diff_profiles_is_empty_for_identical_content(self):
        self.assertEqual(diff_profiles(_profile(), _profile()), ())

    def test_diff_is_reversible(self):
        before = _seeded()
        after = _seeded()
        after.register(
            replace(profiles_default.DYNAMIC_ACTION, plan_mode="single", notes="reverted"),
            overwrite=True,
        )
        after.register(_profile(name="new", environment="drone", goal_type="g", state="s"))
        after.unregister(*profiles_default.HANDHELD.key)

        forward = diff_registries(before, after)
        backward = forward.inverse()

        # inverse swaps the sides ...
        self.assertEqual(set(backward.added), set(forward.removed))
        self.assertEqual(set(backward.removed), set(forward.added))
        # ... and matches the diff you'd get by swapping the arguments outright.
        direct = diff_registries(after, before)
        self.assertEqual(set(backward.added), set(direct.added))
        self.assertEqual(set(backward.removed), set(direct.removed))
        self.assertEqual(
            [(d.key, d.fields) for d in backward.changed],
            [(d.key, d.fields) for d in direct.changed],
        )
        # applying the inverse restores the original registry content, byte for byte.
        restored = load_registry(registry_to_dicts(after))
        for key in backward.added:
            restored.register(profile_from_dict(backward.added[key]), overwrite=True)
        for key in backward.removed:
            restored.unregister(*key)
        for d in backward.changed:
            restored.register(profile_from_dict(d.after), overwrite=True)
        self.assertEqual(registry_hash(restored), registry_hash(before))

    def test_diff_to_dict_is_json_safe(self):
        before = _seeded()
        after = _seeded()
        after.register(
            replace(profiles_default.HANDHELD, plan_mode="strip"), overwrite=True
        )
        after.register(_profile(name="new", environment="drone", goal_type="g", state="s"))
        payload = diff_registries(before, after).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)


def _sides_the_record_reports_absent(record: dict) -> frozenset[str]:
    """Which sides of a serialized :class:`FieldDelta` a consumer can conclude had no value.

    The wire form carries an explicit ``before_absent``/``after_absent`` flag, so this consumer
    reads the flag. It deliberately does NOT pattern-match the value slot against a token: that was
    the whole defect — a field legitimately holding the token was indistinguishable from a field
    that was not there — and ``None`` in the value slot is itself a legitimate profile value
    (``goal_density_guidance.max_recommended_goals``).
    """
    return frozenset(side for side in ("before", "after") if record[f"{side}_absent"])


class SerializationTableCompletenessTests(unittest.TestCase):
    """The serialization tables must cover the dataclass, or a knob leaves the profile's identity.

    ``profile_to_dict`` / ``profile_from_dict`` / ``content_hash`` all drive off two module-level
    tables (``_SCALAR_FIELDS``, ``_NESTED_TYPES``) plus one hand-written ``stat_thresholds``
    branch. A ``WatchProfile`` field added without an entry in one of those three places is
    silently dropped from the serialized body — and therefore from the content hash and from
    every field-level diff — so two materially different profiles would share one address and a
    diff would call them unchanged.

    ``SerializationTests.test_every_watchprofile_field_is_serialized`` does not catch this: it
    compares the OUTPUT dict's keys, which a field emitted by some future ad-hoc branch outside
    the tables would satisfy while still never reaching the diff walk.
    """

    def test_the_three_serialization_tables_cover_every_watchprofile_field(self):
        tabled = set(profile_versioning._SCALAR_FIELDS) | set(profile_versioning._NESTED_TYPES) | {
            "stat_thresholds"
        }
        declared = {f.name for f in dataclasses.fields(WatchProfile)}
        self.assertEqual(
            tabled,
            declared,
            "a WatchProfile field is missing from _SCALAR_FIELDS/_NESTED_TYPES/stat_thresholds; "
            "it will be excluded from profile_to_dict, content_hash and every diff",
        )

    def test_the_three_serialization_tables_are_pairwise_disjoint(self):
        scalars = set(profile_versioning._SCALAR_FIELDS)
        nested = set(profile_versioning._NESTED_TYPES)
        thresholds = {"stat_thresholds"}
        self.assertEqual(scalars & nested, set())
        self.assertEqual(scalars & thresholds, set())
        self.assertEqual(nested & thresholds, set())
        # ... and _SCALAR_FIELDS carries no duplicate, which set arithmetic above would hide.
        self.assertEqual(
            len(profile_versioning._SCALAR_FIELDS), len(set(profile_versioning._SCALAR_FIELDS))
        )

    def test_each_nested_table_entry_names_the_live_annotated_type(self):
        # Not just "some dataclass": the type the table constructs on load must be the type the
        # field is annotated with today, or a round-trip rebuilds a different object.
        hints = get_type_hints(WatchProfile)
        for name, cls in profile_versioning._NESTED_TYPES.items():
            with self.subTest(name):
                self.assertIs(cls, hints[name])
                self.assertTrue(dataclasses.is_dataclass(cls))

    def test_scalar_fields_are_listed_in_declaration_order(self):
        # The table's own docstring claims "canonical declaration order"; pin it so the claim
        # cannot quietly stop being true.
        declared_scalars = tuple(
            f.name
            for f in dataclasses.fields(WatchProfile)
            if f.name not in profile_versioning._NESTED_TYPES and f.name != "stat_thresholds"
        )
        self.assertEqual(profile_versioning._SCALAR_FIELDS, declared_scalars)

    def test_changing_any_single_field_changes_the_content_hash(self):
        """The failure mode the tables exist to prevent, asserted directly and field by field.

        Driven off ``dataclasses.fields`` rather than a literal list, so adding a field to
        ``WatchProfile`` fails here too until someone supplies a distinguishing value for it.
        """
        base = _profile()
        alternates = {
            "name": "other",
            "environment": "wearable",
            "goal_type": "dynamic",
            "state": "transition",
            "salience_stat": "p95",
            "stat_thresholds": {"block_max": 0.21},
            "plan_mode": "single",
            "cadence": replace(base.cadence, base_fps=1.0),
            "gate": replace(base.gate, refractory_s=14.0),
            "goal_density_guidance": replace(base.goal_density_guidance, max_recommended_goals=1),
            "notes": "different",
        }
        self.assertEqual(
            set(alternates),
            {f.name for f in dataclasses.fields(WatchProfile)},
            "a new WatchProfile field has no distinguishing value here — add one and confirm "
            "flipping it moves the content hash",
        )
        for name, value in alternates.items():
            with self.subTest(name):
                variant = _profile(**{name: value})
                self.assertNotEqual(getattr(base, name), getattr(variant, name))
                self.assertNotEqual(content_hash(base), content_hash(variant))


#: The string the sentinel used to BE. Written out rather than imported, because the whole point of
#: the fix is that this text is an ordinary profile value with no special meaning to the differ.
ABSENT_LOOKALIKE = "<absent>"


class AbsentSentinelCollisionTests(unittest.TestCase):
    """``ABSENT`` must not alias any value a profile field can legitimately hold.

    ``diff_profiles`` puts ``ABSENT`` in the value slot to mean "this dotted path exists on only
    one side". The sentinel's own comment reasons about exactly that hazard and rules out ``None``
    because ``None`` is a legitimate value — but ``notes`` (and ``name``, and the free-text
    ``environment`` / ``goal_type`` / ``state``) are unconstrained strings, so the string
    ``"<absent>"`` was a legitimate value on identical grounds. ``diff.summary()`` is the documented
    review surface (docs/sources/source-profiles.md), and it used to render both meanings the same.
    """

    def test_the_sentinel_is_not_a_value_any_profile_field_can_hold(self):
        """The fix, stated once: a private object, not a string, not None, and falsy."""
        self.assertNotIsInstance(ABSENT, (str, bytes, int, float, tuple, dict))
        self.assertIsNot(ABSENT, None)
        self.assertNotEqual(ABSENT, ABSENT_LOOKALIKE)
        self.assertFalse(ABSENT)
        self.assertEqual(repr(ABSENT), "(absent)")
        self.assertIs(profile_versioning.ABSENT, ABSENT)  # one singleton, compared by identity

    def test_a_genuinely_one_sided_path_reports_that_side_absent(self):
        # The sentinel doing its job: stat_thresholds entries really can vanish.
        (delta,) = diff_profiles(
            _profile(stat_thresholds={}), _profile(stat_thresholds={"block_max": 0.2})
        )
        self.assertEqual(delta.path, "stat_thresholds.block_max")
        self.assertEqual(
            _sides_the_record_reports_absent(delta.to_dict()), frozenset({"before"})
        )

    def test_a_field_holding_the_literal_absent_string_is_not_read_as_a_missing_field(self):
        before = _profile(notes="")
        after = _profile(notes=ABSENT_LOOKALIKE)
        (delta,) = diff_profiles(before, after)

        # Ground truth: both sides carry a real value for `notes` ...
        self.assertEqual(before.notes, "")
        self.assertEqual(after.notes, ABSENT_LOOKALIKE)
        # ... and `notes` is an always-emitted scalar, so neither side is one-sided.
        self.assertIn("notes", profile_to_dict(before))
        self.assertIn("notes", profile_to_dict(after))

        # Reversing it — the documented revert path — carries the real text back, and does NOT
        # produce the record a removal would produce.
        self.assertEqual(
            delta.inverse(), FieldDelta(path="notes", before=ABSENT_LOOKALIKE, after="")
        )
        self.assertNotEqual(delta.inverse(), FieldDelta(path="notes", before=ABSENT, after=""))
        # The human rendering now separates the two: a real string stays quoted, the sentinel does
        # not, so `summary()` can be read without guessing.
        self.assertEqual(delta.describe(), "notes: '' -> '<absent>'")
        self.assertEqual(
            FieldDelta(path="stat_thresholds.block_max", before=ABSENT, after=0.2).describe(),
            "stat_thresholds.block_max: (absent) -> 0.2",
        )

        # So: no side of this delta is absent, and the record does not claim one is.
        self.assertEqual(_sides_the_record_reports_absent(delta.to_dict()), frozenset())
        self.assertEqual(delta.to_dict()["after"], ABSENT_LOOKALIKE)

    def test_the_wire_form_reports_one_sidedness_by_flag_and_stays_json_serializable(self):
        """The serialized record must survive ``json.dumps`` — a sentinel object in a value slot
        would not — and must not need ``None`` to mean "missing", since ``None`` is a real value."""
        removal = FieldDelta(path="stat_thresholds.block_max", before=0.2, after=ABSENT)
        record = removal.to_dict()
        self.assertEqual(
            record,
            {
                "path": "stat_thresholds.block_max",
                "before": 0.2,
                "after": None,
                "before_absent": False,
                "after_absent": True,
            },
        )
        self.assertEqual(json.loads(json.dumps(record)), record)

        # A path whose value genuinely IS None is reported with both flags clear, so a consumer
        # reading the flag cannot confuse the two.
        real_none = FieldDelta(
            path="goal_density_guidance.max_recommended_goals", before=3, after=None
        )
        self.assertEqual(
            _sides_the_record_reports_absent(real_none.to_dict()), frozenset()
        )
        self.assertIsNone(real_none.to_dict()["after"])

    def test_inverse_swaps_the_absent_flags_with_the_values(self):
        """``inverse()`` moves the sentinel itself, so the derived flags follow it for free."""
        removal = FieldDelta(path="stat_thresholds.block_max", before=0.2, after=ABSENT)
        addition = removal.inverse()
        self.assertEqual(
            (removal.before_absent, removal.after_absent), (False, True)
        )
        self.assertEqual(
            (addition.before_absent, addition.after_absent), (True, False)
        )
        self.assertEqual(
            _sides_the_record_reports_absent(addition.to_dict()), frozenset({"before"})
        )
        self.assertEqual(addition.inverse(), removal)

    def test_the_profile_format_is_untouched_by_the_fix(self):
        """Blast radius: serialization, hashing and the envelope were always unambiguous.

        Only the delta's *rendering* confused the two meanings, so the fix is a sentinel that does
        not alias a valid value — the profile contract still accepts the literal text and is NOT
        validated to forbid it, which would push a diff-format artifact into that contract.
        """
        plain = _profile(notes="")
        literal = _profile(notes=ABSENT_LOOKALIKE)

        # Round-trip is still exact, both ways.
        self.assertEqual(profile_from_dict(profile_to_dict(literal)), literal)
        self.assertEqual(profile_to_dict(literal)["notes"], ABSENT_LOOKALIKE)
        # Identity still separates them.
        self.assertNotEqual(content_hash(plain), content_hash(literal))
        # And the shippable envelope carries the string through untouched.
        vp = stamp(literal, eval_run_id="run-a", harness_version="0.7.0")
        self.assertEqual(VersionedProfile.from_json(vp.to_json()).profile, literal)

        # At registry level the two profiles are reported as *changed*, with distinct addresses,
        # and the field line inside the summary now reads as the edit it is.
        before_reg, after_reg = WatchProfileRegistry(), WatchProfileRegistry()
        before_reg.register(plain)
        after_reg.register(literal)
        diff = diff_registries(before_reg, after_reg)
        self.assertEqual(len(diff.changed), 1)
        self.assertNotEqual(diff.changed[0].before_hash, diff.changed[0].after_hash)
        self.assertIn("notes: '' -> '<absent>'", diff.summary())
        self.assertNotIn("(absent)", diff.summary())


if __name__ == "__main__":
    unittest.main()
