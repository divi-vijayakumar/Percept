"""Card P4 — profile versioning + provenance: content-addressed, auditable, reversible.

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

import json
import unittest
from dataclasses import replace

from percept import profiles_default
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


if __name__ == "__main__":
    unittest.main()
