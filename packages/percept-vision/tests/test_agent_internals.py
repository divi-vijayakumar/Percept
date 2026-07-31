"""Direct unit coverage for the orchestrator's private helpers, and for ``percept.debug``.

``percept/agent.py`` is exercised indirectly by well over a hundred test files, but the helpers that
enforce its ownership boundaries have had no direct test. Each one guards a specific way a provider
and the runtime can end up sharing a mutable object:

- ``_freeze_provider_value`` / ``_thaw_provider_value`` — a provider mutating a value the runtime
  still holds.
- ``_VerdictSnapshot`` — verdict immutability across the dispatch boundary.
- ``_ObservationSnapshot`` / ``_ObservationCandidate`` — observation identity through the lane.
- ``_readonly_provider_value`` / ``_provider_input_copy`` — provider-owned mutation of caller input.
- ``_normalize_plan_mode`` — an unknown plan mode silently accepted.
- ``_concat_audio`` — mixed audio formats concatenated into nonsense.
- ``_is_valid_verdict`` — the agent-side caller of the ONE verdict-validation authority
  (``percept.contracts.verdict_validation``, ADR-001 §7). It and the judge pool's
  ``_valid_verdict`` were once two implementations of one rule, and they disagreed on exactly the
  terms that matter; these tests pin that they now agree on every shared term and differ ONLY in
  the axis they are each supposed to: exact-type here, duck-typed at the provider seam.

Everything here is a pure function or a frozen dataclass: no fixtures, no I/O, no ``Percept``.
"""

from __future__ import annotations

import dataclasses
import io
import struct
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from types import MappingProxyType

from percept.agent import (
    _concat_audio,
    _FrozenMapping,
    _freeze_provider_value,
    _is_valid_verdict,
    _normalize_plan_mode,
    _ObservationCandidate,
    _ObservationSnapshot,
    _provider_input_copy,
    _readonly_provider_value,
    _thaw_provider_value,
    _VerdictSnapshot,
)
from percept.backends.judge_pool import _valid_verdict as _pool_valid_verdict
from percept.contracts import AudioFrame, Box, ClipEvent, Observation, SubjectRef, Verdict
from percept.debug import collect_trace, format_trace, print_trace
from percept.observability import TraceEvent


def _bound_subject(key: str = "track-7") -> SubjectRef:
    return SubjectRef(
        scheme="track",
        namespace="camera",
        key=key,
        resolution="bound",
        scope="session",
        strategy="track-iou",
    )


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class _Opaque:
    """A provider-owned object the freezer does not know how to make immutable."""

    def __init__(self, payload: list) -> None:
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Opaque) and other.payload == self.payload


class _HashableKey:
    """A dict key that is hashable but still carries mutable state."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.notes: list[str] = []

    def __hash__(self) -> int:
        return hash(self.tag)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HashableKey) and other.tag == self.tag


# ---------------------------------------------------------------------------
# _freeze_provider_value / _thaw_provider_value
# ---------------------------------------------------------------------------


class FreezeProviderValueTests(unittest.TestCase):
    def test_containers_map_onto_their_immutable_twins(self) -> None:
        frozen = _freeze_provider_value(
            {"scores": [1, 2], "tags": {"a"}, "pair": (3, 4), "rigid": frozenset({5})}
        )
        self.assertIsInstance(frozen, _FrozenMapping)
        as_dict = dict(frozen.items)
        self.assertEqual(as_dict["scores"], (1, 2))
        self.assertEqual(as_dict["tags"], frozenset({"a"}))
        self.assertEqual(as_dict["pair"], (3, 4))
        self.assertEqual(as_dict["rigid"], frozenset({5}))
        # A list and a tuple both land on tuple; a set and a frozenset both land on frozenset.
        self.assertIs(type(as_dict["scores"]), tuple)
        self.assertIs(type(as_dict["pair"]), tuple)
        self.assertIs(type(as_dict["tags"]), frozenset)
        self.assertIs(type(as_dict["rigid"]), frozenset)

    def test_nesting_is_frozen_all_the_way_down(self) -> None:
        frozen = _freeze_provider_value({"outer": [{"inner": {"deep": [1]}}]})
        (outer,) = dict(frozen.items)["outer"]
        self.assertIsInstance(outer, _FrozenMapping)
        inner = dict(outer.items)["inner"]
        self.assertIsInstance(inner, _FrozenMapping)
        self.assertEqual(dict(inner.items)["deep"], (1,))

    def test_scalars_pass_through_by_identity(self) -> None:
        for value in (None, True, False, 7, 7.5, "seven", b"seven"):
            with self.subTest(value=value):
                self.assertIs(_freeze_provider_value(value), value)

    def test_a_clip_event_is_rebuilt_rather_than_aliased(self) -> None:
        event = ClipEvent(1.0, 2.0, "item concealed")
        frozen = _freeze_provider_value(event)
        self.assertEqual(frozen, event)
        self.assertIsNot(frozen, event)  # the runtime never holds the provider's own object

    def test_a_subject_ref_passes_through_unchanged(self) -> None:
        subject = _bound_subject()
        self.assertIs(_freeze_provider_value(subject), subject)

    def test_an_unsupported_mutable_value_names_its_type(self) -> None:
        for value, name in ((_Opaque([1]), "_Opaque"), (bytearray(b"ab"), "bytearray")):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    _freeze_provider_value(value)
                self.assertEqual(
                    str(caught.exception),
                    f"provider Verdict contains unsupported mutable value {name}",
                )

    def test_a_subject_ref_subclass_is_refused_rather_than_trusted(self) -> None:
        class _WidenedRef(SubjectRef):
            pass

        widened = _WidenedRef(
            scheme="track",
            namespace="camera",
            key="k",
            resolution="bound",
            scope="session",
            strategy="track-iou",
        )
        with self.assertRaises(ValueError):
            _freeze_provider_value(widened)

    def test_the_frozen_value_does_not_alias_the_providers_dict(self) -> None:
        source = {"scores": [1, 2]}
        frozen = _freeze_provider_value(source)
        source["scores"].append(3)
        source["added_later"] = True
        self.assertEqual(dict(frozen.items), {"scores": (1, 2)})


class ThawProviderValueTests(unittest.TestCase):
    def test_thaw_restores_the_mutable_container_types(self) -> None:
        frozen = _freeze_provider_value({"scores": [1, 2], "tags": {"a"}})
        thawed = _thaw_provider_value(frozen)
        self.assertEqual(thawed, {"scores": [1, 2], "tags": {"a"}})
        self.assertIs(type(thawed), dict)
        self.assertIs(type(thawed["scores"]), list)
        self.assertIs(type(thawed["tags"]), set)

    def test_mutating_a_thawed_copy_leaves_the_frozen_original_untouched(self) -> None:
        frozen = _freeze_provider_value({"scores": [1, 2], "tags": {"a"}})
        first = _thaw_provider_value(frozen)
        first["scores"].append(99)
        first["tags"].add("b")
        first["fresh"] = "key"
        second = _thaw_provider_value(frozen)
        self.assertEqual(second, {"scores": [1, 2], "tags": {"a"}})
        self.assertIsNot(first["scores"], second["scores"])

    def test_a_scalar_or_unknown_leaf_thaws_to_itself(self) -> None:
        event = ClipEvent(0.0, None, "x")
        self.assertIs(_thaw_provider_value(event), event)
        self.assertIs(_thaw_provider_value("plain"), "plain")
        self.assertIsNone(_thaw_provider_value(None))


# ---------------------------------------------------------------------------
# _VerdictSnapshot
# ---------------------------------------------------------------------------


class VerdictSnapshotTests(unittest.TestCase):
    def _rich_verdict(self) -> Verdict:
        return Verdict(
            satisfied=True,
            confidence=0.9,
            evidence="a person entered the zone",
            phase="during",
            action_present=True,
            running_answer="step one",
            usage={"input_tokens": 5, "cost_usd": 0.01},
            events=[ClipEvent(0.5, 1.5, "entered")],
            subject=_bound_subject(),
            operation="assert",
            value={"budget": ["40K"]},
            normalized_value=[1, {"unit": "usd"}],
            target_revision_id="rev-1",
            asserted_by="buyer",
        )

    def test_capture_refuses_a_malformed_verdict(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _VerdictSnapshot.capture(Verdict(satisfied=1, confidence=0.5))
        self.assertEqual(str(caught.exception), "provider returned a malformed Verdict")

    def test_capture_refuses_a_usage_that_is_not_a_mapping(self) -> None:
        verdict = Verdict(satisfied=True, confidence=0.5)
        verdict.usage = [("input_tokens", 5)]  # a provider that emitted pairs, not a mapping
        with self.assertRaises(ValueError) as caught:
            _VerdictSnapshot.capture(verdict)
        self.assertEqual(str(caught.exception), "provider Verdict usage must be a mapping")

    def test_to_verdict_round_trips_to_an_equal_verdict(self) -> None:
        original = self._rich_verdict()
        restored = _VerdictSnapshot.capture(original).to_verdict()
        self.assertEqual(restored, original)
        self.assertIs(type(restored.usage), dict)
        self.assertIs(type(restored.events), list)
        self.assertIs(type(restored.value), dict)
        self.assertEqual(restored.value["budget"], ["40K"])

    def test_the_snapshot_is_detached_from_the_provider_object(self) -> None:
        original = self._rich_verdict()
        snapshot = _VerdictSnapshot.capture(original)
        original.usage["input_tokens"] = 9999
        original.value["budget"].append("mutated")
        original.events.append(ClipEvent(9.0, None, "appended late"))
        restored = snapshot.to_verdict()
        self.assertEqual(restored.usage, {"input_tokens": 5, "cost_usd": 0.01})
        self.assertEqual(restored.value, {"budget": ["40K"]})
        self.assertEqual(len(restored.events), 1)

    def test_two_thaws_of_one_snapshot_share_no_mutable_state(self) -> None:
        snapshot = _VerdictSnapshot.capture(self._rich_verdict())
        first, second = snapshot.to_verdict(), snapshot.to_verdict()
        first.usage["input_tokens"] = 0
        first.value["budget"].append("mutated")
        self.assertEqual(second.usage["input_tokens"], 5)
        self.assertEqual(second.value["budget"], ["40K"])

    def test_the_snapshot_itself_is_frozen(self) -> None:
        snapshot = _VerdictSnapshot.capture(self._rich_verdict())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.satisfied = False  # type: ignore[misc]

    def test_a_verdict_carrying_an_unsupported_value_is_refused_not_aliased(self) -> None:
        verdict = Verdict(satisfied=True, confidence=0.5, value=_Opaque([1]))
        with self.assertRaises(ValueError):
            _VerdictSnapshot.capture(verdict)


# ---------------------------------------------------------------------------
# _ObservationSnapshot / _ObservationCandidate
# ---------------------------------------------------------------------------


class ObservationSnapshotTests(unittest.TestCase):
    def test_capture_round_trips_and_coerces_the_numeric_fields(self) -> None:
        observation = Observation(label="person", box=Box(0, 0, 1, 1), confidence=1, attrs={"c": "red"})
        snapshot = _ObservationSnapshot.capture(observation)
        self.assertIsInstance(snapshot.attrs, _FrozenMapping)
        for coordinate in (snapshot.box.x, snapshot.box.y, snapshot.box.w, snapshot.box.h):
            self.assertIs(type(coordinate), float)
        self.assertIs(type(snapshot.confidence), float)
        restored = snapshot.to_observation()
        self.assertEqual(restored, observation)
        self.assertIs(type(restored.attrs), dict)

    def test_the_snapshot_is_detached_from_the_detectors_attrs(self) -> None:
        attrs = {"colour": "red"}
        snapshot = _ObservationSnapshot.capture(
            Observation(label="person", box=Box(0.0, 0.0, 1.0, 1.0), confidence=0.5, attrs=attrs)
        )
        attrs["colour"] = "blue"
        attrs["added"] = "late"
        self.assertEqual(snapshot.to_observation().attrs, {"colour": "red"})

    def test_an_observation_subclass_is_refused(self) -> None:
        class _WidenedObservation(Observation):
            pass

        with self.assertRaises(ValueError) as caught:
            _ObservationSnapshot.capture(
                _WidenedObservation(label="p", box=Box(0.0, 0.0, 1.0, 1.0), confidence=0.5)
            )
        self.assertEqual(str(caught.exception), "detector returned a malformed Observation")

    def test_a_box_subclass_is_refused(self) -> None:
        class _WidenedBox(Box):
            pass

        with self.assertRaises(ValueError) as caught:
            _ObservationSnapshot.capture(
                Observation(label="p", box=_WidenedBox(0.0, 0.0, 1.0, 1.0), confidence=0.5)
            )
        self.assertEqual(str(caught.exception), "detector returned a malformed Observation")

    def test_attrs_that_are_not_a_mapping_are_refused(self) -> None:
        observation = Observation(label="p", box=Box(0.0, 0.0, 1.0, 1.0), confidence=0.5)
        observation.attrs = [("colour", "red")]  # type: ignore[assignment]
        with self.assertRaises(ValueError) as caught:
            _ObservationSnapshot.capture(observation)
        self.assertEqual(str(caught.exception), "Observation attrs must be a mapping")


class ObservationCandidateTests(unittest.TestCase):
    def _candidate(self, **overrides) -> _ObservationCandidate:
        fields = dict(
            token=None,
            dispatch=None,
            frame=None,
            goals=(),
            source_t=0.0,
            ingest_observed_ts=0.0,
        )
        fields.update(overrides)
        return _ObservationCandidate(**fields)

    def test_skip_detect_defaults_to_false_because_an_unset_flag_is_the_unguarded_path(self) -> None:
        self.assertIs(self._candidate().skip_detect, False)
        self.assertIs(self._candidate(skip_detect=True).skip_detect, True)

    def test_the_candidate_is_frozen(self) -> None:
        self.assertTrue(_ObservationCandidate.__dataclass_params__.frozen)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self._candidate().skip_detect = True  # type: ignore[misc]

    def test_skip_detect_is_the_last_field_so_positional_construction_is_unchanged(self) -> None:
        names = [f.name for f in dataclasses.fields(_ObservationCandidate)]
        self.assertEqual(names[-1], "skip_detect")


# ---------------------------------------------------------------------------
# _readonly_provider_value / _provider_input_copy
# ---------------------------------------------------------------------------


class ReadonlyProviderValueTests(unittest.TestCase):
    def test_a_mapping_becomes_a_proxy_that_refuses_assignment(self) -> None:
        readonly = _readonly_provider_value({"type": "watch", "count": 0})
        self.assertIsInstance(readonly, MappingProxyType)
        with self.assertRaises(TypeError):
            readonly["count"] = 1
        self.assertEqual(dict(readonly), {"type": "watch", "count": 0})

    def test_nested_containers_are_detached_and_read_only(self) -> None:
        readonly = _readonly_provider_value(
            {"nested": {"seen": [1, 2]}, "tags": {"a"}, "pair": (1, 2)}
        )
        self.assertIsInstance(readonly["nested"], MappingProxyType)
        self.assertIs(type(readonly["nested"]["seen"]), tuple)
        self.assertIs(type(readonly["tags"]), frozenset)
        self.assertIs(type(readonly["pair"]), tuple)
        with self.assertRaises(TypeError):
            readonly["nested"]["seen"] = ()

    def test_the_proxy_does_not_alias_the_callers_dict(self) -> None:
        source = {"count": 0, "seen": [1]}
        readonly = _readonly_provider_value(source)
        source["count"] = 99
        source["seen"].append(2)
        self.assertEqual(readonly["count"], 0)
        self.assertEqual(readonly["seen"], (1,))

    def test_keys_are_deep_copied_not_shared(self) -> None:
        key = _HashableKey("subject")
        readonly = _readonly_provider_value({key: "value"})
        (copied_key,) = list(readonly)
        self.assertEqual(copied_key, key)
        self.assertIsNot(copied_key, key)
        copied_key.notes.append("provider scribbled here")
        self.assertEqual(key.notes, [])

    def test_an_unknown_object_is_deep_copied_rather_than_shared(self) -> None:
        opaque = _Opaque([1, 2])
        detached = _readonly_provider_value(opaque)
        self.assertEqual(detached, opaque)
        self.assertIsNot(detached, opaque)
        detached.payload.append(3)
        self.assertEqual(opaque.payload, [1, 2])

    def test_scalars_pass_through_by_identity(self) -> None:
        for value in (None, True, 7, 7.5, "seven", b"seven"):
            with self.subTest(value=value):
                self.assertIs(_readonly_provider_value(value), value)


class ProviderInputCopyTests(unittest.TestCase):
    def test_a_read_only_proxy_becomes_a_fresh_mutable_dict(self) -> None:
        readonly = _readonly_provider_value({"nested": {"count": 0}})
        writable = _provider_input_copy(readonly)
        self.assertIs(type(writable), dict)
        self.assertIs(type(writable["nested"]), dict)
        writable["nested"]["count"] = 99
        writable["added_by_provider"] = True
        self.assertEqual(readonly["nested"]["count"], 0)
        self.assertNotIn("added_by_provider", readonly)

    def test_a_frozenset_becomes_a_mutable_set(self) -> None:
        writable = _provider_input_copy(frozenset({1, 2}))
        self.assertIs(type(writable), set)
        writable.add(3)
        self.assertEqual(writable, {1, 2, 3})

    def test_provider_mutation_cannot_reach_the_callers_frozen_request(self) -> None:
        caller_state = {"type": "watch", "count": 0, "seen": [1], "tags": {"a"}}
        frozen_request = _readonly_provider_value(caller_state)
        provider_view = _provider_input_copy(frozen_request)
        provider_view["count"] = 41
        provider_view["tags"].add("b")
        self.assertEqual(caller_state["count"], 0)
        self.assertEqual(caller_state["tags"], {"a"})
        self.assertEqual(frozen_request["count"], 0)
        self.assertEqual(frozen_request["tags"], frozenset({"a"}))

    def test_two_provider_calls_get_independent_copies(self) -> None:
        frozen_request = _readonly_provider_value({"seen": {"deep": 1}})
        first = _provider_input_copy(frozen_request)
        second = _provider_input_copy(frozen_request)
        first["seen"]["deep"] = 2
        self.assertEqual(second["seen"]["deep"], 1)

    def test_a_plain_dict_is_copied_not_aliased(self) -> None:
        source = {"count": 0}
        copied = _provider_input_copy(source)
        copied["count"] = 5
        self.assertEqual(source["count"], 0)

    def test_keys_are_deep_copied_not_shared(self) -> None:
        key = _HashableKey("subject")
        copied = _provider_input_copy({key: "value"})
        (copied_key,) = list(copied)
        self.assertIsNot(copied_key, key)
        copied_key.notes.append("provider scribbled here")
        self.assertEqual(key.notes, [])

    def test_an_unknown_object_is_deep_copied(self) -> None:
        opaque = _Opaque([1])
        copied = _provider_input_copy(opaque)
        self.assertIsNot(copied, opaque)
        copied.payload.append(2)
        self.assertEqual(opaque.payload, [1])

    def test_scalars_pass_through_by_identity(self) -> None:
        for value in (None, True, 7, 7.5, "seven", b"seven"):
            with self.subTest(value=value):
                self.assertIs(_provider_input_copy(value), value)


# ---------------------------------------------------------------------------
# _normalize_plan_mode
# ---------------------------------------------------------------------------


class NormalizePlanModeTests(unittest.TestCase):
    def test_a_live_mode_is_returned_unchanged_and_silently(self) -> None:
        for mode in ("single", "strip", "window", "event_capsule"):
            with self.subTest(mode=mode):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertEqual(_normalize_plan_mode(mode), mode)
                self.assertEqual([str(w.message) for w in caught], [])

    def test_an_unrecognized_mode_is_returned_unchanged_and_silently(self) -> None:
        # _normalize_plan_mode only maps DEPRECATED modes; refusing an invented mode is
        # _planned_perception's job, so this helper must neither rewrite nor warn about it.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(_normalize_plan_mode("montage"), "montage")
            self.assertEqual(_normalize_plan_mode(""), "")
        self.assertEqual([str(w.message) for w in caught], [])

    def test_the_deprecated_montage_mode_maps_to_window_with_exactly_one_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(_normalize_plan_mode("sprite"), "window")
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        message = str(caught[0].message)
        self.assertIn("'sprite'", message)
        self.assertIn("'window'", message)

    def test_each_call_warns_so_no_call_site_is_silently_deprecated(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _normalize_plan_mode("sprite")
            _normalize_plan_mode("sprite")
        self.assertEqual(len(caught), 2)


# ---------------------------------------------------------------------------
# _is_valid_verdict — the agent-side twin of the judge pool's _valid_verdict
# ---------------------------------------------------------------------------


class IsValidVerdictTests(unittest.TestCase):
    def test_a_well_formed_verdict_is_valid(self) -> None:
        self.assertTrue(_is_valid_verdict(Verdict(satisfied=True, confidence=0.9, evidence="e")))
        self.assertTrue(_is_valid_verdict(Verdict(satisfied=False, confidence=0)))

    def test_a_verdict_shaped_duck_type_is_not_a_verdict(self) -> None:
        class _LooksLikeAVerdict:
            satisfied = True
            confidence = 0.9
            evidence = "e"

        self.assertFalse(_is_valid_verdict(_LooksLikeAVerdict()))
        # The pool's twin is duck-typed and accepts it. That asymmetry is fail-closed — the
        # stricter side is the one guarding the gate — so it is recorded, not asserted away.
        self.assertTrue(_pool_valid_verdict(_LooksLikeAVerdict()))

    def test_the_partial_injection_shape_is_rejected(self) -> None:
        # FakeJudge's "partial" error mode: Verdict-shaped, missing required fields.
        self.assertFalse(_is_valid_verdict({"satisfied": True}))
        self.assertFalse(_is_valid_verdict(None))
        self.assertFalse(_is_valid_verdict({"unexpected_shape": True, "raw": "garbled"}))

    def test_a_truthy_int_satisfied_is_rejected(self) -> None:
        self.assertFalse(_is_valid_verdict(Verdict(satisfied=1, confidence=0.9)))
        self.assertFalse(_is_valid_verdict(Verdict(satisfied="yes", confidence=0.9)))

    def test_a_string_confidence_is_rejected(self) -> None:
        self.assertFalse(_is_valid_verdict(Verdict(satisfied=True, confidence="0.9")))
        self.assertFalse(_is_valid_verdict(Verdict(satisfied=True, confidence=None)))

    def test_an_integer_confidence_is_accepted(self) -> None:
        self.assertTrue(_is_valid_verdict(Verdict(satisfied=True, confidence=1)))

    def test_a_confidence_outside_zero_to_one_is_rejected_like_the_pool_twin(self) -> None:
        # BUG-F-01-01's pin, and the reason the two twins were collapsed onto one authority: the
        # agent side used to accept every one of these, and `percept.gate.classify` reads nan/inf/
        # 7.5/True as a confident "yes" (`nan < threshold` is False). No subTest, kept from when
        # this was xfail(strict) — a subTest failure inside an xfail is reported against the
        # subtest, leaving the enclosing test an XPASS.
        candidates = [
            Verdict(satisfied=True, confidence=confidence, evidence="e")
            for confidence in (float("nan"), float("inf"), 7.5, -0.5, True)
        ]
        self.assertEqual(
            [_pool_valid_verdict(v) for v in candidates],
            [False] * len(candidates),  # the twin already refuses every one of them
        )
        self.assertEqual(
            [_is_valid_verdict(v) for v in candidates],
            [False] * len(candidates),
        )

    def test_the_two_implementations_of_the_rule_still_have_one_shared_core(self) -> None:
        # Whatever else differs, both must reject a non-bool `satisfied` and a non-numeric
        # `confidence`: that is the part the gate dereferences without a guard of its own.
        for verdict in (
            Verdict(satisfied=1, confidence=0.9, evidence="e"),
            Verdict(satisfied=True, confidence="0.9", evidence="e"),
        ):
            with self.subTest(verdict=verdict):
                self.assertFalse(_is_valid_verdict(verdict))
                self.assertFalse(_pool_valid_verdict(verdict))


# ---------------------------------------------------------------------------
# _concat_audio
# ---------------------------------------------------------------------------


class ConcatAudioTests(unittest.TestCase):
    def test_no_chunks_is_no_clip(self) -> None:
        self.assertIsNone(_concat_audio([]))

    def test_typed_frames_join_into_one_frame_carrying_the_heads_format(self) -> None:
        head = AudioFrame(
            pcm=_pcm([1, 2]), sample_rate=8000, channels=1, sample_format="s16le",
            t_start=0.0, t_end=0.25,
        )
        tail = AudioFrame(
            pcm=_pcm([3, 4, 5]), sample_rate=8000, channels=1, sample_format="s16le",
            t_start=0.25, t_end=0.625,
        )
        clip = _concat_audio([head, tail])
        self.assertIsInstance(clip, AudioFrame)
        self.assertEqual(clip.pcm, head.pcm + tail.pcm)
        self.assertEqual(clip.sample_rate, 8000)
        self.assertEqual(clip.channels, 1)
        self.assertEqual(clip.sample_format, "s16le")
        self.assertEqual(clip.t_start, 0.0)
        self.assertEqual(clip.t_end, 0.625)

    def test_the_carried_format_is_read_off_the_frames_not_hardcoded(self) -> None:
        head = AudioFrame(pcm=b"\x01\x02", sample_rate=44_100, channels=2, sample_format="u8")
        tail = AudioFrame(pcm=b"\x03\x04", sample_rate=44_100, channels=2, sample_format="u8")
        clip = _concat_audio([head, tail])
        self.assertEqual(
            (clip.sample_rate, clip.channels, clip.sample_format), (44_100, 2, "u8")
        )
        self.assertEqual(clip.pcm, b"\x01\x02\x03\x04")

    def test_a_non_frame_in_a_typed_list_is_never_flattened(self) -> None:
        frame = AudioFrame(pcm=_pcm([1]), sample_rate=8000, channels=1)
        self.assertIsNone(_concat_audio([frame, b"raw bytes"]))
        self.assertIsNone(_concat_audio([frame, ("audio/wav", b"blob")]))

    def test_mixed_rate_channel_count_or_sample_format_returns_none(self) -> None:
        base = AudioFrame(pcm=_pcm([1, 2]), sample_rate=8000, channels=1, sample_format="s16le")
        others = (
            AudioFrame(pcm=_pcm([3, 4]), sample_rate=16_000, channels=1, sample_format="s16le"),
            AudioFrame(pcm=_pcm([3, 4]), sample_rate=8000, channels=2, sample_format="s16le"),
            AudioFrame(pcm=b"\x00\x00\x00\x00", sample_rate=8000, channels=1, sample_format="f32le"),
        )
        for other in others:
            with self.subTest(other=(other.sample_rate, other.channels, other.sample_format)):
                self.assertIsNone(_concat_audio([base, other]))

    def test_an_encoded_and_raw_sequence_is_never_flattened(self) -> None:
        raw = AudioFrame(pcm=_pcm([1, 2]), sample_rate=8000, channels=1)
        encoded = AudioFrame(pcm=b"", encoded=b"RIFF....", mime="audio/wav")
        self.assertIsNone(_concat_audio([raw, encoded]))
        self.assertIsNone(_concat_audio([encoded, raw]))

    def test_encoded_only_frames_pass_the_latest_through(self) -> None:
        first = AudioFrame(pcm=b"", encoded=b"RIFF-one", mime="audio/wav")
        second = AudioFrame(pcm=b"", encoded=b"RIFF-two", mime="audio/wav")
        self.assertIs(_concat_audio([first, second]), second)
        self.assertIs(_concat_audio([first]), first)

    def test_pcm_that_is_not_a_whole_number_of_interleaved_frames_returns_none(self) -> None:
        odd_mono = AudioFrame(pcm=b"\x00\x00\x00", sample_rate=8000, channels=1, sample_format="s16le")
        self.assertIsNone(_concat_audio([odd_mono]))
        half_stereo = AudioFrame(pcm=_pcm([1]), sample_rate=8000, channels=2, sample_format="s16le")
        self.assertIsNone(_concat_audio([half_stereo, half_stereo]))
        short_float = AudioFrame(pcm=b"\x00\x00", sample_rate=8000, channels=1, sample_format="f32le")
        self.assertIsNone(_concat_audio([short_float]))

    def test_mime_pairs_concatenate_only_when_every_mime_matches(self) -> None:
        self.assertEqual(
            _concat_audio([("audio/wav", b"one"), ("audio/wav", bytearray(b"two"))]),
            ("audio/wav", b"onetwo"),
        )
        self.assertIsNone(_concat_audio([("audio/wav", b"one"), ("audio/ogg", b"two")]))
        self.assertIsNone(_concat_audio([("audio/wav", b"one"), b"two"]))

    def test_raw_byte_buffers_concatenate(self) -> None:
        self.assertEqual(_concat_audio([b"one", bytearray(b"two")]), b"onetwo")
        self.assertIs(type(_concat_audio([bytearray(b"one")])), bytes)
        self.assertIsNone(_concat_audio([b"one", ("audio/wav", b"two")]))

    def test_an_opaque_descriptor_passes_the_latest_through(self) -> None:
        first, second = object(), object()
        self.assertIs(_concat_audio([first, second]), second)


# ---------------------------------------------------------------------------
# percept.debug
# ---------------------------------------------------------------------------


class DebugTraceRenderingTests(unittest.TestCase):
    def test_format_trace_renders_the_trace_goal_stage_and_data(self) -> None:
        event = TraceEvent(
            trace_id="camera@12.0",
            stage="judge.finished",
            t=12.0,
            goal_id="caffeine",
            data={"satisfied": True, "confidence": 0.92},
        )
        self.assertEqual(
            format_trace(event),
            "[trace=camera@12.0 goal=caffeine] judge.finished satisfied=True confidence=0.92",
        )

    def test_a_falsy_goal_id_drops_the_goal_segment(self) -> None:
        for goal_id in (None, ""):
            with self.subTest(goal_id=goal_id):
                event = TraceEvent(
                    trace_id="camera@1.0",
                    stage="source.frame",
                    goal_id=goal_id,
                    data={"w": 48},
                )
                self.assertEqual(format_trace(event), "[trace=camera@1.0] source.frame w=48")

    def test_empty_data_leaves_no_trailing_body_and_no_trailing_space(self) -> None:
        event = TraceEvent(trace_id="camera@1.0", stage="edge.decision", goal_id="zone")
        rendered = format_trace(event)
        self.assertEqual(rendered, "[trace=camera@1.0 goal=zone] edge.decision")
        self.assertEqual(rendered, rendered.rstrip())

    def test_print_trace_writes_to_the_file_it_is_given(self) -> None:
        sink = io.StringIO()
        event = TraceEvent(trace_id="camera@1.0", stage="gate.classified", data={"action": "fire"})
        print_trace(event, file=sink)
        self.assertEqual(sink.getvalue(), "[trace=camera@1.0] gate.classified action=fire\n")

    def test_print_trace_defaults_to_stderr_so_it_never_pollutes_stdout(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        event = TraceEvent(trace_id="camera@1.0", stage="gate.classified")
        with redirect_stdout(out), redirect_stderr(err):
            print_trace(event)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "[trace=camera@1.0] gate.classified\n")

    def test_collect_trace_appends_every_event_in_order(self) -> None:
        collected: list = []
        callback = collect_trace(collected)
        events = [
            TraceEvent(trace_id="camera@1.0", stage="source.frame"),
            TraceEvent(trace_id="camera@1.0", stage="judge.started"),
            TraceEvent(trace_id="camera@1.0", stage="gate.classified"),
        ]
        for event in events:
            callback(event)
        self.assertEqual([e.stage for e in collected], [e.stage for e in events])
        self.assertIs(collected[0], events[0])

    def test_collect_trace_appends_into_the_caller_list_rather_than_a_copy(self) -> None:
        collected: list = ["pre-existing"]
        collect_trace(collected)(TraceEvent(trace_id="camera@1.0", stage="source.frame"))
        self.assertEqual(len(collected), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
