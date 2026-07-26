from __future__ import annotations

import struct

import pytest

from percept.contracts import AudioFrame
from percept.events import AudioUnit
from percept.fused_sync import (
    _AcceptedMediaUnit,
    _FusedCoverageUncertainty,
    _FusedDiscontinuity,
    _FusedSyncPolicy,
    _FusedSynchronizer,
)


def _audio(seq: int, start: float, end: float, *, watermark=None, generation=0):
    rate = 10
    samples = round((end - start) * rate)
    frame = AudioFrame(
        pcm=struct.pack("<" + "h" * samples, *range(samples)),
        sample_rate=rate,
        channels=1,
        sample_format="s16le",
        t_start=start,
        t_end=end,
    )
    unit = AudioUnit(
        seq=seq,
        t_start=start,
        t_end=end,
        chunk=frame,
        event_t=end,
        sample_rate=rate,
        channels=1,
        sample_format="s16le",
        source_id="audio",
        source_epoch=0,
        source_sequence=seq,
        observed_time=end + 10,
        source_time=end,
    )
    return _AcceptedMediaUnit(
        modality="audio",
        payload=unit,
        source_id="audio",
        source_epoch=0,
        source_sequence=seq,
        generation=generation,
        watch_application="app",
        watch_revision=1,
        event_start=start,
        event_end=end,
        observed_start=end + 10,
        observed_end=end + 10,
        clock_domain="source",
        clock_authority="media:fixture",
        clock_confidence=1.0,
        provenance={"format": {"sample_rate": rate}},
        source_watermark=watermark,
    )


def _video(seq: int, at: float, *, watermark=None, generation=0):
    return _AcceptedMediaUnit(
        modality="video",
        payload=("image/jpeg", b"\xff\xd8x\xff\xd9"),
        source_id="video",
        source_epoch=0,
        source_sequence=seq,
        generation=generation,
        watch_application="app",
        watch_revision=1,
        event_start=at,
        event_end=at,
        observed_start=at + 10,
        observed_end=at + 10,
        clock_domain="source",
        clock_authority="media:fixture",
        clock_confidence=1.0,
        provenance={"format": {"media_type": "image/jpeg"}},
        source_watermark=watermark,
    )


def test_pending_window_closes_once_at_watermark_and_eos_is_idempotent():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=1.0, allowed_lateness_s=0.2, max_skew_s=0.1),
        watch_application="app",
        watch_revision=1,
    )
    anchor = _video(0, 1.0, watermark=1.0)
    assert sync.accept(anchor)
    assert sync.accept(_audio(0, 0.0, 1.0, watermark=1.0))
    first = sync.bundle_for(anchor)
    assert _FusedCoverageUncertainty.WATERMARK_PENDING in first.coverage_reasons
    assert sync.accept(_audio(1, 1.0, 1.2, watermark=1.2))
    ready = sync.drain_ready()
    assert len(ready) == 1
    assert ready[0].complete
    assert sync.drain_ready() == ()
    assert sync.close() == ()
    assert sync.close() == ()


def test_forged_anchor_and_second_source_are_rejected_without_borrowing():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=1.0),
        watch_application="app",
        watch_revision=1,
    )
    anchor = _video(0, 1.0, watermark=2.0)
    assert sync.accept(anchor)
    forged = _video(0, 1.0, watermark=2.0)
    with pytest.raises(ValueError, match="not accepted"):
        sync.bundle_for(forged)
    other = _video(1, 2.0, watermark=2.0)
    object.__setattr__(other, "source_id", "other")
    with pytest.raises(ValueError, match="one video source"):
        sync.accept(other)


def test_generation_and_clock_authority_isolate_evidence():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=1.0, max_skew_s=0.0),
        watch_application="app",
        watch_revision=1,
    )
    old = _video(0, 1.0, watermark=1.0)
    assert sync.accept(old)
    generation = sync.advance_generation(_FusedDiscontinuity.RECONNECT)
    assert not sync.accept(old)
    current = _video(0, 2.0, watermark=2.0, generation=generation)
    assert sync.accept(current)
    mismatched = _audio(0, 1.0, 2.0, watermark=2.0, generation=generation)
    object.__setattr__(mismatched, "clock_authority", "media:other")
    assert sync.accept(mismatched)
    bundle = sync.bundle_for(current)
    assert _FusedCoverageUncertainty.CLOCK_DOMAIN_MISMATCH in bundle.coverage_reasons


def test_startup_late_audio_does_not_infer_a_shorter_coverage_window():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=1.0, max_skew_s=0.0),
        watch_application="app",
        watch_revision=1,
    )
    anchor = _video(0, 1.0, watermark=2.0)
    assert sync.accept(anchor)
    assert sync.accept(_audio(0, 0.9, 1.0, watermark=2.0))
    bundle = sync.bundle_for(anchor)
    assert not bundle.complete
    assert _FusedCoverageUncertainty.MISSING_AUDIO in bundle.coverage_reasons


def test_trusted_post_gap_boundary_allows_complete_recovery_without_old_media():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=1.0, max_skew_s=0.0),
        watch_application="app",
        watch_revision=1,
    )
    assert sync.accept(_video(0, 1.0, watermark=1.0))
    generation = sync.advance_generation(_FusedDiscontinuity.GAP)
    sync.set_generation_start(2.0)
    anchor = _video(0, 2.5, watermark=3.0, generation=generation)
    assert sync.accept(anchor)
    assert sync.accept(
        _audio(0, 2.0, 2.5, watermark=3.0, generation=generation)
    )
    bundle = sync.bundle_for(anchor)
    assert bundle.complete
    assert bundle.audio_event_interval == (2.0, 2.5)


def test_provenance_is_deep_frozen_and_rechunking_is_interval_exact():
    provenance = {"format": {"rate": 10}, "path": ["capture", "pcm"]}
    accepted = _audio(0, 0.0, 2.0, watermark=3.0)
    object.__setattr__(accepted, "provenance", provenance)
    # Construct again through the acceptance firewall; later caller mutation cannot alter it.
    accepted = _AcceptedMediaUnit(**{
        **accepted.__dict__,
        "provenance": provenance,
    })
    provenance["format"]["rate"] = 99
    provenance["path"].append("mutated")
    assert accepted.provenance["format"]["rate"] == 10
    assert accepted.provenance["path"] == ("capture", "pcm")

    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=0.5, max_skew_s=0.0),
        watch_application="app",
        watch_revision=1,
    )
    anchor = _video(0, 1.0, watermark=1.0)
    assert sync.accept(anchor)
    assert sync.accept(accepted)
    bundle = sync.bundle_for(anchor)
    sliced = bundle.audio_units[0].payload
    assert (sliced.t_start, sliced.t_end) == (0.5, 1.0)
    assert len(sliced.chunk.pcm) == 10


def test_stalled_audio_anchor_retention_is_bounded_without_raising():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(
            window_s=1.0,
            max_units_per_modality=3,
            max_bytes_per_modality=1024,
        ),
        watch_application="app",
        watch_revision=1,
    )
    for seq in range(20):
        anchor = _video(seq, float(seq), watermark=float(seq))
        assert sync.accept(anchor)
        sync.bundle_for(anchor)
    closed = sync.close()
    assert len(closed) <= 3
    assert all(
        _FusedCoverageUncertainty.BOUNDS_EXCEEDED in bundle.coverage_reasons
        or _FusedCoverageUncertainty.EOS_INCOMPLETE in bundle.coverage_reasons
        for bundle in closed
    )


def test_make_incident_preserves_runtime_local_fused_bundle():
    from percept.agent import FireEvent
    from percept.incidents import make_incident

    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=0.5, max_skew_s=0.0),
        watch_application="app",
        watch_revision=1,
    )
    anchor = _video(0, 1.0, watermark=2.0)
    assert sync.accept(anchor)
    assert sync.accept(_audio(0, 0.5, 1.0, watermark=2.0))
    bundle = sync.bundle_for(
        anchor,
        evidence_seq=0,
        judgment_event_id="judgment:test",
    )
    fire = FireEvent("g", "fire", "done")
    fire._fused_evidence_bundle = bundle
    incident = make_incident(fire, ts=1.0)
    assert incident._fused_evidence_bundle is bundle


def test_public_async_fused_watch_uses_split_provider_path():
    import asyncio

    from percept import Percept
    from percept.adapters import Verdict
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.harness.tier0 import gray_frame
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import (
        STILL,
        W,
        H,
        _collect,
        _profile,
    )

    core = Percept.create(
        vision=FakeVision(
            script={"changed|audio": Verdict(satisfied=True, confidence=0.99)}
        ),
        discover_plugins=False,
    )
    receipt = apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "async-av",
            threshold=0.1,
            modalities=frozenset({"video", "audio"}),
        ),
    )
    assert receipt.status == "applied"
    admitted_jobs = []
    original_prepare = core.prepare_fused_judgment

    def capture_prepared_job(*args, **kwargs):
        job = original_prepare(*args, **kwargs)
        if job is not None:
            admitted_jobs.append(job)
            assert job.bundle.evidence_seq == job.token.evidence_seq
            assert job.bundle.judgment_event_id is not None
            assert job.frames == tuple(
                unit.payload for unit in job.bundle.video_units[:-1]
            )
            assert job.frame is job.bundle.video_units[-1].payload
        return job

    core.prepare_fused_judgment = capture_prepared_job
    incidents = asyncio.run(
        _collect(
            core,
            [(0.0, STILL, "prime"), (1.0, gray_frame(W, H, 255), "changed")],
            audio_source=[
                (
                    0.5,
                    AudioFrame(pcm=b"\x01\x00" * 8_000, sample_rate=16_000),
                    0.5,
                )
            ],
            audio_gate=None,
            judge_dispatch="async",
        )
    )
    assert len(incidents) == 1
    bundle = getattr(incidents[0], "_fused_evidence_bundle")
    assert bundle.judgment_event_id == incidents[0].judgment_event_id
    assert bundle.evidence_seq == incidents[0].evidence_seq
    matched = [
        job
        for job in admitted_jobs
        if job.bundle.judgment_event_id == incidents[0].judgment_event_id
    ]
    assert len(matched) == 1
    assert matched[0].token.evidence_seq == incidents[0].evidence_seq


def test_fused_provider_exception_payload_is_redacted_from_runtime_outputs():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.harness.tier0 import gray_frame
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import (
        STILL,
        W,
        H,
        _collect,
        _profile,
    )

    secret = "SECRET-SENTINEL-IN-PROVIDER-URL"

    traces = []
    health = []
    core = Percept.create(
        vision=FakeVision(),
        discover_plugins=False,
        on_trace=traces.append,
    )
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "redacted-av",
            threshold=0.1,
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"

    async def leaking_dispatch(*args, **kwargs):
        raise RuntimeError(f"https://user:{secret}@provider.invalid/request")

    original_prepare = core.prepare_fused_judgment
    semantic_calls = []
    original_document_fold = core._document_episode_fold
    original_claim_fold = core._claim_episode_fold
    original_gate_decide = core.gate.decide

    def document_fold_spy(*args, **kwargs):
        semantic_calls.append("document")
        return original_document_fold(*args, **kwargs)

    def claim_fold_spy(*args, **kwargs):
        semantic_calls.append("claim")
        return original_claim_fold(*args, **kwargs)

    def gate_decide_spy(*args, **kwargs):
        semantic_calls.append("gate")
        return original_gate_decide(*args, **kwargs)

    core._document_episode_fold = document_fold_spy
    core._claim_episode_fold = claim_fold_spy
    core.gate.decide = gate_decide_spy

    def prepare_with_failing_provider(*args, **kwargs):
        job = original_prepare(*args, **kwargs)
        return None if job is None else replace(job, dispatch=leaking_dispatch)

    core.prepare_fused_judgment = prepare_with_failing_provider
    incidents = asyncio.run(
        _collect(
            core,
            [(0.0, STILL, "prime"), (1.0, gray_frame(W, H, 255), "changed")],
            audio_source=[
                (
                    0.5,
                    AudioFrame(pcm=b"\x01\x00" * 8_000, sample_rate=16_000),
                    0.5,
                )
            ],
            audio_gate=None,
            on_health=health.append,
            judge_dispatch="async",
        )
    )
    assert any(event.kind == "judge_degraded" for event in health)
    assert semantic_calls == []
    exposed = repr(
        {
            "traces": traces,
            "health": health,
            "actions": core.action_log.entries(),
            "incidents": incidents,
        }
    )
    assert secret not in exposed


def test_provider_owned_verdict_mutation_cannot_change_fused_completion():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.adapters import Verdict
    from percept.backends.fake import FakeVision
    from percept.contracts import ClipEvent
    from percept.contracts.watchplan import WatchPlan
    from percept.harness.tier0 import gray_frame
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import (
        STILL,
        W,
        H,
        _collect,
        _profile,
    )

    traces = []
    provider_owned = Verdict(
        satisfied=True,
        confidence=0.99,
        evidence="snapshot-original",
        usage={"input_tokens": 1},
        events=[ClipEvent(0.1, label="original")],
        value={"nested": ["original"]},
    )
    core = Percept.create(
        vision=FakeVision(),
        discover_plugins=False,
        on_trace=traces.append,
    )
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "snapshot-av",
            threshold=0.1,
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"

    async def retained_provider(*args, **kwargs):
        loop = asyncio.get_running_loop()

        def mutate_after_return():
            provider_owned.satisfied = False
            provider_owned.confidence = 0.0
            provider_owned.evidence = "mutated"
            provider_owned.usage["input_tokens"] = 999
            provider_owned.events.append(ClipEvent(0.2, label="mutated"))
            provider_owned.value["nested"].append("mutated")

        loop.call_soon(mutate_after_return)
        return provider_owned

    original_prepare = core.prepare_fused_judgment

    def prepare_with_retained_provider(*args, **kwargs):
        job = original_prepare(*args, **kwargs)
        return None if job is None else replace(job, dispatch=retained_provider)

    core.prepare_fused_judgment = prepare_with_retained_provider
    incidents = asyncio.run(
        _collect(
            core,
            [(0.0, STILL, "prime"), (1.0, gray_frame(W, H, 255), "changed")],
            audio_source=[
                (
                    0.5,
                    AudioFrame(pcm=b"\x01\x00" * 8_000, sample_rate=16_000),
                    0.5,
                )
            ],
            audio_gate=None,
            judge_dispatch="async",
        )
    )
    assert len(incidents) == 1
    finished = [event for event in traces if event.stage == "judge.finished"]
    assert len(finished) == 1
    assert finished[0].data["satisfied"] is True
    assert finished[0].data["confidence"] == 0.99
    usage = [event for event in traces if event.stage == "judge.usage"]
    assert len(usage) == 1
    assert usage[0].data["input_tokens"] == 1


def test_fused_bundle_identity_mismatch_emits_typed_completed_drop():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "identity-av",
            threshold=0.1,
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"
    binding = core._audio_judge_bindings["fused"]
    revision = core.goals.revision
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(window_s=0.5, max_skew_s=0.0),
        watch_application=binding.application_key,
        watch_revision=revision,
    )
    anchor = replace(
        _video(0, 1.0, watermark=2.0),
        watch_application=binding.application_key,
        watch_revision=revision,
    )
    audio = replace(
        _audio(0, 0.5, 1.0, watermark=2.0),
        watch_application=binding.application_key,
        watch_revision=revision,
    )
    assert sync.accept(anchor)
    assert sync.accept(audio)
    job = core.prepare_fused_judgment(
        sync.bundle_for(anchor),
        goal_id="fused",
        source="camera",
        source_t=1.0,
        goals_revision=revision,
        source_generation=core.source_continuity_generation("camera"),
        fused_generation=sync.generation,
    )
    assert job is not None
    completion = asyncio.run(core.dispatch_fused_judgment(job))
    bad_bundle = replace(job.bundle, evidence_seq=job.bundle.evidence_seq + 1)
    bad_completion = replace(completion, job=replace(job, bundle=bad_bundle))
    assert core.reduce_fused_completion(
        bad_completion,
        live_fused_generation=sync.generation,
    ) == []
    records = core.drain_fused_trace_records()
    drops = [item for item in records if item[2] == "judgment.dropped"]
    assert len(drops) == 1
    assert drops[0][3]["reason"] == "evidence_identity"
    assert drops[0][3]["slot"] == "completed"


def test_provider_scheduler_rejects_closed_or_fatal_offer_without_spend():
    from percept.runtime.worker import RuntimeStats, _JudgeScheduler

    class Goals:
        revision = 0

    class Core:
        goals = Goals()

        async def perceive_judged(self, frame, source, **kwargs):
            return []

    for state in ("closed", "fatal"):
        stats = RuntimeStats()
        scheduler = _JudgeScheduler(Core(), "camera", stats)
        if state == "closed":
            scheduler._closing = True
        else:
            scheduler._fatal = RuntimeError("fatal")
        assert not scheduler.can_accept
        assert scheduler.submit(1.0, "candidate") is False
        assert (scheduler.active, scheduler.pending) == (0, 0)
        assert stats.judge_calls == 0
        assert stats.perceive_dispatches == 0
