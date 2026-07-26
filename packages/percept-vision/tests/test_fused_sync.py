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


def test_emitted_anchor_dedup_is_constant_space_and_resets_with_generation():
    sync = _FusedSynchronizer(
        _FusedSyncPolicy(
            window_s=0.1,
            max_skew_s=0.0,
            max_units_per_modality=8,
            max_bytes_per_modality=4096,
        ),
        watch_application="app",
        watch_revision=1,
    )
    latest = None
    for seq in range(500):
        at = float(seq + 1)
        latest = _video(seq, at, watermark=at)
        assert sync.accept(latest)
        pending = sync.bundle_for(latest)
        assert _FusedCoverageUncertainty.WATERMARK_PENDING in pending.coverage_reasons
        assert sync.accept(_audio(seq, at - 0.1, at, watermark=at))
        ready = sync.drain_ready()
        assert len(ready) == 1
        assert ready[0].video_units[-1].source_sequence == seq
        assert len(sync._lanes[next(
            identity for identity in sync._lanes if identity[1] == "video"
        )].units) <= 8
        assert len(sync._lanes[next(
            identity for identity in sync._lanes if identity[1] == "audio"
        )].units) <= 8

    assert not hasattr(sync, "_emitted_anchor_ids")
    assert sync._emitted_anchor_sequence == 499
    assert latest is not None
    # Re-visiting the most recent accepted anchor cannot put it back into pending.
    sync.bundle_for(latest)
    assert sync.drain_ready() == ()

    generation = sync.advance_generation(_FusedDiscontinuity.RECONNECT)
    assert sync._emitted_anchor_sequence is None
    restarted = _video(0, 1_000.0, watermark=1_000.0, generation=generation)
    assert sync.accept(restarted)
    sync.bundle_for(restarted)
    assert sync.accept(
        _audio(
            0,
            999.9,
            1_000.0,
            watermark=1_000.0,
            generation=generation,
        )
    )
    assert len(sync.drain_ready()) == 1
    assert sync._emitted_anchor_sequence == 0


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
        return (
            None
            if job is None
            else replace(
                job,
                provider=replace(job.provider, dispatch=leaking_dispatch),
            )
        )

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
        return (
            None
            if job is None
            else replace(
                job,
                provider=replace(job.provider, dispatch=retained_provider),
            )
        )

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
    bad_completion = replace(completion, job=replace(job, evidence=bad_bundle))
    assert core.reduce_fused_completion(
        bad_completion,
        live_fused_generation=sync.generation,
    ) == []
    records = core.drain_fused_trace_records()
    drops = [item for item in records if item[2] == "judgment.dropped"]
    assert len(drops) == 1
    assert drops[0][3]["reason"] == "evidence_identity"
    assert drops[0][3]["slot"] == "completed"


def test_fused_admission_and_reduction_ledgers_evict_as_one_replay_firewall():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    core._fused_admission_ledger_max = 2
    for index in range(3):
        assert apply_watch_plan(
            core,
            WatchPlan(
                id=f"fused-{index}",
                condition=f"condition {index}",
                sense="fused",
            ),
            source_profile=_profile(
                f"ledger-av-{index}",
                threshold=0.1,
                modalities=frozenset({"video", "audio"}),
            ),
        ).status == "applied"
    revision = core.goals.revision
    jobs = []
    completions = []
    syncs = []
    for index in range(3):
        goal_id = f"fused-{index}"
        binding = core._audio_judge_bindings[goal_id]
        sync = _FusedSynchronizer(
            _FusedSyncPolicy(window_s=0.5, max_skew_s=0.0),
            watch_application=binding.application_key,
            watch_revision=revision,
        )
        video = replace(
            _video(index, float(index + 1), watermark=float(index + 2)),
            watch_application=binding.application_key,
            watch_revision=revision,
        )
        audio = replace(
            _audio(
                index,
                float(index) + 0.5,
                float(index + 1),
                watermark=float(index + 2),
            ),
            watch_application=binding.application_key,
            watch_revision=revision,
        )
        assert sync.accept(video)
        assert sync.accept(audio)
        job = core.prepare_fused_judgment(
            sync.bundle_for(video),
            goal_id=goal_id,
            source="camera",
            source_t=float(index + 1),
            goals_revision=revision,
            source_generation=core.source_continuity_generation("camera"),
            fused_generation=sync.generation,
        )
        jobs.append(job)
        completions.append(asyncio.run(core.dispatch_fused_judgment(job)))
        syncs.append(sync)

    assert len(core._fused_latest_admission) == 2
    assert len(core._fused_reduced_admissions) <= 2
    assert ("fused-0", None) not in core._fused_latest_admission
    # Once both paired rows are evicted, an old completion cannot become valid
    # again: missing latest-admission identity fails closed.
    assert core.reduce_fused_completion(
        completions[0], live_fused_generation=syncs[0].generation
    ) == []
    for index in (1, 2):
        core.reduce_fused_completion(
            completions[index], live_fused_generation=syncs[index].generation
        )
        assert core.reduce_fused_completion(
            completions[index], live_fused_generation=syncs[index].generation
        ) == []
    assert len(core._fused_latest_admission) == 2
    assert len(core._fused_reduced_admissions) <= 2


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


def test_fused_pending_supersession_records_exact_dropped_job_identity():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.runtime.worker import RuntimeStats, _LegacyJudgeScheduler
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "supersede-av",
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
    base = core.prepare_fused_judgment(
        sync.bundle_for(anchor),
        goal_id="fused",
        source="camera",
        source_t=1.0,
        goals_revision=revision,
        source_generation=core.source_continuity_generation("camera"),
        fused_generation=sync.generation,
    )
    pending = replace(
        base,
        token=replace(
            base.token,
            admission_id="pending-exact",
            evidence_seq=base.token.evidence_seq + 1,
        ),
    )
    newer = replace(
        base,
        token=replace(
            base.token,
            admission_id="newer-exact",
            evidence_seq=base.token.evidence_seq + 2,
        ),
    )

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked(job, _source=None, **_kwargs):
            started.set()
            await release.wait()
            return None

        scheduler = _LegacyJudgeScheduler(
            core,
            "camera",
            RuntimeStats(),
            perceive=blocked,
        )
        assert scheduler.submit(1.0, base)
        await started.wait()
        assert scheduler.submit(2.0, pending)
        assert scheduler.submit(3.0, newer)
        scheduler_trace = scheduler.drain_trace()
        fused_trace = core.drain_fused_trace_records()
        release.set()
        await scheduler.flush()
        return scheduler_trace, fused_trace

    scheduler_trace, fused_trace = asyncio.run(exercise())
    superseded = [
        data for kind, data in scheduler_trace if kind == "judge_superseded"
    ]
    assert superseded == [
        {
            "source": "camera",
            "reason": "superseded_by_newer_admission",
            "dropped_t": 2.0,
            "admitted_t": 3.0,
            "admission_id": "pending-exact",
            "evidence_seq": pending.token.evidence_seq,
            "goal_id": "fused",
            "slot": "pending",
        }
    ]
    drops = [data for _, _, stage, data in fused_trace if stage == "judgment.dropped"]
    assert drops == [
        {
            "goal_id": "fused",
            "admission_id": "pending-exact",
            "evidence_seq": pending.token.evidence_seq,
            "reason": "superseded_by_newer_admission",
            "slot": "pending",
        }
    ]


def test_public_midrun_mixed_watch_revision_runs_through_shared_owner():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.harness.tier0 import gray_frame
    from percept.runtime import RuntimeStats, run
    from percept.runtime.envelope import legacy_video_envelope
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import (
        STILL,
        W,
        H,
        _profile,
    )

    vision = FakeVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "dynamic-guard-av",
            threshold=0.1,
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"
    initial_revision = core.goals.revision
    exposed_frames = []
    delivered = []
    health = []
    stats = RuntimeStats()
    provider_calls = []
    original_perceive = core.perceive_judged
    original_fused_dispatch = core.dispatch_fused_judgment

    async def counted_perceive(*args, **kwargs):
        provider_calls.append("ordinary")
        return await original_perceive(*args, **kwargs)

    async def counted_fused_dispatch(*args, **kwargs):
        provider_calls.append("fused")
        return await original_fused_dispatch(*args, **kwargs)

    core.perceive_judged = counted_perceive
    core.dispatch_fused_judgment = counted_fused_dispatch

    class ClockedAsyncSource:
        clock_authority = "test.replay.shared"
        clock_confidence = 1.0

        def __aiter__(self):
            return source()

    class ClockedAudio(list):
        clock_authority = "test.replay.shared"
        clock_confidence = 1.0

    async def source():
        yield replace(
            legacy_video_envelope(
                (0.0, STILL, "prime"),
                source_id="video",
                sequence=0,
                observed_time=0.0,
            ),
            clock_authority="test.replay.shared",
            clock_confidence=1.0,
        )
        assert apply_watch_plan(
            core,
            WatchPlan(id="ordinary", condition="someone entered", sense="video"),
            source_profile=_profile("dynamic-guard-video", threshold=0.1),
        ).status == "applied"
        yield replace(
            legacy_video_envelope(
                (1.0, gray_frame(W, H, 255), "must-not-reach-runtime"),
                source_id="video",
                sequence=1,
                observed_time=1.0,
            ),
            clock_authority="test.replay.shared",
            clock_confidence=1.0,
        )

    async def exercise():
        async for incident in run(
            core,
            source=ClockedAsyncSource(),
            audio_source=ClockedAudio(),
            audio_gate=None,
            timebase="media",
            on_frame=lambda *args: exposed_frames.append(args),
            on_incident=delivered.append,
            on_health=health.append,
            stats=stats,
        ):
            delivered.append(incident)

    asyncio.run(exercise())
    assert core.goals.revision == initial_revision + 1
    assert len(exposed_frames) == 2
    assert exposed_frames[0][1] == "prime"
    assert exposed_frames[1][1] == "must-not-reach-runtime"
    assert stats.frames == 2
    assert stats.perceive_dispatches == 1
    assert provider_calls == []
    assert delivered == []
    actions = core.action_log.entries()
    assert [(action.kind, action.goal_id) for action in actions] == [
        ("ask", "ordinary")
    ]


def test_separate_iterable_fusion_requires_clock_proof_before_iteration():
    import asyncio

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.runtime import run
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import _profile

    class Probe:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(())

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "unqualified-av",
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"
    video = Probe()
    audio = Probe()

    async def exercise():
        with pytest.raises(ValueError, match="explicit matching clock_authority"):
            async for _ in run(
                core,
                source=video,
                audio_source=audio,
                audio_gate=None,
                timebase="media",
            ):
                pass

    asyncio.run(exercise())
    assert video.iterations == 0
    assert audio.iterations == 0


def test_heterogeneous_iterable_clock_claim_is_rejected_at_changed_envelope():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts import AudioFrame
    from percept.contracts.watchplan import WatchPlan
    from percept.harness.tier0 import gray_frame
    from percept.runtime import run
    from percept.runtime.envelope import legacy_audio_envelope, legacy_video_envelope
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import STILL, W, H, _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "heterogeneous-av",
            threshold=0.1,
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"

    def video(sequence, at, frame, authority):
        return replace(
            legacy_video_envelope(
                (at, frame, f"frame-{sequence}"),
                source_id="video",
                sequence=sequence,
                observed_time=at,
            ),
            clock_authority=authority,
            clock_confidence=1.0,
        )

    source = [
        video(0, 0.0, STILL, "clock:A"),
        video(1, 1.0, gray_frame(W, H, 255), "clock:B"),
    ]
    audio = [
        replace(
            legacy_audio_envelope(
                (
                    0.5,
                    AudioFrame(pcm=b"\x01\x00" * 8_000, sample_rate=16_000),
                    0.5,
                ),
                source_id="audio",
                sequence=0,
                observed_time=0.5,
            ),
            clock_authority="clock:A",
            clock_confidence=1.0,
        )
    ]

    async def exercise():
        with pytest.raises(ValueError, match="explicit matching clock_authority"):
            async for _ in run(
                core,
                source=source,
                audio_source=audio,
                audio_gate=None,
                timebase="media",
            ):
                pass

    asyncio.run(exercise())


def test_async_iterable_clock_declaration_is_checked_on_every_envelope():
    import asyncio
    from dataclasses import replace

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts.watchplan import WatchPlan
    from percept.runtime import run
    from percept.runtime.envelope import legacy_video_envelope
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import STILL, _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "async-clock-av",
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"

    class Video:
        clock_authority = "clock:A"
        clock_confidence = 1.0

        async def __aiter__(self):
            for sequence, authority in enumerate(("clock:A", "clock:B")):
                yield replace(
                    legacy_video_envelope(
                        (float(sequence), STILL, f"frame-{sequence}"),
                        source_id="video",
                        sequence=sequence,
                        observed_time=float(sequence),
                    ),
                    clock_authority=authority,
                    clock_confidence=1.0,
                )

    class Audio:
        clock_authority = "clock:A"
        clock_confidence = 1.0

        async def __aiter__(self):
            if False:
                yield None

    async def exercise():
        with pytest.raises(ValueError, match="video envelope changed"):
            async for _ in run(
                core,
                source=Video(),
                audio_source=Audio(),
                audio_gate=None,
                timebase="media",
            ):
                pass

    asyncio.run(exercise())


def test_wall_clock_iterables_preserve_runtime_derived_authority_without_claims():
    import asyncio

    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts import AudioFrame
    from percept.contracts.watchplan import WatchPlan
    from percept.runtime import run
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import STILL, _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "wall-av",
            modalities=frozenset({"video", "audio"}),
            timebase="wall",
        ),
    ).status == "applied"

    async def exercise():
        return [
            incident
            async for incident in run(
                core,
                source=[(1.0, STILL, "prime")],
                audio_source=[
                    (
                        1.0,
                        AudioFrame(pcm=b"\x01\x00" * 160, sample_rate=16_000),
                        0.1,
                    )
                ],
                audio_gate=None,
                timebase="wall",
            )
        ]

    assert asyncio.run(exercise()) == []


def test_same_container_preserves_media_authority_without_envelope_claims(
    monkeypatch,
):
    import asyncio

    import percept.runtime.worker as worker
    from percept import Percept
    from percept.backends.fake import FakeVision
    from percept.contracts import AudioFrame
    from percept.contracts.watchplan import WatchPlan
    from percept.harness.tier0 import gray_frame
    from percept.runtime import run
    from percept.runtime.envelope import legacy_audio_envelope, legacy_video_envelope
    from percept.watchplan_apply import apply_watch_plan
    from tests.test_watchplan_runtime_binding import STILL, W, H, _profile

    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    assert apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "container-av",
            modalities=frozenset({"video", "audio"}),
        ),
    ).status == "applied"

    async def fake_video(_source, _fps, **_kwargs):
        yield legacy_video_envelope(
            (0.0, STILL, "prime"),
            source_id="video",
            sequence=0,
            observed_time=0.0,
        )
        yield legacy_video_envelope(
            (1.0, gray_frame(W, H, 255), "changed"),
            source_id="video",
            sequence=1,
            observed_time=1.0,
        )

    async def fake_audio(_source, **_kwargs):
        yield worker._AudioIngress(
            envelope=legacy_audio_envelope(
                (
                    0.5,
                    AudioFrame(pcm=b"\x01\x00" * 8_000, sample_rate=16_000),
                    0.5,
                ),
                source_id="audio",
                sequence=0,
                observed_time=0.5,
            )
        )

    monkeypatch.setattr(worker, "_aiter", fake_video)
    monkeypatch.setattr(worker, "_audio_aiter", fake_audio)

    async def exercise():
        return [
            incident
            async for incident in run(
                core,
                source="qualified.mp4",
                audio_source="qualified.mp4",
                audio_gate=None,
                timebase="media",
            )
        ]

    asyncio.run(exercise())
