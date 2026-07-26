#!/usr/bin/env python3
"""Source-free public product smoke for a candidate percept-vision wheel.

Run this script with the Python interpreter into which the exact candidate wheel
was installed. It exercises only shipped ``percept`` modules; the script itself
remains repository release tooling.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import percept
from percept import Percept, Verdict
from percept.audio_input import (
    AudioInputCapability,
    audio_frame_from_units,
    audio_script_token,
    canonicalize_audio_unit,
)
from percept.backends.fake import FakeVision
from percept.backends.judge_pool import JudgeCapability, JudgePool, JudgeRoute
from percept.config import Settings
from percept.contracts import AudioFrame, WatchPlan
from percept.domain_pack import get_domain_pack, watch_template
from percept.events import AudioUnit
from percept.harness.tier0 import gray_frame
from percept.resolver import ConfigResolver
from percept.runtime import RuntimeStats, run
from percept.runtime.envelope import legacy_audio_envelope, legacy_video_envelope
from percept.runtime_compiler import runtime_compile
from percept.sourceprofile import SourceProfile
from percept.watchplan_apply import apply_watch_plan


ROOT = Path(__file__).resolve().parents[1]
SDK_SOURCE = (ROOT / "packages" / "percept-vision").resolve()
FORBIDDEN_ARTIFACT_PARTS = {
    ".env",
    "architectural-probes",
    "internal-docs",
    "scratch",
}


class _ClockedReplay(list):
    clock_authority = "percept.wheel-smoke.shared"
    clock_confidence = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_artifact_names(names: list[str], *, wheel: bool) -> None:
    for name in names:
        parts = set(Path(name).parts)
        forbidden = parts & FORBIDDEN_ARTIFACT_PARTS
        if forbidden:
            raise AssertionError(
                f"artifact contains forbidden internal path {name!r}: "
                + ", ".join(sorted(forbidden))
            )
    if wheel:
        unexpected = [
            name
            for name in names
            if not name.startswith("percept/")
            and ".dist-info/" not in name
        ]
        if unexpected:
            raise AssertionError(
                "wheel contains unexpected top-level namespace: "
                + ", ".join(unexpected[:5])
            )
        if any(name.startswith("eval/") for name in names):
            raise AssertionError("wheel must not ship the source-only top-level eval package")


def _artifact_identity(wheel: Path | None, sdist: Path | None) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    if wheel is not None:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            _audit_artifact_names(names, wheel=True)
            build_info = archive.read("percept/_build_info.py").decode("utf-8")
        installed = percept.build_info()
        for field, value in {
            "PACKAGE_VERSION": installed.package_version,
            "GIT_COMMIT": installed.git_commit,
            "CONTRACT_VERSION": installed.contract_version,
            "BUILD_TIMESTAMP": installed.build_timestamp,
        }.items():
            if repr(value) not in build_info:
                raise AssertionError(
                    f"installed {field} does not match selected wheel"
                )
        identity["wheel_sha256"] = _sha256(wheel)
    if sdist is not None:
        import tarfile

        with tarfile.open(sdist, "r:gz") as archive:
            _audit_artifact_names(archive.getnames(), wheel=False)
        identity["sdist_sha256"] = _sha256(sdist)
    return identity


def _profile(name: str, modalities: frozenset[str]) -> SourceProfile:
    return SourceProfile(
        name=name,
        modalities=modalities,
        temporal_unit="frame",
        admission_strategy="pixel-change",
        evidence_strategy="frame",
        identity_strategy="track_continuity",
        default_timebase="media",
        parameters={
            "salience_stat": "block_max",
            "salience_threshold": 0.1,
        },
    )


async def _incidents(core: Percept, **kwargs: Any) -> list[Any]:
    observed: list[Any] = []
    yielded = [
        incident
        async for incident in run(
            core,
            on_incident=observed.append,
            judge_dispatch="inline",
            **kwargs,
        )
    ]
    if yielded != observed:
        raise AssertionError("subscriber and async iterator observed different Incidents")
    return yielded


async def _video_smoke() -> dict[str, Any]:
    still = gray_frame(24, 18, 0)
    changed = gray_frame(24, 18, 255)
    core = Percept.create(
        vision=FakeVision(
            script={
                "prime": Verdict(False, 0.99, "absent"),
                "changed": Verdict(True, 0.99, "present"),
                "maybe": Verdict(True, 0.1, "uncertain"),
            }
        ),
        mode="offline",
        discover_plugins=False,
    )
    receipt = apply_watch_plan(
        core,
        WatchPlan(id="video", condition="the object appears", sense="video"),
        source_profile=_profile("wheel-video", frozenset({"video"})),
    )
    if receipt.status not in {"applied", "applied_with_warnings"}:
        raise AssertionError(f"video Watch failed to apply: {receipt}")
    stats = RuntimeStats()
    incidents = await _incidents(
        core,
        source=[(0.0, still, "prime"), (1.0, changed, "changed")],
        stats=stats,
    )
    if len(incidents) != 1 or incidents[0].concern_id != "video":
        raise AssertionError("compiled video Watch did not produce one rising-edge Incident")

    ask_core = Percept.create(
        vision=FakeVision(script={"maybe": Verdict(True, 0.1, "uncertain")}),
        mode="offline",
        discover_plugins=False,
    )
    ask_receipt = apply_watch_plan(
        ask_core,
        WatchPlan(id="ask", condition="the reading is uncertain", sense="video"),
    )
    if ask_receipt.status not in {"applied", "applied_with_warnings"}:
        raise AssertionError(f"ask Watch failed to apply: {ask_receipt}")
    asks = await ask_core.perceive_judged("maybe", t=0.0)
    if len(asks) != 1 or asks[0].action != "ask":
        raise AssertionError("low-confidence installed-wheel judgment did not emit ask")
    return {
        "incidents": len(incidents),
        "asks": len(asks),
        "admissions": stats.admission_by_goal,
    }


def _audio_candidate() -> AudioUnit:
    chunk = AudioFrame(
        pcm=struct.pack("<h", 32767),
        sample_rate=4,
        channels=1,
        sample_format="s16le",
        t_start=0.0,
        t_end=0.25,
    )
    return AudioUnit(
        seq=0,
        t_start=0.0,
        t_end=0.25,
        event_t=0.25,
        chunk=chunk,
        energy=1.0,
    )


def _audio_token(candidate: AudioUnit) -> str:
    capability = AudioInputCapability(modes=frozenset({"audio_only"}))
    canonical = canonicalize_audio_unit(
        candidate,
        capability=capability,
        source_id="audio",
        source_epoch=0,
        source_sequence=0,
        observed_time=0.25,
        source_time=0.25,
    )
    return audio_script_token(
        audio_frame_from_units((canonical,), capability=capability)
    )


async def _audio_smoke() -> dict[str, Any]:
    candidate = _audio_candidate()
    core = Percept.create(
        vision=FakeVision(
            script={
                _audio_token(candidate): Verdict(True, 0.99, "heard")
            }
        ),
        mode="offline",
        discover_plugins=False,
    )
    receipt = apply_watch_plan(
        core,
        WatchPlan(id="audio", condition="the sound occurs", sense="audio"),
    )
    if receipt.status not in {"applied", "applied_with_warnings"}:
        raise AssertionError(f"audio Watch failed to apply: {receipt}")
    stats = RuntimeStats()
    incidents = await _incidents(
        core,
        source=None,
        audio_source=[candidate],
        audio_settle_s=0.0,
        audio_wake={"high": 0.1, "low": 0.05},
        stats=stats,
    )
    if len(incidents) != 1 or incidents[0].snippet.kind != "audio":
        raise AssertionError("compiled audio Watch did not retain exact audio evidence")
    return {
        "incidents": len(incidents),
        "admissions": stats.admission_by_goal,
    }


async def _fused_smoke() -> dict[str, Any]:
    still = gray_frame(24, 18, 0)
    changed = gray_frame(24, 18, 255)
    core = Percept.create(
        vision=FakeVision(
            script={
                "changed|audio": Verdict(True, 0.99, "seen and heard")
            }
        ),
        mode="offline",
        discover_plugins=False,
    )
    receipt = apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
        source_profile=_profile(
            "wheel-fused",
            frozenset({"video", "audio"}),
        ),
    )
    if receipt.status not in {"applied", "applied_with_warnings"}:
        raise AssertionError(f"fused Watch failed to apply: {receipt}")
    stats = RuntimeStats()
    video_source = _ClockedReplay(
        replace(
            legacy_video_envelope(
                item,
                source_id="video",
                sequence=sequence,
                observed_time=float(item[0]),
            ),
            clock_authority=_ClockedReplay.clock_authority,
            clock_confidence=_ClockedReplay.clock_confidence,
        )
        for sequence, item in enumerate(
            [(0.0, still, "prime"), (1.0, changed, "changed")]
        )
    )
    audio_source = _ClockedReplay(
        replace(
            legacy_audio_envelope(
                item,
                source_id="audio",
                sequence=sequence,
                observed_time=float(item[0]),
            ),
            clock_authority=_ClockedReplay.clock_authority,
            clock_confidence=_ClockedReplay.clock_confidence,
        )
        for sequence, item in enumerate(
            [
                (
                    0.5,
                    AudioFrame(
                        pcm=b"\x01\x00" * 8_000,
                        sample_rate=16_000,
                    ),
                    0.5,
                )
            ]
        )
    )
    incidents = await _incidents(
        core,
        source=video_source,
        audio_source=audio_source,
        audio_gate=None,
        stats=stats,
    )
    if len(incidents) != 1:
        raise AssertionError("compiled fused Watch did not produce one Incident")
    if getattr(incidents[0], "_fused_evidence_bundle", None) is None:
        raise AssertionError("fused Incident lost its synchronized evidence bundle")
    return {
        "incidents": len(incidents),
        "admissions": stats.admission_by_goal,
    }


def _pack_smoke() -> list[str]:
    compiled_names: list[str] = []
    for ref in (
        "physical_scene_starter@1/package_appeared",
        "ego_motion_starter@1/scene_transition",
    ):
        pack = get_domain_pack(ref.partition("/")[0])
        plan = watch_template(ref)
        resolved = ConfigResolver().resolve(
            plan,
            domain_pack=pack,
            settings=Settings(),
        )
        compiled = runtime_compile(resolved)
        core = Percept.create(
            vision=FakeVision(),
            settings=compiled.settings,
            mode="offline",
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            plan,
            domain_pack=pack,
            settings=compiled.settings,
        )
        if receipt.status not in {"applied", "applied_with_warnings"}:
            raise AssertionError(f"starter pack Watch failed to apply: {receipt}")
        if core.goals.get(compiled.goal.id).id != compiled.goal.id:
            raise AssertionError("compiled starter pack goal identity changed at apply")
        compiled_names.append(ref)
    return compiled_names


async def _deadline_smoke() -> dict[str, Any]:
    class SlowJudge:
        async def judge(self, *_args: Any, **_kwargs: Any) -> Verdict:
            await asyncio.sleep(0.2)
            return Verdict(True, 0.99, "too late")

    pool = JudgePool(
        [
            JudgeRoute(
                "slow",
                SlowJudge(),
                JudgeCapability(
                    evidence_shapes=frozenset({"single_image"}),
                    latency_slo_s=0.01,
                    modes=frozenset({"reflex"}),
                    max_inflight=1,
                ),
            )
        ],
        deadline_s=0.03,
    )
    started = time.monotonic()
    verdict = await pool.judge("condition", b"frame")
    elapsed = time.monotonic() - started
    await asyncio.sleep(0)
    if verdict.confidence != 0.0 or elapsed >= 0.1:
        raise AssertionError("JudgePool failed to bound the reflex deadline")
    if pool.snapshot()["routes"]["slow"]["inflight"] != 0:
        raise AssertionError("JudgePool leaked in-flight work after deadline")
    return {
        "elapsed_s": elapsed,
        "outcomes": [receipt.outcome for receipt in pool.receipts()],
    }


async def _run_smoke() -> dict[str, Any]:
    return {
        "video": await _video_smoke(),
        "audio": await _audio_smoke(),
        "fused": await _fused_smoke(),
        "packs": _pack_smoke(),
        "deadline": await _deadline_smoke(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="verify public multimodal paths from an installed percept wheel"
    )
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument(
        "--allow-source",
        action="store_true",
        help="test the smoke mechanics from a checkout; never use for qualification",
    )
    args = parser.parse_args(argv)

    module_path = Path(percept.__file__ or "").resolve()
    if not args.allow_source and (
        module_path == SDK_SOURCE or SDK_SOURCE in module_path.parents
    ):
        raise SystemExit(
            f"WHEEL SMOKE BLOCKED: percept resolved from source tree: {module_path}"
        )
    if not args.allow_source and args.wheel is None:
        raise SystemExit("WHEEL SMOKE BLOCKED: --wheel is required")
    for path in (args.wheel, args.sdist):
        if path is not None and not path.is_file():
            raise SystemExit(f"WHEEL SMOKE BLOCKED: artifact missing: {path}")

    report = {
        "schema_version": "percept.wheel_smoke/1",
        "percept_module": str(module_path),
        "build_info": percept.build_info().to_dict(),
        "artifacts": _artifact_identity(args.wheel, args.sdist),
        "product_paths": asyncio.run(_run_smoke()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
