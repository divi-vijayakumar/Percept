"""Phase-A v0.5 runner — score the tracks and emit the DRAFT Snapshot Card.

Offline smoke (no API, renders the benchmark card with stub cells):
    PYTHONPATH=packages/percept-vision python3.12 -m eval.bench.run
Live sampled:
    PERCEPT_LIVE=1 PYTHONPATH=... python3.12 -m eval.bench.run --live --n 20

Live mode intentionally skips legacy dense/raw media-bearing tracks. Publishable benchmark runs should
use edge/E2E candidate calls only; dense call counts are modeled from the sampling schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

from . import card
from . import (
    track_acoustic_pt,
    track_counting,
    track_cuespeak,
    track_edge_recall,
    track_perception_test_e2e,
    track_relational,
    track_timeliness,
    track_timespeak,
)
from .types import apply_acceptance_gates


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "?"


def _manifest_ref(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return f"{p.as_posix()} sha256:{digest}"


def _backend_label(live: bool) -> str:
    if not live:
        return "fake (offline stub)"
    return os.environ.get("PERCEPT_VISION_MODEL") or os.environ.get("PERCEPT_GEMINI_MODEL") or "gemini"


def _live_prerequisite_blockers(
    *,
    manifest: str | None,
    release: bool,
    n: int | None = None,
    tracks: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return blockers without initializing a provider or fetching benchmark media."""
    from percept.config import missing_credential

    backend = os.environ.get("PERCEPT_VISION_BACKEND", "gemini")
    blockers: list[str] = []
    missing = missing_credential(backend)
    if missing:
        blockers.append(f"{backend} credential missing (set one of: {', '.join(missing)})")
    from .budget import (
        BudgetPlanError,
        build_budget_enforcer,
        live_budget_blockers,
    )

    configured_model = os.environ.get("PERCEPT_VISION_MODEL")
    if not configured_model:
        blockers.append("PERCEPT_VISION_MODEL must explicitly pin the budgeted provider model")
    if "pt-action-e2e" in tracks:
        if backend != "openrouter":
            blockers.append(
                "pt-action-e2e qualification requires "
                "PERCEPT_VISION_BACKEND=openrouter"
            )
        if not configured_model or not configured_model.lower().startswith(
            "qwen/"
        ):
            blockers.append(
                "pt-action-e2e qualification requires an explicitly pinned "
                "Qwen model through OpenRouter"
            )
    raw_budget = os.environ.get("PERCEPT_BENCH_BUDGET_PLAN")
    budget_blockers = live_budget_blockers(
        provider=backend,
        model=configured_model,
        raw=raw_budget,
    )
    blockers.extend(budget_blockers)
    budget = None
    if not budget_blockers:
        try:
            budget = build_budget_enforcer(
                raw_budget or "",
                provider=backend,
                model=configured_model,
            )
        except BudgetPlanError:
            pass
    if manifest and n is not None and (not tracks or "pt-action-e2e" in tracks):
        from .perception_test import (
            GoldenSetCapacityError,
            action_media_minutes,
            golden_action_capacity,
            select_golden_action_events,
        )

        capacity = golden_action_capacity(manifest=manifest)
        if release and n != capacity["licensed"]:
            blockers.append(
                f"full pt-action-e2e qualification requires N={capacity['licensed']} "
                f"for every pinned licensed case; got N={n}"
            )
        elif n > capacity["licensed"]:
            blockers.append(
                f"pt-action-e2e requested N={n}, but the pinned manifest has only "
                f"{capacity['licensed']} licensed action cases"
            )
        elif n > capacity["verified_local"]:
            blockers.append(
                f"pt-action-e2e requested N={n}, but only "
                f"{capacity['verified_local']} of {capacity['licensed']} licensed pinned "
                "cases have verified local annotations/media"
            )
        elif budget is not None:
            try:
                selected_events = select_golden_action_events(
                    n,
                    manifest=manifest,
                    require_media=True,
                )
                selected_minutes = action_media_minutes(selected_events)
            except (GoldenSetCapacityError, ImportError, OSError, ValueError) as exc:
                blockers.append(f"selected media duration could not be verified: {exc}")
            else:
                media_cap = budget.media_minutes_cap
                if selected_minutes > media_cap + 1e-12:
                    blockers.append(
                        f"selected pinned media is {selected_minutes:.6f} minutes, "
                        f"exceeding the effective media cap of {media_cap:.6f} minutes"
                    )
    if release:
        if not manifest:
            blockers.append("release qualification requires a pinned manifest")
        else:
            try:
                from .validate import ValidationError, validate_golden_manifest

                data = validate_golden_manifest(manifest, release=True)
            except (FileNotFoundError, ValidationError) as exc:
                blockers.append(f"release manifest is not ready: {exc}")
            else:
                missing_media: list[str] = []
                for item in data["splits"]["test"]:
                    raw = item.get("media_path")
                    if not raw:
                        missing_media.append(f"{item['id']} (no media_path)")
                        continue
                    path = Path(raw)
                    if not path.is_absolute():
                        path = Path.cwd() / path
                    if not path.is_file() or path.stat().st_size == 0:
                        missing_media.append(item["id"])
                if missing_media:
                    blockers.append(
                        "release media missing/empty for " + ", ".join(missing_media)
                    )
    return blockers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="run a real configured provider")
    mode.add_argument(
        "--offline",
        action="store_true",
        help="run the deterministic/keyless OSS scorecard (does not qualify a provider)",
    )
    ap.add_argument("--n", type=int, default=20, help="relational sample size")
    ap.add_argument("--out", default=None, help="also write the benchmark card to this path")
    ap.add_argument("--manifest", default=None, help="path to the pinned data manifest to stamp on the benchmark card")
    ap.add_argument("--mode", default="sampled", choices=["sampled", "full"], help="scorecard mode")
    ap.add_argument("--device", default="?", help="device/runtime descriptor for the benchmark card header")
    ap.add_argument("--fps", default="1", help="sampling/capture fps for the benchmark card header")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds/runs represented")
    ap.add_argument(
        "--require-release",
        action="store_true",
        help="exit non-zero unless the rendered benchmark card is stamped RELEASE",
    )
    args = ap.parse_args(argv)

    if not args.manifest:
        print("SCORECARD EXECUTION FAILED: a non-empty benchmark manifest is required")
        return 1
    try:
        from .validate import ValidationError, validate_golden_manifest

        manifest_data = validate_golden_manifest(
            args.manifest,
            release=False,
        )
    except (FileNotFoundError, ValidationError) as exc:
        print(f"SCORECARD EXECUTION FAILED: manifest audit failed: {exc}")
        return 1

    if args.live:
        from eval._dotenv import load_dotenv
        load_dotenv()
        blockers = _live_prerequisite_blockers(
            manifest=args.manifest,
            release=args.require_release,
            n=args.n,
            tracks=["pt-action-e2e"],
        )
        if blockers:
            print("BENCHMARK LIVE GATE: BLOCKED")
            for blocker in blockers:
                print(f"  - {blocker}")
            return 2
    elif args.require_release:
        print("BENCHMARK RELEASE GATE: BLOCKED — release qualification requires --live")
        return 2

    results = []
    if args.live:
        # OSS launch qualification has one canonical live lane. The other
        # modules remain directly runnable diagnostics until they are migrated
        # to the compiled runtime-v2 result contract.
        tracks = (track_perception_test_e2e,)
    else:
        tracks = (track_relational, track_counting, track_acoustic_pt, track_timeliness, track_edge_recall,
                  track_cuespeak, track_timespeak)
    previous_mode = os.environ.get("PERCEPT_MODE")
    previous_manifest = os.environ.get("PERCEPT_BENCH_MANIFEST")
    os.environ["PERCEPT_BENCH_MANIFEST"] = str(Path(args.manifest).resolve())
    if not args.live:
        os.environ["PERCEPT_MODE"] = "offline"
    try:
        for mod in tracks:
            try:
                results.append(mod.score(live=args.live, n=args.n))
            except Exception as exc:  # A track failure must not sink the whole benchmark report.
                from .types import TrackResult
                results.append(TrackResult(track=getattr(mod, "TRACK", mod.__name__),
                                           notes=f"FAILED: {type(exc).__name__}: {exc}",
                                           excluded=[f"{getattr(mod, 'TRACK', '?')}: {exc}"],
                                           status="failed"))
    finally:
        if not args.live:
            if previous_mode is None:
                os.environ.pop("PERCEPT_MODE", None)
            else:
                os.environ["PERCEPT_MODE"] = previous_mode
        if previous_manifest is None:
            os.environ.pop("PERCEPT_BENCH_MANIFEST", None)
        else:
            os.environ["PERCEPT_BENCH_MANIFEST"] = previous_manifest

    if args.require_release:
        from .validate import manifest_runner_map

        mappings = list(
            manifest_runner_map(
                manifest_data,
                require_all_tracks=True,
            ).values()
        )
        for result in results:
            mapping = next(
                (
                    item
                    for item in mappings
                    if item["result_track"] == result.track
                ),
                None,
            )
            acceptance = (
                mapping.get("acceptance")
                if mapping is not None
                else None
            )
            if isinstance(acceptance, dict) and acceptance:
                apply_acceptance_gates(result, acceptance)

    execution_blockers = card.execution_blockers(results)
    if execution_blockers:
        print("SCORECARD EXECUTION FAILED: " + "; ".join(execution_blockers))
        return 1

    backend = _backend_label(args.live)
    auroc = next((r.cells.get(("core-replay", "auroc")) for r in results if r.track == "relational"), None)
    manifest_ref = _manifest_ref(args.manifest)
    stamp = card.stamp_of(args.mode, results, manifest_ref)
    out = card.render_card(
        results,
        mode=args.mode,
        backend=backend,
        commit=_commit(),
        manifest=manifest_ref,
        fps=args.fps,
        device=args.device,
        seeds=args.seeds,
        auroc=auroc,
    )
    print(out)
    if args.out:
        artifact = Path(args.out)
        artifact.write_text(out)
        if not artifact.exists() or artifact.stat().st_size == 0:
            print(f"SCORECARD EXECUTION FAILED: output is missing or empty: {artifact}")
            return 1
        print(f"  benchmark card written -> {args.out}")
    if args.require_release and stamp != "RELEASE":
        print("SCORECARD RELEASE GATE: BLOCKED — rendered benchmark card is DRAFT")
        return 1
    if args.live:
        print(f"BENCHMARK LIVE GATE: {'PASS' if stamp == 'RELEASE' else 'DRAFT'}")
    else:
        print(
            "OFFLINE REPRODUCTION: PASS — deterministic/keyless scorecard executed; "
            "Cloud provider cost gate: BLOCKED (not run)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
