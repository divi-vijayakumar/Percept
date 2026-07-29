"""AUDIO stage validation — the channel Stage 2 and Stage 4 never measured.

The stage suites SV-0..SV-4 measured the VIDEO path. Audio had exactly two results: SV-0.3 (unit
ordering) PASS and SV-3.6 (judge parity) PASS. Stage-1 admission was REFUSED — the validation corpus
carried no audio ground truth — and Stage-2 local perception and Stage-4 cognition were never
measured at all. This module closes that gap against a corpus of real, human-annotated SOUND events
(``build_audio_corpus.py``).

    SV-A1.6R  VAD / acoustic admission ..... retry of the refused SV-1.6, now with audio ground
                                             truth. Recall + economy of the shipped audio gate, and
                                             the threshold frontier behind it.
    SV-A2.1   Audio senses (determinism) ... pcm_energy / normalize_audio_unit /
                                             canonicalize_audio_unit / EnergyVad — exactness.
    SV-A2.2   Audio capture (real media) ... byte-identical re-capture of the same file, and the
                                             economy of one second of listening.
    SV-A4.1   Gate decision + state ........ rising edge fires ONCE, refractory honoured, repeated
                                             occurrences yield DISTINCT Incidents — on the audio path.
    SV-A4.7   Evidence integrity ........... ZERO TOLERANCE. An audio Incident's evidence must
                                             resolve to the exact PCM that was fed. Hash in, hash out.
    SV-A4.8   Drop reconciliation .......... units in == admitted + suppressed + dropped-with-receipt.
    SV-A4.9   Audio judge boundedness ...... what bounds the compiled audio dispatch, given that it
                                             refuses a JudgePool.

Everything runs through the real public runtime path (``percept.run(audio_source=...)``) with
scripted, network-free judges. No provider is called; audio judging via a provider is out of scope.

Run standalone::

    <venv-with-the-wheel-under-test>/bin/python eval/stage-validation/sv_audio.py

It prints a PASS/FAIL table and writes ``results/sv_audio.json`` + ``results/sv_audio.md``.
``run_all()`` returns the same structure it serializes.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import percept
from percept import Percept
from percept.adapters import Verdict
from percept.audio_input import (
    MAX_AUDIO_INPUT_BYTES,
    MAX_AUDIO_PROVIDER_BYTES,
    AudioInputCapability,
    audio_frame_from_units,
    audio_script_token,
    canonicalize_audio_unit,
    prepare_audio_judge_binding,
)
from percept.backends.fake import FakeVision
from percept.config import load_settings
from percept.contracts import AudioFrame, WatchPlan
from percept.events import AudioUnit, AudioWindow
from percept.harness.detectors import AcousticOnset
from percept.runtime import frame_energy
from percept.runtime._vad import EnergyVad, endpoint_energy_sequence
from percept.runtime.audio import (
    AudioSourceConfig,
    normalize_audio_unit,
    open_audio_source,
    pcm_energy,
)
from percept.runtime.worker import (
    RuntimeStats,
    _amplitude_threshold_to_energy,
    _LegacyJudgeScheduler,
    run,
)
from percept.watchplan_apply import apply_watch_plan

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"
JSON_PATH = RESULTS_DIR / "sv_audio.json"
MD_PATH = RESULTS_DIR / "sv_audio.md"

sys.path.insert(0, str(HERE))
import build_audio_corpus as corpus_builder  # noqa: E402

# --------------------------------------------------------------------------------------------
# shipped defaults, asserted at runtime rather than hardcoded as belief
# --------------------------------------------------------------------------------------------

#: ``run(audio_gate="acoustic-onset")`` with no ``audio_wake`` builds ``AcousticOnset()``.
SHIPPED_ONSET_HIGH = AcousticOnset().high
SHIPPED_ONSET_LOW = AcousticOnset().low
#: ``run(utterance_threshold=0.01)`` — an AMPLITUDE knob, squared into mean-square energy.
SHIPPED_UTTERANCE_THRESHOLD = 0.01
SHIPPED_HANGOVER_FRAMES = 8
SHIPPED_MIN_SPEECH_FRAMES = 3
#: ``AudioSourceConfig.chunk_s`` — the cadence every file/container/mic audio source runs at.
CAPTURE_CHUNK_S = AudioSourceConfig().chunk_s

#: One capture chunk of slack when asking "did the gate wake for this annotated event?". An onset is
#: reported at the chunk's END, so an event starting mid-chunk is reported up to one chunk late.
ADMISSION_TOLERANCE_S = CAPTURE_CHUNK_S

#: The onset thresholds swept for the frontier. ``low`` is held at ``high / 4`` throughout, the
#: shipped 0.09 / 0.0225 ratio, so the sweep moves ONE knob and stays on the shipped gate's shape.
ONSET_SWEEP = (
    0.09, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001,
    3e-4, 1e-4, 5e-5, 3e-5, 1e-5, 3e-6, 1e-6,
)
#: The ``utterance_threshold`` values swept. These are AMPLITUDE, exactly as the public knob is.
UTTERANCE_SWEEP = (0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001)

CAP = AudioInputCapability(modes=frozenset({"audio_only"}))


# --------------------------------------------------------------------------------------------
# result model (same shape the other SV suites emit, so report.py reads it unchanged)
# --------------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "data": _jsonable(self.data),
        }


@dataclass
class Defect:
    id: str
    severity: str  # "blocker" | "high" | "medium" | "low"
    where: str  # file:line
    summary: str
    repro: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TestResult:
    id: str
    title: str
    checks: list[Check] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    defects: list[Defect] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def check(self, name: str, passed: bool, detail: str = "", **data: Any) -> Check:
        item = Check(name=name, passed=bool(passed), detail=detail, data=data)
        self.checks.append(item)
        return item

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        if not self.checks:
            return "ERROR"
        return "PASS" if all(c.passed for c in self.checks) else "FAIL"

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "error": self.error,
            "checks": [c.to_json() for c in self.checks],
            "measurements": _jsonable(self.measurements),
            "defects": [d.to_json() for d in self.defects],
            "notes": list(self.notes),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(obj)).hexdigest(), "len": len(obj)}
    return repr(obj)


# --------------------------------------------------------------------------------------------
# corpus + real-media energy extraction
# --------------------------------------------------------------------------------------------

_CORPUS: dict[str, Any] | None = None
_ENERGY_CACHE: dict[str, list[tuple[float, float, float]]] = {}


def corpus(*, rebuild: bool = False) -> dict[str, Any]:
    """The audio corpus, built in-process.

    Built rather than read from disk on purpose: ``build_audio_corpus.build()`` is deterministic
    (seeded selection over a sorted list) and re-probes every media file, so the corpus the suite
    measured is exactly the corpus it reports — no stale file can drift between them. The full
    document is embedded in ``results/sv_audio.json``.
    """
    global _CORPUS
    if _CORPUS is None or rebuild:
        _CORPUS = corpus_builder.build()
    return _CORPUS


def cases_of(dataset: str) -> list[dict[str, Any]]:
    return [c for c in corpus()["cases"] if c["dataset"] == dataset]


async def _capture(path: Path, *, chunk_s: float = CAPTURE_CHUNK_S) -> list[dict[str, Any]]:
    """Capture one media file's audio through the RUNTIME's own source, not a private decoder."""
    rows: list[dict[str, Any]] = []
    async for envelope in open_audio_source(str(path), chunk_s=chunk_s):
        payload = envelope.payload
        rows.append(
            {
                "seq": envelope.sequence,
                "t_start": float(payload.t_start),
                "t_end": float(payload.t_end),
                "energy": float(envelope.signal),
                "pcm": bytes(payload.pcm),
                "sample_rate": payload.sample_rate,
                "channels": payload.channels,
                "sample_format": payload.sample_format,
            }
        )
    return rows


def capture(path: Path, *, chunk_s: float = CAPTURE_CHUNK_S) -> list[dict[str, Any]]:
    return asyncio.run(_capture(path, chunk_s=chunk_s))


def energies(case: dict[str, Any]) -> list[tuple[float, float, float]]:
    """``(t_start, t_end, energy)`` per capture chunk for one case, cached across tests."""
    key = case["case_id"]
    if key not in _ENERGY_CACHE:
        rows = capture(REPO_ROOT / case["video_path"])
        _ENERGY_CACHE[key] = [(r["t_start"], r["t_end"], r["energy"]) for r in rows]
    return _ENERGY_CACHE[key]


# --------------------------------------------------------------------------------------------
# SV-A1.6R — VAD / acoustic admission with real audio ground truth
# --------------------------------------------------------------------------------------------


def onset_times(rows: Sequence[tuple[float, float, float]], *, high: float, low: float) -> list[float]:
    """Wake instants the SHIPPED gate would report, driven at the shipped capture cadence.

    Uses ``percept.harness.detectors.AcousticOnset`` itself — the object ``run()`` constructs — fed
    ``(t_end, energy)`` exactly as ``_run_audio_only`` feeds it.
    """
    detector = AcousticOnset(high=high, low=low)
    out: list[float] = []
    for _, t_end, energy in rows:
        out.extend(b.t for b in detector.push(t_end, energy))
    return out


def admission_recall(
    cases: Sequence[dict[str, Any]],
    *,
    high: float,
    low: float,
    positives_key: str | None = "sound_positives_any",
    tolerance: float = ADMISSION_TOLERANCE_S,
) -> dict[str, Any]:
    """Recall / economy of the acoustic-onset gate over a set of cases."""
    hits = total = admissions = chunks = 0
    per_case: list[dict[str, Any]] = []
    for case in cases:
        rows = energies(case)
        positives = (
            (case["provenance"].get(positives_key) if positives_key else None)
            or case["positives"]
        )
        wakes = onset_times(rows, high=high, low=low)
        case_hits = sum(
            1
            for start, end in positives
            if any(start - tolerance <= t <= end + tolerance for t in wakes)
        )
        hits += case_hits
        total += len(positives)
        admissions += len(wakes)
        chunks += len(rows)
        per_case.append(
            {
                "case_id": case["case_id"],
                "positives": len(positives),
                "hits": case_hits,
                "wakes": len(wakes),
                "chunks": len(rows),
            }
        )
    return {
        "high": high,
        "low": low,
        "recall": (hits / total) if total else None,
        "hits": hits,
        "positives": total,
        "admissions": admissions,
        "chunks": chunks,
        "economy": (admissions / chunks) if chunks else None,
        "per_case": per_case,
    }


def utterance_profile(cases: Sequence[dict[str, Any]], *, threshold_amplitude: float) -> dict[str, Any]:
    """Speech fraction + endpoints the shipped ``EnergyVad`` produces at one utterance threshold."""
    energy_threshold = _amplitude_threshold_to_energy(
        threshold_amplitude, name="utterance_threshold"
    )
    endpoints = speech = frames = 0
    seconds = 0.0
    for case in cases:
        rows = energies(case)
        vad = EnergyVad(
            threshold=energy_threshold,
            hangover_frames=SHIPPED_HANGOVER_FRAMES,
            min_speech_frames=SHIPPED_MIN_SPEECH_FRAMES,
        )
        for t_start, t_end, energy in rows:
            event = vad.push(min(1.0, energy))
            frames += 1
            seconds += t_end - t_start
            speech += 1 if event.speech else 0
            endpoints += 1 if event.endpoint else 0
    return {
        "utterance_threshold": threshold_amplitude,
        "energy_threshold": energy_threshold,
        "frames": frames,
        "media_seconds": round(seconds, 2),
        "speech_fraction": (speech / frames) if frames else None,
        "endpoints": endpoints,
        "endpoints_per_minute": (endpoints / (seconds / 60.0)) if seconds else None,
    }


def _golden_sound_events() -> list[dict[str, Any]]:
    """The golden `perception-test-acoustic` track's OWN case anchors, from its own loader."""
    bench = corpus_builder.load_pt_bench()
    if bench is None:
        return []
    try:
        return list(bench.golden_sound_events(require_media=False))
    except Exception:  # noqa: BLE001 - the manifest's media may not be staged
        return []


def sv_a1_6r_vad_admission() -> TestResult:
    r = TestResult(
        "SV-A1.6R",
        "VAD / acoustic admission recall against real audio ground truth (retry of the refused SV-1.6)",
    )
    doc = corpus()
    annotated = [c for c in doc["cases"] if c["gt_strength"] == "audio_annotated"]
    points = [c for c in doc["cases"] if c["gt_strength"] == "audio_annotated_point"]
    checks = doc["ground_truth_checks"]

    # ---- the precondition SV-1.6 refused on ---------------------------------------------------
    unit_error = checks.get("pt_unit_max_error_s")
    r.check(
        "ground truth exists: >= 20 interval-annotated audio cases",
        len(annotated) >= 20,
        f"{len(annotated)} cases with human-annotated SOUND intervals "
        f"(+{len(points)} point-annotated)",
        n_interval=len(annotated),
        n_point=len(points),
    )
    r.check(
        "timestamp UNITS verified, not assumed (microseconds vs frame_ids/frame_rate)",
        unit_error is not None and unit_error <= 1e-3,
        f"max |timestamps/1e6 - frame_ids/frame_rate| = {unit_error} s over "
        f"{checks.get('pt_unit_samples')} interval bounds",
        max_unit_error_s=unit_error,
        bounds_checked=checks.get("pt_unit_samples"),
    )
    outside = [
        (c["case_id"], p)
        for c in annotated
        for p in (c["provenance"]["sound_positives_any"] + c["positives"])
        if c["audio"]["duration_s"] and p[1] > c["audio"]["duration_s"] + 0.5
    ]
    r.check(
        "every ground-truth interval falls inside its media's audio duration",
        not outside,
        f"{len(outside)} interval(s) past the end of the audio stream" if outside
        else f"all {sum(len(c['positives']) for c in annotated)} intervals inside the media",
        offenders=outside[:5],
    )
    no_audio = [c["case_id"] for c in annotated + points if not c.get("audio")]
    r.check(
        "every selected media file HAS an audio stream (ffprobe-probed, never assumed)",
        not no_audio,
        f"{len(annotated) + len(points)} cases probed, {len(no_audio)} without audio",
        without_audio=no_audio,
    )

    # The golden manifest's own acoustic track is the definition this corpus was told to reuse.
    # Reusing it meant reading it, and reading it surfaced a ground-truth defect of exactly the
    # class this section exists to catch.
    golden = _golden_sound_events()
    silent_anchors = [
        (event["video_id"], event["label"])
        for event in golden
        if str(event.get("label", "")).strip() in corpus_builder.NON_SOUND_LABELS
    ]
    r.check(
        "the golden `perception-test-acoustic` track's case anchors are SOUND events, not silence",
        bool(golden) and not silent_anchors,
        f"golden_sound_events() returned {len(golden)} anchor(s); "
        f"{len(silent_anchors)} carry a non-sound label: {silent_anchors}",
        golden_anchors=[(e["video_id"], e["label"]) for e in golden],
        silent_anchors=silent_anchors,
    )
    if silent_anchors:
        r.defects.append(
            Defect(
                id="GOLDEN-ACOUSTIC-MUTED-ANCHOR",
                severity="high",
                where="packages/percept-vision/eval/bench/perception_test.py:197",
                summary=(
                    "golden_sound_events() takes segments[0] (_event_from_row, "
                    "perception_test.py:172-190) as each acoustic row's ground-truth event. For "
                    "both rows in the golden `perception-test-acoustic` track "
                    f"({', '.join(v for v, _ in silent_anchors)}) segment 0 is labelled 'Muted' — "
                    "the annotators' marker for a stretch with NO audible source. label_phrase "
                    "then renders it as 'right now there is an audible muted sound or event', so "
                    "the track's recall / fusion_fp / recall_flip metrics would be scored against "
                    "SILENCE. The valid-split annotations carry no 'Muted' label, so only the "
                    "checked-in train-split sample.json rows are affected — which is exactly what "
                    "the golden manifest pins. This corpus filters NON_SOUND_LABELS instead."
                ),
                repro=(
                    "python -c \"import importlib.util,sys;"
                    "s=importlib.util.spec_from_file_location('pt','packages/percept-vision/eval/"
                    "bench/perception_test.py');m=importlib.util.module_from_spec(s);"
                    "s.loader.exec_module(m);print([(e['video_id'],e['label']) for e in "
                    "m.golden_sound_events()])\""
                ),
            )
        )

    # ---- the shipped gate's operating point ---------------------------------------------------
    shipped = admission_recall(annotated, high=SHIPPED_ONSET_HIGH, low=SHIPPED_ONSET_LOW)
    all_energies = [e for c in annotated for _, _, e in energies(c)]
    all_energies.sort()

    def _q(p: float) -> float:
        return all_energies[min(len(all_energies) - 1, int(p * len(all_energies)))]

    r.check(
        "the SHIPPED audio gate admits at least one annotated sound event",
        (shipped["hits"] or 0) > 0,
        f"AcousticOnset(high={SHIPPED_ONSET_HIGH}, low={SHIPPED_ONSET_LOW}) woke "
        f"{shipped['admissions']} time(s) across {shipped['chunks']} capture chunks and matched "
        f"{shipped['hits']}/{shipped['positives']} annotated sound events "
        f"(recall {shipped['recall']}); corpus energy p99={_q(0.99):.3g}, max={all_energies[-1]:.3g}",
        **{k: shipped[k] for k in ("recall", "hits", "positives", "admissions", "chunks", "economy")},
        corpus_energy_p50=_q(0.5),
        corpus_energy_p99=_q(0.99),
        corpus_energy_max=all_energies[-1],
    )

    # ---- the frontier -------------------------------------------------------------------------
    frontier = [
        admission_recall(annotated, high=high, low=high / 4.0) for high in ONSET_SWEEP
    ]
    for point in frontier:
        point.pop("per_case", None)
    best = max(frontier, key=lambda p: (p["recall"] or 0.0))
    usable = [
        p for p in frontier if (p["recall"] or 0.0) > 0.0 and (p["economy"] or 1.0) < 0.25
    ]
    r.check(
        "a threshold exists where the gate has non-zero recall AND spends < 25% of chunks",
        bool(usable),
        f"{len(usable)}/{len(frontier)} swept thresholds are on a usable frontier; best recall "
        f"{best['recall']:.3f} at high={best['high']:.3g} (economy {best['economy']:.4f})",
        best_high=best["high"],
        best_recall=best["recall"],
        best_economy=best["economy"],
    )

    # Label-specific recall: the gate cannot tell one sound from another, so the any-sound target
    # is the fair one; the label target is reported beside it so the gap is visible, not hidden.
    best_label = admission_recall(
        annotated, high=best["high"], low=best["high"] / 4.0, positives_key=None
    )
    best_label.pop("per_case", None)

    # ---- the amplitude/energy conversion behind the saturation --------------------------------
    converted = _amplitude_threshold_to_energy(
        SHIPPED_UTTERANCE_THRESHOLD, name="utterance_threshold"
    )
    r.check(
        "utterance_threshold is an AMPLITUDE knob squared into mean-square energy",
        converted == SHIPPED_UTTERANCE_THRESHOLD ** 2 == 1e-4,
        f"utterance_threshold={SHIPPED_UTTERANCE_THRESHOLD} reaches EnergyVad as energy "
        f"{converted:g} — two orders of magnitude below the number a reader sees",
        amplitude=SHIPPED_UTTERANCE_THRESHOLD,
        energy=converted,
    )

    speech_cases = points or annotated
    shipped_utterance = utterance_profile(
        speech_cases, threshold_amplitude=SHIPPED_UTTERANCE_THRESHOLD
    )
    r.check(
        "the shipped utterance endpointer endpoints >= 1 utterance per minute of speech material",
        (shipped_utterance["endpoints_per_minute"] or 0.0) >= 1.0,
        f"utterance_threshold={SHIPPED_UTTERANCE_THRESHOLD} -> speech_fraction "
        f"{shipped_utterance['speech_fraction']:.3f}, {shipped_utterance['endpoints']} endpoint(s) "
        f"over {shipped_utterance['media_seconds']:.0f}s "
        f"({shipped_utterance['endpoints_per_minute']:.2f}/min) on the speech-bearing slice",
        **{k: v for k, v in shipped_utterance.items() if k != "frames"},
    )

    utterance_frontier = {
        "speech_slice": [
            utterance_profile(speech_cases, threshold_amplitude=t) for t in UTTERANCE_SWEEP
        ],
        "sound_slice": [
            utterance_profile(annotated, threshold_amplitude=t) for t in UTTERANCE_SWEEP
        ],
    }

    # ---- defects -------------------------------------------------------------------------------
    if (shipped["hits"] or 0) == 0:
        r.defects.append(
            Defect(
                id="AUDIO-GATE-DEAF",
                severity="blocker",
                where="percept/harness/detectors.py:233",
                summary=(
                    f"AcousticOnset's shipped thresholds (high={SHIPPED_ONSET_HIGH}, "
                    f"low={SHIPPED_ONSET_LOW}) are mean-square ENERGY, and no capture chunk in "
                    f"{len(annotated)} annotated clips ({shipped['chunks']} chunks, "
                    f"{sum(c['duration_s'] for c in annotated):.0f}s) reaches "
                    f"{SHIPPED_ONSET_HIGH}: corpus max energy is {all_energies[-1]:.3g}. "
                    "run(audio_gate='acoustic-onset') with default audio_wake therefore NEVER "
                    "admits on this material — an audio Watch is silently deaf, with no health "
                    "event to say so."
                ),
                repro=(
                    "eval/stage-validation/sv_audio.py --only SV-A1.6R "
                    "(check: 'the SHIPPED audio gate admits at least one annotated sound event')"
                ),
            )
        )
    if (shipped_utterance["endpoints_per_minute"] or 0.0) < 1.0:
        r.defects.append(
            Defect(
                id="AUDIO-VAD-HANGOVER-UNITS",
                severity="high",
                where="percept/runtime/_vad.py:54",
                summary=(
                    "EnergyVad's defaults are dimensionless FRAME counts (hangover_frames=8, "
                    "min_speech_frames=3) ported from percept-edge/src/vad.js, but the Python "
                    f"runtime feeds it capture chunks of AudioSourceConfig.chunk_s={CAPTURE_CHUNK_S}s "
                    f"(percept/runtime/audio.py:56). The effective hangover is therefore "
                    f"{SHIPPED_HANGOVER_FRAMES * CAPTURE_CHUNK_S:.2f}s of CONTINUOUS silence and the "
                    f"minimum utterance {SHIPPED_MIN_SPEECH_FRAMES * CAPTURE_CHUNK_S:.2f}s. "
                    "Combined with utterance_threshold=0.01 arriving as energy 1e-4, continuous "
                    "narration reads as one unbroken speech run and endpoints essentially never — "
                    "the saturation observed on omnipro, reproduced here."
                ),
                repro=(
                    "eval/stage-validation/sv_audio.py --only SV-A1.6R "
                    "(measurements.utterance_frontier)"
                ),
            )
        )

    r.measurements = {
        "corpus": {
            "n_cases": doc["n_cases"],
            "by_gt_strength": doc["cases_by_gt_strength"],
            "coverage": doc["coverage"],
            "refused_datasets": doc["refused_datasets"],
            "interval_positives": sum(len(c["positives"]) for c in annotated),
            "any_sound_positives": sum(
                len(c["provenance"]["sound_positives_any"]) for c in annotated
            ),
            "media_seconds": round(sum(c["duration_s"] for c in annotated), 1),
        },
        "admission_tolerance_s": ADMISSION_TOLERANCE_S,
        "capture_chunk_s": CAPTURE_CHUNK_S,
        "shipped_operating_point": {
            k: shipped[k]
            for k in ("high", "low", "recall", "hits", "positives", "admissions", "chunks", "economy")
        },
        "onset_frontier": frontier,
        "best_frontier_point": {k: best[k] for k in ("high", "low", "recall", "economy")},
        "best_point_label_specific_recall": {
            k: best_label[k] for k in ("recall", "hits", "positives")
        },
        "corpus_energy_quantiles": {
            "p10": _q(0.10), "p50": _q(0.50), "p90": _q(0.90),
            "p99": _q(0.99), "max": all_energies[-1],
        },
        "shipped_utterance_profile": shipped_utterance,
        "utterance_frontier": utterance_frontier,
    }
    r.notes = [
        "Recall is scored against the ANY-SOUND ground truth (every annotated sound segment in the "
        "clip, merged): an energy-threshold gate cannot distinguish one sound label from another, "
        "so scoring it against a single label would measure the wrong thing. Label-specific recall "
        "at the same operating point is reported beside it.",
        f"An admission counts as a hit when a wake instant lands within {ADMISSION_TOLERANCE_S}s of "
        "the annotated interval — one capture chunk, because an onset is reported at the chunk's END.",
        "No default is proposed. The frontier is reported; picking an operating point needs a "
        "consequence model (miss cost vs. spend) this suite does not have.",
        f"The speech slice is SMALL — {len(speech_cases)} case(s), "
        f"{shipped_utterance['media_seconds']:.0f}s. The local OmniPro media is the benchmark's "
        "VISUAL subset (19 of 245 videos), so only a handful of rows with an audio-carried trigger "
        "have media on disk. Read the endpoint counts as an order of magnitude, not a rate; the "
        "speech-fraction column is the load-bearing measurement and it is stable across the slice.",
    ]
    return r


# --------------------------------------------------------------------------------------------
# SV-A2.1 — audio senses: determinism + the documented single-quantity contract
# --------------------------------------------------------------------------------------------


def _pcm16(samples: Sequence[int]) -> bytes:
    return struct.pack("<" + "h" * len(samples), *samples)


def sv_a2_1_audio_senses() -> TestResult:
    r = TestResult("SV-A2.1", "Audio senses — determinism and the one-energy-quantity contract")

    case = cases_of("perception_test_sound")[0]
    rows = capture(REPO_ROOT / case["video_path"])
    real_chunks = [row["pcm"] for row in rows[:24]]

    # ---- pcm_energy is a pure function of the bytes --------------------------------------------
    repeats = [[pcm_energy(chunk) for chunk in real_chunks] for _ in range(3)]
    r.check(
        "pcm_energy is bit-exactly repeatable on real PCM",
        repeats[0] == repeats[1] == repeats[2],
        f"{len(real_chunks)} real capture chunks x3 passes, identical float results",
        n_chunks=len(real_chunks),
    )

    # The docstring (percept/runtime/audio.py:88) claims pcm_energy IS frame_energy's definition and
    # deliberately not RMS. Verified against the actual function, on real bytes.
    deltas = []
    for chunk in real_chunks:
        samples = struct.unpack("<" + "h" * (len(chunk) // 2), chunk)
        deltas.append(abs(pcm_energy(chunk) - frame_energy([s / 32768.0 for s in samples])))
    r.check(
        "pcm_energy == runtime.frame_energy on the same signal (the documented single quantity)",
        max(deltas) == 0.0,
        f"max |pcm_energy - frame_energy| = {max(deltas):.3g} over {len(deltas)} real chunks",
        max_delta=max(deltas),
    )
    r.check(
        "pcm_energy is mean-square, NOT RMS (full-scale square wave -> 1.0)",
        pcm_energy(_pcm16([32767] * 8)) < 1.0
        and abs(pcm_energy(_pcm16([-32768] * 8)) - 1.0) == 0.0,
        f"full-negative-scale energy = {pcm_energy(_pcm16([-32768] * 8))}, "
        f"full-positive-scale = {pcm_energy(_pcm16([32767] * 8))}",
    )
    r.check(
        "pcm_energy rejects a misaligned s16le payload instead of guessing",
        _raises(lambda: pcm_energy(b"\x00\x01\x02")),
        "odd-length s16le PCM raises ValueError",
    )

    # ---- normalize_audio_unit: interval + energy derived, never invented ------------------------
    frame = AudioFrame(
        pcm=real_chunks[0], sample_rate=16000, channels=1, sample_format="s16le",
    )
    unit_a = normalize_audio_unit((10.0, frame, None), seq=0)
    unit_b = normalize_audio_unit((10.0, frame, None), seq=0)
    expected_duration = len(real_chunks[0]) / 2 / 16000
    r.check(
        "normalize_audio_unit derives the interval from the PAYLOAD's true duration",
        unit_a.t_end == 10.0
        and abs((unit_a.t_end - unit_a.t_start) - expected_duration) == 0.0,
        f"[{unit_a.t_start}, {unit_a.t_end}] for a {expected_duration}s payload anchored at t=10.0",
        t_start=unit_a.t_start, t_end=unit_a.t_end, expected_duration=expected_duration,
    )
    r.check(
        "normalize_audio_unit is deterministic and computes the missing energy from the PCM",
        (unit_a.t_start, unit_a.t_end, unit_a.energy)
        == (unit_b.t_start, unit_b.t_end, unit_b.energy)
        and unit_a.energy == pcm_energy(real_chunks[0]),
        f"energy {unit_a.energy} == pcm_energy(payload) {pcm_energy(real_chunks[0])}",
    )

    # ---- canonicalize_audio_unit: the trust boundary copies, it does not alias -------------------
    mutable = bytearray(real_chunks[1])
    live = AudioFrame(
        pcm=mutable, sample_rate=16000, channels=1, sample_format="s16le",
        t_start=0.0, t_end=len(mutable) / 2 / 16000,
    )
    candidate = AudioUnit(
        seq=0, t_start=live.t_start, t_end=live.t_end, event_t=live.t_end,
        chunk=live, energy=pcm_energy(bytes(mutable)),
    )
    canonical = canonicalize_audio_unit(
        candidate, capability=CAP, source_id="audio", source_epoch=0, source_sequence=0,
        observed_time=1.0, source_time=live.t_end,
    )
    frozen = bytes(canonical.chunk.pcm)
    mutable[:] = b"\0" * len(mutable)
    r.check(
        "canonicalize_audio_unit COPIES the payload (a later mutation cannot rewrite evidence)",
        bytes(canonical.chunk.pcm) == frozen == real_chunks[1],
        "the caller's buffer was zeroed after canonicalization; the canonical PCM is unchanged",
        sha256=hashlib.sha256(frozen).hexdigest(),
    )
    twin = canonicalize_audio_unit(
        AudioUnit(
            seq=0, t_start=live.t_start, t_end=live.t_end, event_t=live.t_end,
            chunk=AudioFrame(
                pcm=real_chunks[1], sample_rate=16000, channels=1, sample_format="s16le",
                t_start=live.t_start, t_end=live.t_end,
            ),
            energy=pcm_energy(real_chunks[1]),
        ),
        capability=CAP, source_id="audio", source_epoch=0, source_sequence=0,
        observed_time=1.0, source_time=live.t_end,
    )
    r.check(
        "canonicalization is deterministic (same bytes in -> same judge token out)",
        audio_script_token(audio_frame_from_units((canonical,), capability=CAP))
        == audio_script_token(audio_frame_from_units((twin,), capability=CAP)),
        "the frozen provider token is a pure function of the canonical PCM",
    )

    # ---- EnergyVad: the JS twin's state machine, deterministic ---------------------------------
    sequence = [e for _, _, e in energies(case)]
    clipped = [min(1.0, e) for e in sequence]
    vad = EnergyVad(threshold=1e-5, hangover_frames=8, min_speech_frames=3)
    first = [i for i, ev in enumerate(vad.push(e) for e in clipped) if ev.endpoint]
    vad.reset()
    second = [i for i, ev in enumerate(vad.push(e) for e in clipped) if ev.endpoint]
    batch = endpoint_energy_sequence(clipped, threshold=1e-5, hangover_frames=8, min_speech_frames=3)
    r.check(
        "EnergyVad: same energy sequence -> identical endpoints, and reset() re-arms exactly",
        first == second == batch,
        f"{len(first)} endpoints on {len(clipped)} real chunks; streaming == reset-replay == batch",
        endpoints=first,
    )
    r.check(
        "EnergyVad refuses an out-of-range energy instead of clamping silently",
        _raises(lambda: EnergyVad().push(1.5)) and _raises(lambda: EnergyVad().push(float("nan"))),
        "energy outside [0, 1] and non-finite energy both raise ValueError",
    )
    r.check(
        "EnergyVad rejects a non-positive hangover/min-speech configuration",
        _raises(lambda: EnergyVad(hangover_frames=0))
        and _raises(lambda: EnergyVad(min_speech_frames=0))
        and _raises(lambda: EnergyVad(threshold=1.5)),
        "the three constructor bounds all fail closed",
    )

    r.measurements = {
        "probe_case": case["case_id"],
        "real_chunks_used": len(real_chunks),
        "pcm_energy_vs_frame_energy_max_delta": max(deltas),
        "vad_endpoints_on_probe_case": len(first),
    }
    return r


def _raises(fn: Callable[[], Any]) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001 - the point is that it fails closed
        return True
    return False


# --------------------------------------------------------------------------------------------
# SV-A2.2 — audio capture on real media: determinism + economy
# --------------------------------------------------------------------------------------------


def sv_a2_2_audio_capture() -> TestResult:
    r = TestResult("SV-A2.2", "Audio capture on real media — byte determinism and economy")

    probes = cases_of("perception_test_sound")[:5]
    mismatches: list[str] = []
    contiguity: list[str] = []
    timings: list[dict[str, Any]] = []

    for case in probes:
        path = REPO_ROOT / case["video_path"]
        start = time.perf_counter()
        first = capture(path)
        elapsed = time.perf_counter() - start
        second = capture(path)
        digest_a = hashlib.sha256(b"".join(r_["pcm"] for r_ in first)).hexdigest()
        digest_b = hashlib.sha256(b"".join(r_["pcm"] for r_ in second)).hexdigest()
        if digest_a != digest_b or [r_["energy"] for r_ in first] != [r_["energy"] for r_ in second]:
            mismatches.append(case["case_id"])
        for previous, current in zip(first, first[1:]):
            if current["t_start"] != previous["t_end"] or current["seq"] != previous["seq"] + 1:
                contiguity.append(case["case_id"])
                break
        media_s = first[-1]["t_end"] if first else 0.0
        pcm_bytes = sum(len(r_["pcm"]) for r_ in first)
        timings.append(
            {
                "case_id": case["case_id"],
                "chunks": len(first),
                "media_s": round(media_s, 3),
                "pcm_bytes": pcm_bytes,
                "wall_s": round(elapsed, 4),
                "realtime_factor": round(media_s / elapsed, 1) if elapsed else None,
                "bytes_per_media_s": round(pcm_bytes / media_s) if media_s else None,
                "chunks_per_media_s": round(len(first) / media_s, 3) if media_s else None,
                "pcm_sha256": digest_a,
            }
        )

    r.check(
        "re-capturing the same media yields BYTE-IDENTICAL PCM and identical energies",
        not mismatches,
        f"{len(probes)} files captured twice; {len(mismatches)} differed",
        mismatched=mismatches,
    )
    r.check(
        "captured units are contiguous and sequentially numbered (no silent gap)",
        not contiguity,
        f"{sum(t['chunks'] for t in timings)} chunks across {len(probes)} files, "
        f"{len(set(contiguity))} file(s) with a gap or sequence jump",
        offenders=sorted(set(contiguity)),
    )
    # Cadence is exact for every FULL chunk; the final chunk of a file is a short read at EOF and
    # is legitimately shorter. Asserting chunks/second instead would fail on that tail alone.
    interior_durations: set[float] = set()
    tail_too_long: list[str] = []
    for case in probes:
        rows = capture(REPO_ROOT / case["video_path"])
        interior_durations.update(round(r_["t_end"] - r_["t_start"], 9) for r_ in rows[:-1])
        if rows and (rows[-1]["t_end"] - rows[-1]["t_start"]) > CAPTURE_CHUNK_S + 1e-9:
            tail_too_long.append(case["case_id"])
    r.check(
        "every full capture chunk spans exactly the shipped chunk_s (tail chunk may be shorter)",
        interior_durations == {round(CAPTURE_CHUNK_S, 9)} and not tail_too_long,
        f"interior chunk durations {sorted(interior_durations)} against chunk_s={CAPTURE_CHUNK_S}; "
        f"{len(tail_too_long)} file(s) with an over-long tail chunk",
        interior_durations=sorted(interior_durations),
        overlong_tails=tail_too_long,
    )
    bytes_per_s = {t["bytes_per_media_s"] for t in timings}
    expected_bytes = AudioSourceConfig().sample_rate * AudioSourceConfig().channels * 2
    r.check(
        "one second of listening costs exactly one second of 16 kHz mono s16 PCM",
        bytes_per_s == {expected_bytes},
        f"observed {sorted(bytes_per_s)} B/s against {expected_bytes} B/s "
        f"({AudioSourceConfig().sample_rate} Hz x {AudioSourceConfig().channels} ch x 2 B)",
        observed=sorted(bytes_per_s), expected=expected_bytes,
    )
    slowest = min(t["realtime_factor"] or 0.0 for t in timings)
    r.check(
        "capture keeps up with real time by a wide margin (>= 20x)",
        slowest >= 20.0,
        f"slowest observed {slowest:.0f}x realtime across {len(probes)} files",
        realtime_factors=[t["realtime_factor"] for t in timings],
    )

    window = AudioWindow(4.0)
    r.check(
        "the retention window is bounded by seconds AND by unit/byte caps",
        window._max_units == 4096 and window._max_bytes == 16 * 1024 * 1024,
        f"AudioWindow(4.0) -> max_units={window._max_units}, max_bytes={window._max_bytes}; "
        f"a 4 s window at {expected_bytes} B/s holds {4 * expected_bytes} B, "
        f"{100 * 4 * expected_bytes / window._max_bytes:.2f}% of the byte cap",
        max_units=window._max_units, max_bytes=window._max_bytes,
    )

    r.measurements = {
        "per_file": timings,
        "capture_chunk_s": CAPTURE_CHUNK_S,
        "bytes_per_media_second": expected_bytes,
        "window_seconds_default": 4.0,
    }
    return r


# --------------------------------------------------------------------------------------------
# audio Watch driving harness (the REAL public path)
# --------------------------------------------------------------------------------------------

RATE = 16_000
UNIT_S = CAPTURE_CHUNK_S
UNIT_SAMPLES = int(RATE * UNIT_S)

YES = Verdict(satisfied=True, confidence=0.95, evidence="scripted: the sound is present")
NO = Verdict(satisfied=False, confidence=0.95, evidence="scripted: the sound is absent")


def synth_unit(seq: int, amplitude: int, *, start: float | None = None) -> AudioUnit:
    """One capture-shaped unit of constant-amplitude PCM — energy is exactly (a/32768)^2."""
    start = seq * UNIT_S if start is None else start
    pcm = _pcm16([amplitude] * UNIT_SAMPLES)
    chunk = AudioFrame(
        pcm=pcm, sample_rate=RATE, channels=1, sample_format="s16le",
        t_start=start, t_end=start + UNIT_S,
    )
    return AudioUnit(
        seq=seq, t_start=chunk.t_start, t_end=chunk.t_end, event_t=chunk.t_end,
        chunk=chunk, energy=min(1.0, (amplitude / 32768.0) ** 2),
    )


def real_units(case: dict[str, Any], count: int) -> tuple[list[AudioUnit], list[dict[str, Any]]]:
    """The first ``count`` capture chunks of a REAL corpus media file, as ordered AudioUnits."""
    rows = capture(REPO_ROOT / case["video_path"])[:count]
    units = []
    for row in rows:
        chunk = AudioFrame(
            pcm=row["pcm"], sample_rate=row["sample_rate"], channels=row["channels"],
            sample_format=row["sample_format"], t_start=row["t_start"], t_end=row["t_end"],
        )
        units.append(
            AudioUnit(
                seq=row["seq"], t_start=row["t_start"], t_end=row["t_end"],
                event_t=row["t_end"], chunk=chunk, energy=row["energy"],
            )
        )
    return units, rows


def canon(unit: AudioUnit) -> AudioUnit:
    return canonicalize_audio_unit(
        unit, capability=CAP, source_id="audio", source_epoch=0,
        source_sequence=unit.seq, observed_time=10.0, source_time=unit.t_end,
    )


def token_for(units: Sequence[AudioUnit]) -> str:
    """The scripted-judge key the compiled audio path derives from an evidence tuple.

    Built with the SDK's own ``audio_frame_from_units``/``audio_script_token``, so a mismatch
    between what the harness predicts and what the runtime assembles shows up as a MISSING script
    entry (no fire) rather than as a silently-passing test.
    """
    return audio_script_token(
        audio_frame_from_units(tuple(canon(u) for u in units), capability=CAP)
    )


def audio_core(
    script: dict[str, Verdict],
    *,
    refractory_s: float = 0.0,
    condition: str = "right now there is an audible sound or event",
    goal_id: str = "audio",
) -> Percept:
    settings = load_settings()
    settings = dataclasses.replace(
        settings, gate=dataclasses.replace(settings.gate, refractory_s=refractory_s)
    )
    core = Percept.create(
        vision=FakeVision(script=dict(script)), discover_plugins=False, settings=settings
    )
    receipt = apply_watch_plan(
        core, WatchPlan(id=goal_id, condition=condition, sense="audio")
    )
    if receipt.status != "applied":
        raise RuntimeError(f"audio WatchPlan not applied: {receipt.status}")
    return core


@dataclass
class AudioRun:
    incidents: list
    health: list
    trace: list
    stats: RuntimeStats


def drive_audio(core: Percept, units: Iterable, **kwargs: Any) -> AudioRun:
    """Run the REAL public audio path and collect every observable channel."""
    health: list = []
    trace: list = []
    stats: RuntimeStats = kwargs.pop("stats", None) or RuntimeStats()
    incidents: list = []

    async def _main() -> None:
        async for incident in run(
            core,
            source=None,
            audio_source=units,
            audio_settle_s=kwargs.pop("audio_settle_s", 0.0),
            audio_gate=kwargs.pop("audio_gate", "acoustic-onset"),
            audio_wake=kwargs.pop("audio_wake", {"high": 0.2, "low": 0.05}),
            judge_dispatch=kwargs.pop("judge_dispatch", "inline"),
            stats=stats,
            on_health=health.append,
            on_trace=trace.append,
            **kwargs,
        ):
            incidents.append(incident)

    asyncio.run(_main())
    return AudioRun(incidents=incidents, health=health, trace=trace, stats=stats)


# --------------------------------------------------------------------------------------------
# SV-A4.1 — gate decision + state on the audio path
# --------------------------------------------------------------------------------------------

LOUD = int(0.5 * 32767)
QUIET = int(0.02 * 32767)


def _wake_pattern(n: int, wake_at: Sequence[int]) -> list[AudioUnit]:
    amplitudes = [QUIET] * n
    for index in wake_at:
        amplitudes[index] = LOUD
    return [synth_unit(i, a) for i, a in enumerate(amplitudes)]


def sv_a4_1_gate_decision() -> TestResult:
    r = TestResult("SV-A4.1", "Audio cognition — rising edge, refractory, distinct Incidents")

    wake_at = (2, 6, 10, 14)
    units = _wake_pattern(16, wake_at)
    # Each acoustic wake admits the whole retained window; alternating satisfied/unsatisfied
    # verdicts make every second admission a genuine rising edge.
    script = {
        token_for(units[: index + 1]): (YES if position % 2 == 0 else NO)
        for position, index in enumerate(wake_at)
    }

    out = drive_audio(audio_core(script), units)
    fire_ts = [i.ts for i in out.incidents]
    reasons = [
        e.data.get("gate_reason") for e in out.trace if getattr(e, "stage", "") == "gate.classified"
    ]
    r.check(
        "every acoustic wake reaches the judge exactly once",
        (out.stats.escalations, out.stats.judge_calls, out.stats.perceive_dispatches)
        == (len(wake_at), len(wake_at), len(wake_at)),
        f"{len(wake_at)} wakes -> escalations={out.stats.escalations}, "
        f"judge_calls={out.stats.judge_calls}, perceive_dispatches={out.stats.perceive_dispatches}",
        escalations=out.stats.escalations, judge_calls=out.stats.judge_calls,
    )
    r.check(
        "the audio gate fires on the RISING EDGE and only there",
        fire_ts == [units[2].t_end, units[10].t_end] and reasons == [
            "rising_edge", "not_satisfied", "rising_edge", "not_satisfied"
        ],
        f"fired at {fire_ts}; gate reasons {reasons}",
        fire_ts=fire_ts, gate_reasons=reasons,
    )

    # a held-true occurrence must fire once and then report already_fired
    held_units = _wake_pattern(16, wake_at)
    held_script = {token_for(held_units[: i + 1]): YES for i in wake_at}
    held = drive_audio(audio_core(held_script), held_units)
    held_reasons = [
        e.data.get("gate_reason")
        for e in held.trace
        if getattr(e, "stage", "") == "gate.classified"
    ]
    r.check(
        "a held-true audio occurrence fires exactly ONCE",
        len(held.incidents) == 1 and held_reasons.count("already_fired") == len(wake_at) - 1,
        f"{len(held.incidents)} incident(s); gate reasons {held_reasons}",
        n_incidents=len(held.incidents), reasons=held_reasons,
    )

    # distinctness of repeated occurrences
    ids = [i.id for i in out.incidents]
    judgment_ids = [i.judgment_event_id for i in out.incidents]
    evidence_seqs = [i.evidence_seq for i in out.incidents]
    evidence_hashes = [
        hashlib.sha256(b"".join(bytes(f.chunk.pcm) for f in i.snippet.frames)).hexdigest()
        for i in out.incidents
    ]
    r.check(
        "a repeated audio occurrence yields a DISTINCT Incident",
        len(set(ids)) == len(ids) == 2
        and len(set(judgment_ids)) == len(judgment_ids)
        and len(set(evidence_seqs)) == len(evidence_seqs)
        and len(set(evidence_hashes)) == len(evidence_hashes),
        f"{len(ids)} incidents; {len(set(ids))} distinct ids, "
        f"{len(set(judgment_ids))} distinct judgment_event_ids, "
        f"{len(set(evidence_seqs))} distinct evidence_seqs, "
        f"{len(set(evidence_hashes))} distinct evidence hashes",
        ids=ids, judgment_event_ids=judgment_ids, evidence_seqs=evidence_seqs,
    )

    # refractory
    refractory_s = 2.5
    ref_units = _wake_pattern(16, wake_at)
    ref_script = {
        token_for(ref_units[: index + 1]): (YES if position % 2 == 0 else NO)
        for position, index in enumerate(wake_at)
    }
    ref = drive_audio(audio_core(ref_script, refractory_s=refractory_s), ref_units)
    ref_ts = [i.ts for i in ref.incidents]
    suppressed = [h for h in ref.health if h.kind == "gate_suppressed"]
    r.check(
        "refractory suppresses an audio re-fire inside the window, with a typed receipt",
        ref_ts == [units[2].t_end]
        and len(suppressed) == 1
        and (suppressed[0].data or {}).get("reason") == "refractory"
        and (suppressed[0].data or {}).get("at") == units[10].t_end,
        f"refractory_s={refractory_s}: fired at {ref_ts}, suppressed "
        f"{[( (h.data or {}).get('reason'), (h.data or {}).get('at')) for h in suppressed]}",
        fire_ts=ref_ts,
        suppressed=[(h.data or {}).get("reason") for h in suppressed],
    )
    r.check(
        "the second occurrence WOULD have fired without refractory (the suppression is the cause)",
        len(out.incidents) == 2 and len(ref.incidents) == 1,
        f"same wakes and verdicts: {len(out.incidents)} fires at refractory_s=0.0, "
        f"{len(ref.incidents)} at refractory_s={refractory_s}",
    )

    r.measurements = {
        "wake_indices": list(wake_at),
        "unit_seconds": UNIT_S,
        "baseline_fire_ts": fire_ts,
        "refractory_s": refractory_s,
        "refractory_fire_ts": ref_ts,
        "gate_reasons": reasons,
        "held_true_reasons": held_reasons,
    }
    return r


# --------------------------------------------------------------------------------------------
# SV-A4.7 — evidence integrity on REAL audio: hash in, hash out
# --------------------------------------------------------------------------------------------


def _wav_data_chunk(raw: bytes) -> bytes:
    """The PCM payload of a RIFF/WAVE container, located by walking its chunks."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE container")
    offset = 12
    while offset + 8 <= len(raw):
        tag = raw[offset:offset + 4]
        size = int.from_bytes(raw[offset + 4:offset + 8], "little")
        body = raw[offset + 8:offset + 8 + size]
        if tag == b"data":
            return body
        offset += 8 + size + (size % 2)
    raise ValueError("RIFF/WAVE container has no data chunk")


def sv_a4_7_evidence_integrity() -> TestResult:
    r = TestResult("SV-A4.7", "Audio evidence integrity — an Incident resolves to the exact PCM fed")

    case = max(cases_of("perception_test_sound"), key=lambda c: max(e for _, _, e in energies(c)))
    units, rows = real_units(case, 12)
    fed_hashes = [hashlib.sha256(bytes(u.chunk.pcm)).hexdigest() for u in units]

    # chain of custody: the bytes under test came from the file the corpus pinned
    media_path = REPO_ROOT / case["video_path"]
    media_sha = corpus_builder.sha256_file(media_path)
    r.check(
        "the media under test is the exact file the corpus pinned",
        media_sha == case["video_sha256"],
        f"{case['video_path']} sha256 {media_sha[:16]}... matches the corpus pin",
        case_id=case["case_id"], media_sha256=media_sha,
    )

    # a wake threshold derived from THIS file's own energies, so an onset is guaranteed
    peak = max(u.energy for u in units)
    floor = min(u.energy for u in units)
    high_energy = (peak + floor) / 2.0
    script = {token_for(units[: i + 1]): YES for i in range(len(units))}
    out = drive_audio(
        audio_core(script),
        units,
        audio_wake={"high": math.sqrt(high_energy), "low": math.sqrt(max(floor, 1e-12))},
    )

    r.check(
        "the real-audio Watch produced at least one Incident to audit",
        len(out.incidents) >= 1,
        f"{len(out.incidents)} incident(s) from {len(units)} real capture chunks of "
        f"{case['case_id']}",
        n_incidents=len(out.incidents),
    )
    if not out.incidents:
        r.error = "no Incident produced; evidence integrity could not be audited"
        return r

    incident = out.incidents[0]
    snippet = incident.snippet
    got_hashes = [hashlib.sha256(bytes(f.chunk.pcm)).hexdigest() for f in snippet.frames]
    got_seqs = [f.seq for f in snippet.frames]
    expected_seqs = list(range(len(got_seqs)))

    r.check(
        "the Incident carries an AUDIO snippet, not a substituted modality",
        snippet is not None and snippet.kind == "audio",
        f"snippet.kind={getattr(snippet, 'kind', None)}",
    )
    r.check(
        "every evidence frame's PCM hashes to the EXACT bytes that were fed",
        got_hashes == fed_hashes[: len(got_hashes)] and got_seqs == expected_seqs,
        f"{len(got_hashes)} evidence frames; hashes {'match' if got_hashes == fed_hashes[:len(got_hashes)] else 'DIFFER'}; "
        f"seqs {got_seqs}",
        fed_sha256=fed_hashes[: len(got_hashes)],
        evidence_sha256=got_hashes,
    )
    r.check(
        "the concatenated evidence is byte-identical to the concatenated capture",
        hashlib.sha256(b"".join(bytes(f.chunk.pcm) for f in snippet.frames)).hexdigest()
        == hashlib.sha256(b"".join(r_["pcm"] for r_ in rows[: len(got_hashes)])).hexdigest(),
        "no re-encode, no resample, no substitution between capture and Incident",
    )
    r.check(
        "the evidence interval equals the real media interval it was taken from",
        (snippet.t_start, snippet.t_end)
        == (rows[0]["t_start"], rows[len(got_hashes) - 1]["t_end"]),
        f"snippet [{snippet.t_start}, {snippet.t_end}] vs capture "
        f"[{rows[0]['t_start']}, {rows[len(got_hashes) - 1]['t_end']}]",
    )
    # The judge saw the same bytes the Incident carries. The provider payload is a canonical
    # 44-byte-header PCM WAV (`_wav_bytes`), so the comparison is against its data chunk — parsed
    # from the RIFF structure rather than assuming a fixed offset.
    judged = audio_frame_from_units(tuple(canon(u) for u in units[: len(got_hashes)]), capability=CAP)
    judged_pcm = _wav_data_chunk(bytes(judged.encoded)) if judged.encoded else bytes(judged.pcm)
    evidence_pcm = b"".join(bytes(f.chunk.pcm) for f in snippet.frames)
    r.check(
        "what the judge was shown and what the Incident carries are the SAME bytes",
        judged_pcm == evidence_pcm
        and judged.sample_rate == rows[0]["sample_rate"]
        and judged.channels == rows[0]["channels"],
        f"provider payload is {judged.mime} ({len(judged.encoded or b'')} B); its PCM data chunk "
        f"{'matches' if judged_pcm == evidence_pcm else 'DIFFERS from'} the retained evidence "
        f"({len(evidence_pcm)} B) at {judged.sample_rate} Hz x {judged.channels} ch",
        judged_sha256=hashlib.sha256(judged_pcm).hexdigest(),
        evidence_sha256_concat=hashlib.sha256(evidence_pcm).hexdigest(),
    )
    r.check(
        "evidence energy is the real signal, not a placeholder",
        [f.energy for f in snippet.frames] == [rows[i]["energy"] for i in range(len(got_hashes))]
        and any(f.energy > 0 for f in snippet.frames),
        f"evidence energies {[round(f.energy, 8) for f in snippet.frames][:6]}...",
    )

    r.measurements = {
        "case_id": case["case_id"],
        "media_sha256": media_sha,
        "units_fed": len(units),
        "evidence_frames": len(got_hashes),
        "evidence_interval": [snippet.t_start, snippet.t_end],
        "evidence_sha256": hashlib.sha256(
            b"".join(bytes(f.chunk.pcm) for f in snippet.frames)
        ).hexdigest(),
        "wake_high_energy": high_energy,
    }
    return r


# --------------------------------------------------------------------------------------------
# SV-A4.8 — drop reconciliation on the audio path
# --------------------------------------------------------------------------------------------


def sv_a4_8_drop_reconciliation() -> TestResult:
    r = TestResult("SV-A4.8", "Audio drop reconciliation — units in == admitted + suppressed + receipted")

    wake_at = (2, 6, 10)
    valid = _wake_pattern(12, wake_at)
    malformed = [("not-a-tuple",), 42]           # two carriers that cannot be normalized
    offered = list(valid) + malformed

    suppress_window = range(4, 9)                # suppress five consecutive VALID units
    seen = [0]

    def muted() -> bool:
        seen[0] += 1
        return seen[0] - 1 in suppress_window

    script = {token_for(valid[: i + 1]): YES for i in wake_at}
    out = drive_audio(audio_core(script), offered, muted=muted)

    rejected = [h for h in out.health if h.kind == "audio_unit_rejected"]
    capped = [h for h in out.health if h.kind == "audio_evidence_capped"]
    ended = next((h for h in out.health if h.kind == "runtime_ended"), None)
    coverage = ((ended.data or {}).get("source_coverage") or {}).get("audio") or {}

    n_suppressed = len([i for i in suppress_window if i < len(valid)])
    accounted = out.stats.audio_chunks + len(rejected)
    unaccounted = len(offered) - accounted

    r.check(
        "every malformed carrier is dropped WITH a typed receipt",
        len(rejected) == len(malformed),
        f"{len(malformed)} malformed carriers -> {len(rejected)} audio_unit_rejected event(s)",
        receipts=[(h.data or {}).get("error", "")[:80] for h in rejected],
    )
    r.check(
        "accepted units are counted exactly once",
        out.stats.audio_chunks == len(valid) - n_suppressed,
        f"{len(valid)} valid units, {n_suppressed} suppressed -> audio_chunks="
        f"{out.stats.audio_chunks}",
        audio_chunks=out.stats.audio_chunks,
    )
    r.check(
        "admissions reconcile with judge dispatches and goal admissions",
        out.stats.escalations == out.stats.judge_calls == out.stats.perceive_dispatches
        == out.stats.goal_admissions,
        f"escalations={out.stats.escalations}, judge_calls={out.stats.judge_calls}, "
        f"perceive_dispatches={out.stats.perceive_dispatches}, "
        f"goal_admissions={out.stats.goal_admissions}",
    )
    r.check(
        "the end-of-run coverage report accounts for every unit offered to the runtime",
        unaccounted == 0,
        f"{len(offered)} units offered; audio_chunks={out.stats.audio_chunks} + "
        f"receipts={len(rejected)} = {accounted}; {unaccounted} unit(s) vanished with no counter "
        f"and no health event. runtime_ended reports {coverage}",
        offered=len(offered),
        audio_chunks=out.stats.audio_chunks,
        receipts=len(rejected),
        suppressed_expected=n_suppressed,
        unaccounted=unaccounted,
        source_coverage=coverage,
    )
    r.check(
        "mute/pause suppression emits a receipt naming how many units it swallowed",
        any(h.kind in ("audio_suppressed", "audio_units_suppressed") for h in out.health),
        f"health kinds observed: {sorted({h.kind for h in out.health})}",
        health_kinds=sorted({h.kind for h in out.health}),
    )

    # the provider-cap path DOES receipt its drops — shown here as the contrast
    cap_units = [synth_unit(0, QUIET), synth_unit(1, LOUD)]
    cap_core = audio_core({token_for(cap_units[:2]): YES, token_for(cap_units[1:2]): YES})
    original = cap_core._audio_judge_bindings["audio"]
    bounded = dataclasses.replace(
        original,
        capability=AudioInputCapability(
            modes=frozenset({"audio_only", "fused"}),
            max_input_bytes=MAX_AUDIO_INPUT_BYTES,
            max_provider_bytes=len(cap_units[1].chunk.pcm) + 44,
        ),
    )
    cap_core._audio_judge_bindings["audio"] = bounded
    cap_core._audio_judge_binding_tokens["audio"] = bounded.binding_token
    cap_out = drive_audio(cap_core, cap_units)
    cap_events = [h for h in cap_out.health if h.kind == "audio_evidence_capped"]
    r.check(
        "evidence trimmed by the provider byte cap is dropped WITH a receipt",
        len(cap_events) == 1 and (cap_events[0].data or {}).get("dropped_units") == 1,
        f"{len(cap_events)} audio_evidence_capped event(s): "
        f"{[(h.data or {}).get('dropped_units') for h in cap_events]}",
        events=[_jsonable(h.data) for h in cap_events],
    )

    if unaccounted:
        r.defects.append(
            Defect(
                id="AUDIO-SUPPRESSED-NO-RECEIPT",
                severity="high",
                where="percept/runtime/worker.py:2760",
                summary=(
                    "In the compiled audio-only path, a unit arriving while muted()/paused() is "
                    "`continue`d with no counter increment and no RuntimeHealthEvent. "
                    f"{unaccounted} of {len(offered)} units offered in this run are therefore "
                    "invisible: RuntimeStats has no audio-suppressed field (only frames_dropped, "
                    "which is the video path), SourceTracker never sees them "
                    f"(runtime_ended source_coverage reports excluded={coverage.get('excluded')}), "
                    "and no health event names them. Units in != admitted + suppressed + "
                    "dropped-with-receipt: the suppressed term has no term."
                ),
                repro=(
                    "eval/stage-validation/sv_audio.py --only SV-A4.8 "
                    "(check: 'the end-of-run coverage report accounts for every unit offered')"
                ),
            )
        )

    r.measurements = {
        "units_offered": len(offered),
        "valid_units": len(valid),
        "malformed_carriers": len(malformed),
        "suppressed_units": n_suppressed,
        "audio_chunks": out.stats.audio_chunks,
        "rejection_receipts": len(rejected),
        "evidence_capped_receipts": len(capped),
        "unaccounted_units": unaccounted,
        "escalations": out.stats.escalations,
        "judge_calls": out.stats.judge_calls,
        "runtime_ended_source_coverage": coverage,
        "health_kinds": sorted({h.kind for h in out.health}),
    }
    return r


# --------------------------------------------------------------------------------------------
# SV-A4.9 — what bounds the compiled audio judge path
# --------------------------------------------------------------------------------------------


def sv_a4_9_boundedness() -> TestResult:
    r = TestResult("SV-A4.9", "Audio judge boundedness — what bounds the compiled audio dispatch")

    from percept.backends.judge_pool import JudgeCapability, JudgePool, JudgeRoute

    capability = JudgeCapability(
        evidence_shapes=frozenset({"audio", "fused", "single_image"}),
        latency_slo_s=1.0,
        modes=frozenset({"reflex"}),
    )
    pool = JudgePool([JudgeRoute(name="fake", adapter=FakeVision(), capability=capability)],
                     mode="reflex")
    pool_error: str | None = None
    try:
        prepare_audio_judge_binding(pool, mode="audio_only", application_key="sv-audio")
    except Exception as exc:  # noqa: BLE001 - the rejection IS the observation
        pool_error = f"{type(exc).__name__}: {exc}"

    pool_core = Percept.create(vision=pool, discover_plugins=False)
    receipt = apply_watch_plan(
        pool_core, WatchPlan(id="audio", condition="a sound occurs", sense="audio")
    )

    r.check(
        "a JudgePool that DECLARES the 'audio' evidence shape can back a compiled audio Watch",
        pool_error is None and receipt.status == "applied",
        f"JudgePool declares evidence_shapes={sorted(capability.evidence_shapes)} yet "
        f"prepare_audio_judge_binding raised [{pool_error}] and apply_watch_plan returned "
        f"{receipt.status!r}; the compiled audio path accepts only "
        f"{sorted(k.__qualname__ for k in _trusted_audio_judges())}",
        pool_error=pool_error,
        apply_status=receipt.status,
        declared_shapes=sorted(capability.evidence_shapes),
    )

    # ---- what DOES bound it --------------------------------------------------------------------
    binding = audio_core({})._audio_judge_bindings["audio"]
    caps = binding.capability
    r.check(
        "the audio path is bounded by explicit byte caps on input and provider payload",
        caps.max_input_bytes == MAX_AUDIO_INPUT_BYTES == 16 * 1024 * 1024
        and caps.max_provider_bytes == MAX_AUDIO_PROVIDER_BYTES == 16 * 1024 * 1024,
        f"AudioInputCapability(max_input_bytes={caps.max_input_bytes}, "
        f"max_provider_bytes={caps.max_provider_bytes}) — "
        f"{caps.max_provider_bytes / (RATE * 2):.0f}s of 16 kHz mono s16 audio",
        max_input_bytes=caps.max_input_bytes,
        max_provider_bytes=caps.max_provider_bytes,
        provider_seconds=caps.max_provider_bytes / (RATE * 2),
    )
    window = AudioWindow(4.0)
    r.check(
        "the retained audio timeline is bounded by seconds, unit count AND bytes",
        window._seconds == 4.0 and window._max_units == 4096 and window._max_bytes == 16 * 1024 * 1024,
        f"AudioWindow(seconds={window._seconds}, max_units={window._max_units}, "
        f"max_bytes={window._max_bytes})",
    )

    # the async scheduler — the audio path's actual concurrency bound
    import inspect as _inspect

    signature = _inspect.signature(_LegacyJudgeScheduler.__init__)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name in ("stall_after_s", "backpressure_after", "decision_deadline_s")
    }
    scheduler = _LegacyJudgeScheduler(
        audio_core({}), "camera", RuntimeStats(), perceive=lambda *a, **k: None
    )
    r.check(
        "compiled audio at the DEFAULT judge_dispatch='async' gets a single-flight scheduler",
        scheduler._active is None and scheduler._pending is None
        and not isinstance(scheduler._pending, (list, tuple, dict, set)),
        "in-flight and pending are single scalar slots (J<=1, P<=1 by construction), with "
        f"stall/backpressure/deadline defaults {defaults}",
        scheduler_defaults=_jsonable(defaults),
    )

    # the inline path has no such bound: prove it by timing a slow judge on the capture loop
    class SlowFake(FakeVision):
        pass

    delay_s = 0.25
    units = _wake_pattern(8, (2, 6))
    slow_script = {token_for(units[: i + 1]): YES for i in (2, 6)}
    slow_core = audio_core(slow_script)
    original_perceive = slow_core.perceive_audio_judged

    async def slow_perceive(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(delay_s)
        return await original_perceive(*args, **kwargs)

    slow_core.perceive_audio_judged = slow_perceive  # type: ignore[method-assign]
    started = time.perf_counter()
    drive_audio(slow_core, units, judge_dispatch="inline")
    inline_elapsed = time.perf_counter() - started
    r.check(
        "judge_dispatch='inline' makes the audio capture loop WAIT on cognition (documented, not a bug)",
        inline_elapsed >= 2 * delay_s,
        f"2 admissions x {delay_s}s judge latency stalled the pump for {inline_elapsed:.2f}s — "
        "the inline path has no deadline; the bound comes only from judge_dispatch='async'",
        inline_elapsed_s=round(inline_elapsed, 3),
    )

    if pool_error is not None:
        r.defects.append(
            Defect(
                id="AUDIO-JUDGEPOOL-REJECTED",
                severity="medium",
                where="percept/audio_input.py:1036",
                summary=(
                    "prepare_audio_judge_binding accepts only an exact percept.backends.fake."
                    "FakeVision or percept.backends.gemini.GeminiVision instance, so a JudgePool — "
                    "including one whose JudgeCapability explicitly declares the 'audio' and "
                    "'fused' evidence shapes (percept/backends/judge_pool.py:35) — is rejected and "
                    "apply_watch_plan returns 'rejected'. The compiled audio path therefore never "
                    "inherits the pool's per-route capacity (max_inflight), latency SLO/deadline, "
                    "hedging or JudgeRouteReceipts. It is NOT unbounded: the byte caps, the "
                    "AudioWindow caps and the default judge_dispatch='async' single-flight "
                    "scheduler bound it — but the bound is the runtime's, not the pool's, and "
                    "there is no receipt trail per route."
                ),
                repro="eval/stage-validation/sv_audio.py --only SV-A4.9",
            )
        )

    r.measurements = {
        "judgepool_error": pool_error,
        "judgepool_apply_status": receipt.status,
        "trusted_audio_judges": sorted(
            f"{k.__module__}.{k.__qualname__}" for k in _trusted_audio_judges()
        ),
        "max_input_bytes": caps.max_input_bytes,
        "max_provider_bytes": caps.max_provider_bytes,
        "provider_cap_seconds_16k_mono": caps.max_provider_bytes / (RATE * 2),
        "scheduler_defaults": _jsonable(defaults),
        "inline_stall_s": round(inline_elapsed, 3),
        "bounds": [
            "AudioInputCapability.max_input_bytes / max_provider_bytes (16 MiB each)",
            "AudioWindow seconds + max_units(4096) + max_bytes(16 MiB)",
            "_LegacyJudgeScheduler single-flight + one-latest-pending (judge_dispatch='async')",
            "judge_stall_after_s=20.0, judge_backpressure_after=3, judge_decision_deadline_s=None",
        ],
        "not_inherited_from_judgepool": [
            "JudgeCapability.max_inflight (per-route capacity)",
            "JudgeCapability.latency_slo_s / JudgePool deadline_s + hedge_after_s",
            "JudgeRouteReceipt trail",
        ],
    }
    return r


def _trusted_audio_judges() -> Any:
    import percept.audio_input as module

    return module._TRUSTED_AUDIO_JUDGES


# --------------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------------

TESTS: list[tuple[str, Callable[[], TestResult]]] = [
    ("SV-A1.6R", sv_a1_6r_vad_admission),
    ("SV-A2.1", sv_a2_1_audio_senses),
    ("SV-A2.2", sv_a2_2_audio_capture),
    ("SV-A4.1", sv_a4_1_gate_decision),
    ("SV-A4.7", sv_a4_7_evidence_integrity),
    ("SV-A4.8", sv_a4_8_drop_reconciliation),
    ("SV-A4.9", sv_a4_9_boundedness),
]


def runtime_identity() -> dict:
    module = Path(percept.__file__).resolve()
    return {
        "percept_version": getattr(percept, "__version__", None),
        "percept_path": str(module),
        "in_site_packages": "site-packages" in module.parts,
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def assert_runtime_under_test() -> None:
    """Refuse to measure a source tree. The runtime under test is an INSTALLED wheel."""
    identity = runtime_identity()
    if not identity["in_site_packages"]:
        raise SystemExit(
            "sv_audio.py measures an INSTALLED percept wheel, not a source checkout.\n"
            f"  percept resolved to {identity['percept_path']}\n"
            "Create a venv, pip install the wheel under test, and run this file with that "
            "interpreter."
        )


def run_all(only: list[str] | None = None) -> dict:
    selected = [(name, fn) for name, fn in TESTS if not only or name in only]
    results: list[TestResult] = []
    for name, fn in selected:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 - a crashed test is a result, never a silence
            failed = TestResult(name, fn.__doc__ or name)
            failed.error = f"{type(exc).__name__}: {exc}"
            results.append(failed)

    doc = corpus()
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "suite": "sv_audio",
        "title": "Audio stage validation (Stage 1 admission, Stage 2 senses, Stage 4 cognition)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": runtime_identity(),
        "totals": counts,
        "corpus": doc,
        "defects": [d.to_json() for r in results for d in r.defects],
        "tests": [r.to_json() for r in results],
    }


def _print_table(report: dict) -> None:
    print()
    print(f"{'TEST':<11} {'STATUS':<7} {'CHECKS':<9} TITLE")
    print("-" * 108)
    for test in report["tests"]:
        passed = sum(1 for c in test["checks"] if c["passed"])
        print(f"{test['id']:<11} {test['status']:<7} {passed}/{len(test['checks']):<7} {test['title'][:70]}")
    print("-" * 108)
    totals = report["totals"]
    print(f"PASS {totals.get('PASS', 0)}  FAIL {totals.get('FAIL', 0)}  ERROR {totals.get('ERROR', 0)}")
    for test in report["tests"]:
        for check in test["checks"]:
            if not check["passed"]:
                print(f"  FAIL  {test['id']}  {check['name']}")
                print(f"        {check['detail']}")
        if test["error"]:
            print(f"  ERROR {test['id']}  {test['error']}")
    if report["defects"]:
        print()
        print("DEFECTS")
        for defect in report["defects"]:
            print(f"  [{defect['severity']}] {defect['id']}  {defect['where']}")
            print(f"      {defect['summary'][:160]}")


def render_markdown(report: dict) -> str:
    out: list[str] = []
    add = out.append
    doc = report["corpus"]
    totals = report["totals"]

    add("# SV-AUDIO — audio stage validation")
    add("")
    add(f"Generated: {report['generated']}")
    add(f"Runtime under test: `percept {report['runtime']['percept_version']}` "
        f"at `{report['runtime']['percept_path']}`")
    add("")
    add(f"**PASS {totals.get('PASS', 0)} · FAIL {totals.get('FAIL', 0)} · "
        f"ERROR {totals.get('ERROR', 0)}**")
    add("")
    add("| Test | Status | Checks | Title |")
    add("|---|---|---|---|")
    for test in report["tests"]:
        passed = sum(1 for c in test["checks"] if c["passed"])
        add(f"| `{test['id']}` | **{test['status']}** | {passed}/{len(test['checks'])} | {test['title']} |")
    add("")

    # ---- corpus ------------------------------------------------------------------------------
    add("## The corpus")
    add("")
    add(f"**{doc['n_cases']} audio-bearing cases**, seeded selection (`seed={doc['selection_seed']}`), "
        "every media file ffprobe-verified to carry an audio stream.")
    add("")
    add("| Source | Built | Ground-truth strength | What the positives are |")
    add("|---|---|---|---|")
    strength_note = {
        "perception_test_sound": ("audio_annotated", "human-annotated SOUND intervals (`sound_localisation`)"),
        "omnipro_audio": ("audio_annotated_point", "instants a speech/sound-carried event became answerable"),
        "charades_audible": ("proxy", "VISUAL action intervals whose label names an audible event"),
    }
    for name, info in doc["coverage"].items():
        strength, what = strength_note.get(name, ("?", "?"))
        add(f"| `{name}` | {info['built']}/{info['requested']} | `{strength}` | {what} |")
    add("")
    add("Refused sources:")
    add("")
    for name, info in doc["refused_datasets"].items():
        add(f"- `{name}` — {info['verdict']} ({info['with_audio_stream']}/{info['probed']} probed "
            "files carry an audio stream)")
    checks = doc["ground_truth_checks"]
    add("")
    add("Ground-truth defence (the three defect classes already found in the video corpus):")
    add("")
    add(f"- **units** — Perception Test `sound_localisation` timestamps are MICROSECONDS. Verified: "
        f"max `|timestamps/1e6 - frame_ids/frame_rate|` = **{checks.get('pt_unit_max_error_s')} s** over "
        f"{checks.get('pt_unit_samples')} interval bounds.")
    add(f"- **containment** — {checks.get('pt_dropped', {}).get('interval_past_media_end')} intervals "
        "past the end of the media (dropped, not clamped silently); every retained interval was "
        "re-checked against the ffprobe audio duration.")
    add(f"- **seeding** — selection is `random.Random({doc['selection_seed']}).shuffle` over a "
        "**sorted** candidate list, never filesystem order.")
    add("")

    # ---- per test ----------------------------------------------------------------------------
    for test in report["tests"]:
        add(f"## {test['id']} — {test['title']}")
        add("")
        add(f"Status: **{test['status']}**")
        add("")
        add("| Check | Result | Measured |")
        add("|---|---|---|")
        for check in test["checks"]:
            mark = "PASS" if check["passed"] else "**FAIL**"
            detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
            add(f"| {check['name']} | {mark} | {detail} |")
        add("")
        if test["error"]:
            add(f"ERROR: `{test['error']}`")
            add("")
        for note in test.get("notes", []):
            add(f"> {note}")
            add("")

        if test["id"] == "SV-A1.6R":
            measured = test["measurements"]
            add("### VAD / onset threshold frontier")
            add("")
            add(f"Corpus: {measured['corpus']['media_seconds']} s of annotated audio, "
                f"{measured['corpus']['any_sound_positives']} annotated sound events. "
                f"Admission tolerance {measured['admission_tolerance_s']} s "
                f"(one capture chunk at `chunk_s={measured['capture_chunk_s']}`).")
            add("")
            add("| `audio_wake.high` (energy) | `low` | recall | hits/positives | wakes | economy (wakes/chunk) |")
            add("|---|---|---|---|---|---|")
            for point in measured["onset_frontier"]:
                shipped = " **(shipped default)**" if point["high"] == SHIPPED_ONSET_HIGH else ""
                recall = "—" if point["recall"] is None else f"{point['recall']:.3f}"
                economy = "—" if point["economy"] is None else f"{point['economy']:.4f}"
                add(f"| {point['high']:.3g}{shipped} | {point['low']:.3g} | {recall} | "
                    f"{point['hits']}/{point['positives']} | {point['admissions']} | {economy} |")
            add("")
            quantiles = measured["corpus_energy_quantiles"]
            add(f"Corpus chunk-energy distribution: p10 `{quantiles['p10']:.3g}`, "
                f"p50 `{quantiles['p50']:.3g}`, p90 `{quantiles['p90']:.3g}`, "
                f"p99 `{quantiles['p99']:.3g}`, max `{quantiles['max']:.3g}`.")
            add("")
            add("### `utterance_threshold` sweep (the saturation)")
            add("")
            add("`utterance_threshold` is an **amplitude** knob; `_amplitude_threshold_to_energy` "
                "squares it before it reaches `EnergyVad` "
                f"(`{SHIPPED_UTTERANCE_THRESHOLD}` → energy `{SHIPPED_UTTERANCE_THRESHOLD ** 2:g}`). "
                f"`hangover_frames={SHIPPED_HANGOVER_FRAMES}` at "
                f"`chunk_s={measured['capture_chunk_s']}` means an endpoint needs "
                f"**{SHIPPED_HANGOVER_FRAMES * measured['capture_chunk_s']:.2f} s of continuous "
                "silence**.")
            add("")
            for slice_name, rows in measured["utterance_frontier"].items():
                add(f"**{slice_name}**")
                add("")
                add("| `utterance_threshold` | energy | speech fraction | endpoints | endpoints/min |")
                add("|---|---|---|---|---|")
                for row in rows:
                    shipped = " **(shipped)**" if row["utterance_threshold"] == SHIPPED_UTTERANCE_THRESHOLD else ""
                    add(f"| {row['utterance_threshold']:g}{shipped} | {row['energy_threshold']:.3g} | "
                        f"{row['speech_fraction']:.3f} | {row['endpoints']} | "
                        f"{row['endpoints_per_minute']:.2f} |")
                add("")

    # ---- defects ------------------------------------------------------------------------------
    add("## Defects")
    add("")
    if not report["defects"]:
        add("None found.")
    else:
        for defect in report["defects"]:
            add(f"### `{defect['id']}` — {defect['severity']}")
            add("")
            add(f"**Where:** `{defect['where']}`")
            add("")
            add(defect["summary"])
            add("")
            add(f"**Repro:** `{defect['repro']}`")
            add("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", help="run only these test ids (repeatable)")
    ap.add_argument("--json-out", type=Path, default=JSON_PATH)
    ap.add_argument("--md-out", type=Path, default=MD_PATH)
    ap.add_argument("--allow-source-tree", action="store_true",
                    help="skip the installed-wheel guard (development only)")
    args = ap.parse_args(argv)

    if not args.allow_source_tree:
        assert_runtime_under_test()

    report = run_all(args.only)
    _print_table(report)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=False, default=repr))
    args.md_out.write_text(render_markdown(report))
    print()
    print(f"json -> {args.json_out}")
    print(f"md   -> {args.md_out}")
    return 1 if report["totals"].get("FAIL") or report["totals"].get("ERROR") else 0


if __name__ == "__main__":
    raise SystemExit(main())
