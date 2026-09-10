"""Run the frozen local VLM review while loading model weights only once."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import time

from mlx_vlm import apply_chat_template, generate, load

LABELS = {
    "true_eligible_entry",
    "not_entry_tracking_error",
    "left_censored_or_already_inside",
    "ambiguous",
    "unusable",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_response(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    response = json.loads(text[start : end + 1])
    if response.get("label") not in LABELS:
        raise ValueError(f"invalid label: {response.get('label')!r}")
    required = {
        "confidence",
        "person_description",
        "outside_observation",
        "crossing_observation",
        "uncertainty",
    }
    missing = required - response.keys()
    if missing:
        raise ValueError(f"missing response fields: {sorted(missing)}")
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("prompt", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--seed-result", type=pathlib.Path)
    parser.add_argument("--start-case", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    prompt_text = args.prompt.read_text()
    results = []
    if args.seed_result:
        results.append(json.loads(args.seed_result.read_text()))
    output = {
        "schema": "percept.benchmark.local-vlm-adjudication.v1",
        "execution_identity": "local-model-freeze-v3.json",
        "evidentiary_status": "model-assisted calibration triage; not ground truth",
        "provider_calls": 0,
        "provider_cost_usd": 0,
        "model_path": str(args.model),
        "model_safetensors_sha256": sha256(args.model / "model.safetensors"),
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load(str(args.model))
    for case in manifest["cases"]:
        ordinal = int(case["case_id"].rsplit("-", 1)[1])
        if ordinal < args.start_case:
            continue
        started = time.monotonic()
        formatted = apply_chat_template(
            processor,
            model.config,
            prompt_text,
            num_images=len(case["ordered_review_frames"]),
            num_audios=0,
        )
        generated = generate(
            model,
            processor,
            formatted,
            image=case["ordered_review_frames"],
            temperature=0.0,
            max_tokens=350,
            verbose=False,
        )
        record = {
            "case_id": case["case_id"],
            "incident_id": case["incident_id"],
            "source_id": case["source_id"],
            "source_seconds": case["source_seconds"],
            "response": parse_response(generated.text),
            "raw_response": generated.text,
            "elapsed_seconds": time.monotonic() - started,
            "prompt_tokens": generated.prompt_tokens,
            "generation_tokens": generated.generation_tokens,
            "peak_memory_gb": generated.peak_memory,
            "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        output["results"].append(record)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

