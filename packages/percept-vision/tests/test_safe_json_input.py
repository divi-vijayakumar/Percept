"""Adversarial tests for viability artifact input at the local filesystem boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import percept._safe_json as safe_json
from percept.eval.viability_frontier import (
    ViabilityInputError,
    load_viability_evidence,
    load_viability_manifest,
    main,
)


def test_safe_json_receipt_binds_exact_descriptor_snapshot(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"watch":"door"}', encoding="utf-8")

    result = safe_json.load_json_file(source)

    assert result.value == {"watch": "door"}
    assert result.receipt.path == str(source)
    assert result.receipt.canonical_path == str(source.resolve())
    assert result.receipt.byte_count == source.stat().st_size
    assert result.receipt.sha256.startswith("sha256:")
    assert len(result.receipt.sha256) == len("sha256:") + 64
    assert (result.receipt.device, result.receipt.inode) == (
        source.stat().st_dev,
        source.stat().st_ino,
    )
    assert result.receipt.modified_time_ns == source.stat().st_mtime_ns


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is not supported")
def test_fifo_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "manifest.json"
    os.mkfifo(fifo)
    code = """
import sys
from percept.eval.viability_frontier import ViabilityInputError, load_viability_manifest
try:
    load_viability_manifest(sys.argv[1])
except ViabilityInputError:
    raise SystemExit(0)
raise SystemExit(3)
"""

    completed = subprocess.run(
        [sys.executable, "-c", code, str(fifo)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        timeout=3,
    )

    assert completed.returncode == 0


@pytest.mark.skipif(not Path("/dev/zero").exists(), reason="/dev/zero is not available")
def test_character_device_is_rejected_before_reading():
    with pytest.raises(ViabilityInputError, match="not a regular file"):
        load_viability_manifest("/dev/zero")


def test_final_symlink_is_rejected(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(target)

    with pytest.raises(ViabilityInputError, match="not a regular file"):
        load_viability_manifest(link)


def test_hardlink_is_rejected(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "hardlinked.json"
    os.link(target, link)

    with pytest.raises(ViabilityInputError, match="exactly one hard link"):
        load_viability_manifest(link)


def test_oversized_sparse_file_is_rejected_without_materializing_it(tmp_path):
    source = tmp_path / "too-large.json"
    with source.open("wb") as stream:
        stream.truncate(safe_json.SAFE_JSON_MAX_BYTES + 1)

    with pytest.raises(ViabilityInputError, match="exceeds the .*byte limit"):
        load_viability_manifest(source)


def test_primitive_rejects_empty_input(tmp_path):
    source = tmp_path / "empty.json"
    source.write_bytes(b"")

    with pytest.raises(safe_json.SafeJSONInputError, match="input is empty"):
        safe_json.load_json_file(source)
    with pytest.raises(ViabilityInputError, match="evidence is empty"):
        load_viability_evidence(source)


def test_embedded_nul_is_a_controlled_public_input_error(capsys):
    invalid = "manifest.json\0suffix"

    with pytest.raises(ViabilityInputError, match="path inspection failed"):
        load_viability_manifest(invalid)
    assert main(["--manifest", invalid, "--evidence", "unused.jsonl"]) == 2
    assert "VIABILITY INPUT ERROR" in capsys.readouterr().err


def test_invalid_utf8_is_a_controlled_public_input_error(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_bytes(b'{"watch":"\xff"}')

    with pytest.raises(ViabilityInputError, match="not valid UTF-8"):
        load_viability_manifest(source)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"one","schema_version":"two"}',
        '{"watch":{"condition":"one","condition":"two"}}',
    ],
)
def test_manifest_rejects_duplicate_keys_at_every_depth(tmp_path, payload):
    source = tmp_path / "manifest.json"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ViabilityInputError, match="duplicate JSON object key"):
        load_viability_manifest(source)


@pytest.mark.parametrize(
    "payload",
    [
        '{"case_id":"one","case_id":"two"}\n',
        '{"case_id":"one","event":{"id":"one","id":"two"}}\n',
    ],
)
def test_jsonl_rejects_duplicate_keys_at_every_depth(tmp_path, payload):
    source = tmp_path / "evidence.jsonl"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ViabilityInputError, match="duplicate JSON object key"):
        load_viability_evidence(source)


def test_json_array_evidence_also_rejects_nested_duplicate_keys(tmp_path):
    source = tmp_path / "evidence.json"
    source.write_text('[{"event":{"id":"one","id":"two"}}]', encoding="utf-8")

    with pytest.raises(ViabilityInputError, match="duplicate JSON object key"):
        load_viability_evidence(source)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_non_finite_constants(tmp_path, constant):
    source = tmp_path / "evidence.json"
    source.write_text(f'[{{"score":{constant}}}]', encoding="utf-8")

    with pytest.raises(ViabilityInputError, match="non-finite JSON number"):
        load_viability_evidence(source)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_jsonl_rejects_non_finite_constants(tmp_path, constant):
    source = tmp_path / "evidence.jsonl"
    source.write_text(f'{{"score":{constant}}}\n', encoding="utf-8")

    with pytest.raises(ViabilityInputError, match="non-finite JSON number"):
        load_viability_evidence(source)


def test_excessive_json_nesting_is_a_controlled_error(tmp_path):
    source = tmp_path / "evidence.json"
    source.write_text("[" * 20_000 + "0" + "]" * 20_000, encoding="utf-8")

    with pytest.raises(ViabilityInputError, match="cannot parse evidence"):
        load_viability_evidence(source)


def test_path_replacement_during_read_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps([{"case_id": "old"}]), encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps([{"case_id": "new"}]), encoding="utf-8")
    original_read = os.read
    replaced = False

    def replacing_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(fd, size)
        if not replaced:
            replaced = True
            replacement.replace(source)
        return chunk

    monkeypatch.setattr(safe_json.os, "read", replacing_read)
    with pytest.raises(ViabilityInputError, match="replaced"):
        load_viability_evidence(source)


def test_in_place_mutation_during_read_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps([{"case_id": "old"}]), encoding="utf-8")
    original_read = os.read
    mutated = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(fd, size)
        if not mutated:
            mutated = True
            source.write_text(json.dumps([{"case_id": "new"}]), encoding="utf-8")
        return chunk

    monkeypatch.setattr(safe_json.os, "read", mutating_read)
    with pytest.raises(ViabilityInputError, match="changed"):
        load_viability_evidence(source)


def test_removal_during_read_is_a_controlled_error(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps([{"case_id": "old"}]), encoding="utf-8")
    original_read = os.read
    removed = False

    def removing_read(fd: int, size: int) -> bytes:
        nonlocal removed
        chunk = original_read(fd, size)
        if not removed:
            removed = True
            source.unlink()
        return chunk

    monkeypatch.setattr(safe_json.os, "read", removing_read)
    with pytest.raises(ViabilityInputError, match="path inspection failed"):
        load_viability_evidence(source)
