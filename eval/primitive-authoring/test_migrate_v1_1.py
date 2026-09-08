"""The v1.1 frozen-pin migration is text-only, complete, and reproducible."""

from __future__ import annotations

import json

import migrate_v1_1 as migration


def test_migration_covers_current_unique_goals_without_provider_authoring():
    source = json.loads(migration.DEFAULT_SOURCE.read_text())
    diff = json.loads(migration.DEFAULT_DIFF.read_text())
    corpus = json.loads(migration.DEFAULT_CORPUS.read_text())
    payload = migration.migrate_payload(source, diff, corpus)

    assert payload["migration"]["provider_calls"] == 0
    assert len(payload["entries"]) == len({case["goal_text"] for case in corpus["cases"]})
    assert {row["after"] for row in payload["migration"]["rekeyed"]} == {
        "a person completes a bench press repetition",
        "a person completes a jumping jack",
        "a person completes a pull-up",
        "a person completes a push-up",
    }


def test_genuine_kind_disagreements_remain_visible_not_repaired_from_labels():
    payload = migration.migrate_payload(
        json.loads(migration.DEFAULT_SOURCE.read_text()),
        json.loads(migration.DEFAULT_DIFF.read_text()),
        json.loads(migration.DEFAULT_CORPUS.read_text()),
    )
    assert payload["scorecard"]["hit"] == 91
    assert len(payload["scorecard"]["mismatches"]) == 9
