"""Corpus 1 — SUBJECT IDENTITY: what makes two sightings the same *thing*.

The collapse this corpus pins: today's identity is IoU + label. Two invoices in the SAME screen
region with the SAME label are, to that model, one subject — even when their content keys differ.
Case ``same_region_same_label_different_keys`` is that exact input, labelled with the answer the
current model *cannot express* (two distinct subjects).
"""

from __future__ import annotations

from ._common import DOC_BOX, frame_step, labels_doc, obs, stream_doc

NAME = "subject_identity"
GENERATOR = __name__


def _doc_step(t: float, token: str, key: str | None, *, label: str = "document") -> dict:
    """A document sighting. ``key=None`` models an unreadable/absent content key."""
    attrs = {"content_key": key} if key else {"content_key_readable": "false"}
    verdict = {
        "satisfied": True,
        "confidence": 0.90 if key else 0.35,
        "evidence": f"scripted: {label} in region, key={key or 'UNREADABLE'}",
        # ADDITIVE fields P0 will introduce; ScriptedJudge attaches them today (see scripted.py).
        "subject_key": f"{label}:{key}" if key else None,
        "binding": "resolved" if key else "unknown",
    }
    return frame_step(t, token, observations=[obs(label, box=DOC_BOX, **attrs)], verdict=verdict)


def build() -> tuple[dict, dict]:
    cases = [
        {
            "id": "same_region_same_label_different_keys",
            "description": (
                "Two invoices shown in the SAME screen region with the SAME label 'document', "
                "differing only in content key (INV-2291 then INV-2294)."
            ),
            "steps": [_doc_step(0.0, "invoice-2291", "INV-2291"), _doc_step(4.0, "invoice-2294", "INV-2294")],
        },
        {
            "id": "unreadable_key_binds_unknown",
            "description": "A document whose content key is blurred/absent — nothing legible to bind to.",
            "steps": [_doc_step(0.0, "invoice-blurred", None)],
        },
        {
            "id": "reappearance_after_unrelated_document",
            "description": (
                "INV-2291, then an unrelated INV-2999, then INV-2291 again — the same document "
                "returning after another subject held the region."
            ),
            "steps": [
                _doc_step(0.0, "invoice-2291", "INV-2291"),
                _doc_step(6.0, "invoice-2999", "INV-2999"),
                _doc_step(12.0, "invoice-2291-again", "INV-2291"),
            ],
        },
    ]

    labels = [
        {
            "id": "same_region_same_label_different_keys",
            "expect": {
                "distinct_subject_count": 2,
                "subjects": [
                    {"key": "document:INV-2291", "binding": "resolved", "step_indices": [0]},
                    {"key": "document:INV-2294", "binding": "resolved", "step_indices": [1]},
                ],
                "identity_signal": "content_key",
                "collapses_under_iou_label_identity": True,
                "wrong_answer": {"distinct_subject_count": 1, "why": "IoU ~1.0 and identical label"},
            },
            "proves": (
                "The identity model must be able to express TWO subjects when region and label are "
                "identical and only the content key differs."
            ),
            "does_not_prove": (
                "That any real OCR/VLM reads the key correctly off a real screen — key extraction "
                "quality is a benchmark question, not a capability question."
            ),
        },
        {
            "id": "unreadable_key_binds_unknown",
            "expect": {
                "distinct_subject_count": 1,
                "subjects": [{"key": None, "binding": "unknown", "step_indices": [0]}],
                "must_not_fabricate_key": True,
                "must_not_report_no_subject": True,
                "wrong_answers": [
                    {"why": "invented a key from position/ordinal", "example_key": "document:1"},
                    {"why": "dropped the sighting entirely", "distinct_subject_count": 0},
                ],
            },
            "proves": (
                "An unreadable key yields the honest third answer binding='unknown' — neither a "
                "fabricated key nor a silent disappearance of the subject."
            ),
            "does_not_prove": "How often a real judge fails to read a key, or how it should recover afterwards.",
        },
        {
            "id": "reappearance_after_unrelated_document",
            "expect": {
                "distinct_subject_count": 2,
                "subjects": [
                    {"key": "document:INV-2291", "binding": "resolved", "step_indices": [0, 2]},
                    {"key": "document:INV-2999", "binding": "resolved", "step_indices": [1]},
                ],
                "durable_key": True,
                "session_ordinal_forbidden": True,
                "wrong_answer": {
                    "distinct_subject_count": 3,
                    "why": "keys minted per sighting (a session ordinal), so the return looks new",
                },
            },
            "proves": (
                "The subject key is DURABLE content-derived state: the same document resolves to the "
                "same key after an unrelated subject intervened."
            ),
            "does_not_prove": "That keys stay stable across restarts, storage, or long real-world drift.",
        },
    ]

    return stream_doc(NAME, "frames", GENERATOR, cases), labels_doc(NAME, labels)
