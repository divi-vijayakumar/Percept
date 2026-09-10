import pathlib

from reconstruct_review_cases import DEFAULT_EVIDENCE_ROOT, reconstruct


def test_reconstructs_exact_zero_gt_review_set() -> None:
    payload = reconstruct(DEFAULT_EVIDENCE_ROOT)
    assert payload["case_count"] == 15
    assert {item["source_id"] for item in payload["cases"]} == {
        "VIRAT_S_050200_00_000106_000380",
        "VIRAT_S_050201_08_001567_001647",
        "VIRAT_S_050201_11_002090_002220",
        "VIRAT_S_050202_00_000000_000105",
        "VIRAT_S_050202_04_000690_000750",
        "VIRAT_S_050203_08_001686_001885",
    }
    assert len({item["incident_id"] for item in payload["cases"]}) == 15
    assert all(item["source_id"].startswith("VIRAT_S_0502") for item in payload["cases"])

