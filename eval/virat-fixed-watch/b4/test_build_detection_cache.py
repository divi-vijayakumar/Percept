import pathlib

import pytest

from build_detection_cache import build, source_group


def test_source_group() -> None:
    path = pathlib.Path("VIRAT_S_050201_00_000012_000116.mp4")
    assert source_group(path) == "0502"


def test_qualification_directory_is_rejected_before_model_import(tmp_path) -> None:
    with pytest.raises(ValueError, match="calibration-exposed or validation"):
        build(
            media_dir=tmp_path / "qualification-0503",
            weights=tmp_path / "weights",
            zone_manifest=tmp_path / "zones",
            output_dir=tmp_path / "out",
            sample_hz=1.0,
            minimum_size=320,
            threads=1,
        )
