import pathlib
import tempfile
import unittest
import zipfile

from build_media_manifest import (
    annotation_inventory,
    annotation_stems,
    parse_official_listings,
)


class MediaManifestTests(unittest.TestCase):
    def test_listing_parser_ignores_headers_and_reads_media_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = pathlib.Path(directory) / "listing.txt"
            listing.write_text(
                "Archive: videos.zip\n"
                "  123 Stored 123 0% 2025-01-22 14:09 deadbeef "
                "videos_original/VIRAT_S_010000_00_000000_000165.mp4\n"
                "-------- --------\n"
            )
            self.assertEqual(
                {"VIRAT_S_010000_00_000000_000165.mp4": 123},
                parse_official_listings([listing]),
            )

    def test_annotation_inventory_excludes_directory_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = pathlib.Path(directory) / "annotations.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "annotations/VIRAT_S_050200.viratdata.objects.txt",
                    "1 1 0 0 0 1 1 1\n",
                )
                archive.writestr(
                    "annotations/VIRAT_S_050201.viratdata.objects.txt/", ""
                )
            self.assertEqual({"VIRAT_S_050200"}, annotation_stems(archive_path))
            self.assertEqual(
                {
                    "rows": 1,
                    "minimum_frame": 0,
                    "maximum_frame": 0,
                    "person_rows": 1,
                    "person_tracks": 1,
                    "malformed_rows": 0,
                },
                annotation_inventory(archive_path)["VIRAT_S_050200"],
            )


if __name__ == "__main__":
    unittest.main()
