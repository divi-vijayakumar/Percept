import asyncio
import pathlib
import unittest


from rehearsal import rehearse, summary


class RehearsalTests(unittest.TestCase):
    def test_latency_summary_is_hand_calculable(self):
        self.assertEqual(
            {
                "p50": 2.5,
                "p95": 3.8499999999999996,
                "p99": 3.9699999999999998,
                "max": 4,
            },
            summary([1, 2, 3, 4]),
        )

    def test_qualification_path_is_rejected_before_capture(self):
        path = pathlib.Path("media/qualification-0000/example.mp4")
        with self.assertRaisesRegex(ValueError, "only explicitly exposed calibration"):
            asyncio.run(rehearse(path))


if __name__ == "__main__":
    unittest.main()
