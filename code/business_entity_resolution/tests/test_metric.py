from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.metric import entity_f05, evaluate_match_lists, load_match_lists


class EntityMetricTests(unittest.TestCase):
    def test_empty_and_singleton_competition_edge_cases(self) -> None:
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05(set(), {"S2-1"}), 0.0)
        self.assertEqual(entity_f05({"S2-1"}, {"S2-1"}), 1.0)
        self.assertEqual(entity_f05({"S2-1"}, {"S2-1", "S3-1"}), 0.0)
        self.assertEqual(entity_f05({"S2-1"}, set()), 0.0)

    def test_multi_match_uses_f05_formula(self) -> None:
        score = entity_f05({"S2-1", "S3-1"}, {"S2-1", "S2-2"})
        self.assertAlmostEqual(score, 0.5)

    def test_macro_report_and_grouping(self) -> None:
        truth = {"S1-1": set(), "S1-2": {"S2-1"}, "S1-3": {"S2-1", "S3-1"}}
        predicted = {"S1-1": set(), "S1-2": {"S2-1"}, "S1-3": {"S2-1", "S2-2"}}
        report = evaluate_match_lists(truth, predicted, {"S1-1": "India", "S1-2": "India", "S1-3": "US"})
        self.assertAlmostEqual(float(report["macro_f0_5"]), (1 + 1 + 0.5) / 3)
        self.assertEqual(report["macro_f0_5_by_group"], {"India": 1.0, "US": 0.5})

    def test_rejects_incomplete_and_duplicate_tsv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.tsv"
            path.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate matched ID"):
                load_match_lists(path)
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            evaluate_match_lists({"S1-1": set()}, {})


if __name__ == "__main__":
    unittest.main()
