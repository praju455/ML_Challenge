from __future__ import annotations

import unittest

from src.train import _sample_source, best_threshold


class TrainingSelectionTests(unittest.TestCase):
    def test_source_sampling_is_deterministic(self) -> None:
        self.assertEqual(_sample_source("S1-7", 0.5, 2026), _sample_source("S1-7", 0.5, 2026))

    def test_threshold_optimizes_entity_macro_f05(self) -> None:
        groups = ["S1-1", "S1-1", "S1-2"]
        candidates = ["S2-1", "S2-2", "S3-1"]
        probabilities = [0.9, 0.6, 0.8]
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        threshold, report = best_threshold(groups, candidates, probabilities, truth, [0.5, 0.7, 0.85])
        self.assertEqual(threshold, 0.85)
        self.assertEqual(report["macro_f0_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
