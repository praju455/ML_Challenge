from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from utils.validate_submission import validate_submission


def _write_source(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("entity_id", "business_name", "business_address", "country"))
        for entity_id in ids:
            writer.writerow((entity_id, "name", "address", "country"))


def _write_tsv(path: Path, header: tuple[str, str], rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        _write_source(self.root / "test_source1.tsv", ["S1-1", "S1-2"])
        _write_source(self.root / "test_source2.tsv", ["S2-1"])
        _write_source(self.root / "test_source3.tsv", ["S3-1"])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_valid_submission_and_candidate_subset_passes(self) -> None:
        _write_tsv(
            self.root / "candidates.tsv",
            ("source1_entity_id", "candidate_entity_ids"),
            [("S1-1", "S2-1,S3-1"), ("S1-2", "")],
        )
        _write_tsv(
            self.root / "matching.tsv",
            ("source1_entity_id", "matched_entity_ids"),
            [("S1-1", "S3-1"), ("S1-2", "")],
        )
        report = validate_submission(
            self.root / "matching.tsv",
            self.root / "test_source1.tsv",
            self.root / "test_source2.tsv",
            self.root / "test_source3.tsv",
            self.root / "candidates.tsv",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["predicted_pairs"], 1)

    def test_rejects_candidate_file_and_incomplete_prediction(self) -> None:
        _write_tsv(
            self.root / "wrong.tsv",
            ("source1_entity_id", "candidate_entity_ids"),
            [("S1-1", "S2-1")],
        )
        with self.assertRaisesRegex(ValueError, "expected header"):
            validate_submission(
                self.root / "wrong.tsv",
                self.root / "test_source1.tsv",
                self.root / "test_source2.tsv",
                self.root / "test_source3.tsv",
            )

        _write_tsv(
            self.root / "incomplete.tsv",
            ("source1_entity_id", "matched_entity_ids"),
            [("S1-1", "S2-1")],
        )
        report = validate_submission(
            self.root / "incomplete.tsv",
            self.root / "test_source1.tsv",
            self.root / "test_source2.tsv",
            self.root / "test_source3.tsv",
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["issues"]["missing_source1_rows"], 1)

    def test_rejects_invalid_or_non_candidate_prediction(self) -> None:
        _write_tsv(
            self.root / "candidates.tsv",
            ("source1_entity_id", "candidate_entity_ids"),
            [("S1-1", "S2-1"), ("S1-2", "")],
        )
        _write_tsv(
            self.root / "matching.tsv",
            ("source1_entity_id", "matched_entity_ids"),
            [("S1-1", "S3-1,S3-1"), ("S1-2", "S2-404")],
        )
        report = validate_submission(
            self.root / "matching.tsv",
            self.root / "test_source1.tsv",
            self.root / "test_source2.tsv",
            self.root / "test_source3.tsv",
            self.root / "candidates.tsv",
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["issues"]["duplicate_target_in_row"], 1)
        self.assertEqual(report["issues"]["prediction_not_in_candidate_pairs"], 2)
        self.assertEqual(report["issues"]["unknown_target_id"], 1)


if __name__ == "__main__":
    unittest.main()
