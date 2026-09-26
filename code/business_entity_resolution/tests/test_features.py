from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.features import FEATURE_COLUMNS, Entity, generate_feature_file, pair_features


def _write_normalized(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "entity_id",
                "business_name",
                "business_address",
                "country",
                "clean_name",
                "clean_address",
            )
        )
        for entity_id, name, address, country in rows:
            writer.writerow((entity_id, name, address, country, name, address))


def _write_candidates(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "candidate_entity_ids"))
        writer.writerows(rows)


class PairFeatureTests(unittest.TestCase):
    def test_exact_numeric_and_country_features_are_explicit(self) -> None:
        source = Entity("S1-1", "acme cafe 7", "10 lake road", "India")
        target = Entity("S2-1", "acme cafe 7", "10 lake road", "India")
        values = pair_features(source, target)
        self.assertEqual(set(values), set(FEATURE_COLUMNS[3:]))
        self.assertEqual(values["name_exact"], 1.0)
        self.assertEqual(values["address_numeric_jaccard"], 1.0)
        self.assertEqual(values["country_equal"], 1.0)

    def test_country_is_not_a_filtering_rule(self) -> None:
        source = Entity("S1-1", "acme cafe", "10 lake road", "India")
        target = Entity("S3-1", "acme coffee", "10 lake rd", "France")
        values = pair_features(source, target)
        self.assertEqual(values["country_equal"], 0.0)
        self.assertGreater(values["name_sequence_ratio"], 0.0)


class FeatureFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source1 = self.root / "source1.tsv"
        self.source2 = self.root / "source2.tsv"
        self.source3 = self.root / "source3.tsv"
        _write_normalized(
            self.source1,
            [
                ("S1-1", "acme cafe 7", "10 lake road", "India"),
                ("S1-2", "empty", "9 hill lane", "US"),
            ],
        )
        _write_normalized(self.source2, [("S2-1", "acme cafe 7", "10 lake road", "India")])
        _write_normalized(self.source3, [("S3-1", "acme coffee", "10 lake rd", "France")])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_feature_contract_preserves_candidates_and_adds_labels(self) -> None:
        candidates = self.root / "candidates.tsv"
        truth = self.root / "truth.tsv"
        _write_candidates(candidates, [("S1-1", "S2-1,S3-1"), ("S1-2", "")])
        with truth.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("source1_entity_id", "matched_entity_ids"))
            writer.writerow(("S1-1", "S2-1"))
            writer.writerow(("S1-2", ""))
        output = self.root / "features.tsv"
        report_path = self.root / "report.json"
        report = generate_feature_file(
            self.source1,
            [self.source2, self.source3],
            candidates,
            output,
            self.root / "targets.sqlite",
            report_path,
            truth,
            self.root / "truth.sqlite",
            rebuild_target_index=True,
            rebuild_truth_index=True,
        )
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), FEATURE_COLUMNS)
        self.assertEqual([(row["candidate_entity_id"], row["is_match"]) for row in rows], [("S2-1", "1"), ("S3-1", "0")])
        self.assertEqual(rows[1]["country_equal"], "0.000000")
        self.assertEqual(report["source1_records"], 2)
        self.assertEqual(report["feature_rows"], 2)
        self.assertEqual(report["positive_feature_rows"], 1)
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["labels_present"], True)

    def test_unknown_candidate_and_misaligned_rows_are_rejected(self) -> None:
        candidates = self.root / "candidates.tsv"
        _write_candidates(candidates, [("S1-2", "S2-404"), ("S1-1", "")])
        with self.assertRaisesRegex(ValueError, "expected Source-1 row S1-1"):
            generate_feature_file(
                self.source1,
                [self.source2, self.source3],
                candidates,
                self.root / "features.tsv",
                self.root / "targets.sqlite",
                self.root / "report.json",
                rebuild_target_index=True,
            )


if __name__ == "__main__":
    unittest.main()
