from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.mine_equivalences import MiningConfig, run_mining


class EvidenceBasedMiningTests(unittest.TestCase):
    @staticmethod
    def _write_normalized(path: Path, rows: list[tuple[str, str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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
            for entity_id, clean_name, clean_address in rows:
                writer.writerow((entity_id, clean_name, clean_address, "X", clean_name, clean_address))

    def test_mines_repeated_true_pair_replacements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            self._write_normalized(
                normalized / "train/train_source1.tsv",
                [
                    ("S1-1", "acme corp", "7 lake rd"),
                    ("S1-2", "bright corp", "9 hill rd"),
                    ("S1-3", "smith and sons", "1 main street"),
                    ("S1-4", "jones and sons", "2 main street"),
                ],
            )
            self._write_normalized(
                normalized / "train/train_source2.tsv",
                [
                    ("S2-1", "acme corporation", "7 lake road"),
                    ("S2-2", "bright corporation", "9 hill road"),
                    ("S2-3", "smith sons", "1 main street"),
                    ("S2-4", "jones sons", "2 main street"),
                ],
            )
            self._write_normalized(normalized / "train/train_source3.tsv", [])
            ground_truth = root / "ground_truth.tsv"
            with ground_truth.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source1_entity_id", "matched_entity_ids"))
                writer.writerow(("S1-1", "S2-1"))
                writer.writerow(("S1-2", "S2-2"))
                writer.writerow(("S1-3", "S2-3"))
                writer.writerow(("S1-4", "S2-4"))

            output_path = root / "equivalences.tsv"
            report = run_mining(
                normalized_data_dir=normalized,
                ground_truth_path=ground_truth,
                database_path=root / "records.sqlite",
                output_path=output_path,
                report_path=root / "report.json",
                config=MiningConfig(
                    min_support=2,
                    min_purity=0.5,
                    min_confidence=0.5,
                    min_drop_support=2,
                    min_drop_purity=0.9,
                    min_drop_coverage=0.2,
                    batch_entities=2,
                    query_batch_size=3,
                    insert_batch_size=2,
                ),
            )
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        mappings = {(row["field"], row["variant"]): row for row in rows}
        self.assertEqual(mappings[("name", "corp")]["canonical"], "corporation")
        self.assertEqual(mappings[("address", "rd")]["canonical"], "road")
        self.assertEqual(mappings[("name", "and")]["canonical"], "")
        self.assertEqual(report["alignment"]["resolved_edges"], 4)
        self.assertEqual(report["selected_equivalences"], 3)


if __name__ == "__main__":
    unittest.main()
