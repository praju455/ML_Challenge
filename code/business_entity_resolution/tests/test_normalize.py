from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.normalize import normalize_text, transform_all, transform_file


class NormalizeTextTests(unittest.TestCase):
    def test_unicode_case_whitespace_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_text("  LLC Moncada Léarning—Center  "),
            "llc moncada léarning center",
        )
        self.assertEqual(normalize_text("Ｂ＋ Retail (Inc.)"), "b retail inc")
        self.assertEqual(normalize_text("मॉडर्न  फाइनेंस"), "मॉडर्न फाइनेंस")

    def test_empty_values(self) -> None:
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("..."), "")


class TransformTests(unittest.TestCase):
    @staticmethod
    def _write_source(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("entity_id", "business_name", "business_address", "country"))
            writer.writerow(("S1-1", "ACME Corp.", "7 Lake Rd", "US"))
            writer.writerow(("S1-2", "मॉडर्न फाइनेंस", "", "India"))

    def test_transform_preserves_originals_and_adds_clean_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.tsv"
            output_path = root / "output.tsv"
            self._write_source(input_path)
            report = transform_file(input_path, output_path)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["empty_clean_address"], 1)
        self.assertEqual(rows[0]["business_name"], "ACME Corp.")
        self.assertEqual(rows[0]["clean_name"], "acme corp")
        self.assertEqual(rows[0]["clean_address"], "7 lake rd")
        self.assertEqual(rows[1]["clean_name"], "मॉडर्न फाइनेंस")

    def test_all_six_files_and_report(self) -> None:
        relative_files = (
            "train/train_source1.tsv",
            "train/train_source2.tsv",
            "train/train_source3.tsv",
            "test/test_source1.tsv",
            "test/test_source2.tsv",
            "test/test_source3.tsv",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "dataset"
            for relative in relative_files:
                self._write_source(data_dir / relative)
            summary = transform_all(
                data_dir=data_dir,
                output_dir=root / "processed",
                report_path=root / "report.json",
                max_rows_per_file=1,
            )
            self.assertTrue((root / "report.json").is_file())
            self.assertTrue((root / "processed/test/test_source3.tsv").is_file())
        self.assertEqual(summary["total_rows"], 6)


if __name__ == "__main__":
    unittest.main()
