from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.blocking import (
    BlockingConfig,
    CandidateBlocker,
    Record,
    SQLiteLexicalIndex,
    character_shingles,
    generate_candidate_file,
    minhash_band_keys,
)
from src.eval_blocking import evaluate_blocking


def _encoder(value: str) -> tuple[str, str]:
    return (value[:4].upper(), "")


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
        for entity_id, name, address in rows:
            writer.writerow((entity_id, name, address, "Example", name, address))


class SQLiteBlockingTests(unittest.TestCase):
    def test_lexical_strategies_are_repeatable_and_idempotent(self) -> None:
        config = BlockingConfig(minhash_permutations=8, minhash_bands=4, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "blocking.sqlite"
            records = [
                Record("S2-1", "acme limited", "7 lake road"),
                Record("S3-1", "acme ltd", "7 lake rd"),
                Record("S2-2", "other bakery", "9 hill lane"),
            ]
            with SQLiteLexicalIndex(database, config, phonetic_encoder=_encoder) as index:
                self.assertEqual(index.add_many(records), 3)
                self.assertEqual(index.add_many(records), 0)
                self.assertEqual(index.target_count, 3)
                self.assertIn("S2-1", index.sorted_neighbors("acme limited"))
                self.assertIn("S2-1", index.phonetic_neighbors("acme ltd"))
                blocker = CandidateBlocker(index, lambda _record: {"S3-1"})
                candidates = blocker.candidates(Record("S2-1", "acme ltd", "7 lake rd"))
                self.assertNotIn("S2-1", candidates)
                self.assertIn("S3-1", candidates)

    def test_shingles_and_minhash_are_deterministic(self) -> None:
        config = BlockingConfig(minhash_permutations=8, minhash_bands=4)
        self.assertEqual(character_shingles("abc"), ("abc",))
        self.assertEqual(
            minhash_band_keys("acme cafe 7 lake road", config),
            minhash_band_keys("acme cafe 7 lake road", config),
        )
        self.assertEqual(len(minhash_band_keys("acme cafe", config)), 4)


class CandidateOutputTests(unittest.TestCase):
    def test_candidate_tsv_has_one_row_per_source1_and_union_of_strategies(self) -> None:
        class FakeAnn:
            def __init__(self, _config: BlockingConfig) -> None:
                pass

            def build(self, _paths: list[Path], max_rows_per_file: int | None = None) -> dict[str, object]:
                return {"backend": "fake", "targets": 2}

            @staticmethod
            def neighbors(record: Record) -> set[str]:
                return {"S3-2"} if record.entity_id == "S1-1" else set()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "source1.tsv"
            source2 = root / "source2.tsv"
            source3 = root / "source3.tsv"
            _write_normalized(source1, [("S1-1", "acme ltd", "7 lake road"), ("S1-2", "", "")])
            _write_normalized(source2, [("S2-1", "acme limited", "7 lake rd")])
            _write_normalized(source3, [("S3-2", "acme ltd", "7 lake road")])
            output = root / "candidates.tsv"
            report = root / "report.json"
            with (
                patch("src.blocking.CharacterNgramANN", FakeAnn),
                patch("src.blocking.default_phonetic_encoder", _encoder),
            ):
                result = generate_candidate_file(
                    source1_path=source1,
                    target_paths=[source2, source3],
                    output_path=output,
                    database_path=root / "index.sqlite",
                    report_path=report,
                    config=BlockingConfig(minhash_permutations=8, minhash_bands=4),
                )
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["source1_entity_id"] for row in rows], ["S1-1", "S1-2"])
            self.assertIn("S3-2", rows[0]["candidate_entity_ids"].split(","))
            self.assertEqual(rows[1]["candidate_entity_ids"], "")
            self.assertEqual(result["candidate_count"]["median"], 1)


class BlockingEvaluationTests(unittest.TestCase):
    def test_evaluation_reports_recall_and_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.tsv"
            candidates = root / "candidates.tsv"
            with truth.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source1_entity_id", "matched_entity_ids"))
                writer.writerow(("S1-1", "S2-1,S3-1"))
                writer.writerow(("S1-2", ""))
            with candidates.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source1_entity_id", "candidate_entity_ids"))
                writer.writerow(("S1-1", "S2-1,S2-2"))
                writer.writerow(("S1-2", ""))
            report = evaluate_blocking(truth, candidates, target_count=4, validation_fraction=1)
            self.assertEqual(report["validation_entities"], 2)
            self.assertEqual(report["blocking_recall_ceiling"], 0.5)
            self.assertEqual(report["reduction_ratio"], 0.75)
            self.assertEqual(report["zero_match_entities_with_empty_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
