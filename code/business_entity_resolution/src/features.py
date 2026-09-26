"""Create reproducible, candidate-backed pair features for entity resolution.

The module accepts normalized Source-1 and target TSVs plus the Phase-2 candidate
contract.  It emits one row per candidate pair; Source-1 rows with no candidates
correctly emit no pair-feature rows.  ``country_equal`` is a regular feature only:
country is never used to reject a candidate or select a code path.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


NORMALIZED_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "clean_name",
    "clean_address",
)
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

PAIR_FEATURE_COLUMNS = (
    "name_exact",
    "address_exact",
    "name_token_jaccard",
    "address_token_jaccard",
    "name_char_trigram_jaccard",
    "address_char_trigram_jaccard",
    "name_sequence_ratio",
    "address_sequence_ratio",
    "name_numeric_jaccard",
    "address_numeric_jaccard",
    "name_length_ratio",
    "address_length_ratio",
    "name_token_count_ratio",
    "address_token_count_ratio",
    "name_prefix_ratio",
    "address_prefix_ratio",
    "country_equal",
)
FEATURE_COLUMNS = ("source1_entity_id", "candidate_entity_id", "is_match", *PAIR_FEATURE_COLUMNS)
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    clean_name: str
    clean_address: str
    country: str


def _require_exact_header(path: Path, fieldnames: Sequence[str] | None, expected: Sequence[str]) -> None:
    if tuple(fieldnames or ()) != tuple(expected):
        raise ValueError(f"{path}: expected exact TSV header {tuple(expected)}")


def iter_normalized_entities(path: Path) -> Iterator[Entity]:
    """Stream one normalized source file with strict column and ID checks."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in NORMALIZED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}: line {line_number} has extra tab-separated fields")
            entity_id = (row.get("entity_id") or "").strip()
            if not entity_id:
                raise ValueError(f"{path}: line {line_number} has an empty entity_id")
            if entity_id in seen:
                raise ValueError(f"{path}: duplicate entity_id at line {line_number}: {entity_id}")
            seen.add(entity_id)
            yield Entity(
                entity_id=entity_id,
                clean_name=row.get("clean_name") or "",
                clean_address=row.get("clean_address") or "",
                country=row.get("country") or "",
            )


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _char_trigrams(value: str) -> set[str]:
    compact = " ".join(value.split())
    if not compact:
        return set()
    if len(compact) <= 3:
        return {compact}
    return {compact[index : index + 3] for index in range(len(compact) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _symmetric_ratio(left: int, right: int) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return min(left, right) / max(left, right)


def _prefix_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    shared = 0
    for first, second in zip(left, right):
        if first != second:
            break
        shared += 1
    return shared / max(len(left), len(right))


def _sequence_ratio(left: str, right: str) -> float:
    """Use RapidFuzz when installed; retain a deterministic stdlib fallback for tests."""

    try:
        from rapidfuzz.fuzz import ratio
    except ImportError:  # pragma: no cover - CI uses this fallback when dependencies are absent
        return difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    return ratio(left, right) / 100.0


def pair_features(source: Entity, target: Entity) -> dict[str, float]:
    """Return the fixed numeric feature schema for one already-blocked pair."""

    name_tokens = _tokens(source.clean_name), _tokens(target.clean_name)
    address_tokens = _tokens(source.clean_address), _tokens(target.clean_address)
    return {
        "name_exact": float(bool(source.clean_name) and source.clean_name == target.clean_name),
        "address_exact": float(bool(source.clean_address) and source.clean_address == target.clean_address),
        "name_token_jaccard": _jaccard(*name_tokens),
        "address_token_jaccard": _jaccard(*address_tokens),
        "name_char_trigram_jaccard": _jaccard(
            _char_trigrams(source.clean_name), _char_trigrams(target.clean_name)
        ),
        "address_char_trigram_jaccard": _jaccard(
            _char_trigrams(source.clean_address), _char_trigrams(target.clean_address)
        ),
        "name_sequence_ratio": _sequence_ratio(source.clean_name, target.clean_name),
        "address_sequence_ratio": _sequence_ratio(source.clean_address, target.clean_address),
        "name_numeric_jaccard": _jaccard(
            set(_NUMBER.findall(source.clean_name)), set(_NUMBER.findall(target.clean_name))
        ),
        "address_numeric_jaccard": _jaccard(
            set(_NUMBER.findall(source.clean_address)), set(_NUMBER.findall(target.clean_address))
        ),
        "name_length_ratio": _symmetric_ratio(len(source.clean_name), len(target.clean_name)),
        "address_length_ratio": _symmetric_ratio(
            len(source.clean_address), len(target.clean_address)
        ),
        "name_token_count_ratio": _symmetric_ratio(len(name_tokens[0]), len(name_tokens[1])),
        "address_token_count_ratio": _symmetric_ratio(
            len(address_tokens[0]), len(address_tokens[1])
        ),
        "name_prefix_ratio": _prefix_ratio(source.clean_name, target.clean_name),
        "address_prefix_ratio": _prefix_ratio(source.clean_address, target.clean_address),
        "country_equal": float(bool(source.country) and source.country == target.country),
    }


class TargetStore:
    """Disk-backed target lookup used to stream candidate features without a RAM join."""

    def __init__(self, path: Path, rebuild: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if rebuild:
            path.unlink(missing_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS targets (
                entity_id TEXT PRIMARY KEY,
                clean_name TEXT NOT NULL,
                clean_address TEXT NOT NULL,
                country TEXT NOT NULL
            )
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TargetStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0])

    def add_many(self, entities: Iterable[Entity]) -> int:
        rows: list[tuple[str, str, str, str]] = []
        inserted = 0
        for entity in entities:
            rows.append((entity.entity_id, entity.clean_name, entity.clean_address, entity.country))
            if len(rows) >= 10_000:
                inserted += self._insert(rows)
                rows.clear()
        inserted += self._insert(rows)
        return inserted

    def _insert(self, rows: list[tuple[str, str, str, str]]) -> int:
        if not rows:
            return 0
        before = self.count
        with self.connection:
            self.connection.executemany("INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?)", rows)
        if self.count - before != len(rows):
            raise ValueError("target sources contain duplicate entity IDs")
        return len(rows)

    def fetch_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        found: dict[str, Entity] = {}
        for start in range(0, len(entity_ids), 900):
            batch = entity_ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT entity_id, clean_name, clean_address, country FROM targets "
                f"WHERE entity_id IN ({placeholders})",
                tuple(batch),
            )
            for entity_id, name, address, country in rows:
                found[entity_id] = Entity(entity_id, name, address, country)
        return found


class TruthStore:
    """Optional disk-backed labels. Test rows receive an empty ``is_match`` value."""

    def __init__(self, path: Path, rebuild: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if rebuild:
            path.unlink(missing_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                source1_entity_id TEXT NOT NULL,
                candidate_entity_id TEXT NOT NULL,
                PRIMARY KEY (source1_entity_id, candidate_entity_id)
            ) WITHOUT ROWID
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TruthStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0])

    def load(self, path: Path) -> int:
        if self.count:
            return self.count
        rows: list[tuple[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _require_exact_header(path, reader.fieldnames, GROUND_TRUTH_COLUMNS)
            for line_number, row in enumerate(reader, start=2):
                source_id = (row["source1_entity_id"] or "").strip()
                if not source_id:
                    raise ValueError(f"{path}: line {line_number} has an empty Source-1 ID")
                targets = [value for value in (row["matched_entity_ids"] or "").split(",") if value]
                if len(targets) != len(set(targets)):
                    raise ValueError(f"{path}: duplicate matched ID at line {line_number}")
                rows.extend((source_id, target_id) for target_id in targets)
                if len(rows) >= 50_000:
                    self._insert(rows)
                    rows.clear()
        self._insert(rows)
        return self.count

    def _insert(self, rows: list[tuple[str, str]]) -> None:
        if rows:
            with self.connection:
                self.connection.executemany("INSERT OR IGNORE INTO labels VALUES (?, ?)", rows)

    def matching_ids(self, source_id: str, candidate_ids: Sequence[str]) -> set[str]:
        matches: set[str] = set()
        for start in range(0, len(candidate_ids), 900):
            batch = candidate_ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT candidate_entity_id FROM labels WHERE source1_entity_id = ? "
                f"AND candidate_entity_id IN ({placeholders})",
                (source_id, *batch),
            )
            matches.update(row[0] for row in rows)
        return matches


def _candidate_rows(path: Path) -> Iterator[tuple[str, list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_exact_header(path, reader.fieldnames, CANDIDATE_COLUMNS)
        for line_number, row in enumerate(reader, start=2):
            source_id = (row["source1_entity_id"] or "").strip()
            values = [value for value in (row["candidate_entity_ids"] or "").split(",") if value]
            if len(values) != len(set(values)):
                raise ValueError(f"{path}: duplicate candidate ID at line {line_number}")
            yield source_id, values


def _build_target_store(store: TargetStore, target_paths: Sequence[Path]) -> int:
    if store.count:
        return store.count
    for target_path in target_paths:
        store.add_many(iter_normalized_entities(target_path))
    return store.count


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_feature_file(
    source1_path: Path,
    target_paths: Sequence[Path],
    candidate_path: Path,
    output_path: Path,
    target_database_path: Path,
    report_path: Path,
    ground_truth_path: Path | None = None,
    truth_database_path: Path | None = None,
    rebuild_target_index: bool = False,
    rebuild_truth_index: bool = False,
) -> dict[str, object]:
    """Atomically create the fixed feature TSV and its audit report."""

    if bool(ground_truth_path) != bool(truth_database_path):
        raise ValueError("ground_truth_path and truth_database_path must be supplied together")
    with TargetStore(target_database_path, rebuild=rebuild_target_index) as targets:
        target_count = _build_target_store(targets, target_paths)
        truth: TruthStore | None = None
        if ground_truth_path and truth_database_path:
            truth = TruthStore(truth_database_path, rebuild=rebuild_truth_index)
            truth.load(ground_truth_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(output_path.suffix + ".partial")
            source_count = 0
            candidate_pairs = 0
            feature_rows = 0
            positive_rows = 0
            candidate_iterator = _candidate_rows(candidate_path)
            try:
                with temporary.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    for source in iter_normalized_entities(source1_path):
                        source_count += 1
                        try:
                            candidate_source_id, candidate_ids = next(candidate_iterator)
                        except StopIteration as error:
                            raise ValueError(f"{candidate_path}: missing Source-1 row {source.entity_id}") from error
                        if candidate_source_id != source.entity_id:
                            raise ValueError(
                                f"{candidate_path}: expected Source-1 row {source.entity_id}, got {candidate_source_id}"
                            )
                        candidate_pairs += len(candidate_ids)
                        candidates = targets.fetch_many(candidate_ids)
                        unknown = sorted(set(candidate_ids) - set(candidates))
                        if unknown:
                            raise ValueError(
                                f"{candidate_path}: unknown candidate target for {source.entity_id}: {unknown[0]}"
                            )
                        positives = truth.matching_ids(source.entity_id, candidate_ids) if truth else set()
                        for candidate_id in candidate_ids:
                            values = pair_features(source, candidates[candidate_id])
                            label = "" if truth is None else str(int(candidate_id in positives))
                            positive_rows += label == "1"
                            writer.writerow(
                                {
                                    "source1_entity_id": source.entity_id,
                                    "candidate_entity_id": candidate_id,
                                    "is_match": label,
                                    **{column: f"{values[column]:.6f}" for column in PAIR_FEATURE_COLUMNS},
                                }
                            )
                            feature_rows += 1
                    try:
                        extra_source_id, _ = next(candidate_iterator)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError(f"{candidate_path}: extra Source-1 row {extra_source_id}")
                os.replace(temporary, output_path)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            if truth:
                truth.close()

    report = {
        "source1_records": source_count,
        "target_records": target_count,
        "candidate_pairs": candidate_pairs,
        "feature_rows": feature_rows,
        "positive_feature_rows": positive_rows if ground_truth_path else None,
        "labels_present": bool(ground_truth_path),
        "feature_columns": list(FEATURE_COLUMNS),
        "country_usage": "country_equal is a feature only; it never filters candidates",
        "output": str(output_path),
    }
    _write_json_atomic(report, report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1-path", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True, action="append")
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-database-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--ground-truth-path", type=Path)
    parser.add_argument("--truth-database-path", type=Path)
    parser.add_argument("--rebuild-target-index", action="store_true")
    parser.add_argument("--rebuild-truth-index", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = generate_feature_file(
            source1_path=args.source1_path,
            target_paths=args.target_path,
            candidate_path=args.candidate_path,
            output_path=args.output_path,
            target_database_path=args.target_database_path,
            report_path=args.report_path,
            ground_truth_path=args.ground_truth_path,
            truth_database_path=args.truth_database_path,
            rebuild_target_index=args.rebuild_target_index,
            rebuild_truth_index=args.rebuild_truth_index,
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
