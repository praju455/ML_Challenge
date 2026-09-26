"""Validate an entity-resolution submission before leaderboard upload.

The leaderboard accepts only ``matching_results.tsv``. Candidate files use a
different header and are retained for the final package, not uploaded during the
challenge. This validator checks the full test-ID universe without loading it all
into Python memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence


MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
SOURCE_ID_HEADER = "entity_id"
MAX_EXAMPLES = 10


def _open_tsv(path: Path, expected_header: tuple[str, ...]) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        actual_header = tuple(reader.fieldnames or ())
        if actual_header != expected_header:
            raise ValueError(f"{path}: expected header {expected_header}, got {actual_header}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}: line {line_number} has extra tab-separated fields")
            yield {key: row.get(key) or "" for key in expected_header}


def _iter_source_ids(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if SOURCE_ID_HEADER not in (reader.fieldnames or ()):
            raise ValueError(f"{path}: missing {SOURCE_ID_HEADER!r} column")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}: line {line_number} has extra tab-separated fields")
            entity_id = (row.get(SOURCE_ID_HEADER) or "").strip()
            if not entity_id:
                raise ValueError(f"{path}: line {line_number} has an empty entity_id")
            yield entity_id


def _record_issue(issues: Counter[str], examples: dict[str, list[str]], kind: str, value: str) -> None:
    issues[kind] += 1
    if len(examples.setdefault(kind, [])) < MAX_EXAMPLES:
        examples[kind].append(value)


def _split_ids(value: str, source_id: str, issues: Counter[str], examples: dict[str, list[str]]) -> list[str]:
    if not value:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for raw_id in value.split(","):
        candidate_id = raw_id.strip()
        if not candidate_id:
            _record_issue(issues, examples, "empty_list_item", source_id)
            continue
        if raw_id != candidate_id:
            _record_issue(issues, examples, "whitespace_in_list_item", f"{source_id}:{raw_id!r}")
        if candidate_id in seen:
            _record_issue(issues, examples, "duplicate_target_in_row", f"{source_id}:{candidate_id}")
            continue
        seen.add(candidate_id)
        output.append(candidate_id)
    return output


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE source1 (entity_id TEXT PRIMARY KEY, seen INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID;
        CREATE TABLE targets (entity_id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE candidate_pairs (
            source1_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            PRIMARY KEY (source1_entity_id, target_entity_id)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _insert_ids(connection: sqlite3.Connection, table: str, entity_ids: Iterator[str]) -> tuple[int, list[str]]:
    rows = 0
    duplicates: list[str] = []
    batch: list[tuple[str]] = []
    for entity_id in entity_ids:
        batch.append((entity_id,))
        if len(batch) >= 50_000:
            rows, duplicates = _flush_ids(connection, table, batch, rows, duplicates)
            batch.clear()
    if batch:
        rows, duplicates = _flush_ids(connection, table, batch, rows, duplicates)
    return rows, duplicates


def _flush_ids(
    connection: sqlite3.Connection,
    table: str,
    batch: list[tuple[str]],
    rows: int,
    duplicates: list[str],
) -> tuple[int, list[str]]:
    for entity_id, in batch:
        cursor = connection.execute(f"INSERT OR IGNORE INTO {table}(entity_id) VALUES (?)", (entity_id,))
        if cursor.rowcount:
            rows += 1
        elif len(duplicates) < MAX_EXAMPLES:
            duplicates.append(entity_id)
    connection.commit()
    return rows, duplicates


def validate_submission(
    matching_results_path: Path,
    test_source1_path: Path,
    test_source2_path: Path,
    test_source3_path: Path,
    candidate_pairs_path: Path | None = None,
    work_dir: Path | None = None,
) -> dict[str, object]:
    """Return an auditable validation report; callers decide whether to upload."""

    issues: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="entity-resolution-validation-", dir=work_dir))
    database_path = root / "validation.sqlite"
    connection = _connect(database_path)
    try:
        source1_count, source1_duplicates = _insert_ids(connection, "source1", _iter_source_ids(test_source1_path))
        target2_count, target2_duplicates = _insert_ids(connection, "targets", _iter_source_ids(test_source2_path))
        target3_count, target3_duplicates = _insert_ids(connection, "targets", _iter_source_ids(test_source3_path))
        for duplicate in source1_duplicates:
            _record_issue(issues, examples, "duplicate_input_source1_id", duplicate)
        for duplicate in (*target2_duplicates, *target3_duplicates):
            _record_issue(issues, examples, "duplicate_input_target_id", duplicate)

        if candidate_pairs_path is not None:
            _load_candidates(connection, candidate_pairs_path, issues, examples)

        matching_rows = 0
        predicted_pairs = 0
        for row in _open_tsv(matching_results_path, MATCHING_HEADER):
            matching_rows += 1
            source_id = row["source1_entity_id"]
            cursor = connection.execute("UPDATE source1 SET seen = seen + 1 WHERE entity_id = ?", (source_id,))
            if not cursor.rowcount:
                _record_issue(issues, examples, "unknown_source1_id", source_id)
            elif connection.execute("SELECT seen FROM source1 WHERE entity_id = ?", (source_id,)).fetchone()[0] > 1:
                _record_issue(issues, examples, "duplicate_source1_row", source_id)

            for target_id in _split_ids(row["matched_entity_ids"], source_id, issues, examples):
                predicted_pairs += 1
                if not connection.execute("SELECT 1 FROM targets WHERE entity_id = ?", (target_id,)).fetchone():
                    _record_issue(issues, examples, "unknown_target_id", f"{source_id}:{target_id}")
                if candidate_pairs_path is not None and not connection.execute(
                    "SELECT 1 FROM candidate_pairs WHERE source1_entity_id = ? AND target_entity_id = ?",
                    (source_id, target_id),
                ).fetchone():
                    _record_issue(issues, examples, "prediction_not_in_candidate_pairs", f"{source_id}:{target_id}")

        missing = [
            row[0]
            for row in connection.execute(
                "SELECT entity_id FROM source1 WHERE seen = 0 ORDER BY entity_id LIMIT ?", (MAX_EXAMPLES,)
            )
        ]
        missing_count = int(connection.execute("SELECT COUNT(*) FROM source1 WHERE seen = 0").fetchone()[0])
        if missing_count:
            issues["missing_source1_rows"] += missing_count
            examples["missing_source1_rows"] = missing

        report = {
            "status": "PASS" if not issues else "FAIL",
            "matching_rows": matching_rows,
            "expected_source1_rows": source1_count,
            "target_entities": target2_count + target3_count,
            "predicted_pairs": predicted_pairs,
            "candidate_pairs_checked": candidate_pairs_path is not None,
            "issues": dict(sorted(issues.items())),
            "examples": examples,
        }
        return report
    finally:
        connection.close()
        for path in root.glob("*"):
            path.unlink(missing_ok=True)
        root.rmdir()


def _load_candidates(
    connection: sqlite3.Connection,
    candidate_pairs_path: Path,
    issues: Counter[str],
    examples: dict[str, list[str]],
) -> None:
    batch: list[tuple[str, str]] = []
    for row in _open_tsv(candidate_pairs_path, CANDIDATE_HEADER):
        source_id = row["source1_entity_id"]
        if not connection.execute("SELECT 1 FROM source1 WHERE entity_id = ?", (source_id,)).fetchone():
            _record_issue(issues, examples, "candidate_unknown_source1_id", source_id)
        for target_id in _split_ids(row["candidate_entity_ids"], source_id, issues, examples):
            if not connection.execute("SELECT 1 FROM targets WHERE entity_id = ?", (target_id,)).fetchone():
                _record_issue(issues, examples, "candidate_unknown_target_id", f"{source_id}:{target_id}")
            batch.append((source_id, target_id))
            if len(batch) >= 50_000:
                with connection:
                    connection.executemany(
                        "INSERT OR IGNORE INTO candidate_pairs VALUES (?, ?)", batch
                    )
                batch.clear()
    if batch:
        with connection:
            connection.executemany("INSERT OR IGNORE INTO candidate_pairs VALUES (?, ?)", batch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-results", type=Path, required=True)
    parser.add_argument("--test-source1", type=Path, required=True)
    parser.add_argument("--test-source2", type=Path, required=True)
    parser.add_argument("--test-source3", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path)
    parser.add_argument("--work-dir", type=Path, help="Temporary SQLite location with ample free disk")
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_submission(
            matching_results_path=args.matching_results,
            test_source1_path=args.test_source1,
            test_source2_path=args.test_source2,
            test_source3_path=args.test_source3,
            candidate_pairs_path=args.candidate_pairs,
            work_dir=args.work_dir,
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
