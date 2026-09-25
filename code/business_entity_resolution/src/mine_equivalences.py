"""Mine token equivalences from aligned, normalized training records.

The miner uses ground-truth match edges as evidence and counts token frequencies in
the training corpus. A disk-backed SQLite index avoids loading millions of records
into RAM. No country names, legal suffixes, address terms, or language-specific
vocabularies are embedded in the algorithm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


TRAIN_FILES = (
    (1, Path("train/train_source1.tsv")),
    (2, Path("train/train_source2.tsv")),
    (3, Path("train/train_source3.tsv")),
)
NORMALIZED_COLUMNS = ("entity_id", "clean_name", "clean_address")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
OUTPUT_COLUMNS = (
    "field",
    "variant",
    "canonical",
    "mapping_type",
    "support",
    "variant_frequency",
    "canonical_frequency",
    "purity",
    "coverage",
    "confidence",
)


@dataclass(frozen=True)
class MiningConfig:
    min_support: int = 50
    min_purity: float = 0.70
    min_confidence: float = 0.65
    min_drop_support: int = 200
    min_drop_purity: float = 0.90
    min_drop_coverage: float = 0.20
    batch_entities: int = 5_000
    query_batch_size: int = 800
    insert_batch_size: int = 50_000
    max_evidence_pairs: int = 2_000_000


def _validate_columns(fieldnames: Sequence[str] | None, required: Sequence[str], path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path}: missing TSV header")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-131072")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            entity_id TEXT PRIMARY KEY,
            source INTEGER NOT NULL,
            clean_name TEXT NOT NULL,
            clean_address TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS loaded_sources (
            source INTEGER PRIMARY KEY,
            relative_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            rows INTEGER NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS records_source_idx ON records(source)")
    connection.commit()
    return connection


def _iter_normalized_rows(path: Path) -> Iterator[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _validate_columns(reader.fieldnames, NORMALIZED_COLUMNS, path)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}: row {row_number} has extra tab-separated fields")
            yield (
                row["entity_id"],
                row.get("clean_name") or "",
                row.get("clean_address") or "",
            )


def build_record_index(
    normalized_data_dir: Path,
    database_path: Path,
    insert_batch_size: int = 50_000,
) -> dict[str, int]:
    """Build or resume the SQLite record index one source file at a time."""

    connection = _connect(database_path)
    loaded = {
        int(source): (str(relative_path), int(file_size), int(modified_ns), int(rows))
        for source, relative_path, file_size, modified_ns, rows in connection.execute(
            "SELECT source, relative_path, file_size, modified_ns, rows FROM loaded_sources"
        )
    }
    report: dict[str, int] = {}
    try:
        for source, relative_path in TRAIN_FILES:
            path = normalized_data_dir / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            fingerprint = (str(relative_path), stat.st_size, stat.st_mtime_ns)
            if source in loaded and loaded[source][:3] == fingerprint:
                report[str(relative_path)] = loaded[source][3]
                continue

            connection.execute("DELETE FROM records WHERE source = ?", (source,))
            connection.commit()
            batch: list[tuple[str, int, str, str]] = []
            rows = 0
            for entity_id, clean_name, clean_address in _iter_normalized_rows(path):
                batch.append((entity_id, source, clean_name, clean_address))
                if len(batch) >= insert_batch_size:
                    connection.executemany(
                        "INSERT INTO records(entity_id, source, clean_name, clean_address) VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    connection.commit()
                    rows += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO records(entity_id, source, clean_name, clean_address) VALUES (?, ?, ?, ?)",
                    batch,
                )
                connection.commit()
                rows += len(batch)
            connection.execute(
                """
                INSERT OR REPLACE INTO loaded_sources(
                    source, relative_path, file_size, modified_ns, rows
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (source, str(relative_path), stat.st_size, stat.st_mtime_ns, rows),
            )
            connection.commit()
            report[str(relative_path)] = rows
    finally:
        connection.close()
    return report


def _batched(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_records(
    connection: sqlite3.Connection,
    entity_ids: Iterable[str],
    query_batch_size: int,
) -> dict[str, tuple[str, str]]:
    unique_ids = sorted(set(entity_ids))
    records: dict[str, tuple[str, str]] = {}
    for batch in _batched(unique_ids, query_batch_size):
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT entity_id, clean_name, clean_address FROM records WHERE entity_id IN ({placeholders})"
        for entity_id, clean_name, clean_address in connection.execute(query, tuple(batch)):
            records[entity_id] = (clean_name, clean_address)
    return records


def _single_token_event(left: str, right: str) -> tuple[str, str] | None:
    """Return the only unmatched token on each side, allowing one side to be empty."""

    left_only = list((Counter(left.split()) - Counter(right.split())).elements())
    right_only = list((Counter(right.split()) - Counter(left.split())).elements())
    if len(left_only) > 1 or len(right_only) > 1:
        return None
    if not left_only and not right_only:
        return None
    left_token = left_only[0] if left_only else ""
    right_token = right_only[0] if right_only else ""
    if left_token and not left_token.isalpha():
        return None
    if right_token and not right_token.isalpha():
        return None
    if left_token == right_token:
        return None
    return tuple(sorted((left_token, right_token)))


def _prune_evidence(counter: Counter[tuple[str, str, str]], keep: int) -> None:
    retained = counter.most_common(keep)
    counter.clear()
    counter.update(dict(retained))


def collect_alignment_evidence(
    database_path: Path,
    ground_truth_path: Path,
    config: MiningConfig,
    max_ground_truth_rows: int | None = None,
) -> tuple[Counter[tuple[str, str, str]], dict[str, int]]:
    """Count one-token substitutions/deletions across true matched record pairs."""

    evidence: Counter[tuple[str, str, str]] = Counter()
    report = {
        "ground_truth_rows": 0,
        "match_edges": 0,
        "resolved_edges": 0,
        "missing_record_edges": 0,
        "name_events": 0,
        "address_events": 0,
    }
    connection = sqlite3.connect(database_path)

    def process_batch(batch: list[tuple[str, list[str]]]) -> None:
        ids = [source1_id for source1_id, _ in batch]
        ids.extend(target for _, targets in batch for target in targets)
        records = fetch_records(connection, ids, config.query_batch_size)
        for source1_id, targets in batch:
            source_record = records.get(source1_id)
            for target in targets:
                report["match_edges"] += 1
                target_record = records.get(target)
                if source_record is None or target_record is None:
                    report["missing_record_edges"] += 1
                    continue
                report["resolved_edges"] += 1
                for field, left, right in (
                    ("name", source_record[0], target_record[0]),
                    ("address", source_record[1], target_record[1]),
                ):
                    event = _single_token_event(left, right)
                    if event is not None:
                        evidence[(field, event[0], event[1])] += 1
                        report[f"{field}_events"] += 1
        if len(evidence) > config.max_evidence_pairs:
            _prune_evidence(evidence, config.max_evidence_pairs // 2)

    batch: list[tuple[str, list[str]]] = []
    try:
        with ground_truth_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _validate_columns(reader.fieldnames, GROUND_TRUTH_COLUMNS, ground_truth_path)
            for row_number, row in enumerate(reader, start=1):
                if max_ground_truth_rows is not None and row_number > max_ground_truth_rows:
                    break
                targets = [token.strip() for token in row["matched_entity_ids"].split(",") if token.strip()]
                batch.append((row["source1_entity_id"], targets))
                report["ground_truth_rows"] += 1
                if len(batch) >= config.batch_entities:
                    process_batch(batch)
                    batch.clear()
            if batch:
                process_batch(batch)
    finally:
        connection.close()
    return evidence, report


def collect_candidate_frequencies(
    normalized_data_dir: Path,
    evidence: Mapping[tuple[str, str, str], int],
    min_support: int,
) -> dict[str, Counter[str]]:
    """Count only tokens participating in supported alignment events."""

    candidates = {"name": set(), "address": set()}
    for (field, left, right), support in evidence.items():
        if support < min_support:
            continue
        if left:
            candidates[field].add(left)
        if right:
            candidates[field].add(right)

    frequencies = {"name": Counter(), "address": Counter()}
    for _, relative_path in TRAIN_FILES:
        path = normalized_data_dir / relative_path
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _validate_columns(reader.fieldnames, NORMALIZED_COLUMNS, path)
            for row in reader:
                for field, column in (("name", "clean_name"), ("address", "clean_address")):
                    wanted = candidates[field]
                    if wanted:
                        frequencies[field].update(token for token in (row.get(column) or "").split() if token in wanted)
    return frequencies


def select_equivalences(
    evidence: Mapping[tuple[str, str, str], int],
    frequencies: Mapping[str, Counter[str]],
    config: MiningConfig,
) -> list[dict[str, object]]:
    """Select at most one high-confidence canonical mapping per variant token."""

    event_totals: Counter[tuple[str, str]] = Counter()
    for (field, left, right), support in evidence.items():
        if left:
            event_totals[(field, left)] += support
        if right:
            event_totals[(field, right)] += support

    best: dict[tuple[str, str], dict[str, object]] = {}
    for (field, left, right), support in evidence.items():
        if support < config.min_support:
            continue
        if not left:
            variant, canonical, mapping_type = right, "", "drop"
        elif not right:
            variant, canonical, mapping_type = left, "", "drop"
        elif len(left) < len(right):
            variant, canonical, mapping_type = left, right, "replace"
        elif len(right) < len(left):
            variant, canonical, mapping_type = right, left, "replace"
        else:
            left_frequency = frequencies[field][left]
            right_frequency = frequencies[field][right]
            variant, canonical = (
                (left, right) if left_frequency <= right_frequency else (right, left)
            )
            mapping_type = "replace"

        variant_frequency = frequencies[field][variant]
        canonical_frequency = frequencies[field][canonical] if canonical else 0
        purity = support / max(1, event_totals[(field, variant)])
        coverage = min(1.0, support / max(1, variant_frequency))
        confidence = 0.75 * purity + 0.25 * coverage

        if mapping_type == "drop":
            if support < config.min_drop_support:
                continue
            if purity < config.min_drop_purity or coverage < config.min_drop_coverage:
                continue
        elif purity < config.min_purity or confidence < config.min_confidence:
            continue

        row = {
            "field": field,
            "variant": variant,
            "canonical": canonical,
            "mapping_type": mapping_type,
            "support": support,
            "variant_frequency": variant_frequency,
            "canonical_frequency": canonical_frequency,
            "purity": round(purity, 6),
            "coverage": round(coverage, 6),
            "confidence": round(confidence, 6),
        }
        key = (field, variant)
        incumbent = best.get(key)
        if incumbent is None or (row["confidence"], row["support"], row["canonical"]) > (
            incumbent["confidence"], incumbent["support"], incumbent["canonical"]
        ):
            best[key] = row
    return sorted(best.values(), key=lambda row: (str(row["field"]), str(row["variant"])))


def _write_tsv_atomic(rows: Iterable[Mapping[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})
                count += 1
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_mining(
    normalized_data_dir: Path,
    ground_truth_path: Path,
    database_path: Path,
    output_path: Path,
    report_path: Path,
    config: MiningConfig,
    max_ground_truth_rows: int | None = None,
) -> dict[str, object]:
    indexed_rows = build_record_index(
        normalized_data_dir,
        database_path,
        insert_batch_size=config.insert_batch_size,
    )
    evidence, evidence_report = collect_alignment_evidence(
        database_path,
        ground_truth_path,
        config,
        max_ground_truth_rows=max_ground_truth_rows,
    )
    frequencies = collect_candidate_frequencies(
        normalized_data_dir,
        evidence,
        min_support=config.min_support,
    )
    rows = select_equivalences(evidence, frequencies, config)
    output_count = _write_tsv_atomic(rows, output_path)
    summary = {
        "config": asdict(config),
        "database_path": str(database_path),
        "indexed_rows": indexed_rows,
        "alignment": evidence_report,
        "distinct_evidence_pairs": len(evidence),
        "selected_equivalences": output_count,
        "output_path": str(output_path),
    }
    _write_json_atomic(summary, report_path)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-data-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--min-support", type=_positive_int, default=MiningConfig.min_support)
    parser.add_argument("--min-purity", type=_probability, default=MiningConfig.min_purity)
    parser.add_argument("--min-confidence", type=_probability, default=MiningConfig.min_confidence)
    parser.add_argument("--min-drop-support", type=_positive_int, default=MiningConfig.min_drop_support)
    parser.add_argument("--min-drop-purity", type=_probability, default=MiningConfig.min_drop_purity)
    parser.add_argument("--min-drop-coverage", type=_probability, default=MiningConfig.min_drop_coverage)
    parser.add_argument("--batch-entities", type=_positive_int, default=MiningConfig.batch_entities)
    parser.add_argument("--query-batch-size", type=_positive_int, default=MiningConfig.query_batch_size)
    parser.add_argument("--insert-batch-size", type=_positive_int, default=MiningConfig.insert_batch_size)
    parser.add_argument("--max-ground-truth-rows", type=_positive_int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MiningConfig(
        min_support=args.min_support,
        min_purity=args.min_purity,
        min_confidence=args.min_confidence,
        min_drop_support=args.min_drop_support,
        min_drop_purity=args.min_drop_purity,
        min_drop_coverage=args.min_drop_coverage,
        batch_entities=args.batch_entities,
        query_batch_size=args.query_batch_size,
        insert_batch_size=args.insert_batch_size,
    )
    summary = run_mining(
        normalized_data_dir=args.normalized_data_dir,
        ground_truth_path=args.ground_truth_path,
        database_path=args.database_path,
        output_path=args.output_path,
        report_path=args.report_path,
        config=config,
        max_ground_truth_rows=args.max_ground_truth_rows,
    )
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
