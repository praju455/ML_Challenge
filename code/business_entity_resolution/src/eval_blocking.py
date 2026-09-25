"""Evaluate blocking recall ceiling and reduction ratio on labeled entities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Iterator, Sequence


def _in_validation(entity_id: str, fraction: float, seed: int) -> bool:
    payload = f"{seed}\0{entity_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return bucket / (2**64) < fraction


def iter_ground_truth(path: Path) -> Iterator[tuple[str, set[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ("source1_entity_id", "matched_entity_ids")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path}: expected exact TSV header {expected}")
        for row in reader:
            matches = {value for value in (row["matched_entity_ids"] or "").split(",") if value}
            yield row["source1_entity_id"], matches


def load_candidates(path: Path) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ("source1_entity_id", "candidate_entity_ids")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path}: expected exact TSV header {expected}")
        for row_number, row in enumerate(reader, start=2):
            source_id = row["source1_entity_id"]
            if source_id in candidates:
                raise ValueError(f"{path}: duplicate Source-1 row at line {row_number}: {source_id}")
            values = [value for value in (row["candidate_entity_ids"] or "").split(",") if value]
            if len(values) != len(set(values)):
                raise ValueError(f"{path}: duplicate candidate ID at line {row_number}")
            candidates[source_id] = set(values)
    return candidates


def evaluate_blocking(
    ground_truth_path: Path,
    candidate_path: Path,
    target_count: int,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> dict[str, object]:
    if not 0 < validation_fraction <= 1:
        raise ValueError("validation_fraction must be in (0, 1]")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    candidates = load_candidates(candidate_path)
    entity_count = 0
    true_edges = 0
    recovered_edges = 0
    complete_entities = 0
    candidate_pairs = 0
    counts: list[int] = []
    zero_match_entities = 0
    zero_match_with_no_candidates = 0
    missing_rows: list[str] = []

    for source_id, truth in iter_ground_truth(ground_truth_path):
        if not _in_validation(source_id, validation_fraction, seed):
            continue
        entity_count += 1
        if source_id not in candidates:
            missing_rows.append(source_id)
            predicted_candidates: set[str] = set()
        else:
            predicted_candidates = candidates[source_id]
        count = len(predicted_candidates)
        counts.append(count)
        candidate_pairs += count
        true_edges += len(truth)
        recovered_edges += len(truth & predicted_candidates)
        complete_entities += truth <= predicted_candidates
        if not truth:
            zero_match_entities += 1
            zero_match_with_no_candidates += not predicted_candidates

    if not entity_count:
        raise ValueError("validation split contains no Source-1 entities")
    search_space = entity_count * target_count
    sorted_counts = sorted(counts)
    report = {
        "validation_entities": entity_count,
        "target_entities": target_count,
        "true_match_edges": true_edges,
        "recovered_true_match_edges": recovered_edges,
        "blocking_recall_ceiling": recovered_edges / true_edges if true_edges else 1.0,
        "complete_entity_coverage": complete_entities / entity_count,
        "candidate_pairs": candidate_pairs,
        "cartesian_pairs": search_space,
        "reduction_ratio": 1.0 - (candidate_pairs / search_space),
        "candidate_count": {
            "minimum": min(counts),
            "median": statistics.median(sorted_counts),
            "maximum": max(counts),
            "mean": statistics.fmean(counts),
        },
        "zero_match_entities": zero_match_entities,
        "zero_match_entities_with_empty_candidates": zero_match_with_no_candidates,
        "missing_candidate_rows": len(missing_rows),
        "missing_candidate_examples": missing_rows[:5],
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-path", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_blocking(
            ground_truth_path=args.ground_truth_path,
            candidate_path=args.candidate_path,
            target_count=args.target_count,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"blocking recall ceiling: {report['blocking_recall_ceiling']:.6f}")
    print(f"reduction ratio: {report['reduction_ratio']:.6f}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
