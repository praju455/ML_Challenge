"""Exact per-entity macro F0.5 metric used for validation and submission checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence


MATCH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def entity_f05(truth: set[str], predicted: set[str]) -> float:
    """Score one Source-1 entity using the competition's F0.5 edge-case rules.

    A correctly empty list earns 1.0. An incorrect empty/non-empty decision earns
    0.0. For a true singleton, any false-positive candidate earns 0.0 even if the
    true target is present. Other non-empty entities use the ordinary F-beta formula
    with beta=0.5.
    """

    if not truth:
        return 1.0 if not predicted else 0.0
    if len(truth) == 1 and predicted - truth:
        return 0.0
    if not predicted:
        return 0.0
    true_positives = len(truth & predicted)
    if not true_positives:
        return 0.0
    precision = true_positives / len(predicted)
    recall = true_positives / len(truth)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def load_match_lists(path: Path) -> dict[str, set[str]]:
    """Load strict, deduplicated Source-1 match lists from a TSV artifact."""

    rows: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MATCH_COLUMNS:
            raise ValueError(f"{path}: expected exact TSV header {MATCH_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            source_id = (row["source1_entity_id"] or "").strip()
            if not source_id:
                raise ValueError(f"{path}: empty Source-1 ID at line {line_number}")
            if source_id in rows:
                raise ValueError(f"{path}: duplicate Source-1 ID at line {line_number}: {source_id}")
            matched = [value for value in (row["matched_entity_ids"] or "").split(",") if value]
            if len(matched) != len(set(matched)):
                raise ValueError(f"{path}: duplicate matched ID at line {line_number}: {source_id}")
            rows[source_id] = set(matched)
    return rows


def evaluate_match_lists(
    truth_by_source: Mapping[str, set[str]],
    predicted_by_source: Mapping[str, set[str]],
    group_by_source: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return macro F0.5 plus auditable precision/recall and group diagnostics."""

    truth_ids = set(truth_by_source)
    predicted_ids = set(predicted_by_source)
    missing = sorted(truth_ids - predicted_ids)
    extra = sorted(predicted_ids - truth_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing predictions for {len(missing)} Source-1 IDs (e.g. {missing[0]})")
        if extra:
            details.append(f"unexpected predictions for {len(extra)} Source-1 IDs (e.g. {extra[0]})")
        raise ValueError("; ".join(details))

    scores: list[float] = []
    true_positive_count = 0
    predicted_count = 0
    truth_count = 0
    groups: dict[str, list[float]] = {}
    for source_id in sorted(truth_ids):
        truth = set(truth_by_source[source_id])
        predicted = set(predicted_by_source[source_id])
        score = entity_f05(truth, predicted)
        scores.append(score)
        true_positive_count += len(truth & predicted)
        predicted_count += len(predicted)
        truth_count += len(truth)
        if group_by_source is not None:
            group = group_by_source.get(source_id, "<missing>")
            groups.setdefault(group, []).append(score)

    report: dict[str, object] = {
        "entities": len(scores),
        "macro_f0_5": statistics.fmean(scores) if scores else 0.0,
        "micro_precision": true_positive_count / predicted_count if predicted_count else 0.0,
        "micro_recall": true_positive_count / truth_count if truth_count else 0.0,
        "true_positive_pairs": true_positive_count,
        "predicted_pairs": predicted_count,
        "true_pairs": truth_count,
        "correct_empty_entities": sum(
            not truth_by_source[source_id] and not predicted_by_source[source_id]
            for source_id in truth_ids
        ),
        "wrong_empty_entities": sum(
            not truth_by_source[source_id] and bool(predicted_by_source[source_id])
            for source_id in truth_ids
        ),
    }
    if group_by_source is not None:
        report["macro_f0_5_by_group"] = {
            group: statistics.fmean(values) for group, values in sorted(groups.items())
        }
    return report


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-path", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_match_lists(
            load_match_lists(args.ground_truth_path), load_match_lists(args.prediction_path)
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.report_path:
        _write_json_atomic(report, args.report_path)
    print(f"macro F0.5: {float(report['macro_f0_5']):.6f}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
