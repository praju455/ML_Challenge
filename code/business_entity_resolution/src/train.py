"""Train and calibrate a local LightGBM candidate-pair classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from src.features import FEATURE_COLUMNS, PAIR_FEATURE_COLUMNS
from src.metric import evaluate_match_lists, load_match_lists


def _sample_source(source_id: str, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(f"{seed}\0{source_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / (2**64) < fraction


def load_labeled_features(path: Path, sample_fraction: float = 1.0, seed: int = 2026):
    """Load a deterministic Source-1 sample; never split rows from one entity."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("install requirements.txt before training") from error
    values: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FEATURE_COLUMNS:
            raise ValueError(f"{path}: unexpected feature header")
        for line_number, row in enumerate(reader, start=2):
            source_id = row["source1_entity_id"]
            label = row["is_match"]
            if label not in {"0", "1"}:
                raise ValueError(f"{path}: line {line_number} has no train label")
            if not _sample_source(source_id, sample_fraction, seed):
                continue
            values.append([float(row[column]) for column in PAIR_FEATURE_COLUMNS])
            labels.append(int(label))
            groups.append(source_id)
    if not values or len(set(labels)) < 2:
        raise ValueError("sample contains fewer than two label classes")
    return np.asarray(values, dtype=np.float32), np.asarray(labels), np.asarray(groups, dtype=object)


def best_threshold(
    groups: Sequence[str], candidate_ids: Sequence[str], probabilities: Sequence[float], truth: dict[str, set[str]],
    thresholds: Sequence[float],
) -> tuple[float, dict[str, object]]:
    all_groups = set(groups)
    truth_subset = {source_id: truth[source_id] for source_id in all_groups}
    best: tuple[float, dict[str, object]] | None = None
    for threshold in thresholds:
        predicted = {source_id: set() for source_id in all_groups}
        for source_id, candidate_id, probability in zip(groups, candidate_ids, probabilities):
            if probability >= threshold:
                predicted[source_id].add(candidate_id)
        report = evaluate_match_lists(truth_subset, predicted)
        if best is None or float(report["macro_f0_5"]) > float(best[1]["macro_f0_5"]):
            best = (float(threshold), report)
    if best is None:
        raise ValueError("no thresholds supplied")
    return best


def train_model(feature_path: Path, ground_truth_path: Path, model_path: Path, report_path: Path,
                sample_fraction: float = 0.05, seed: int = 2026) -> dict[str, object]:
    """Fit → calibrate → tune threshold using disjoint Source-1 entity groups."""
    try:
        import numpy as np
        from lightgbm import LGBMClassifier
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as error:
        raise RuntimeError("install requirements.txt with lightgbm and joblib before training") from error
    x, y, groups = load_labeled_features(feature_path, sample_fraction, seed)
    candidate_ids: list[str] = []
    with feature_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if _sample_source(row["source1_entity_id"], sample_fraction, seed):
                candidate_ids.append(row["candidate_entity_id"])
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_index, holdout_index = next(outer.split(x, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    calibration_local, validation_local = next(inner.split(x[holdout_index], y[holdout_index], groups[holdout_index]))
    calibration_index, validation_index = holdout_index[calibration_local], holdout_index[validation_local]
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    model.fit(x[train_index], y[train_index])
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(x[calibration_index], y[calibration_index])
    probabilities = calibrator.predict_proba(x[validation_index])[:, 1]
    truth = load_match_lists(ground_truth_path)
    validation_groups = groups[validation_index]
    if any(source_id not in truth for source_id in validation_groups):
        raise ValueError("ground truth is missing sampled Source-1 IDs")
    threshold, metrics = best_threshold(validation_groups, np.asarray(candidate_ids)[validation_index], probabilities,
                                        truth, np.linspace(0.05, 0.95, 37))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrator": calibrator, "threshold": threshold, "feature_columns": PAIR_FEATURE_COLUMNS}, model_path)
    report = {"sample_fraction": sample_fraction, "feature_rows": int(len(y)), "source1_entities": int(len(set(groups))),
              "split_source1_entities": {"train": int(len(set(groups[train_index]))), "calibration": int(len(set(groups[calibration_index]))), "validation": int(len(set(validation_groups)))},
              "threshold": threshold, "validation": metrics, "model": str(model_path)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--ground-truth-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        report = train_model(**vars(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
