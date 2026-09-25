"""Run all Phase 1 stages inside a SageMaker Processing job.

The Processing input must contain the competition ``train/`` and ``test/``
directories. Only final normalized files, learned equivalences, and audit reports
are written to the Processing output channel; the large SQLite index is ephemeral.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ENTRYPOINT_DIR = Path(__file__).resolve().parent
LOCAL_PROJECT_ROOT = ENTRYPOINT_DIR.parent
for import_root in (ENTRYPOINT_DIR, LOCAL_PROJECT_ROOT):
    if (import_root / "src").is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.mine_equivalences import MiningConfig, run_mining  # noqa: E402
from src.normalize import transform_all  # noqa: E402


DEFAULT_DATA_DIR = Path("/opt/ml/processing/input/dataset")
DEFAULT_WORK_DIR = Path("/opt/ml/processing/work")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/processing/output/results")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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


def run_phase1(
    data_dir: Path,
    work_dir: Path,
    output_dir: Path,
    max_rows_per_file: int | None = None,
    max_ground_truth_rows: int | None = None,
) -> dict[str, object]:
    """Run base normalization, mine equivalences, and apply accepted mappings."""

    started_at = datetime.now(timezone.utc)
    base_dir = work_dir / "normalized-base"
    database_path = work_dir / "equivalence-records.sqlite"
    artifacts_dir = output_dir / "artifacts" / "normalization" / "full"
    final_dir = output_dir / "data" / "processed" / "final"
    equivalence_path = artifacts_dir / "token_equivalences.tsv"

    base_summary = transform_all(
        data_dir=data_dir,
        output_dir=base_dir,
        report_path=work_dir / "base-normalization-report.json",
        max_rows_per_file=max_rows_per_file,
    )
    mining_summary = run_mining(
        normalized_data_dir=base_dir,
        ground_truth_path=data_dir / "train" / "train_ground_truth.tsv",
        database_path=database_path,
        output_path=equivalence_path,
        report_path=artifacts_dir / "equivalence-report.json",
        config=MiningConfig(),
        max_ground_truth_rows=max_ground_truth_rows,
    )
    final_summary = transform_all(
        data_dir=data_dir,
        output_dir=final_dir,
        report_path=artifacts_dir / "normalization-report.json",
        equivalence_path=equivalence_path,
        max_rows_per_file=max_rows_per_file,
    )

    completed_at = datetime.now(timezone.utc)
    manifest = {
        "phase": "phase1",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
        "bounded_run": max_rows_per_file is not None or max_ground_truth_rows is not None,
        "max_rows_per_file": max_rows_per_file,
        "max_ground_truth_rows": max_ground_truth_rows,
        "base_normalization": base_summary,
        "equivalence_mining": mining_summary,
        "final_normalization": final_summary,
    }
    _write_json_atomic(manifest, output_dir / "phase1-manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows-per-file", type=_positive_int)
    parser.add_argument("--max-ground-truth-rows", type=_positive_int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_phase1(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        max_rows_per_file=args.max_rows_per_file,
        max_ground_truth_rows=args.max_ground_truth_rows,
    )
    json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
