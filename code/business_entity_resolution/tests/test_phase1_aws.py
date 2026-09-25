from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from aws.phase1_job import run_phase1
from aws.submit_phase1 import build_job_plan, main, parse_s3_uri


ROLE_ARN = "arn:aws:iam::123456789012:role/test-sagemaker-role"


def _launcher_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "role_arn": ROLE_ARN,
        "input_s3_uri": "s3://challenge-bucket/raw/dataset",
        "output_s3_prefix": "s3://challenge-bucket/runs/phase1",
        "region": "ap-south-1",
        "instance_type": "ml.m5.xlarge",
        "volume_size_gb": 50,
        "max_runtime_seconds": 21_600,
        "smoke_rows": None,
        "job_name": "entity-resolution-phase1-test",
        "wait": False,
        "execute": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Phase1AwsLauncherTests(unittest.TestCase):
    def test_dry_run_plan_is_non_billable_and_unique(self) -> None:
        plan = build_job_plan(_launcher_args(smoke_rows=1_000))
        self.assertEqual(plan["action"], "DRY_RUN_ONLY")
        self.assertFalse(plan["creates_billable_processing_job"])
        self.assertEqual(
            plan["output_s3_uri"],
            "s3://challenge-bucket/runs/phase1/entity-resolution-phase1-test",
        )
        self.assertEqual(
            plan["job_arguments"],
            [
                "--max-rows-per-file",
                "1000",
                "--max-ground-truth-rows",
                "1000",
            ],
        )

    def test_rejects_gpu_or_overlapping_prefixes(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU instances only"):
            build_job_plan(_launcher_args(instance_type="ml.g5.xlarge"))
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            build_job_plan(
                _launcher_args(output_s3_prefix="s3://challenge-bucket/raw/dataset/output")
            )

    def test_s3_uri_requires_a_prefix(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_s3_uri("s3://challenge-bucket")

    def test_cli_defaults_to_dry_run_without_importing_aws_sdk(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--role-arn",
                    ROLE_ARN,
                    "--input-s3-uri",
                    "s3://challenge-bucket/raw/dataset",
                    "--output-s3-prefix",
                    "s3://challenge-bucket/runs/phase1",
                    "--job-name",
                    "entity-resolution-phase1-test",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["action"], "DRY_RUN_ONLY")


class Phase1JobTests(unittest.TestCase):
    @staticmethod
    def _write_source(path: Path, entity_id: str, name: str, address: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("entity_id", "business_name", "business_address", "country"))
            writer.writerow((entity_id, name, address, "India"))

    def test_entrypoint_runs_all_three_phase1_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "dataset"
            for split in ("train", "test"):
                for source in (1, 2, 3):
                    self._write_source(
                        data_dir / split / f"{split}_source{source}.tsv",
                        f"{split}-S{source}-1",
                        "Acme Corp.",
                        "7 Lake Rd.",
                    )
            with (data_dir / "train/train_ground_truth.tsv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source1_entity_id", "matched_entity_ids"))
                writer.writerow(("train-S1-1", "train-S2-1,train-S3-1"))

            output_dir = root / "output"
            manifest = run_phase1(
                data_dir=data_dir,
                work_dir=root / "work",
                output_dir=output_dir,
                max_rows_per_file=1,
                max_ground_truth_rows=1,
            )

            self.assertTrue(manifest["bounded_run"])
            self.assertEqual(manifest["final_normalization"]["total_rows"], 6)
            self.assertTrue((output_dir / "phase1-manifest.json").is_file())
            self.assertTrue(
                (output_dir / "data/processed/final/test/test_source3.tsv").is_file()
            )
            self.assertTrue(
                (output_dir / "artifacts/normalization/full/token_equivalences.tsv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
