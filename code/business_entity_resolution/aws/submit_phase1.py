"""Plan or submit a cost-guarded SageMaker Processing job for Phase 1.

Dry-run is the default. No AWS resources are created and no files are uploaded
unless ``--execute`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGION = "ap-south-1"
DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
ALLOWED_INSTANCE_TYPES = ("ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge")
REQUIRED_INPUT_FILES = (
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
)
ROLE_ARN_PATTERN = re.compile(
    r"^arn:(?:aws|aws-us-gov|aws-cn):iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+$"
)
REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-[0-9]+$")
JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:-*[A-Za-z0-9])*$")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_s3_uri(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.params or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(f"invalid S3 URI: {value!r}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    if not prefix:
        raise argparse.ArgumentTypeError("S3 URI must include a non-root prefix")
    if any(part in {".", ".."} for part in prefix.split("/")):
        raise argparse.ArgumentTypeError("S3 URI prefix cannot contain '.' or '..' segments")
    return parsed.netloc, prefix


def _validate_non_overlapping_s3(input_uri: str, output_uri: str) -> None:
    input_bucket, input_prefix = parse_s3_uri(input_uri)
    output_bucket, output_prefix = parse_s3_uri(output_uri)
    if input_bucket != output_bucket:
        return
    input_parts = input_prefix.split("/")
    output_parts = output_prefix.split("/")
    shared = min(len(input_parts), len(output_parts))
    if input_parts[:shared] == output_parts[:shared]:
        raise ValueError("input and output S3 prefixes must not overlap")


def _default_job_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"entity-resolution-phase1-{timestamp}"


def build_job_plan(args: argparse.Namespace) -> dict[str, object]:
    if not ROLE_ARN_PATTERN.fullmatch(args.role_arn):
        raise ValueError("--role-arn must be a complete IAM role ARN")
    if not REGION_PATTERN.fullmatch(args.region):
        raise ValueError("--region is not a valid AWS region name")
    if args.instance_type not in ALLOWED_INSTANCE_TYPES:
        choices = ", ".join(ALLOWED_INSTANCE_TYPES)
        raise ValueError(f"Phase 1 permits CPU instances only: {choices}")
    if not 30 <= args.volume_size_gb <= 200:
        raise ValueError("--volume-size-gb must be between 30 and 200")
    if not 1_800 <= args.max_runtime_seconds <= 21_600:
        raise ValueError("--max-runtime-seconds must be between 1800 and 21600")
    if len(args.job_name) > 63 or not JOB_NAME_PATTERN.fullmatch(args.job_name):
        raise ValueError("--job-name must be 1-63 letters, numbers, or interior hyphens")

    parse_s3_uri(args.input_s3_uri)
    output_bucket, output_prefix = parse_s3_uri(args.output_s3_prefix)
    run_output_uri = f"s3://{output_bucket}/{output_prefix}/{args.job_name}"
    _validate_non_overlapping_s3(args.input_s3_uri, run_output_uri)

    job_arguments: list[str] = []
    if args.smoke_rows is not None:
        job_arguments.extend(
            [
                "--max-rows-per-file",
                str(args.smoke_rows),
                "--max-ground-truth-rows",
                str(args.smoke_rows),
            ]
        )
    return {
        "action": "SUBMIT" if args.execute else "DRY_RUN_ONLY",
        "creates_billable_processing_job": bool(args.execute),
        "job_name": args.job_name,
        "region": args.region,
        "role_arn": args.role_arn,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "volume_size_gb": args.volume_size_gb,
        "max_runtime_seconds": args.max_runtime_seconds,
        "framework": "scikit-learn",
        "framework_version": "1.4-2",
        "input_s3_uri": args.input_s3_uri.rstrip("/"),
        "output_s3_uri": run_output_uri,
        "code_s3_uri": f"s3://{output_bucket}/{output_prefix}/code",
        "smoke_rows": args.smoke_rows,
        "job_arguments": job_arguments,
        "wait_for_completion": bool(args.wait),
        "cost_guards": [
            "dry-run unless --execute is supplied",
            "one CPU instance only",
            "hard maximum runtime",
            "unique per-job output prefix",
            "required S3 inputs checked before submission",
        ],
    }


def _bucket_region(s3_client: object, bucket: str) -> str:
    response = s3_client.get_bucket_location(Bucket=bucket)
    return response.get("LocationConstraint") or "us-east-1"


def _preflight_s3(plan: dict[str, object], boto_session: object) -> dict[str, int]:
    s3_client = boto_session.client("s3")
    input_bucket, input_prefix = parse_s3_uri(str(plan["input_s3_uri"]))
    output_bucket, output_prefix = parse_s3_uri(str(plan["output_s3_uri"]))
    expected_region = str(plan["region"])
    for bucket in sorted({input_bucket, output_bucket}):
        actual_region = _bucket_region(s3_client, bucket)
        if actual_region != expected_region:
            raise ValueError(
                f"S3 bucket {bucket!r} is in {actual_region}, but the job region is {expected_region}"
            )

    sizes: dict[str, int] = {}
    for relative_path in REQUIRED_INPUT_FILES:
        key = f"{input_prefix}/{relative_path}"
        response = s3_client.head_object(Bucket=input_bucket, Key=key)
        sizes[relative_path] = int(response["ContentLength"])

    existing = s3_client.list_objects_v2(
        Bucket=output_bucket,
        Prefix=f"{output_prefix}/",
        MaxKeys=1,
    )
    if int(existing.get("KeyCount", 0)):
        raise ValueError(f"refusing to reuse non-empty output prefix {plan['output_s3_uri']}")
    return sizes


def submit_job(args: argparse.Namespace, plan: dict[str, object]) -> None:
    try:
        import boto3
        from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
        from sagemaker.session import Session
        from sagemaker.sklearn.estimator import SKLearn
    except ImportError as error:
        raise RuntimeError(
            "AWS launcher dependencies are missing; install requirements-aws.txt"
        ) from error

    boto_session = boto3.Session(region_name=str(plan["region"]))
    input_sizes = _preflight_s3(plan, boto_session)
    print(
        json.dumps(
            {
                "preflight": "passed",
                "input_files": len(input_sizes),
                "input_bytes": sum(input_sizes.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    sagemaker_session = Session(boto_session=boto_session)
    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version=str(plan["framework_version"]),
        role=str(plan["role_arn"]),
        instance_count=1,
        instance_type=str(plan["instance_type"]),
        volume_size_in_gb=int(plan["volume_size_gb"]),
        max_runtime_in_seconds=int(plan["max_runtime_seconds"]),
        base_job_name="entity-resolution-phase1",
        code_location=str(plan["code_s3_uri"]),
        sagemaker_session=sagemaker_session,
        env={"PYTHONUNBUFFERED": "1"},
        tags=[
            {"Key": "Project", "Value": "amazon-ml-challenge"},
            {"Key": "Phase", "Value": "phase1"},
        ],
    )
    processor.run(
        code="phase1_job.py",
        source_dir=str(PROJECT_ROOT / "aws"),
        dependencies=[str(PROJECT_ROOT / "src")],
        inputs=[
            ProcessingInput(
                input_name="dataset",
                source=str(plan["input_s3_uri"]),
                destination="/opt/ml/processing/input/dataset",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="phase1-results",
                source="/opt/ml/processing/output/results",
                destination=str(plan["output_s3_uri"]),
            )
        ],
        arguments=list(plan["job_arguments"]),
        wait=bool(plan["wait_for_completion"]),
        logs=bool(plan["wait_for_completion"]),
        job_name=str(plan["job_name"]),
    )
    print(
        json.dumps(
            {
                "submitted": True,
                "job_name": plan["job_name"],
                "region": plan["region"],
                "output_s3_uri": plan["output_s3_uri"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution-role ARN")
    parser.add_argument(
        "--input-s3-uri",
        required=True,
        help="S3 prefix whose root contains train/ and test/",
    )
    parser.add_argument(
        "--output-s3-prefix",
        required=True,
        help="S3 parent prefix; a unique job-name child is added automatically",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--volume-size-gb", type=_positive_int, default=50)
    parser.add_argument("--max-runtime-seconds", type=_positive_int, default=21_600)
    parser.add_argument("--smoke-rows", type=_positive_int)
    parser.add_argument("--job-name", default=_default_job_name())
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Stream logs and wait; without this flag the managed job continues asynchronously",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit the billable job; omission is always a no-write dry run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_job_plan(args)
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if args.execute:
            submit_job(args, plan)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
