"""Stream and normalize the six competition source files safely.

This base-normalization stage is deliberately non-semantic: it does not guess that
one word is a synonym or abbreviation of another. Data-driven equivalences are mined
and validated in a separate stage before they are allowed to affect matching keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Iterator, Sequence


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
SOURCE_FILES = (
    Path("train/train_source1.tsv"),
    Path("train/train_source2.tsv"),
    Path("train/train_source3.tsv"),
    Path("test/test_source1.tsv"),
    Path("test/test_source2.tsv"),
    Path("test/test_source3.tsv"),
)
OUTPUT_COLUMNS = SOURCE_COLUMNS + ("clean_name", "clean_address")


def normalize_text(value: object) -> str:
    """Return a Unicode-aware, case-folded, punctuation-separated representation.

    NFKC unifies compatibility characters, including full-width Latin characters.
    Letters, numbers, and combining marks from every script are retained. All other
    characters become token boundaries. Numeric evidence is intentionally preserved.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    output: list[str] = []
    previous_was_separator = True
    for character in text:
        category = unicodedata.category(character)
        if category[0] in {"L", "N", "M"}:
            output.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            output.append(" ")
            previous_was_separator = True
    return "".join(output).strip()


def _validate_header(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path}: missing TSV header")
    missing = [column for column in SOURCE_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def iter_source_rows(path: Path, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    """Yield validated source rows without loading the file into memory."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _validate_header(reader.fieldnames, path)
        for row_number, row in enumerate(reader, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            if None in row:
                raise ValueError(f"{path}: row {row_number + 1} has extra tab-separated fields")
            yield {column: (row.get(column) or "") for column in SOURCE_COLUMNS}


def transform_file(
    input_path: Path,
    output_path: Path,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Normalize one TSV while preserving original columns and row order."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    rows = 0
    empty_names = 0
    empty_addresses = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in iter_source_rows(input_path, max_rows=max_rows):
                clean_name = normalize_text(row["business_name"])
                clean_address = normalize_text(row["business_address"])
                writer.writerow({**row, "clean_name": clean_name, "clean_address": clean_address})
                rows += 1
                empty_names += not bool(clean_name)
                empty_addresses += not bool(clean_address)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": rows,
        "empty_clean_name": empty_names,
        "empty_clean_address": empty_addresses,
    }


def _input_size(data_dir: Path) -> int:
    return sum((data_dir / relative_path).stat().st_size for relative_path in SOURCE_FILES)


def ensure_output_capacity(data_dir: Path, output_dir: Path, safety_multiplier: float = 2.0) -> None:
    """Fail before a full run when normalized copies would exhaust the destination."""

    output_dir.mkdir(parents=True, exist_ok=True)
    required = int(_input_size(data_dir) * safety_multiplier)
    available = shutil.disk_usage(output_dir).free
    if available < required:
        raise OSError(
            "Insufficient free disk for full normalized output: "
            f"estimated {required / (1024**3):.2f} GiB required, "
            f"{available / (1024**3):.2f} GiB available. "
            "Use --max-rows-per-file for a smoke test or run the full stage on the strong machine."
        )


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


def transform_all(
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
    max_rows_per_file: int | None = None,
    overwrite: bool = False,
    skip_disk_check: bool = False,
) -> dict[str, object]:
    """Normalize all six source files and emit an auditable JSON report."""

    for relative_path in SOURCE_FILES:
        if not (data_dir / relative_path).is_file():
            raise FileNotFoundError(data_dir / relative_path)
    if max_rows_per_file is None and not skip_disk_check:
        ensure_output_capacity(data_dir, output_dir)

    reports = [
        transform_file(
            data_dir / relative_path,
            output_dir / relative_path,
            max_rows=max_rows_per_file,
            overwrite=overwrite,
        )
        for relative_path in SOURCE_FILES
    ]
    summary = {
        "normalization": {
            "unicode_form": "NFKC",
            "case_operation": "casefold",
            "retained_unicode_categories": ["L*", "M*", "N*"],
            "semantic_rewrites": False,
        },
        "files": reports,
        "total_rows": sum(int(report["rows"]) for report in reports),
    }
    _write_json_atomic(summary, report_path)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/ and test/")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for normalized TSV files")
    parser.add_argument("--report-path", type=Path, required=True, help="Destination for the normalization report")
    parser.add_argument("--max-rows-per-file", type=_positive_int, help="Limit rows per file for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing normalized outputs")
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Bypass the full-run free-space guard after storage has been checked manually",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = transform_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        max_rows_per_file=args.max_rows_per_file,
        overwrite=args.overwrite,
        skip_disk_check=args.skip_disk_check,
    )
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
