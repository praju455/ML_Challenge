"""Score test pair features and write a complete, validated matching-results TSV."""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from typing import Sequence
from src.features import FEATURE_COLUMNS, PAIR_FEATURE_COLUMNS
from utils.validate_submission import validate_submission

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")

def _test_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if "entity_id" not in (r.fieldnames or ()): raise ValueError(f"{path}: missing entity_id")
        ids = [row["entity_id"] for row in r]
    if len(ids) != len(set(ids)): raise ValueError(f"{path}: duplicate entity_id")
    return ids

def predict_file(feature_path: Path, model_path: Path, test_source1_path: Path, output_path: Path, model_loader=None) -> dict[str, object]:
    if model_loader is None:
        try:
            import joblib
        except ImportError as e: raise RuntimeError("install requirements.txt before prediction") from e
        model_loader = joblib.load
    artifact = model_loader(model_path)
    if tuple(artifact.get("feature_columns", ())) != PAIR_FEATURE_COLUMNS: raise ValueError("model feature schema does not match")
    threshold, model = float(artifact["threshold"]), artifact["calibrator"]
    ids, predicted = _test_ids(test_source1_path), {}
    with feature_path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if tuple(r.fieldnames or ()) != FEATURE_COLUMNS: raise ValueError(f"{feature_path}: unexpected feature header")
        rows = list(r)
    if any(row["is_match"] for row in rows): raise ValueError("test feature file must have blank is_match values")
    if rows:
        x = [[float(row[c]) for c in PAIR_FEATURE_COLUMNS] for row in rows]
        probabilities = [float(row[1]) for row in model.predict_proba(x)]
        for row, probability in zip(rows, probabilities):
            if probability >= threshold: predicted.setdefault(row["source1_entity_id"], set()).add(row["candidate_entity_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True); temporary = output_path.with_suffix(output_path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(MATCH_HEADER)
            for source_id in ids: w.writerow((source_id, ",".join(sorted(predicted.get(source_id, set())))))
        os.replace(temporary, output_path)
    finally: temporary.unlink(missing_ok=True)
    return {"source1_rows": len(ids), "predicted_pairs": sum(map(len, predicted.values())), "threshold": threshold, "output": str(output_path)}

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-path", "model-path", "test-source1", "test-source2", "test-source3", "candidate-pairs", "output-path", "work-dir", "report-path"): p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        report = predict_file(a.feature_path, a.model_path, a.test_source1, a.output_path)
        report["validation"] = validate_submission(a.output_path, a.test_source1, a.test_source2, a.test_source3, a.candidate_pairs, a.work_dir)
        a.report_path.parent.mkdir(parents=True, exist_ok=True); a.report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    except (OSError, RuntimeError, ValueError) as e: print(f"error: {e}", file=sys.stderr); return 2
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["validation"]["status"] == "PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
