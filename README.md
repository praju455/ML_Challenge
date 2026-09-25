# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains a reproducible entity-resolution pipeline for matching noisy business records from Source 1 against Sources 2 and 3. The target metric is macro-averaged per-entity F0.5, which weights precision more heavily than recall.

## Constraints

- No external lookup APIs, geocoding, registry services, or internet access at inference time.
- No country-specific filters, branches, or hardcoded country allowlists.
- Train and validation splits are made by Source-1 entity to prevent leakage.
- All competition tabular input and output uses explicit tab separators.
- Large generated data, indices, models, and submissions remain outside Git.

## Verified data scale

| File | Data rows |
|---|---:|
| Train Source 1 | 2,206,821 |
| Train Source 2 | 5,034,616 |
| Train Source 3 | 5,285,603 |
| Train ground truth | 2,206,821 |
| Test Source 1 | 1,732,544 |
| Test Source 2 | 4,887,273 |
| Test Source 3 | 5,082,316 |

Train contains `India` and `US`; test contains `France`, `India`, and `US`. The zero-external-match rate in train is 5.5848%.

The global train and test comparison spaces are approximately 22.775 trillion and 17.273 trillion pairs. The character-vector blocking path will therefore use compressed FAISS approximate-nearest-neighbor search rather than global brute force.

## Project layout

```text
code/business_entity_resolution/
├── src/                 Pipeline modules
├── tests/               Unit and integration tests
├── config/              Reproducible run configuration
├── requirements.txt     Pinned dependencies
├── data/                Local generated data; ignored by Git
├── artifacts/           Models and run metadata; ignored by Git
└── output/              Candidate pairs and predictions; ignored by Git
```

The supplied competition files are expected under `student_resource/` at the repository root. That directory is intentionally ignored by Git.

## Compute split

Development and deterministic sample tests run locally. Full normalization and later compute-heavy stages run as short-lived SageMaker jobs against S3; the teammate can monitor or reproduce those jobs. This avoids producing multi-gigabyte intermediate files on the local 16 GB Mac and avoids paying for an always-on endpoint.

## Progress

- [x] Phase 0: inspect real files, sizes, schemas, countries, match multiplicity, and noise.
- [x] Phase 1: chunked normalization and automatic equivalence mining.
  - [x] Phase 1A: safe, chunked Unicode base normalization.
  - [x] Phase 1B: evidence-based automatic equivalence mining.
  - [x] Phase 1C: apply accepted mappings and validate normalized outputs.
  - [x] Phase 1D: cost-guarded SageMaker Processing launcher (full run pending on AWS).
- [ ] Phase 2: four-strategy blocking and blocking diagnostics.
- [ ] Phase 3: pairwise feature generation.
- [ ] Phase 4: calibrated classifier training and F0.5 threshold tuning.
- [ ] Phase 5: exact metric and country-generalization stress test.
- [ ] Phase 6: test inference and submission validation.

## Current commands

Run the normalization tests:

```bash
cd code/business_entity_resolution
python3 -m unittest discover -s tests -v
```

Run a bounded normalization smoke test against the supplied data:

```bash
cd code/business_entity_resolution
python3 -m src.normalize \
  --data-dir ../../student_resource/dataset \
  --output-dir data/processed/sample \
  --report-path artifacts/normalization/sample/report.json \
  --max-rows-per-file 1000 \
  --overwrite
```

See `code/business_entity_resolution/README.md` for the full-data command and normalization behavior. Commands for later phases will be added only after their implementations and tests pass.

The AWS launcher is dry-run by default and requires an explicit `--execute` before it can upload code or start a billable Processing job. See the pipeline README for S3 layout, smoke-job, full-job, and shutdown instructions.

After the full normalized training files exist on the strong machine, mine token equivalences from true matched pairs:

```bash
cd code/business_entity_resolution
python3 -m src.mine_equivalences \
  --normalized-data-dir data/processed/full \
  --ground-truth-path ../../student_resource/dataset/train/train_ground_truth.tsv \
  --database-path data/interim/equivalence_records.sqlite \
  --output-path artifacts/normalization/full/token_equivalences.tsv \
  --report-path artifacts/normalization/full/equivalence_report.json
```

Apply the accepted table to produce the final normalized files:

```bash
cd code/business_entity_resolution
python3 -m src.normalize \
  --data-dir ../../student_resource/dataset \
  --output-dir data/processed/final \
  --report-path artifacts/normalization/final/report.json \
  --equivalence-path artifacts/normalization/full/token_equivalences.tsv
```
