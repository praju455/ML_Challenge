# Business entity resolution pipeline

This directory contains the executable pipeline. Run commands from this directory so `src` resolves as a local Python package.

## Phase 1A: safe base normalization

`src/normalize.py` streams all six source files and appends `clean_name` and `clean_address`, preserving the original columns and row order.

Normalization uses Unicode NFKC, Unicode case folding, whitespace collapse, and punctuation/symbol boundaries. It retains letters, combining marks, and numbers from every script. It deliberately performs no guessed semantic rewrites. The next Phase 1 unit mines equivalences from aligned true training pairs rather than rewriting tokens based only on spelling similarity.

Outputs are written atomically, and existing files are not replaced unless `--overwrite` is provided.

## Phase 1B: evidence-based equivalence mining

`src/mine_equivalences.py` builds a resumable, disk-backed SQLite index of the three normalized training sources. It then examines true Source-1-to-Source-2/3 match edges and records cases where the matched records differ by exactly one token. Repeated substitution or optional-token evidence is combined with full-corpus token frequencies before a mapping is accepted.

This design learns mappings from the data without a hand-written suffix list. It also prevents spelling-only guesses such as rewriting `lake` as `lakeside`.

### Run tests

```bash
python3 -m unittest discover -s tests -v
```

### Local smoke test

This limits every input file to 1,000 rows and is safe for development machines:

```bash
python3 -m src.normalize \
  --data-dir ../../student_resource/dataset \
  --output-dir data/processed/sample \
  --report-path artifacts/normalization/sample/report.json \
  --max-rows-per-file 1000 \
  --overwrite
```

### Full run on the strong machine

```bash
python3 -m src.normalize \
  --data-dir ../../student_resource/dataset \
  --output-dir data/processed/full \
  --report-path artifacts/normalization/full/report.json
```

The full command checks free disk space before scanning. It will stop early if a complete normalized copy would leave insufficient space. Use `--skip-disk-check` only after manually verifying that the destination has enough capacity.

Generated outputs are intentionally ignored by Git. The JSON report records normalization behavior, row counts, missing clean fields, and output paths.

After the normalized files exist on the strong machine, mine the equivalence table:

```bash
python3 -m src.mine_equivalences \
  --normalized-data-dir data/processed/full \
  --ground-truth-path ../../student_resource/dataset/train/train_ground_truth.tsv \
  --database-path data/interim/equivalence_records.sqlite \
  --output-path artifacts/normalization/full/token_equivalences.tsv \
  --report-path artifacts/normalization/full/equivalence_report.json
```

The SQLite index is resumable at source-file boundaries. Keep it on the strong machine; it is generated data and is ignored by Git.

### Apply the accepted mappings

Generate the final normalized files from the original inputs after mining finishes:

```bash
python3 -m src.normalize \
  --data-dir ../../student_resource/dataset \
  --output-dir data/processed/final \
  --report-path artifacts/normalization/final/report.json \
  --equivalence-path artifacts/normalization/full/token_equivalences.tsv
```

The report records the exact mapping path and the number of name and address mappings applied. Token mappings are bounded to complete tokens, mapping chains are flattened, deletion mappings are supported, and cycles are rejected.

## Phase 2: unioned blocking

`src/blocking.py` produces one tab-separated candidate-list row for every Source-1
entity. It unions four independent strategies: token-sort sorted neighbourhood,
Double Metaphone over leading name tokens, character n-gram TF-IDF neighbours, and
MinHash LSH over name-and-address shingles. Country is not a blocking key.

For bounded local samples, character neighbours use exact sklearn brute-force cosine
search. At the real target scale, the same module switches to compressed FAISS IVF
search. Install the pinned Phase 2 dependencies before running either path:

```bash
python3 -m pip install -r requirements.txt
```

After Phase 1 has produced final normalized files, run a small local smoke test:

```bash
python3 -m src.blocking \
  --source1-path data/processed/final/train/train_source1.tsv \
  --target-path data/processed/final/train/train_source2.tsv \
  --target-path data/processed/final/train/train_source3.tsv \
  --output-path output/train_candidate_lists.sample.tsv \
  --database-path data/interim/blocking.sample.sqlite \
  --report-path artifacts/blocking/sample/report.json \
  --max-rows-per-file 1000 \
  --rebuild-index
```

Measure the candidate-set recall ceiling before training any classifier. The evaluator
uses a deterministic Source-1-level validation split, so all matches for an entity
remain in the same split:

```bash
python3 -m src.eval_blocking \
  --ground-truth-path ../../student_resource/dataset/train/train_ground_truth.tsv \
  --candidate-path output/train_candidate_lists.sample.tsv \
  --target-count 2000 \
  --report-path artifacts/blocking/sample/evaluation.json
```

Do not compare candidate diagnostics from a truncated sample to a full-data score.
Use the full target count and candidate file for the final blocking recall ceiling and
reduction ratio.

## Phase 3: pair features

`src/features.py` turns the Phase-2 candidate lists into one deterministic feature
row per candidate pair. Its fixed TSV schema contains IDs, an `is_match` label
column, fuzzy name/address similarity, token and character n-gram overlap, numeric
overlap, structural ratios, and `country_equal`. For test data `is_match` is blank;
for training data it is `0` or `1` from ground truth. Country is a feature only and
never filters candidates.

The generator streams Source-1 and candidate rows in lockstep, so it rejects missing,
reordered, duplicate, or unknown candidate IDs before model training. Target records
and optional truth edges are stored in generated SQLite indices rather than loaded
into memory. These indices, feature TSVs, and reports are generated artifacts and
must remain outside Git.

Create labeled training features only after the full train blocking file meets the
recall-ceiling target:

```bash
python3 -m src.features \
  --source1-path data/processed/final/train/train_source1.tsv \
  --target-path data/processed/final/train/train_source2.tsv \
  --target-path data/processed/final/train/train_source3.tsv \
  --candidate-path output/train_candidate_pairs.tsv \
  --ground-truth-path ../../student_resource/dataset/train/train_ground_truth.tsv \
  --output-path output/train_pair_features.tsv \
  --target-database-path data/interim/features/train_targets.sqlite \
  --truth-database-path data/interim/features/train_truth.sqlite \
  --report-path artifacts/features/train/report.json \
  --rebuild-target-index \
  --rebuild-truth-index
```

Generate test features with the identical schema after the trained candidate contract
has passed validation. The blank `is_match` column is intentional:

```bash
python3 -m src.features \
  --source1-path data/processed/final/test/test_source1.tsv \
  --target-path data/processed/final/test/test_source2.tsv \
  --target-path data/processed/final/test/test_source3.tsv \
  --candidate-path output/candidate_pairs.tsv \
  --output-path output/test_pair_features.tsv \
  --target-database-path data/interim/features/test_targets.sqlite \
  --report-path artifacts/features/test/report.json \
  --rebuild-target-index
```

## Phase 5: exact validation metric

`src/metric.py` implements the competition metric used to choose every model
threshold: macro-averaged per-Source-1 F0.5. A correct empty match list scores `1.0`;
an incorrect empty/non-empty decision scores `0.0`; and any false positive on a true
singleton scores `0.0`. It refuses duplicate, missing, or unexpected Source-1 rows so
a partial artifact cannot look like a valid validation result.

Run it only on a held-out training prediction file with the same Source-1 coverage as
the supplied truth file:

```bash
python3 -m src.metric \
  --ground-truth-path ../../student_resource/dataset/train/train_ground_truth.tsv \
  --prediction-path output/validation_matching_results.tsv \
  --report-path artifacts/validation/metric.json
```

## Submission validation

Only `matching_results.tsv` is uploaded to the leaderboard. Run this check before
every upload. It rejects candidate files, incomplete Source-1 coverage, unknown
target IDs, duplicate rows, duplicate IDs within a prediction, and predictions that
were not present in the supplied candidate set.

```bash
python3 -m utils.validate_submission \
  --matching-results output/matching_results.tsv \
  --candidate-pairs output/candidate_pairs.tsv \
  --test-source1 ../../student_resource/dataset/test/test_source1.tsv \
  --test-source2 ../../student_resource/dataset/test/test_source2.tsv \
  --test-source3 ../../student_resource/dataset/test/test_source3.tsv \
  --work-dir data/interim/submission_validation \
  --report-path artifacts/submission_validation.json
```

Upload only when the command prints `"status": "PASS"`. `candidate_pairs.tsv` is
kept for the final package and must never be uploaded as `matching_results.tsv`.

## Run Phase 1 with SageMaker Processing

SageMaker Studio is the control plane only. Phase 1 runs as a finite Processing job, writes its results to S3, and releases its instance automatically. It does not create a model endpoint.

The launcher has these safeguards:

- Dry-run is the default; only `--execute` can submit a billable job.
- Phase 1 is restricted to one CPU `ml.m5` instance.
- Runtime is capped at six hours and storage is capped at 200 GB.
- All seven required S3 inputs and their bucket regions are checked before submission.
- Every run gets a unique output child prefix and refuses to reuse a non-empty prefix.

### 1. Prepare the Studio control environment

Open a small Studio JupyterLab or Code Editor space only when needed, use the smallest available instance, and keep idle shutdown enabled. In its terminal:

```bash
git clone https://github.com/praju455/ML_Challenge.git
cd ML_Challenge/code/business_entity_resolution
python3 -m venv .venv-aws
source .venv-aws/bin/activate
python -m pip install -r requirements-aws.txt
```

The isolated environment avoids replacing the SageMaker SDK bundled with Studio.

### 2. Put the supplied data in S3

Create a private S3 bucket in `ap-south-1`. The input prefix must have this exact shape:

```text
s3://YOUR_BUCKET/raw/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

From a machine with the AWS CLI configured, upload only the competition files:

```bash
aws s3 sync ../../student_resource/dataset s3://YOUR_BUCKET/raw \
  --region ap-south-1 \
  --exclude "*" \
  --include "train/*.tsv" \
  --include "test/*.tsv"
```

Do not upload the dataset to GitHub.

### 3. Inspect the job plan without spending credits

Copy the execution-role ARN from the SageMaker domain or user profile. This command performs no AWS writes:

```bash
python3 -m aws.submit_phase1 \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
  --input-s3-uri s3://YOUR_BUCKET/raw \
  --output-s3-prefix s3://YOUR_BUCKET/runs/phase1 \
  --smoke-rows 1000 \
  --job-name entity-resolution-phase1-smoke
```

Confirm that the printed plan says `"action": "DRY_RUN_ONLY"`.

### 4. Submit a bounded smoke job

Add `--execute` only after reviewing the dry-run plan:

```bash
python3 -m aws.submit_phase1 \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
  --input-s3-uri s3://YOUR_BUCKET/raw \
  --output-s3-prefix s3://YOUR_BUCKET/runs/phase1 \
  --smoke-rows 1000 \
  --job-name entity-resolution-phase1-smoke \
  --execute
```

The command returns after submission. Monitor it in SageMaker Studio under **Jobs → Processing jobs**. A Processing job releases compute automatically when it succeeds, fails, or reaches its runtime cap.

### 5. Submit the full Phase 1 run

Use a new job name and omit `--smoke-rows`:

```bash
python3 -m aws.submit_phase1 \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
  --input-s3-uri s3://YOUR_BUCKET/raw \
  --output-s3-prefix s3://YOUR_BUCKET/runs/phase1 \
  --job-name entity-resolution-phase1-full-v1 \
  --execute
```

The default is one `ml.m5.xlarge`, a 50 GB volume, and a six-hour maximum. Do not increase the instance size unless the smoke/full-job telemetry proves it is necessary. Results are written below the job-specific S3 prefix as:

```text
data/processed/final/...
artifacts/normalization/full/token_equivalences.tsv
artifacts/normalization/full/equivalence-report.json
artifacts/normalization/full/normalization-report.json
phase1-manifest.json
```

After submitting, stop the Studio JupyterLab/Code Editor app when it is no longer needed. Never create a real-time endpoint for this batch pipeline.
