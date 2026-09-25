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
s3://YOUR_BUCKET/raw/dataset/
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
aws s3 sync ../../student_resource/dataset s3://YOUR_BUCKET/raw/dataset \
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
  --input-s3-uri s3://YOUR_BUCKET/raw/dataset \
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
  --input-s3-uri s3://YOUR_BUCKET/raw/dataset \
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
  --input-s3-uri s3://YOUR_BUCKET/raw/dataset \
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
