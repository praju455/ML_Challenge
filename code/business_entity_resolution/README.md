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
