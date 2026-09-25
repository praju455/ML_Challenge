# Business entity resolution pipeline

This directory contains the executable pipeline. Run commands from this directory so `src` resolves as a local Python package.

## Phase 1A: safe base normalization

`src/normalize.py` streams all six source files and appends `clean_name` and `clean_address`, preserving the original columns and row order.

Normalization uses Unicode NFKC, Unicode case folding, whitespace collapse, and punctuation/symbol boundaries. It retains letters, combining marks, and numbers from every script. It deliberately performs no guessed semantic rewrites. The next Phase 1 unit mines equivalences from aligned true training pairs rather than rewriting tokens based only on spelling similarity.

Outputs are written atomically, and existing files are not replaced unless `--overwrite` is provided.

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
