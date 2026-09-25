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

Development and deterministic sample tests run locally. Full normalization, FAISS indexing, feature generation, training, calibration, stress testing, and test inference run on the stronger teammate machine. This avoids producing multi-gigabyte intermediate files on the local 16 GB Mac.

## Progress

- [x] Phase 0: inspect real files, sizes, schemas, countries, match multiplicity, and noise.
- [ ] Phase 1: chunked normalization and automatic equivalence mining.
- [ ] Phase 2: four-strategy blocking and blocking diagnostics.
- [ ] Phase 3: pairwise feature generation.
- [ ] Phase 4: calibrated classifier training and F0.5 threshold tuning.
- [ ] Phase 5: exact metric and country-generalization stress test.
- [ ] Phase 6: test inference and submission validation.

Exact run commands will be added as their corresponding modules are implemented and verified.
