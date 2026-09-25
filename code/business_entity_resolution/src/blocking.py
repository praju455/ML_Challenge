"""Generate Source-1 blocking candidates with four independent strategies.

The lexical strategies use a disk-backed SQLite index so the target sources do not
need to fit in RAM. Character n-gram nearest neighbours use sklearn for bounded
development samples and FAISS for full-scale data. Country is deliberately absent
from every blocking key: unseen country labels must follow the same path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


NORMALIZED_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "clean_name",
    "clean_address",
)
PRIME_64 = 18_446_744_073_709_551_557


@dataclass(frozen=True)
class Record:
    entity_id: str
    clean_name: str
    clean_address: str
    country: str = ""


@dataclass(frozen=True)
class BlockingConfig:
    sorted_window: int = 5
    phonetic_tokens: int = 2
    phonetic_limit: int = 50
    minhash_permutations: int = 16
    minhash_bands: int = 4
    minhash_limit: int = 50
    ann_top_k: int = 20
    ann_faiss_threshold: int = 250_000
    ann_fit_sample: int = 100_000
    ann_features: int = 65_536
    ann_components: int = 64
    batch_size: int = 10_000
    seed: int = 2026

    def validate(self) -> None:
        positive = {
            "sorted_window": self.sorted_window,
            "phonetic_tokens": self.phonetic_tokens,
            "phonetic_limit": self.phonetic_limit,
            "minhash_permutations": self.minhash_permutations,
            "minhash_bands": self.minhash_bands,
            "minhash_limit": self.minhash_limit,
            "ann_top_k": self.ann_top_k,
            "ann_faiss_threshold": self.ann_faiss_threshold,
            "ann_fit_sample": self.ann_fit_sample,
            "ann_features": self.ann_features,
            "ann_components": self.ann_components,
            "batch_size": self.batch_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"blocking values must be positive: {', '.join(invalid)}")
        if self.minhash_permutations % self.minhash_bands:
            raise ValueError("minhash_permutations must be divisible by minhash_bands")


def iter_normalized_records(path: Path, max_rows: int | None = None) -> Iterator[Record]:
    """Stream normalized records with strict TSV/header validation."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in NORMALIZED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            if None in row:
                raise ValueError(f"{path}: row {row_number + 1} has extra tab-separated fields")
            entity_id = (row.get("entity_id") or "").strip()
            if not entity_id:
                raise ValueError(f"{path}: row {row_number + 1} has an empty entity_id")
            yield Record(
                entity_id=entity_id,
                clean_name=row.get("clean_name") or "",
                clean_address=row.get("clean_address") or "",
                country=row.get("country") or "",
            )


def token_sort_key(text: str) -> str:
    return " ".join(sorted(text.split()))


def combined_text(record: Record) -> str:
    if record.clean_name and record.clean_address:
        return f"{record.clean_name} {record.clean_address}"
    return record.clean_name or record.clean_address


def default_phonetic_encoder(text: str) -> tuple[str, str]:
    """Return Double Metaphone keys; the dependency is loaded only when needed."""

    try:
        from metaphone import doublemetaphone
    except ImportError as error:  # pragma: no cover - exercised in installed runtime
        raise RuntimeError(
            "Double Metaphone support is missing; install the pinned requirements.txt"
        ) from error
    primary, secondary = doublemetaphone(text)
    return primary or "", secondary or ""


def phonetic_keys(
    clean_name: str,
    leading_tokens: int,
    encoder: Callable[[str], tuple[str, str]] = default_phonetic_encoder,
) -> tuple[str, ...]:
    prefix = " ".join(clean_name.split()[:leading_tokens])
    if not prefix:
        return ()
    return tuple(dict.fromkeys(key for key in encoder(prefix) if key))


def character_shingles(text: str, width: int = 3, maximum: int = 256) -> tuple[str, ...]:
    compact = " ".join(text.split())
    if not compact:
        return ()
    if len(compact) <= width:
        return (compact,)
    shingles = sorted({compact[index : index + width] for index in range(len(compact) - width + 1)})
    if len(shingles) <= maximum:
        return tuple(shingles)
    step = len(shingles) / maximum
    return tuple(shingles[int(index * step)] for index in range(maximum))


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _minhash_coefficients(permutations: int, seed: int) -> tuple[tuple[int, int], ...]:
    rng = random.Random(seed)
    return tuple((rng.randrange(1, PRIME_64), rng.randrange(0, PRIME_64)) for _ in range(permutations))


def minhash_band_keys(text: str, config: BlockingConfig) -> tuple[str, ...]:
    shingles = character_shingles(text)
    if not shingles:
        return ()
    coefficients = _minhash_coefficients(config.minhash_permutations, config.seed)
    hashed = tuple(_stable_u64(shingle) for shingle in shingles)
    signature = [
        min((coefficient * value + offset) % PRIME_64 for value in hashed)
        for coefficient, offset in coefficients
    ]
    rows_per_band = config.minhash_permutations // config.minhash_bands
    keys: list[str] = []
    for band in range(config.minhash_bands):
        start = band * rows_per_band
        values = signature[start : start + rows_per_band]
        payload = band.to_bytes(2, "big") + b"".join(value.to_bytes(8, "big") for value in values)
        keys.append(hashlib.blake2b(payload, digest_size=12).hexdigest())
    return tuple(keys)


class SQLiteLexicalIndex:
    """Disk-backed indices for sorted-neighbourhood, phonetic, and MinHash LSH."""

    def __init__(
        self,
        path: Path,
        config: BlockingConfig,
        phonetic_encoder: Callable[[str], tuple[str, str]] | None = None,
        rebuild: bool = False,
    ) -> None:
        config.validate()
        self.path = path
        self.config = config
        self.phonetic_encoder = phonetic_encoder or default_phonetic_encoder
        path.parent.mkdir(parents=True, exist_ok=True)
        if rebuild:
            path.unlink(missing_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                entity_id TEXT PRIMARY KEY,
                sort_key TEXT NOT NULL,
                phonetic_primary TEXT NOT NULL,
                phonetic_secondary TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS targets_sort_key
                ON targets(sort_key, entity_id);
            CREATE INDEX IF NOT EXISTS targets_phonetic_primary
                ON targets(phonetic_primary, entity_id);
            CREATE INDEX IF NOT EXISTS targets_phonetic_secondary
                ON targets(phonetic_secondary, entity_id);
            CREATE TABLE IF NOT EXISTS minhash_buckets (
                band_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (band_key, entity_id),
                FOREIGN KEY (entity_id) REFERENCES targets(entity_id)
            ) WITHOUT ROWID;
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteLexicalIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def target_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0])

    def add_many(self, records: Iterable[Record]) -> int:
        target_rows: list[tuple[str, str, str, str]] = []
        bucket_rows: list[tuple[str, str]] = []
        inserted = 0

        def flush() -> None:
            nonlocal inserted
            if not target_rows:
                return
            before = self.target_count
            with self.connection:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?)", target_rows
                )
                self.connection.executemany(
                    "INSERT OR IGNORE INTO minhash_buckets VALUES (?, ?)", bucket_rows
                )
            inserted += self.target_count - before
            target_rows.clear()
            bucket_rows.clear()

        for record in records:
            phonetics = phonetic_keys(
                record.clean_name,
                self.config.phonetic_tokens,
                self.phonetic_encoder,
            )
            primary = phonetics[0] if phonetics else ""
            secondary = phonetics[1] if len(phonetics) > 1 else ""
            target_rows.append(
                (record.entity_id, token_sort_key(record.clean_name), primary, secondary)
            )
            bucket_rows.extend(
                (band_key, record.entity_id)
                for band_key in minhash_band_keys(combined_text(record), self.config)
            )
            if len(target_rows) >= self.config.batch_size:
                flush()
        flush()
        return inserted

    def sorted_neighbors(self, clean_name: str) -> set[str]:
        key = token_sort_key(clean_name)
        if not key:
            return set()
        limit = self.config.sorted_window
        lower = self.connection.execute(
            """
            SELECT entity_id FROM targets
            WHERE sort_key < ?
            ORDER BY sort_key DESC, entity_id DESC LIMIT ?
            """,
            (key, limit),
        ).fetchall()
        upper = self.connection.execute(
            """
            SELECT entity_id FROM targets
            WHERE sort_key >= ?
            ORDER BY sort_key, entity_id LIMIT ?
            """,
            (key, limit),
        ).fetchall()
        return {str(row[0]) for row in (*lower, *upper)}

    def phonetic_neighbors(self, clean_name: str) -> set[str]:
        keys = phonetic_keys(clean_name, self.config.phonetic_tokens, self.phonetic_encoder)
        if not keys:
            return set()
        placeholders = ",".join("?" for _ in keys)
        query = f"""
            SELECT entity_id FROM targets
            WHERE phonetic_primary IN ({placeholders})
               OR phonetic_secondary IN ({placeholders})
            ORDER BY entity_id LIMIT ?
        """
        rows = self.connection.execute(
            query,
            (*keys, *keys, self.config.phonetic_limit),
        ).fetchall()
        return {str(row[0]) for row in rows}

    def minhash_neighbors(self, record: Record) -> set[str]:
        band_keys = minhash_band_keys(combined_text(record), self.config)
        if not band_keys:
            return set()
        placeholders = ",".join("?" for _ in band_keys)
        rows = self.connection.execute(
            f"""
            SELECT entity_id, COUNT(*) AS band_hits
            FROM minhash_buckets
            WHERE band_key IN ({placeholders})
            GROUP BY entity_id
            ORDER BY band_hits DESC, entity_id
            LIMIT ?
            """,
            (*band_keys, self.config.minhash_limit),
        ).fetchall()
        return {str(row[0]) for row in rows}


class CharacterNgramANN:
    """Character TF-IDF ANN with a bounded sklearn or full-scale FAISS backend."""

    def __init__(self, config: BlockingConfig) -> None:
        self.config = config
        self.backend = "unbuilt"
        self.target_ids: list[str] = []
        self.vectorizer: object | None = None
        self.reducer: object | None = None
        self.index: object | None = None

    @staticmethod
    def _imports() -> tuple[object, object, object, object]:
        try:
            import numpy as np
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neighbors import NearestNeighbors
        except ImportError as error:  # pragma: no cover - depends on installed environment
            raise RuntimeError(
                "Character n-gram ANN dependencies are missing; install requirements.txt"
            ) from error
        return np, TfidfVectorizer, TruncatedSVD, NearestNeighbors

    @staticmethod
    def _ann_text(text: str) -> str:
        """Keep records with both fields blank indexable without guessing a match."""

        return text if text else "blankrecord"

    def _make_vectorizer(self, vectorizer_type: object, np: object) -> object:
        return vectorizer_type(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=self.config.ann_features,
            sublinear_tf=True,
            dtype=np.float32,
        )

    def build(self, target_paths: Sequence[Path], max_rows_per_file: int | None = None) -> dict[str, object]:
        np, vectorizer_type, reducer_type, neighbors_type = self._imports()
        rng = random.Random(self.config.seed)
        sample: list[str] = []
        small_texts: list[str] = []
        small_ids: list[str] = []
        total = 0
        for path in target_paths:
            for record in iter_normalized_records(path, max_rows=max_rows_per_file):
                text = self._ann_text(combined_text(record))
                total += 1
                if len(sample) < self.config.ann_fit_sample:
                    sample.append(text)
                else:
                    replacement = rng.randrange(total)
                    if replacement < len(sample):
                        sample[replacement] = text
                if total <= self.config.ann_faiss_threshold:
                    small_ids.append(record.entity_id)
                    small_texts.append(text)
                elif small_texts:
                    small_ids.clear()
                    small_texts.clear()
        if not total:
            raise ValueError("cannot build ANN index without target records")

        vectorizer = self._make_vectorizer(vectorizer_type, np)
        sample_matrix = vectorizer.fit_transform(sample)
        self.vectorizer = vectorizer

        if total <= self.config.ann_faiss_threshold:
            matrix = vectorizer.transform(small_texts)
            index = neighbors_type(metric="cosine", algorithm="brute", n_jobs=-1)
            index.fit(matrix)
            self.backend = "sklearn-brute"
            self.index = index
            self.target_ids = small_ids
            return {"backend": self.backend, "targets": total, "dimensions": matrix.shape[1]}

        max_components = min(sample_matrix.shape[0] - 1, sample_matrix.shape[1] - 1)
        if max_components < 1:
            raise ValueError(
                "FAISS blocking needs at least two distinct sampled records and character features"
            )
        components = min(self.config.ann_components, max_components)
        reducer = reducer_type(n_components=components, random_state=self.config.seed)
        sample_dense = reducer.fit_transform(sample_matrix).astype("float32", copy=False)
        norms = np.linalg.norm(sample_dense, axis=1, keepdims=True)
        sample_dense /= np.maximum(norms, 1e-12)
        try:
            import faiss
        except ImportError as error:  # pragma: no cover - depends on installed environment
            raise RuntimeError(
                "The real target count requires FAISS; install the pinned faiss-cpu dependency"
            ) from error
        nlist = max(1, min(4096, int(math.sqrt(total)), len(sample_dense)))
        quantizer = faiss.IndexFlatIP(components)
        index = faiss.IndexIVFFlat(quantizer, components, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(sample_dense)
        index.nprobe = min(32, nlist)

        self.reducer = reducer
        self.index = index
        self.target_ids = []
        batch_texts: list[str] = []
        for path in target_paths:
            for record in iter_normalized_records(path, max_rows=max_rows_per_file):
                self.target_ids.append(record.entity_id)
                batch_texts.append(self._ann_text(combined_text(record)))
                if len(batch_texts) >= self.config.batch_size:
                    self._faiss_add(batch_texts, np)
                    batch_texts.clear()
        if batch_texts:
            self._faiss_add(batch_texts, np)
        self.backend = "faiss-ivf-flat"
        return {"backend": self.backend, "targets": total, "dimensions": components, "nlist": nlist}

    def _faiss_add(self, texts: Sequence[str], np: object) -> None:
        matrix = self.vectorizer.transform(texts)
        dense = self.reducer.transform(matrix).astype("float32", copy=False)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense /= np.maximum(norms, 1e-12)
        self.index.add(dense)

    def neighbors(self, record: Record) -> set[str]:
        if self.index is None or self.vectorizer is None:
            raise RuntimeError("ANN index has not been built")
        text = self._ann_text(combined_text(record))
        query = self.vectorizer.transform([text])
        top_k = min(self.config.ann_top_k, len(self.target_ids))
        if self.backend == "sklearn-brute":
            _distances, indices = self.index.kneighbors(query, n_neighbors=top_k)
        else:
            np = self._imports()[0]
            dense = self.reducer.transform(query).astype("float32", copy=False)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            dense /= np.maximum(norms, 1e-12)
            _scores, indices = self.index.search(dense, top_k)
        return {
            self.target_ids[int(index)]
            for index in indices[0]
            if 0 <= int(index) < len(self.target_ids)
        }


class CandidateBlocker:
    def __init__(
        self,
        lexical_index: SQLiteLexicalIndex,
        ann_neighbors: Callable[[Record], set[str]],
    ) -> None:
        self.lexical_index = lexical_index
        self.ann_neighbors = ann_neighbors

    def candidates_by_strategy(self, record: Record) -> dict[str, set[str]]:
        return {
            "sorted_neighborhood": self.lexical_index.sorted_neighbors(record.clean_name),
            "double_metaphone": self.lexical_index.phonetic_neighbors(record.clean_name),
            "char_ngram_ann": self.ann_neighbors(record),
            "minhash_lsh": self.lexical_index.minhash_neighbors(record),
        }

    def candidates(self, record: Record) -> set[str]:
        candidates: set[str] = set()
        for values in self.candidates_by_strategy(record).values():
            candidates.update(values)
        candidates.discard(record.entity_id)
        return candidates


def build_lexical_index(
    database_path: Path,
    target_paths: Sequence[Path],
    config: BlockingConfig,
    max_rows_per_file: int | None = None,
    rebuild: bool = False,
) -> tuple[SQLiteLexicalIndex, int]:
    index = SQLiteLexicalIndex(database_path, config, rebuild=rebuild)
    if index.target_count and not rebuild:
        return index, index.target_count
    for path in target_paths:
        index.add_many(iter_normalized_records(path, max_rows=max_rows_per_file))
    return index, index.target_count


def generate_candidate_file(
    source1_path: Path,
    target_paths: Sequence[Path],
    output_path: Path,
    database_path: Path,
    report_path: Path,
    config: BlockingConfig,
    max_rows_per_file: int | None = None,
    rebuild_index: bool = False,
) -> dict[str, object]:
    """Build all four strategies and atomically write the union per Source-1 row."""

    config.validate()
    lexical, target_count = build_lexical_index(
        database_path,
        target_paths,
        config,
        max_rows_per_file=max_rows_per_file,
        rebuild=rebuild_index,
    )
    try:
        ann = CharacterNgramANN(config)
        ann_report = ann.build(target_paths, max_rows_per_file=max_rows_per_file)
        blocker = CandidateBlocker(lexical, ann.neighbors)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".partial")
        counts: list[int] = []
        strategy_nonempty = {
            "sorted_neighborhood": 0,
            "double_metaphone": 0,
            "char_ngram_ann": 0,
            "minhash_lsh": 0,
        }
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source1_entity_id", "candidate_entity_ids"))
                for record in iter_normalized_records(source1_path, max_rows=max_rows_per_file):
                    by_strategy = blocker.candidates_by_strategy(record)
                    candidates: set[str] = set()
                    for strategy, values in by_strategy.items():
                        strategy_nonempty[strategy] += bool(values)
                        candidates.update(values)
                    candidates.discard(record.entity_id)
                    ordered = sorted(candidates)
                    writer.writerow((record.entity_id, ",".join(ordered)))
                    counts.append(len(ordered))
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        lexical.close()

    source1_count = len(counts)
    sorted_counts = sorted(counts)
    report = {
        "source1_records": source1_count,
        "target_records": target_count,
        "candidate_pairs": sum(counts),
        "candidate_count": {
            "minimum": min(counts, default=0),
            "median": statistics.median(sorted_counts) if source1_count else 0,
            "maximum": max(counts, default=0),
            "mean": (sum(counts) / source1_count) if source1_count else 0.0,
        },
        "strategy_nonempty_source1": strategy_nonempty,
        "ann": ann_report,
        "config": config.__dict__,
        "output": str(output_path),
        "database": str(database_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".partial")
    try:
        with temporary_report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_report, report_path)
    finally:
        temporary_report.unlink(missing_ok=True)
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1-path", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True, action="append")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--max-rows-per-file", type=_positive_int)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--sorted-window", type=_positive_int, default=5)
    parser.add_argument("--ann-top-k", type=_positive_int, default=20)
    parser.add_argument("--ann-faiss-threshold", type=_positive_int, default=250_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BlockingConfig(
        sorted_window=args.sorted_window,
        ann_top_k=args.ann_top_k,
        ann_faiss_threshold=args.ann_faiss_threshold,
        seed=args.seed,
    )
    try:
        report = generate_candidate_file(
            source1_path=args.source1_path,
            target_paths=args.target_path,
            output_path=args.output_path,
            database_path=args.database_path,
            report_path=args.report_path,
            config=config,
            max_rows_per_file=args.max_rows_per_file,
            rebuild_index=args.rebuild_index,
        )
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
