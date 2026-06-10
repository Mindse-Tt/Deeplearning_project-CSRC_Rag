"""Build a Chinese-native dense index using BAAI/bge-small-zh-v1.5.

Replaces the legacy English MiniLM + svd_tfidf path. Produces two artifacts:

    data/processed/chunk_embeddings_bge.npy   float32 (N, D) L2-normalised
    data/processed/chunk_id_order_bge.json    list[str] chunk_ids, same row order

Usage
-----
    python scripts/build_dense_index_bge.py \
        --model BAAI/bge-small-zh-v1.5 \
        --batch-size 32 \
        --output-npy data/processed/chunk_embeddings_bge.npy \
        --output-order data/processed/chunk_id_order_bge.json

Notes
-----
- bge models do NOT need the query-instruction prefix when encoding the corpus;
  the instruction is only applied at query time (handled by the retrieval engine).
- L2-normalised so cosine similarity = dot product.
- For 29,314 chunks:
    CPU (int8 fallback) ~3h, GPU (fp16) ~20min.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.settings import PROCESSED_DIR
from csrc_rag.utils import read_jsonl

LOGGER = logging.getLogger("build_dense_index_bge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Chinese dense retrieval index with BAAI/bge-small-zh-v1.5.",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="HuggingFace model id. Use bge-m3 if resources allow.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=PROCESSED_DIR / "event_chunks.jsonl",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=PROCESSED_DIR / "chunk_embeddings_bge.npy",
    )
    parser.add_argument(
        "--output-order",
        type=Path,
        default=PROCESSED_DIR / "chunk_id_order_bge.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' | 'cuda' | 'mps'. Default: auto.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max token length. bge-small-zh-v1.5 supports 512.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For smoke-testing. Only encode first N chunks.",
    )
    parser.add_argument(
        "--cache-folder",
        default="artifacts/models",
        help="HuggingFace cache folder (kept out of git).",
    )
    return parser.parse_args()


def load_chunks(path: Path, limit: int | None) -> tuple[list[str], list[str]]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    doc_ids = [row["chunk_id"] for row in rows]
    texts = [row["retrieval_text"] for row in rows]
    return doc_ids, texts


def encode_corpus(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
    max_seq_length: int,
    cache_folder: str | None = None,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        ) from exc

    LOGGER.info("Loading model %s (cache_folder=%s)", model_name, cache_folder)
    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_folder,
    )
    model.max_seq_length = max_seq_length

    LOGGER.info("Encoding %d chunks (batch_size=%d)", len(texts), batch_size)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def write_artifacts(
    npy_path: Path,
    order_path: Path,
    doc_ids: list[str],
    vectors: np.ndarray,
) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), vectors)
    order_path.write_text(
        json.dumps(doc_ids, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    doc_ids, texts = load_chunks(args.chunks_path, args.limit)
    if not doc_ids:
        raise SystemExit("No chunks found; aborting.")

    import time as _time
    t0 = _time.perf_counter()

    vectors = encode_corpus(
        texts=texts,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_seq_length=args.max_seq_length,
        cache_folder=args.cache_folder,
    )

    build_time_s = round(_time.perf_counter() - t0, 2)

    if vectors.shape[0] != len(doc_ids):
        raise RuntimeError(
            f"Vector count {vectors.shape[0]} != doc_id count {len(doc_ids)}",
        )

    write_artifacts(args.output_npy, args.output_order, doc_ids, vectors)

    # Resolve device actually used (sentence-transformers may auto-select).
    try:
        import torch  # type: ignore
        resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:  # pragma: no cover
        resolved_device = args.device or "cpu"

    summary = {
        "backend": "bge_prebuilt",
        "model": args.model,
        "documents": len(doc_ids),
        "dim": int(vectors.shape[1]),
        "npy_path": str(args.output_npy.relative_to(PROJECT_ROOT)),
        "order_path": str(args.output_order.relative_to(PROJECT_ROOT)),
        "normalized": True,
        "build_time_s": build_time_s,
        "num_chunks": len(doc_ids),
        "batch_size": args.batch_size,
        "device": resolved_device,
        "max_seq_length": args.max_seq_length,
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "note": "Configure configs/models.json -> dense_retrieval.bge_small_zh to point here.",
    }
    summary_path = PROCESSED_DIR / "dense_index_summary_bge.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote summary to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
