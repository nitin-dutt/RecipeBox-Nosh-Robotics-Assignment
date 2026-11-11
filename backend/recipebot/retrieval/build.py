"""Build the retrieval index:  uv run python -m recipebot.retrieval.build

Resumable: embeddings are computed in length-sorted chunks and checkpointed, so an interruption loses at most one
chunk. BM25 is built in seconds; the embedding step is the slow part (about 80 documents/second on CPU).
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recipebot.retrieval.bm25 import BM25Index
from recipebot.retrieval.dense import DenseIndex, Embedder, FastEmbedder
from recipebot.retrieval.hybrid import HybridIndex
from recipebot.retrieval.text import DOC_VERSION, db_fingerprint, load_docs
from recipebot.settings import get_settings
from recipebot.store.db import Database

log = logging.getLogger("recipebot.retrieval.build")


def build_index(db: Database, out_dir: Path, embedder: Embedder, chunk: int = 8192) -> HybridIndex:
    t0 = time.time()
    docs = load_docs(db)
    ids = [d.id for d in docs]
    log.info("%d documents loaded", len(docs))
    bm25 = BM25Index.build(ids, [d.bm25_tokens for d in docs])
    log.info("BM25 built: %d terms (%.1fs)", len(bm25.vocab), time.time() - t0)

    texts = [d.dense_text for d in docs]
    order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))      # similar lengths batch together
    tmp = out_dir / "_chunks"
    tmp.mkdir(parents=True, exist_ok=True)
    vectors = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    done_before = time.time()
    for ci, start in enumerate(range(0, len(order), chunk)):
        part = order[start:start + chunk]
        path = tmp / f"chunk_{ci:04d}.npy"
        stamp = tmp / f"chunk_{ci:04d}.md5"
        digest = hashlib.md5("\n".join(texts[i] for i in part).encode("utf-8")).hexdigest()
        if path.exists() and stamp.exists() and stamp.read_text() == digest:   # never reuse a chunk of other text
            vecs = np.load(path)
        else:
            t1 = time.time()
            vecs = embedder.embed_documents([texts[i] for i in part])
            np.save(path, vecs)
            stamp.write_text(digest)
            rate = len(part) / (time.time() - t1)
            left = (len(order) - start - len(part)) / rate / 60
            log.info("chunk %d: %d docs, %.0f docs/s, ~%.1f min left", ci, len(part), rate, left)
        vectors[part] = vecs
    dense = DenseIndex.build(vectors, ids)
    meta = {"model": embedder.name, "dim": embedder.dim, "n_docs": len(ids), "doc_version": DOC_VERSION,
            "db_fingerprint": db_fingerprint(db), "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bm25_terms": len(bm25.vocab), "rrf_k": 60}
    index = HybridIndex(ids, bm25, dense, embedder, meta)
    index.save(out_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    log.info("index saved to %s in %.1f min (embedding phase %.1f min)", out_dir, (time.time() - t0) / 60,
             (time.time() - done_before) / 60)
    return index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = get_settings()
    db = Database(cfg.db_path)
    build_index(db, cfg.index_dir, FastEmbedder(cfg.models_dir))


if __name__ == "__main__":
    main()
