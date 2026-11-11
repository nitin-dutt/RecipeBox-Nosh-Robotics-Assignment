"""Dense retrieval: local embeddings + an exact FAISS inner-product index.

Exact (flat) search is used on purpose: 124k vectors of 384 floats is small enough that approximate search would
only add recall loss and non-determinism. Filtering is done with a FAISS ID selector so the search itself only
ever looks at eligible recipes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 (quantised ONNX) through fastembed: CPU only, no API, no key, no per-call cost."""

    name = "BAAI/bge-small-en-v1.5"
    dim = 384

    def __init__(self, cache_dir: Path, batch_size: int = 64) -> None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding(self.name, cache_dir=str(cache_dir), max_length=128)
        self._batch = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts, batch_size=self._batch)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(list(self._model.query_embed([text])), dtype=np.float32)[0]


def normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


class DenseIndex:
    def __init__(self, index: faiss.Index, ids: list[str]) -> None:
        self.index, self.ids = index, ids

    @classmethod
    def build(cls, vectors: np.ndarray, ids: list[str]) -> DenseIndex:
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(normalise(vectors))
        return cls(idx, ids)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "dense.faiss"))

    @classmethod
    def load(cls, folder: Path, ids: list[str]) -> DenseIndex:
        return cls(faiss.read_index(str(folder / "dense.faiss")), ids)

    def search(self, qvec: np.ndarray, k: int, allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        q = normalise(qvec.reshape(1, -1))
        params = None
        total = self.index.ntotal
        if allowed_rows is not None:
            if allowed_rows.size == 0:
                return []
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(allowed_rows.astype(np.int64))
            total = int(allowed_rows.size)
        n = min(k, total)
        while True:
            scores, rows = (self.index.search(q, n, params=params) if params else self.index.search(q, n))
            out = [(int(r), float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0]
            # Equal scores (e.g. duplicate recipes) must be ordered by ID, not by whichever FAISS happened to return
            # at the cutoff: keep widening until the k-th score is strictly better than the last one fetched.
            if n >= total or len(out) < n or k > len(out) or round(out[k - 1][1], 6) > round(out[-1][1], 6):
                break
            n = min(n * 2, total)
        out.sort(key=lambda x: (-round(x[1], 6), self.ids[x[0]]))
        return out[:k]

    def vector(self, row: int) -> np.ndarray:
        return self.index.reconstruct(int(row))
