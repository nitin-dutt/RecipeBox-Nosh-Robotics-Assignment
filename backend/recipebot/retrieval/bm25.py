"""BM25 over a sparse matrix. Fast (milliseconds over 124k documents), exact, no external service.

Weights are precomputed per (document, term): idf * tf*(k1+1) / (tf + k1*(1 - b + b*len/avglen)).
A query is scored by summing the columns of its terms. Filtering is a row mask applied BEFORE taking the top k.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from recipebot.retrieval.text import tokenize


class BM25Index:
    def __init__(self, ids: list[str], vocab: dict[str, int], idf: np.ndarray, weights: sp.csc_matrix) -> None:
        self.ids, self.vocab, self.idf, self.weights = ids, vocab, idf, weights

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, ids: list[str], docs_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75) -> BM25Index:
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        lengths = np.fromiter((len(t) for t in docs_tokens), dtype=np.float32, count=len(docs_tokens))
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(tf)
        n, m = len(docs_tokens), len(vocab)
        r, c, tf = np.array(rows), np.array(cols), np.array(tfs, dtype=np.float32)
        df = np.bincount(c, minlength=m).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = float(lengths.mean()) or 1.0
        denom = tf + k1 * (1.0 - b + b * lengths[r] / avg)
        w = idf[c] * tf * (k1 + 1.0) / denom
        return cls(ids, vocab, idf, sp.csc_matrix((w, (r, c)), shape=(n, m), dtype=np.float32))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        sp.save_npz(folder / "bm25_weights.npz", self.weights)
        np.save(folder / "bm25_idf.npy", self.idf)
        (folder / "bm25_vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, ids: list[str]) -> BM25Index:
        return cls(ids, json.loads((folder / "bm25_vocab.json").read_text(encoding="utf-8")),
                   np.load(folder / "bm25_idf.npy"), sp.load_npz(folder / "bm25_weights.npz").tocsc())

    # ------------------------------------------------------------------ query
    def query_terms(self, query: str) -> tuple[list[int], list[str]]:
        known, unknown = [], []
        for t in dict.fromkeys(tokenize(query)):
            (known if t in self.vocab else unknown).append(t)
        return [self.vocab[t] for t in known], unknown

    def search(self, query: str, k: int, allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        """-> [(row, score)] best first. Rows with score 0 (no query term) are never returned."""
        term_ids, _ = self.query_terms(query)
        if not term_ids:
            return []
        scores = np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()
        if allowed_rows is not None:
            masked = np.zeros_like(scores)
            masked[allowed_rows] = scores[allowed_rows]
            scores = masked
        cand = np.flatnonzero(scores > 0)
        if cand.size == 0:
            return []
        if cand.size > k:
            cand = cand[np.argpartition(-scores[cand], k - 1)[:k]]
        order = sorted(cand.tolist(), key=lambda r: (-scores[r], self.ids[r]))   # deterministic tie-break by ID
        return [(r, float(scores[r])) for r in order]
