"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion (RRF).

Contract with the rest of the system:
  * Retrieval PROPOSES and RANKS candidates; it never decides eligibility. Callers pass `allowed_ids` (the recipes
    that already satisfy every structured filter) and both rankers only look inside that set.
  * Deterministic: exact search, ties broken by recipe ID.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from recipebot.retrieval.bm25 import BM25Index
from recipebot.retrieval.dense import DenseIndex, Embedder
from recipebot.retrieval.text import DOC_VERSION, db_fingerprint
from recipebot.store.db import Database

Mode = Literal["hybrid", "keyword", "semantic"]
RRF_K = 60
# A recipe found ONLY by meaning (no query word in it) must still be close to the best meaning match. Without this the
# dense side pads a short keyword result with its nearest vectors however weak: "gulab jamun" has 3 real recipes and the
# list was filled out with "Mock Guac" and "Ras El Hanout" (cosine 0.62 against 0.83 for the real ones).
DENSE_ONLY_MARGIN = 0.12      # keep a meaning-only hit only if within this cosine distance of the best meaning score


class StaleIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class Retrieved:
    id: str
    rrf: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None


class HybridIndex:
    def __init__(self, ids: list[str], bm25: BM25Index, dense: DenseIndex, embedder: Embedder, meta: dict) -> None:
        self.ids, self.bm25, self.dense, self.embedder, self.meta = ids, bm25, dense, embedder, meta
        self.row_of = {rid: i for i, rid in enumerate(ids)}

    # ------------------------------------------------------------------ persistence
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.bm25.save(folder)
        self.dense.save(folder)
        (folder / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, embedder: Embedder, db: Database | None = None) -> HybridIndex:
        if not (folder / "meta.json").exists():
            raise FileNotFoundError(f"No retrieval index in {folder}. Build it: uv run python -m recipebot.retrieval.build")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta["doc_version"] != DOC_VERSION or meta["model"] != embedder.name:
            raise StaleIndexError("Index was built with a different document format or embedding model; rebuild it.")
        if db is not None and meta["db_fingerprint"] != db_fingerprint(db):
            raise StaleIndexError("The recipe database changed since the index was built; rebuild it.")
        ids = json.loads((folder / "ids.json").read_text(encoding="utf-8"))
        return cls(ids, BM25Index.load(folder, ids), DenseIndex.load(folder, ids), embedder, meta)

    # ------------------------------------------------------------------ search
    def rows_for(self, allowed_ids) -> np.ndarray | None:
        if allowed_ids is None:
            return None
        return np.fromiter((self.row_of[i] for i in allowed_ids if i in self.row_of), dtype=np.int64)

    def search(self, query: str, k: int = 20, allowed_ids=None, mode: Mode = "hybrid",
               pool: int = 300) -> tuple[list[Retrieved], dict]:
        t0 = time.time()
        allowed = self.rows_for(allowed_ids)
        pool = max(pool, k)
        bm_hits = self.bm25.search(query, pool, allowed) if mode in ("hybrid", "keyword") else []
        dn_hits = (self.dense.search(self.embedder.embed_query(query), pool, allowed)
                   if mode in ("hybrid", "semantic") else [])
        bm = {r: (rank, s) for rank, (r, s) in enumerate(bm_hits, start=1)}
        dn = {r: (rank, s) for rank, (r, s) in enumerate(dn_hits, start=1)}
        best_dense = max((s for _, s in dn.values()), default=0.0)
        floor = best_dense - DENSE_ONLY_MARGIN
        weak = {r for r, (_, s) in dn.items() if r not in bm and s < floor}
        fused: list[Retrieved] = []
        for r in (set(bm) | set(dn)) - weak:
            score = sum(1.0 / (RRF_K + x[0]) for x in (bm.get(r), dn.get(r)) if x)
            fused.append(Retrieved(self.ids[r], score, *(bm.get(r) or (None, None)), *(dn.get(r) or (None, None))))
        fused.sort(key=lambda h: (-round(h.rrf, 12), h.id))
        _, unknown = self.bm25.query_terms(query)
        info = {"mode": mode, "pool": pool, "eligible": None if allowed is None else int(allowed.size),
                "bm25_candidates": len(bm), "dense_candidates": len(dn), "unknown_words": unknown, "dropped_weak_semantic": len(weak),
                "ms": round((time.time() - t0) * 1000, 1)}
        return fused[:k], info

    def similar(self, recipe_id: str, k: int = 10, allowed_ids=None) -> list[Retrieved]:
        """Nearest recipes by meaning to an existing recipe (the recipe itself is excluded)."""
        row = self.row_of[recipe_id]
        allowed = self.rows_for(allowed_ids)
        hits = self.dense.search(self.dense.vector(row), k + 1, allowed)
        out = [Retrieved(self.ids[r], 1.0 / (RRF_K + i), dense_rank=i, dense_score=s)
               for i, (r, s) in enumerate((h for h in hits if h[0] != row), start=1)]
        return out[:k]
