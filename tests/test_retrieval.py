"""Retrieval logic on the mini database with a deterministic fake embedder (no model download, no randomness).

The fake embeds each text as a normalised bag of hashed words, so similar words give similar vectors and every
property below (exactness, filter-first, fusion arithmetic, staleness) is checked without the real model.
"""
from __future__ import annotations

import random
import zlib

import numpy as np
import pytest

from recipebot.domain.models import Filters, SearchSpec
from recipebot.retrieval.bm25 import BM25Index
from recipebot.retrieval.build import build_index
from recipebot.retrieval.dense import DenseIndex, normalise
from recipebot.retrieval.hybrid import RRF_K, HybridIndex, StaleIndexError
from recipebot.retrieval.text import load_docs, tokenize
from recipebot.tools import semantic
from recipebot.tools.registry import TOOLS
from recipebot.tools.search import search_recipes
from recipebot.tools.semantic import semantic_search, set_index, similar_recipes


class FakeEmbedder:
    name, dim = "fake-bow", 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        return v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def index(mini_db, tmp_path_factory):
    folder = tmp_path_factory.mktemp("index")
    idx = build_index(mini_db, folder, FakeEmbedder(), chunk=4)
    set_index(idx)
    idx.folder = folder
    yield idx
    set_index(None)


# ------------------------------------------------------------------ BM25
def test_bm25_ranks_title_and_ingredient_matches(index):
    hits = index.bm25.search("penne", 10)
    top_ids = [index.ids[r] for r, _ in hits]
    assert set(top_ids) == {"R1", "R2", "R3", "R8"}                  # exactly the recipes mentioning penne
    assert all(s > 0 for _, s in hits)


def test_bm25_unknown_words_are_reported_not_scored(index):
    assert index.bm25.search("zzzunknown", 5) == []
    ids, unknown = index.bm25.query_terms("penne zzzunknown")
    assert len(ids) == 1 and unknown == ["zzzunknown"]


def test_bm25_mask_is_applied_before_taking_top_k(index):
    allowed = np.array([index.row_of["R3"]])
    assert [index.ids[r] for r, _ in index.bm25.search("penne", 1, allowed)] == ["R3"]


def test_bm25_ties_are_broken_by_id(index):
    ids = [index.ids[r] for r, s in index.bm25.search("penne", 10)]
    same = [i for i in ids if i in ("R2", "R8")]                       # identical documents, identical scores
    assert same == sorted(same)


# ------------------------------------------------------------------ dense: exact, filtered search
def test_dense_filtered_search_equals_brute_force(index):
    rng = random.Random(7)
    q = index.embedder.embed_query("tomato soup with onion")
    mat = index.dense.index.reconstruct_n(0, len(index.ids))
    for _ in range(8):
        allowed = np.array(sorted(rng.sample(range(len(index.ids)), 6)))
        got = [r for r, _ in index.dense.search(q, 4, allowed)]
        sims = mat[allowed] @ (q / np.linalg.norm(q))
        order = sorted(range(len(allowed)), key=lambda i: (-round(float(sims[i]), 6), index.ids[allowed[i]]))
        assert got == [int(allowed[i]) for i in order[:4]]


def test_dense_search_with_empty_allowed_set(index):
    assert index.dense.search(index.embedder.embed_query("x"), 5, np.array([], dtype=np.int64)) == []


# ------------------------------------------------------------------ fusion
def test_rrf_scores_are_the_published_formula(index):
    hits, _ = index.search("penne tomato", k=20)
    for h in hits:
        expected = sum(1.0 / (RRF_K + r) for r in (h.bm25_rank, h.dense_rank) if r)
        assert h.rrf == pytest.approx(expected)
    both = [h for h in hits if h.bm25_rank and h.dense_rank]
    assert both and hits[0].bm25_rank is not None and hits[0].dense_rank is not None


def test_hybrid_never_returns_a_recipe_outside_the_allowed_set(index):
    rng, universe = random.Random(11), list(index.ids)
    for _ in range(25):
        allowed = rng.sample(universe, rng.randint(1, len(universe)))
        hits, info = index.search("penne tomato garlic soup", k=50, allowed_ids=allowed)
        assert {h.id for h in hits} <= set(allowed) and info["eligible"] == len(allowed)


def test_modes_use_the_requested_rankers(index):
    kw, _ = index.search("penne", 10, mode="keyword")
    sm, _ = index.search("penne", 10, mode="semantic")
    assert all(h.dense_rank is None for h in kw) and all(h.bm25_rank is None for h in sm)


def test_search_is_deterministic(index):
    a = index.search("tomato soup", 10)[0]
    b = index.search("tomato soup", 10)[0]
    assert a == b


# ------------------------------------------------------------------ the point of the design: filter first
def test_similarity_alone_leaks_a_restricted_recipe_but_filter_first_does_not(index, mini_db):
    leak, _ = index.search("penne cream butter parmesan", k=3, mode="semantic")
    assert "R3" in {h.id for h in leak}                                 # the dairy recipe is the nearest neighbour
    spec = SearchSpec(query="penne cream butter parmesan", mode="semantic", exclude_classes=["dairy"], limit=20)
    res = semantic_search(spec, mini_db)
    flagged = {r[0] for r in mini_db.query("SELECT recipe_id FROM recipe_flags WHERE \"class\"='dairy'")}
    assert not ({h.id for h in res.recipes} & flagged)
    assert res.recipes and any("BEFORE ranking" in n for n in res.computation.notes)


def test_hybrid_results_carry_retrieval_evidence_and_matching_lines(index, mini_db):
    res = search_recipes(SearchSpec(query="tomato soup", mode="hybrid", limit=3), mini_db)
    top = res.recipes[0]
    assert top.retrieval["rrf"] == top.score and top.retrieval["bm25_rank"] is not None
    assert any("tomato" in m.text for m in top.matched_ingredients)
    assert "NOT guaranteed" in " ".join(res.computation.notes)


def test_requiring_an_ingredient_is_a_hard_filter_not_a_similarity(index, mini_db):
    res = search_recipes(SearchSpec(query="something comforting", mode="hybrid", include_ingredients=["tomato"],
                                    limit=20), mini_db)
    tomato = {r[0] for r in mini_db.query("SELECT recipe_id FROM ingredients_parsed WHERE name='tomato' OR parent='tomato'")}
    assert res.recipes and {h.id for h in res.recipes} <= tomato


def test_lexical_search_falls_back_to_meaning_when_no_word_matches(index, mini_db):
    res = search_recipes(SearchSpec(query="penne chowder"), mini_db)    # 'chowder' appears nowhere, 'penne' does
    assert res.total_matches > 0                                        # the any-word rule already handles this
    none = search_recipes(SearchSpec(query="zzzyyy"), mini_db)
    assert none.total_matches == 0 or any("No recipe matched the words" in n for n in none.computation.notes)


def test_pure_filters_without_a_query_ignore_the_index(index, mini_db):
    res = search_recipes(SearchSpec(mode="hybrid", include_ingredients=["tomato"]), mini_db)
    assert res.total_matches > 0 and all(h.retrieval is None for h in res.recipes)


# ------------------------------------------------------------------ similar recipes
def test_similar_recipes_excludes_the_seed_and_respects_filters(index, mini_db):
    res = similar_recipes("R1", Filters(exclude_classes=["dairy"]), k=5, db=mini_db)
    ids = [i.id for i in res.items]
    assert "R1" not in ids and "R3" not in ids                         # R3 is dairy; excluded before ranking
    top = res.items[0]
    assert top.id in ("R2", "R8") and "penne" in top.shared_ingredients and "parmesan cheese" in top.only_in_seed_recipe
    sims = [i.similarity for i in res.items]
    assert sims == sorted(sims, reverse=True)


def test_similar_unknown_id_is_an_error(index, mini_db):
    with pytest.raises(KeyError):
        similar_recipes("nope", db=mini_db)


# ------------------------------------------------------------------ persistence and staleness
def test_save_load_roundtrip_gives_identical_results(index, mini_db):
    loaded = HybridIndex.load(index.folder, FakeEmbedder(), mini_db)
    assert loaded.search("tomato garlic", 8)[0] == index.search("tomato garlic", 8)[0]


def test_index_built_for_another_database_is_rejected(index, match_db):
    with pytest.raises(StaleIndexError, match="database changed"):
        HybridIndex.load(index.folder, FakeEmbedder(), match_db)


def test_index_built_with_another_model_is_rejected(index, mini_db):
    class Other(FakeEmbedder):
        name = "other-model"

    with pytest.raises(StaleIndexError, match="embedding model"):
        HybridIndex.load(index.folder, Other(), mini_db)


def test_missing_index_has_an_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="retrieval.build"):
        HybridIndex.load(tmp_path, FakeEmbedder())


def test_build_is_resumable_and_deterministic(mini_db, tmp_path):
    first = build_index(mini_db, tmp_path / "a", FakeEmbedder(), chunk=3)
    again = build_index(mini_db, tmp_path / "b", FakeEmbedder(), chunk=5)       # different chunking, same result
    assert first.search("garlic", 10)[0] == again.search("garlic", 10)[0]
    assert np.allclose(first.dense.index.reconstruct_n(0, 11), again.dense.index.reconstruct_n(0, 11))


# ------------------------------------------------------------------ plumbing
def test_registry_exposes_similar_recipes_and_validates_mode():
    assert "similar_recipes" in TOOLS
    with pytest.raises(Exception):
        SearchSpec(query="x", mode="vibes")
    assert semantic.index_available() is True


def test_fingerprint_notices_a_changed_ingredient_parse():
    """When the parser/lexicon improves, an index built from the old parse must be detected as stale."""
    import duckdb

    from conftest import _build
    from recipebot.retrieval.text import db_fingerprint
    from recipebot.store.db import Database

    con = duckdb.connect(":memory:")
    _build(con)
    db = Database(connection=con)
    before = db_fingerprint(db)
    assert db_fingerprint(db) == before
    con.execute("UPDATE ingredients_parsed SET parent = 'changed' WHERE name = 'tomato'")
    assert db_fingerprint(db) != before
    con.close()


def test_meaning_only_hits_must_be_close_to_the_best_match():
    """A short keyword result is not padded with weak look-alikes (found with "gulab jamun" -> "Mock Guac")."""
    from types import SimpleNamespace as NS
    from recipebot.retrieval.hybrid import HybridIndex
    bm25 = NS(search=lambda q, pool, allowed: [(0, 30.0), (1, 25.0)], query_terms=lambda q: ([], []))
    dense = NS(search=lambda v, pool, allowed: [(0, 0.83), (1, 0.78), (2, 0.62), (3, 0.74)])
    ix = HybridIndex(["a", "b", "c", "d"], bm25, dense, NS(embed_query=lambda q: None), {})
    hits, info = ix.search("x", k=10)
    ids = [h.id for h in hits]
    assert ids[:2] == ["a", "b"] and "c" not in ids                      # 0.62 is far below the best (0.83)
    assert "d" in ids                                                      # 0.74 is within 0.12 of 0.83: a real semantic neighbour
    assert info["dropped_weak_semantic"] == 1
    none_lex = NS(search=lambda q, pool, allowed: [], query_terms=lambda q: ([], []))
    ix2 = HybridIndex(["a", "b", "c", "d"], none_lex, dense, NS(embed_query=lambda q: None), {})
    assert [h.id for h in ix2.search("aubergine", k=10)[0]] == ["a", "b", "d"]   # no keyword match at all: meaning still works
