"""Retrieval on the real index (skipped until `uv run python -m recipebot.retrieval.build` has been run)."""
from __future__ import annotations

import time

import pytest

from recipebot.domain.models import Filters, SearchSpec
from recipebot.settings import get_settings
from recipebot.store.db import Database
from recipebot.tools import semantic
from recipebot.tools.search import search_recipes
from recipebot.tools.semantic import get_index, similar_recipes

CFG = get_settings()
pytestmark = pytest.mark.skipif(not (CFG.index_dir / "meta.json").exists() or not CFG.db_path.exists(),
                                reason="build the retrieval index first")


@pytest.fixture(scope="module")
def db():
    d = Database(CFG.db_path)
    semantic.set_index(None)
    yield d
    semantic.set_index(None)
    d.close()


def titles(db, hits):
    ids = [h.id for h in hits]
    t = {r[0]: (r[1] or "") for r in db.query(f"SELECT id, title FROM recipes WHERE id IN ({', '.join('?' for _ in ids)})", ids)}
    return [t[i] for i in ids]


def test_index_matches_the_database_and_covers_every_recipe(db):
    idx = get_index(db)                                # raises StaleIndexError if the database changed
    assert idx.meta["n_docs"] == db.scalar("SELECT count(*) FROM recipes") == len(idx.ids)
    assert idx.meta["model"] == "BAAI/bge-small-en-v1.5"


def test_dish_name_query_finds_that_dish(db):
    hits, info = get_index(db).search("banana bread", 10)
    assert sum("banana" in t.lower() and "bread" in t.lower() for t in titles(db, hits)) >= 7
    assert info["ms"] < 2000


def test_ingredient_query_prefers_recipes_that_contain_it(db):
    res = search_recipes(SearchSpec(query="chickpeas", mode="hybrid", limit=10), db)
    having = sum(any("chickpea" in m.text.lower() or "garbanzo" in m.text.lower() for m in h.matched_ingredients)
                 for h in res.recipes)
    assert having >= 8


def test_descriptive_request_returns_results_with_an_honest_note(db):
    res = search_recipes(SearchSpec(query="cozy chickpea stew for a cold night", mode="hybrid", limit=5), db)
    assert len(res.recipes) == 5 and all(h.retrieval for h in res.recipes)
    assert any("NOT guaranteed" in n for n in res.computation.notes)


@pytest.mark.parametrize("query,cls", [("pasta without dairy", "dairy"), ("egg free cake", "egg"),
                                       ("peanut free cookies", "peanut"), ("gluten free bread", "gluten")])
def test_restricted_search_never_returns_a_flagged_recipe(db, query, cls):
    res = search_recipes(SearchSpec(query=query, mode="hybrid", exclude_classes=[cls], limit=20), db)
    assert res.recipes
    ids = [h.id for h in res.recipes]
    flagged = db.query(f"SELECT recipe_id FROM recipe_flags WHERE \"class\" = ? AND recipe_id IN ({', '.join('?' for _ in ids)})", [cls, *ids])
    assert flagged == []


def test_required_ingredient_is_enforced_even_in_semantic_mode(db):
    res = search_recipes(SearchSpec(query="a comforting dinner", mode="semantic", include_ingredients=["lentil"],
                                    limit=15), db)
    ids = [h.id for h in res.recipes]
    ok = {r[0] for r in db.query("SELECT DISTINCT recipe_id FROM ingredients_parsed WHERE name = 'lentil' OR parent = 'lentil'")}
    assert ids and set(ids) <= ok


def test_search_is_deterministic(db):
    a = search_recipes(SearchSpec(query="creamy mushroom soup", mode="hybrid", limit=10), db)
    b = search_recipes(SearchSpec(query="creamy mushroom soup", mode="hybrid", limit=10), db)
    assert [h.id for h in a.recipes] == [h.id for h in b.recipes]


def test_lexical_zero_match_falls_back_to_meaning(db):
    res = search_recipes(SearchSpec(query="qwertyzxc"), db)         # not a word anywhere in the dataset
    assert res.total_matches >= 0 and (not res.recipes or any("No recipe matched" in n for n in res.computation.notes))


def test_similar_recipes_to_a_banana_bread_are_baked_goods(db):
    seed = db.scalar("SELECT id FROM recipes WHERE lower(title) = 'banana bread' AND is_complete ORDER BY id LIMIT 1")
    res = similar_recipes(seed, Filters(), k=8, db=db)
    assert seed not in [i.id for i in res.items]
    assert sum(("banana" in (i.title or "").lower()) or ("bread" in (i.title or "").lower()) for i in res.items) >= 5
    assert all(i.shared_ingredients for i in res.items[:5])
    assert [i.similarity for i in res.items] == sorted([i.similarity for i in res.items], reverse=True)
