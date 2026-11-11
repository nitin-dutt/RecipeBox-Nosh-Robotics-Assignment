"""The brief's example questions against the real database, cross-checked by an independent Python computation."""
from __future__ import annotations

import duckdb
import pytest

from recipebot.domain.models import Filters, SearchSpec
from recipebot.settings import get_settings
from recipebot.store.db import Database
from recipebot.tools import stats
from recipebot.tools.search import search_recipes

DB = get_settings().db_path
pytestmark = pytest.mark.skipif(not DB.exists(), reason="run `uv run python -m recipebot.ingestion` first")


@pytest.fixture(scope="module")
def db():
    d = Database(DB)
    yield d
    d.close()


def test_garlic_and_onion_count_matches_independent_python_set_logic(db):
    sql_count = stats.count_recipes(Filters(include_ingredients=["garlic", "onion"]), db)
    # independent path: pull (recipe, name, parent) rows and intersect sets in Python (no EXISTS pipeline)
    complete = {r[0] for r in db.query("SELECT id FROM recipes WHERE is_complete")}
    rows = db.query("SELECT recipe_id, name, parent FROM ingredients_parsed WHERE status='ok' "
                    "AND (name IN ('garlic','onion') OR parent IN ('garlic','onion'))")
    garlic = {r for r, n, p in rows if n == "garlic" or p == "garlic"} & complete
    onion = {r for r, n, p in rows if n == "onion" or p == "onion"} & complete
    assert sql_count.count == len(garlic & onion)
    assert sql_count.denominator == len(complete)
    assert sql_count.percentage == round(100 * len(garlic & onion) / len(complete), 2)


def test_penne_search_then_no_dairy_refines_the_same_result_set(db):
    base = search_recipes(SearchSpec(query="penne", limit=50), db)
    nodairy = search_recipes(SearchSpec(query="penne", exclude_classes=["dairy"], limit=50), db)
    assert 0 < nodairy.total_matches < base.total_matches
    # none of the returned recipes carries a definite or possible dairy flag, and each still matches "penne"
    flagged = {r[0] for r in db.query("SELECT recipe_id FROM recipe_flags WHERE \"class\"='dairy'")}
    assert not ({h.id for h in nodairy.recipes} & flagged)
    assert all(h.matched_ingredients or "penne" in h.title.lower() for h in nodairy.recipes)
    # the refinement is exactly the unrestricted set minus the flagged recipes
    removed = db.scalar("""SELECT count(*) FROM recipes r WHERE r.is_complete
        AND (lower(r.title) LIKE '%penne%' OR EXISTS (SELECT 1 FROM ingredients_parsed p WHERE p.recipe_id=r.id
             AND p.status='ok' AND (p.name='penne' OR p.parent='penne')))
        AND EXISTS (SELECT 1 FROM recipe_flags f WHERE f.recipe_id=r.id AND f."class"='dairy')""")
    assert base.total_matches - nodairy.total_matches == removed
    assert nodairy.disclaimer


def test_tomato_fewest_steps_lists_five_with_ids_and_discloses_ties(db):
    r = stats.rank_by_steps(Filters(include_ingredients=["tomato"]), k=5, db=db)
    assert len(r.ranking) == 5 and all(x.id for x in r.ranking)
    steps = [x.steps for x in r.ranking]
    assert steps == sorted(steps)
    assert r.n_tied_at_cutoff >= 5 and any("share the cutoff" in n for n in r.computation.notes)


def test_funnel_is_monotonic_and_ends_at_the_count(db):
    c = stats.count_recipes(Filters(include_ingredients=["garlic"], exclude_ingredients=["egg"],
                                    exclude_classes=["dairy"], max_steps=10), db)
    rem = [f.remaining for f in c.computation.funnel]
    assert rem == sorted(rem, reverse=True) and rem[-1] == c.count


def test_stats_use_the_full_filtered_set_not_the_search_page(db):
    total = search_recipes(SearchSpec(query="chicken", limit=5), db).total_matches
    counted = stats.count_recipes(Filters(include_ingredients=["chicken"]), db).count
    assert total > 5 and counted > 5       # a 5-row page can never be the denominator or the count
