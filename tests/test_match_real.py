"""Task 2 on the real database: the brief's example, its follow-up, a no-match case, and independent re-checks."""
from __future__ import annotations

import re
import time

import pytest

from recipebot.domain.models import AvailableIngredient as AI
from recipebot.domain.models import Filters, MatchRequest
from recipebot.settings import get_settings
from recipebot.store.db import Database
from recipebot.tools.match import match_recipes, parse_available_text

DB = get_settings().db_path
pytestmark = pytest.mark.skipif(not DB.exists(), reason="run `uv run python -m recipebot.ingestion` first")


@pytest.fixture(scope="module")
def db():
    d = Database(DB)
    yield d
    d.close()


def example(**kw):
    return MatchRequest(available=parse_available_text("eggs, onions, tomatoes, and rice"),
                        constraints=Filters(exclude_classes=["peanut"], max_steps=6), k=5, **kw)


def test_example_request_returns_five_eligible_ranked_recipes_quickly(db):
    t = time.time()
    r = match_recipes(example(), db)
    assert time.time() - t < 10
    assert r.status == "matches" and len(r.matches) == 5
    scores = [m.score for m in r.matches]
    assert scores == sorted(scores, reverse=True)
    for m in r.matches:
        assert m.id and m.title and m.why and m.used and m.steps <= 6
        assert len(m.used) + len(m.unused_available) == 4


def test_hard_constraints_hold_for_every_returned_recipe(db):
    r = match_recipes(MatchRequest(available=parse_available_text("eggs, onions, tomatoes, and rice"),
                                   constraints=Filters(exclude_classes=["peanut"], max_steps=6), k=20), db)
    ids = [m.id for m in r.matches]
    ph = ", ".join("?" for _ in ids)
    flagged = db.query(f"SELECT recipe_id FROM recipe_flags WHERE \"class\"='peanut' AND recipe_id IN ({ph})", ids)
    assert flagged == []
    over = db.query(f"SELECT id FROM recipes WHERE n_steps_sentences > 6 AND id IN ({ph})", ids)
    assert over == []
    lines = db.query(f"SELECT text FROM recipe_ingredients WHERE recipe_id IN ({ph})", ids)
    assert not any(re.search(r"peanut", t.lower()) for (t,) in lines)


def test_independent_recount_of_used_ingredients(db):
    """Recompute 'which of my ingredients does this recipe use' in plain Python from the parsed rows."""
    r = match_recipes(MatchRequest(available=parse_available_text("eggs, onions, tomatoes, and rice"),
                                   constraints=Filters(exclude_classes=["peanut"], max_steps=6), k=10), db)
    for m in r.matches:
        rows = db.query("SELECT name, parent, optional FROM ingredients_parsed WHERE recipe_id = ? AND status='ok'", [m.id])
        names = {n for n, p, o in rows if not o} | {p for n, p, o in rows if not o}
        want = {t for t in ("egg", "onion", "tomato", "rice") if t in names}
        assert {u.available for u in m.used} == want, (m.title, want, [u.available for u in m.used])


def test_follow_up_exclude_eggs_removes_every_recipe_that_mentions_egg(db):
    base = example()
    follow = base.model_copy(update={"constraints": base.constraints.model_copy(update={"exclude_ingredients": ["eggs"]}),
                                     "k": 20})
    r = match_recipes(follow, db)
    assert r.status == "matches"
    assert any("egg" in s and "exclude" in s for s in r.ignored_available)
    ids = [m.id for m in r.matches]
    ph = ", ".join("?" for _ in ids)
    text = [t.lower() for (t,) in db.query(f"SELECT text FROM recipe_ingredients WHERE recipe_id IN ({ph})", ids)]
    assert not any(re.search(r"\beggs?\b", t) for t in text)
    assert all("egg" not in [u.available for u in m.used] for m in r.matches)
    first = match_recipes(example(), db)
    assert [m.id for m in first.matches] != [m.id for m in r.matches[:5]]       # the results actually changed


def test_no_match_is_stated_and_alternatives_are_labelled(db):
    r = match_recipes(MatchRequest(available=[AI(name="rice"), AI(name="tomato")], constraints=Filters(max_steps=0)), db)
    assert r.status == "no_match" and r.matches == []
    assert r.closest_alternatives
    for alt in r.closest_alternatives:
        assert alt.violates and any("steps" in v.detail for v in alt.violates)


def test_unusable_combination_explains_itself(db):
    r = match_recipes(MatchRequest(available=[AI(name="peanuts")], constraints=Filters(exclude_classes=["peanut"])), db)
    assert r.status == "no_match" and any("peanut" in s for s in r.ignored_available)


def test_quantities_on_real_data_are_checked_or_marked_unverified(db):
    r = match_recipes(MatchRequest(available=[AI(name="egg", quantity=2), AI(name="rice", quantity=2, unit="cup"),
                                              AI(name="onion", quantity=1)],
                                   constraints=Filters(max_steps=6), k=10), db)
    seen = {u.quantity_check for m in r.matches for u in m.used}
    assert seen <= {"sufficient", "short", "not_verified"} and seen
    assert [m.score for m in r.matches] == sorted([m.score for m in r.matches], reverse=True)
