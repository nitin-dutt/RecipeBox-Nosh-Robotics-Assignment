"""Task 2 (ingredient matching) on the hand-checkable MATCH database (see conftest.MATCH).

The user has eggs, onions, tomatoes and rice; hard constraints: no peanuts, at most 6 steps. Hand-computed:

  recipe  used  still-needed  score(0.25)  steps
  M8       4         0           4.00         5     red onion counts as onion
  M1       3         1           2.75         4     soy sauce missing; salt is pantry
  M9       4         8           2.00         6     uses everything but needs 8 more
  M5       2         1           1.75         2     vinegar missing; olive oil is pantry
  M10      2         1           1.75         3     garlic powder missing
  M2       2         1           1.75         3     butter missing
  M7       1         0           1.00         1
  M6       1         3           0.25         3     green onions are NOT onions
  M3 (7 steps) and M4 (peanuts) are not eligible.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from recipebot.domain.models import AvailableIngredient as AI
from recipebot.domain.models import Filters, MatchRequest
from recipebot.tools.match import match_recipes, parse_available_text
from recipebot.tools.registry import call_tool

BASE = Filters(exclude_classes=["peanut"], max_steps=6)


def req(available=None, **kw):
    kw.setdefault("constraints", BASE)
    return MatchRequest(available=available or parse_available_text("eggs, onions, tomatoes, and rice"), **kw)


def ids(res):
    return [m.id for m in res.matches]


# ------------------------------------------------------------------ the brief's example
def test_example_request_balanced_ranking(match_db):
    r = match_recipes(req(k=8), match_db)
    assert r.status == "matches"
    assert ids(r) == ["M8", "M1", "M9", "M5", "M10", "M2", "M7", "M6"]
    assert [m.score for m in r.matches] == [4.0, 2.75, 2.0, 1.75, 1.75, 1.75, 1.0, 0.25]
    assert r.matches[0].n_used == 4 and r.matches[0].n_missing == 0


def test_strict_priority_ranking_puts_use_everything_first(match_db):
    r = match_recipes(req(k=8, ranking="priority"), match_db)
    assert ids(r) == ["M8", "M9", "M1", "M5", "M10", "M2", "M7", "M6"]
    assert all(m.score is None for m in r.matches)


def test_hard_constraints_are_never_relaxed(match_db):
    r = match_recipes(req(k=20), match_db)
    assert "M3" not in ids(r) and "M4" not in ids(r)          # 7 steps / peanuts
    assert len(r.matches) == 8 and r.closest_alternatives == []
    assert [(f.label, f.remaining) for f in r.computation.funnel][0] == ("population", 10)
    assert r.computation.funnel[-1].remaining == 8


def test_every_result_explains_its_rank(match_db):
    r = match_recipes(req(k=5), match_db)
    assert r.matches[0].why.startswith("Score 4.00 = 4 used - 0.25 x 0 still needed")
    assert r.matches[1].why.endswith("Ranked below #1 because its score is lower (2.75 vs 4.00).")
    assert r.matches[2].why.endswith("Ranked below #2 because its score is lower (2.00 vs 2.75).")
    m5, m10 = r.matches[3], r.matches[4]
    assert "Same score as #4" in m10.why and "more steps (3 vs 2)" in m10.why and m5.score == m10.score


def test_scores_obey_the_published_formula(match_db):
    for m in match_recipes(req(k=8), match_db).matches:
        assert m.score == pytest.approx(m.n_used - 0.25 * (m.n_missing + m.n_short))
        assert m.n_have + m.n_short + m.n_missing == m.n_required
        assert len(m.used) == m.n_used and len(m.used) + len(m.unused_available) == m.n_available


def test_matched_missing_and_pantry_are_shown_per_recipe(match_db):
    m1 = next(m for m in match_recipes(req(k=8), match_db).matches if m.id == "M1")
    assert [x.matched_with for x in m1.matched] == ["egg", "rice", "onion"]
    assert [x.line.text for x in m1.missing] == ["2 tablespoons soy sauce"]
    assert [x.line.text for x in m1.assumed_pantry] == ["1 teaspoon salt"]
    assert m1.matched[0].line.original == "2 eggs"            # verbatim source line is kept


# ------------------------------------------------------------------ equivalence rules
def test_red_onion_is_an_onion_but_green_onion_is_not(match_db):
    r = match_recipes(req(k=8), match_db)
    m8 = next(m for m in r.matches if m.id == "M8")
    assert any("onion" in (x.how or "") and "red onion" in x.line.text for x in m8.matched)
    m6 = next(m for m in r.matches if m.id == "M6")
    assert "onion" not in [u.available for u in m6.used]
    assert any("green onion" in x.line.text for x in m6.missing)


def test_a_different_form_does_not_match_and_is_reported_as_a_near_miss(match_db):
    r = match_recipes(req(available=[AI(name="garlic"), AI(name="rice")], constraints=Filters(), k=20), match_db)
    m10 = next(m for m in r.matches if m.id == "M10")
    assert any("garlic powder" in x.line.text for x in m10.missing)
    assert any("garlic" in n and "different form" in n for n in m10.near_misses)
    m3 = next(m for m in r.matches if m.id == "M3")
    assert "garlic" in [u.available for u in m3.used]            # real garlic cloves do match


def test_pantry_items_are_assumed_and_can_be_overridden(match_db):
    default = next(m for m in match_recipes(req(k=8), match_db).matches if m.id == "M7")
    assert default.n_required == 1 and default.n_missing == 0 and len(default.assumed_pantry) == 2
    none = next(m for m in match_recipes(req(k=8, pantry=[]), match_db).matches if m.id == "M7")
    assert none.n_required == 3 and none.n_missing == 2 and none.assumed_pantry == []


# ------------------------------------------------------------------ follow-up: "Exclude eggs too"
def test_exclude_eggs_updates_results_and_explains_the_conflict(match_db):
    r = match_recipes(req(constraints=BASE.model_copy(update={"exclude_ingredients": ["eggs"]}), k=8), match_db)
    assert ids(r) == ["M5", "M10", "M7"]                            # every recipe with egg text is gone
    assert any(s.startswith("egg") and "exclude" in s for s in r.ignored_available)
    assert "egg" not in [u.available for m in r.matches for u in m.used]


def test_having_onion_does_not_conflict_with_excluding_only_red_onion(match_db):
    r = match_recipes(req(constraints=BASE.model_copy(update={"exclude_ingredients": ["red onion"]})), match_db)
    assert not any("onion" in s for s in r.ignored_available)
    assert "M8" not in ids(r)


def test_available_ingredient_in_an_excluded_class_cannot_be_used(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="peanut")], constraints=BASE), match_db)
    assert r.status == "no_match" and r.matches == [] and r.ignored_available and "peanut" in r.ignored_available[0]


# ------------------------------------------------------------------ no match: honest, and labelled
def test_no_eligible_recipe_is_stated_and_alternatives_are_labelled(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="rice")], constraints=Filters(max_steps=0)), match_db)
    assert r.status == "no_match" and r.matches == []
    assert r.closest_alternatives and "VIOLATE" in r.message
    for alt in r.closest_alternatives:
        assert alt.violates and alt.relaxed_constraint == "at most 0 steps"
        assert any("steps" in v.detail for v in alt.violates)
    assert r.computation.funnel[-1].remaining == 0


def test_alternatives_can_be_switched_off(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="rice")], constraints=Filters(max_steps=0),
                                   offer_violating_alternatives=False), match_db)
    assert r.status == "no_match" and r.closest_alternatives == []


def test_alternative_that_breaks_an_exclusion_shows_the_offending_lines(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="peanuts")], constraints=Filters()), match_db)
    # not excluded here: peanuts are usable, M4 is the match
    assert ids(r) == ["M4"]
    r2 = match_recipes(MatchRequest(available=[AI(name="rice")], constraints=Filters(exclude_classes=["peanut"], max_steps=0)), match_db)
    assert r2.status == "no_match"
    viol = [v for a in r2.closest_alternatives for v in a.violates]
    assert viol and all(v.constraint in ("at most 0 steps", "no peanut") for v in viol)


def test_unrecognised_ingredient_is_reported_not_guessed(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="quinoa flour")], constraints=Filters()), match_db)
    assert r.status == "no_match" and r.unrecognised_ingredients and r.closest_alternatives == []


# ------------------------------------------------------------------ quantities
def test_quantity_sufficiency_for_counts(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="egg", quantity=2)], constraints=Filters(max_steps=6), k=10), match_db)
    by = {m.id: m for m in r.matches}
    assert by["M1"].matched[0].status == "have" and "needs 2" in by["M1"].matched[0].note       # exactly enough
    assert by["M2"].matched[0].status == "short" and by["M2"].n_short == 1                        # needs 3
    assert "short by 1" in by["M2"].matched[0].note
    # M2: uses 1 (egg); still needs tomato + butter (2 missing) and the 1 short egg line counts as unmet too
    assert by["M2"].n_missing == 2 and by["M2"].score == pytest.approx(1 - 0.25 * (2 + 1))


def test_quantity_not_verified_across_mass_and_volume(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="rice", quantity=300, unit="g")], constraints=Filters(max_steps=6), k=10), match_db)
    m7 = next(m for m in r.matches if m.id == "M7")
    item = m7.matched[0]
    assert item.status == "have_quantity_unverified" and "not verified" in item.note and "density" in item.note
    assert m7.used[0].quantity_check == "not_verified"


def test_quantity_volume_is_compared_when_units_are_comparable(match_db):
    ok = match_recipes(MatchRequest(available=[AI(name="rice", quantity=2, unit="cup")], constraints=Filters(max_steps=6), k=10), match_db)
    assert next(m for m in ok.matches if m.id == "M7").matched[0].status == "have"
    short = match_recipes(MatchRequest(available=[AI(name="rice", quantity=0.25, unit="cup")], constraints=Filters(max_steps=6), k=10), match_db)
    assert next(m for m in short.matches if m.id == "M7").matched[0].status == "short"


def test_no_quantities_means_name_level_only_and_says_so(match_db):
    r = match_recipes(req(), match_db)
    assert any("No quantities were given" in n for n in r.computation.notes)
    assert "not verified" in r.quantity_policy or "Otherwise" in r.quantity_policy


def test_quantity_ranking_is_sorted_by_the_effective_score(match_db):
    r = match_recipes(MatchRequest(available=[AI(name="egg", quantity=1), AI(name="rice"), AI(name="onion")],
                                   constraints=Filters(max_steps=6), k=8), match_db)
    scores = [m.score for m in r.matches]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ plumbing
def test_parse_available_text_extracts_quantities_and_units():
    got = {a.name: (a.quantity, a.unit) for a in parse_available_text("2 eggs, 500g rice and an onion")}
    assert got == {"egg": (2.0, None), "rice": (500.0, "g"), "onion": (1.0, None)}


def test_computation_record_is_complete(match_db):
    c = match_recipes(req(), match_db).computation
    assert c.tool == "match_recipes" and c.sql.startswith("WITH s0") and "score" in c.method.lower()
    assert any("Assumed on hand" in n for n in c.notes)


def test_registry_exposes_and_validates_match_recipes():
    with pytest.raises(ValidationError):
        call_tool("match_recipes", {"available": []})                  # at least one ingredient is required
    with pytest.raises(ValidationError):
        call_tool("match_recipes", {"available": [{"name": "egg"}], "priority": ["cheapest"]})
