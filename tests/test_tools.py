"""Tool behaviour on the hand-checkable mini database (see conftest.MINI for the 11 recipes).

Complete recipes: R1-R6, R8-R11 = 10 (R7 has no ingredients/instructions). R2 and R8 are exact duplicates.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from recipebot.domain.models import Filters, Population, SearchSpec
from recipebot.tools import capabilities, stats
from recipebot.tools.recipe import get_recipe
from recipebot.tools.registry import TOOLS, call_tool
from recipebot.tools.search import query_tokens, search_recipes

pytestmark = pytest.mark.usefixtures("mini_db")


def ids(result):
    return {h.id for h in result.recipes}


# ------------------------------------------------------------------ counting and percentages
def test_population_and_percentage_denominator():
    c = stats.count_recipes(Filters(include_ingredients=["tomato"]))
    assert c.denominator == 10                      # complete recipes only
    assert c.count == 4                             # R2, R5, R8, R10; R1's "tomato sauce" is a different product
    assert c.percentage == pytest.approx(40.0)
    wider = stats.count_recipes(Filters(include_ingredients=["tomato"], match_level="base"))
    assert wider.count == 5                         # base level also counts tomato sauce (R1)


def test_garlic_and_onion_excludes_garlic_powder_by_default():
    c = stats.count_recipes(Filters(include_ingredients=["garlic", "onion"]))
    assert c.count == 1                             # only R5; R4 has garlic POWDER
    c_base = stats.count_recipes(Filters(include_ingredients=["garlic", "onion"], match_level="base"))
    assert c_base.count == 2                        # base level widens to garlic powder (R4)
    assert [s.remaining for s in c.computation.funnel][-1] == 1


def test_incomplete_recipes_only_counted_when_requested():
    all_ = stats.count_recipes(Filters(population=Population(complete_only=False)))
    assert all_.denominator == 11 and all_.count == 11
    assert stats.count_recipes(Filters()).count == 10


def test_dedupe_collapses_exact_duplicates():
    kept = stats.count_recipes(Filters(population=Population(dedupe=True)))
    assert kept.count == 9                          # R2/R8 collapse to one


def test_source_filter():
    c = stats.count_recipes(Filters(population=Population(sources=["ar"])))
    assert c.count == 5                             # R1 R2 R5 R9 R10


def test_exclude_ingredient_is_broad_and_text_based():
    c = stats.count_recipes(Filters(exclude_ingredients=["garlic"]))
    assert c.count == 10 - 5                        # R1 R2 R4(garlic powder) R5 R8 contain garlic text
    assert stats.count_recipes(Filters(exclude_ingredients=["egg"])).count == 8   # R9 and R11


def test_unknown_term_falls_back_to_text_and_is_reported():
    r = search_recipes(SearchSpec(include_ingredients=["safflower"]))
    assert "safflower" in r.unresolved_terms and ids(r) == {"R11"}
    assert any("not a known ingredient" in n for n in r.computation.notes)


def test_group_expands_pasta_to_members():
    r = search_recipes(SearchSpec(include_ingredients=["pasta"], limit=20))
    assert {"R1", "R2", "R3", "R8"} <= ids(r)
    assert any("group pasta" in n for n in r.computation.notes)


# ------------------------------------------------------------------ restrictions
def test_dairy_strict_removes_definite_and_possible():
    r = search_recipes(SearchSpec(query="penne", exclude_classes=["dairy"], limit=20))
    assert ids(r) == {"R2", "R8"}                   # R1 parmesan, R3 cream/butter/parmesan removed
    assert r.disclaimer and "do not certify" in r.disclaimer
    assert [f.label for f in r.computation.funnel][0] == "population"


def test_margarine_is_maybe_so_strict_drops_it_and_lenient_keeps_it_with_a_warning():
    strict = search_recipes(SearchSpec(query="cookies", exclude_classes=["dairy"]))
    assert strict.total_matches == 0 and strict.excluded_only_by_maybe == 1
    lenient = search_recipes(SearchSpec(query="cookies", exclude_classes=["dairy"], restriction_mode="lenient"))
    assert ids(lenient) == {"R9"}
    w = lenient.recipes[0].warnings
    assert w and w[0].restriction == "dairy" and w[0].status == "maybe" and "margarine" in w[0].lines[0].text


def test_peanut_in_an_or_list_is_not_missed():
    """The recall bug found in phase 2: 'canola, safflower or peanut oil' must never pass a no-peanut filter."""
    r = search_recipes(SearchSpec(query="rice", exclude_classes=["peanut"], limit=20))
    assert "R11" not in ids(r)
    len_ = search_recipes(SearchSpec(query="rice", exclude_classes=["peanut"], restriction_mode="lenient"))
    note = next(w for h in len_.recipes if h.id == "R11" for w in h.warnings)
    assert note.via == "text" and "peanut oil" in note.lines[0].text


def test_peanut_butter_is_a_definite_peanut_hit_but_not_dairy():
    assert "R6" not in ids(search_recipes(SearchSpec(query="noodles", exclude_classes=["peanut"])))
    assert "R6" in ids(search_recipes(SearchSpec(query="noodles", exclude_classes=["dairy"])))


# ------------------------------------------------------------------ search behaviour
def test_search_returns_ids_titles_and_matching_ingredient_evidence():
    r = search_recipes(SearchSpec(query="penne", limit=3))
    top = r.recipes[0]
    assert top.id and top.title and any("penne" in m.text for m in top.matched_ingredients)
    assert top.matched_ingredients[0].original is not None   # verbatim source line is preserved


def test_all_words_then_any_word_fallback_is_disclosed():
    r = search_recipes(SearchSpec(query="penne smoothie"))
    assert r.total_matches > 0 and any("ANY word" in n for n in r.computation.notes)


def test_optional_fields_only_when_requested():
    base = search_recipes(SearchSpec(query="soup")).recipes[0]
    assert base.ingredients is None and base.instructions is None
    full = search_recipes(SearchSpec(query="soup", include_ingredient_lines=True, include_instructions=True)).recipes[0]
    assert len(full.ingredients) == 4 and "Simmer" in full.instructions


def test_ordering_is_deterministic():
    a = [h.id for h in search_recipes(SearchSpec(query="penne", limit=20)).recipes]
    b = [h.id for h in search_recipes(SearchSpec(query="penne", limit=20)).recipes]
    assert a == b


def test_max_steps_filter_uses_the_stated_rule():
    r = search_recipes(SearchSpec(include_ingredients=["tomato"], max_steps=1, limit=20))
    assert ids(r) == {"R10"}                        # "Blend." is a single sentence


# ------------------------------------------------------------------ ranking and ties
def test_rank_by_steps_orders_and_discloses_ties():
    k = stats.rank_by_steps(Filters(include_ingredients=["tomato"]), k=2)
    assert k.ranking[0].id == "R10" and k.ranking[0].steps == 1
    assert [x.rank for x in k.ranking] == [1, 2]
    assert "recipe ID" in k.tie_break
    big = stats.rank_by_steps(Filters(), k=1, order="fewest")
    assert big.n_tied_at_cutoff >= 1


# ------------------------------------------------------------------ aggregates
def test_top_ingredients_counts_recipes_once_and_skips_filter_term():
    t = stats.top_ingredients(Filters(include_ingredients=["tomato"]), k=5)
    names = [i.ingredient for i in t.items]
    assert "tomato" not in names and t.computation.n_analyzed == 4
    assert t.items[0].ingredient == "garlic" and t.items[0].recipes == 3   # R2, R5, R8
    assert all(0 < i.percentage <= 100 for i in t.items)


def test_ingredient_count_summary_matches_hand_calculation():
    s = stats.ingredient_count_summary(Filters())
    counts = sorted([5, 4, 4, 4, 4, 3, 4, 4, 3, 3])  # R1..R6,R8..R11 ingredient lines
    assert s.groups[0].n == 10 and s.groups[0].mean == pytest.approx(sum(counts) / 10, abs=0.01)


# ------------------------------------------------------------------ computation record
def test_every_result_carries_a_computation_record():
    for res in (search_recipes(SearchSpec(query="soup")), stats.count_recipes(Filters()),
                stats.top_ingredients(Filters()), stats.rank_by_steps(Filters())):
        c = res.computation
        assert c.sql.startswith("WITH s0") and c.method and c.funnel and c.n_analyzed >= 0
        assert c.funnel[0].label == "population" and c.denominator is not None


# ------------------------------------------------------------------ recipes, refusals, registry
def test_get_recipe_returns_clean_and_original_text():
    rec = get_recipe("R1")
    assert rec.title == "Penne with Tomato Sauce" and rec.ingredients[2].text.startswith("3 cloves garlic")
    assert rec.ingredients[0].original == "1 pound penne" and get_recipe("nope") is None


@pytest.mark.parametrize("q,attr", [
    ("how many calories are in the lasagna", "nutrition"), ("how long does it take to cook", "cooking time"),
    ("how many servings does it make", "servings"), ("which is the most popular", "ratings and popularity"),
    ("what cuisine is this", "cuisine"),
])
def test_unsupported_attributes_are_refused_with_explanation(q, attr):
    r = capabilities.check_attribute(q)
    assert r and r.attribute == attr and r.reason and r.can_offer_instead


def test_supported_questions_are_not_refused():
    assert capabilities.check_attribute("which recipes contain garlic") is None


def test_registry_validates_arguments():
    assert {"search_recipes", "count_recipes", "rank_by_steps", "check_supported"} <= set(TOOLS)
    with pytest.raises(ValidationError):
        call_tool("count_recipes", {"filters": {"match_level": "fuzzy"}})
    out = call_tool("count_recipes", {"filters": {"include_ingredients": ["tomato"]}})
    assert out.count >= 1


def test_query_tokens_drop_filler_words():
    assert query_tokens("Find me a penne pasta recipe, please") == ["penne", "pasta"]
