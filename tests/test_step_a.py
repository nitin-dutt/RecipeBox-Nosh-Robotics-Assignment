"""Hardening round 1: instruction recall net, incomplete recipes never 'clear', composite names, config versioning."""
from __future__ import annotations

from types import SimpleNamespace

import duckdb
import pytest
from conftest import _build

from recipebot.domain import versioning
from recipebot.domain.ingredient import decompose
from recipebot.domain.instruction_scan import scan_instructions
from recipebot.domain.models import AvailableIngredient as AI
from recipebot.domain.models import Filters, MatchRequest, Population, SearchSpec
from recipebot.store.db import Database, StaleDataError, verify_lexicon
from recipebot.tools.match import match_recipes
from recipebot.tools.search import search_recipes


def classes(text):
    return {c for c, _ in scan_instructions(text)}


# ------------------------------------------------------------------ instruction scan
@pytest.mark.parametrize("text", [
    "Cream the shortening and sugar until fluffy.",
    "Roll the dough into walnut-sized balls.",
    "Strain through cheese cloth.",
    "Cook until the sauce is the color of peanut butter.",
    "Bake until deep golden brown.",
])
def test_ingredient_words_used_as_verbs_shapes_or_colours_are_not_flagged(text):
    assert classes(text) == set()


def test_toast_as_a_verb_is_not_bread_but_the_walnuts_it_toasts_still_count():
    found = classes("Toast the walnuts on a baking sheet.")
    assert "gluten" not in found and "tree_nut" in found


@pytest.mark.parametrize("text,cls", [
    ("Brush the top with melted butter.", "dairy"),
    ("Heat a deep pot of peanut oil to 350 degrees.", "peanut"),
    ("Serve with sour cream.", "dairy"),
    ("Top with crumbled gorgonzola.", "dairy"),
    ("Stir in the nuts.", "peanut"),
    ("Spread on toast and serve.", "gluten"),
])
def test_real_ingredients_that_only_the_instructions_mention_are_flagged(text, cls):
    assert cls in classes(text)


def test_cream_as_a_verb_does_not_remove_the_real_ingredient_next_to_it():
    assert "dairy" in classes("Cream butter and brown sugar until light.")


def test_snippet_points_at_the_real_hit_not_a_guarded_look_alike():
    text = "Simmer the coconut milk for 5 minutes. At the very end, stir in 2 tablespoons butter and serve."
    snippet = dict(scan_instructions(text))["dairy"]
    assert "butter" in snippet


def test_instruction_hits_are_stored_as_maybe_never_yes(mini_db):
    assert mini_db.query("SELECT DISTINCT status FROM instruction_flags") in ([], [("maybe",)])


# ------------------------------------------------------------------ a tiny database with instruction-only usage
DATA = [
    ("I1", "ar", "Glazed Apple", ["2 apples"], "Slice the apples. Brush with melted butter."),
    ("I2", "ar", "Dry Rub Chicken", ["1 chicken breast", "1 teaspoon salt"], "Cream the salt. Roll into walnut-sized balls."),
    ("I3", "fn", "Mystery Dish", [], ""),
]


@pytest.fixture(scope="module")
def idb():
    con = duckdb.connect(":memory:")
    _build(con, DATA)
    db = Database(connection=con)
    yield db
    con.close()


def test_strict_filter_drops_the_recipe_that_only_mentions_butter_in_its_instructions(idb):
    r = search_recipes(SearchSpec(query="apple", exclude_classes=["dairy"]), idb)
    assert r.total_matches == 0 and r.excluded_only_by_maybe == 1


def test_lenient_filter_keeps_it_and_shows_the_instruction_snippet(idb):
    r = search_recipes(SearchSpec(query="apple", exclude_classes=["dairy"], restriction_mode="lenient"), idb)
    note = r.recipes[0].warnings[0]
    assert note.restriction == "dairy" and note.via == "instructions" and note.lines == []
    assert "butter" in note.snippet


def test_a_recipe_with_no_ingredient_list_is_never_reported_as_free(idb):
    wide = Filters(population=Population(complete_only=False), exclude_classes=["peanut"])
    from recipebot.tools import stats
    c = stats.count_recipes(wide, idb)
    assert c.denominator == 2                              # I3 (no ingredients) is excluded even though asked for
    assert any("Incomplete recipes are excluded" in n for n in c.computation.notes)
    # without any restriction, incomplete recipes can still be counted when asked
    assert stats.count_recipes(Filters(population=Population(complete_only=False)), idb).denominator == 3


def test_matcher_reports_instruction_mentions_as_violations_of_the_original_constraints(idb):
    r = match_recipes(MatchRequest(available=[AI(name="apple")], constraints=Filters(exclude_classes=["dairy"])), idb)
    assert r.status == "no_match" and r.matches == []
    alt = r.closest_alternatives[0]
    assert alt.recipe.id == "I1" and "instructions mention" in alt.violates[0].detail


# ------------------------------------------------------------------ composite names
@pytest.mark.parametrize("name", ["chili oil and lime", "plain yogurt and cilantro", "basmati rice and naan bread",
                                  "sweet potato sour cream and mint", "bread with seeds"])
def test_composite_names_do_not_borrow_the_last_word_as_parent(name):
    assert decompose(name)[2] == name


def test_normal_names_still_roll_up():
    assert decompose("dark rum")[2] == "rum" and decompose("salmon")[2] == "salmon"


# ------------------------------------------------------------------ config versioning
def test_fingerprint_changes_when_any_config_file_changes(tmp_path, monkeypatch):
    for n in ("ingredients.yaml", "taxonomy.yaml", "groups.yaml"):
        (tmp_path / n).write_text("a: 1")
    monkeypatch.setattr(versioning, "get_settings", lambda: SimpleNamespace(config_dir=tmp_path))
    before = versioning.lexicon_fingerprint()
    assert versioning.lexicon_fingerprint() == before
    (tmp_path / "taxonomy.yaml").write_text("a: 2")
    assert versioning.lexicon_fingerprint() != before
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "ingredient_labels.json").write_text("{}")      # a reviewed overlay is part of the version
    assert versioning.lexicon_fingerprint() not in (before,)


def test_tools_refuse_data_built_from_a_different_config():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE build_meta(key VARCHAR, value VARCHAR)")
    con.execute("INSERT INTO build_meta VALUES ('lexicon_fingerprint', 'deadbeef')")
    with pytest.raises(StaleDataError, match="--stage2"):
        verify_lexicon(Database(connection=con))
    con.execute("DELETE FROM build_meta")
    con.execute("INSERT INTO build_meta VALUES ('lexicon_fingerprint', ?)", [versioning.lexicon_fingerprint()])
    verify_lexicon(Database(connection=con))                                  # matching version: fine
    verify_lexicon(Database(connection=duckdb.connect(":memory:")))           # hand-built database: no check


def test_real_database_matches_the_config_on_disk():
    from recipebot.settings import get_settings
    if not get_settings().db_path.exists():
        pytest.skip("no database")
    verify_lexicon(Database(get_settings().db_path))
