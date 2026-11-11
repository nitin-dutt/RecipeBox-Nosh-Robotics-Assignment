"""Unit tests for the cleaning rules. Cases are taken from real records in the dataset."""
from __future__ import annotations

from collections import Counter

import pytest

from recipebot.ingestion.cleaning import (
    clean_ingredient_line,
    clean_ingredients,
    clean_instructions,
    clean_record,
    is_section_header,
    normalize_text,
    split_sentences,
)


def hits() -> tuple[Counter, callable]:
    c: Counter = Counter()
    return c, lambda rule: c.update([rule])


# ---------------------------------------------------------------- advertisement
def test_advertisement_suffix_removed():
    c, h = hits()
    assert clean_ingredient_line("2 tablespoons butter ADVERTISEMENT", h) == "2 tablespoons butter"
    assert c["advertisement_removed"] == 1


def test_bare_advertisement_line_is_dropped():
    c, h = hits()
    out = clean_ingredients(["1 onion ADVERTISEMENT", "ADVERTISEMENT"], h)
    assert out.lines == ["1 onion"]
    assert c["empty_ingredient_line_dropped"] == 1


# ---------------------------------------------------------------- encoding / markup
def test_broken_entities_decoded():
    c, h = hits()
    assert normalize_text("sweet piment&ocute;n (smoked paprika)", h) == "sweet pimentón (smoked paprika)"
    assert normalize_text("red jalape&ntild;eo", h) == "red jalapeño"
    assert normalize_text("Salt &amp; pepper", h) == "Salt & pepper"
    assert normalize_text("saut&eactue; at 350&Deg;F", h) == "sauté at 350°F"
    assert normalize_text("mini M&M;'s and PB&J;", h) == "mini M&M's and PB&J"
    assert normalize_text("a pinch&151;or two", h) == "a pinch—or two"


def test_unicode_fractions_and_spaces():
    _, h = hits()
    assert normalize_text("1½ cups flour", h) == "1 1/2 cups flour"
    assert normalize_text("½ cup milk", h) == "1/2 cup milk"


def test_brand_marks_removed_from_clean_text():
    _, h = hits()
    assert normalize_text("1 package Kraft® cheese", h) == "1 package Kraft cheese"
    assert normalize_text("Philadelphia™ cream cheese", h) == "Philadelphia cream cheese"


def test_epicurious_markup_recovered():
    _, h = hits()
    garbled = 'epi:recipeLink id="102487"<buttermilk corn="" bread="">/epi:recipeLink<, very coarsely crumbled</buttermilk>'
    assert normalize_text(garbled, h) == "buttermilk corn bread, very coarsely crumbled"
    assert normalize_text("<special>Special equipment: 2-quart dish</special>", h) == "Special equipment: 2-quart dish"
    assert normalize_text("Accompaniment: creamy polentaepi:recipelink</epi:recipelink>", h) == "Accompaniment: creamy polenta"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("manufacturer¿s instructions", "manufacturer's instructions"),
        ("confectioners¿ sugar", "confectioners' sugar"),
        ("paper towel¿lined plate", "paper towel-lined plate"),
        ("Recipe Creations¿ refrigerated dough", "Recipe Creations refrigerated dough"),
    ],
)
def test_lossy_char_repaired(raw, expected):
    c, h = hits()
    assert normalize_text(raw, h) == expected
    assert c["lossy_char_repaired"] == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Preheat oven to 425¿F/220¿C", "Preheat oven to 425°F/220°C"),
        ("1¿2 cup sugar and 1 1¿4 teaspoon salt", "1/2 cup sugar and 1 1/4 teaspoon salt"),
        ("a 9 ¿13-inch pan", "a 9 x 13-inch pan"),
        ("60%¿70% cacao", "60%-70% cacao"),
        ("melt the butter ¿ heat it", "melt the butter — heat it"),
    ],
)
def test_lossy_char_numeric_contexts(raw, expected):
    _, h = hits()
    assert normalize_text(raw, h) == expected


def test_all_caps_abbreviation_is_ingredient_not_header():
    assert not is_section_header("EVOO")


def test_lossy_char_unresolved_is_flagged_not_guessed():
    c, h = hits()
    assert "¿" in normalize_text("add 1 ¿2 teaspoon salt", h)
    assert c["lossy_char_unresolved"] == 1


def test_correct_accents_untouched():
    _, h = hits()
    assert normalize_text("Sauté at 350°F with crème fraîche", h) == "Sauté at 350°F with crème fraîche"


# ---------------------------------------------------------------- section headers
@pytest.mark.parametrize("line", ["For the sauce:", "FILLING", "Topping:", "For the crust"])
def test_headers_detected(line):
    assert is_section_header(line)


@pytest.mark.parametrize(
    "line", ["Garnish: ground nutmeg", "1 cup flour", "Salt and pepper to taste", "2 eggs", "MSG"]
)
def test_real_ingredients_not_headers(line):
    assert not is_section_header(line)


def test_header_becomes_group_label():
    _, h = hits()
    out = clean_ingredients(["For the sauce:", "1 cup cream", "2 eggs", "Topping:", "1 tsp salt"], h)
    assert out.lines == ["1 cup cream", "2 eggs", "1 tsp salt"]
    assert out.groups == ["For the sauce", "For the sauce", "Topping"]


# ---------------------------------------------------------------- instructions
def test_duplicated_instruction_halves_removed():
    c, h = hits()
    raw = "Beat the eggs.\nBake it.\nBeat the eggs.\nBake it."
    out = clean_instructions(raw, h)
    assert out.text == "Beat the eggs.\nBake it."
    assert out.n_paragraphs == 2
    assert c["duplicate_instructions_removed"] == 1


def test_non_duplicate_not_collapsed():
    _, h = hits()
    out = clean_instructions("Mix.\nBake.\nMix.\nServe.", h)
    assert out.n_paragraphs == 4


def test_boilerplate_removed_and_not_counted_as_steps():
    c, h = hits()
    raw = "Watch how to make this recipe.\nPreheat the oven to 350 degrees F.\nBake 20 minutes. Photograph by Con Poulos"
    out = clean_instructions(raw, h)
    assert out.text == "Preheat the oven to 350 degrees F.\nBake 20 minutes."
    assert out.n_paragraphs == 2
    assert c["boilerplate_line_removed"] == 2


@pytest.mark.parametrize(
    "credit",
    ["Photograph by Karl Juengel/Studio D", "Photograph by Gentl & Hyers", "Photograph by Penny de los Santos"],
)
def test_photo_credit_variants_removed(credit):
    _, h = hits()
    out = clean_instructions(f"Serve warm.\n{credit}", h)
    assert out.text == "Serve warm." and out.n_paragraphs == 1
    out = clean_instructions(f"Serve warm. {credit}", h)
    assert out.text == "Serve warm."


def test_real_step_starting_with_watch_is_kept():
    _, h = hits()
    out = clean_instructions("Watch as you beat and stop when it gets thick.", h)
    assert out.n_paragraphs == 1


def test_sentence_counting():
    assert split_sentences("Mix well. Add 1.5 cups of water. Bake at 350 degrees F. Cool.") == [
        "Mix well.", "Add 1.5 cups of water.", "Bake at 350 degrees F.", "Cool.",
    ]


# ---------------------------------------------------------------- idempotence & records
def test_cleaning_is_idempotent():
    _, h = hits()
    raw = {
        "title": "  Rago&ocute;t ",
        "ingredients": ["1½ cups milk ADVERTISEMENT", "For the sauce:", "2 eggs"],
        "instructions": "Mix.\nBake.\nMix.\nBake.",
        "picture_link": None,
    }
    first = clean_record("id1", "epi", raw, h)
    again = clean_record(
        "id1", "epi",
        {"title": first.title, "ingredients": first.ingredients, "instructions": first.instructions, "picture_link": None},
        h,
    )
    assert (again.title, again.ingredients, again.instructions) == (first.title, first.ingredients, first.instructions)


def test_incomplete_record_flagged_and_raw_preserved():
    _, h = hits()
    raw = {"title": "Mystery", "ingredients": [], "instructions": "", "picture_link": None}
    rec = clean_record("id2", "fn", raw, h)
    assert not rec.is_complete and not rec.has_ingredients and rec.has_title
    assert rec.title_raw == "Mystery"
