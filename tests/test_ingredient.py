"""Ingredient parser tests. Inputs are real lines from the cleaned dataset."""
from __future__ import annotations

import pytest

from recipebot.domain.ingredient import parse_ingredient_line, singular


def one(text):
    items = parse_ingredient_line(text)
    assert len(items) == 1, items
    return items[0]


# ------------------------------------------------------------------ quantities and units
def test_simple_quantity_and_unit():
    p = one("2 tablespoons butter")
    assert (p.qty_min, p.unit, p.unit_kind, p.name) == (2.0, "tbsp", "volume", "butter")
    assert p.qty_base_min == pytest.approx(29.574, rel=1e-3)  # millilitres


def test_mixed_number_and_range():
    assert one("1 1/2 cups all-purpose flour").qty_min == 1.5
    r = one("5 to 6 medium size Yukon gold potatoes, peeled and quartered")
    assert (r.qty_min, r.qty_max, r.name) == (5.0, 6.0, "potato")


def test_pack_size_is_multiplied_out():
    p = one("2 (10.75 ounce) cans condensed cream of chicken soup")
    assert p.unit == "oz" and p.unit_kind == "mass"
    assert p.qty_base_min == pytest.approx(21.5 * 28.3495, rel=1e-3)


def test_hyphenated_can_size_and_word_number():
    p = one("One 14-ounce can chopped tomatoes")
    assert p.qty_min == 1 and p.name == "tomato" and p.qty_base_min == pytest.approx(14 * 28.3495, rel=1e-3)


def test_leading_dot_pack_size():
    assert one("1 (.25 ounce) package active dry yeast").name == "yeast"
    assert one("1/4 ounce package active dry yeast").name == "yeast"


def test_count_units_have_no_base_quantity():
    p = one("3 cloves garlic, minced")
    assert (p.unit, p.unit_kind, p.name, p.qty_base_min, p.prep) == ("clove", "count", "garlic", None, "minced")


def test_no_quantity_line():
    p = one("Salt, to taste")
    assert p.qty_min is None and p.name == "salt"


def test_pinch_without_number():
    p = one("Pinch of salt")
    assert (p.unit, p.name) == ("pinch", "salt")


def test_whole_cloves_is_the_spice():
    assert one("2 whole cloves").name == "clove"


# ------------------------------------------------------------------ names, descriptors, synonyms
@pytest.mark.parametrize(
    "text,name",
    [
        ("1 large egg", "egg"),
        ("3 large eggs, separated", "egg"),
        ("4 skinless, boneless chicken breast halves", "chicken breast"),
        ("1/4 cup extra-virgin olive oil", "olive oil"),
        ("2 teaspoons ground cinnamon", "cinnamon"),
        ("1/2 cup unsalted butter, softened", "butter"),
        ("1 cup sun-dried tomatoes", "sun dried tomato"),
        ("2 scallions, sliced", "green onion"),
        ("1 can garbanzo beans, drained", "chickpea"),
        ("1/2 cup confectioners' sugar", "powdered sugar"),
        ("1 teaspoon kosher salt", "salt"),
        ("1/2 teaspoon freshly ground black pepper", "black pepper"),
        ("1 cup Pillsbury all-purpose flour", "all purpose flour"),
        ("1 yellow onion, diced", "onion"),
        ("2 Roma tomatoes", "tomato"),
        ("1 cup half-and-half", "half and half"),
        ("1/2 teaspoon cream of tartar", "cream of tartar"),
        ("1 package cream cheese, softened", "cream cheese"),
        ("2 cups whole wheat flour", "whole wheat flour"),
        ("1 pound ground beef", "ground beef"),
        ("3 stalks celery, chopped", "celery"),
        ("a handful of basil leaves", "basil"),
    ],
)
def test_canonical_names(text, name):
    assert one(text).name == name


def test_products_that_differ_stay_different():
    assert one("1 teaspoon garlic powder").name == "garlic powder"
    assert one("3 cloves garlic").name == "garlic"
    assert one("1 can coconut milk").name == "coconut milk"
    assert one("1 cup milk").name == "milk"
    assert one("1 tablespoon tomato paste").name == "tomato paste"
    assert one("1 sweet potato").name == "sweet potato"
    assert one("1 green bell pepper").name != "black pepper"


# ------------------------------------------------------------------ base / form / parent
def test_name_base_form_parent():
    p = one("2 teaspoons garlic powder")
    assert (p.name, p.base, p.form) == ("garlic powder", "garlic", "powder")  # name != garlic, base == garlic
    t = one("1 tablespoon tomato paste")
    assert (t.name, t.base, t.form) == ("tomato paste", "tomato", "paste")
    bp = one("1 teaspoon baking powder")
    assert (bp.base, bp.form) == ("baking powder", None)  # atomic: not a kind of "baking"
    q = one("1 tablespoon lemon juice")
    assert (q.base, q.form) == ("lemon", "juice")
    r = one("1 red onion")
    assert (r.name, r.parent) == ("red onion", "onion")
    c = one("1 cup cheddar cheese")
    assert (c.name, c.parent) == ("cheddar cheese", "cheese")
    d = one("1 cup dried thyme")
    assert (d.base, d.form) == ("thyme", "dried")
    e = one("2 egg yolks")
    assert (e.base, e.form) == ("egg", "yolk")


# ------------------------------------------------------------------ alternatives and lists
def test_inline_or_alternative_shares_head_noun():
    p = one("1/2 cup chicken or vegetable stock")
    assert p.name == "chicken broth" and p.alternatives == ["vegetable broth"] and p.alt_source == "inline_or"


def test_comma_or_substitute_is_recorded_as_evidence():
    p = one("1 tablespoon Chinese black vinegar, or substitute good-quality balsamic vinegar")
    assert p.alternatives == ["balsamic vinegar"] and p.alt_source == "comma_or"


def test_paren_or_alternatives():
    p = one("2 tablespoons fig spread or preserves (or you can use plum, apricot, etc.)")
    assert "plum" in p.alternatives and "apricot" in p.alternatives and "etc" not in p.alternatives


def test_or_to_taste_is_not_an_alternative():
    assert one("1/2 teaspoon salt, or to taste").alternatives == []


def test_salt_and_pepper_split_into_two():
    items = parse_ingredient_line("Salt and freshly ground black pepper")
    assert [i.name for i in items] == ["salt", "black pepper"]
    items = parse_ingredient_line("1/2 tsp each salt and pepper")
    assert [i.name for i in items] == ["salt", "black pepper"] and all(i.qty_min == 0.5 for i in items)


def test_juice_and_zest_of():
    assert one("Juice of 1 lemon").name == "lemon juice"
    items = parse_ingredient_line("Zest and juice of 1 lemon")
    assert sorted(i.name for i in items) == ["lemon juice", "lemon zest"]


def test_optional_flag():
    assert one("a pinch of nutmeg (optional)").optional
    assert not one("1 cup flour").optional


def test_equipment_lines_are_not_ingredients():
    assert one("Special equipment: a deep-fat thermometer").status == "not_ingredient"


def test_nothing_raises_on_garbage():
    for t in ["", "   ", "(", ")))", "1", "1/0 cup", "¿", "ADVERTISEMENT", "Garnish:", "3-4", "x" * 500]:
        parse_ingredient_line(t)  # must not raise


@pytest.mark.parametrize(
    "word,expected",
    [("tomatoes", "tomato"), ("berries", "berry"), ("leaves", "leaf"), ("molasses", "molasses"),
     ("cookies", "cookie"), ("cloves", "clove"), ("asparagus", "asparagus"), ("peaches", "peach"),
     ("cheeses", "cheese"), ("hummus", "hummus"), ("chips", "chip")],
)
def test_singular(word, expected):
    assert singular(word) == expected


# ------------------------------------------------------------------ head-noun parents (found by the retrieval probe)
@pytest.mark.parametrize(
    "text,parent",
    [
        ("1 pound firm tofu", "tofu"), ("2 salmon fillets", "salmon"), ("1 tablespoon Dijon mustard", "mustard"),
        ("1 pound ground pork", "pork"), ("4 flour tortillas", "tortilla"), ("2 ounces dark rum", "rum"),
        ("1 cup Greek yogurt", "yogurt"), ("1 cup sharp cheddar cheese", "cheese"), ("1 tuna can albacore tuna", "tuna"),
    ],
)
def test_variety_names_roll_up_to_their_parent(text, parent):
    assert one(text).parent == parent


@pytest.mark.parametrize(
    "text,name",
    [
        ("1 can coconut milk", "coconut milk"), ("1/2 cup peanut butter", "peanut butter"),
        ("1 tuna steak", "tuna steak"), ("2 slices turkey bacon", "turkey bacon"), ("1 cup ice cream", "ice cream"),
        ("1 (10 ounce) frozen pound cake", "pound cake"),
    ],
)
def test_look_alikes_do_not_roll_up(text, name):
    p = one(text)
    assert p.name == name and p.parent == name


def test_quantity_words_do_not_leak_into_names():
    assert one("1 cup plus 2 tablespoons sugar").name == "sugar"
    assert one("1 fluid ounce vodka").name == "vodka"


def test_interchangeable_is_smaller_than_the_parent_list():
    from recipebot.tools.match import _members
    assert _members("onion")[0].interch is True
    assert _members("cheese")[0].interch is False and _members("tofu")[0].interch is False


# ------------------------------------------------------------------ bugs found by reading the LLM review queue
@pytest.mark.parametrize(
    "text,name",
    [
        ("8 hot dogs", "hot dog"),                                            # plural of a protected phrase
        ("1 (14.5 ounce) can peeled and diced tomatoes", "tomato"),           # was "and tomato": stray conjunction
        ("6 potatoes - peeled and cubed", "potato"),                          # was "potato and"
        ("1/2 pound bacon - cooked and crumbled", "bacon"),                   # was "bacon and"
        ("1 cup/200 g sugar", "sugar"),                                       # dual measure, was "g sugar"
        ("4 tablespoons/60 ml extra-virgin olive oil", "olive oil"),          # was "ml olive oil"
        ("water to cover", "water"),                                          # was "water cover"
        ("5 to 6 turns freshly ground black pepper", "black pepper"),         # was "turn black pepper"
        ("4 firm, ripe tomatoes, cut into 1-inch pieces", "firm tomato"),     # was "firm"
        ("1/2 cup roasted, unsalted peanuts", "roasted peanut"),              # was "roasted": the peanuts were lost
    ],
)
def test_parser_regressions_from_the_review_queue(text, name):
    assert one(text).name == name


def test_roasted_peanut_is_a_peanut_and_cooking_states_are_not_forms():
    p = one("1/2 cup roasted, unsalted peanuts")
    assert p.parent == "peanut" and p.form is None
    assert one("1 cup toasted pecans").name == "pecan"


def test_dual_measure_keeps_the_first_quantity():
    p = one("1 cup/200 g sugar")
    assert (p.qty_min, p.unit) == (1.0, "cup")


def test_conjunctions_inside_a_real_name_are_kept():
    assert one("1 cup half-and-half").name == "half and half"
    assert one("1 can condensed cream of mushroom soup").name == "condensed cream of mushroom soup"


def test_square_bracket_notes_are_dropped_like_parentheses():
    # found through the vegetarian filter: the note's words became part of the name, so "chicken stock" was not recognised
    p = parse_ingredient_line("4 cups (1 liter) chicken stock [for a vegetarian soup, use Garlic Broth ]")[0]
    assert p.name == "chicken broth"           # stock is a synonym of broth in the lexicon
    assert parse_ingredient_line("6 medium, ripe peaches [must be freestone]")[0].name == "peach"


def test_chicken_stock_with_a_vegetarian_aside_is_still_not_vegetarian():
    from recipebot.domain.taxonomy import get_taxonomy
    p = parse_ingredient_line("4 cups chicken stock [for a vegetarian soup, use Garlic Broth ]")[0]
    flags = get_taxonomy().classify_item(p.name, p.alternatives)
    assert flags.get("poultry") == "yes" and flags.get("vegetarian") == "yes"
