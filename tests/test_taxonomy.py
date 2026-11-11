"""Taxonomy tests: every false-positive guard and every definite hit is pinned by a real dataset phrase."""
from __future__ import annotations

import pytest

from recipebot.domain.taxonomy import get_taxonomy

T = get_taxonomy()


def status(name: str, cls: str, alts: tuple[str, ...] = ()) -> str | None:
    return T.classify_item(name, list(alts), rules_only=True).get(cls)    # these tests pin the RULES; the LLM overlay has its own tests


@pytest.mark.parametrize(
    "name,cls",
    [
        ("butter", "dairy"), ("heavy cream", "dairy"), ("parmesan cheese", "dairy"), ("buttermilk", "dairy"),
        ("sour cream", "dairy"), ("milk chocolate", "dairy"), ("condensed cream of mushroom soup", "dairy"),
        ("egg", "egg"), ("mayonnaise", "egg"), ("egg yolk", "egg"), ("egg noodle", "egg"),
        ("peanut oil", "peanut"), ("creamy peanut butter", "peanut"),
        ("walnut", "tree_nut"), ("almond milk", "tree_nut"), ("pine nut", "tree_nut"),
        ("penne", "gluten"), ("all purpose flour", "gluten"), ("flour tortilla", "gluten"), ("breadcrumb", "gluten"),
        ("soy sauce", "soy"), ("tofu", "soy"),
        ("shrimp", "shellfish"), ("oyster sauce", "shellfish"), ("salmon fillet", "fish"), ("fish sauce", "fish"),
        ("bacon", "pork"), ("ground beef", "beef"), ("chicken broth", "poultry"), ("ground lamb", "lamb"),
        ("sesame oil", "sesame"), ("white wine", "alcohol"), ("honey", "vegan"), ("bacon", "vegetarian"),
    ],
)
def test_definite_hits(name, cls):
    assert status(name, cls) == "yes"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("coconut milk", "dairy"), ("almond milk", "dairy"), ("peanut butter", "dairy"), ("cocoa butter", "dairy"),
        ("cream of tartar", "dairy"), ("butternut squash", "dairy"), ("butter bean", "dairy"),
        ("butter flavored shortening", "dairy"), ("bean curd", "dairy"), ("swiss chard", "dairy"),
        ("dairy free cheese", "dairy"), ("vegan butter", "dairy"),
        ("eggplant", "egg"), ("egg replacer", "egg"), ("flaxseed egg", "egg"), ("rice noodle", "egg"),
        ("nutmeg", "tree_nut"), ("butternut squash", "tree_nut"), ("water chestnut", "tree_nut"),
        ("pine nut", "peanut"), ("macadamia nut", "peanut"),
        ("rice flour", "gluten"), ("almond flour", "gluten"), ("corn tortilla", "gluten"), ("rice noodle", "gluten"),
        ("pasta sauce", "gluten"), ("gluten free pasta", "gluten"), ("buckwheat", "gluten"),
        ("oyster mushroom", "shellfish"), ("crab apple", "shellfish"), ("goldfish cracker", "fish"),
        ("turkey bacon", "pork"), ("hamburger bun", "beef"), ("ham steak", "beef"), ("poultry seasoning", "poultry"),
        ("graham cracker", "pork"), ("crab meat", "meat"), ("meatless crumble", "meat"), ("vegetarian sausage", "meat"),
        ("pigeon pea", "game"), ("goat cheese", "game"), ("buffalo sauce", "game"),
        ("root beer", "alcohol"), ("red wine vinegar", "alcohol"), ("champagne vinegar", "alcohol"),
        ("scotch bonnet", "alcohol"), ("kale", "alcohol"), ("crumb", "alcohol"),
        ("sugar", "vegan"),
    ],
)
def test_false_positive_guards(name, cls):
    assert status(name, cls) is None, f"{name} wrongly flagged {cls}: {T.classify_name(name, rules_only=True)}"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("margarine", "dairy"), ("chocolate chip", "dairy"), ("vanilla extract", "alcohol"),
        ("worcestershire sauce", "fish"), ("coconut", "tree_nut"), ("pesto", "tree_nut"),
        ("crab boil seasoning", "shellfish"), ("hummus", "sesame"), ("tortilla chip", "gluten"),
        ("rolled oat", "gluten"), ("hoisin sauce", "soy"), ("sausage", "pork"),
    ],
)
def test_possible_hits_are_maybe_not_yes(name, cls):
    assert status(name, cls) == "maybe"


def test_composites_roll_up():
    assert status("bacon", "meat") == "yes"
    assert status("shrimp", "vegetarian") == "yes"
    assert status("butter", "vegan") == "yes"
    assert status("egg", "vegan") == "yes"
    assert status("gelatin", "vegetarian") == "yes"
    assert status("gelatin", "meat") is None  # gelatin is not meat, but is not vegetarian


def test_or_alternatives_downgrade_to_maybe():
    # the cook can choose the compliant option, so it is not a definite violation
    assert status("butter", "dairy", ("margarine",)) == "maybe"
    assert status("butter", "dairy", ("cream",)) == "yes"          # both options dairy
    assert status("chicken broth", "poultry", ("vegetable broth",)) == "maybe"
    assert status("olive oil", "dairy", ("vegetable oil",)) is None


# ------------------------------------------------------------------ found by the evaluation audit
def _tax():
    from recipebot.domain.taxonomy import get_taxonomy
    return get_taxonomy()


def test_exemption_covers_only_the_phrase_it_modifies():
    t = _tax()
    assert t.classify_name("vegetarian chicken nuggets") == {}                     # the exempt word modifies chicken
    line = "4 cups chicken stock for a vegetarian soup use garlic broth"           # ... but here it does not
    assert t.classify_name(line).get("poultry") == "yes" and t.classify_name(line).get("vegetarian") == "yes"


def test_real_bacon_bits_are_pork_and_imitation_ones_are_not():
    t = _tax()
    assert t.classify_name("real bacon bit", rules_only=True).get("pork") == "yes"
    assert t.classify_name("imitation bacon bit", rules_only=True) == {}


def test_a_human_decision_on_a_base_class_reaches_the_composites():
    # review decision: plain "bacon bit" is a possible pork. The vegetarian/meat classes must see it as "maybe" too,
    # otherwise a strict vegetarian search still returns recipes with bacon bits (the audit found three).
    flags = _tax().classify_name("bacon bit")
    assert flags.get("pork") == "maybe" and flags.get("meat") == "maybe" and flags.get("vegetarian") == "maybe"
