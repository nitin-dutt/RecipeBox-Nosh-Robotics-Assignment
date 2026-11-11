"""INDEPENDENT checks for the evaluation: plain regular expressions over the RAW ingredient lines.

They deliberately do not use our parser, lexicon, taxonomy or LLM labels, so they can disagree with the system. A disagreement is
then either a system miss (a leak) or a mistake in these simple patterns; the audit lists them so a human can decide which.
"""
from __future__ import annotations

import re

# phrases that contain a restricted word but are NOT the restricted food (removed before matching)
_NOT_DAIRY = (r"coconut (?:milk|cream)|cream of coconut|almond (?:milk|butter)|soy ?milk|oat ?milk|rice ?milk|cashew (?:milk|cream|butter)|"
              r"peanut butter|nut butter|cocoa butter|apple butter|shea butter|sunflower (?:seed )?butter|butter (?:beans?|lettuce|bean)|"
              r"coconut whipped cream|cream of tartar|non-?dairy|dairy[- ]free|vegan (?:butter|cheese|cream(?: cheese)?)|plant[- ]based (?:milk|butter|cheese)|"
              r"nutritional yeast|cream soda|root beer")
_DAIRY = r"\b(?:milk|cream|cheese|cheeses|butter|yogurt|yoghurt|whey|ghee|buttermilk|custard|half[- ]and[- ]half|parmesan|mozzarella|cheddar|ricotta|feta)\b"
_NOT_MEAT = r"vegetarian (?:chicken|beef|bacon|sausage|burger|ground)|veggie (?:burger|sausage|chicken)|meatless|imitation (?:crab|bacon)|chicken[- ]style|beef[- ]style|vegan (?:bacon|sausage)"
_MEAT = (r"\b(?:chicken|beef|pork|bacon|ham|sausages?|turkey|lamb|veal|steaks?(?! (?:sauce|seasoning|spice))|prosciutto|pepperoni|salami|chorizo|fish|salmon|tuna|cod|tilapia|"
         r"shrimp|prawns?|crab|lobster|clams?|oysters?|mussels?|anchov(?:y|ies)|sardines?|duck|venison|bison|(?:spare|short|baby back|pork|beef) ribs|spareribs|brisket|hot dogs?|gelatin|lard|"
         r"ground meat|pancetta|mortadella|bratwurst|kielbasa)\b")
_NOT_EGG = r"egg ?plant|egg (?:replacer|substitute)|vegan egg|egg-?free"
_EGG = r"\beggs?\b|\begg yolks?\b|\begg whites?\b|\bmayonnaise\b"
_PEANUT = r"\b(?:peanuts?|groundnuts?)\b"

_SHELLFISH = r"\b(?:shrimps?|prawns?|crabs?|lobsters?|clams?|oysters?|mussels?|scallops?|crawfish|crayfish)\b"
_PORK = r"\b(?:pork|bacon|ham|prosciutto|pancetta|lard|spareribs|chorizo|pepperoni|salami)\b"
_NOT_PORK = r"turkey bacon|vegan bacon|imitation bacon|bacon (?:flavou?red|bits? \(imitation\))|ham(?:burger)s?|graham"
_GLUTEN = r"\b(?:flour|wheat|breadcrumbs?|bread|pasta|noodles?|spaghetti|macaroni|penne|couscous|barley|rye|tortillas?|crackers?|pita|bagels?)\b"
_NOT_GLUTEN = (r"gluten[- ]free|(?:rice|almond|coconut|sorghum|tapioca|amaranth|soy|potato|chickpea|garbanzo|teff|millet|quinoa|oat|corn|cassava|arrowroot|bean|"
               r"buckwheat|cashew|sweet rice|brown rice|white rice) flour|rice noodles?|(?:cellophane|glass|bean thread|bean threads) noodles?|bean threads?|glass noodles?|corn ?(?:meal|starch)|"
               r"corn tortillas?|buckwheat|cornflour|(?:pasta|spaghetti|pizza|marinara|tomato|jar(?:red)?|prepared) sauce|rice paper")
_TREENUT = r"\b(?:almonds?|walnuts?|pecans?|cashews?|pistachios?|hazelnuts?|macadamias?|pine nuts?|brazil nuts?|chestnuts?)\b"

_NOT_TREENUT = r"water chestnuts?"

CLASSES = {
    "shellfish": ("$^", _SHELLFISH),
    "pork": (_NOT_PORK, _PORK),
    "gluten": (_NOT_GLUTEN, _GLUTEN),
    "tree_nut": (_NOT_TREENUT, _TREENUT),
    "dairy": (_NOT_DAIRY, _DAIRY),
    "vegetarian": (_NOT_MEAT, _MEAT),     # i.e. "contains meat, poultry, fish or shellfish"
    "egg": (_NOT_EGG, _EGG),
    "peanut": ("$^", _PEANUT),
}


def violations(lines: list[str], cls: str) -> list[str]:
    """Raw lines that, by simple text rules, contain a food of class `cls` (the lines a user with that restriction would object to)."""
    skip, hit = CLASSES[cls]
    bad = []
    for ln in lines:
        low = re.sub(skip, " ", ln.lower())
        if re.search(hit, low):
            bad.append(ln)
    return bad


def words_present(lines: list[str], words: list[str]) -> bool:
    """True if every word appears (as a whole word) somewhere in the recipe's lines."""
    text = " ".join(lines).lower()
    return all(re.search(rf"\b{re.escape(w)}s?\b", text) for w in words)


def words_in_lines(lines: list[str], words: list[str]) -> list[str]:
    """Lines containing ANY of the words as whole words (plural allowed). Used for named foods ("grapes", "mushrooms")."""
    rx = "|".join(rf"\b{re.escape(w)}(?:s|es)?\b" for w in words)
    return [ln for ln in lines if re.search(rx, ln.lower())]
