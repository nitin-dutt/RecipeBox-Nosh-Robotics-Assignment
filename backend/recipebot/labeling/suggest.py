"""Pre-filled recommendations for the human review queue.

These are SUGGESTIONS written down with their criteria, so a reviewer can accept them in one go or overrule any of them.
Criteria (the same ones that produced the gold set):
  raise "maybe" on a common ingredient
     accept   the allergen is a declared ingredient of typical commercial versions (milk solids in chocolate, wheat in
              taco seasoning, parmesan in pesto, soy in cake mixes)
     reject   only a processing aid or a rare variant would make it true, or accepting it would make a strict filter
              useless for an everyday ingredient: highly refined oils and sprays (soy), dried pasta (egg), plain or crusty
              bread (soy, egg), distilled spirits (gluten), worcestershire (alcohol = trace from vinegar)
  LLM says LESS than the rules
     reject   by default (the rules stay), because a wrong "does not contain" is the dangerous error
     accept   only when the rule is a plain false positive (spaghetti squash is not wheat; cream-style corn has no cream)
  LLM says "not an ingredient"
     accept   when the text is equipment, packaging, a markup fragment or a separator
     reject   anything that could still name a food
"""
from __future__ import annotations

import re

_OIL_LIKE = re.compile(r"\b(oil|spray|shortening|whipped topping|margarine spread)\b")
_DRIED_PASTA_OR_BREAD = re.compile(r"\b(spaghetti|spaghettini|penne|macaroni|linguine|linguini|fettuccine|rigatoni|ziti|"
                                   r"orzo|pasta|noodle|bread|bun|roll|crumb)\b")
_SPIRITS = re.compile(r"\b(sake|whiskey|whisky|bourbon|vodka|rum|gin|tequila|brandy|rye)\b")
_RULE_FALSE_POSITIVES = {("spaghetti squash", "gluten"), ("cream style corn", "dairy"), ("creamed corn", "dairy"),
                         ("sherry wine vinegar", "alcohol"), ("marshmallow cream", "dairy"), ("marshmallow creme", "dairy")}
_NON_FOOD = re.compile(
    r"\b(skewer|toothpick|parchment|foil|jar|lid|thermometer|mill|cheesecloth|twine|pan|weight|plank|wood chip|"
    r"lollipop stick|canning|available|preheat|oven|after first day|fermentation|wrap|paper towel|string)\b")
_SEPARATORS = {"em", "/em", "strong", "/strong", "or", "and", "red", "white", "yellow", "green", "black", "brown", "dark",
               "light", "soft", "dry", "dried", "strong", "round", "blanched", "cleaned", "washed", "unbleached", "fully",
               "roasted", "firm", "piece", "topping", "chip", "dog"}


def suggest(row: dict) -> str:
    kind, name, cls = row["kind"], row["name"], row.get("class", "")
    if kind == "raise_maybe":
        if cls == "soy" and _OIL_LIKE.search(name):
            return "reject"
        if cls == "soy" and re.search(r"\bbread\b", name):
            return "reject"
        if cls == "egg" and _DRIED_PASTA_OR_BREAD.search(name):
            return "reject"
        if cls == "gluten" and _SPIRITS.search(name):
            return "reject"
        if cls == "alcohol" and "worcestershire" in name:
            return "reject"
        if cls == "dairy" and re.search(r"graham|crusty", name):
            return "reject"
        return "accept"
    if kind == "lower_class":
        return "accept" if (name, cls) in _RULE_FALSE_POSITIVES else "reject"
    if kind == "not_ingredient":
        return "accept" if (_NON_FOOD.search(name) or name in _SEPARATORS) else "reject"
    return ""
