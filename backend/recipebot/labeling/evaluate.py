"""Score a labeling method against a small hand-made gold set.

The numbers that matter for safety are recall: of the (ingredient, restriction) pairs the gold set says apply, how many
did the method flag at all (yes or maybe)? A missed definite allergen is the dangerous error; a false alarm only
hides a recipe. So the report keeps them apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

RANK = {None: 0, "none": 0, "maybe": 1, "yes": 2}
BASE_CLASSES = ("dairy", "egg", "peanut", "tree_nut", "gluten", "soy", "fish", "shellfish", "sesame", "pork", "beef",
                "poultry", "lamb", "game", "alcohol")


@dataclass
class Pred:
    is_ingredient: bool
    parent: str
    classes: dict[str, str]


def load_gold(path: Path) -> dict[str, dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    def level(v):   # YAML 1.1 reads the bare word `yes` as the boolean True: map it back
        return "yes" if v is True else ("maybe" if v == "maybe" else None if v in (False, None, "none") else str(v))

    return {n: {"is_ingredient": v.get("is_ingredient", True), "parent": v.get("parent", n),
                "classes": {c: level(l) for c, l in (v.get("classes") or {}).items() if level(l)},
                "group": v.get("group", "")} for n, v in doc["names"].items()}


def score(preds: dict[str, Pred], gold: dict[str, dict]) -> dict:
    names = [n for n in gold if n in preds]
    gold_flag = gold_yes = tp = yes_caught = yes_exact = pred_flag = false_alarm = 0
    missed_yes: list[tuple] = []
    missed_any: list[tuple] = []
    alarms: list[tuple] = []
    parent_ok = parent_n = ing_ok = 0
    for n in names:
        g, p = gold[n], preds[n]
        ing_ok += g["is_ingredient"] == p.is_ingredient
        if g["is_ingredient"]:
            parent_n += 1
            parent_ok += g["parent"] == p.parent
        for c in BASE_CLASSES:
            gl, pl = g["classes"].get(c), (p.classes.get(c) if p.is_ingredient or not g["is_ingredient"] else None)
            if not g["is_ingredient"]:
                pl = p.classes.get(c)
            if RANK[gl]:
                gold_flag += 1
                if RANK[pl]:
                    tp += 1
                else:
                    missed_any.append((n, c, gl))
            if gl == "yes":
                gold_yes += 1
                yes_caught += bool(RANK[pl])
                yes_exact += pl == "yes"
                if not RANK[pl]:
                    missed_yes.append((n, c))
            if RANK[pl]:
                pred_flag += 1
                if not RANK[gl]:
                    false_alarm += 1
                    alarms.append((n, c, pl))
    return {
        "names": len(names),
        "recall_any_flag": round(tp / gold_flag, 3) if gold_flag else None,
        "recall_definite_flagged": round(yes_caught / gold_yes, 3) if gold_yes else None,
        "recall_definite_exact": round(yes_exact / gold_yes, 3) if gold_yes else None,
        "precision_of_flags": round(tp / pred_flag, 3) if pred_flag else None,
        "false_alarm_pairs": false_alarm,
        "parent_accuracy": round(parent_ok / parent_n, 3) if parent_n else None,
        "is_ingredient_accuracy": round(ing_ok / len(names), 3) if names else None,
        "gold_flagged_pairs": gold_flag, "gold_definite_pairs": gold_yes,
        "missed_definite": missed_yes, "missed_any": missed_any, "false_alarms": alarms,
    }
