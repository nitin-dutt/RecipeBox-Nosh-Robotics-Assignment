"""Retrieval quality probe.  uv run python eval/retrieval_probe.py

Compares keyword (BM25), semantic (embeddings) and hybrid (RRF) retrieval where the right answer is computable
from the structured data, and measures how often similarity ALONE returns recipes that break a restriction.
Writes eval/retrieval_probe_results.json (kept in the repo as evidence for the evaluation report).
"""
from __future__ import annotations

import json
from pathlib import Path

from recipebot.domain.models import Filters, SearchSpec
from recipebot.settings import get_settings
from recipebot.store.db import Database
from recipebot.tools.filters import FilterPlan
from recipebot.tools.search import search_recipes
from recipebot.tools.semantic import _eligible_ids, get_index
from recipebot.tools.terms import TermResolver

K = 10
INGREDIENT_QUERIES = ["chickpeas", "tofu", "salmon", "pumpkin", "lentils", "coconut milk", "zucchini", "eggplant",
                      "quinoa", "shrimp", "spinach", "mushrooms"]
TITLE_QUERIES = ["banana bread", "chicken noodle soup", "apple pie", "lasagna", "pad thai", "chocolate chip cookies",
                 "caesar salad", "beef stew"]
DESCRIPTIVE = ["comforting winter soup", "light summer salad", "spicy vegetarian curry", "quick weeknight pasta",
               "something sweet with chocolate and nuts", "hearty breakfast", "cozy chickpea stew"]
RESTRICTED = [  # (query a person would type, restriction class they mean)
    ("pasta without dairy", "dairy"), ("dairy free creamy soup", "dairy"), ("vegan dessert", "dairy"),
    ("egg free cake", "egg"), ("peanut free cookies", "peanut"), ("gluten free bread", "gluten"),
    ("nut free brownies", "tree_nut"), ("shellfish free seafood stew", "shellfish"),
]


def main() -> None:
    cfg = get_settings()
    db = Database(cfg.db_path)
    index = get_index(db)
    resolver = TermResolver(db)
    # measure against the same population the tools use by default: complete recipes only
    complete, *_ = _eligible_ids(FilterPlan(Filters(), db, resolver), db)
    out: dict = {"k": K, "ingredient_queries": [], "title_queries": [], "descriptive": [], "restriction_leakage": []}

    # 1) "recipes with X": precision@K = share of the top K that really contain X (ground truth from structured data)
    for q in INGREDIENT_QUERIES:
        ids, *_ = _eligible_ids(FilterPlan(Filters(include_ingredients=[q]), db, resolver), db)
        gold = set(ids)
        row = {"query": q, "recipes_that_contain_it": len(gold)}
        for mode in ("keyword", "semantic", "hybrid"):
            hits, _ = index.search(q, K, allowed_ids=complete, mode=mode)
            row[mode] = round(sum(h.id in gold for h in hits) / max(len(hits), 1), 2)
        out["ingredient_queries"].append(row)

    # 2) dish titles: share of the top K whose title contains every query word
    for q in TITLE_QUERIES:
        words = q.split()
        row = {"query": q}
        for mode in ("keyword", "semantic", "hybrid"):
            hits, _ = index.search(q, K, allowed_ids=complete, mode=mode)
            titles = {r[0]: (r[1] or "").lower() for r in db.query(
                f"SELECT id, title FROM recipes WHERE id IN ({', '.join('?' for _ in hits)})", [h.id for h in hits])}
            row[mode] = round(sum(all(w in titles[h.id] for w in words) for h in hits) / max(len(hits), 1), 2)
        out["title_queries"].append(row)

    # 3) descriptive requests: no ground truth, so the evidence is the list itself
    for q in DESCRIPTIVE:
        hits, _ = index.search(q, 5, allowed_ids=complete)
        titles = {r[0]: r[1] for r in db.query(
            f"SELECT id, title FROM recipes WHERE id IN ({', '.join('?' for _ in hits)})", [h.id for h in hits])}
        out["descriptive"].append({"query": q, "top5": [titles[h.id] for h in hits]})

    # 4) restriction leakage: similarity alone vs filter-first
    for q, cls in RESTRICTED:
        flagged_yes = {r[0] for r in db.query('SELECT recipe_id FROM recipe_flags WHERE "class" = ? AND status = \'yes\'', [cls])}
        flagged_any = {r[0] for r in db.query('SELECT recipe_id FROM recipe_flags WHERE "class" = ?', [cls])}
        row = {"query": q, "restriction": cls}
        for mode in ("semantic", "hybrid"):
            hits, _ = index.search(q, K, allowed_ids=complete, mode=mode)
            row[f"{mode}_definite_violations_in_top{K}"] = sum(h.id in flagged_yes for h in hits)
            row[f"{mode}_any_flag_in_top{K}"] = sum(h.id in flagged_any for h in hits)
        spec = SearchSpec(query=q, mode="hybrid", exclude_classes=[cls], limit=K)
        res = search_recipes(spec, db)
        row["filter_first_violations_in_top10"] = sum(h.id in flagged_any for h in res.recipes)
        row["filter_first_returned"] = len(res.recipes)
        out["restriction_leakage"].append(row)

    def mean(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 3)

    out["summary"] = {
        "ingredient_precision_at_10": {m: mean(out["ingredient_queries"], m) for m in ("keyword", "semantic", "hybrid")},
        "title_precision_at_10": {m: mean(out["title_queries"], m) for m in ("keyword", "semantic", "hybrid")},
        "leakage_definite_violations_total_in_80_results": {
            "semantic": sum(r[f"semantic_definite_violations_in_top{K}"] for r in out["restriction_leakage"]),
            "hybrid": sum(r[f"hybrid_definite_violations_in_top{K}"] for r in out["restriction_leakage"]),
            "filter_first": sum(r["filter_first_violations_in_top10"] for r in out["restriction_leakage"])},
        "index": index.meta,
    }
    path = Path(__file__).with_name("retrieval_probe_results.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
