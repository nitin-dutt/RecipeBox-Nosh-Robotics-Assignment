"""Test fixtures. `mini_db` builds a tiny in-memory database through the REAL cleaning, parsing and taxonomy code,
so the tool tests exercise the same pipeline as production but with answers small enough to verify by hand."""
from __future__ import annotations

import duckdb
import pytest

from recipebot.domain.ingredient import normalize_for_scan, parse_ingredient_line
from recipebot.domain.instruction_scan import scan_instructions
from recipebot.domain.taxonomy import get_taxonomy
from recipebot.ingestion.cleaning import clean_record
from recipebot.store.db import Database, set_db

# id, source, title, ingredients, instructions
MINI = [
    ("R1", "ar", "Penne with Tomato Sauce", ["1 pound penne", "2 cups tomato sauce", "3 cloves garlic, minced",
     "2 tablespoons olive oil", "1/2 cup grated parmesan cheese"], "Boil the penne.\nMake the sauce. Combine and serve."),
    ("R2", "ar", "Dairy Free Penne", ["1 pound penne", "4 tomatoes, chopped", "3 cloves garlic, minced",
     "2 tablespoons olive oil"], "Boil the penne. Toss with tomatoes and garlic. Serve."),
    ("R3", "fn", "Penne Alfredo", ["1 pound penne", "1 cup heavy cream", "4 tablespoons butter",
     "1 cup parmesan cheese"], "Boil. Simmer cream and butter. Toss."),
    ("R4", "epi", "Garlic Bread", ["1 loaf bread", "2 teaspoons garlic powder", "1/2 cup butter", "1 onion, diced"],
     "Mix. Spread. Bake."),
    ("R5", "ar", "Tomato Soup", ["4 tomatoes", "1 onion, chopped", "2 cloves garlic", "2 cups vegetable broth"],
     "Simmer everything. Blend. Serve hot."),
    ("R6", "fn", "Peanut Noodles", ["8 ounces noodles", "1/2 cup peanut butter", "2 tablespoons soy sauce"],
     "Cook noodles. Stir sauce. Combine."),
    ("R7", "fn", "Mystery Dish", [], ""),
    ("R8", "epi", "Dairy Free Penne", ["1 pound penne", "4 tomatoes, chopped", "3 cloves garlic, minced",
     "2 tablespoons olive oil"], "Boil the penne. Toss with tomatoes and garlic. Serve."),
    ("R9", "ar", "Margarine Cookies", ["2 cups flour", "1 cup sugar", "1 cup margarine", "1 egg"], "Mix. Bake."),
    ("R10", "ar", "Tomato Basil Smoothie", ["2 tomatoes", "1/4 cup fresh basil leaves", "1 cup ice"], "Blend."),
    ("R11", "fn", "Fried Rice", ["2 cups cooked rice", "2 tablespoons canola, safflower or peanut oil", "1 egg"],
     "Heat the oil. Fry the rice. Add egg."),
]


def _build(con: duckdb.DuckDBPyConnection, data=None) -> None:
    data = data or MINI
    tax = get_taxonomy()
    con.execute("""CREATE TABLE recipes(id VARCHAR, source VARCHAR, title VARCHAR, title_raw VARCHAR,
        instructions VARCHAR, instructions_raw VARCHAR, n_ingredients INTEGER, n_steps_paragraphs INTEGER,
        n_steps_sentences INTEGER, picture_link VARCHAR, has_title BOOLEAN, has_ingredients BOOLEAN,
        has_instructions BOOLEAN, is_complete BOOLEAN, dup_group_id VARCHAR, dup_group_size INTEGER)""")
    con.execute("CREATE TABLE recipe_ingredients(recipe_id VARCHAR, position INTEGER, text VARCHAR, text_raw VARCHAR, group_label VARCHAR)")
    con.execute("""CREATE TABLE ingredients_parsed(recipe_id VARCHAR, position INTEGER, item_idx INTEGER, status VARCHAR,
        name VARCHAR, base VARCHAR, form VARCHAR, parent VARCHAR, qty_min DOUBLE, qty_max DOUBLE, unit VARCHAR,
        unit_kind VARCHAR, qty_base_min DOUBLE, qty_base_max DOUBLE, optional BOOLEAN, alternatives VARCHAR[])""")
    con.execute("""CREATE TABLE ingredient_flags(recipe_id VARCHAR, position INTEGER, item_idx INTEGER, "class" VARCHAR,
        status VARCHAR, via VARCHAR, name VARCHAR, optional BOOLEAN)""")
    cleaned, keys = [], {}
    for rid, src, title, ings, ins in data:
        rec = clean_record(rid, src, {"title": title, "ingredients": ings, "instructions": ins, "picture_link": None},
                           lambda _r: None)
        cleaned.append(rec)
        if rec.has_title and rec.has_ingredients:
            keys.setdefault((rec.title.lower(), tuple(i.lower() for i in rec.ingredients)), []).append(rid)
    dups = {rid: (f"g{n}", len(ids)) for n, ids in enumerate(k for k in keys.values() if len(k) > 1) for rid in ids}
    con.execute("CREATE TABLE instruction_flags(recipe_id VARCHAR, \"class\" VARCHAR, status VARCHAR, snippet VARCHAR)")
    flagged: dict[str, set[str]] = {}
    for rec in cleaned:
        gid, gsize = dups.get(rec.id, (None, 1))
        con.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [rec.id, rec.source, rec.title, rec.title_raw, rec.instructions, rec.instructions_raw,
                     len(rec.ingredients), rec.n_steps_paragraphs, rec.n_steps_sentences, None, rec.has_title,
                     rec.has_ingredients, rec.has_instructions, rec.is_complete, gid, gsize])
        for pos, (text, raw) in enumerate(zip(rec.ingredients, rec.ingredient_raws, strict=True), start=1):
            con.execute("INSERT INTO recipe_ingredients VALUES (?,?,?,?,?)", [rec.id, pos, text, raw, None])
            items = parse_ingredient_line(text)
            line_flags = tax.classify_name(normalize_for_scan(text))
            seen: set[str] = set()
            for idx, p in enumerate(items):
                con.execute("INSERT INTO ingredients_parsed VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            [rec.id, pos, idx, p.status, p.name or None, p.base or None, p.form, p.parent or None,
                             p.qty_min, p.qty_max, p.unit, p.unit_kind, p.qty_base_min, p.qty_base_max, p.optional,
                             p.alternatives])
                if p.status != "ok":
                    continue
                for cls, st in tax.classify_item(p.name, p.alternatives).items():
                    seen.add(cls)
                    flagged.setdefault(rec.id, set()).add(cls)
                    con.execute("INSERT INTO ingredient_flags VALUES (?,?,?,?,?,?,?,?)",
                                [rec.id, pos, idx, cls, st, "name", p.name, False])
            for cls in line_flags:
                flagged.setdefault(rec.id, set()).add(cls)
                if cls not in seen:
                    con.execute("INSERT INTO ingredient_flags VALUES (?,?,?,?,?,?,?,?)",
                                [rec.id, pos, 0, cls, "maybe", "text", None, False])
    for rec in cleaned:
        for cls, snippet in scan_instructions(rec.instructions, tax):
            if cls not in flagged.get(rec.id, set()):
                con.execute("INSERT INTO instruction_flags VALUES (?,?,?,?)", [rec.id, cls, "maybe", snippet])
    con.execute("""CREATE TABLE recipe_flags AS SELECT recipe_id, "class",
        CASE WHEN bool_or(status='yes') THEN 'yes' ELSE 'maybe' END AS status, count(*) AS n_lines
        FROM (SELECT recipe_id, "class", status FROM ingredient_flags
              UNION ALL SELECT recipe_id, "class", status FROM instruction_flags)
        GROUP BY recipe_id, "class" """)
    con.execute("""CREATE TABLE ingredient_vocab AS SELECT name, any_value(base) AS base, any_value(form) AS form,
        any_value(parent) AS parent, count(*) AS n_lines, count(DISTINCT recipe_id) AS n_recipes
        FROM ingredients_parsed WHERE status='ok' AND name IS NOT NULL GROUP BY name""")


# Task 2 fixtures: the user has [eggs, onions, tomatoes, rice]
MATCH = [
    ("M1", "ar", "Egg Fried Rice", ["2 eggs", "2 cups cooked rice", "1 onion, chopped", "2 tablespoons soy sauce",
     "1 teaspoon salt"], "Cook rice. Scramble eggs. Fry onion. Combine."),
    ("M2", "ar", "Tomato Egg Scramble", ["3 eggs", "2 tomatoes", "1 tablespoon butter", "salt to taste"],
     "Beat eggs. Fry tomatoes. Add eggs."),
    ("M3", "ar", "Spanish Rice", ["1 cup rice", "1 onion", "2 tomatoes", "1 green bell pepper", "2 cloves garlic",
     "2 cups chicken broth"], "One. Two. Three. Four. Five. Six. Seven."),
    ("M4", "ar", "Peanut Rice", ["1 cup rice", "1 onion", "1 tomato", "1/2 cup peanuts", "1 egg"], "Cook. Mix. Serve."),
    ("M5", "ar", "Onion Tomato Salad", ["2 onions", "3 tomatoes", "2 tablespoons olive oil", "1 tablespoon vinegar"],
     "Slice. Toss."),
    ("M6", "ar", "Egg Drop Soup", ["2 eggs", "4 cups chicken broth", "2 green onions", "1 tablespoon cornstarch"],
     "Boil broth. Drizzle eggs. Garnish."),
    ("M7", "ar", "Plain Rice", ["1 cup rice", "2 cups water", "1 teaspoon salt"], "Boil."),
    ("M8", "ar", "Red Onion Tomato Egg Rice", ["2 eggs", "1 red onion", "1 tomato", "1/2 cup rice"],
     "Cook rice. Fry onion and tomato. Add egg. Mix. Serve."),
    ("M9", "ar", "Big Stew", ["2 eggs", "2 onions", "3 tomatoes", "1 cup rice", "1 pound beef", "2 carrots",
     "2 potatoes", "1 cup peas", "2 cups beef broth", "1 bay leaf", "1 tablespoon paprika", "1 cup red wine"],
     "One. Two. Three. Four. Five. Six."),
    ("M10", "ar", "Garlic Rice", ["1 cup rice", "1 teaspoon garlic powder", "1 onion, minced"], "Fry. Add rice. Serve."),
]


@pytest.fixture(scope="session")
def match_db():
    con = duckdb.connect(":memory:")
    _build(con, MATCH)
    db = Database(connection=con)
    yield db
    con.close()


@pytest.fixture(scope="session")
def mini_db():
    con = duckdb.connect(":memory:")
    _build(con)
    db = Database(connection=con)
    set_db(db)
    yield db
    set_db(None)
    con.close()
