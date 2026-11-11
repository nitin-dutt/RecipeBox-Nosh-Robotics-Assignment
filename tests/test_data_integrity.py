"""Integration invariants over the built database. Skipped if the DB has not been built (run the ingestion first)."""
from __future__ import annotations

import duckdb
import pytest

from recipebot.settings import get_settings

DB = get_settings().db_path
pytestmark = pytest.mark.skipif(not DB.exists(), reason="run `uv run python -m recipebot.ingestion` first")


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(DB), read_only=True)
    yield c
    c.close()


def one(con, sql):
    return con.execute(sql).fetchone()[0]


def test_record_counts_reconcile(con):
    assert one(con, "select count(*) from recipes") == one(con, "select count(distinct id) from recipes") == 124647


def test_no_advertisement_left(con):
    assert one(con, "select count(*) from recipe_ingredients where text ilike '%advertisement%'") == 0


def test_raw_text_is_preserved(con):
    # every cleaned ingredient line keeps its raw source line
    assert one(con, "select count(*) from recipe_ingredients where text_raw is null or text_raw = ''") == 0
    assert one(con, "select count(*) from recipes where title_raw is null and has_title") == 0


def test_no_orphans(con):
    assert one(con, "select count(*) from ingredients_parsed p left join recipes r on r.id=p.recipe_id where r.id is null") == 0
    assert one(con, "select count(*) from ingredient_flags f left join recipes r on r.id=f.recipe_id where r.id is null") == 0


def test_every_ingredient_line_was_parsed(con):
    lines = one(con, "select count(*) from recipe_ingredients")
    parsed_lines = one(con, "select count(distinct (recipe_id, position)) from ingredients_parsed")
    assert parsed_lines == lines


def test_allergen_recall_peanut_sesame_tree_nut(con):
    """A line that literally mentions these allergens must always carry a flag (yes or maybe)."""
    for cls, rx in {"peanut": "peanut", "sesame": "sesame", "tree_nut": "almond|walnut|pecan|cashew|pistachio|hazelnut"}.items():
        missing = one(con, f"""select count(*) from recipe_ingredients i
            where regexp_matches(lower(i.text), '{rx}') and lower(i.text) not like '%free%'
            and not exists (select 1 from ingredient_flags f where f.recipe_id=i.recipe_id
                            and f.position=i.position and f."class"='{cls}')""")
        assert missing == 0, f"{cls}: {missing} lines mention the allergen but carry no flag"


def test_known_names(con):
    assert one(con, "select n_recipes from ingredient_vocab where name='penne'") > 300
    assert one(con, "select count(*) from ingredient_vocab where name='sweet potato' and parent='sweet potato'") == 1
    assert one(con, "select count(*) from ingredient_vocab where name='green onion' and parent='green onion'") == 1
