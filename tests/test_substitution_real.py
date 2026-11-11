"""Task 3 on the real database: honest verdicts, evidence that really exists, and no silent restriction leaks."""
from __future__ import annotations

import pytest

from recipebot.domain.models import SubstitutionRequest
from recipebot.settings import get_settings
from recipebot.store.db import Database
from recipebot.substitution.engine import suggest_substitution

S = get_settings()
pytestmark = pytest.mark.skipif(not S.db_path.exists() or not S.index_dir.exists(),
                                reason="run the ingestion and index build first")


@pytest.fixture(scope="module")
def db():
    d = Database(S.db_path)
    yield d
    d.close()


def one(db, sql):
    rid = db.scalar(sql)
    assert rid, "scenario recipe not found"
    return rid


def test_banana_bread_without_egg_never_claims_more_than_the_evidence_gives(db):
    rid = one(db, """select id from recipes where lower(title)='banana bread' and is_complete
                     and id in (select recipe_id from recipe_flags where "class"='egg') order by id limit 1""")
    r = suggest_substitution(SubstitutionRequest(recipe_id=rid, remove=["egg"]), db)
    t = r.targets[0]
    assert t.verdict in ("weak", "conflicting", "none", "supported")
    if t.verdict != "supported":
        assert all(not l.changed or l.evidence_level != "supported" for l in r.adapted_ingredients)
    for c in t.candidates:                                    # every cited recipe id really exists
        for e in c.evidence:
            assert db.scalar("select count(*) from recipes where id=?", [e.id]) == 1
    assert r.summary


def test_cookie_butter_has_a_supported_replacement_with_real_recipe_ids(db):
    rid = one(db, """select r.id from recipes r where r.is_complete and lower(r.title) like '%chocolate chip cookie%'
                     and exists (select 1 from ingredients_parsed p where p.recipe_id=r.id and p.name='butter') order by r.id limit 1""")
    r = suggest_substitution(SubstitutionRequest(recipe_id=rid, remove=["butter"]), db)
    t = r.targets[0]
    assert t.verdict == "supported" and t.chosen
    assert t.candidates and t.candidates[0].evidence


def test_dairy_free_request_is_never_reported_as_adapted_while_dairy_remains(db):
    rid = one(db, """select r.id from recipes r where r.is_complete and lower(r.title) like '%pasta%' and exists
                     (select 1 from ingredients_parsed p where p.recipe_id=r.id and p.name='heavy cream') order by length(r.title), r.id limit 1""")
    r = suggest_substitution(SubstitutionRequest(recipe_id=rid, exclude_classes=["dairy"]), db)
    dairy = [c for c in r.restriction_check if c.restriction == "dairy"]
    assert dairy
    if r.status == "adapted":
        assert all(c.status == "no_term_found" for c in dairy)
    else:
        assert r.remaining_issues
