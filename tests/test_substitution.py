"""Task 3 (substitution) on a hand-designed database where every outcome is predictable.

Recipe families (see FAMILIES): the evidence the engine can find is exactly what is written here.
  Banana Bread   8 recipes with egg; 4 'Vegan Banana Bread' without egg, 3 of them with applesauce      -> supported variant
  Chocolate Cake 6 with egg; 5 egg-free, each using a DIFFERENT replacement (1 recipe each)             -> weak / scattered
  Sugar Cookies  5 recipes that list 'butter or margarine'                                              -> direct evidence
  Dinner Rolls   5 recipes with 'Beaten egg or milk, for brushing'                                      -> context-free alternative
  Creamy Pasta   4 with cream+parmesan; 4 dairy-free with coconut milk + nutritional yeast              -> class-based adaptation
"""
from __future__ import annotations

import duckdb
import pytest
from conftest import _build
from test_retrieval import FakeEmbedder

from recipebot.domain.models import SubstitutionRequest
from recipebot.retrieval.build import build_index
from recipebot.store.db import Database
from recipebot.substitution import engine, roles
from recipebot.substitution.engine import Evidence, _fit, pick, suggest_substitution, verdict_for
from recipebot.substitution.rewrite import Change, mention_words, rewrite_instructions
from recipebot.tools.registry import TOOLS, call_tool

BANANA_BASE = ["1 1/2 cups all-purpose flour", "1 teaspoon baking soda", "1/2 teaspoon salt", "1 cup white sugar",
               "2 eggs, beaten", "1/4 cup butter, melted", "3 bananas, mashed"]
FLOUR_SUGAR = ["2 cups all-purpose flour", "1 cup sugar", "1 teaspoon baking soda"]


def fam(prefix, title, n, lines, instr="Mix. Bake."):
    return [(f"{prefix}{i}", "ar", title, lines, instr) for i in range(1, n + 1)]


FAMILIES = (
    [("B0", "ar", "Banana Bread", BANANA_BASE,
      "Preheat the oven to 350 degrees F. Mix in slightly beaten eggs, melted butter and mashed bananas. Bake for 60 minutes.")]
    + fam("B", "Banana Bread", 8, FLOUR_SUGAR + ["1 egg", "3 bananas", "1/4 cup butter"])
    + fam("V", "Vegan Banana Bread", 3, FLOUR_SUGAR + ["3 bananas", "1/2 cup applesauce", "1/4 cup vegetable oil"])
    + [("V4", "ar", "Vegan Banana Bread", FLOUR_SUGAR + ["3 bananas", "1/4 cup vegetable oil"], "Mix. Bake.")]
    # chocolate cake: scattered egg-free evidence
    + [("C0", "ar", "Chocolate Cake", ["2 cups all-purpose flour", "1 cup sugar", "1/2 cup cocoa powder", "2 eggs", "1 cup milk",
                                       "1 teaspoon baking soda"], "Preheat the oven. Beat the eggs. Mix all and bake.")]
    + fam("CE", "Chocolate Cake", 5, FLOUR_SUGAR + ["1/2 cup cocoa powder", "2 eggs", "1 cup milk"])
    + [("CF1", "ar", "Chocolate Cake", FLOUR_SUGAR + ["1/2 cup cocoa powder", "1 cup milk", "1/4 cup flaxseed"], "Mix. Bake."),
       ("CF2", "ar", "Chocolate Cake", FLOUR_SUGAR + ["1/2 cup cocoa powder", "1 cup milk", "1/2 cup silken tofu"], "Mix. Bake."),
       ("CF3", "ar", "Chocolate Cake", FLOUR_SUGAR + ["1/2 cup cocoa powder", "1 cup milk", "1/2 cup applesauce"], "Mix. Bake."),
       ("CF4", "ar", "Chocolate Cake", FLOUR_SUGAR + ["1/2 cup cocoa powder", "1 cup milk", "1 tablespoon vinegar"], "Mix. Bake."),
       ("CF5", "ar", "Chocolate Cake", FLOUR_SUGAR + ["1/2 cup cocoa powder", "1 cup milk", "1/2 cup yogurt"], "Mix. Bake.")]
    # cookies: direct 'butter or margarine' evidence, plus 'egg substitute' which must never be offered for egg
    + [("K0", "ar", "Sugar Cookies", ["2 cups all-purpose flour", "1 cup sugar", "1 cup butter, softened", "1 egg",
                                      "1 teaspoon vanilla extract"], "Cream the butter and sugar. Beat in the egg. Bake.")]
    + fam("K", "Sugar Cookies", 5, ["2 cups all-purpose flour", "1 cup sugar", "1 cup butter or margarine", "1 egg or egg substitute"])
    + [("K6", "ar", "Sugar Cookies", ["2 cups all-purpose flour", "1 cup sugar", "1 cup margarine", "1 egg"], "Mix. Bake.")]
    # dinner rolls: 'egg or milk' appears here, in a different kind of dish
    + fam("R", "Dinner Rolls", 5, ["4 cups all-purpose flour", "1 cup warm water", "2 tablespoons sugar", "Beaten egg or milk, for brushing"])
    # creamy pasta: class-based adaptation
    + [("P0", "ar", "Creamy Pasta", ["8 ounces penne", "1 cup heavy cream", "1/2 cup parmesan cheese", "2 cloves garlic",
                                     "1 tablespoon olive oil"], "Boil the penne. Stir in the heavy cream and parmesan cheese. Serve.")]
    + fam("P", "Creamy Pasta", 4, ["8 ounces penne", "1 cup heavy cream", "1/2 cup parmesan cheese", "2 cloves garlic", "1 tablespoon olive oil"])
    + fam("PD", "Creamy Pasta", 4, ["8 ounces penne", "1 cup coconut milk", "1/4 cup nutritional yeast", "2 cloves garlic",
                                    "1 tablespoon olive oil"], "Boil. Stir in the coconut milk and nutritional yeast.")
    # nothing similar and dairy-free to learn from
    + [("Z0", "ar", "Zebra Lentil Soup", ["1 cup heavy cream", "1 cup lentils", "1 teaspoon cumin"], "Simmer the lentils. Stir in the cream.")]
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(engine, "VOCAB_MIN_LINES", 1)             # tiny database: do not hide rare names as debris
    con = duckdb.connect(":memory:")
    _build(con, FAMILIES)
    db = Database(con if False else None, connection=con) if False else Database(connection=con)
    index = build_index(db, tmp_path_factory.mktemp("subidx"), FakeEmbedder(), chunk=8)
    yield db, index
    mp.undo()
    con.close()


def run(world, **kw):
    db, index = world
    kw.setdefault("neighbours", 14)
    return suggest_substitution(SubstitutionRequest(**kw), db, index)


# ------------------------------------------------------------------ supported variant evidence
def test_egg_free_variants_that_share_a_replacement_make_it_supported(world):
    r = run(world, recipe_id="B0", remove=["egg"])
    t = r.targets[0]
    assert t.verdict == "supported" and t.chosen == "applesauce"
    cand = t.candidates[0]
    assert cand.tier == "variant" and cand.n_supporting == 3 and cand.prevalence_without == pytest.approx(0.75)
    assert {e.id for e in cand.evidence if e.kind == "variant_without"} <= {"V1", "V2", "V3"}      # cites real recipes
    assert t.n_similar_without == 4 and t.n_similar_with >= 8
    assert r.status == "adapted"


def test_adapted_list_replaces_the_line_and_instructions_are_edited_and_listed(world):
    r = run(world, recipe_id="B0", remove=["egg"])
    texts = [l.text for l in r.adapted_ingredients]
    assert "2 eggs, beaten" not in texts and any("applesauce" in x for x in texts)
    changed = [l for l in r.adapted_ingredients if l.changed]
    assert len(changed) == 1 and changed[0].original == "2 eggs, beaten" and changed[0].evidence_level == "supported"
    assert "applesauce" in r.revised_instructions and "eggs" not in r.revised_instructions
    assert r.instruction_edits and r.instruction_edits[0].action == "replace" and "eggs" in r.instruction_edits[0].before


def test_effects_are_labelled_as_predictions_not_facts(world):
    r = run(world, recipe_id="B0", remove=["egg"])
    assert r.predicted_effects and all(e.startswith("PREDICTION") for e in r.predicted_effects)
    assert "not verified" in r.predicted_effects[0] and "lift" in r.predicted_effects[0]    # egg's lost role is named
    assert "not certify" in r.disclaimer and "No external web search" in r.evidence_policy


# ------------------------------------------------------------------ weak and conflicting evidence
def test_scattered_replacements_are_leads_only_and_nothing_is_invented(world):
    r = run(world, recipe_id="C0", remove=["egg"])
    t = r.targets[0]
    assert t.verdict in ("weak", "conflicting") and t.chosen is None and t.replacement_line is None
    assert {c.substitute for c in t.candidates} & {"flaxseed", "silken tofu", "applesauce", "yogurt"}   # leads are shown...
    assert all(c.n_supporting + c.n_alternative_similar < 2 for c in t.candidates if c.tier == "indirect")
    assert not any(l.changed and l.text for l in r.adapted_ingredients)                                # ...but no replacement applied
    assert any(l.changed and not l.text and "egg" in (l.original or "") for l in r.adapted_ingredients)   # the removal is SHOWN, not silent
    assert r.status in ("partial", "unsupported") and r.external_search_suggested
    assert any("WITHOUT a supported replacement" in i for i in r.remaining_issues)
    assert len(t.omission_evidence) >= 1                       # recipes that simply have no egg, cited by ID


def test_the_egg_line_is_removed_only_because_the_user_asked_and_the_sentence_is_deleted(world):
    r = run(world, recipe_id="C0", remove=["egg"])
    assert "2 eggs" not in [l.text for l in r.adapted_ingredients]
    assert any(e.action == "delete" and "Beat the eggs" in e.before for e in r.instruction_edits)
    assert "Beat the eggs" not in r.revised_instructions


# ------------------------------------------------------------------ direct evidence and its guards
def test_alternatives_that_similar_recipes_list_themselves_are_direct_evidence(world):
    r = run(world, recipe_id="K0", remove=["butter"])
    t = r.targets[0]
    c = next(c for c in t.candidates if c.substitute == "margarine")
    assert c.tier == "direct" and c.n_alternative_similar >= 3 and t.verdict == "supported"
    assert t.chosen == "margarine" and t.replacement_line == "1 cup margarine"          # amount copied from the original line
    assert any(e.kind == "alternative_listed" and "margarine" in e.line for e in c.evidence)


def test_a_substitute_from_the_same_ingredient_family_is_never_offered(world):
    r = run(world, recipe_id="K0", remove=["egg"])
    offered = {c.substitute for c in r.targets[0].candidates}
    assert not offered & {"egg substitute", "egg white", "egg yolk"}                      # 'egg substitute' is mostly egg
    assert r.targets[0].chosen != "egg substitute"


def test_an_alternative_seen_only_in_other_kinds_of_dish_does_not_count(world):
    r = run(world, recipe_id="B0", remove=["egg"])
    milk = [c for c in r.targets[0].candidates if c.substitute == "milk"]
    assert not milk or (milk[0].tier == "indirect" and milk[0].n_alternative_similar == 0)
    assert r.targets[0].chosen == "applesauce"                                           # not milk from the egg-wash lines


# ------------------------------------------------------------------ class-based adaptation is checked afterwards
def test_dairy_free_adaptation_changes_every_flagged_line_and_verifies_the_result(world):
    r = run(world, recipe_id="P0", exclude_classes=["dairy"])
    assert {t.ingredient for t in r.targets} == {"heavy cream", "parmesan cheese"}
    chosen = {t.ingredient: t.chosen for t in r.targets}
    assert chosen == {"heavy cream": "coconut milk", "parmesan cheese": "nutritional yeast"}
    texts = " | ".join(l.text for l in r.adapted_ingredients)
    assert "heavy cream" not in texts and "parmesan" not in texts and "coconut milk" in texts
    check = r.restriction_check[0]
    assert check.restriction == "dairy" and check.status == "no_term_found"
    assert r.status == "adapted" and not r.remaining_issues
    assert "coconut milk" in r.revised_instructions and "nutritional yeast" in r.revised_instructions


def test_candidates_in_the_excluded_class_are_dropped_and_counted(world):
    r = run(world, recipe_id="P0", exclude_classes=["dairy"])
    cream = next(t for t in r.targets if t.ingredient == "heavy cream")
    assert all("dairy" not in (c.substitute or "") for c in cream.candidates)
    assert "dropped" in cream.verdict_reason or cream.candidates


def test_when_nothing_supports_a_replacement_the_line_stays_and_the_restriction_is_reported(world):
    r = run(world, recipe_id="Z0", exclude_classes=["dairy"], neighbours=8)
    assert r.targets[0].chosen is None
    assert any("cream" in l.text for l in r.adapted_ingredients)                       # the cream line is still there
    assert r.restriction_check[0].status == "still_present"
    assert r.status in ("partial", "unsupported") and any("still contains dairy" in i for i in r.remaining_issues)


# ------------------------------------------------------------------ edges
def test_a_request_that_matches_nothing_changes_nothing(world):
    r = run(world, recipe_id="B0", remove=["tofu"])
    assert r.targets == [] and not any(l.changed for l in r.adapted_ingredients) and "nothing needed to change" in r.summary


def test_unknown_recipe_and_empty_request_are_clear_errors(world):
    with pytest.raises(KeyError):
        run(world, recipe_id="nope", remove=["egg"])
    with pytest.raises(ValueError):
        run(world, recipe_id="B0")


def test_the_computation_record_states_the_method_and_the_evidence_scope(world):
    c = run(world, recipe_id="B0", remove=["egg"]).computation
    assert c.tool == "suggest_substitution" and "No language model is involved" in c.method and c.funnel[0].remaining > 0
    assert any("same dish name" in f.label for f in c.funnel)


# ------------------------------------------------------------------ the rules, tested directly
def cand(tier="variant", n=3, p_wo=0.6, p_w=0.1, alt_sim=0, covered=("binder", "moisture"), missing=("leavening",)):
    from recipebot.domain.models import SubstituteCandidate
    return SubstituteCandidate(substitute="x", tier=tier, n_supporting=n, prevalence_without=p_wo, prevalence_with=p_w,
                               n_alternative_lines=alt_sim, n_alternative_similar=alt_sim, roles_covered=list(covered),
                               roles_missing=list(missing))


def ev(cands, n_without=10, scope="closest"):
    return Evidence(n_without, 20, scope, cands, [])


def test_verdict_supported_needs_a_clear_lift_and_a_good_role_fit():
    assert verdict_for(ev([cand()]))[0] == "supported"
    assert verdict_for(ev([cand(p_wo=0.9, p_w=0.85)]))[0] == "weak"                       # as common where the ingredient is kept
    assert verdict_for(ev([cand(covered=("moisture",), missing=("binder", "leavening", "richness", "structure"))]))[0] == "weak"


def test_verdict_conflicting_when_many_different_things_are_used_and_none_dominates():
    assert verdict_for(ev([cand(n=1, p_wo=0.1, p_w=0.0, tier="indirect")], n_without=12))[0] == "conflicting"
    assert verdict_for(ev([]))[0] == "none"


def test_pick_applies_only_what_the_evidence_allows():
    assert pick(ev([cand()]), "supported").substitute == "x"
    assert pick(ev([cand(n=1, tier="indirect")]), "weak") is None                          # one recipe is not enough
    assert pick(ev([cand(tier="indirect", n=2, p_wo=0.3, p_w=0.0)]), "weak") is not None   # two recipes and a role fit
    assert pick(ev([cand(tier="indirect", n=2, p_wo=0.1, p_w=0.3)]), "weak") is None        # more common where it is kept
    assert pick(ev([cand()]), "conflicting") is None and pick(ev([cand()]), "none") is None


def test_role_fit_is_the_share_of_the_original_jobs_covered():
    assert _fit(cand(covered=("a", "b"), missing=("c", "d"))) == 0.5
    assert _fit(cand(covered=(), missing=())) is None
    assert roles.compare([], ["fat"])[2] and roles.compare(["fat"], [])[2]               # unknown roles give a note, not a verdict


def test_rewriter_changes_only_sentences_that_mention_the_ingredient():
    text = "Preheat the oven. Beat the eggs. Mix the flour with the eggs and the bananas, then bake for an hour."
    new, edits = rewrite_instructions(text, [Change(mention_words("egg"), "applesauce")])
    assert "Preheat the oven." in new and new.count("applesauce") == 2 and len(edits) == 2
    new, edits = rewrite_instructions(text, [Change(mention_words("egg"), None)])
    assert [e.action for e in edits] == ["delete", "review"] and "Beat the eggs" not in new


def test_registry_exposes_and_validates_the_tool():
    assert "suggest_substitution" in TOOLS
    with pytest.raises(Exception):
        call_tool("suggest_substitution", {"recipe_id": "x", "restriction_mode": "relaxed"})
