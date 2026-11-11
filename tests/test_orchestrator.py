"""Phase 7: session state, planning guards, grounding validation and the SSE API, with a scripted fake LLM.

No test here calls Gemini: the fake returns the plan a good model would return, so we test OUR code's behaviour
(state diffs, guards, refusals, fallback, validation), which is what must be deterministic.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from recipebot.api.app import create_app
from recipebot.llm.gemini import LLMUnavailable
from recipebot.orchestrator import ground
from recipebot.orchestrator.bot import Chatbot, detect_restrictions, fallback_plan
from recipebot.orchestrator.plan import TurnPlan


class FakeLLM:
    """Plays back plans (for the TurnPlan schema) and narrations (for the Narration schema)."""

    def __init__(self, plans=(), narrations=(), fail=False):
        self.plans, self.narrations, self.fail = list(plans), list(narrations), fail
        self.calls = []

    def generate_json(self, system, user, schema, max_output_tokens=0, accept=None):
        self.calls.append(schema.__name__)
        if self.fail:
            raise LLMUnavailable("down")
        if schema.__name__ == "TurnPlan":
            p = self.plans.pop(0)
            return (p if isinstance(p, dict) else p.model_dump()), False
        n = self.narrations.pop(0) if self.narrations else None
        if n is None:
            raise LLMUnavailable("no narration scripted")
        return {"intro": n}, False

    def stats(self):
        return {"api_calls": len(self.calls)}


def bot(llm, narrate=False):
    return Chatbot(llm, narrate=narrate)


# ------------------------------------------------------------------------------------------------ session state
def test_followup_is_a_diff_on_the_session_filters(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne"),
                     TurnPlan(intent="search", add_exclude_classes=["dairy"])]))
    r1 = b.turn("s", "show me penne recipes")
    assert {s.recipe_id for s in r1.sources} >= {"R1", "R3"}
    r2 = b.turn("s", "I can't have dairy")
    ids = {s.recipe_id for s in r2.sources}
    assert ids and ids.isdisjoint({"R1", "R3"})                  # parmesan / cream / butter recipes are gone
    assert r2.active_filters["query"] == "penne" and r2.active_filters["exclude_classes"] == ["dairy"]


def test_new_scope_forgets_old_filters(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne", add_exclude_classes=["dairy"]),
                     TurnPlan(intent="search", scope="new", query="soup")]))
    b.turn("s", "dairy free penne")
    r = b.turn("s", "now soup")
    assert r.active_filters.get("exclude_classes") is None and r.active_filters["query"] == "soup"


def test_sessions_are_isolated(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne", add_exclude_classes=["dairy"]),
                     TurnPlan(intent="search", scope="new", query="penne")]))
    b.turn("a", "no dairy penne")
    r = b.turn("b", "penne")
    assert "exclude_classes" not in r.active_filters


def test_ordinal_reference_resolves_against_the_last_list(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne"), TurnPlan(intent="get_recipe", ref_index=2)]))
    r1 = b.turn("s", "penne")
    r2 = b.turn("s", "show me the second one")
    assert r2.sources[0].recipe_id == r1.sources[1].recipe_id
    assert "Ingredients:" in r2.answer


def test_ambiguous_reference_asks_instead_of_guessing(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne"), TurnPlan(intent="get_recipe")]))
    b.turn("s", "penne")
    r = b.turn("s", "show me that recipe")
    assert r.status == "needs_input" and "number" in r.answer.lower()


def test_unknown_result_number_is_reported(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne"), TurnPlan(intent="get_recipe", ref_index=9)]))
    b.turn("s", "penne")
    assert "no result number 9" in b.turn("s", "the ninth").answer


# ------------------------------------------------------------------------------------------------ Task routing
def test_count_answer_comes_from_the_tool(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="count", scope="new", add_include=["garlic", "onion"])]))
    r = b.turn("s", "how many recipes have garlic and onion?")
    assert r.computation and r.computation["denominator"] and r.computation["sql"]
    assert str(r.computation["n_analyzed"]) in r.answer or "of" in r.answer


def test_match_followup_reuses_the_ingredients_and_adds_a_constraint(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="match", scope="new", available_text="eggs, tomatoes, rice"),
                     TurnPlan(intent="match", add_exclude=["peanut"])]))
    b.turn("s", "I have eggs, tomatoes and rice")
    r = b.turn("s", "no peanuts")
    assert r.active_filters["available"] == "eggs, tomatoes, rice"
    assert "peanut" in r.active_filters["exclude_ingredients"] + r.active_filters["exclude_classes"]


def _stub_substitution(monkeypatch):
    """The engine has its own tests; here we only check that the orchestrator calls it with the right arguments."""
    from dataclasses import replace
    from recipebot.domain.models import Computation, SubstitutionResult
    from recipebot.orchestrator import execute as ex
    seen = {}

    def handler(req):
        seen["req"] = req
        comp = Computation(tool="suggest_substitution", description="d", population="p", filters={}, funnel=[],
                           n_analyzed=1, method="m", sql="s")
        return SubstitutionResult(status="partial", summary="stub summary", recipe_id=req.recipe_id, recipe_title="T",
                                  targets=[], adapted_ingredients=[], revised_instructions="", instruction_edits=[],
                                  predicted_effects=[], restriction_check=[], remaining_issues=["still check"],
                                  external_search_suggested=False, evidence_policy="e", disclaimer="Text check only.",
                                  computation=comp)

    monkeypatch.setitem(ex.TOOLS, "suggest_substitution", replace(ex.TOOLS["suggest_substitution"], handler=handler))
    return seen


def test_substitution_runs_for_a_listed_recipe_with_the_right_request(mini_db, monkeypatch):
    seen = _stub_substitution(monkeypatch)
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne alfredo"),
                     TurnPlan(intent="substitute", ref_index=1, add_exclude_classes=["dairy"])]))
    r1 = b.turn("s", "penne alfredo")
    r = b.turn("s", "make the first one dairy-free")
    assert r.intent == "substitute" and seen["req"].recipe_id == r1.sources[0].recipe_id
    assert seen["req"].exclude_classes == ["dairy"] and "stub summary" in r.answer
    assert b.sessions.get("s").focus_id == seen["req"].recipe_id


def test_substitution_without_a_target_asks_what_to_change(mini_db, monkeypatch):
    _stub_substitution(monkeypatch)
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne alfredo"), TurnPlan(intent="substitute", ref_index=1)]))
    b.turn("s", "penne alfredo")
    assert b.turn("s", "change it").status == "needs_input"


def test_unsupported_attribute_is_refused_without_calling_a_search_tool(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="unsupported", unsupported_text="calories")]))
    r = b.turn("s", "how many calories in the penne?")
    assert r.status == "refused" and r.sources == [] and "no nutrition" in r.answer


def test_mixed_request_answers_the_supported_part_and_names_the_unsupported_part(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne")]))
    r = b.turn("s", "penne recipes and how long they take to cook")
    assert r.sources and "Not answered" in r.answer


# ------------------------------------------------------------------------------------------------ guards
@pytest.mark.parametrize("msg,expected", [
    ("I can't have dairy", ["dairy"]), ("gluten-free pasta", ["gluten"]), ("dairy free please", ["dairy"]),
    ("I'm allergic to peanuts and eggs", ["egg", "peanut"]), ("no nuts", ["tree_nut"]),
    ("I love peanuts", []), ("lots of cheese please", []), ("without butter", []),
])
def test_detect_restrictions(msg, expected):
    assert sorted(detect_restrictions(msg)) == sorted(expected)


def test_guard_adds_a_restriction_the_llm_forgot(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="noodles")]))      # plan forgot "no peanuts"
    r = b.turn("s", "noodles but I'm allergic to peanuts")
    assert r.active_filters["exclude_classes"] == ["peanut"]
    assert "R6" not in {s.recipe_id for s in r.sources}
    assert (r.trace.get("allergen_guard", {}).get("classes_added") or r.trace.get("restrictions_added_by_guard")) == ["peanut"]


def test_guard_never_removes_a_restriction(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne", add_exclude_classes=["dairy"]),
                     TurnPlan(intent="search", query="penne", remove_exclude_classes=["dairy"])]))
    b.turn("s", "no dairy penne")
    r = b.turn("s", "dairy is fine now")
    assert r.active_filters.get("exclude_classes") is None                           # explicit user relaxation only


# ------------------------------------------------------------------------------------------------ degraded modes
def test_llm_down_falls_back_to_rules_and_still_answers(mini_db):
    r = bot(FakeLLM(fail=True)).turn("s", "penne")
    assert r.sources and r.trace["planner"].startswith("rules")


def test_invalid_plan_falls_back_instead_of_crashing(mini_db):
    r = bot(FakeLLM([{"intent": "dance"}])).turn("s", "penne")
    assert r.trace["planner"].startswith("rules") and r.sources


def test_no_llm_configured_at_all(mini_db):
    assert bot(None).turn("s", "I have eggs and rice").intent == "match"


def test_fallback_plan_shapes():
    assert fallback_plan("I have eggs, rice").intent == "match"
    assert fallback_plan("hello").intent == "chat"
    assert fallback_plan("tomato soup").query == "tomato soup"


def test_empty_result_says_so_and_does_not_loosen(mini_db):
    r = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne", add_exclude=["penne"])])).turn("s", "x")
    assert r.status == "no_results" and "did not loosen" in r.answer


# ------------------------------------------------------------------------------------------------ narration + grounding
def test_valid_lead_in_is_added_and_the_facts_stay_tool_derived(mini_db):
    plan = TurnPlan(intent="count", scope="new", add_include=["garlic"])
    first = bot(FakeLLM([plan])).turn("s", "q")
    r = bot(FakeLLM([plan], ["Here is the count you asked for."]), narrate=True).turn("s", "q")
    assert r.answer == "Here is the count you asked for.\n\n" + first.grounded_answer
    assert r.trace["narration"] == "accepted"


@pytest.mark.parametrize("bad", ["There are 4242 of them.", "See [9].", "This is allergy-free.", "Recipe 0123456789abcdef0123456789abcdef"])
def test_lead_in_that_adds_something_is_rejected(mini_db, bad):
    plan = TurnPlan(intent="count", scope="new", add_include=["garlic"])
    first = bot(FakeLLM([plan])).turn("s", "q")
    r = bot(FakeLLM([plan], [bad]), narrate=True).turn("s", "q")
    assert r.answer == first.grounded_answer and r.trace["narration"] == "rejected"


def test_refusals_and_questions_are_never_sent_to_the_narrator(mini_db):
    llm = FakeLLM([TurnPlan(intent="unsupported", unsupported_text="calories")], ["should not be used"])
    bot(llm, narrate=True).turn("s", "calories?")
    assert llm.calls == ["TurnPlan"]


def test_validator_unit():
    g = "Found 12 recipes. 1. Penne [1]\n2. Soup [2]"
    assert ground.validate("12 recipes: Penne [1], Soup [2]", g, 2).ok
    assert not ground.validate("13 recipes: Penne [1], Soup [2]", g, 2).ok
    assert not ground.validate("12 recipes: Penne [1], Soup [2], Pie [3]", g, 2).ok
    assert not ground.validate("12 recipes. This is allergy-free. [1][2]", g, 2).ok
    assert not ground.validate("12 recipes [1][2] id 0123456789abcdef0123456789abcdef", g, 2).ok


# ------------------------------------------------------------------------------------------------ API
def test_chat_endpoint_streams_the_documented_events(mini_db):
    client = TestClient(create_app(bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne")]))))
    r = client.post("/chat", json={"message": "penne"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [e.split("\n")[0][7:] for e in r.text.strip().split("\n\n")]
    assert events[0] == "session" and events[-1] == "done"
    assert {"active_filters", "sources", "computation", "token"} <= set(events)
    srcs = json.loads(next(e for e in r.text.split("\n\n") if e.startswith("event: sources")).split("data: ")[1])
    assert srcs and {"n", "recipe_id", "title", "source"} <= set(srcs[0])


def test_chat_endpoint_validates_input_and_hides_errors(mini_db):
    client = TestClient(create_app(bot(FakeLLM())))
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}


# ------------------------------------------------------------------------------------------------ a title query is never silently dropped
def test_count_with_a_dish_query_uses_the_query(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="count", scope="new", query="penne alfredo")]))
    r = b.turn("s", "how many penne alfredo recipes?")
    assert r.computation["tool"] == "search_recipes" and "penne" in r.answer and " of " in r.answer


def test_rank_with_a_dish_query_only_lists_matching_recipes(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="rank_steps", scope="new", query="penne", k=3)]))
    r = b.turn("s", "penne recipes with the fewest steps")
    assert r.sources and all("penne" in (s.title or "").lower() for s in r.sources)


def test_top_ingredients_with_a_dish_query_asks_instead_of_ignoring_it(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="top_ingredients", scope="new", query="penne alfredo")]))
    assert b.turn("s", "common ingredients in penne recipes").status == "needs_input"


def test_recipe_endpoint_returns_the_stored_recipe_or_404(mini_db):
    client = TestClient(create_app(bot(FakeLLM())))
    ok = client.get("/recipe", params={"id": "R2"})
    assert ok.status_code == 200 and ok.json()["title"] == "Dairy Free Penne" and ok.json()["ingredients"]
    assert client.get("/recipe", params={"id": "no/such.id"}).status_code == 404


# ------------------------------------------------------------------------------------------------ vegetarian toggle
@pytest.fixture
def veg_db(monkeypatch):
    import duckdb
    from recipebot.store import db as dbmod
    from conftest import _build
    data = [("V1", "ar", "Chicken Noodle Soup", ["2 cups chicken broth", "1 pound chicken breast", "1 cup noodles"], "Boil. Serve."),
            ("V2", "ar", "Vegetable Noodle Soup", ["2 cups vegetable broth", "2 carrots", "1 cup noodles"], "Boil. Serve."),
            ("V3", "fn", "Beef Noodle Stir Fry", ["1 pound beef", "1 cup noodles", "2 tablespoons soy sauce"], "Fry. Serve.")]
    con = duckdb.connect(":memory:")
    _build(con, data)
    monkeypatch.setattr(dbmod, "_default", dbmod.Database(connection=con))
    yield
    con.close()


def test_vegetarian_toggle_is_a_hard_filter_applied_in_code(veg_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="noodle"), TurnPlan(intent="search", scope="new", query="noodle")]))
    plain = b.turn("a", "noodle recipes")
    veg = b.turn("b", "noodle recipes", vegetarian=True)
    assert {"V1", "V2", "V3"} <= {s.recipe_id for s in plain.sources}
    assert {s.recipe_id for s in veg.sources} == {"V2"}
    assert veg.active_filters["vegetarian_only"] is True and "vegetarian only" in veg.answer


def test_the_model_cannot_switch_the_vegetarian_toggle_off(veg_db):
    plans = [TurnPlan(intent="search", scope="new", query="noodle"),
             TurnPlan(intent="search", scope="new", query="noodle", remove_exclude_classes=["vegetarian"])]
    b = bot(FakeLLM(plans))
    b.turn("s", "noodles", vegetarian=True)
    r = b.turn("s", "allow meat too please")                    # no toggle value sent: the setting stays
    assert {s.recipe_id for s in r.sources} == {"V2"}


def test_toggle_is_sent_by_the_api_and_can_be_turned_off(veg_db):
    plans = [TurnPlan(intent="search", scope="new", query="noodle")] * 2
    client = TestClient(create_app(bot(FakeLLM(plans))))
    on = client.post("/chat", json={"message": "noodles", "session_id": "x", "vegetarian": True}).text
    off = client.post("/chat", json={"message": "noodles", "session_id": "x", "vegetarian": False}).text
    assert "Chicken Noodle Soup" not in on and "Chicken Noodle Soup" in off


# ------------------------------------------------------------------------------------------------ gaps closed after the brief review
def test_average_and_median_ingredient_counts_are_reachable_from_chat(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="ingredient_stats", scope="new"),
                     TurnPlan(intent="ingredient_stats", scope="new", by_source=True)]))
    r = b.turn("a", "what is the median number of ingredients per recipe?")
    assert r.computation["tool"] == "ingredient_count_summary" and "average" in r.answer and "median" in r.answer
    assert "all recipes:" in r.answer
    by = b.turn("b", "and per collection?")
    assert "Allrecipes" in by.answer and "Food Network" in by.answer


def test_ingredient_stats_with_a_dish_query_asks_instead_of_ignoring_it(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="ingredient_stats", scope="new", query="penne alfredo")]))
    assert b.turn("s", "average ingredients in penne alfredo recipes").status == "needs_input"


def test_the_count_explains_how_the_ingredient_word_was_matched(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="count", scope="new", add_include=["garlic"])]))
    notes = " ".join(b.turn("s", "how many recipes have garlic?").computation["notes"])
    assert "Matching rule for 'garlic'" in notes and "garlic powder" in notes and "NOT counted as garlic" in notes


def test_the_substitution_answer_contains_the_full_adapted_recipe(mini_db, monkeypatch):
    from dataclasses import replace
    from recipebot.domain.models import AdaptedLine, Computation, InstructionEdit, SubstitutionResult
    from recipebot.orchestrator import execute as ex

    def handler(req):
        comp = Computation(tool="suggest_substitution", description="d", population="p", filters={}, funnel=[],
                           n_analyzed=1, method="m", sql="s")
        return SubstitutionResult(
            status="adapted", summary="Adapted.", recipe_id=req.recipe_id, recipe_title="Cookies", targets=[],
            adapted_ingredients=[AdaptedLine(position=1, text="2 cups flour", changed=False),
                                 AdaptedLine(position=2, text="1 cup margarine", changed=True, original="1 cup butter", evidence_level="weak"),
                                 AdaptedLine(position=3, text="", changed=True, original="1 egg")],
            revised_instructions="Cream the margarine.\nBake 10 minutes.",
            instruction_edits=[InstructionEdit(step=1, action="replace", before="Cream the butter.", after="Cream the margarine.")],
            predicted_effects=[], restriction_check=[], remaining_issues=[], external_search_suggested=False,
            evidence_policy="e", disclaimer="Text check only.", computation=comp)

    monkeypatch.setitem(ex.TOOLS, "suggest_substitution", replace(ex.TOOLS["suggest_substitution"], handler=handler))
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne"), TurnPlan(intent="substitute", ref_index=1, remove=["butter"])]))
    b.turn("s", "penne")
    r = b.turn("s", "replace the butter in the first one")
    a = r.grounded_answer                      # the FULL text; the screen gets a shorter one (the lists live in the adapted recipe)
    assert "Updated ingredient list" not in r.answer and "Open the adapted recipe above" in r.answer
    assert r.sources[0].kind == "adapted" and [i["status"] for i in r.sources[0].view["ingredients"]][1:] == ["changed", "removed"]
    assert "Updated ingredient list" in a and "- 2 cups flour" in a
    assert "**1 cup margarine** (replaces: 1 cup butter; tentative, weak evidence)" in a
    assert "(removed: 1 egg)" in a
    assert "Revised instructions:" in a and "1. Cream the margarine." in a and "2. Bake 10 minutes." in a
    assert "please read them before cooking" in a


# ------------------------------------------------------------------------------------------------ the guard for a forgotten step limit
@pytest.mark.parametrize("msg,expected", [
    ("at most 6 steps please", 6), ("no more than 5 instruction steps", 5), ("under 4 steps", 3), ("fewer than 10 steps", 9),
    ("6 steps or fewer", 6), ("in 3 steps or less", 3), ("max 8 steps", 8), ("fewest steps", None), ("I have 6 eggs", None),
])
def test_detect_max_steps(msg, expected):
    from recipebot.orchestrator.bot import detect_max_steps
    assert detect_max_steps(msg) == expected


def test_guard_adds_a_step_limit_the_llm_forgot(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="match", scope="new", available_text="penne, tomatoes")]))     # plan forgot the limit
    r = b.turn("s", "I have penne and tomatoes, at most 4 steps")
    assert r.active_filters["max_steps"] == 4 and r.trace["step_limit_added_by_guard"] == 4


def test_guard_does_not_override_a_step_limit_the_plan_set(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="match", scope="new", available_text="penne", max_steps=3)]))
    assert b.turn("s", "I have penne, at most 4 steps").active_filters["max_steps"] == 3
