"""Phase 10: serving scaling, taste adjustment, web search for weak evidence, latency settings."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recipebot.api.app import create_app
from recipebot.llm.gemini import GeminiClient, LLMUnavailable
from recipebot.orchestrator.bot import Chatbot, detect_scale, enforce_scale, fallback_plan
from recipebot.orchestrator.plan import TurnPlan
from recipebot.tools import web as web_tools
from recipebot.tools.amounts import format_amount, scale_line, scale_lines
from recipebot.tools.taste import adjust_for_taste, axes_of, clean_profile

from test_orchestrator import FakeLLM


def bot(llm):
    return Chatbot(llm, narrate=False)


# ------------------------------------------------------------------------------------------------ amounts
@pytest.mark.parametrize("line,factor,expected", [
    ("2 cups flour", 2, "4 cups flour"),
    ("1 cup sugar", 2, "2 cups sugar"),
    ("2 cups sugar", 0.5, "1 cup sugar"),
    ("1 1/2 cups milk", 2, "3 cups milk"),
    ("1/2 cup butter", 3, "1 1/2 cups butter"),
    ("1/4 teaspoon salt", 0.5, "1/8 teaspoon salt"),
    ("2-3 cloves garlic", 2, "4-6 cloves garlic"),
    ("1 (14 ounce) can tomatoes", 2, "2 (14 ounce) cans tomatoes"),
    ("1 egg", 2, "2 eggs"),
    ("4 tomatoes", 0.5, "2 tomatoes"),
    ("1 tablespoon oil", 1.5, "1 1/2 tablespoons oil"),
])
def test_scale_line(line, factor, expected):
    assert scale_line(line, factor) == expected


def test_lines_without_an_amount_or_with_a_package_size_are_not_scaled():
    assert scale_line("salt to taste", 2) is None
    assert scale_line("8-ounce package cream cheese", 2) is None          # a size, not a count
    out = scale_lines(["2 cups flour", "salt to taste", "8-ounce package cream cheese"], 2)
    assert [s.scaled for s in out] == [True, False, False]
    assert out[2].reason == "amount is a package size"


def test_factor_one_changes_nothing_and_formatting_is_friendly():
    assert all(not s.scaled for s in scale_lines(["2 cups flour"], 1))
    assert format_amount(0.333) == "1/3" and format_amount(2.5) == "2 1/2" and format_amount(12.4) == "12"
    assert format_amount(0.97) == "1"


# ------------------------------------------------------------------------------------------------ scale intent
@pytest.mark.parametrize("msg,expected", [
    ("double it", {"scale_factor": 2.0}),
    ("make it half", {"scale_factor": 0.5}),
    ("triple the recipe", {"scale_factor": 3.0}),
    ("scale it 1.5 times", {"scale_factor": 1.5}),
    ("it serves 4, make it for 6", {"servings_from": 4, "servings_to": 6}),
    ("make it for 8 people", {"servings_to": 8}),
])
def test_detect_scale(msg, expected):
    assert detect_scale(msg) == expected


def test_guard_restores_numbers_the_planner_dropped():
    plan = TurnPlan(intent="scale", ref_index=1)
    assert enforce_scale(plan, "double the first one") == {"scale_factor": 2.0} and plan.scale_factor == 2.0
    kept = TurnPlan(intent="scale", scale_factor=3)
    assert enforce_scale(kept, "double it") == {} and kept.scale_factor == 3          # the plan's own number wins
    assert enforce_scale(TurnPlan(intent="search"), "double it") == {}


def test_fallback_planner_understands_scaling():
    assert fallback_plan("double it").intent == "scale"
    assert fallback_plan("make it for 6 people").servings_to == 6


def test_scale_answer_scales_the_amounts_and_does_not_claim_servings(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="scale", ref_index=1, scale_factor=2)]))
    b.turn("s", "margarine cookies")
    r = b.turn("s", "double it")
    assert r.intent == "scale" and r.status == "ok"
    assert "**4 cups flour**" in r.answer and "**2 cups sugar**" in r.answer
    assert "NOT scaled" in r.answer                                  # honest about instructions and times
    assert "no servings" not in r.answer.lower()                      # the servings refusal must not fire on a scale request


def test_scale_without_a_factor_asks_and_says_why(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="scale", ref_index=1, servings_to=6)]))
    b.turn("s", "margarine cookies")
    r = b.turn("s", "make it for 6")
    assert r.status == "needs_input" and "does not say how many people" in r.answer


def test_servings_from_and_to_give_the_factor(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="scale", ref_index=1, servings_from=4, servings_to=6)]))
    b.turn("s", "margarine cookies")
    r = b.turn("s", "it serves 4, make it for 6")
    assert "scaled by 1.5x (from 4 to 6 servings)" in r.answer and "**3 cups flour**" in r.answer


def test_absurd_factor_is_refused(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="scale", ref_index=1, scale_factor=500)]))
    b.turn("s", "margarine cookies")
    assert b.turn("s", "x500").status == "needs_input"


# ------------------------------------------------------------------------------------------------ taste
def test_profile_is_validated():
    assert clean_profile({"sweet": "LOW", "salt": "extreme", "spice": "high", "x": "low"}) == {"sweet": "low", "spice": "high"}
    assert clean_profile(None) == {}


@pytest.mark.parametrize("line,axes", [
    ("1 teaspoon salt", ["salt"]), ("1/2 cup butter, salted", []), ("2 tablespoons soy sauce", ["salt"]),
    ("1/4 teaspoon cayenne pepper", ["spice"]), ("1 red bell pepper", []), ("1 teaspoon red pepper flakes", ["spice"]),
    ("salt and pepper to taste", ["salt", "spice"]), ("1 cup sugar", ["sweet"]), ("1 cup sugar snap peas", []),
    ("2 tablespoons honey", ["sweet"]), ("1 jalapeno, minced", ["spice"]), ("2 cups flour", []),
])
def test_axes_of(line, axes):
    assert axes_of(line) == axes


def test_low_spice_halves_the_hot_lines_only():
    box = adjust_for_taste(["1 teaspoon cayenne pepper", "2 cups rice", "1 cup sugar"], {"spice": "low"})
    assert [(a.axis, a.suggested, a.change) for a in box.adjustments] == [("spice", "1/2 teaspoon cayenne pepper", "Reduce by about 50%")]
    assert "not from the dataset" in box.note


def test_high_levels_increase_and_medium_changes_nothing():
    up = adjust_for_taste(["2 tablespoons honey"], {"sweet": "high"}).adjustments[0]
    assert up.suggested == "3 tablespoons honey" and up.change == "Increase by about 50%"
    flat = adjust_for_taste(["2 tablespoons honey"], {"sweet": "medium", "salt": "medium", "spice": "medium"})
    assert flat.adjustments == [] and "as written" in flat.note


def test_a_line_with_no_amount_gets_advice_not_a_number():
    a = adjust_for_taste(["salt and pepper to taste"], {"salt": "low", "spice": "high"}).adjustments
    assert {x.axis for x in a} == {"salt", "spice"} and all(x.suggested is None and x.advice for x in a)


def test_baking_keeps_sugar_and_salt_changes_small():
    lines = ["2 cups flour", "1 teaspoon baking powder", "1 cup sugar", "1/2 teaspoon salt"]
    box = adjust_for_taste(lines, {"sweet": "low", "salt": "low"})
    sugar = next(a for a in box.adjustments if a.axis == "sweet")
    assert box.baking and sugar.change == "Reduce by about 25%" and sugar.suggested == "3/4 cup sugar"
    assert "baking" in box.note


def test_nothing_to_change_is_said_plainly():
    box = adjust_for_taste(["2 cups rice", "1 pound chicken"], {"spice": "high"})
    assert box.adjustments == [] and "nothing to change" in box.note


def test_recipe_endpoint_scales_and_adds_the_taste_box(mini_db):
    client = TestClient(create_app(bot(FakeLLM())))
    body = client.get("/recipe", params={"id": "R9", "scale": 2, "sweet": "low"}).json()
    assert body["ingredients"][0]["text"] == "4 cups flour" and body["ingredients"][0]["original"].startswith("2 cups")
    sugar = [a for a in body["taste"]["adjustments"] if a["axis"] == "sweet"]
    assert sugar and sugar[0]["line"] == "2 cups sugar" and sugar[0]["suggested"] == "1 cup sugar"
    assert client.get("/recipe", params={"id": "R9", "scale": 99}).status_code == 422
    plain = client.get("/recipe", params={"id": "R9"}).json()
    assert plain["taste"]["adjustments"] == [] and plain["ingredients"][0]["text"] == "2 cups flour"


def test_chat_taste_profile_reaches_the_recipe_answer(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="peanut noodles"), TurnPlan(intent="get_recipe", ref_index=1)]))
    client = TestClient(create_app(b))
    client.post("/chat", json={"message": "peanut noodles", "session_id": "t", "taste": {"salt": "low", "bogus": "high"}})
    r = b.turn("t", "show it")
    assert "Adjusted to your taste (salt low)" in r.answer and "soy sauce" in r.answer
    assert b.sessions.get("t").taste == {"salt": "low"}


# ------------------------------------------------------------------------------------------------ web search
class FakeWeb:
    def __init__(self, hits=None):
        self.hits, self.queries = hits if hits is not None else [
            web_tools.WebHit("Egg replacers", "https://example.com/eggs", "Use applesauce.")], []

    def search(self, query, max_results=3):
        self.queries.append(query)
        return self.hits


@pytest.fixture
def web():
    fake = FakeWeb()
    web_tools.set_searcher(fake)
    yield fake
    web_tools.set_searcher(None)


def _weak_substitution(monkeypatch, verdict):
    from dataclasses import replace
    from recipebot.domain.models import Computation, SubstitutionResult
    from recipebot.orchestrator import execute as ex
    from types import SimpleNamespace as NS

    def handler(req):
        comp = Computation(tool="suggest_substitution", description="d", population="p", filters={}, funnel=[],
                           n_analyzed=1, method="m", sql="s")
        target = NS(ingredient="egg", verdict=verdict, verdict_reason="r", candidates=[], chosen=None, why="you asked to remove 'egg'")
        return SubstitutionResult.model_construct(
            status="partial", summary="s", recipe_id=req.recipe_id, recipe_title="Margarine Cookies", targets=[target],
            adapted_ingredients=[], revised_instructions="", instruction_edits=[], predicted_effects=[],
            restriction_check=[], remaining_issues=[], external_search_suggested=True, evidence_policy="e",
            disclaimer="Text check only.", computation=comp)

    monkeypatch.setitem(ex.TOOLS, "suggest_substitution", replace(ex.TOOLS["suggest_substitution"], handler=handler))


def test_weak_evidence_triggers_a_labelled_web_search(mini_db, monkeypatch, web):
    _weak_substitution(monkeypatch, "weak")
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="substitute", ref_index=1, remove=["egg"])]))
    b.turn("s", "margarine cookies")
    r = b.turn("s", "replace the egg")
    assert web.queries == ["egg substitute in Margarine Cookies"]
    assert r.web and r.web[0]["url"] == "https://example.com/eggs" and r.web[0]["domain"] == "example.com"
    assert "NOT from the dataset" in r.answer and "Web: Egg replacers (example.com)" in r.answer


def test_supported_evidence_never_searches_the_web(mini_db, monkeypatch, web):
    _weak_substitution(monkeypatch, "supported")
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="substitute", ref_index=1, remove=["egg"])]))
    b.turn("s", "margarine cookies")
    r = b.turn("s", "replace the egg")
    assert web.queries == [] and r.web == []


def test_web_search_off_or_empty_never_breaks_the_answer(mini_db, monkeypatch):
    _weak_substitution(monkeypatch, "none")
    web_tools.set_searcher(None)
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="substitute", ref_index=1, remove=["egg"])]))
    b.turn("s", "margarine cookies")
    assert b.turn("s", "replace the egg").web == []
    empty = FakeWeb(hits=[])
    web_tools.set_searcher(empty)
    b2 = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                      TurnPlan(intent="substitute", ref_index=1, remove=["egg"])]))
    b2.turn("s", "margarine cookies")
    assert "returned nothing" in b2.turn("s", "replace the egg").answer
    assert empty.queries == ["egg substitute in Margarine Cookies", "best substitute for egg in cooking"]   # one retry, less specific
    web_tools.set_searcher(None)


def test_web_hits_are_sanitised():
    bad = web_tools._safe_hit("Title\x00 with\x1b control", "javascript:alert(1)", "x")
    assert bad is None
    ok = web_tools._safe_hit("T" * 500, "https://www.site.com/a", "line\x00one" + "y" * 500)
    assert len(ok.title) == 140 and len(ok.snippet) == 220 and "\x00" not in ok.snippet and ok.domain == "site.com"


def test_api_sends_a_web_event(mini_db, monkeypatch, web):
    _weak_substitution(monkeypatch, "weak")
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="substitute", ref_index=1, remove=["egg"])]))
    client = TestClient(create_app(b))
    client.post("/chat", json={"message": "margarine cookies", "session_id": "w"})
    text = client.post("/chat", json={"message": "replace the egg", "session_id": "w"}).text
    assert "event: web" in text and "https://example.com/eggs" in text


# ------------------------------------------------------------------------------------------------ latency
def test_the_chat_makes_one_model_call_per_turn_by_default(mini_db):
    from recipebot.settings import Settings
    assert Settings().narrate_with_llm is False
    llm = FakeLLM([TurnPlan(intent="search", scope="new", query="penne")])
    r = Chatbot(llm, narrate=False).turn("s", "penne")
    assert llm.calls == ["TurnPlan"] and r.status == "ok"


def test_interactive_client_gives_up_instead_of_waiting(tmp_path, monkeypatch):
    """A long rate-limit wait must turn into LLMUnavailable (the rule-based planner then answers), not a 30 s stall."""
    from pydantic import SecretStr
    from google.genai import errors
    c = GeminiClient(SecretStr("k"), "m", tmp_path, min_interval_s=0, max_wait_s=8.0, request_timeout_s=12)

    class Boom:
        class models:
            @staticmethod
            def generate_content(**kw):
                e = errors.APIError.__new__(errors.APIError)
                e.code, e.message = 429, "quota per minute, retry in 40s"
                e.args = ("429 retry in 40s",)
                raise e

    c._client = Boom()
    with pytest.raises(LLMUnavailable):
        c.generate_json("s", "u", TurnPlan)


def test_a_dropped_connection_becomes_llm_unavailable(tmp_path):
    from pydantic import SecretStr
    c = GeminiClient(SecretStr("k"), "m", tmp_path, min_interval_s=0, request_timeout_s=1)

    class Boom:
        class models:
            @staticmethod
            def generate_content(**kw):
                raise TimeoutError("read timed out")

    c._client = Boom()
    with pytest.raises(LLMUnavailable):
        c.generate_json("s", "u", TurnPlan)


def test_warm_up_pays_the_first_call_cost_in_the_background_and_never_raises(mini_db):
    llm = FakeLLM([TurnPlan(intent="chat")])
    Chatbot(llm, narrate=False).warm_up()
    assert llm.calls == ["TurnPlan"]                                  # one throwaway planner call
    Chatbot(FakeLLM(fail=True), narrate=False).warm_up()               # a failing model must not break startup
    Chatbot(None, narrate=False).warm_up()                            # and neither does having no model at all


def test_the_timeout_can_never_go_below_the_services_minimum(tmp_path):
    from pydantic import SecretStr
    assert GeminiClient(SecretStr("k"), "m", tmp_path, request_timeout_s=3).request_timeout_s == 10.0   # the API answers 400 below 10 s
    assert GeminiClient(SecretStr("k"), "m", tmp_path).request_timeout_s is None                          # batch jobs: no limit


def test_the_adapted_recipe_is_a_source_with_its_own_view_and_the_leads_are_listed_as_rows(mini_db, monkeypatch):
    from dataclasses import replace
    from recipebot.domain.models import AdaptedLine, Computation, SubstitutionResult
    from recipebot.orchestrator import execute as ex
    from types import SimpleNamespace as NS

    def handler(req):
        comp = Computation(tool="suggest_substitution", description="d", population="p", filters={}, funnel=[],
                           n_analyzed=1, method="m", sql="s")
        cand = NS(substitute="applesauce", tier="indirect", n_supporting=1, n_alternative_lines=0,
                  evidence=[NS(id="R3", title="Penne Alfredo", line="1 cup applesauce")])
        target = NS(ingredient="egg", verdict="weak", verdict_reason="r", candidates=[cand], chosen=None,
                    why="you asked to remove 'egg'")
        lines = [AdaptedLine(position=1, text="2 cups flour", changed=False),
                 AdaptedLine(position=2, text="", changed=True, original="1 egg")]
        return SubstitutionResult.model_construct(
            status="partial", summary="s", recipe_id=req.recipe_id, recipe_title="Margarine Cookies", targets=[target],
            adapted_ingredients=lines, revised_instructions="Mix.\nBake.", instruction_edits=[NS(step=1, action="review")],
            predicted_effects=[], restriction_check=[], remaining_issues=[], external_search_suggested=False,
            evidence_policy="e", disclaimer="Text check only.", computation=comp)

    monkeypatch.setitem(ex.TOOLS, "suggest_substitution", replace(ex.TOOLS["suggest_substitution"], handler=handler))
    web_tools.set_searcher(None)
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="margarine cookies"),
                     TurnPlan(intent="substitute", ref_index=1, remove=["egg"])]))
    b.turn("s", "margarine cookies")
    r = b.turn("s", "remove the egg")
    first = r.sources[0]
    assert first.kind == "adapted" and [i["status"] for i in first.view["ingredients"]] == ["same", "removed"]
    assert first.view["instructions"] == ["Mix.", "Bake."] and any("removed, no replacement" in n for n in first.view["notes"])
    assert all(s.kind == "recipe" and s.view is None for s in r.sources[1:])
    assert "2. Penne Alfredo [2] (shows: 1 cup applesauce)" in r.answer         # the lead is a numbered row, not only an evidence entry
    from recipebot.api.app import events
    import json
    stream = "".join(events(r, "v"))
    frame = next(x for x in stream.split("\n\n") if x.startswith("event: sources"))
    sent = json.loads(frame.split("data: ", 1)[1])
    assert sent[0]["kind"] == "adapted" and sent[0]["view"]["ingredients"][1]["status"] == "removed" and sent[1]["kind"] == "recipe"


# ------------------------------------------------------------------------------------------------ vegetarian mode (Indian "veg")
@pytest.fixture
def veg2(monkeypatch):
    import duckdb
    from recipebot.store import db as dbmod
    from conftest import _build
    data = [("E1", "ar", "Egg Fried Noodles", ["2 eggs", "1 cup noodles"], "Fry."),
            ("E2", "ar", "Plain Noodles Soup", ["2 cups vegetable broth", "1 cup noodles"], "Boil."),
            ("E3", "ar", "Chicken Noodles", ["1 pound chicken", "1 cup noodles"], "Fry."),
            ("E4", "ar", "Carrot Noodles", ["2 carrots", "1 cup noodles"], "Boil.")]
    con = duckdb.connect(":memory:")
    _build(con, data)
    monkeypatch.setattr(dbmod, "_default", dbmod.Database(connection=con))
    yield
    con.close()


def test_vegetarian_mode_also_leaves_out_recipes_with_egg(veg2):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="noodles")]))
    r = b.turn("a", "noodles", vegetarian=True)
    ids = {s.recipe_id for s in r.sources}
    assert "E1" not in ids and "E3" not in ids and {"E2", "E4"} <= ids        # no egg, no chicken; plain "noodles" lines stay


def test_asking_for_non_veg_food_in_veg_mode_says_so_and_offers_the_switch(veg2):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="butter chicken")] * 2))
    r = b.turn("a", "butter chicken recipe", vegetarian=True)
    assert r.status == "needs_input" and r.suggest == "turn_off_vegetarian" and r.sources == []
    assert "chicken" in r.answer and "Turn the switch off" in r.answer
    assert b.sessions.get("a").query is None                                    # the blocked request did not change the filters
    off = b.turn("a", "butter chicken recipe", vegetarian=False)                  # with the switch off the same request is a normal search
    assert off.suggest is None and off.status in ("ok", "no_results")


@pytest.mark.parametrize("q,blocked", [("chicken curry", True), ("egg fried rice", True), ("mutton biryani", True), ("omelette", True),
                                       ("banana cake", False), ("paneer butter masala", False), ("noodle soup", False), ("tofu stir fry", False)])
def test_which_requests_count_as_non_vegetarian(q, blocked):
    from recipebot.orchestrator.execute import nonveg_words
    assert bool(nonveg_words(q)) is blocked


def test_the_done_event_carries_the_suggestion(veg2):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="fish fry")]))
    text = TestClient(create_app(b)).post("/chat", json={"message": "fish fry", "session_id": "z", "vegetarian": True}).text
    assert '"suggest": "turn_off_vegetarian"' in text


# ------------------------------------------------------------------------------------------------ allergen guard
@pytest.mark.parametrize("msg,expected", [
    ("I'm allergic to grapes", ["grapes"]), ("allergic to peanuts and tree nuts", ["peanuts", "tree nuts"]),
    ("I can't eat shrimp, or mushrooms", ["shrimp", "mushrooms"]), ("I am lactose intolerant", []),
    ("avoid cilantro please", ["cilantro"]), ("no more than 5 steps", []), ("I can't have more than 6 steps", []),
    ("allergic to kiwi too", ["kiwi"]),
])
def test_stated_allergens(msg, expected):
    from recipebot.orchestrator.bot import stated_allergens
    assert stated_allergens(msg) == expected


def test_a_class_the_user_never_mentioned_is_dropped_and_the_named_food_is_excluded():
    """Found by the user: 'I'm allergic to grapes' came back from the planner as 'no egg' and grapes were not excluded."""
    from recipebot.orchestrator.bot import enforce_allergens
    plan = TurnPlan(intent="search", add_exclude_classes=["egg"])
    report = enforce_allergens(plan, "I'm allergic to grapes")
    assert plan.add_exclude_classes == [] and plan.add_exclude == ["grapes"]
    assert report == {"classes_dropped": ["egg"], "ingredients_added": ["grapes"]}


@pytest.mark.parametrize("msg,cls", [("I'm allergic to peanuts", "peanut"), ("I'm lactose intolerant", "dairy"), ("no dairy please", "dairy"),
                                     ("gluten-free pasta", "gluten"), ("I'm allergic to shrimp", "shellfish"), ("I can't eat eggs", "egg"),
                                     ("allergic to almonds", "tree_nut"), ("I'm vegetarian", "vegetarian"), ("no pork", "pork")])
def test_a_class_the_user_did_state_is_kept(msg, cls):
    from recipebot.orchestrator.bot import enforce_allergens
    plan = TurnPlan(intent="search", add_exclude_classes=[cls])
    assert enforce_allergens(plan, msg).get("classes_dropped") is None and cls in plan.add_exclude_classes


def test_a_named_allergen_that_belongs_to_a_class_gets_the_class_even_if_the_planner_forgot(mini_db):
    from recipebot.orchestrator.bot import enforce_allergens
    plan = TurnPlan(intent="search")
    enforce_allergens(plan, "I'm allergic to shrimp")
    assert "shellfish" in plan.add_exclude_classes and plan.add_exclude == []


def test_the_whole_turn_filters_the_allergen_even_when_the_planner_guesses_wrong(mini_db):
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="penne"),
                     TurnPlan(intent="search", add_exclude_classes=["egg"])]))        # the planner's wrong guess
    b.turn("s", "penne")
    r = b.turn("s", "I'm allergic to tomatoes")
    assert r.active_filters.get("exclude_ingredients") == ["tomatoes"] and "exclude_classes" not in r.active_filters
    assert r.trace["allergen_guard"]["classes_dropped"] == ["egg"]
    assert r.sources
    for src in r.sources:                                                                 # no shown recipe has a tomato line
        lines = " ".join(x[0] for x in mini_db.query("SELECT text FROM recipe_ingredients WHERE recipe_id = ?", [src.recipe_id])).lower()
        assert "tomato" not in lines


def test_a_class_without_synonyms_is_dropped_when_the_message_does_not_mention_it():
    """Regression: an empty regex alternative made every class without a synonym list look 'stated' (found by the extended eval)."""
    from recipebot.orchestrator.bot import class_is_stated
    assert class_is_stated("peanut", "I'm lactose intolerant") is False
    assert class_is_stated("sesame", "no dairy please") is False
    assert class_is_stated("peanut", "I'm allergic to groundnut? no, peanuts") is True


@pytest.mark.parametrize("msg,cls", [("I'm lactose intolerant", "dairy"), ("I have a peanut allergy", "peanut"), ("gluten intolerance", "gluten"),
                                     ("egg allergy, show me cakes", "egg"), ("dairy sensitivity", "dairy")])
def test_condition_phrases_are_restrictions(msg, cls):
    from recipebot.orchestrator.bot import detect_restrictions
    assert cls in detect_restrictions(msg)


# ------------------------------------------------------------------------------------------------ faults found by the extended evaluation
@pytest.mark.parametrize("msg,expected", [
    ("and how many of those are vegetarian?", ["vegetarian"]), ("only vegan ones", ["vegan"]), ("Show me vegetarian soups", ["vegetarian"]),
    ("non-vegetarian dishes", []), ("I'm lactose intolerant", ["dairy"]), ("recipes with dairy", [])])
def test_a_requested_vegetarian_or_vegan_is_a_restriction(msg, expected):
    from recipebot.orchestrator.bot import detect_restrictions
    assert detect_restrictions(msg) == expected


def test_a_request_that_asks_to_confirm_safety_is_not_refused_whole():
    from recipebot.orchestrator.bot import enforce_safety_request
    plan = TurnPlan(intent="unsupported", unsupported_text="confirm they are safe")
    assert enforce_safety_request(plan, "I have a nut allergy, show me cookie recipes and confirm they are safe") is True
    assert plan.intent == "search" and plan.query == "cookie"
    other = TurnPlan(intent="unsupported", unsupported_text="calories")
    assert enforce_safety_request(other, "how many calories are in cookie recipes") is False and other.intent == "unsupported"
    no_dish = TurnPlan(intent="unsupported", unsupported_text="is it safe")
    assert enforce_safety_request(no_dish, "is it safe?") is False


def test_veg_mode_leaves_out_an_egg_line_that_was_parsed_to_a_junk_name(monkeypatch):
    import duckdb
    from recipebot.store import db as dbmod
    from conftest import _build
    data = [("J1", "ar", "Scrambled Eggs", ["Allow 2 medium eggs per person for an appetizer or light snack", "3 tbsp butter"], "Mix. Cook."),
            ("J2", "ar", "Plain Noodle Soup", ["2 cups vegetable broth", "1 cup noodles"], "Boil."),
            ("J3", "ar", "Egg Noodle Bake", ["2 cups egg noodles", "1 cup peas"], "Bake.")]
    con = duckdb.connect(":memory:")
    _build(con, data)
    monkeypatch.setattr(dbmod, "_default", dbmod.Database(connection=con))
    b = bot(FakeLLM([TurnPlan(intent="search", scope="new", query="noodles")]))
    ids = {s.recipe_id for s in b.turn("a", "noodles", vegetarian=True).sources}
    assert "J1" not in ids and "J2" in ids          # a plain "may contain egg" noodles line is allowed, a line with the word eggs is not
    con.close()


def test_a_single_ingredient_word_in_the_dish_query_is_moved_to_the_ingredient_filter(mini_db):
    from recipebot.orchestrator.bot import normalize_ingredient_query
    plan = TurnPlan(intent="rank_steps", query="tomato", k=5)
    assert normalize_ingredient_query(plan) == "tomato" and plan.add_include == ["tomato"] and plan.query is None
    dish = TurnPlan(intent="count", query="penne alfredo")                      # two words: a dish, left alone
    assert normalize_ingredient_query(dish) is None and dish.query == "penne alfredo"
    search = TurnPlan(intent="search", query="tomato")                          # a search keeps its query
    assert normalize_ingredient_query(search) is None
    unknown = TurnPlan(intent="count", query="flurbleberries")
    assert normalize_ingredient_query(unknown) is None
    given = TurnPlan(intent="count", query="tomato", add_include=["basil"])
    assert normalize_ingredient_query(given) is None
