"""The ingredient labeling pipeline: raise-only merge, human review, evaluation maths, and the free-tier-safe client.
Nothing here calls the real API."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from pydantic import SecretStr

from recipebot.domain.taxonomy import Taxonomy, get_taxonomy
from recipebot.labeling import evaluate, merge
from recipebot.labeling.llm_labeler import label_names, load_labels, save_labels
from recipebot.labeling.schema import ClassFlag, NameLabel
from recipebot.llm.gemini import BudgetExceeded, DailyQuotaExhausted, GeminiClient, LLMUnavailable


def lab(name, classes=None, parent=None, is_ingredient=True, reason="r"):
    return NameLabel(name=name, is_ingredient=is_ingredient, product=name, parent=parent or name, form="none",
                     classes=[ClassFlag(cls=c, level=l) for c, l in (classes or {}).items()], reason=reason)


TAX = get_taxonomy()


# ------------------------------------------------------------------ merge: the asymmetric safety rule
def test_llm_raising_to_yes_is_applied_automatically():
    cmp = merge.compare({"chourico": lab("chourico", {"pork": "yes"})}, TAX, {}, {"chourico": 2})
    assert cmp.raises == {"chourico": {"pork": "yes"}} and not cmp.raises_review


def test_llm_raising_to_maybe_on_a_rare_name_is_applied_automatically():
    cmp = merge.compare({"rare thing": lab("rare thing", {"egg": "maybe"})}, TAX, {}, {"rare thing": 3})
    assert cmp.raises == {"rare thing": {"egg": "maybe"}}


def test_llm_maybe_on_a_common_name_waits_for_a_human():
    cmp = merge.compare({"spaghetti": lab("spaghetti", {"gluten": "yes", "egg": "maybe"})}, TAX, {}, {"spaghetti": 578})
    assert "egg" not in cmp.raises.get("spaghetti", {})                       # not applied
    assert [(r["name"], r["class"], r["llm_says"]) for r in cmp.raises_review] == [("spaghetti", "egg", "maybe")]


def test_llm_can_never_lower_a_rule_flag_it_goes_to_review_and_the_rule_stays():
    assert TAX.classify_name("margarine", rules_only=True)["dairy"] == "maybe"
    cmp = merge.compare({"margarine": lab("margarine", {})}, TAX, {}, {"margarine": 1100})
    assert "margarine" not in cmp.raises
    assert [(r["class"], r["rules_say"], r["llm_says"]) for r in cmp.lowers if r["name"] == "margarine"][0] == ("dairy", "maybe", "none")


def test_llm_saying_not_an_ingredient_is_only_a_suggestion():
    cmp = merge.compare({"hot dog": lab("hot dog", is_ingredient=False)}, TAX, {"hot dog": "hot dog"}, {"hot dog": 50})
    assert cmp.not_ingredient and cmp.not_ingredient[0]["name"] == "hot dog"


def test_llm_parents_are_not_used_by_default():
    cmp = merge.compare({"lemon juice": lab("lemon juice", parent="juice")}, TAX, {"lemon juice": "lemon juice"}, {"lemon juice": 99})
    assert cmp.parents == []
    on = merge.compare({"lemon juice": lab("lemon juice", parent="juice")}, TAX, {"lemon juice": "lemon juice"},
                       {"lemon juice": 99}, use_llm_parents=True)
    assert on.parents


def test_overlay_only_ever_raises(tmp_path):
    path = tmp_path / "labels.json"
    c1 = merge.Comparison(raises={"x": {"dairy": "yes"}})
    merge.write_overlay(path, c1, "m")
    c2 = merge.Comparison(raises={"x": {"dairy": "maybe", "egg": "maybe"}})        # a later, weaker label
    merge.write_overlay(path, c2, "m")
    got = json.loads(path.read_text())["labels"]["x"]["classes"]
    assert got == {"dairy": "yes", "egg": "maybe"}                                  # yes is kept; egg is added


# ------------------------------------------------------------------ taxonomy: overlay and human overrides
def make_tax(tmp_path, overlay=None, human=None):
    cfg = tmp_path / "cfg"
    (cfg / "labels").mkdir(parents=True)
    src = get_taxonomy_path()
    (cfg / "taxonomy.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    if overlay:
        (cfg / "labels" / "ingredient_labels.json").write_text(json.dumps({"labels": overlay}))
    if human:
        (cfg / "ingredient_overrides.yaml").write_text(yaml.safe_dump({"classes": human}))
    return Taxonomy(cfg / "taxonomy.yaml")


def get_taxonomy_path():
    from recipebot.settings import get_settings
    return get_settings().config_dir / "taxonomy.yaml"


def test_overlay_raises_but_cannot_lower(tmp_path):
    t = make_tax(tmp_path, overlay={"chourico": {"classes": {"pork": "yes"}},
                                    "margarine": {"classes": {"dairy": "none", "egg": "maybe"}}})
    assert t.classify_name("chourico")["pork"] == "yes"
    out = t.classify_name("margarine")
    assert out["dairy"] == "maybe"                       # the rule's flag survives an overlay that says "none"
    assert out["egg"] == "maybe"                         # but a raise is applied


def test_human_decision_wins_in_both_directions(tmp_path):
    t = make_tax(tmp_path, human={"margarine": {"dairy": "none"}, "spaghetti": {"egg": "maybe"}})
    assert "dairy" not in t.classify_name("margarine")                                # a person lowered it
    assert t.classify_name("spaghetti")["egg"] == "maybe"                             # a person raised it
    assert "egg" not in t.classify_name("spaghetti", rules_only=True)                 # rules_only ignores both layers


def test_rules_only_ignores_the_overlay(tmp_path):
    t = make_tax(tmp_path, overlay={"chourico": {"classes": {"pork": "yes"}}})
    assert "pork" not in t.classify_name("chourico", rules_only=True)


# ------------------------------------------------------------------ review queue and decisions
def test_review_roundtrip_applies_accepted_decisions_only(tmp_path):
    cmp = merge.Comparison(
        raises_review=[{"kind": "raise_maybe", "name": "spaghetti", "class": "egg", "n_lines": 578, "rules_say": "none",
                        "llm_says": "maybe", "llm_reason": "x"}],
        lowers=[{"kind": "lower_class", "name": "margarine", "class": "dairy", "n_lines": 1100, "rules_say": "maybe",
                 "llm_says": "none", "llm_reason": "y"}],
        not_ingredient=[{"kind": "not_ingredient", "name": "hot dog", "class": "", "n_lines": 50,
                         "rules_say": "ingredient", "llm_says": "not an ingredient", "llm_reason": "z"}])
    csv_path, decisions, overrides = tmp_path / "r.csv", tmp_path / "d.json", tmp_path / "o.yaml"
    assert merge.write_review_queue(csv_path, cmp, decisions, suggest=lambda r: "reject") == 3
    rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    rows[1] = rows[1].rstrip(",") if False else rows[1]
    import csv as _csv
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        data = list(_csv.DictReader(f))
    for r in data:                                                    # a person decides
        r["decision"] = {"spaghetti": "accept", "margarine": "reject", "hot dog": ""}[r["name"]]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=merge.REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(data)
    summary = merge.apply_review(csv_path, decisions, overrides, "tester")
    assert summary["accepted_total"] == 1 and summary["rejected_total"] == 1
    doc = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    assert doc["classes"] == {"spaghetti": {"egg": "maybe"}} and doc["not_ingredient"] == []
    # decided rows are not asked again; the undecided one (hot dog) still is
    left = merge.write_review_queue(tmp_path / "again.csv", cmp, decisions)
    assert left == 1


# ------------------------------------------------------------------ evaluation maths
GOLD = {
    "a": {"is_ingredient": True, "parent": "a", "classes": {"dairy": "yes"}, "group": "t"},
    "b": {"is_ingredient": True, "parent": "b", "classes": {"egg": "maybe"}, "group": "t"},
    "c": {"is_ingredient": True, "parent": "c", "classes": {}, "group": "t"},
    "d": {"is_ingredient": False, "parent": "d", "classes": {}, "group": "t"},
}


def test_score_counts_misses_false_alarms_and_exactness():
    preds = {"a": evaluate.Pred(True, "a", {"dairy": "maybe"}),            # flagged, but only maybe
             "b": evaluate.Pred(True, "x", {}),                            # missed + wrong parent
             "c": evaluate.Pred(True, "c", {"soy": "maybe"}),              # false alarm
             "d": evaluate.Pred(False, "d", {})}
    s = evaluate.score(preds, GOLD)
    assert s["recall_any_flag"] == 0.5 and s["recall_definite_flagged"] == 1.0 and s["recall_definite_exact"] == 0.0
    assert s["false_alarm_pairs"] == 1 and s["precision_of_flags"] == 0.5
    assert s["parent_accuracy"] == pytest.approx(2 / 3, abs=1e-3) and s["is_ingredient_accuracy"] == 1.0
    assert s["missed_any"] == [("b", "egg", "maybe")]


def test_gold_yaml_reads_the_word_yes_correctly(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("names:\n  x: {classes: {dairy: yes, egg: maybe, soy: no}}\n", encoding="utf-8")
    assert evaluate.load_gold(p)["x"]["classes"] == {"dairy": "yes", "egg": "maybe"}


def test_the_real_gold_file_is_well_formed():
    from recipebot.labeling.run import GOLD as GOLD_PATH
    gold = evaluate.load_gold(GOLD_PATH)
    assert len(gold) >= 150 and {"trap", "common", "mid", "tail"} <= {g["group"] for g in gold.values()}
    assert all(l in ("yes", "maybe") for g in gold.values() for l in g["classes"].values())


# ------------------------------------------------------------------ the free-tier-safe client
class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(text=step, usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4))


def client_with(tmp_path, script, **kw):
    c = GeminiClient(SecretStr("SECRETKEY123"), "fake-model", tmp_path / "cache", min_interval_s=0, **kw)
    c._client = SimpleNamespace(models=FakeModels(script))
    return c


GOOD = json.dumps([{"name": "x", "is_ingredient": True, "product": "x", "parent": "x", "form": "none", "classes": [], "reason": ""}])


def test_identical_requests_are_served_from_the_cache(tmp_path):
    c = client_with(tmp_path, [GOOD])
    first, cached1 = c.generate_json("sys", "u", list[NameLabel])
    second, cached2 = c.generate_json("sys", "u", list[NameLabel])
    assert (cached1, cached2) == (False, True) and first == second and c.calls == 1 and c.cache_hits == 1


def test_a_rejected_response_is_not_cached(tmp_path):
    c = client_with(tmp_path, ["[]", GOOD])
    _, _ = c.generate_json("sys", "u", list[NameLabel], accept=lambda p: bool(p))
    parsed, cached = c.generate_json("sys", "u", list[NameLabel], accept=lambda p: bool(p))
    assert cached is False and parsed and c.calls == 2                           # the bad answer was asked again


def test_call_cap_protects_the_daily_budget(tmp_path):
    c = client_with(tmp_path, [GOOD, GOOD], max_calls=1)
    c.generate_json("sys", "one", list[NameLabel])
    with pytest.raises(BudgetExceeded):
        c.generate_json("sys", "two", list[NameLabel])


def test_daily_quota_stops_cleanly_without_retrying_or_leaking_the_key(tmp_path):
    from google.genai import errors
    err = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                             "message": "Quota exceeded GenerateRequestsPerDayPerProjectPerModel-FreeTier key=SECRETKEY123"}})
    c = client_with(tmp_path, [err, GOOD])
    with pytest.raises(DailyQuotaExhausted) as e:
        c.generate_json("sys", "u", list[NameLabel])
    assert "SECRETKEY123" not in str(e.value) and c.calls == 1                    # no retry burned more quota


def test_other_api_errors_never_leak_the_key(tmp_path):
    from google.genai import errors
    err = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "bad key=SECRETKEY123"}})
    c = client_with(tmp_path, [err])
    with pytest.raises(LLMUnavailable) as e:
        c.generate_json("sys", "u", list[NameLabel])
    assert "SECRETKEY123" not in str(e.value)


def test_missing_key_is_a_clear_error(tmp_path):
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        GeminiClient(None, "m", tmp_path)


def test_the_key_is_not_visible_in_settings_repr():
    from recipebot.settings import get_settings
    assert "AIza" not in repr(get_settings()) and "**********" in repr(get_settings().gemini_api_key) or get_settings().gemini_api_key is None


# ------------------------------------------------------------------ labeler: incomplete answers are retried smaller
def test_labeler_retries_missing_names_in_smaller_batches_and_reports_unlabelled(tmp_path):
    names = [f"n{i}" for i in range(12)]

    def answer(batch):
        return json.dumps([{"name": n, "is_ingredient": True, "product": n, "parent": n, "form": "none", "classes": [], "reason": ""}
                           for n in batch])

    class Responder(FakeModels):
        """First call answers only 4 of the 12 requested names (an incomplete answer); later calls answer in full."""
        def generate_content(self, model, contents, config):
            self.calls += 1
            asked = json.loads(contents)
            return SimpleNamespace(text=answer(asked[:4] if self.calls == 1 else asked),
                                   usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1))

    c = client_with(tmp_path, [])
    c._client = SimpleNamespace(models=Responder([]))
    got = label_names(c, names, batch_size=12)
    # call 1 (12 asked, 4 answered, rejected) -> missing 8 retried as [5, 3] names -> 3 calls in total
    assert set(got) == set(names) and c.calls == 3


def test_labeler_drops_invented_names_and_unknown_classes(tmp_path):
    bad = json.dumps([{"name": "ghost", "is_ingredient": True, "product": "ghost", "parent": "ghost", "form": "none", "classes": [], "reason": ""}])
    c = client_with(tmp_path, [bad] * 6, max_calls=6)
    got = label_names(c, ["real"], batch_size=5)
    assert got == {}                                                               # never accepts a label for an unasked name


def test_labels_roundtrip_through_storage(tmp_path):
    p = tmp_path / "labels.jsonl"
    save_labels(p, {"x": lab("x", {"dairy": "yes"})}, "m")
    save_labels(p, {"y": lab("y")}, "m")
    got = load_labels(p)
    assert set(got) == {"x", "y"} and got["x"].class_map() == {"dairy": "yes"}
