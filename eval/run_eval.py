"""Evaluation harness.   uv run python eval/run_eval.py <command>

  det        deterministic cases (eval/cases.yaml): no LLM calls, same result every run
  audit      restriction-leak audit: many queries x restriction classes, checked with INDEPENDENT raw-text regexes
  conv       real test conversations with the real Gemini planner (eval/conversations.yaml), quota-aware and cached
  baseline   the same questions to a one-shot LLM with no tools, to show where ungrounded answers go wrong
  report     assemble eval/report.md from the JSON results above
Results are written to eval/results/*.json so the report can be rebuilt without re-running anything.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import independent as ind  # noqa: E402

from recipebot.domain.models import SearchSpec  # noqa: E402
from recipebot.llm.gemini import LLMUnavailable  # noqa: E402
from recipebot.orchestrator.bot import Chatbot  # noqa: E402
from recipebot.orchestrator.plan import TurnPlan  # noqa: E402
from recipebot.store.db import get_db  # noqa: E402
from recipebot.tools.search import search_recipes  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def save(name: str, data) -> None:
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def raw_lines(db, rid: str) -> list[str]:
    return [r[0] or "" for r in db.query("SELECT coalesce(text_raw, text) FROM recipe_ingredients WHERE recipe_id=? ORDER BY position", [rid])]


# ------------------------------------------------------------------------------------------------ deterministic cases
class ScriptedLLM:
    """Returns the scripted plan for a message; refuses narration (so answers are exactly the grounded text)."""

    def __init__(self, plans: dict[str, dict]):
        self.plans = plans

    def generate_json(self, system, user, schema, max_output_tokens=0, accept=None):
        if schema.__name__ != "TurnPlan":
            raise LLMUnavailable("narration disabled in deterministic evaluation")
        msg = user.split("USER MESSAGE: ", 1)[1]
        return TurnPlan(**self.plans[msg]).model_dump(), False


def run_checks(case, result, db):
    out = []
    for chk in case.get("checks", []):
        (name, arg), = (chk.items() if isinstance(chk, dict) else [(chk, None)])
        ok, detail = CHECKS[name](arg, result, db)
        out.append({"check": name, "arg": arg, "ok": ok, "detail": detail})
    return out


def _ids(result):
    return [s.recipe_id for s in result.sources]


def c_status(arg, r, db):
    return r.status == arg, f"status={r.status}"


def c_min_sources(arg, r, db):
    return len(r.sources) >= arg, f"{len(r.sources)} recipes shown"


def c_adapted_view(arg, r, db):
    """The adapted recipe is a source with its own sheet (ingredients with status, revised steps), and the FULL text is kept for audit."""
    v = r.sources[0].view if r.sources else None
    ok = bool(v and v["ingredients"] and v["instructions"] and r.sources[0].kind == "adapted"
              and "Updated ingredient list" in r.grounded_answer and "Revised instructions" in r.grounded_answer)
    return ok, "adapted sheet present" if ok else "no adapted sheet on the first source"


def c_no_sources(arg, r, db):
    return len(r.sources) == 0, f"{len(r.sources)} recipes shown"


def c_text_contains(arg, r, db):
    return arg.lower() in r.answer.lower(), f"looking for {arg!r}"


def c_text_regex(arg, r, db):
    m = re.search(arg, r.answer)
    return bool(m), (m.group(0) if m else f"no match for /{arg}/")


def c_no_violations(arg, r, db):
    bad = []
    for s in r.sources:
        v = ind.violations(raw_lines(db, s.recipe_id), arg)
        if v:
            bad.append(f"{s.title}: {v[0][:70]}")
    return not bad, (f"{len(bad)} of {len(r.sources)} recipes break '{arg}' by raw text: {bad[:3]}" if bad else f"0 of {len(r.sources)} recipes break '{arg}'")


def c_steps_at_most(arg, r, db):
    steps = [db.scalar("SELECT n_steps_sentences FROM recipes WHERE id=?", [i]) for i in _ids(r)]
    return all(s <= arg for s in steps), f"steps={steps}"


def c_steps_ascending(arg, r, db):
    steps = [db.scalar("SELECT n_steps_sentences FROM recipes WHERE id=?", [i]) for i in _ids(r)]
    return steps == sorted(steps), f"steps={steps}"


def c_active_filter(arg, r, db):
    v = r.active_filters.get(arg["key"])
    ok = (arg["value"] in v) if isinstance(v, list) else (v == arg["value"])
    return ok, f"{arg['key']}={v}"


def c_cited_ids_exist(arg, r, db):
    missing = [i for i in _ids(r) if db.scalar("SELECT count(*) FROM recipes WHERE id=?", [i]) != 1]
    return not missing, f"{len(_ids(r))} cited, {len(missing)} not in the dataset"


def c_note_contains(arg, r, db):
    notes = " ".join((r.computation or {}).get("notes", []))
    return arg.lower() in notes.lower(), f"looking for {arg!r} in the calculation notes"


def c_has_calculation(arg, r, db):
    c = r.computation or {}
    return bool(c.get("sql") and c.get("method") and c.get("n_analyzed") is not None), f"computation keys: {sorted(c)[:6]}"


def c_denominator_is_complete_recipes(arg, r, db):
    gold = db.scalar("SELECT count(*) FROM recipes WHERE is_complete")
    got = (r.computation or {}).get("denominator")
    return got == gold, f"denominator {got}, independent SQL says {gold}"


def c_count_close(arg, r, db):
    conds = " AND ".join(
        f"bool_or(regexp_matches(lower(coalesce(text_raw,text)), '\\b{w}s?\\b'))" for w in arg["words"])
    gold = db.scalar(f"SELECT count(*) FROM (SELECT recipe_id FROM recipe_ingredients WHERE recipe_id IN "
                     f"(SELECT id FROM recipes WHERE is_complete) GROUP BY recipe_id HAVING {conds})")
    m = re.search(r"^(\d+) of", r.answer, re.M)
    ours = int(m.group(1)) if m else -1
    ratio = ours / gold if gold else 0
    return arg["low"] <= ratio <= arg["high"], f"ours {ours}, plain-text count {gold}, ratio {ratio:.2f} (allowed {arg['low']}-{arg['high']})"


def c_alternatives_labelled(arg, r, db):
    if not r.sources:
        return True, "no recipes shown at all"
    return "BREAKS a constraint" in r.answer and "Closest alternatives" in r.answer, f"{len(r.sources)} near-misses shown"


def c_no_safety_claim(arg, r, db):
    bad = re.search(r"\b(100% safe|is safe|are safe|guarantee|allergy[- ]free|hypoallergenic)\b", r.answer, re.I)
    return not bad, "no safety claim" if not bad else f"found {bad.group(0)!r}"


def c_dairy_claim_consistent(arg, r, db):
    if "Text check for dairy: still present" in r.answer:
        ok = bool(re.search(r"Partly adapted|stays in the recipe|Nothing applied|No supported substitute|nothing applied", r.answer))
        return ok, "dairy still present and the answer says so" if ok else "dairy still present but the answer sounds finished"
    return True, "no dairy term left in the text, or none claimed"


def c_no_word(arg, r, db):
    """Independent: no shown recipe has a raw ingredient line containing any of these words (a named allergen such as grapes)."""
    words = arg if isinstance(arg, list) else [arg]
    bad = []
    for s in r.sources:
        v = ind.words_in_lines(raw_lines(db, s.recipe_id), words)
        if v:
            bad.append(f"{s.title}: {v[0][:60]}")
    return not bad, (f"{len(bad)} of {len(r.sources)} recipes contain {words}: {bad[:3]}" if bad else f"0 of {len(r.sources)} recipes contain {words}")


def c_answer_lacks(arg, r, db):
    return arg.lower() not in r.answer.lower(), f"{arg!r} must not appear"


def c_suggest(arg, r, db):
    return r.suggest == arg, f"suggest={r.suggest}"


def c_filter_absent(arg, r, db):
    v = r.active_filters.get(arg["key"])
    if arg.get("value") is None:
        return v in (None, [], ""), f"{arg['key']}={v}"
    return (arg["value"] not in v) if isinstance(v, list) else (v != arg["value"]), f"{arg['key']}={v}"


def c_trace_has(arg, r, db):
    node = r.trace
    for k in arg["path"].split("."):
        node = node.get(k) if isinstance(node, dict) else None
    ok = node == arg["value"] if "value" in arg else node is not None
    return ok, f"trace[{arg['path']}]={node}"


def c_opened_ref(arg, r, db):
    """The recipe opened by 'the Nth one' really is the Nth recipe of the previous list."""
    prev = getattr(r, "prev", None)
    ids = _ids(prev) if prev else []
    ok = len(ids) >= arg and bool(r.sources) and r.sources[0].recipe_id == ids[arg - 1]
    return ok, f"opened {r.sources[0].recipe_id if r.sources else None}, list item {arg} was {ids[arg - 1] if len(ids) >= arg else None}"


def c_same_recipe_as_previous(arg, r, db):
    prev = getattr(r, "prev", None)
    ok = bool(prev and prev.sources and r.sources) and r.sources[0].recipe_id == prev.sources[0].recipe_id
    return ok, f"{r.sources[0].recipe_id if r.sources else None} vs previous {prev.sources[0].recipe_id if prev and prev.sources else None}"


def c_web_labelled(arg, r, db):
    ok = bool(r.web) and "NOT from the dataset" in r.grounded_answer and all(w["url"].startswith("http") for w in r.web)
    return ok, f"{len(r.web)} web hits"


def c_no_web(arg, r, db):
    return not r.web, f"{len(r.web)} web hits"


def c_removed_line(arg, r, db):
    v = r.sources[0].view if r.sources else None
    hit = [i["text"] for i in (v or {}).get("ingredients", []) if i["status"] == "removed" and arg.lower() in i["text"].lower()]
    return bool(hit), f"removed lines: {hit}"


def _lead_value(line: str):
    """Independent leading-amount parser (does not use recipebot.tools.amounts): '1 1/2 cups' -> 1.5; None for ranges and sizes."""
    if re.match(r"^\s*\d+\s*(?:-|to)\s*\d", line) or re.match(r"^\s*\d+(?:\.\d+)?-[A-Za-z]", line):
        return None
    m = re.match(r"^\s*(\d+)\s+(\d+)/(\d+)\b", line)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.match(r"^\s*(\d+)/(\d+)\b", line)
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\b", line)
    return float(m.group(1)) if m else None


def c_scale_independent(arg, r, db):
    """Independent arithmetic: each ingredient amount in the answer equals the stored amount times the factor (rounded to a cooking fraction)."""
    if not r.sources:
        return False, "no recipe"
    stored = [x[0] for x in db.query("SELECT text FROM recipe_ingredients WHERE recipe_id=? ORDER BY position", [r.sources[0].recipe_id])]
    block = r.grounded_answer.split("Ingredients (changed amounts in bold):", 1)[-1].split("Instructions (unchanged):", 1)[0]
    shown = [re.sub(r"^\s*-\s*|\*\*", "", ln) for ln in block.splitlines() if ln.strip().startswith("-")]
    bad, checked = [], 0
    for i, line in enumerate(stored[:len(shown)]):
        base, got = _lead_value(line), _lead_value(shown[i])
        if base is None:
            continue
        want = base * arg["factor"]
        checked += 1
        tol = max(0.07 * want, 0.51 if want >= 10 else 0.07)
        if got is None or abs(got - want) > tol:
            bad.append(f"{line!r} x{arg['factor']} -> {shown[i]!r} (wanted about {want:g})")
    return checked > 0 and not bad, (f"{checked} amounts checked, {len(bad)} wrong: {bad[:2]}" if bad else f"{checked} amounts checked, all correct")


_TASTE_WORDS = {
    "spice": r"chil[ie]s?|chilli|cayenne|jalape|serrano|habanero|chipotle|pepper flakes|hot sauce|sriracha|tabasco|harissa|(?:black|white|ground|cracked) pepper|peppercorn|pepper,? to taste",
    "salt": r"\bsalt\b|soy sauce|fish sauce|tamari|worcestershire",
    "sweet": r"\bsugar\b|honey|syrup|molasses|agave|jaggery",
}
_TASTE_NOT = r"unsalted|salted|saltine|sugar snap|bell pepper|sugar[- ]free|low[- ]sodium"


def c_taste_lines_match(arg, r, db):
    """Independent: every line the taste box changes names a food of that axis (precision), and every such line of the recipe is changed (recall)."""
    body = r.grounded_answer.split("Adjusted to your taste", 1)[-1] if "Adjusted to your taste" in r.grounded_answer else ""
    changed = [re.sub(r"^\s*-\s*", "", ln).split(": Reduce")[0].split(": Increase")[0] for ln in body.splitlines()
               if ln.strip().startswith("-") and ("Reduce by" in ln or "Increase by" in ln)]
    axes = arg["axes"]
    union = "|".join(_TASTE_WORDS[a] for a in axes)
    wrong = [c for c in changed if not re.search(union, c.lower())]
    ids = _ids(r)
    stored = [x[0] for x in db.query("SELECT text FROM recipe_ingredients WHERE recipe_id=? ORDER BY position", [ids[0]])] if ids else []
    expected = [ln for ln in stored if re.search(union, ln.lower()) and not re.search(_TASTE_NOT, ln.lower())]
    missed = [e for e in expected if not any(e.lower().startswith(c.lower()[:12]) or c.lower() in e.lower() or e.lower() in c.lower() for c in changed)]
    ok = not wrong and len(missed) <= arg.get("allow_missed", 0)
    return ok, f"changed {len(changed)} lines ({len(wrong)} not of axis {axes}: {wrong[:2]}); {len(missed)} axis lines missed: {missed[:2]}"


def c_status_in(arg, r, db):
    return r.status in arg, f"status={r.status}"


def c_veg_blocked(arg, r, db):
    ok = r.status == "needs_input" and r.suggest == "turn_off_vegetarian" and not r.sources and "Turn the switch off" in r.answer
    return ok, f"status={r.status} suggest={r.suggest} sources={len(r.sources)}"


def c_source_count_between(arg, r, db):
    return arg[0] <= len(r.sources) <= arg[1], f"{len(r.sources)} recipes shown"


def c_titles_contain(arg, r, db):
    """Relevance: at least `min` of the shown recipes have the word in the title (independent of retrieval)."""
    n = sum(1 for s in r.sources if s.title and re.search(rf"\b{re.escape(arg['word'])}", s.title.lower()))
    return n >= arg.get("min", 1), f"{n} of {len(r.sources)} titles contain {arg['word']!r}"


CHECKS = {k[2:]: v for k, v in globals().items() if k.startswith("c_")}


class FakeWeb:
    """A canned web searcher so the deterministic evaluation never touches the network."""

    def search(self, query, max_results=3):
        from recipebot.tools.web import WebHit
        return [WebHit("Egg replacers for baking", "https://example.org/egg-replacers", "Applesauce or flax can replace egg.")]


def load_cases():
    out = []
    for name in ("cases.yaml", "cases_extended.yaml"):
        p = HERE / name
        if p.exists():
            out += yaml.safe_load(p.read_text(encoding="utf-8")) or []
    return out


def cmd_det():
    from recipebot.tools import web as web_tools
    db = get_db()
    report = []
    for case in load_cases():
        web_tools.set_searcher(FakeWeb() if case.get("web") == "fake" else None)      # never the real network
        plans = {t["say"]: t["plan"] for t in case["turns"]}
        bot = Chatbot(ScriptedLLM(plans), narrate=False, search_mode="hybrid")
        result, prev, checks = None, None, []
        for n, t in enumerate(case["turns"], start=1):
            veg = t.get("vegetarian", case.get("vegetarian"))
            result = bot.turn(case["id"], t["say"], vegetarian=veg, taste=t.get("taste", case.get("taste")))
            result.prev = prev
            for c in run_checks({"checks": t.get("checks", [])}, result, db):
                checks.append({**c, "turn": n})
            prev = result
        for c in run_checks(case, result, db):
            checks.append({**c, "turn": len(case["turns"])})
        passed = all(c["ok"] for c in checks)
        report.append({"id": case["id"], "task": case["task"], "category": case.get("category", ""), "why": case.get("why", ""),
                       "turns": len(case["turns"]), "passed": passed, "checks": checks, "answer_excerpt": result.answer[:600]})
        print(("PASS " if passed else "FAIL ") + case["id"])
        for c in checks:
            if not c["ok"]:
                print("     ", f"turn {c['turn']}", c["check"], "->", c["detail"])
    web_tools.set_searcher(None)
    n = sum(r["passed"] for r in report)
    print(f"\n{n}/{len(report)} cases passed ({sum(len(r['checks']) for r in report)} checks)")
    save("det", report)


# ------------------------------------------------------------------------------------------------ leak audit
QUERIES = ["pasta", "soup", "salad", "cookies", "cake", "bread", "stir fry", "curry", "casserole", "pie", "sandwich", "tacos",
           "pizza", "rice", "noodles", "breakfast", "muffins", "pancakes", "stew", "dip", "sauce", "chili", "burger", "roast",
           "lasagna", "brownies", "dinner", "appetizer", "dessert", "chicken"]
AUDIT_CLASSES = {"vegetarian": "vegetarian", "dairy": "dairy", "egg": "egg", "peanut": "peanut", "shellfish": "shellfish", "pork": "pork",
                 "tree_nut": "tree_nut", "gluten": "gluten", "veg_mode": "vegetarian"}   # veg_mode = the UI switch: no meat/fish AND no definite egg


def cmd_audit(k: int = 25):
    db = get_db()
    out = {"k": k, "queries": QUERIES, "classes": {}}
    for label, cls in AUDIT_CLASSES.items():
        checked = leaks = 0
        examples = []
        for q in QUERIES:
            spec = (SearchSpec(query=q, exclude_classes=["vegetarian"], exclude_classes_definite=["egg"], limit=k, mode="hybrid")
                    if label == "veg_mode" else SearchSpec(query=q, exclude_classes=[cls], limit=k, mode="hybrid"))
            res = search_recipes(spec, db)
            for h in res.recipes:
                checked += 1
                lines = raw_lines(db, h.id)
                bad = ind.violations(lines, "vegetarian") + ind.violations(lines, "egg") if label == "veg_mode" else ind.violations(lines, label)
                if bad:
                    leaks += 1
                    examples.append({"query": q, "id": h.id, "title": h.title, "line": bad[0],
                                     "warnings": [w.restriction for w in h.warnings]})
        out["classes"][label] = {"recipes_checked": checked, "leaks": leaks, "rate": round(leaks / checked, 4) if checked else None,
                                 "examples": examples}
        print(f"{label:11} checked {checked:4}  independent-text leaks {leaks:3}  ({100 * leaks / max(checked, 1):.1f}%)")
    save("audit", out)


# ------------------------------------------------------------------------------------------------ conversations (real LLM)
def cmd_conv():
    from recipebot.orchestrator.bot import build_chatbot
    db = get_db()
    convs = yaml.safe_load((HERE / "conversations.yaml").read_text(encoding="utf-8"))
    from recipebot.tools import web as web_tools
    web_tools.set_searcher(None)             # conversations are scored without the network; web search has its own cases
    bot = build_chatbot()      # the configuration that ships: one model call per turn, no LLM lead-in
    out = []
    for conv in convs:
        turns, prev = [], None
        for t in conv["turns"]:
            r = bot.turn(conv["id"], t["say"], vegetarian=t.get("vegetarian", conv.get("vegetarian")),
                         taste=t.get("taste", conv.get("taste")))
            r.prev, prev = prev, r
            checks = []
            for chk in t.get("checks", []):
                (name, arg), = (chk.items() if isinstance(chk, dict) else [(chk, None)])
                ok, detail = CHECKS[name](arg, r, db)
                checks.append({"check": name, "arg": arg, "ok": ok, "detail": detail})
            turns.append({"say": t["say"], "intent": r.intent, "status": r.status, "plan": r.trace.get("plan"),
                          "planner": r.trace.get("planner"), "narration": r.trace.get("narration"),
                          "guard_added": r.trace.get("restrictions_added_by_guard"), "allergen_guard": r.trace.get("allergen_guard"),
                          "planner_seconds": r.trace.get("planner_seconds"), "active_filters": r.active_filters,
                          "answer": r.answer, "sources": [{"n": s.n, "id": s.recipe_id, "title": s.title} for s in r.sources],
                          "seconds": r.trace.get("seconds"), "checks": checks, "passed": all(c["ok"] for c in checks)})
            print(("PASS " if turns[-1]["passed"] else "FAIL "), conv["id"], "|", t["say"][:60], "|", turns[-1]["intent"], turns[-1]["status"])
            for c in checks:
                if not c["ok"]:
                    print("      ", c["check"], "->", c["detail"])
        out.append({"id": conv["id"], "title": conv["title"], "turns": turns})
    save("conversations", {"stats": bot.llm.stats() if bot.llm else None, "conversations": out})
    print("LLM:", bot.llm.stats() if bot.llm else "none")


# ------------------------------------------------------------------------------------------------ one-shot baseline
BASELINE_SYSTEM = ("You are a helpful recipe assistant. The user is asking about the 'Recipe Box' dataset of about 125,000 recipes "
                   "(Allrecipes, Epicurious, Food Network). Answer the question as well as you can.")


def cmd_baseline():
    from pydantic import BaseModel

    from recipebot.llm.gemini import GeminiClient
    from recipebot.settings import get_settings

    class Ans(BaseModel):
        answer: str

    s = get_settings()
    llm = GeminiClient(s.gemini_api_key, s.gemini_model, s.llm_cache_dir, max_calls=20, min_interval_s=4.0)
    db = get_db()
    qs = {
        "count": "How many recipes in the dataset use both garlic and onion? Give the exact number and the percentage.",
        "ids": "List 5 penne pasta recipes from the dataset that contain no dairy. Give each recipe's title and its recipe ID in the dataset.",
        "calories": "How many calories are in the dataset's banana bread recipe?",
        "match": "I have eggs, onions, tomatoes and rice. List 5 recipes from the dataset that use them, with recipe IDs, using at most 6 steps and no peanuts.",
    }
    out = {}
    for key, q in qs.items():
        raw, _ = llm.generate_json(BASELINE_SYSTEM, q, Ans, 1500)
        out[key] = {"question": q, "answer": raw["answer"]}
        print("\n###", key, "\n", raw["answer"][:700])
    # ---- score what can be scored
    gold_count = db.scalar("SELECT count(*) FROM (SELECT recipe_id FROM recipe_ingredients WHERE recipe_id IN (SELECT id FROM recipes WHERE is_complete) "
                           "GROUP BY recipe_id HAVING bool_or(regexp_matches(lower(coalesce(text_raw,text)),'\\bgarlic\\b')) "
                           "AND bool_or(regexp_matches(lower(coalesce(text_raw,text)),'\\bonions?\\b')))")
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]{2,}", out["count"]["answer"])]
    out["count"]["gold_plain_text_count"] = gold_count
    out["count"]["numbers_in_answer"] = nums
    out["count"]["declined"] = bool(re.search(r"(do not|don't|cannot|can't|unable|no) (have )?(access|way|ability)|not able to", out["count"]["answer"], re.I))
    id_re = re.compile(r"(?:recipe )?ID\b:?\s*#?([A-Za-z0-9./]+)", re.I)
    for key in ("ids", "match"):
        ans = out[key]["answer"]
        ids = set(id_re.findall(ans)) | set(re.findall(r"\b[A-Za-z0-9./]{28,40}\b", ans))
        real = {i for i in ids if db.scalar("SELECT count(*) FROM recipes WHERE id=?", [i]) == 1}
        out[key]["ids_quoted"], out[key]["ids_real"] = len(ids), len(real)
        out[key]["id_lengths"] = sorted({len(i) for i in ids})
        titles = re.findall(r"\d+\.\s*(?:Recipe ID \d+:\s*)?([A-Z][^(:\n\-]{6,70}?)\s*(?:\(|:| -)", ans)
        found = [t for t in titles if db.scalar("SELECT count(*) FROM recipes WHERE lower(title)=lower(?)", [t.strip()]) > 0]
        out[key]["titles_quoted"], out[key]["titles_in_dataset"] = len(titles), len(found)
    cal = out["calories"]["answer"]
    out["calories"]["gave_a_number"] = bool(re.search(r"\d+\s*(kcal|calories|cal)\b", cal, re.I))
    out["calories"]["said_dataset_lacks_it"] = bool(re.search(r"(dataset|data).{0,50}(no|not|doesn't|does not|lack|without).{0,40}(nutrition|calor)|(no|not|doesn't|does not).{0,40}(nutrition|calor).{0,30}(dataset|data|field)", cal, re.I))
    save("baseline", {"stats": llm.stats(), "results": out})
    print("\nLLM:", llm.stats())


# ------------------------------------------------------------------------------------------------ report
def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def compute_metrics() -> dict:
    """Every number the project reports about itself, computed from the result files (nothing typed by hand)."""
    import statistics as st
    det, audit, conv, base = load("det"), load("audit"), load("conversations"), load("baseline")
    m: dict = {}
    if det:
        by_cat: dict = {}
        for c in det:
            d = by_cat.setdefault(c.get("category") or "core", {"cases": 0, "passed": 0})
            d["cases"] += 1
            d["passed"] += int(c["passed"])
        checks = [x for c in det for x in c["checks"]]
        by_check: dict = {}
        for x in checks:
            d = by_check.setdefault(x["check"], {"n": 0, "ok": 0})
            d["n"] += 1
            d["ok"] += int(x["ok"])
        m["deterministic"] = {"cases": len(det), "passed": sum(c["passed"] for c in det), "checks": len(checks),
                              "checks_passed": sum(x["ok"] for x in checks), "multi_turn_cases": sum(1 for c in det if c.get("turns", 1) > 1),
                              "by_category": by_cat, "by_check": by_check}
    if audit:
        m["leak_audit"] = {k: {"checked": v["recipes_checked"], "leaks": v["leaks"]} for k, v in audit["classes"].items()}
    if conv:
        turns = [t for c in conv["conversations"] for t in c["turns"]]
        live = [t for t in turns if (t.get("planner_seconds") or 0) >= 0.3]        # a cached replay takes ~0 s and would flatter the numbers
        secs = [t["seconds"] for t in live if t.get("seconds") is not None]
        psecs = [t["planner_seconds"] for t in live]
        m["conversations"] = {
            "conversations": len(conv["conversations"]), "turns": len(turns), "turns_passed": sum(t["passed"] for t in turns),
            "planner_was_llm": sum(1 for t in turns if t.get("planner") == "llm"),
            "allergen_guard_fired": sum(1 for t in turns if t.get("allergen_guard")),
            "other_guard_fired": sum(1 for t in turns if t.get("guard_added")),
            "uncached_turns": len(live),
            "turn_seconds": {"median": round(st.median(secs), 2), "p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 2), "max": max(secs)} if secs else None,
            "planner_seconds": {"median": round(st.median(psecs), 2), "max": max(psecs)} if psecs else None}
    lat = load("latency")
    if lat:
        m["latency"] = {k: v for k, v in lat.items() if k != "rows"}
    if base:
        r = base["results"]
        m["baseline"] = {"ids_quoted": r["ids"]["ids_quoted"] + r["match"]["ids_quoted"], "ids_real": r["ids"]["ids_real"] + r["match"]["ids_real"],
                         "calories_gave_number": r["calories"]["gave_a_number"], "said_no_nutrition": r["calories"]["said_dataset_lacks_it"]}
    return m


LATENCY_MESSAGES = ["show me eggplant parmesan recipes", "what percentage of recipes use ginger", "I have lentils, carrots and ginger",
                    "recipes with cinnamon and apples without nuts", "which 3 recipes with oats have the fewest steps", "how many recipes use turmeric",
                    "show me stuffed pepper recipes", "open the second one", "double the first one", "I'm allergic to kiwi",
                    "show me coconut curry recipes", "make the first one dairy-free", "what are the most common ingredients in cookie recipes",
                    "show me lemon chicken recipes"]


def cmd_latency():
    """Live latency of the shipping configuration: fresh messages (no cache), spaced like a person, one session. Costs ~14 model calls."""
    import statistics as st
    import time
    from recipebot.orchestrator.bot import build_chatbot
    from recipebot.tools import web as web_tools
    web_tools.set_searcher(None)
    bot = build_chatbot()
    bot.warm_up()
    rows = []
    for msg in LATENCY_MESSAGES:
        time.sleep(4.5)
        r = bot.turn("latency", msg)
        rows.append({"say": msg, "intent": r.intent, "seconds": r.trace["seconds"], "planner_seconds": r.trace.get("planner_seconds"),
                     "planner": r.trace.get("planner")})
        print(f"{r.trace['seconds']:6.2f}s  model {r.trace.get('planner_seconds')}s  {r.intent:12s} {msg}")
    secs = [x["seconds"] for x in rows]
    out = {"turns": len(rows), "median": round(st.median(secs), 2), "p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 2), "max": max(secs),
           "model_median": round(st.median([x["planner_seconds"] for x in rows if x["planner_seconds"]]), 2),
           "planner_was_llm": sum(1 for x in rows if x["planner"] == "llm"), "rows": rows}
    save("latency", out)
    print({k: v for k, v in out.items() if k != "rows"})


def cmd_metrics():
    m = compute_metrics()
    save("metrics", m)
    print(json.dumps(m, indent=2))


def metrics_markdown(m: dict) -> list[str]:
    L = ["## 0. Metrics at a glance", "",
         "Computed by `eval/run_eval.py metrics` from the result files. Higher is better unless stated.", ""]
    d = m.get("deterministic")
    if d:
        L += [f"- **Deterministic cases:** {d['passed']}/{d['cases']} pass ({_pct(d['passed'], d['cases'])}); {d['checks_passed']}/{d['checks']} individual checks; "
              f"{d['multi_turn_cases']} are multi-turn follow-up cases.", "",
              "| Category | Cases | Passed |", "|---|---|---|"]
        for k, v in sorted(d["by_category"].items()):
            L.append(f"| {k} | {v['cases']} | {v['passed']} |")
        L.append("")
        indep = {k: v for k, v in d["by_check"].items() if k in ("no_violations", "no_word", "scale_independent", "taste_lines_match", "count_close",
                                                                 "denominator_is_complete_recipes", "cited_ids_exist", "no_safety_claim", "steps_at_most", "steps_ascending")}
        L += ["| Independent check | Times run | Passed |", "|---|---|---|"] + [f"| {k} | {v['n']} | {v['ok']} |" for k, v in sorted(indep.items())] + [""]
    a = m.get("leak_audit")
    if a:
        tot = sum(v["checked"] for v in a.values())
        lk = sum(v["leaks"] for v in a.values())
        L += [f"- **Restriction leaks (audit, lower is better):** {lk} of {tot} recipes checked ({_pct(lk, tot)}), by class: " +
              ", ".join(f"{k} {v['leaks']}/{v['checked']}" for k, v in a.items()) + ".", ""]
    c = m.get("conversations")
    if c:
        L += [f"- **Real-planner conversations:** {c['turns_passed']}/{c['turns']} turns pass across {c['conversations']} conversations; "
              f"the model planner was used on {c['planner_was_llm']}/{c['turns']} turns (the rest fell back to rules).",
              f"- **Code guards firing on real planner output:** allergen guard {c['allergen_guard_fired']} turns, restriction/step guard {c['other_guard_fired']} turns.",
              ]
        if False:
            L.append(f"- **Latency per turn (seconds, lower is better; {c['uncached_turns']} turns with a live model call):** median {c['turn_seconds']['median']}, 90th percentile {c['turn_seconds']['p90']}, max {c['turn_seconds']['max']}"
                     + (f"; model call alone: median {c['planner_seconds']['median']}, max {c['planner_seconds']['max']}." if c["planner_seconds"] else "."))
        L.append("")
    lt = m.get("latency")
    if lt:
        L += [f"- **Latency per turn, live (seconds, lower is better):** median {lt['median']}, 90th percentile {lt['p90']}, max {lt['max']} over {lt['turns']} fresh messages "
              f"({lt['planner_was_llm']} planned by the model); the model call alone has median {lt['model_median']}. "
              "Before the phase 10 changes a turn took 5-15 s (two model calls, pacing, default reasoning).", ""]
    b = m.get("baseline")
    if b:
        L += [f"- **One-shot LLM baseline (for contrast):** {b['ids_real']} of {b['ids_quoted']} recipe IDs it quoted exist; "
              f"it {'did not say' if not b['said_no_nutrition'] else 'said'} the dataset lacks nutrition data.", ""]
    return L


def cmd_report():
    det, audit, conv, base = load("det"), load("audit"), load("conversations"), load("baseline")
    L = ["# Evaluation report", "",
         "Generated by `eval/run_eval.py report` from `eval/results/*.json`. Numbers in sections 0, 1, 2, 4 and 5 are produced by code from the result files; sections 3 and 6 are written by hand from those results.", ""]
    L += metrics_markdown(compute_metrics())
    if det:
        n = sum(c["passed"] for c in det)
        L += [f"## 1. Deterministic cases: {n}/{len(det)} passed", "",
              "No LLM is called: the plan a good planner would produce is scripted, so results are identical on every run. Checks are written "
              "independently of the system (raw-text regexes, direct SQL, independent arithmetic). Some cases script a WRONG plan on purpose "
              "(planner fault injection) to prove the code guards repair it.", "",
              "| Case | Category | Turns | Result | What was checked |", "|---|---|---|---|---|"]
        for c in det:
            L.append(f"| {c['id']} | {c.get('category') or 'core'} | {c.get('turns', 1)} | {'pass' if c['passed'] else '**FAIL**'} | " +
                     "; ".join(f"{x['check']}: {x['detail']}" for x in c["checks"])[:300].replace("|", "/") + " |")
        L.append("")
    if audit:
        L += ["## 2. Restriction-leak audit (independent text check)", "",
              f"{len(audit['queries'])} dish queries x {len(audit['classes'])} restriction classes, top {audit['k']} results each, strict mode."
              " A *leak* is a returned recipe "
              "whose RAW ingredient text contains a food of the excluded class by a simple regex that does not use our parser or labels. "
              "Some leaks are mistakes in the simple regex; they are triaged in section 3.", "",
              "| Restriction | Recipes checked | Leaks by independent text | Rate |", "|---|---|---|---|"]
        for k, v in audit["classes"].items():
            L.append(f"| {k} | {v['recipes_checked']} | {v['leaks']} | {100 * (v['rate'] or 0):.1f}% |")
        L.append("")
    before = load("audit_before_fixes")
    if audit and before:
        L += ["## 3. What the audit found, and what was fixed (written by hand from the audit output)", "",
              "| Restriction | Leaks before fixes | Leaks after fixes |", "|---|---|---|"]
        for k in audit["classes"]:
            if k in before["classes"]:
                L.append(f"| {k} | {before['classes'][k]['leaks']} | {audit['classes'][k]['leaks']} |")
        L += ["", "Triage of the 11 leaks found first (every one was read by a person):", "",
              "- **8 were mistakes in the independent regex, not in the system**: `celery ribs`, `kale ribs`, `steak sauce`, `coconut whipped cream`, "
              "`vegan cream cheese`, and an equipment line. The regex was tightened.",
              "- **3 were real misses**: recipes with `bacon bits` passed the vegetarian filter. Two causes, both in the taxonomy: "
              "(1) a human review decision on a base class (bacon bits: possible pork) was applied after the composite classes (meat, vegetarian) "
              "were derived, so they never saw it; (2) a guard phrase erased the match for `real bacon bits`.",
              "- **A related weakness found by hand while building the vegetarian switch**: any line containing the word `vegetarian` was exempt from the "
              "meat check, which cleared `chicken stock [for a vegetarian soup, use garlic broth]`. The exemption now covers only the phrase it modifies.",
              "- After the fixes one leak remains out of 3,000 recipes checked, and it is a false alarm of the regex (`hot dog buns` read as hot dogs).", ""]
    if conv:
        total = sum(len(c["turns"]) for c in conv["conversations"])
        ok = sum(t["passed"] for c in conv["conversations"] for t in c["turns"])
        L += [f"## 4. Real test conversations ({len(conv['conversations'])} conversations, {total} turns, {ok} turns pass their checks)", ""]
        for c in conv["conversations"]:
            L += [f"### {c['title']}", ""]
            for t in c["turns"]:
                L += [f"**User:** {t['say']}", "", f"*planner: {t['planner']}, intent: {t['intent']}, status: {t['status']}, "
                      f"filters: {t['active_filters']}, {'PASS' if t['passed'] else 'FAIL'}*", "", "> " + t["answer"][:900].replace("\n", "\n> "), ""]
    if base:
        r = base["results"]
        L += ["## 5. One-shot LLM baseline (same model, no tools)", "",
              "The same Gemini model answers the same questions without the dataset. This is the failure the grounded design prevents.", "",
              f"- **Count:** the baseline {'declined to give a number' if r['count']['declined'] else 'stated ' + str(r['count']['numbers_in_answer'])}; the grounded system computes it "
              f"(plain-text count over the dataset: {r['count']['gold_plain_text_count']}). A refusal is honest but gives the user nothing.",
              f"- **Recipe IDs (penne, no dairy):** it quoted {r['ids']['ids_quoted']} IDs (lengths {r['ids']['id_lengths']}; real IDs are 31 characters); "
              f"{r['ids']['ids_real']} exist in the dataset. Titles quoted: {r['ids']['titles_quoted']}, of which {r['ids']['titles_in_dataset']} exist as a title.",
              f"- **Matching (eggs, onions, tomatoes, rice):** it quoted {r['match']['ids_quoted']} IDs, {r['match']['ids_real']} exist; it also wrote step-by-step "
              f"instructions for recipes that are not in the dataset. Titles quoted: {r['match']['titles_quoted']}, of which {r['match']['titles_in_dataset']} exist as a title.",
              "- **Calories:** it " + ("gave a number" if r["calories"]["gave_a_number"] else "gave no number") + " and " +
              ("said the dataset lacks nutrition data." if r["calories"]["said_dataset_lacks_it"] else
               "did NOT say the dataset has no nutrition data (it evaded by saying it depends on the recipe)."), ""]
        for k, v in r.items():
            L += [f"<details><summary>Baseline answer: {k}</summary>", "", "> " + v["answer"][:1200].replace("\n", "\n> "), "", "</details>", ""]
    L += ["## 6. What worked, what failed, and why (written by hand)", "",
          "**Worked**", "",
          "- Hard constraints held: 0 real restriction leaks in 3,000 audited results after the fixes; the vegetarian switch and stated restrictions are applied in code.",
          "- Statistics are computed, not guessed, and state the filters, N, denominator and SQL; the count answer explains what counts as the ingredient.",
          "- Refusals: calories, cooking time and similar are refused with what the dataset does have; mixed requests answer the supported part.",
          "- Honest uncertainty: egg-free banana bread is reported as weak evidence, with nothing applied and the leads shown.", "",
          "**Failed, and how it was found and fixed**", "",
          "- A real conversation showed the planner dropping a stated hard constraint (`at most 6 steps`). A code guard now re-reads the message, like the one for restrictions.",
          "- A dish word was silently ignored by the count/rank tools (found by running the real model). They now route through search or ask the user.",
          "- Restriction leaks found by the independent audit (section 3). The unit tests had not seen them because they used small hand-made data.",
          "- Recipe IDs containing `/` failed to load in the UI; unit tests used IDs like `R2`.",
          "- Phase 10, found by the user in the UI: 'I'm allergic to grapes' came back from the planner as 'no egg' and grapes were not excluded. "
          "Two code guards now (a) drop any restriction class the user's words do not support and (b) always exclude the food the user names. "
          "While testing the guard the extended evaluation found a bug in the guard itself (an empty regex alternative made classes without "
          "synonyms always count as 'stated'); fixed and pinned by a test.",
          "- Phase 10, found by the extended evaluation: the planner dropped 'vegetarian' from a count follow-up (a wrong number was presented as the answer) "
          "and refused a whole cookie request because the user also asked to 'confirm they are safe'; the veg switch let 'Scrambled Eggs' through "
          "because the parser turned its line into a junk name flagged only as 'maybe egg'. All three are code fixes with regression cases.",
          "- Triage again: 22 of the first 23 'leaks' in the new audit classes were mistakes in my new regexes (pasta sauce, tapioca flour, water chestnuts); "
          "one was a real miss (the Scrambled Eggs line).", "",
          "**Still limited**", "",
          "- The evidence for substitutions is often weak (a swap that appears in 1 of 7 similar recipes is a lead, not proof), and amounts for a swap are copied from one similar recipe.",
          "- Role knowledge (what egg does in a cake) is curated for about 70 ingredients; the instruction rewriter is word replacement and needs a human read.",
          "- The audit regexes are simple. They are an independent check, not ground truth; the gold set for ingredient labels was written by the author and used to tune the rules (in-sample).",
          "- Web search for weak evidence exists but is scored only with a canned searcher (the real network is flaky and not reproducible); its labelling and its absence for supported verdicts are checked.",
          "- Taste adjustments are a labelled rule of thumb; the evaluation checks WHICH lines change and the arithmetic, not whether cooks like the result.",
          "- The real-planner set is small (about 50 turns) and uses one model; the deterministic set injects planner faults to cover the rest.", ""]
    (HERE / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote eval/report.md")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "det"
    {"det": cmd_det, "audit": cmd_audit, "conv": cmd_conv, "baseline": cmd_baseline, "report": cmd_report, "metrics": cmd_metrics, "latency": cmd_latency}[cmd]()
