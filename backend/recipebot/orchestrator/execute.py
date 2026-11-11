"""Turn a validated TurnPlan into tool calls and a deterministic, fully grounded answer.

Everything the user reads that is a fact (a title, an ID, a number, a verdict) is produced HERE from tool output.
The LLM may later rephrase this text (see `bot.py`), but the rephrasing is checked against it (see `ground.py`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from recipebot.domain.models import (Filters, MatchRequest, SearchSpec, SubstitutionRequest)
from recipebot.orchestrator.plan import TurnPlan
from recipebot.orchestrator.session import SessionState, ShownRecipe, merge_unique
from recipebot.tools import capabilities
from recipebot.tools import web as web_tools
from recipebot.tools.match import parse_available_text
from recipebot.tools.amounts import scale_lines
from recipebot.tools.taste import adjust_for_taste, box_to_text
from recipebot.tools.registry import (CountInput, GetRecipeInput, IngredientCountInput, RankInput, SimilarInput, TOOLS,
                                      TopIngredientsInput)

FILTER_INTENTS = {"search", "count", "top_ingredients", "ingredient_stats", "rank_steps", "match"}


@dataclass
class Source:
    n: int
    recipe_id: str
    title: str | None
    source: str
    cited: bool = True
    evidence: list[str] = field(default_factory=list)
    kind: str = "recipe"                 # "adapted": the row opens the ADAPTED recipe (see `view`), not the stored original
    view: dict | None = None             # the adapted recipe sheet: ingredients with status, revised steps, notes


class SourceBook:
    """Assigns the [n] numbers shown to the user. One number per recipe per turn."""

    def __init__(self) -> None:
        self.items: list[Source] = []

    def add(self, recipe_id: str, title: str | None, source: str = "", evidence: list[str] | None = None) -> int:
        for s in self.items:
            if s.recipe_id == recipe_id:
                s.evidence += [e for e in (evidence or []) if e not in s.evidence]
                return s.n
        self.items.append(Source(len(self.items) + 1, recipe_id, title, source, evidence=list(evidence or [])))
        return len(self.items)


@dataclass
class Outcome:
    intent: str
    text: str
    tool: str | None = None
    args: dict | None = None
    sources: list[Source] = field(default_factory=list)
    computation: dict | None = None
    status: str = "ok"                       # ok | refused | needs_input | no_results
    notes: list[str] = field(default_factory=list)
    shown: list[ShownRecipe] | None = None   # becomes the new "last results" when set
    web: list[dict] = field(default_factory=list)   # web pages found for weak evidence (untrusted; shown as links only)
    display: str | None = None            # a shorter text for the screen; `text` stays the full grounded answer
    suggest: str | None = None            # an action the UI offers as a button, e.g. "turn_off_vegetarian"


# --------------------------------------------------------------------------------------------- state diff
def apply_plan(state: SessionState, plan: TurnPlan) -> None:
    """Apply the filter part of the plan to the session. Only intents that search the dataset change the filters."""
    if plan.intent not in FILTER_INTENTS:
        return
    if plan.scope == "new":
        state.query, state.include, state.exclude, state.classes = None, [], [], []
        state.max_steps, state.mode, state.focus_id = None, "strict", None
        if plan.intent != "match":
            state.available_text = None
    if plan.query is not None:
        state.query = plan.query.strip() or None
    state.include = merge_unique(state.include, plan.add_include, [])
    state.exclude = merge_unique(state.exclude, plan.add_exclude, plan.remove_exclude)
    state.classes = merge_unique(state.classes, list(plan.add_exclude_classes), list(plan.remove_exclude_classes))
    if plan.clear_max_steps:
        state.max_steps = None
    if plan.max_steps is not None:
        state.max_steps = plan.max_steps
    if plan.lenient is not None:
        state.mode = "lenient" if plan.lenient else "strict"
    if plan.intent == "match" and plan.available_text:
        state.available_text = plan.available_text


def _describe(state: SessionState) -> str:
    f = state.active_filters()
    f.pop("restriction_mode", None)
    parts = []
    for k, v in f.items():
        if v is True:
            parts.append(k.replace("_", " "))
            continue
        val = ", ".join(map(str, v)) if isinstance(v, list) else v
        parts.append(f"{k.replace('_', ' ')}: {val}")
    return "; ".join(parts) or "no filters"


def _comp(result) -> dict | None:
    c = getattr(result, "computation", None)
    return c.model_dump() if c is not None else None


def _short(lines, n=2) -> str:
    return "; ".join(l.text for l in lines[:n])


# --------------------------------------------------------------------------------------------- per-intent runners
def _search(state: SessionState, plan: TurnPlan) -> Outcome:
    spec = SearchSpec(**state.filters().model_dump(), query=state.query, limit=plan.k or 5,
                      include_ingredient_lines=False, mode=state.search_mode)
    res = TOOLS["search_recipes"].handler(spec)
    book, shown = SourceBook(), []
    if res.total_matches == 0:
        txt = (f"No recipe matches ({_describe(state)}). I did not loosen any filter. "
               + (f"{res.excluded_only_by_maybe} recipe(s) were left out only because they MAY contain a restricted ingredient; "
                  "say so if you want those included. " if res.excluded_only_by_maybe else "")
               + ("I could not understand: " + ", ".join(res.unresolved_terms) + "." if res.unresolved_terms else ""))
        return Outcome("search", txt.strip(), "search_recipes", spec.model_dump(), [], _comp(res), "no_results")
    if state.search_mode != "lexical" and state.query:     # hybrid retrieval ranks a pool; it does not count matches
        out = [f"The {res.shown} most relevant recipes ({_describe(state)}), ranked by combined keyword and meaning similarity "
               "(this is a relevance ranking, not a count of all matches):"]
    else:
        out = [f"Found {res.total_matches} matching recipes ({_describe(state)}); showing {res.shown}:"]
    for r in res.recipes:
        n = book.add(r.id, r.title, r.source, [l.text for l in r.matched_ingredients[:3]])
        shown.append(ShownRecipe(n, r.id, r.title, r.source))
        line = f"{n}. {r.title} [{n}] ({r.n_steps_sentences} steps, {r.n_ingredients} ingredients, source {r.source})"
        if r.matched_ingredients:
            line += f" - matched: {_short(r.matched_ingredients)}"
        if r.warnings:
            line += " - uncertain: " + "; ".join(f"may contain {w.restriction}" for w in r.warnings)
        out.append(line)
    if res.excluded_only_by_maybe:
        out.append(f"({res.excluded_only_by_maybe} more recipes were excluded only because they MAY contain a restricted ingredient.)")
    if res.disclaimer:
        out.append(res.disclaimer)
    return Outcome("search", "\n".join(out), "search_recipes", spec.model_dump(), book.items, _comp(res), shown=shown)


def _count(state: SessionState, plan: TurnPlan) -> Outcome:
    if state.query:                                   # count_recipes has no text query: count through the search tool
        spec = SearchSpec(**state.filters().model_dump(), query=state.query, limit=1)
        res = TOOLS["search_recipes"].handler(spec)
        base = TOOLS["count_recipes"].handler(CountInput(filters=Filters()))
        pct = 100 * res.total_matches / base.denominator
        txt = (f"{res.total_matches} of {base.denominator} recipes ({pct:.1f}%) match ({_describe(state)}). "
               "The match for the dish words is a keyword match on title and ingredients.")
        return Outcome("count", txt, "search_recipes", spec.model_dump(), [], _comp(res))
    inp = CountInput(filters=state.filters())
    res = TOOLS["count_recipes"].handler(inp)
    txt = (f"{res.count} of {res.denominator} recipes ({res.percentage:.1f}%) match ({_describe(state)}). "
           f"For reference, that is {res.percentage_of_all:.1f}% of all {res.denominator_all} recipes.")
    return Outcome("count", txt, "count_recipes", inp.model_dump(), [], _comp(res))


def _top(state: SessionState, plan: TurnPlan) -> Outcome:
    if state.query:
        return Outcome("top_ingredients", f"I can compute common ingredients for an ingredient or restriction filter, but not for a "
                       f"dish-name search ('{state.query}'). Tell me an ingredient to filter on, or clear the search words.",
                       status="needs_input")
    inp = TopIngredientsInput(filters=state.filters(), k=plan.k or 10, level=plan.top_level)
    res = TOOLS["top_ingredients"].handler(inp)
    rows = [f"{i+1}. {x.ingredient}: {x.recipes} recipes ({x.percentage:.1f}%)" for i, x in enumerate(res.items)]
    return Outcome("top_ingredients", f"Most common ingredients ({_describe(state)}):\n" + "\n".join(rows),
                   "top_ingredients", inp.model_dump(), [], _comp(res))


def _ingredient_stats(state: SessionState, plan: TurnPlan) -> Outcome:
    if state.query:
        return Outcome("ingredient_stats", f"I can work out ingredient counts for an ingredient or restriction filter, but not for a "
                       f"dish-name search ('{state.query}'). Tell me an ingredient to filter on, or clear the search words.",
                       status="needs_input")
    inp = IngredientCountInput(filters=state.filters(), group_by="source" if plan.by_source else "none")
    res = TOOLS["ingredient_count_summary"].handler(inp)
    names = {"ar": "Allrecipes", "epi": "Epicurious", "fn": "Food Network"}
    rows = [f"- {names.get(g.group, g.group)}: {g.n} recipes, average {g.mean:g} ingredient lines, median {g.median:g}, "
            f"from {g.minimum:g} to {g.maximum:g} (90th percentile {g.p90:g})" for g in res.groups]
    return Outcome("ingredient_stats", f"Ingredient lines per recipe ({_describe(state)}):\n" + "\n".join(rows),
                   "ingredient_count_summary", inp.model_dump(), [], _comp(res))


def _rank(state: SessionState, plan: TurnPlan) -> Outcome:
    if state.query:
        if plan.order == "most":
            return Outcome("rank_steps", "I can rank a dish-name search by fewest steps, but not by most. Filter on an ingredient instead.",
                           status="needs_input")
        k = plan.k or 5
        pooled = state.search_mode != "lexical"      # hybrid retrieval ranks by relevance and ignores `sort`: rank a relevance pool in code
        spec = SearchSpec(**state.filters().model_dump(), query=state.query, limit=50 if pooled else k, sort="fewest_steps",
                          mode=state.search_mode)
        res = TOOLS["search_recipes"].handler(spec)
        recipes = sorted(res.recipes, key=lambda r: r.n_steps_sentences)[:k] if pooled else res.recipes    # stable: ties keep relevance order
        book, shown, rows = SourceBook(), [], []
        for r in recipes:
            n = book.add(r.id, r.title, r.source)
            shown.append(ShownRecipe(n, r.id, r.title, r.source))
            rows.append(f"{n}. {r.title} [{n}]: {r.n_steps_sentences} steps ({r.n_ingredients} ingredients)")
        head = (f"Of the {len(res.recipes)} most relevant recipes ({_describe(state)}; a relevance pool, not every match), the fewest steps:"
                if pooled else f"Of {res.total_matches} matching recipes ({_describe(state)}), the fewest steps:")
        txt = (head + "\n" + "\n".join(rows)
               + "\nSteps are counted as sentences; ties are ordered by relevance, then recipe ID.")
        return Outcome("rank_steps", txt, "search_recipes", spec.model_dump(), book.items, _comp(res),
                       "ok" if rows else "no_results", shown=shown)
    inp = RankInput(filters=state.filters(), k=plan.k or 5, order=plan.order)
    res = TOOLS["rank_by_steps"].handler(inp)
    book, shown, rows = SourceBook(), [], []
    for r in res.ranking:
        n = book.add(r.id, r.title, r.source)
        shown.append(ShownRecipe(n, r.id, r.title, r.source))
        rows.append(f"{n}. {r.title} [{n}]: {r.steps} steps ({r.n_ingredients} ingredients)")
    tie = (f"\nNote: {res.n_tied_at_cutoff} recipes tie at the cut-off; {res.tie_break}"
           if res.n_tied_at_cutoff > 1 else "")
    word = "fewest" if plan.order == "fewest" else "most"
    return Outcome("rank_steps", f"Recipes with the {word} steps ({_describe(state)}):\n" + "\n".join(rows) + tie,
                   "rank_by_steps", inp.model_dump(), book.items, _comp(res), shown=shown)


def _match(state: SessionState, plan: TurnPlan) -> Outcome:
    if not state.available_text:
        return Outcome("match", "Tell me which ingredients you have, for example: eggs, onions, tomatoes and rice.",
                       status="needs_input")
    avail = parse_available_text(state.available_text)
    if not avail:
        return Outcome("match", "I could not read any ingredients from that. Try: eggs, onions, tomatoes and rice.",
                       status="needs_input")
    req = MatchRequest(available=avail, constraints=state.filters(), k=plan.k or 5)
    res = TOOLS["match_recipes"].handler(req)
    book, shown, out = SourceBook(), [], []

    def line(m, prefix=""):
        n = book.add(m.id, m.title, m.source, [u.available for u in m.used])
        shown.append(ShownRecipe(n, m.id, m.title, m.source))
        miss = ", ".join(i.line.text for i in m.missing[:4]) or "nothing"
        return f"{prefix}{n}. {m.title} [{n}] ({m.steps} steps): uses {len(m.used)} of your {m.n_available}; still needed: {miss}. {m.why}"

    if res.status == "matches":
        out.append(f"{len(res.matches)} recipes fit your ingredients and constraints ({_describe(state)}):")
        out += [line(m) for m in res.matches]
    else:
        out.append(res.message)
        if res.closest_alternatives:
            out.append("Closest alternatives - each BREAKS a constraint, so none of these meet your request:")
            for a in res.closest_alternatives:
                out.append(line(a.recipe, "  ") + " Breaks: " + "; ".join(v.detail for v in a.violates))
    if res.ignored_available:
        out.append("Ignored because you excluded them: " + ", ".join(res.ignored_available) + ".")
    if res.unrecognised_ingredients:
        out.append("I did not recognise: " + ", ".join(res.unrecognised_ingredients) + ".")
    out.append(res.quantity_policy)
    return Outcome("match", "\n".join(out), "match_recipes", req.model_dump(), book.items, _comp(res),
                   "ok" if res.status == "matches" else "no_results", shown=shown)


def _substitute(state: SessionState, plan: TurnPlan) -> Outcome:
    rid, problem = state.resolve_recipe(plan.ref_index, plan.recipe_id)
    if problem:
        return Outcome("substitute", problem, status="needs_input")
    classes = list(plan.add_exclude_classes)
    if not plan.remove and not classes:
        return Outcome("substitute", "What should I remove or avoid in that recipe (for example egg, or dairy)?",
                       status="needs_input")
    req = SubstitutionRequest(recipe_id=rid, remove=list(plan.remove), exclude_classes=classes,
                              restriction_mode=state.mode)
    res = TOOLS["suggest_substitution"].handler(req)
    book = SourceBook()
    base = TOOLS["get_recipe"].handler(GetRecipeInput(recipe_id=res.recipe_id))
    n0 = book.add(res.recipe_id, res.recipe_title, base.source if base else "")
    out = [f"{n0}. {res.recipe_title} [{n0}] (the recipe being adapted)", res.summary]
    for t in res.targets:
        ev = []
        for c in t.candidates[:3]:
            ns = [book.add(e.id, e.title, "", [e.line]) for e in c.evidence[:3]]
            ev.append(f"{c.substitute} ({c.tier}, {c.n_supporting} similar recipes"
                      + (f"; {c.n_alternative_lines} list it as an alternative" if c.n_alternative_lines else "")
                      + ") " + "".join(f"[{n}]" for n in ns))
        out.append(f"- {t.ingredient}: evidence is {t.verdict.upper()}. {t.verdict_reason}"
                   + (f" Leads: {'; '.join(ev)}." if ev else "")
                   + (f" Chosen: {t.chosen}." if t.chosen else
                      " The line is removed, with no replacement applied." if "you asked to remove" in t.why else " Nothing applied."))
    leads = [s for s in book.items if s.n != n0]
    if leads:
        out.append("Recipes behind these swaps (open one to see how it is made):")
        for s in leads:
            out.append(f"{s.n}. {s.title} [{s.n}] (shows: {'; '.join(s.evidence[:2]) or 'a similar recipe'})")
    changed = [l for l in res.adapted_ingredients if l.changed]
    adapted = bool(changed or res.instruction_edits)
    if adapted:
        book.items[0].kind, book.items[0].view = "adapted", _adapted_view(res)
    head_end = len(out)               # the on-screen answer keeps everything before this; the lists live in the adapted recipe
    if adapted:
        out.append("Updated ingredient list (changed lines in bold):")
        for l in res.adapted_ingredients:
            if not l.changed:
                out.append(f"- {l.text}")
            elif not l.text.strip():
                out.append(f"- (removed: {l.original})")
            else:
                how = "tentative, weak evidence" if l.evidence_level == "weak" else "supported by similar recipes"
                out.append(f"- **{l.text}** (replaces: {l.original}; {how})")
        steps = [x.strip() for x in res.revised_instructions.splitlines() if x.strip()]
        out.append("Revised instructions:")
        out += [f"{i}. {x}" for i, x in enumerate(steps, 1)] if len(steps) > 1 else steps
        if res.instruction_edits:
            out.append("The instructions were edited by simple word replacement, please read them before cooking. Edited: "
                       + "; ".join(f"step {e.step} ({e.action})" for e in res.instruction_edits[:8]) + ".")
    else:
        out.append("No change was applied, so the recipe stays exactly as it is.")
    tail_start = len(out)
    out += [f"Predicted effect (a prediction, not verified): {e}" for e in res.predicted_effects[:2]]
    out += [f"Text check for {c.restriction}: {c.status.replace('_', ' ')}" for c in res.restriction_check]
    out += list(res.remaining_issues)
    web = _web_for_weak_evidence(res, out)
    out.append(res.disclaimer)
    state.focus_id = rid
    display = None
    if adapted:
        # What the user reads. The full text above (lists, steps, edits) stays as the grounded answer for audit and tests; on screen
        # the same facts are in the adapted recipe (ingredients, steps, "what was changed"), so they are not repeated here.
        tail = [l for l in out[tail_start:] if not l.startswith("- Web: ") and "removed WITHOUT a supported" not in l
                and not l.startswith("The dataset's evidence for") or "returned nothing" in l]
        display = "\n".join(out[:head_end] + ["Open the adapted recipe above for the full ingredient list, the revised steps and "
                                              "what was changed."] + tail)
    return Outcome("substitute", "\n".join(out), "suggest_substitution", req.model_dump(), book.items, _comp(res),
                   "ok" if res.status == "adapted" else "no_results", web=web, display=display)


def _adapted_view(res) -> dict:
    """The adapted recipe as the recipe window shows it: every ingredient line with its status, the revised steps, and notes."""
    ings = []
    for l in res.adapted_ingredients:
        if not l.changed:
            ings.append({"text": l.text, "original": None, "status": "same"})
        elif not l.text.strip():
            ings.append({"text": l.original or "", "original": None, "status": "removed"})
        else:
            ings.append({"text": l.text, "original": l.original, "status": "changed"})
    steps = [x.strip() for x in res.revised_instructions.splitlines() if x.strip()]
    notes = [f"{t.ingredient}: removed, no replacement applied (dataset evidence {t.verdict})." for t in res.targets
             if not t.chosen and "you asked to remove" in t.why]
    notes += [f"{t.ingredient}: replaced with {t.chosen} (dataset evidence {t.verdict})." for t in res.targets if t.chosen]
    if res.instruction_edits:
        notes.append("Instructions were edited by simple word replacement; read every step before cooking: "
                     + "; ".join(f"step {e.step} ({e.action})" for e in res.instruction_edits[:8]) + ".")
    notes += list(res.remaining_issues)
    return {"ingredients": ings, "instructions": steps, "notes": notes, "disclaimer": res.disclaimer}


def _web_for_weak_evidence(res, out: list[str]) -> list[dict]:
    """Where the dataset's evidence is weak, conflicting or missing, also search the web. Results are labelled as NOT
    from the dataset and are never used to pick or apply a substitute."""
    weak = [t for t in res.targets if t.verdict != "supported"]
    searcher = web_tools.get_searcher() if weak else None
    if not searcher:
        return []
    found: list[dict] = []
    for t in weak[:2]:
        hits = (searcher.search(f"{t.ingredient} substitute in {res.recipe_title}", 3)
                or searcher.search(f"best substitute for {t.ingredient} in cooking", 3))     # a dish title can be too specific
        if not hits:
            out.append(f"The dataset's evidence for {t.ingredient} is {t.verdict}, so I tried a web search, but it returned nothing.")
            continue
        out.append(f"The dataset's evidence for {t.ingredient} is {t.verdict}, so I also searched the web. These pages are "
                   "NOT from the dataset and I have not checked them; read them before relying on them:")
        for h in hits:
            out.append(f"- Web: {h.title} ({h.domain})")
            found.append({"ingredient": t.ingredient, **h.as_dict()})
    return found


def _scale(state: SessionState, plan: TurnPlan) -> Outcome:
    rid, problem = state.resolve_recipe(plan.ref_index, plan.recipe_id)
    if problem:
        return Outcome("scale", problem, status="needs_input")
    factor = plan.scale_factor
    if factor is None and plan.servings_from and plan.servings_to:
        factor = plan.servings_to / plan.servings_from
    if factor is None:
        return Outcome("scale", "The dataset does not say how many people a recipe serves, so I scale by a factor. Say how many it "
                       "serves now (for example 'it serves 4, make it for 6'), or give a factor such as double, half or 1.5 times.",
                       status="needs_input")
    if not 0.1 <= factor <= 20:
        return Outcome("scale", "Please pick a factor between 0.1 and 20 times.", status="needs_input")
    res = TOOLS["get_recipe"].handler(GetRecipeInput(recipe_id=rid))
    if res is None:
        return Outcome("scale", f"I have no recipe with ID {rid}.", status="no_results")
    book = SourceBook()
    n = book.add(res.id, res.title, res.source)
    scaled = scale_lines([i.text for i in res.ingredients], factor)
    sv = f" (from {plan.servings_from} to {plan.servings_to} servings)" if plan.servings_from and plan.servings_to else ""
    out = [f"{n}. {res.title} [{n}] (scaled by {factor:g}x{sv})", "Ingredients (changed amounts in bold):"]
    for s in scaled:
        out.append(f"- **{s.text}**" if s.scaled else f"- {s.text}")
    skipped = [s for s in scaled if not s.scaled and factor != 1]
    if skipped:
        out.append("Left as written, nothing to scale: " + "; ".join(s.original for s in skipped[:6]) + ".")
    out.append("Cooking times, pan sizes and any amounts written inside the instructions are NOT scaled. "
               + ("For a big change, season gradually and taste as you go." if factor >= 2 or factor <= 0.5 else
                  "Check the instructions for amounts before you cook."))
    box = adjust_for_taste([s.text for s in scaled], state.taste)
    out.append("Instructions (unchanged):")
    out.append(res.instructions)
    out += box_to_text(box)
    state.focus_id = res.id
    return Outcome("scale", "\n".join(out), "scale_recipe", {"recipe_id": rid, "factor": factor}, book.items)


def _get_recipe(state: SessionState, plan: TurnPlan) -> Outcome:
    rid, problem = state.resolve_recipe(plan.ref_index, plan.recipe_id)
    if problem:
        return Outcome("get_recipe", problem, status="needs_input")
    res = TOOLS["get_recipe"].handler(GetRecipeInput(recipe_id=rid))
    if res is None:
        return Outcome("get_recipe", f"I have no recipe with ID {rid}.", status="no_results")
    book = SourceBook()
    n = book.add(res.id, res.title, res.source)
    state.focus_id = res.id
    txt = (f"{n}. {res.title} [{n}] (source {res.source}, {res.n_steps_sentences} steps)\nIngredients:\n"
           + "\n".join(f"- {i.text}" for i in res.ingredients) + f"\nInstructions:\n{res.instructions}")
    box = box_to_text(adjust_for_taste([i.text for i in res.ingredients], state.taste))
    if box:
        txt += "\n" + "\n".join(box)
    return Outcome("get_recipe", txt, "get_recipe", {"recipe_id": rid}, book.items)


def _similar(state: SessionState, plan: TurnPlan) -> Outcome:
    rid, problem = state.resolve_recipe(plan.ref_index, plan.recipe_id)
    if problem:
        return Outcome("similar", problem, status="needs_input")
    inp = SimilarInput(recipe_id=rid, filters=state.filters(), k=plan.k or 5)
    res = TOOLS["similar_recipes"].handler(inp)
    book, shown, rows = SourceBook(), [], []
    for r in res.items:
        n = book.add(r.id, r.title, r.source)
        shown.append(ShownRecipe(n, r.id, r.title, r.source))
        rows.append(f"{n}. {r.title} [{n}] (similarity {r.similarity:.2f}; shares {len(r.shared_ingredients)} ingredients)")
    return Outcome("similar", f"Recipes most similar to {res.seed_title}:\n" + "\n".join(rows), "similar_recipes",
                   inp.model_dump(), book.items, _comp(res), shown=shown)


def _refusal(text: str) -> Outcome | None:
    r = capabilities.check_attribute(text)
    if r is None:
        return None
    body = (f"I can't answer that: the dataset has no {r.attribute} information. {r.reason}\n"
            f"What I can do instead: " + "; ".join(r.can_offer_instead) + ".")
    return Outcome("unsupported", body, "check_supported", {"request": text}, [], None, "refused")


RUNNERS = {"search": _search, "count": _count, "top_ingredients": _top, "ingredient_stats": _ingredient_stats, "rank_steps": _rank, "match": _match,
           "substitute": _substitute, "get_recipe": _get_recipe, "similar": _similar, "scale": _scale}

CHAT_TEXT = ("I answer questions about the Recipe Box dataset (about 125,000 recipes): find recipes, count them or work "
             "out percentages, list common ingredients, rank by number of steps, suggest recipes from the ingredients "
             "you have, and adapt a recipe (for example dairy-free) using what similar recipes do. Every answer shows "
             "the recipes it comes from. The dataset has no nutrition, cooking time, servings, cuisine, cost or rating.")


# the taxonomy knows meat, poultry, fish and shellfish; egg is listed by word so that dish names that merely CONTAIN egg ("cake") are not blocked
_EXTRA_NONVEG = {"egg", "eggs", "omelette", "omelet", "keema", "gosht", "murgh", "kebab", "kabab"}


def nonveg_words(text: str | None) -> list[str]:
    """Words of a request that name meat, poultry, fish, shellfish or egg (the taxonomy decides, plus a few Indian dish words)."""
    if not text:
        return []
    from recipebot.domain.ingredient import normalize_for_scan
    from recipebot.domain.taxonomy import get_taxonomy
    tax, found = get_taxonomy(), []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in found or len(w) < 3:
            continue
        cls = tax.classify_name(normalize_for_scan(w))
        if cls.get("vegetarian") == "yes" or w in _EXTRA_NONVEG:
            found.append(w)
    return found


def _veg_conflict(state: SessionState, plan: TurnPlan) -> Outcome | None:
    """With the Vegetarian switch on, a search that NAMES a non-vegetarian food would only return look-alikes or nothing.
    Say what is happening and offer to turn the switch off, instead of showing results that do not answer the request."""
    if not state.vegetarian or plan.intent not in {"search", "count", "rank_steps", "similar"}:
        return None
    words = nonveg_words(" ".join([plan.query or "", *plan.add_include]))
    if not words:
        return None
    return Outcome(plan.intent, f"Vegetarian only is on, and it leaves out meat, fish and egg, so I will not search for "
                   f"{', '.join(words)} while it is on. Turn the switch off to see these recipes.",
                   status="needs_input", suggest="turn_off_vegetarian")


def execute(state: SessionState, plan: TurnPlan, message: str) -> Outcome:
    if plan.intent == "unsupported":
        out = _refusal(plan.unsupported_text or message) or Outcome(
            "unsupported", "The dataset cannot answer that. It holds only titles, ingredient lines and instructions.",
            status="refused")
    elif plan.intent == "chat":
        out = Outcome("chat", CHAT_TEXT)
    elif plan.intent == "clarify":
        out = Outcome("clarify", plan.question or "Could you tell me a bit more about what you are looking for?",
                      status="needs_input")
    elif (blocked := _veg_conflict(state, plan)) is not None:
        out = blocked                          # Vegetarian only is on and the request needs meat, fish or egg: say so, do not search
    else:
        apply_plan(state, plan)
        out = RUNNERS[plan.intent](state, plan)
        # a request that mixes a supported part with an unsupported one: answer the part we can, and say so
        side = None if plan.intent == "scale" else capabilities.check_attribute(message)   # "for 6 servings" is a scale request
        if side is not None:
            out.notes.append(f"Not answered: the dataset has no {side.attribute} information. {side.reason}")
            out.text += f"\nNot answered: the dataset has no {side.attribute} information."
    if out.shown is not None:
        state.last_results = out.shown
        state.focus_id = None
    return out
