"""The conversation orchestrator.

One user turn =
  1. PLAN      LLM (or a rule-based fallback) turns the message + session state into a validated `TurnPlan`.
  2. GUARD     deterministic checks fix what the LLM may have missed (stated restrictions are never dropped).
  3. EXECUTE   plain code runs the tools; the grounded answer text, sources and computation come from tool output.
  4. NARRATE   the LLM may add a short lead-in; the facts stay the tool-derived text, and `ground.validate` rejects a lead-in that adds anything.
  5. TRACE     everything above is recorded (no secrets) so any answer can be audited.
The LLM never produces a number, an ID, or a recipe fact that reaches the user unchecked.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, get_args

from pydantic import BaseModel, ValidationError

from recipebot.domain.models import RestrictionClass
from recipebot.llm.gemini import BudgetExceeded, DailyQuotaExhausted, LLMUnavailable
from recipebot.orchestrator import ground
from recipebot.orchestrator.execute import Outcome, Source, execute
from recipebot.orchestrator.plan import TurnPlan
from recipebot.orchestrator.session import SessionState, Sessions
from recipebot.settings import ROOT, get_settings
from recipebot.tools.taste import clean_profile
from recipebot.retrieval.hybrid import StaleIndexError
from recipebot.store.db import StaleDataError

log = logging.getLogger("recipebot.bot")
PROMPTS = ROOT / "prompts"
LLM_ERRORS = (DailyQuotaExhausted, BudgetExceeded, LLMUnavailable)


class LLM(Protocol):
    def generate_json(self, system: str, user: str, schema, max_output_tokens: int = ..., accept=...) -> tuple[object, bool]: ...


class Narration(BaseModel):
    intro: str


@dataclass
class TurnResult:
    answer: str
    grounded_answer: str
    intent: str
    status: str
    sources: list[Source]
    computation: dict | None
    active_filters: dict
    notes: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    web: list[dict] = field(default_factory=list)      # untrusted web pages for weak evidence (links only)
    suggest: str | None = None                          # an action the UI offers as a button


# --------------------------------------------------------------------------------------------- guards
_NEG = r"(?:no|without|free of|free from|allergic to|allergy to|allergic|can'?t have|cannot have|can'?t eat|cannot eat|avoid|intolerant to|exclude|excluding|leave out|skip|not)"
_WORDS = {
    "dairy": r"dairy|milk|lactose|cheese|butter", "egg": r"eggs?", "peanut": r"peanuts?", "tree_nut": r"tree nuts?|nuts?",
    "gluten": r"gluten|wheat", "soy": r"soy|soya", "fish": r"fish", "shellfish": r"shellfish", "sesame": r"sesame",
    "pork": r"pork", "beef": r"beef", "vegetarian": r"vegetarian", "vegan": r"vegan", "alcohol": r"alcohol|wine|beer",
}
# words that name a class only through a specific ingredient ("no butter" is an ingredient filter, not the dairy class)
_INGREDIENT_ONLY = {"cheese", "butter"}


def detect_restrictions(message: str) -> list[str]:
    """Restriction classes the user stated with a negation ('no dairy', 'gluten-free', 'allergic to peanuts')."""
    t = message.lower()
    found = []
    for cls, words in _WORDS.items():
        names = [w for w in words.split("|") if w not in _INGREDIENT_ONLY]
        rx = "|".join(names)
        if (re.search(rf"\b(?:{rx})(?:\s|-)free\b", t) or re.search(rf"\b{_NEG}\b[^.,;!?]{{0,25}}?\b(?:{rx})\b", t)
                or re.search(rf"\b(?:{rx})\s+(?:allerg(?:y|ic)|intoleran(?:t|ce)|sensitiv(?:e|ity))\b", t)):
            found.append(cls)
    # "vegetarian" / "vegan" is an adjective that always means a filter when it is asked for ("how many of those are vegetarian?"),
    # unlike "dairy" ("with dairy" is not a restriction). "non-vegetarian" and "not vegetarian" are not.
    for cls in ("vegetarian", "vegan"):
        if cls not in found and re.search(rf"(?<![-\w]){cls}\b", t) and not re.search(rf"\b(?:non|not|isn'?t|aren'?t)[- ]{cls}\b", t):
            found.append(cls)
    return found


_RESTRICTED_INTENTS = {"search", "count", "top_ingredients", "ingredient_stats", "rank_steps", "match", "substitute", "similar"}
_ALLERGY = re.compile(r"\b(?:allergic to|allergy to|allergies to|intolerant to|intolerance to|can'?t (?:have|eat)|cannot (?:have|eat)|"
                      r"(?:do not|don'?t) eat|avoid|must not have|mustn'?t have)\s+([^.;!?]{1,70})", re.I)
_FILLER = {"any", "all", "the", "a", "an", "too", "also", "please", "fresh", "raw", "of", "my", "any kind of"}
_NOT_FOOD = {"more", "less", "than", "step", "steps", "time", "it", "them", "that", "those", "these", "this", "anything", "everything"}
# words a user may use for a class that the class name itself does not contain
_CLASS_SYNONYMS = {"dairy": r"lactose|milk|cheese|butter|cream|yogh?urt|paneer|ghee|curd", "gluten": r"celiac|coeliac|wheat|flour|bread|pasta",
                   "tree_nut": r"nuts?|almonds?|cashews?|walnuts?|pecans?|pistachios?|hazelnuts?", "alcohol": r"wine|beer|liquor|rum|whisk(?:e)?y|vodka",
                   "pork": r"bacon|ham|pig", "shellfish": r"prawns?|shrimps?|crabs?|lobsters?|oysters?|clams?|mussels?",
                   "vegetarian": r"veg|vegetarian", "vegan": r"vegan|plant[- ]based", "soy": r"soy|soya|tofu", "egg": r"eggs?|omelet+e?|mayo(?:nnaise)?"}


def stated_allergens(message: str) -> list[str]:
    """Foods the user said they are allergic to, intolerant to, or cannot eat ("allergic to grapes and nuts" -> grapes, nuts)."""
    found: list[str] = []
    for m in _ALLERGY.finditer(message):
        for part in re.split(r",|;|\band\b|\bor\b|&|\bbut\b", m.group(1).lower()):
            words = [w for w in re.findall(r"[a-z][a-z-]*", part) if w not in _FILLER]
            if not words or len(words) > 3 or any(w in _NOT_FOOD for w in words):
                continue
            term = " ".join(words)
            if term not in found:
                found.append(term)
    return found


def _classes_for_word(term: str) -> set[str]:
    from recipebot.domain.ingredient import normalize_for_scan
    from recipebot.domain.taxonomy import get_taxonomy
    cls = get_taxonomy().classify_name(normalize_for_scan(term))
    return {c for c, st in cls.items() if st == "yes" and c not in ("vegan", "vegetarian", "meat")} | (
        {"vegetarian"} if cls.get("vegetarian") == "yes" else set())


def class_is_stated(cls: str, message: str) -> bool:
    """Is a restriction class that the planner proposed actually supported by the user's own words? (A class is never guessed.)"""
    t = message.lower()
    words = "|".join(p for p in (_WORDS.get(cls, ""), _CLASS_SYNONYMS.get(cls, ""), cls.replace("_", " ")) if p)   # no empty alternative
    if re.search(rf"\b(?:{words})\b", t):
        return True
    return any(cls in _classes_for_word(w) for w in re.findall(r"[a-z]+", t) if len(w) > 2)


def enforce_allergens(plan: TurnPlan, message: str) -> dict:
    """Two code guards for the most safety-relevant part of a message (the planner is a model and can slip):
    1) a restriction CLASS the user's words do not support is dropped (found when "allergic to grapes" came back as "no egg");
    2) every food the user says they are allergic to is excluded: as its class when it belongs to one, else as an ingredient."""
    if plan.intent not in _RESTRICTED_INTENTS:
        return {}
    report: dict = {}
    dropped = [c for c in plan.add_exclude_classes if not class_is_stated(c, message)]
    if dropped:
        plan.add_exclude_classes[:] = [c for c in plan.add_exclude_classes if c not in dropped]
        report["classes_dropped"] = dropped
    added_cls, added_ing = [], []
    for term in stated_allergens(message):
        classes = _classes_for_word(term) - {"vegetarian"}
        for c in classes:
            if c not in plan.add_exclude_classes and c in get_args(RestrictionClass):
                plan.add_exclude_classes.append(c)  # type: ignore[arg-type]
                added_cls.append(c)
        if not classes and term.lower() not in {x.lower() for x in plan.add_exclude}:
            plan.add_exclude.append(term)
            added_ing.append(term)
    if added_cls:
        report["classes_added"] = added_cls
    if added_ing:
        report["ingredients_added"] = added_ing
    return report


_INGREDIENT_INTENTS = {"count", "top_ingredients", "ingredient_stats", "rank_steps"}


def normalize_ingredient_query(plan: TurnPlan) -> str | None:
    """'Which 5 recipes WITH TOMATO have the fewest steps?' must filter on the ingredient. The planner sometimes puts the word in
    `query` (a dish search) instead of `add_include`; for a single known ingredient word the code moves it. -> the word moved."""
    q = (plan.query or "").strip()
    if plan.intent not in _INGREDIENT_INTENTS or not q or plan.add_include or len(q.split()) != 1:
        return None
    from recipebot.store.db import get_db
    from recipebot.tools.terms import TermResolver
    if TermResolver(get_db()).resolve(q).strategy != "structured":
        return None
    plan.add_include.append(q)
    plan.query = None
    return q


_SAFETY_ASK = re.compile(r"\b(?:safe|safety|certify|guarantee|confirm)\b", re.I)
_DISH_REQUEST = re.compile(r"\b(?:show me|give me|find me|find|get me|i want|i'?d like|looking for|suggest)\s+(?:some\s+|any\s+)?(.+?)\s+"
                           r"(?:recipes?|dishes?|ideas?)\b", re.I)


def enforce_safety_request(plan: TurnPlan, message: str) -> bool:
    """'Show me cookie recipes and confirm they are safe': the model sometimes refuses the WHOLE message as unsupported.
    The part that can be answered (the recipes, with the allergy filters) is; the part that cannot (certifying safety) is stated.
    -> True when the plan was converted."""
    if plan.intent != "unsupported" or not _SAFETY_ASK.search(plan.unsupported_text or message):
        return False
    m = _DISH_REQUEST.search(message)
    if not m:
        return False
    plan.intent, plan.scope, plan.query, plan.unsupported_text = "search", "new", m.group(1).strip(" ,"), None
    return True


def enforce_restrictions(plan: TurnPlan, message: str) -> list[str]:
    """Add any stated restriction the plan missed. Only ever ADDS filters (the safe direction). -> classes added."""
    if plan.intent not in _RESTRICTED_INTENTS:
        return []
    added = []
    for cls in detect_restrictions(message):
        if cls in plan.add_exclude_classes or cls in plan.remove_exclude_classes:
            continue
        plan.add_exclude_classes.append(cls)  # type: ignore[arg-type]
        added.append(cls)
    return added


_STEP_WORDS = r"(?:instruction\s+|cooking\s+|recipe\s+)?steps?"
_STEP_PATTERNS = [
    (rf"\b(?:at most|no more than|not more than|maximum of|max(?:imum)?|up to|within|in)\s+(\d+)\s+{_STEP_WORDS}\b", 0),
    (rf"\b(?:under|fewer than|less than)\s+(\d+)\s+{_STEP_WORDS}\b", -1),
    (rf"\b(\d+)\s+{_STEP_WORDS}\s+(?:or\s+(?:fewer|less)|max(?:imum)?|at most)\b", 0),
]


def detect_max_steps(message: str) -> int | None:
    """A step limit the user stated in words ("at most 6 steps", "under 5 steps", "6 steps or fewer")."""
    t = message.lower()
    for rx, adjust in _STEP_PATTERNS:
        m = re.search(rx, t)
        if m:
            return max(int(m.group(1)) + adjust, 0)
    return None


def enforce_step_limit(plan: TurnPlan, message: str) -> int | None:
    """If the plan forgot a stated step limit, add it (a hard constraint, like a restriction). -> the limit added."""
    if plan.intent not in {"search", "count", "top_ingredients", "ingredient_stats", "match"} or plan.max_steps is not None or plan.clear_max_steps:
        return None
    n = detect_max_steps(message)
    if n is not None:
        plan.max_steps = n
    return n


_FACTOR_WORDS = [(r"\b(?:double|twice|2x)\b", 2.0), (r"\b(?:triple|thrice|3x)\b", 3.0), (r"\b(?:half|halve|halved)\b", 0.5)]
_SCALE_CUE = re.compile(r"\b(?:scale|double|triple|halve|serves?|servings?|for\s+\d+\s+(?:people|persons|servings|portions|guests))\b")


def detect_scale(message: str) -> dict:
    """Scale facts stated in words: a factor ("double", "1.5 times") or servings ("it serves 4, make it for 6")."""
    t = message.lower()
    out: dict = {}
    for rx, f in _FACTOR_WORDS:
        if re.search(rx, t):
            out["scale_factor"] = f
            break
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", t) or re.search(r"\bx\s*(\d+(?:\.\d+)?)\b", t)
    if m:
        out["scale_factor"] = float(m.group(1))
    frm = re.search(r"\bserves?\s+(\d+)", t)
    to = re.search(r"\b(?:for|to|feed|feeds)\s+(\d+)\s*(?:people|persons|servings|portions|guests)?", t)
    if frm:
        out["servings_from"] = int(frm.group(1))
    if to and (not frm or int(to.group(1)) != int(frm.group(1))):
        out["servings_to"] = int(to.group(1))
    return out


def enforce_scale(plan: TurnPlan, message: str) -> dict:
    """A scale request keeps the numbers the user typed, even if the planner dropped them. -> what was added."""
    if plan.intent != "scale":
        return {}
    found, added = detect_scale(message), {}
    for key, val in found.items():
        if getattr(plan, key) is None:
            setattr(plan, key, val)
            added[key] = val
    return added


def fallback_plan(message: str) -> TurnPlan:
    """No LLM available: a small rule-based planner so the bot degrades to correct-but-basic instead of failing."""
    t = message.strip()
    low = t.lower()
    if _SCALE_CUE.search(low):
        return TurnPlan(intent="scale", **detect_scale(low))
    m = re.search(r"\bi (?:have|got)\b(.*)", low)
    if m and m.group(1).strip():
        return TurnPlan(intent="match", scope="new", available_text=m.group(1).strip(" .!"))
    if re.match(r"^(hi|hello|hey|help|what can you do)\b", low):
        return TurnPlan(intent="chat")
    return TurnPlan(intent="search", scope="new", query=t)


# --------------------------------------------------------------------------------------------- the bot
class Chatbot:
    def __init__(self, llm: LLM | None, sessions: Sessions | None = None, narrate: bool = True,
                 trace_dir: Path | None = None, search_mode: str = "lexical") -> None:
        self.llm, self.sessions, self.narrate = llm, sessions or Sessions(), narrate
        self.search_mode = search_mode
        self.trace_dir = trace_dir
        self._planner = (PROMPTS / "planner.md").read_text(encoding="utf-8")
        self._narrator = (PROMPTS / "narrator.md").read_text(encoding="utf-8")

    # ------------------------------------------------------------------ steps
    def _plan(self, state: SessionState, message: str, trace: dict) -> TurnPlan:
        if self.llm is None:
            trace["planner"] = "rules (no LLM configured)"
            return fallback_plan(message)
        recent = "\n".join(f"user: {u[:160]}" for u, _ in state.history[-3:]) or "none"
        user = f"{state.prompt_view()}\nrecent user messages:\n{recent}\n\nUSER MESSAGE: {message}"
        t_plan = time.time()
        try:
            raw, cached = self.llm.generate_json(self._planner, user, TurnPlan, 1024)
            trace["planner"], trace["planner_cached"] = "llm", cached
            trace["planner_seconds"] = round(time.time() - t_plan, 2)       # how much of the turn was the model call
            return TurnPlan.model_validate(raw)
        except LLM_ERRORS as e:
            trace["planner"], trace["planner_error"] = "rules (LLM unavailable)", type(e).__name__
        except ValidationError as e:
            trace["planner"], trace["planner_error"] = "rules (invalid plan)", str(e)[:200]
        return fallback_plan(message)

    def _narrate(self, out: Outcome, message: str, trace: dict) -> str:
        if not (self.narrate and self.llm and out.status in ("ok", "no_results") and out.tool
                and out.intent != "get_recipe"):
            return out.text
        user = f"USER QUESTION: {message}\n\nGROUNDED ANSWER (do not repeat it):\n{out.text[:3000]}"
        try:
            raw, cached = self.llm.generate_json(self._narrator, user, Narration, 300)
            intro = Narration.model_validate(raw).intro.strip()
        except (*LLM_ERRORS, ValidationError) as e:
            trace["narration"] = f"skipped ({type(e).__name__})"
            return out.text
        v = ground.validate(intro, out.text, len(out.sources), require_all_citations=False)
        trace["narration"] = "accepted" if (v.ok and intro) else "rejected"
        if not v.ok or not intro:
            trace["narration_problems"] = v.problems
            return out.text
        # the facts are always the tool-derived text; the LLM contributes only a short lead-in sentence
        return f"{intro}\n\n{out.text}"

    # ------------------------------------------------------------------ one turn
    def warm_up(self) -> None:
        """Run one throwaway search so the first real question does not pay for loading the embedding model and index."""
        try:
            execute(SessionState(search_mode=self.search_mode), TurnPlan(intent="search", query="soup"), "warm up")
        except Exception:
            log.warning("warm-up search failed (the first real question will be slower)")
        if self.llm is not None:
            # The first structured-output call with a new schema was measured at 13-15 s against about 1.7 s afterwards
            # (the service prepares the schema on first use). Pay that now, in the background, not on the user's first question.
            # The message is unique each start so the on-disk cache cannot skip the real call.
            try:
                user = f"{SessionState().prompt_view()}\nrecent user messages:\nnone\n\nUSER MESSAGE: hello (warm-up {time.time():.0f})"
                self.llm.generate_json(self._planner, user, TurnPlan, 1024)
            except Exception:
                log.warning("warm-up model call failed (the first real question may be slower)")

    def turn(self, session_id: str, message: str, vegetarian: bool | None = None,
             taste: dict | None = None) -> TurnResult:
        t0 = time.time()
        state = self.sessions.get(session_id)
        state.search_mode = self.search_mode
        if vegetarian is not None:
            state.vegetarian = vegetarian        # the UI toggle is applied in code, whatever the model planned
        if taste is not None:
            state.taste = clean_profile(taste)   # the taste profile is a user setting, validated and never model-written
        trace: dict = {"message": message, "session": session_id}
        plan = self._plan(state, message, trace)
        converted = enforce_safety_request(plan, message)
        if converted:
            trace["safety_request_converted"] = True
        moved = normalize_ingredient_query(plan)
        if moved:
            trace["query_moved_to_ingredient"] = moved
        allergen_report = enforce_allergens(plan, message)
        if allergen_report:
            trace["allergen_guard"] = allergen_report
        added = enforce_restrictions(plan, message)
        steps_added = enforce_step_limit(plan, message)
        scale_added = enforce_scale(plan, message)
        trace["plan"] = plan.model_dump(exclude_defaults=True)
        if scale_added:
            trace["scale_added_by_guard"] = scale_added
        if steps_added is not None:
            trace["step_limit_added_by_guard"] = steps_added
        if added:
            trace["restrictions_added_by_guard"] = added
        try:
            out = execute(state, plan, message)
        except (StaleDataError, StaleIndexError) as e:
            out = Outcome(plan.intent, f"The data needs rebuilding before I can answer: {e}", status="error")
        except Exception as e:                                   # a tool failure must never become an invented answer
            log.exception("tool failure")
            out = Outcome(plan.intent, "I hit an internal error running that, so I will not guess an answer. "
                          "Please rephrase or try again.", status="error")
            trace["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if converted and out.status in ("ok", "no_results"):
            out.text += ("\nNot answered: I cannot confirm safety for any recipe. The checks read only the ingredient text, and "
                         "packaged foods can hide allergens, so please check the labels yourself.")
        answer = self._narrate(out, message, trace)
        if out.display and answer == out.text:
            answer = out.display                 # the screen gets the short form; `grounded_answer` below keeps the full text
        state.history.append((message, out.text[:300]))
        state.history = state.history[-6:]
        trace.update(tool=out.tool, args=out.args, status=out.status, seconds=round(time.time() - t0, 2),
                     llm=self.llm.stats() if hasattr(self.llm, "stats") else None)
        self._write_trace(trace)
        return TurnResult(answer, out.text, out.intent, out.status, out.sources, out.computation,
                          state.active_filters(), out.notes, trace, out.web, out.suggest)

    def _write_trace(self, trace: dict) -> None:
        if not self.trace_dir:
            return
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with open(self.trace_dir / "turns.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


def build_chatbot(narrate: bool | None = None) -> Chatbot:
    """Production wiring: Gemini if a key is configured, else the rule-based fallback.

    Latency: one model call per turn (the plan). The optional LLM lead-in is a second call and is OFF unless
    RECIPEBOT_NARRATE_WITH_LLM=true. Calls give up after 10 s or a long rate-limit wait, and the rule-based planner answers."""
    s = get_settings()
    llm = None
    if s.gemini_api_key is not None:
        from recipebot.llm.gemini import GeminiClient
        llm = GeminiClient(s.gemini_api_key, s.gemini_model, s.llm_cache_dir, max_calls=200, min_interval_s=4.0,
                           max_retries=1, request_timeout_s=10.0, max_wait_s=8.0, thinking_level="minimal")
    return Chatbot(llm, narrate=s.narrate_with_llm if narrate is None else narrate,
                   trace_dir=s.tmp_dir.parent / "traces", search_mode="hybrid")
