"""Server-side conversation state. A follow-up is a DIFF to this state, never a re-prompt of the whole history.

The LLM only proposes the diff (`TurnPlan`); `apply_plan` applies it with plain code, so what the bot filters on is
always visible (`SessionState.active_filters`) and never hidden in a prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from recipebot.domain.models import Filters


@dataclass
class ShownRecipe:
    n: int                       # the citation number the user saw
    id: str
    title: str | None
    source: str = ""


@dataclass
class SessionState:
    query: str | None = None
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    max_steps: int | None = None
    vegetarian: bool = False                   # the UI toggle: a HARD filter set by the user, never by the model
    taste: dict[str, str] = field(default_factory=dict)   # the user's taste profile (sweet/salt/spice: low|medium|high), set by the UI
    search_mode: str = "lexical"            # set by the bot: lexical (no index needed) or hybrid (BM25 + embeddings)
    mode: str = "strict"                       # restriction mode: strict | lenient
    available_text: str | None = None          # what the user said they have (Task 2 follow-ups reuse it)
    last_results: list[ShownRecipe] = field(default_factory=list)
    focus_id: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)   # (user, assistant), short, for the planner

    # ------------------------------------------------------------------ views
    def active_filters(self) -> dict:
        return {k: v for k, v in {"query": self.query, "include_ingredients": self.include,
                                  "exclude_ingredients": self.exclude, "exclude_classes": self.classes,
                                  "max_steps": self.max_steps, "vegetarian_only": self.vegetarian or None, "restriction_mode": self.mode,
                                  "available": self.available_text}.items() if v not in (None, [], "")}

    def filters(self) -> Filters:
        classes = list(self.classes)
        if self.vegetarian:
            # "Veg" here means what a cook in India means: no meat, poultry, fish or game, AND no egg (dairy is allowed).
            if "vegetarian" not in classes:
                classes.append("vegetarian")      # the composite class: no meat, poultry, fish or game ingredient found
        definite = ["egg"] if self.vegetarian and "egg" not in classes else []     # a "may contain egg" line (noodles) does not hide a recipe
        return Filters(include_ingredients=list(self.include), exclude_ingredients=list(self.exclude),
                       exclude_classes=classes, exclude_classes_definite=definite, max_steps=self.max_steps, restriction_mode=self.mode)

    def prompt_view(self) -> str:
        shown = "; ".join(f"{r.n}={r.title}" for r in self.last_results[:10]) or "none"
        return (f"active filters: {self.active_filters() or 'none'}\n"
                f"results shown last turn (number=title): {shown}\n"
                f"focus recipe: {self.focus_id or 'none'}")

    def resolve_recipe(self, ref_index: int | None, recipe_id: str | None) -> tuple[str | None, str | None]:
        """-> (recipe id, problem). A problem is a plain-language reason the reference could not be resolved."""
        if recipe_id:
            return recipe_id, None
        if ref_index is not None:
            for r in self.last_results:
                if r.n == ref_index:
                    return r.id, None
            return None, f"There is no result number {ref_index} in the last list."
        if self.focus_id:
            return self.focus_id, None
        if len(self.last_results) == 1:
            return self.last_results[0].id, None
        if self.last_results:
            return None, "Which recipe do you mean? Give me its number from the last list."
        return None, "Which recipe do you mean? Search for one first, or tell me its name."


def merge_unique(current: list[str], add: list[str], remove: list[str]) -> list[str]:
    drop = {x.lower() for x in remove}
    out = [x for x in current if x.lower() not in drop]
    for x in add:
        if x.lower() not in {y.lower() for y in out}:
            out.append(x)
    return out


class Sessions:
    """In-memory store keyed by session id. (Production: Redis or the database; the state is small and serialisable.)"""

    def __init__(self) -> None:
        self._s: dict[str, SessionState] = {}

    def get(self, sid: str) -> SessionState:
        return self._s.setdefault(sid, SessionState())

    def reset(self, sid: str) -> None:
        self._s.pop(sid, None)
