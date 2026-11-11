"""What the LLM is allowed to produce: a small, flat, validated description of the user's intent. Never an answer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from recipebot.domain.models import RestrictionClass

Intent = Literal["search", "count", "top_ingredients", "ingredient_stats", "rank_steps", "match", "substitute", "get_recipe",
                 "similar", "scale", "unsupported", "chat", "clarify"]


class TurnPlan(BaseModel):
    intent: Intent
    scope: Literal["new", "refine"] = Field(
        "refine", description="new = a fresh topic, forget the old filters; refine = keep the active filters and change only what the user said.")
    query: str | None = Field(None, description="Dish name or words to search for.")
    add_include: list[str] = Field(default_factory=list, description="Ingredients that must be present.")
    add_exclude: list[str] = Field(default_factory=list, description="Ingredients that must NOT be present.")
    add_exclude_classes: list[RestrictionClass] = Field(default_factory=list, description="Restriction classes to avoid: dairy, egg, peanut...")
    remove_exclude: list[str] = Field(default_factory=list, description="Ingredients the user now allows again.")
    remove_exclude_classes: list[RestrictionClass] = Field(default_factory=list, description="Classes the user now allows again.")
    max_steps: int | None = Field(None, description="Upper limit on instruction steps, if the user gave one.")
    clear_max_steps: bool = False
    lenient: bool | None = Field(None, description="true only if the user explicitly accepts uncertain 'possible' matches; false to go back to strict.")
    available_text: str | None = Field(None, description="For intent=match: the ingredients the user HAS, as they said them.")
    k: int | None = Field(None, description="How many results the user asked for.")
    order: Literal["fewest", "most"] = "fewest"
    ref_index: int | None = Field(None, description="Which result of the last list the user means (1 = first).")
    recipe_id: str | None = Field(None, description="Only if the user typed an exact recipe ID.")
    remove: list[str] = Field(default_factory=list, description="For intent=substitute: ingredients to remove or replace.")
    scale_factor: float | None = Field(None, description="For intent=scale: the multiplier (double = 2, half = 0.5, triple = 3).")
    servings_from: int | None = Field(None, description="For intent=scale: how many servings the recipe makes now, if the user said.")
    servings_to: int | None = Field(None, description="For intent=scale: how many servings the user wants.")
    by_source: bool = Field(False, description="For intent=ingredient_stats: break the result down by source collection (Allrecipes, Epicurious, Food Network).")
    top_level: Literal["name", "parent", "base"] = "name"
    question: str | None = Field(None, description="For intent=clarify: ONE short question to the user.")
    unsupported_text: str | None = Field(None, description="For intent=unsupported: the part of the request the dataset cannot answer.")
