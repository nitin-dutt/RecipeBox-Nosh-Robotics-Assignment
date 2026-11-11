"""Typed contracts shared by the tools, the orchestrator and the UI.

Every tool returns its answer together with a `Computation` record. The record is what the UI shows as "how this
was calculated" and what the grounding validator checks answers against.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RestrictionClass = Literal[
    "dairy", "egg", "peanut", "tree_nut", "gluten", "soy", "fish", "shellfish", "sesame", "pork", "beef",
    "poultry", "lamb", "game", "meat", "vegetarian", "vegan", "alcohol",
]
MatchLevel = Literal["exact", "name_or_parent", "base"]
StepRule = Literal["sentences", "paragraphs"]
Source = Literal["ar", "epi", "fn"]


class Population(BaseModel):
    """Which recipes are counted at all (this is also the denominator for percentages)."""
    complete_only: bool = Field(True, description="Only recipes with a title, ingredients and instructions.")
    dedupe: bool = Field(False, description="Count one recipe per exact-duplicate group.")
    sources: list[Source] | None = Field(None, description="ar=Allrecipes, epi=Epicurious, fn=Food Network.")


class Filters(BaseModel):
    population: Population = Field(default_factory=Population)
    include_ingredients: list[str] = Field(default_factory=list, description="ALL of these must be present.")
    exclude_ingredients: list[str] = Field(default_factory=list, description="NONE of these may appear.")
    exclude_classes: list[RestrictionClass] = Field(default_factory=list, description="e.g. dairy, egg, peanut.")
    exclude_classes_definite: list[RestrictionClass] = Field(
        default_factory=list, description="Classes dropped only on DEFINITE (yes) hits, whatever the mode: a preference such as "
        "'no egg' in vegetarian mode, where a 'may contain egg' line (noodles) should not hide the recipe.")
    restriction_mode: Literal["strict", "lenient"] = Field(
        "strict", description="strict drops recipes with yes OR maybe hits; lenient drops only definite (yes) hits.")
    match_level: MatchLevel = Field(
        "name_or_parent", description="exact: product name only. name_or_parent: also varieties (red onion is an "
        "onion). base: also processed forms (garlic powder counts as garlic).")
    max_steps: int | None = Field(None, ge=0)
    min_steps: int | None = Field(None, ge=0)
    step_rule: StepRule = "sentences"
    max_ingredients: int | None = Field(None, ge=0)


class SearchSpec(Filters):
    query: str | None = Field(None, description="Dish name or words, e.g. 'penne pasta'.")
    limit: int = Field(10, ge=1, le=50)
    offset: int = Field(0, ge=0)
    sort: Literal["relevance", "fewest_steps", "fewest_ingredients"] = "relevance"
    include_ingredient_lines: bool = False
    include_instructions: bool = False
    mode: Literal["lexical", "hybrid", "keyword", "semantic"] = Field(
        "lexical",
        description="lexical: every query word must match (precise; falls back to hybrid if nothing matches). "
                    "hybrid: BM25 + embeddings fused with RRF, for descriptive requests ('cozy chickpea stew'). "
                    "keyword: BM25 only. semantic: embeddings only. Hybrid/semantic results are similar in meaning "
                    "and are NOT guaranteed to contain each word; use include_ingredients to require an ingredient.")


class FunnelStep(BaseModel):
    label: str
    remaining: int


class Computation(BaseModel):
    tool: str
    description: str
    population: str
    filters: dict
    funnel: list[FunnelStep]
    n_analyzed: int
    denominator: int | None = None
    method: str
    sql: str
    notes: list[str] = Field(default_factory=list)


class IngredientLine(BaseModel):
    position: int
    text: str = Field(description="Cleaned ingredient line.")
    original: str | None = Field(None, description="Verbatim line from the source file.")


class RestrictionNote(BaseModel):
    restriction: str
    status: Literal["yes", "maybe"]
    lines: list[IngredientLine]
    via: str = "name"
    snippet: str | None = Field(None, description="For via='instructions': the instruction text that mentions it.")


class RecipeHit(BaseModel):
    id: str
    title: str | None
    source: str
    n_ingredients: int
    n_steps_sentences: int
    n_steps_paragraphs: int
    dup_group_size: int = 1
    score: float | None = None
    retrieval: dict | None = Field(None, description="BM25 / dense ranks and RRF score for hybrid retrieval.")
    matched_ingredients: list[IngredientLine] = Field(default_factory=list)
    warnings: list[RestrictionNote] = Field(default_factory=list)
    ingredients: list[IngredientLine] | None = None
    instructions: str | None = None


class SearchResult(BaseModel):
    total_matches: int
    shown: int
    recipes: list[RecipeHit]
    checked_restrictions: list[str] = Field(default_factory=list)
    excluded_only_by_maybe: int = 0
    unresolved_terms: list[str] = Field(default_factory=list)
    computation: Computation
    disclaimer: str | None = None


class RecipeRecord(BaseModel):
    id: str
    source: str
    title: str | None
    title_raw: str | None
    ingredients: list[IngredientLine]
    instructions: str
    instructions_raw: str
    n_steps_sentences: int
    n_steps_paragraphs: int
    picture_link: str | None
    is_complete: bool
    dup_group_id: str | None
    dup_group_size: int


class CountResult(BaseModel):
    count: int
    denominator: int = Field(description="Size of the population the filters start from.")
    percentage: float = Field(description="count / denominator * 100")
    denominator_all: int = Field(description="Every non-empty recipe in the dataset.")
    percentage_of_all: float = Field(description="count / denominator_all * 100")
    computation: Computation


class IngredientFrequency(BaseModel):
    ingredient: str
    recipes: int
    percentage: float


class TopIngredientsResult(BaseModel):
    items: list[IngredientFrequency]
    computation: Computation


class NumericSummary(BaseModel):
    group: str
    n: int
    mean: float
    median: float
    minimum: float
    maximum: float
    p90: float


class NumericSummaryResult(BaseModel):
    metric: str
    groups: list[NumericSummary]
    computation: Computation


class RankedRecipe(BaseModel):
    rank: int
    id: str
    title: str | None
    source: str
    steps: int
    other_steps_measure: int
    n_ingredients: int


class RankResult(BaseModel):
    ranking: list[RankedRecipe]
    n_tied_at_cutoff: int
    tie_break: str
    computation: Computation


class UnsupportedResult(BaseModel):
    supported: Literal[False] = False
    attribute: str
    reason: str
    what_the_dataset_has: list[str]
    can_offer_instead: list[str]


# ---------------------------------------------------------------------------------------------- Task 2: matching
class AvailableIngredient(BaseModel):
    name: str = Field(description="What the user has, e.g. 'eggs', 'red onion'.")
    quantity: float | None = Field(None, description="How much, if the user said.")
    unit: str | None = Field(None, description="Unit for quantity (cup, g, tbsp...). None means 'pieces'.")


PriorityKey = Literal["use_most_available", "fewest_missing", "fewest_steps", "fewest_ingredients"]


class MatchRequest(BaseModel):
    available: list[AvailableIngredient] = Field(min_length=1)
    constraints: Filters = Field(
        default_factory=Filters,
        description="HARD constraints: exclude_ingredients, exclude_classes, max_steps, include_ingredients "
                    "(must use), population. These are never relaxed silently.")
    ranking: Literal["balanced", "priority"] = Field(
        "balanced",
        description="balanced: score = ingredients of yours used - missing_penalty x ingredients still needed; "
                    "ties by the priority keys. priority: strict lexicographic order of the priority keys.")
    missing_penalty: float = Field(0.25, ge=0, le=5, description="Cost of each ingredient the user would still need, "
                                   "in units of 'one of your ingredients used' (balanced ranking).")
    priority: list[PriorityKey] = Field(
        default_factory=lambda: ["use_most_available", "fewest_missing", "fewest_steps"],
        description="Preferences in order of importance. In balanced mode only fewest_steps / fewest_ingredients "
                    "act as tie-breakers after the score.")
    max_missing: int | None = Field(None, ge=0, description="Hard cap on ingredients the user would still need.")
    pantry: list[str] | None = Field(
        None, description="Items assumed on hand. None = default pantry (salt, pepper, water, cooking oils); "
                          "[] = assume nothing.")
    k: int = Field(5, ge=1, le=20)
    offer_violating_alternatives: bool = True


class ItemMatch(BaseModel):
    line: IngredientLine
    status: Literal["have", "have_quantity_unverified", "short", "missing", "pantry", "optional"]
    matched_with: str | None = None
    how: str | None = Field(None, description="Why it counts as a match (same product / variety / alternative).")
    need: str | None = None
    have: str | None = None
    note: str | None = None


class UsedIngredient(BaseModel):
    available: str
    quantity_check: Literal["sufficient", "short", "not_verified", "no_quantity_given"]
    detail: str


class MatchedRecipe(BaseModel):
    rank: int
    id: str
    title: str | None
    source: str
    steps: int
    steps_rule: str
    n_required: int
    n_have: int
    n_short: int
    n_missing: int
    n_used: int
    n_available: int
    used: list[UsedIngredient]
    unused_available: list[str]
    matched: list[ItemMatch]
    missing: list[ItemMatch]
    assumed_pantry: list[ItemMatch]
    optional_ignored: list[ItemMatch]
    near_misses: list[str] = Field(default_factory=list)
    score: float | None = Field(None, description="used - missing_penalty x still needed (balanced only).")
    why: str


class Violation(BaseModel):
    constraint: str
    detail: str
    lines: list[IngredientLine] = Field(default_factory=list)


class ClosestAlternative(BaseModel):
    recipe: MatchedRecipe
    violates: list[Violation]
    relaxed_constraint: str


class MatchResult(BaseModel):
    status: Literal["matches", "no_match"]
    message: str
    matches: list[MatchedRecipe]
    closest_alternatives: list[ClosestAlternative] = Field(default_factory=list)
    unrecognised_ingredients: list[str] = Field(default_factory=list)
    ignored_available: list[str] = Field(default_factory=list)
    assumed_pantry: list[str] = Field(default_factory=list)
    ranking_rule: str
    quantity_policy: str
    computation: Computation


# ---------------------------------------------------------------------------------------------- similar recipes
class SimilarRecipe(BaseModel):
    id: str
    title: str | None
    source: str
    similarity: float = Field(description="Cosine similarity of the embeddings (0 to 1).")
    shared_ingredients: list[str]
    only_in_this_recipe: list[str]
    only_in_seed_recipe: list[str]
    steps: int


class SimilarResult(BaseModel):
    seed_id: str
    seed_title: str | None
    items: list[SimilarRecipe]
    computation: Computation


# ---------------------------------------------------------------------------------------------- Task 3: substitution
class SubstitutionRequest(BaseModel):
    recipe_id: str = Field(description="Stable ID of the recipe to adapt.")
    remove: list[str] = Field(default_factory=list, description="Ingredients to remove or replace, e.g. ['egg'].")
    exclude_classes: list[RestrictionClass] = Field(
        default_factory=list, description="Restriction classes the adapted recipe must avoid, e.g. ['dairy']. Every flagged line is adapted.")
    restriction_mode: Literal["strict", "lenient"] = "strict"
    neighbours: int = Field(400, ge=5, le=1500, description="How many similar recipes to examine for evidence.")
    min_support: int = Field(3, ge=1, le=10, description="Recipes needed before a substitute counts as well supported.")


class EvidenceRecipe(BaseModel):
    id: str
    title: str | None
    similarity: float | None = None
    line: str = Field(description="The ingredient line in that recipe that supports the substitute.")
    kind: Literal["alternative_listed", "variant_without", "omits"]


class SubstituteCandidate(BaseModel):
    substitute: str
    tier: Literal["direct", "variant", "indirect"]
    n_supporting: int = Field(description="Similar recipes WITHOUT the ingredient that contain this substitute.")
    prevalence_without: float = Field(description="Share of similar recipes without the ingredient that contain it.")
    prevalence_with: float = Field(description="Share of similar recipes that still use the ingredient and contain it.")
    n_alternative_lines: int = Field(description="Recipes that list it as an 'or' alternative of the ingredient.")
    n_alternative_similar: int = Field(0, description="Of the recipes most similar to this one, how many list it as an 'or' alternative.")
    roles_covered: list[str] = Field(default_factory=list)
    roles_missing: list[str] = Field(default_factory=list)
    role_note: str = ""
    evidence: list[EvidenceRecipe] = Field(default_factory=list)


class TargetAdaptation(BaseModel):
    line: IngredientLine
    ingredient: str
    why: str
    n_similar_without: int
    n_similar_with: int
    verdict: Literal["supported", "weak", "conflicting", "none"]
    verdict_reason: str
    candidates: list[SubstituteCandidate]
    chosen: str | None = None
    replacement_line: str | None = None
    replacement_source: EvidenceRecipe | None = None
    omission_evidence: list[EvidenceRecipe] = Field(default_factory=list)


class InstructionEdit(BaseModel):
    step: int
    action: Literal["replace", "delete", "review"]
    before: str
    after: str | None = None


class AdaptedLine(BaseModel):
    position: int
    text: str
    changed: bool
    original: str | None = None
    evidence_level: Literal["supported", "weak"] | None = Field(
        None, description="For a changed line: supported by similar recipes, or only weak (tentative).")


class RestrictionCheck(BaseModel):
    restriction: str
    status: Literal["no_term_found", "possible", "still_present"]
    lines: list[str] = Field(default_factory=list)


class SubstitutionResult(BaseModel):
    status: Literal["adapted", "partial", "unsupported"]
    summary: str
    recipe_id: str
    recipe_title: str | None
    targets: list[TargetAdaptation]
    adapted_ingredients: list[AdaptedLine]
    revised_instructions: str
    instruction_edits: list[InstructionEdit]
    predicted_effects: list[str]
    restriction_check: list[RestrictionCheck]
    remaining_issues: list[str]
    external_search_suggested: bool
    evidence_policy: str
    disclaimer: str
    computation: Computation
