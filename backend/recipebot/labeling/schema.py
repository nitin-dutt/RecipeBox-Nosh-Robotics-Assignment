"""Label schema shared by the LLM labeler, the hand-made gold set, the rule baseline and the merge."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClassName = Literal[
    "dairy", "egg", "peanut", "tree_nut", "gluten", "soy", "fish", "shellfish", "sesame", "pork", "beef", "poultry",
    "lamb", "game", "meat", "vegetarian", "vegan", "alcohol",
]
CLASSES: tuple[str, ...] = tuple(ClassName.__args__)   # type: ignore[attr-defined]


class ClassFlag(BaseModel):
    cls: ClassName
    level: Literal["yes", "maybe"]


class NameLabel(BaseModel):
    """What the LLM returns for one ingredient name (structured output)."""
    name: str
    is_ingredient: bool
    product: str
    parent: str
    form: str = Field(description="powder, paste, juice, ... or 'none'")
    classes: list[ClassFlag]
    reason: str = ""

    def class_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in self.classes:
            if out.get(f.cls) != "yes":
                out[f.cls] = f.level
        return out
