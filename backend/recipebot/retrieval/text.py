"""Text used for retrieval: what is indexed per recipe, and how queries/documents are tokenised.

A recipe is represented by its title and its canonical ingredient names (not the instructions): short, dense in
meaning, cheap to embed, and the same facts the structured filters use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from recipebot.domain.ingredient import singular
from recipebot.store.db import Database

DOC_VERSION = "v1-title+ingredient-names"
MAX_INGREDIENT_NAMES = 14
_STOP = {"the", "and", "with", "of", "a", "an", "in", "for", "to", "or", "on", "at", "by", "from"}
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [singular(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    ingredient_names: tuple[str, ...]
    parents: tuple[str, ...]

    @property
    def dense_text(self) -> str:
        names = ", ".join(self.ingredient_names)
        return f"{self.title}. Ingredients: {names}" if names else self.title

    @property
    def bm25_tokens(self) -> list[str]:
        # title counted three times so a title hit outweighs an ingredient hit; parents let "onion" find "red onion"
        t = tokenize(self.title)
        return t * 3 + tokenize(" ".join(self.ingredient_names)) + tokenize(" ".join(self.parents))


def load_docs(db: Database) -> list[Doc]:
    rows = db.query(
        "SELECT r.id, coalesce(r.title, ''), "
        "list(p.name ORDER BY p.position, p.item_idx) FILTER (WHERE p.name IS NOT NULL), "
        "list(p.parent ORDER BY p.position, p.item_idx) FILTER (WHERE p.parent IS NOT NULL) "
        "FROM recipes r LEFT JOIN ingredients_parsed p ON p.recipe_id = r.id AND p.status = 'ok' "
        "GROUP BY r.id, r.title ORDER BY r.id")
    docs = []
    for rid, title, names, parents in rows:
        seen: dict[str, None] = {}
        for n in names or []:
            seen.setdefault(n, None)
        uniq = tuple(list(seen)[:MAX_INGREDIENT_NAMES])
        pseen: dict[str, None] = {}
        for n in parents or []:
            pseen.setdefault(n, None)
        docs.append(Doc(rid, title, uniq, tuple(pseen)))
    return docs


def db_fingerprint(db: Database) -> str:
    """Changes whenever recipe IDs, titles OR parsed ingredient names/parents change.

    The ingredient part matters: when the ingredient parser or lexicon is improved, the text that was embedded and
    indexed is no longer the text the database holds, and a stale index must be detected, not silently trusted.
    """
    titles = db.scalar("SELECT md5(string_agg(id || '|' || coalesce(title, ''), ',' ORDER BY id)) FROM recipes")
    names = db.scalar(
        "SELECT md5(string_agg(recipe_id || ':' || coalesce(name, '') || ':' || coalesce(parent, ''), ',' "
        "ORDER BY recipe_id, position, item_idx)) FROM ingredients_parsed WHERE status = 'ok'")
    return f"{titles}-{names}"
