"""Recall net over the INSTRUCTIONS text.

A recipe can use an ingredient it never lists ("brush with melted butter", "deep fry in peanut oil", "serve with sour
cream"). The ingredient-line parse cannot see that, so every instruction text is also scanned with the same taxonomy.
Hits are always "maybe" (never "yes"): instructions are full of verbs and shapes that look like ingredients
("cream the shortening", "walnut-sized balls", "cheese cloth"), which the instruction guards in taxonomy.yaml remove.
"""
from __future__ import annotations

import re

from recipebot.domain.ingredient import normalize_for_scan
from recipebot.domain.taxonomy import Taxonomy, get_taxonomy


def scan_instructions(text: str, tax: Taxonomy | None = None) -> list[tuple[str, str]]:
    """-> [(class, snippet)] for classes the instruction text mentions. Status is always 'maybe'."""
    tax = tax or get_taxonomy()
    if not text:
        return []
    cleaned = tax.apply_instruction_guards(normalize_for_scan(text))
    out: list[tuple[str, str]] = []
    for cls in tax.classify_name(cleaned):
        hit = tax.match_term(cls, cleaned)
        out.append((cls, _snippet(text, hit, tax) if hit else ""))
    return out


def _snippet(original: str, hit: tuple[str, str], tax: Taxonomy) -> str:
    """Text around the occurrence that really triggered the flag (not one removed by a guard phrase)."""
    term, owner = hit
    plural = rf"(?<![A-Za-z]){re.escape(term)}(?:s|es)?(?![A-Za-z])"
    for m in re.finditer(plural, original, re.IGNORECASE):
        window = normalize_for_scan(original[max(0, m.start() - 25): m.end() + 25])
        window = tax.apply_instruction_guards(window)
        if tax.term_survives_guards(owner, term, window):
            a, b = max(0, m.start() - 50), min(len(original), m.end() + 40)
            return ("..." if a else "") + original[a:b].replace("\n", " ").strip() + ("..." if b < len(original) else "")
    return ""
