"""Deterministic, conservative instruction editing.

The tool edits only sentences that mention a changed ingredient and records every edit as before/after. Replacing a word is
not the same as writing good cooking instructions, so edits are listed explicitly for the reader (and for the orchestrator
LLM, which may later polish the wording using ONLY this edit list). Nothing is invented: no new step, time or temperature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from recipebot.domain.ingredient import singular
from recipebot.domain.models import InstructionEdit
from recipebot.ingestion.cleaning import split_sentences


@dataclass(frozen=True)
class Change:
    names: tuple[str, ...]       # words that identify the ingredient in text, e.g. ("egg",) or ("heavy cream", "cream")
    replacement: str | None      # None = removed without replacement


def _pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    words = sorted({n for n in names if n}, key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b(?:{alt})(?:s|es)?\b", re.IGNORECASE)


def rewrite_instructions(instructions: str, changes: list[Change]) -> tuple[str, list[InstructionEdit]]:
    edits: list[InstructionEdit] = []
    out_paragraphs: list[str] = []
    step = 0
    for paragraph in [p for p in instructions.split("\n") if p.strip()]:
        kept: list[str] = []
        for sentence in split_sentences(paragraph) or [paragraph]:
            step += 1
            new, edit = _edit_sentence(sentence, changes, step)
            if edit:
                edits.append(edit)
            if new:
                kept.append(new)
        if kept:
            out_paragraphs.append(" ".join(kept))
    return "\n".join(out_paragraphs), edits


def _edit_sentence(sentence: str, changes: list[Change], step: int) -> tuple[str | None, InstructionEdit | None]:
    text = sentence
    touched = False
    unresolved: str | None = None
    for ch in changes:
        pat = _pattern(ch.names)
        if not pat.search(text):
            continue
        touched = True
        if ch.replacement:
            text = pat.sub(ch.replacement, text)
        else:
            unresolved = ch.names[0]
    if not touched:
        return sentence, None
    if unresolved is not None:
        words = re.findall(r"[A-Za-z']+", sentence)
        if len(words) <= 8:   # a short sentence that is only about the removed ingredient ("Beat the eggs.")
            return None, InstructionEdit(step=step, action="delete", before=sentence, after=None)
        return sentence, InstructionEdit(step=step, action="review", before=sentence, after=None)
    return text, InstructionEdit(step=step, action="replace", before=sentence, after=text)


def mention_words(name: str) -> tuple[str, ...]:
    """Words a recipe text may use for an ingredient name: the full name and its head word."""
    parts = name.split()
    out = [name]
    if len(parts) > 1:
        out.append(singular(parts[-1]))
    return tuple(dict.fromkeys(out))
