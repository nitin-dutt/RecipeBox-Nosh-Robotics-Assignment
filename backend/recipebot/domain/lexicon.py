"""Loads config/ingredients.yaml into ready-to-use lookup structures."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from recipebot.settings import get_settings


@dataclass(frozen=True)
class Lexicon:
    descriptor_phrase_re: re.Pattern[str]
    descriptors: frozenset[str]
    protected: tuple[str, ...]
    synonyms: dict[str, str]
    collapse: dict[str, frozenset[str]]
    atomic: frozenset[str]
    form_suffixes: frozenset[str]
    form_prefixes: tuple[str, ...]
    varieties: dict[str, tuple[str, ...]]
    trailing_noise: frozenset[str]
    base_overrides: dict[str, tuple[str, str]]
    head_parents: frozenset[str]
    interchangeable: frozenset[str]
    lonely_modifiers: frozenset[str]
    parent_overrides: dict[str, str]
    not_ingredient: frozenset[str]
    distinct_products: frozenset[str]


def _alternation(items: list[str]) -> str:
    return "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True))


def load_lexicon(path: Path | None = None) -> Lexicon:
    path = path or get_settings().config_dir / "ingredients.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ov_path = path.parent / "ingredient_overrides.yaml"
    overrides = (yaml.safe_load(ov_path.read_text(encoding="utf-8")) or {}) if ov_path.exists() else {}
    phrases = raw["descriptor_phrases"]
    return Lexicon(
        descriptor_phrase_re=re.compile(rf"\b(?:{_alternation(phrases)})\b"),
        descriptors=frozenset(raw["descriptors"]),
        protected=tuple(sorted(raw["protected_phrases"], key=len, reverse=True)),
        synonyms={str(k): str(v) for k, v in raw["synonyms"].items()},
        collapse={k: frozenset(v or []) for k, v in raw["collapse"].items()},
        atomic=frozenset(raw["atomic_products"]),
        form_suffixes=frozenset(raw["form_suffixes"]),
        form_prefixes=tuple(sorted(raw["form_prefixes"], key=len, reverse=True)),
        varieties={k: tuple(v) for k, v in raw["varieties"].items()},
        trailing_noise=frozenset(raw["trailing_noise"]),
        base_overrides={k: (v[0], v[1]) for k, v in raw.get("base_overrides", {}).items()},
        head_parents=frozenset(raw.get("head_parents", [])),
        interchangeable=frozenset(raw.get("interchangeable", [])),
        lonely_modifiers=frozenset(raw.get("lonely_modifiers", [])),
        parent_overrides={str(k): str(v) for k, v in (overrides.get("parent") or {}).items()},
        not_ingredient=frozenset(overrides.get("not_ingredient") or []),
        distinct_products=frozenset(raw.get("distinct_products", [])),
    )


@lru_cache
def get_lexicon() -> Lexicon:
    return load_lexicon()
