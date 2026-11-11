"""Fingerprint of everything that decides how ingredient lines are understood.

Stored in the database when it is built. If the configuration (lexicon, taxonomy, groups, reviewed label overlay)
is edited afterwards, the stored data no longer reflects it, and the tools refuse to run on it instead of silently
giving answers from an older understanding of the data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from recipebot.settings import get_settings

CONFIG_FILES = ["ingredients.yaml", "taxonomy.yaml", "groups.yaml"]
OVERLAY = Path("labels") / "ingredient_labels.json"          # reviewed LLM/human labels (optional)
OVERRIDES = "ingredient_overrides.yaml"                        # human decisions on suggestions (optional)


def lexicon_fingerprint() -> str:
    cfg = get_settings().config_dir
    h = hashlib.sha256()
    for name in [*CONFIG_FILES, OVERRIDES, str(OVERLAY)]:
        p = cfg / name
        h.update(name.encode())
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()[:16]
