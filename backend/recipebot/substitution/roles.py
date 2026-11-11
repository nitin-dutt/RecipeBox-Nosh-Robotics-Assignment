"""Culinary role knowledge (config/roles.yaml): general knowledge, NOT dataset evidence. Always presented as a prediction."""
from __future__ import annotations

from functools import lru_cache

import yaml

from recipebot.settings import get_settings


@lru_cache
def _doc() -> dict:
    return yaml.safe_load((get_settings().config_dir / "roles.yaml").read_text(encoding="utf-8"))


def roles_of(*names: str) -> list[str]:
    """Roles for the first name that has an entry (try the product name, then base, then parent)."""
    table = _doc()["roles"]
    for n in names:
        if n and n in table:
            return list(table[n])
    return []


def role_label(role: str) -> str:
    return _doc()["role_labels"].get(role, role)


def loss_effect(role: str) -> str:
    return _doc()["loss_effects"].get(role, "")


def adds_text(name: str) -> str:
    return _doc()["adds"].get(name, "")


def compare(x_roles: list[str], y_roles: list[str]) -> tuple[list[str], list[str], str]:
    """-> (roles the substitute covers, roles it does not, note). Unknown roles on either side give no verdict."""
    if not x_roles:
        return [], [], "the role of the original ingredient is not documented, so only the recipe evidence is used"
    if not y_roles:
        return [], [], "the role of this substitute is not documented, so only the recipe evidence is used"
    # "flavor" is too generic to justify a swap (a banana has flavour too): judge fit on the specific jobs, and only fall back
    # to "flavor" when it is the original's only documented role.
    jobs = [r for r in x_roles if r != "flavor"] or x_roles
    covered = [r for r in jobs if r in y_roles]
    missing = [r for r in jobs if r not in y_roles]
    return covered, missing, ""
