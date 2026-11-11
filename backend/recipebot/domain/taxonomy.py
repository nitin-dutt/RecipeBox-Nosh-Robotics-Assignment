"""Restriction taxonomy: ingredient name -> {class: "yes" | "maybe"}.

Text-only and deliberately conservative. It answers "does the ingredient TEXT contain a restricted item?"
and never claims an allergen-free or medically safe result.

Line-level rule with "or" alternatives ("butter or margarine"):
    every option flagged "yes"          -> yes
    some options flagged, some clear    -> maybe   (the cook can choose a compliant option)
    all options clear                   -> none
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from recipebot.settings import get_settings

YES, MAYBE = "yes", "maybe"
_RANK = {None: 0, "none": 0, MAYBE: 1, YES: 2}
_LEVEL = {0: None, 1: MAYBE, 2: YES}


def _phrase_re(items: list[str]) -> re.Pattern[str] | None:
    items = sorted({i for i in items if i}, key=len, reverse=True)
    if not items:
        return None
    alt = "|".join(re.escape(i).replace(r"\ ", " ") for i in items)
    return re.compile(rf"(?<![a-z])(?:{alt})(?![a-z])")


def _exempt_span_re(items: list[str]) -> re.Pattern[str] | None:
    items = sorted({i for i in items if i}, key=len, reverse=True)
    if not items:
        return None
    alt = "|".join(re.escape(i).replace(r"\ ", " ") for i in items)
    return re.compile(rf"(?<![a-z])(?:{alt})(?![a-z])(?:[\s,/-]+[a-z][a-z-]*){{0,2}}")


@dataclass(frozen=True)
class ClassRule:
    name: str
    label: str
    yes_re: re.Pattern[str] | None
    maybe_re: re.Pattern[str] | None
    guard_re: re.Pattern[str] | None
    exempt_re: re.Pattern[str] | None
    exempt_span_re: re.Pattern[str] | None
    demote_re: re.Pattern[str] | None
    includes: tuple[str, ...]


class Taxonomy:
    def __init__(self, path: Path | None = None) -> None:
        path = path or get_settings().config_dir / "taxonomy.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw = doc["classes"]
        self.instruction_guard_res = [re.compile(g) for g in doc.get("instruction_guards", [])]
        self.rules: dict[str, ClassRule] = {}
        for name, spec in raw.items():
            self.rules[name] = ClassRule(
                name=name, label=spec["label"],
                yes_re=_phrase_re(spec.get("terms") or []),
                maybe_re=_phrase_re(spec.get("maybe_terms") or []),
                guard_re=_phrase_re(spec.get("guards") or []),
                exempt_re=_phrase_re(spec.get("exempt") or []),
                exempt_span_re=_exempt_span_re(spec.get("exempt") or []),
                demote_re=_phrase_re(spec.get("demote") or []),
                includes=tuple(spec.get("includes") or []),
            )
        # direct (non-composite) result per class, resolved lazily with includes
        self.classes = list(self.rules)
        cfg = path.parent
        # Reviewed LLM labels: may only RAISE a level (none -> maybe -> yes), never lower one the rules set.
        self.overlay: dict[str, dict[str, str]] = {}
        overlay_file = cfg / "labels" / "ingredient_labels.json"
        if overlay_file.exists():
            self.overlay = {n: v["classes"] for n, v in json.loads(overlay_file.read_text(encoding="utf-8"))["labels"].items()}
        # Explicit human decisions: they win over both the rules and the LLM (the only way to lower a level).
        self.human: dict[str, dict[str, str]] = {}
        overrides_file = cfg / "ingredient_overrides.yaml"
        if overrides_file.exists():
            self.human = (yaml.safe_load(overrides_file.read_text(encoding="utf-8")) or {}).get("classes", {}) or {}

    @staticmethod
    def _without_exempt(rule: ClassRule, text: str) -> str:
        """Remove each 'exempt word + the (up to) two words it modifies' ("vegetarian chicken nuggets"), keeping the rest of the line.
        The old rule dropped the WHOLE line when an exempt word appeared anywhere, which cleared "chicken stock [for a vegetarian
        soup, use garlic broth]". Now only the modified phrase is exempt."""
        return rule.exempt_span_re.sub(" ", text) if rule.exempt_span_re else text

    def _direct(self, rule: ClassRule, name: str) -> str | None:
        name = self._without_exempt(rule, name)
        text = rule.guard_re.sub(" ", name) if rule.guard_re else name
        # `demote` phrases may not trigger "yes" (e.g. "crab boil seasoning" is a spice mix), but still count as "maybe"
        yes_text = rule.demote_re.sub(" ", text) if rule.demote_re else text
        if rule.yes_re and rule.yes_re.search(yes_text):
            return YES
        if rule.maybe_re and rule.maybe_re.search(text):
            return MAYBE
        if rule.demote_re and rule.demote_re.search(text):
            return MAYBE
        return None

    def apply_instruction_guards(self, text: str) -> str:
        """Remove uses of ingredient words as verbs/shapes/colours ('cream the shortening', 'walnut sized')."""
        for g in self.instruction_guard_res:
            text = g.sub(" ", text)
        return text

    def match_term(self, cls: str, text: str) -> tuple[str, str] | None:
        """(term, owning class) of the first term of `cls` (or a class it includes) found in `text`, for evidence."""
        rule = self.rules[cls]
        text = self._without_exempt(rule, text)
        cleaned = rule.guard_re.sub(" ", text) if rule.guard_re else text
        for rx in (rule.yes_re, rule.maybe_re, rule.demote_re):
            if rx is not None and (m := rx.search(cleaned)):
                return m.group(0), cls
        for inc in rule.includes:
            if (t := self.match_term(inc, text)):
                return t
        return None

    def term_survives_guards(self, owner: str, term: str, window: str) -> bool:
        """True if `term` is still present in `window` after the owner's guard phrases are removed."""
        rule = self.rules[owner]
        cleaned = rule.guard_re.sub(" ", window) if rule.guard_re else window
        return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", cleaned) is not None

    def classify_name(self, name: str, rules_only: bool = False) -> dict[str, str]:
        direct = {c: self._direct(r, name) for c, r in self.rules.items()}
        if not rules_only and name in self.overlay:      # reviewed LLM labels: raise-only, BEFORE composites are derived
            for c, lvl in self.overlay[name].items():
                if c in direct and _RANK[lvl] > _RANK[direct[c]]:
                    direct[c] = lvl
        human = self.human.get(name, {}) if not rules_only else {}
        for c, lvl in human.items():                       # a human decision on a BASE class feeds the composites (meat, vegetarian...)
            if c in self.rules and not self.rules[c].includes:
                direct[c] = lvl if _RANK[lvl] else None
        out: dict[str, str] = {}
        for c, r in self.rules.items():
            status = direct[c]
            for inc in r.includes:
                inc_status = direct[inc]
                if inc_status == YES:
                    status = YES
                elif inc_status == MAYBE and status is None:
                    status = MAYBE
            if status:
                out[c] = status
        # composites that include composites (vegetarian -> meat -> pork...) are resolved by a second pass
        for c, r in self.rules.items():
            for inc in r.includes:
                s = out.get(inc)
                if s == YES:
                    out[c] = YES
                elif s == MAYBE and out.get(c) is None:
                    out[c] = MAYBE
        if human:                                        # a human decision on a COMPOSITE class wins, in either direction
            for c, lvl in human.items():
                if c in self.rules and self.rules[c].includes:
                    if _RANK[lvl] == 0:
                        out.pop(c, None)
                    else:
                        out[c] = lvl
        return out

    def classify_item(self, name: str, alternatives: list[str] | tuple[str, ...] = (),
                      rules_only: bool = False) -> dict[str, str]:
        options = [self.classify_name(name, rules_only), *(self.classify_name(a, rules_only) for a in alternatives)]
        out: dict[str, str] = {}
        for c in self.classes:
            sts = [o.get(c) for o in options]
            if all(s == YES for s in sts):
                out[c] = YES
            elif any(s is not None for s in sts):
                out[c] = MAYBE
        return out


@lru_cache
def get_taxonomy() -> Taxonomy:
    return Taxonomy()
