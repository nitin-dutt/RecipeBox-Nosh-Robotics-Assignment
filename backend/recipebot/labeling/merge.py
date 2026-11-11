"""Merge the rules and the LLM labels under ONE safety rule:

    The LLM may RAISE a restriction level (none -> maybe -> yes) automatically.
    It may never LOWER a level the rules set. Where it disagrees that way, the rules stay in force and the
    disagreement goes to a human review queue. Only an explicit human decision can lower a level.

Why asymmetric: a wrong "contains" costs one skipped recipe; a wrong "does not contain" can hurt a person.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from recipebot.domain.taxonomy import Taxonomy
from recipebot.labeling.evaluate import BASE_CLASSES
from recipebot.labeling.schema import NameLabel

RANK = {None: 0, "none": 0, "maybe": 1, "yes": 2}

# A new "maybe" on a COMMON ingredient (spaghetti -> egg maybe) would make strict filters exclude thousands of recipes,
# so it needs a human decision. Raising to "yes", or to "maybe" on a rare name, is applied automatically.
COMMON_LINES = 100


@dataclass
class Comparison:
    raises: dict[str, dict[str, str]] = field(default_factory=dict)          # applied automatically: name -> {class: level}
    raises_review: list[dict] = field(default_factory=list)                   # "maybe" on a common name: human decides
    lowers: list[dict] = field(default_factory=list)                          # LLM says less than the rules
    parents: list[dict] = field(default_factory=list)                         # LLM proposes another parent
    not_ingredient: list[dict] = field(default_factory=list)
    agree: int = 0
    reasons: dict[str, str] = field(default_factory=dict)


def compare(labels: dict[str, NameLabel], tax: Taxonomy, rule_parent: dict[str, str], n_lines: dict[str, int],
            use_llm_parents: bool = False) -> Comparison:
    out = Comparison()
    for name, lab in labels.items():
        lines = n_lines.get(name, 0)
        rules = tax.classify_name(name, rules_only=True)
        llm = lab.class_map() if lab.is_ingredient else {}
        changed = False
        for cls in BASE_CLASSES:     # vegetarian/vegan/meat are DERIVED from these; the LLM's own answers for them are ignored
            r, l = rules.get(cls), llm.get(cls)
            if RANK[l] > RANK[r]:
                if l == "yes" or lines < COMMON_LINES:
                    out.raises.setdefault(name, {})[cls] = l          # type: ignore[assignment]
                    out.reasons[name] = lab.reason
                else:
                    out.raises_review.append({"kind": "raise_maybe", "name": name, "class": cls, "n_lines": lines,
                                              "rules_say": r or "none", "llm_says": l, "llm_reason": lab.reason})
                changed = True
            elif RANK[l] < RANK[r]:
                out.lowers.append({"kind": "lower_class", "name": name, "class": cls, "n_lines": lines,
                                   "rules_say": r, "llm_says": l or "none", "llm_reason": lab.reason})
                changed = True
        if not lab.is_ingredient and name in rule_parent:
            out.not_ingredient.append({"kind": "not_ingredient", "name": name, "class": "", "n_lines": lines,
                                       "rules_say": "ingredient", "llm_says": "not an ingredient", "llm_reason": lab.reason})
            changed = True
        elif use_llm_parents and lab.is_ingredient and lab.parent and lab.parent != rule_parent.get(name, name):
            out.parents.append({"kind": "parent", "name": name, "class": "", "n_lines": lines,
                                "rules_say": rule_parent.get(name, name), "llm_says": lab.parent,
                                "llm_reason": lab.reason})
            changed = True
        if not changed:
            out.agree += 1
    return out


# ------------------------------------------------------------------------------------------ overlay (auto-accepted raises)
def write_overlay(path: Path, cmp: Comparison, model: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("labels", {})
    for name, classes in cmp.raises.items():
        merged = dict(existing.get(name, {}).get("classes", {}))
        for c, lvl in classes.items():
            if RANK[lvl] > RANK[merged.get(c)]:
                merged[c] = lvl
        existing[name] = {"classes": merged, "source": f"llm:{model}", "reason": cmp.reasons.get(name, "")}
    doc = {"version": 1, "rule": "raise-only: these labels can add restriction flags, never remove one",
           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "labels": dict(sorted(existing.items()))}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(existing)


# ------------------------------------------------------------------------------------------ human review
REVIEW_COLUMNS = ["kind", "name", "class", "n_lines", "rules_say", "llm_says", "llm_reason", "suggested", "decision"]


def _key(row: dict) -> str:
    return f"{row['kind']}|{row['name']}|{row.get('class', '')}"


def load_decisions(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_review_queue(path: Path, cmp: Comparison, decisions_path: Path, suggest=None, limit: int | None = None) -> int:
    """Rows still awaiting a decision, most impactful first. `suggest(row) -> 'accept'|'reject'|''` pre-fills advice."""
    done = load_decisions(decisions_path)
    rows = [r for r in cmp.raises_review + cmp.lowers + cmp.not_ingredient + cmp.parents if _key(r) not in done]
    rows.sort(key=lambda r: (-r["n_lines"], r["name"]))
    if limit:
        rows = rows[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:             # utf-8-sig so Excel opens it correctly
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "suggested": suggest(r) if suggest else "", "decision": ""})
    return len(rows)


def apply_review(csv_path: Path, decisions_path: Path, overrides_path: Path, reviewer: str) -> dict[str, int]:
    """Record the human decisions from the CSV, then regenerate the overrides file from ALL accepted decisions."""
    decisions = load_decisions(decisions_path)
    stamp = datetime.now(timezone.utc).date().isoformat()
    n_new = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            d = (row.get("decision") or "").strip().lower()
            if d in ("accept", "reject"):
                decisions[_key(row)] = {**{k: row[k] for k in ("kind", "name", "class", "rules_say", "llm_says")},
                                        "decision": d, "reviewer": reviewer, "date": stamp}
                n_new += 1
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(json.dumps(decisions, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    classes: dict[str, dict[str, str]] = {}
    parent: dict[str, str] = {}
    not_ing: list[str] = []
    for d in decisions.values():
        if d["decision"] != "accept":
            continue
        if d["kind"] in ("lower_class", "raise_maybe"):
            classes.setdefault(d["name"], {})[d["class"]] = "none" if d["llm_says"] == "none" else d["llm_says"]
        elif d["kind"] == "parent":
            parent[d["name"]] = d["llm_says"]
        elif d["kind"] == "not_ingredient":
            not_ing.append(d["name"])
    doc = {"_note": "Generated from config/labels/review_decisions.json. Human decisions; they win over rules and LLM.",
           "classes": dict(sorted(classes.items())), "parent": dict(sorted(parent.items())),
           "not_ingredient": sorted(set(not_ing))}
    overrides_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"new_decisions": n_new, "accepted_total": sum(d["decision"] == "accept" for d in decisions.values()),
            "rejected_total": sum(d["decision"] == "reject" for d in decisions.values())}
