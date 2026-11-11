"""Grounding validator: LLM-written text is accepted only if it adds nothing the tool output did not contain.

Checks (all deterministic):
* every number in the rephrasing also appears in the grounded text (counts, percentages, steps, rank numbers);
* every citation [n] refers to a real source of this turn;
* every source that the grounded text cites is still cited (nothing silently dropped);
* no raw recipe ID appears in the prose;
* no safety guarantee is introduced ("safe", "guaranteed", "allergy-free", "hypoallergenic").
If any check fails the caller falls back to the grounded text itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?:/\d+)?")
_CITE = re.compile(r"\[(\d+)\]")
_ID = re.compile(r"\b[0-9a-f]{20,}\b|\b[A-Za-z0-9]{28,}\b")
_SAFE = re.compile(r"\b(100% safe|is safe|are safe|guarantee[ds]?|allergy[- ]free|hypoallergenic|medically safe)\b", re.I)


@dataclass
class Verdict:
    ok: bool
    problems: list[str] = field(default_factory=list)


def numbers(text: str) -> set[str]:
    text = _CITE.sub(" ", text)
    return {n.replace(",", ".").rstrip("0").rstrip(".") if "." in n else n for n in _NUM.findall(text)}


def validate(candidate: str, grounded: str, n_sources: int, require_all_citations: bool = True) -> Verdict:
    problems: list[str] = []
    extra = numbers(candidate) - numbers(grounded)
    if extra:
        problems.append(f"numbers not in the tool output: {sorted(extra)}")
    cited = {int(x) for x in _CITE.findall(candidate)}
    bad = {n for n in cited if n < 1 or n > n_sources}
    if bad:
        problems.append(f"citations to sources that do not exist: {sorted(bad)}")
    dropped = {int(x) for x in _CITE.findall(grounded)} - cited
    if require_all_citations and dropped:
        problems.append(f"citations dropped: {sorted(dropped)}")
    if _ID.search(candidate):
        problems.append("a raw recipe ID appears in the prose")
    if _SAFE.search(candidate) and not _SAFE.search(grounded):
        problems.append("introduces a safety guarantee")
    return Verdict(not problems, problems)
