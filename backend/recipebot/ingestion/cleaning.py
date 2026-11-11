"""Pure, unit-tested cleaning rules for the scraped Recipe Box records.

Design rules
------------
* Raw text is never modified; callers keep it next to the cleaned text.
* Every rule reports a name through the ``hit`` callback so the pipeline can count
  exactly how many lines/records each rule touched (data_quality_report.json).
* Rules are idempotent: cleaning already-clean text changes nothing.
"""
from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field

Hit = Callable[[str], None]

# --------------------------------------------------------------------------- text normalisation

_ADVERT = re.compile(r"\s*\bADVERTISEMENT\b\s*")

_FRACTIONS = {
    "¼": "1/4", "½": "1/2", "¾": "3/4", "⅓": "1/3", "⅔": "2/3",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}
_FRACTION_CLASS = "".join(_FRACTIONS)
_FRAC_AFTER_DIGIT = re.compile(rf"(\d)([{_FRACTION_CLASS}])")
_FRAC = re.compile(f"[{_FRACTION_CLASS}]")

_SPACE_CHARS = re.compile("[      ]")
_ZERO_WIDTH = re.compile("[​‌‍﻿]")
_INLINE_WS = re.compile(r"[ \t]{2,}")

# The scraper wrote accents as malformed entities (typos, truncations). The raw files contain
# only ~30 distinct ones, so they are mapped explicitly rather than guessed. Longest first.
_BROKEN_ENTITIES = {
    "&ntild;eo": "ño", "&ntilede;": "ñ", "&nitlde;": "ñ", "&ntlde;": "ñ", "&entilde;": "ñ",
    "&ntild;": "ñ", "&241;": "ñ", "&ocute;": "ó", "&ircirc;": "î", "&eauce;": "é",
    "&eaacute;": "é", "&eactue;": "é", "&ecute;": "é", "&eacut;": "é", "&eucate;": "ú",
    "&eouml;": "ö", "&euuml;": "ü", "&egave;": "è", "Kahl&a;": "Kahlúa",
    "&Deg;": "°", "&176;": "°", "&def;": "°", "&151;": "—", "&endash;": "–", "&Reg;": "®",
}
_BROKEN_ENTITY = re.compile("|".join(re.escape(k) for k in sorted(_BROKEN_ENTITIES, key=len, reverse=True)))
# Brand names with a stray semicolon: "M&M;'s", "S&B;", "PB&J;" -> "M&M's", "S&B", "PB&J"
_AMP_BRAND = re.compile(r"(?<=\w)&(\w{1,2});")

_BRAND_MARKS = re.compile("[®™]")

# Epicurious markup debris
_RECIPELINK = re.compile(r'<?/?epi:recipelink(?:\s+id="\d+")?>?', re.IGNORECASE)
_GARBLED_TAG = re.compile(r'<([A-Za-z][\w:-]*)((?:\s+[\w:-]+="")+)\s*>')
_ANY_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_STRAY_LT = re.compile(r"<(?=[\s,.;)])")


def _rebuild_garbled(m: re.Match[str]) -> str:
    """`<buttermilk corn="" bread="">` was the text 'buttermilk corn bread' eaten by an HTML parser."""
    words = [m.group(1), *re.findall(r'([\w:-]+)=""', m.group(2))]
    return " ".join(words)


def strip_markup(s: str, hit: Hit) -> str:
    if "<" not in s and "epi:" not in s.lower():
        return s
    t = _RECIPELINK.sub("", s)
    t = _GARBLED_TAG.sub(_rebuild_garbled, t)
    t = _ANY_TAG.sub("", t)
    t = _STRAY_LT.sub("", t)
    if t != s:
        hit("markup_stripped")
    return t


def decode_entities(s: str, hit: Hit) -> str:
    if "&" not in s:
        return s

    t = _BROKEN_ENTITY.sub(lambda m: _BROKEN_ENTITIES[m.group(0)], s)
    t = _AMP_BRAND.sub(lambda m: "&" + m.group(1), t)
    t = html.unescape(t)
    if t != s:
        hit("html_entity_decoded")
    return t


def repair_lossy_char(s: str, hit: Hit) -> str:
    """Food Network text has '¿' where an apostrophe, hyphen, dash or ® used to be.

    Only repaired where the context makes the original unambiguous; anything else is left
    untouched and flagged ``lossy_char_unresolved``.
    """
    if "¿" not in s:
        return s
    t = re.sub(r"(?<=\d)¿(?=[FC]\b)", "°", s)        # "425¿F/220¿C"
    t = re.sub(r"(?<=\d)¿(?=\d)", "/", t)            # "1¿2 cup", "1 1¿4 teaspoon"
    t = re.sub(r"(?<=\d) ¿(?=\d+-inch)", " x ", t)   # "9 ¿13-inch pan"
    t = re.sub(r"(?<=%)¿(?=\d)", "-", t)             # "60%¿70% cacao"
    t = re.sub(r"(?<=\s)¿(?=\s)", "—", t)            # "butter ¿ heat it"
    t = re.sub(r"(?<=[A-Za-z])¿s\b", "'s", t)

    def trailing(m: re.Match[str]) -> str:
        word = m.group(1)
        if word[0].isupper():  # "Creations¿ refrigerated" -> a brand mark, removed
            return word
        if word.endswith("s"):  # "confectioners¿ sugar" -> plural possessive
            return word + "'"
        return m.group(0)

    t = re.sub(r"\b(\w+)¿(?=\s|$|[,.;])", trailing, t)
    t = re.sub(r"(?<=[A-Za-z])¿(?=[A-Za-z])", "-", t)  # "towel¿lined"
    hit("lossy_char_repaired" if "¿" not in t else "lossy_char_unresolved")
    return t


def normalize_text(s: str, hit: Hit) -> str:
    """Single-line normalisation applied to titles, ingredient lines and instruction lines."""
    t = decode_entities(s, hit)
    t = strip_markup(t, hit)
    t = _ZERO_WIDTH.sub("", t)
    t = _SPACE_CHARS.sub(" ", t)
    if _FRAC.search(t):
        t = _FRAC.sub(lambda m: _FRACTIONS[m.group(0)], _FRAC_AFTER_DIGIT.sub(r"\1 \2", t))
        hit("unicode_fraction_normalized")
    t = repair_lossy_char(t, hit)
    if _BRAND_MARKS.search(t):
        t = _BRAND_MARKS.sub("", t)
        hit("brand_mark_removed")
    t = _INLINE_WS.sub(" ", t)
    return t.strip()


# --------------------------------------------------------------------------- ingredients

# ALL-CAPS lines that are real ingredients (abbreviations), not section labels
_CAPS_INGREDIENTS = {"EVOO"}
_HEADER_FOR = re.compile(r"^for (the )?[a-z' -]{2,40}$", re.IGNORECASE)


def is_section_header(line: str) -> bool:
    """`For the sauce:`, `FILLING`, `Topping:` - labels, not ingredients."""
    if not line or re.search(r"\d", line):
        return False
    words = line.split()
    if line.endswith(":") and len(words) <= 8:
        return True
    letters = re.sub(r"[^A-Za-z]", "", line)
    if line in _CAPS_INGREDIENTS:
        return False
    if len(letters) > 3 and line.upper() == line and len(words) <= 6:
        return True
    return bool(_HEADER_FOR.match(line) and len(words) <= 6)


def clean_ingredient_line(raw: str, hit: Hit) -> str:
    t = raw
    if "ADVERTISEMENT" in t:
        t = _ADVERT.sub(" ", t)
        hit("advertisement_removed")
    return normalize_text(t, hit)


@dataclass
class CleanIngredients:
    lines: list[str] = field(default_factory=list)   # cleaned ingredient lines (no headers)
    raws: list[str] = field(default_factory=list)    # the raw line behind each cleaned line
    groups: list[str | None] = field(default_factory=list)  # section label per cleaned line


def clean_ingredients(raw_lines: list[str] | None, hit: Hit) -> CleanIngredients:
    out = CleanIngredients()
    group: str | None = None
    for raw in raw_lines or []:
        line = clean_ingredient_line(raw, hit)
        if not line:
            hit("empty_ingredient_line_dropped")
            continue
        if is_section_header(line):
            group = line.rstrip(":").strip()
            hit("section_header_moved_to_group")
            continue
        out.lines.append(line)
        out.raws.append(raw)
        out.groups.append(group)
    return out


# --------------------------------------------------------------------------- instructions

_BOILERPLATE_LINES = [
    re.compile(r"^watch how to make this recipe\.?$", re.IGNORECASE),
    re.compile(r"^photograph(s)? by [A-Z].{0,100}$"),
    re.compile(r"^photo(s)? (by|courtesy of) [A-Z][^.]{0,60}$", re.IGNORECASE),
    re.compile(r"^recipe (is )?courtesy of .{0,120}$", re.IGNORECASE),
]
_TRAILING_CREDIT = re.compile(r"\s*Photograph(?:s)? by [A-Z][^\n]{0,100}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def dedupe_repeated_halves(lines: list[str]) -> tuple[list[str], bool]:
    """Epicurious often stores the whole instruction text twice."""
    n = len(lines)
    if n >= 2 and n % 2 == 0 and lines[: n // 2] == lines[n // 2:]:
        return lines[: n // 2], True
    return lines, False


def split_sentences(line: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(line) if s.strip()]


@dataclass
class CleanInstructions:
    text: str
    n_paragraphs: int
    n_sentences: int


def clean_instructions(raw: str | None, hit: Hit) -> CleanInstructions:
    lines = [ln.strip() for ln in (raw or "").split("\n") if ln.strip()]
    lines, was_dup = dedupe_repeated_halves(lines)
    if was_dup:
        hit("duplicate_instructions_removed")
    cleaned: list[str] = []
    for ln in lines:
        ln = normalize_text(ln, hit)
        if not ln:
            continue
        if any(p.match(ln) for p in _BOILERPLATE_LINES):
            hit("boilerplate_line_removed")
            continue
        stripped = _TRAILING_CREDIT.sub("", ln)
        if stripped != ln:
            hit("boilerplate_line_removed")
            ln = stripped.strip()
        if ln:
            cleaned.append(ln)
    return CleanInstructions(
        text="\n".join(cleaned),
        n_paragraphs=len(cleaned),
        n_sentences=sum(len(split_sentences(ln)) for ln in cleaned),
    )


# --------------------------------------------------------------------------- whole record

@dataclass
class CleanRecord:
    id: str
    source: str
    title: str | None
    title_raw: str | None
    ingredients: list[str]
    ingredient_raws: list[str]
    ingredient_groups: list[str | None]
    ingredients_raw: list[str]
    instructions: str
    instructions_raw: str
    n_steps_paragraphs: int
    n_steps_sentences: int
    picture_link: str | None
    has_title: bool
    has_ingredients: bool
    has_instructions: bool
    is_complete: bool
    flags: list[str]

    def to_json(self, dup_group_id: str | None, dup_group_size: int) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "title_raw": self.title_raw,
            "ingredients": self.ingredients,
            "ingredient_groups": self.ingredient_groups,
            "ingredients_raw": self.ingredients_raw,
            "instructions": self.instructions,
            "instructions_raw": self.instructions_raw,
            "n_steps_paragraphs": self.n_steps_paragraphs,
            "n_steps_sentences": self.n_steps_sentences,
            "picture_link": self.picture_link,
            "is_complete": self.is_complete,
            "dup_group_id": dup_group_id,
            "dup_group_size": dup_group_size,
            "flags": self.flags,
        }


def clean_record(rec_id: str, source: str, raw: dict, count: Hit) -> CleanRecord:
    """Clean one non-empty raw record. ``count`` receives rule names (global tally)."""
    flags: list[str] = []

    def hit(rule: str) -> None:
        count(rule)
        if rule not in flags:
            flags.append(rule)

    title_raw = raw.get("title")
    title = normalize_text(title_raw, hit) if title_raw else ""
    ing = clean_ingredients(raw.get("ingredients"), hit)
    ins = clean_instructions(raw.get("instructions"), hit)

    for text in (title, *ing.lines, ins.text):
        if "�" in text:
            hit("replacement_char_present")
            break

    has_title, has_ing, has_ins = bool(title), bool(ing.lines), bool(ins.text)
    return CleanRecord(
        id=rec_id,
        source=source,
        title=title or None,
        title_raw=title_raw,
        ingredients=ing.lines,
        ingredient_raws=ing.raws,
        ingredient_groups=ing.groups,
        ingredients_raw=list(raw.get("ingredients") or []),
        instructions=ins.text,
        instructions_raw=raw.get("instructions") or "",
        n_steps_paragraphs=ins.n_paragraphs,
        n_steps_sentences=ins.n_sentences,
        picture_link=raw.get("picture_link"),
        has_title=has_title,
        has_ingredients=has_ing,
        has_instructions=has_ins,
        is_complete=has_title and has_ing and has_ins,
        flags=flags,
    )
