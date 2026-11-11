"""Rule-based ingredient-line parser.

One cleaned ingredient line ->  quantity, unit, canonical product name, and its decomposition:

    "2 tablespoons garlic powder, divided"
        name   = "garlic powder"   exact product (used for strict matching)
        base   = "garlic"          product without its processing form
        form   = "powder"
        parent = "garlic"          category/variety parent ("red onion" -> "onion", "cheddar cheese" -> "cheese")

Why name / base / form / parent: the brief says "treat different forms or products as different when the
distinction affects the match". Default matching uses name or parent (so garlic powder is NOT garlic, but a red
onion IS an onion); `base` lets a caller opt in to "any form of garlic" and the answer can say so.

Nothing here guesses silently: lines that yield no name are returned with status "no_name".
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from recipebot.domain.lexicon import Lexicon, get_lexicon

# --------------------------------------------------------------------------- units
# canonical unit -> (kind, factor to base unit). Base units: grams for mass, millilitres for volume.
_UNIT_DEFS: dict[str, tuple[str, float]] = {
    "tsp": ("volume", 4.92892), "tbsp": ("volume", 14.7868), "fl oz": ("volume", 29.5735),
    "cup": ("volume", 236.588), "pint": ("volume", 473.176), "quart": ("volume", 946.353),
    "gallon": ("volume", 3785.41), "ml": ("volume", 1.0), "l": ("volume", 1000.0), "dl": ("volume", 100.0),
    "oz": ("mass", 28.3495), "lb": ("mass", 453.592), "g": ("mass", 1.0), "kg": ("mass", 1000.0),
    "mg": ("mass", 0.001),
}
_COUNT_UNITS = [
    "can", "package", "packet", "clove", "slice", "stick", "bunch", "head", "sprig", "stalk", "pinch",
    "dash", "piece", "jar", "bottle", "container", "sheet", "strip", "fillet", "rib", "ear", "leaf",
    "handful", "drop", "cube", "envelope", "loaf", "bag", "box", "link", "carton", "tub", "tube",
    "splash", "sprinkle", "cutlet", "wedge", "round", "bulb", "knob", "sachet", "pouch", "turn", "grind",
]
_UNIT_ALIASES: dict[str, str] = {
    "teaspoon": "tsp", "tsp": "tsp", "tsps": "tsp",
    "tablespoon": "tbsp", "tbsp": "tbsp", "tbsps": "tbsp", "tbs": "tbsp", "tbl": "tbsp", "tbls": "tbsp",
    "fluid ounce": "fl oz", "fl oz": "fl oz", "fluid oz": "fl oz", "floz": "fl oz",
    "cup": "cup", "c": "cup", "pint": "pint", "pt": "pint", "quart": "quart", "qt": "quart",
    "gallon": "gallon", "gal": "gallon", "milliliter": "ml", "millilitre": "ml", "ml": "ml",
    "liter": "l", "litre": "l", "l": "l", "deciliter": "dl", "dl": "dl",
    "ounce": "oz", "oz": "oz", "pound": "lb", "lb": "lb", "lbs": "lb",
    "gram": "g", "g": "g", "gm": "g", "kilogram": "kg", "kg": "kg", "milligram": "mg", "mg": "mg",
    "pkg": "package",
}
_UNIT_MODIFIERS = {"heaping", "scant", "level", "rounded", "generous", "large", "small", "medium", "big",
                   "good", "whole", "thick", "thin", "healthy", "full", "heaped"}
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1, "half": 0.5,
                 "dozen": 12}

_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d*\.\d+|\d+)"
_QTY_RE = re.compile(rf"^\s*({_NUM})(?:\s*(?:to|-|–|—|or)\s*({_NUM}))?(?!\d)")
_PACK_RE = re.compile(rf"^\s*\(\s*({_NUM})\s*(?:-|\s)?\s*([a-zA-Z. ]+?)\s*\)\s*")
_HYPHEN_SIZE_RE = re.compile(
    r"(?<![\w/])(\d+(?:\.\d+)?)[- ](ounce|oz|pound|lb|gram|g|ml|milliliter|liter|litre|fluid ounce)s?\b"
    r"(?=\s+(?:can|package|packet|jar|bottle|bag|box|container|carton|tub|tube|pkg|pouch)s?\b)",
    re.IGNORECASE,
)
_JUICE_OF_RE = re.compile(r"^(juice|zest|grated zest|grated rind|rind|peel|grated peel)s? of (.+)$", re.IGNORECASE)
_DIMENSION_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:/\d+)?\s*-?\s*(?:(?:to|-)\s*\d+(?:\.\d+)?(?:/\d+)?\s*-?\s*)?(?:inch|inches|in\.|cm|centimeter)s?\b(?:-\w+)?",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_OPTIONAL_RE = re.compile(r"\(?\boptional\b\)?", re.IGNORECASE)
_NOISE_OR = re.compile(r"^(?:or\s+)?(?:to taste|as needed|more|less|so|additional|as desired|if desired|"
                       r"as required|to your taste|\d)", re.IGNORECASE)
_ALT_LEAD = re.compile(r"^(?:or|and/or)\s+(?:you can use|you could use|substitute|use|try|try using|sub)?\s*", re.IGNORECASE)
_CONJUNCTIONS = {"and", "or", "with", "plus"}
# "1 cup/200 g sugar", "4 tablespoons/60 ml olive oil": the second (metric) measure repeats the first
_DUAL_UNIT_RE = re.compile(
    r"(?<=[a-z])\s*/\s*\d+(?:\.\d+)?\s*(?:g|grams?|ml|milliliters?|millilitres?|kg|l|liters?|litres?|oz|ounces?|lbs?|pounds?)\b",
    re.IGNORECASE)
_ALT_STOPWORDS = {"etc", "other", "others", "similar", "your favorite", "favorite", "any", "whatever"}
_SEASONINGS = {"salt", "black pepper", "white pepper", "cayenne pepper", "sugar", "all purpose flour",
               "butter", "olive oil", "vegetable oil", "garlic powder", "onion powder", "paprika",
               "cinnamon", "nutmeg", "pepper"}

_IRREGULAR = {"leaves": "leaf", "halves": "half", "loaves": "loaf", "knives": "knife", "wolves": "wolf",
              "cookies": "cookie", "brownies": "brownie", "smoothies": "smoothie", "veggies": "veggie",
              "pies": "pie", "calories": "calorie", "goodies": "goodie", "hoagies": "hoagie",
              "anchovies": "anchovy", "radishes": "radish"}
_NO_SINGULAR = {"molasses", "hummus", "couscous", "asparagus", "citrus", "swiss", "grits", "brussels",
                "cress", "watercress", "lemongrass", "bass", "series", "species", "less", "this", "plus",
                "gas", "bitters", "greens", "mess", "swiss", "dress", "boss", "moss",
                "hibiscus", "octopus", "cactus", "citrus", "focus", "bonus", "status", "famous"}


def singular(word: str) -> str:
    if word in _NO_SINGULAR or len(word) <= 3:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _unit_lookup(word: str) -> str | None:
    w = word.lower().strip(".,;:")
    if w in _UNIT_ALIASES:
        return _UNIT_ALIASES[w]
    s = singular(w)
    if s in _UNIT_ALIASES:
        return _UNIT_ALIASES[s]
    if s in _UNIT_DEFS or s in _COUNT_UNITS:
        return s
    return None


def unit_info(unit_text: str | None) -> tuple[str, str, float | None] | None:
    """Surface unit -> (canonical unit, kind, grams-or-ml factor). Count units have factor None."""
    if not unit_text:
        return None
    u = _unit_lookup(unit_text)
    if u is None:
        return None
    if u in _UNIT_DEFS:
        kind, factor = _UNIT_DEFS[u]
        return u, kind, factor
    return u, "count", None


def _to_float(token: str) -> float:
    token = token.strip()
    if " " in token:  # mixed number "1 1/2"
        whole, frac = token.split(None, 1)
        n, d = frac.split("/")
        return float(whole) + float(n) / float(d)
    if "/" in token:
        n, d = token.split("/")
        return float(n) / float(d) if float(d) else 0.0
    return float(token)


# --------------------------------------------------------------------------- result type
@dataclass
class ParsedIngredient:
    text: str
    status: str = "ok"                       # ok | no_name
    qty_min: float | None = None
    qty_max: float | None = None
    unit: str | None = None
    unit_kind: str | None = None             # mass | volume | count | None
    qty_base_min: float | None = None        # grams (mass) or millilitres (volume); None if not convertible
    qty_base_max: float | None = None
    name: str = ""
    base: str = ""
    form: str | None = None
    parent: str = ""
    descriptors: list[str] = field(default_factory=list)
    prep: str | None = None
    optional: bool = False
    alternatives: list[str] = field(default_factory=list)   # canonical names of "or" alternatives
    alt_source: str | None = None            # inline_or | comma_or | paren_or

    def to_row(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- name canonicalisation
def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    return "".join(c for c in t if not unicodedata.combining(c))


def _prep_text(s: str) -> str:
    t = fold(s).lower().replace("&", " and ")
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"[-/]", " ", t)
    t = re.sub(r"[^a-z0-9%_ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_scan(text: str) -> str:
    """Lowercase, accent-folded, punctuation-free line text used for the whole-line restriction scan."""
    return " ".join(singular(w) for w in _prep_text(text).split())


def canonical_name(phrase: str, lex: Lexicon | None = None) -> tuple[str, list[str]]:
    """Phrase -> (canonical product name, removed descriptors)."""
    lex = lex or get_lexicon()
    t = _prep_text(phrase)
    if not t:
        return "", []
    for p in lex.protected:  # protect multiword product names (and their plurals) before removing descriptors
        if p in t:
            t = re.sub(rf"\b{re.escape(p)}(?:s|es)?\b", lambda m: m.group(0).replace(" ", "_"), t)
    removed: list[str] = []

    def drop_phrase(m: re.Match[str]) -> str:
        removed.append(m.group(0))
        return " "

    t = lex.descriptor_phrase_re.sub(drop_phrase, t)
    tokens: list[str] = []
    for tok in t.split():
        if tok in lex.descriptors or singular(tok) in lex.descriptors:
            removed.append(tok)
        elif re.fullmatch(r"\d+%?|\d+x\d+|x", tok):
            continue
        else:
            tokens.append(tok)
    words: list[str] = []
    for tok in tokens:
        words.extend(singular(w) for w in tok.split("_"))
    while len(words) > 1 and words[-1] in lex.trailing_noise and words[-3:] != ["half", "and", "half"]:
        removed.append(words.pop())
    while words and words[0] in _CONJUNCTIONS:
        words.pop(0)
    while words and words[-1] in _CONJUNCTIONS and words[-3:] != ["half", "and", "half"]:
        words.pop()
    name = " ".join(words)
    name = lex.synonyms.get(name, name)
    name = _collapse(name, lex)
    return name, removed


def _collapse(name: str, lex: Lexicon) -> str:
    for base, mods in lex.collapse.items():
        if name == base or not name.endswith(" " + base):
            continue
        prefix = name[: -len(base)].strip()
        if prefix in mods or all(w in mods for w in prefix.split()):
            return base
    return name


def decompose(name: str, lex: Lexicon | None = None) -> tuple[str, str | None, str]:
    """name -> (base, form, parent)."""
    lex = lex or get_lexicon()
    if name in lex.parent_overrides:                      # a reviewed human decision beats every rule
        base, form, _ = _decompose_rules(name, lex)
        return base, form, lex.parent_overrides[name]
    return _decompose_rules(name, lex)


def _decompose_rules(name: str, lex: Lexicon) -> tuple[str, str | None, str]:
    parent = name
    for par, variants in lex.varieties.items():
        if name != par and name.endswith(" " + par) and name[: -len(par)].strip() in variants:
            parent = par
            break
    if name in lex.base_overrides:
        base, form = lex.base_overrides[name]
        return base, form, parent
    if name in lex.atomic:
        return name, None, parent
    toks = name.split()
    if len(toks) > 1 and toks[-1] in lex.form_suffixes:
        return " ".join(toks[:-1]), toks[-1], parent
    for pre in lex.form_prefixes:
        if name.startswith(pre + " ") and len(name) > len(pre) + 1:
            return name[len(pre) + 1:], pre, parent
    # form-less names ending in a head word belong to that parent ("maui onion" -> onion)
    composite = bool(set(toks) & {"and", "or", "with", "plus"})   # "chili oil and lime" is two things, not a lime
    if (parent == name and len(toks) > 1 and toks[-1] in lex.head_parents and name not in lex.distinct_products
            and not composite):
        parent = toks[-1]
    return name, None, parent


# --------------------------------------------------------------------------- line parsing
def _split_alternatives(head: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s+(?:or|and/or)\s+", head) if p.strip()]
    if len(parts) > 1:
        first, last = parts[0].split(), parts[-1].split()
        if len(first) == 1 and len(last) >= 2:  # "chicken or vegetable stock" -> "chicken stock"
            parts[0] = f"{first[0]} {' '.join(last[1:])}"
    return parts


def _parse_quantity(s: str) -> tuple[float | None, float | None, str]:
    m = _QTY_RE.match(s)
    if m:
        lo = _to_float(m.group(1))
        hi = _to_float(m.group(2)) if m.group(2) else None
        return lo, hi, s[m.end():].strip()
    first, _, rest = s.partition(" ")
    if first.lower() in _WORD_NUMBERS and rest:
        if first.lower() in ("a", "an") and not (
            _unit_lookup(rest.split()[0]) or rest.split()[0].lower() in _UNIT_MODIFIERS
        ):
            return 1.0, None, rest.strip()  # "a pinch", "an egg"
        return float(_WORD_NUMBERS[first.lower()]), None, rest.strip()
    return None, None, s


def _is_lonely(head: str, lex: Lexicon) -> bool:
    """True when the text before a comma is only descriptors/adjectives, so the real ingredient comes after it."""
    name = canonical_name(head, lex)[0]
    if not name:
        return True
    return all(t in lex.lonely_modifiers for t in name.split())


def _make_name(phrase: str, lex: Lexicon) -> tuple[str, list[str]]:
    name, removed = canonical_name(phrase, lex)
    return name, removed


def parse_ingredient_line(text: str, lex: Lexicon | None = None) -> list[ParsedIngredient]:
    lex = lex or get_lexicon()
    return [p for p in _parse(text, lex)]


@lru_cache(maxsize=400_000)
def parse_cached(text: str) -> tuple[ParsedIngredient, ...]:
    return tuple(_parse(text, get_lexicon()))


def _parse(text: str, lex: Lexicon) -> list[ParsedIngredient]:
    s = text.strip()
    optional = bool(_OPTIONAL_RE.search(s))
    s = _OPTIONAL_RE.sub(" ", s)
    s = re.sub(r"\[[^\[\]]*\]", " ", s)       # [square-bracket notes] are asides, like parentheses ("chicken stock [for a vegetarian soup, ...]")
    s = re.sub(r"\s+,", ",", re.sub(r"\s+", " ", s)).strip(" ,;")
    s = _HYPHEN_SIZE_RE.sub(lambda m: f"({m.group(1)} {m.group(2)})", s)
    s = _DIMENSION_RE.sub(" ", s)
    s = _DUAL_UNIT_RE.sub("", s)

    if re.match(r"^(?:special )?equipment\b", s, re.IGNORECASE):  # Epicurious "Special equipment: ..." lines
        return [ParsedIngredient(text=text, status="not_ingredient", optional=optional)]
    both = re.match(r"^(?:zest and juice|juice and zest) of (.+)$", s, re.IGNORECASE)
    if both:  # "Zest and juice of 1 lemon" -> two ingredients
        return _parse(f"Zest of {both.group(1)}", lex) + _parse(f"Juice of {both.group(1)}", lex)

    # "Juice of 1 lemon" -> qty 1, name "lemon juice"
    jm = _JUICE_OF_RE.match(s)
    juice_kind = None
    if jm:
        juice_kind = jm.group(1).lower().replace("grated ", "").replace("rind", "zest").replace("peel", "zest")
        s = jm.group(2)

    qmin, qmax, rest = _parse_quantity(s)
    pack_size: tuple[float, str] | None = None
    pm = _PACK_RE.match(rest)
    if pm and _unit_lookup(pm.group(2).strip()) in _UNIT_DEFS:
        pack_size = (_to_float(pm.group(1)), _unit_lookup(pm.group(2).strip()))  # type: ignore[arg-type]
        rest = rest[pm.end():]
    # unit (skip size words if they precede a unit)
    toks = rest.split()
    i = 0
    while i + 1 < len(toks) and toks[i].lower() in _UNIT_MODIFIERS and (
        toks[i + 1].lower() in _UNIT_MODIFIERS or _unit_lookup(toks[i + 1])
    ):
        i += 1
    unit = None
    if toks and _unit_lookup(toks[i]) and (qmin is not None or _unit_lookup(toks[i]) in ("pinch", "dash", "splash", "handful", "sprinkle", "knob")):
        unit = _unit_lookup(toks[i])
        i += 1
    elif len(toks) > i + 1 and f"{toks[i].lower()} {toks[i + 1].lower()}" in _UNIT_ALIASES and qmin is not None:
        unit = _UNIT_ALIASES[f"{toks[i].lower()} {toks[i + 1].lower()}"]
        i += 2
    if unit == "fl oz" and toks and toks[i:i + 1] and toks[i].lower() in ("oz", "ounce", "ounces"):
        i += 1
    if unit in _UNIT_DEFS and i < len(toks) and _unit_lookup(toks[i]) in _COUNT_UNITS:
        i += 1  # "1/4 ounce package yeast": the container word is not part of the name
    rest = " ".join(toks[i:])
    if rest.startswith("("):  # "4 cups (2 pints) white chocolate ice cream": drop the parenthetical size note
        rest = re.sub(r"^\([^()]*\)\s*", "", rest)
    rest = re.sub(r"^of\s+", "", rest, flags=re.IGNORECASE)
    if unit == "clove" and not rest.strip(" ,"):  # "2 whole cloves" is the spice, not a garlic count
        unit, rest = None, "clove"

    # alternatives hidden in parentheses: "(or you can use plum, apricot, etc.)"
    alts_raw: list[str] = []
    alt_source = None
    for pm2 in _PAREN_RE.finditer(rest):
        inner = pm2.group(1).strip()
        if re.match(r"^(?:or|and/or)\s+", inner, re.IGNORECASE) and not _NOISE_OR.match(inner):
            alts_raw.extend(a for a in re.split(r",|\bor\b", _ALT_LEAD.sub("", inner)) if a.strip())
            alt_source = "paren_or"
    rest = _PAREN_RE.sub(" ", rest)
    rest = re.sub(r"\s+", " ", rest).strip(" ,;")

    # comma handling: merge descriptor-only heads ("skinless, boneless chicken breast")
    parts = [p.strip() for p in rest.split(",")]
    head, idx = parts[0], 1
    while idx < len(parts) and _is_lonely(head, lex):
        head = f"{head} {parts[idx]}"
        idx += 1
    prep = ", ".join(p for p in parts[idx:] if p) or None
    if prep and re.match(r"^(?:or|and/or)\s", prep, re.IGNORECASE) and not _NOISE_OR.match(prep):
        alts_raw.extend(a for a in re.split(r",|\bor\b", _ALT_LEAD.sub("", prep)) if a.strip())
        alt_source = alt_source or "comma_or"
        prep = None

    head = juice_kind and f"{head} {juice_kind}" or head
    # "each salt and pepper" / "salt and pepper" -> two ingredients
    names_heads: list[str]
    and_split = re.split(r"\s+and\s+", head)
    if len(and_split) == 2 and all(len(p.split()) <= 5 for p in and_split) and (
        qmin is None or re.search(r"\beach\b", text, re.IGNORECASE)
    ):
        c = [canonical_name(p, lex)[0] for p in and_split]
        names_heads = and_split if any(x in _SEASONINGS for x in c) and all(c) else [head]
    else:
        names_heads = [head]

    results: list[ParsedIngredient] = []
    for nh in names_heads:
        inline = _split_alternatives(nh)
        primary_phrase = inline[0] if inline else nh
        name, removed = canonical_name(primary_phrase, lex)
        alt_phrases = inline[1:] + alts_raw
        alt_names = []
        for ap in alt_phrases:
            an, _ = canonical_name(ap, lex)
            if an and an not in _ALT_STOPWORDS and an != name and an not in alt_names:
                alt_names.append(an)
        src = alt_source or ("inline_or" if len(inline) > 1 else None)
        base, form, parent = decompose(name, lex) if name else ("", None, "")
        if name in lex.not_ingredient:
            results.append(ParsedIngredient(text=text, status="not_ingredient", optional=optional))
            continue
        item = ParsedIngredient(
            text=text, status="ok" if name else "no_name", qty_min=qmin, qty_max=qmax, unit=unit,
            unit_kind=("count" if unit in _COUNT_UNITS else _UNIT_DEFS[unit][0]) if unit else None,
            name=name, base=base, form=form, parent=parent, descriptors=removed, prep=prep,
            optional=optional, alternatives=alt_names, alt_source=src if alt_names else None,
        )
        _fill_base_quantity(item, pack_size)
        results.append(item)
    return results


def _fill_base_quantity(item: ParsedIngredient, pack: tuple[float, str] | None) -> None:
    """Quantities in grams/ml, only for mass/volume units. A pack size (2 x 10.75 oz can) is multiplied out."""
    if item.qty_min is None:
        return
    mult_unit, size = item.unit, 1.0
    if pack is not None:  # "2 (10.75 ounce) cans" -> 21.5 oz
        size, mult_unit = pack[0], pack[1]
        item.unit, item.unit_kind = mult_unit, _UNIT_DEFS[mult_unit][0]
    if mult_unit in _UNIT_DEFS:
        factor = _UNIT_DEFS[mult_unit][1]
        item.qty_base_min = round(item.qty_min * size * factor, 3)
        item.qty_base_max = round(item.qty_max * size * factor, 3) if item.qty_max is not None else None
