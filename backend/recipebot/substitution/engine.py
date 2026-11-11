"""Evidence-based substitution (Task 3).

Evidence, strongest first, always from the LOCAL dataset:
  direct    recipes whose own ingredient lines list the substitute as an "or" alternative ("1 cup cream or half and half")
  variant   among the recipes MOST SIMILAR to the one being changed, those that do not use the ingredient: what do they use?
  indirect  weaker versions of the above (1-2 recipes, or only in more distantly related recipes)
Culinary role knowledge (config/roles.yaml) is general knowledge, used for a compatibility check and for the "expected effects"
text, and always labelled as a prediction. Nothing is ever claimed as verified unless a cited recipe shows it.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from recipebot.domain.ingredient import normalize_for_scan, parse_ingredient_line
from recipebot.domain.instruction_scan import scan_instructions
from recipebot.domain.lexicon import get_lexicon
from recipebot.domain.models import (
    AdaptedLine, Computation, EvidenceRecipe, FunnelStep, IngredientLine, RestrictionCheck, SubstituteCandidate,
    SubstitutionRequest, SubstitutionResult, TargetAdaptation,
)
from recipebot.domain.taxonomy import get_taxonomy
from recipebot.store.db import Database, get_db
from recipebot.substitution import roles
from recipebot.substitution.rewrite import Change, mention_words, rewrite_instructions
from recipebot.tools.semantic import get_index
from recipebot.tools.terms import TermResolver

VOCAB_MIN_LINES = 5                   # ignore ingredient names seen on fewer lines than this (parsing debris)
MIN_FIT = 0.4                         # a substitute must cover at least 40% of the documented jobs of what it replaces
MIN_EVIDENCE_RECIPES = 2              # a weak substitute needs at least two recipes behind it before it is even proposed
CLOSE_RANK = 100                      # a neighbour in the top 100 counts as a close variant of the recipe being changed
UNIVERSAL = {"salt", "black pepper", "white pepper", "water", "ice", "cooking spray"}
DISCLAIMER = ("Restriction checks read only the ingredient and instruction text of the adapted recipe; they do not certify it "
              "as safe (packaged ingredients can contain allergens that the text does not mention). Expected effects are "
              "general culinary knowledge, not outcomes verified in the dataset.")
EVIDENCE_POLICY = ("Local dataset first: alternatives that recipes themselves list, and what the most similar recipes WITHOUT the "
                   "ingredient use instead. No external web search was used.")


@dataclass
class Item:
    position: int
    item_idx: int
    name: str
    base: str
    parent: str
    alternatives: list[str]
    qty: float | None
    unit: str | None
    text: str
    original: str


@dataclass
class Target:
    item: Item
    why: str


@dataclass
class NbItem:
    name: str
    base: str
    parent: str
    text: str
    alternatives: list[str] = field(default_factory=list)


def _fmt(n: float) -> str:
    return f"{n:.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------------------------------------ loading
def _load_items(db: Database, rid: str) -> list[Item]:
    rows = db.query(
        "SELECT p.position, p.item_idx, coalesce(p.name, ''), coalesce(p.base, ''), coalesce(p.parent, ''), "
        "coalesce(p.alternatives, []), p.qty_min, p.unit, i.text, i.text_raw FROM ingredients_parsed p "
        "JOIN recipe_ingredients i ON i.recipe_id = p.recipe_id AND i.position = p.position "
        "WHERE p.recipe_id = ? AND p.status = 'ok' ORDER BY p.position, p.item_idx", [rid])
    return [Item(*r) for r in rows]


def _resolve_targets(db: Database, rid: str, items: list[Item], req: SubstitutionRequest,
                     resolver: TermResolver) -> list[Target]:
    found: dict[tuple[int, int], Target] = {}
    for term in req.remove:
        rt = resolver.resolve(term)
        members = set(rt.members)
        pat = re.compile(rt.text_pattern, re.IGNORECASE)
        for it in items:
            if it.name in members or it.parent in members or it.base in members or (
                    rt.strategy == "text" and pat.search(it.text)):
                found.setdefault((it.position, it.item_idx), Target(it, f"you asked to remove {term}"))
    if req.exclude_classes:
        ph = ", ".join("?" for _ in req.exclude_classes)
        st = "" if req.restriction_mode == "strict" else " AND status = 'yes'"
        by_pos = defaultdict(list)
        for it in items:
            by_pos[it.position].append(it)
        for pos, cls, status in db.query(
                f'SELECT position, "class", status FROM ingredient_flags WHERE recipe_id = ? AND "class" IN ({ph}){st}',
                [rid, *req.exclude_classes]):
            for it in by_pos.get(pos, [])[:1]:
                key = (it.position, it.item_idx)
                why = f"flagged {cls} ({'definite' if status == 'yes' else 'possible'})"
                if key in found:
                    if why not in found[key].why:
                        found[key].why += "; " + why
                else:
                    found[key] = Target(it, why)
    return sorted(found.values(), key=lambda t: (t.item.position, t.item.item_idx))


def _same_dish(db: Database, index, rid: str, title: str | None, have: set[str], limit: int = 300) -> list[tuple[str, float]]:
    """Recipes whose TITLE contains every word of this recipe's title (the same dish), with their embedding similarity.

    Embedding similarity alone can bury a relevant variant whose ingredient list differs (an applesauce banana bread with
    zucchini ranked 1,214th). Same-dish recipes are exactly the related recipes a cook would compare."""
    import numpy as np
    from recipebot.tools.search import query_tokens
    tokens = query_tokens(title)
    if not tokens:
        return []
    clauses = " AND ".join("regexp_matches(lower(title), ?)" for _ in tokens)
    ids = [r[0] for r in db.query(f"SELECT id FROM recipes WHERE is_complete AND id <> ? AND {clauses} LIMIT 3000",
                                   [rid, *[rf"\b{re.escape(t)}" for t in tokens]])]
    ids = [i for i in ids if i in index.row_of and i not in have]
    if len(ids) < 5:
        return []
    base = index.dense.vector(index.row_of[rid])
    base = base / (np.linalg.norm(base) or 1.0)
    sims = []
    for i in ids:
        v = index.dense.vector(index.row_of[i])
        sims.append((i, float(base @ (v / (np.linalg.norm(v) or 1.0)))))
    sims.sort(key=lambda t: -t[1])
    return sims[:limit]


def _neighbour_items(db: Database, ids: list[str]) -> dict[str, list[NbItem]]:
    out: dict[str, list[NbItem]] = defaultdict(list)
    for chunk_start in range(0, len(ids), 400):
        chunk = ids[chunk_start:chunk_start + 400]
        ph = ", ".join("?" for _ in chunk)
        for rid, name, base, parent, text, alts in db.query(
                f"SELECT p.recipe_id, coalesce(p.name, ''), coalesce(p.base, ''), coalesce(p.parent, ''), i.text, coalesce(p.alternatives, []) "
                f"FROM ingredients_parsed p JOIN recipe_ingredients i ON i.recipe_id = p.recipe_id AND i.position = p.position "
                f"WHERE p.recipe_id IN ({ph}) AND p.status = 'ok' ORDER BY p.recipe_id, p.position", chunk):
            out[rid].append(NbItem(name, base, parent, text, list(alts)))
    return out


def _ykey(name: str, parent: str) -> str:
    lex = get_lexicon()
    if parent and parent != name and parent in lex.head_parents and name not in lex.distinct_products:
        return parent
    return name


# ------------------------------------------------------------------------------------------------ evidence
@dataclass
class Evidence:
    n_without: int                 # recipes in the evidence scope that do not use the ingredient
    n_with: int
    scope: str                     # "closest" (top CLOSE_RANK neighbours) or "wider" (all neighbours examined)
    candidates: list[SubstituteCandidate]
    omission: list[EvidenceRecipe]
    dropped_restriction: int = 0
    dropped_role: int = 0
    dropped_same_family: int = 0
    notes: list[str] = field(default_factory=list)


def _conflicts(name: str, classes: list[str], mode: str) -> bool:
    flags = get_taxonomy().classify_name(name)
    return any(c in flags and (mode == "strict" or flags[c] == "yes") for c in classes)


def _same_family(y: str, x: Item, vocab_info: dict[str, tuple[str, str]]) -> bool:
    """A substitute must not be the ingredient itself, one of its forms, or another product of its allergen family
    (replacing egg with 'egg substitute', which is mostly egg white, does not remove the egg)."""
    y_base = vocab_info.get(y, ("", ""))[0]
    if y == x.name or (x.base and (y_base == x.base or y == x.base)):
        return True
    tax = get_taxonomy()
    x_yes = {c for c, s in tax.classify_name(x.name).items() if s == "yes" and c not in ("vegan", "vegetarian", "meat")}
    y_yes = {c for c, s in tax.classify_name(y).items() if s == "yes"}
    return bool(x_yes & y_yes)


def _alt_edges(db: Database, x_name: str) -> dict[str, tuple[int, list[tuple[str, str]]]]:
    """Substitutes that recipes themselves list as 'or' alternatives of x (either direction): {alt: (n_recipes, [(id, line)])}."""
    fwd = db.query(
        "SELECT alt, count(DISTINCT recipe_id) FROM (SELECT recipe_id, unnest(alternatives) AS alt FROM ingredients_parsed "
        "WHERE status = 'ok' AND name = ?) GROUP BY alt", [x_name])
    rev = db.query(
        "SELECT name, count(DISTINCT recipe_id) FROM ingredients_parsed WHERE status = 'ok' AND list_contains(alternatives, ?) "
        "GROUP BY name", [x_name])
    counts: Counter[str] = Counter()
    for alt, n in [*fwd, *rev]:
        if alt and alt != x_name:
            counts[alt] += int(n)
    out = {}
    for alt, n in counts.most_common(40):
        ex = db.query(
            "SELECT p.recipe_id, i.text_raw FROM ingredients_parsed p JOIN recipe_ingredients i ON i.recipe_id = p.recipe_id "
            "AND i.position = p.position WHERE (p.name = ? AND list_contains(p.alternatives, ?)) OR (p.name = ? AND "
            "list_contains(p.alternatives, ?)) ORDER BY p.recipe_id LIMIT 3", [x_name, alt, alt, x_name])
        out[alt] = (n, [(r[0], r[1]) for r in ex])
    return out


def build_evidence(db: Database, target: Target, base_keys: set[str], neighbours: list[tuple[str, float, int]],
                   nb_items: dict[str, list[NbItem]], titles: dict[str, str | None], restricted_ok: set[str],
                   req: SubstitutionRequest, vocab_info: dict[str, tuple[str, str]]) -> Evidence:
    x = target.item
    x_names, x_bases = {x.name}, ({x.base} if x.base else set())

    def has_x(rid: str) -> bool:
        return any(i.name in x_names or (i.base and i.base in x_bases) for i in nb_items.get(rid, []))

    with_x = [n for n in neighbours if has_x(n[0])]
    free = [n for n in neighbours if not has_x(n[0]) and n[0] in restricted_ok]
    free_close = [n for n in free if n[2] <= CLOSE_RANK]
    with_close = [n for n in with_x if n[2] <= CLOSE_RANK]
    use_close = len(free_close) >= req.min_support            # enough close variants: judge by them, not by distant relatives
    scope_free, scope_with = (free_close, with_close) if use_close else (free, with_x)
    scope_ids_free = {n[0] for n in scope_free}
    scope_ids_with = {n[0] for n in scope_with}

    cnt_scope_f: Counter[str] = Counter()
    cnt_scope_w: Counter[str] = Counter()
    cnt_close_f: Counter[str] = Counter()
    support: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    names_for_key: dict[str, Counter[str]] = defaultdict(Counter)

    def keys_of(rid: str):
        seen = {}
        for i in nb_items.get(rid, []):
            if i.name in x_names or (i.base and i.base in x_bases) or not i.name:
                continue
            k = _ykey(i.name, i.parent)
            if k in base_keys or k in UNIVERSAL or i.name not in vocab_info:     # vocabulary check is on the PRODUCT name
                continue
            seen.setdefault(k, i)
        return seen

    for rid, sim, rank in free:
        for k, it in keys_of(rid).items():
            names_for_key[k][it.name] += 1
            support[k].append((rid, sim, it.text))
            if rid in scope_ids_free:
                cnt_scope_f[k] += 1
            if rank <= CLOSE_RANK:
                cnt_close_f[k] += 1
    for rid, _sim, _rank in with_x:
        if rid in scope_ids_with:
            for k in keys_of(rid):
                cnt_scope_w[k] += 1

    edges = _alt_edges(db, x.name)
    alt_nb: Counter[str] = Counter()            # similar recipes that themselves list Y as an "or" alternative of X
    for rid, _sim, _rank in neighbours:
        if _rank > CLOSE_RANK:
            continue                                   # an alternative seen only in distant recipes is not evidence here
        hit = set()
        for i in nb_items.get(rid, []):
            if i.name == x.name:
                hit.update(i.alternatives)
            elif x.name in i.alternatives and i.name:
                hit.add(i.name)
        for y in hit:
            alt_nb[y] += 1
    cand_keys = set(support) | {k for k in edges if k in vocab_info}
    ev = Evidence(len(scope_free), len(scope_with), "closest" if use_close else "wider", [], [])
    x_roles = roles.roles_of(x.name, x.base, x.parent)
    classes = list(req.exclude_classes)
    for k in cand_keys:
        n_alt, alt_ex = edges.get(k, (0, []))
        n_f, n_w = cnt_scope_f.get(k, 0), cnt_scope_w.get(k, 0)
        n_close = cnt_close_f.get(k, 0)
        if n_f == 0 and n_alt == 0 and n_close == 0:
            continue
        display = names_for_key[k].most_common(1)[0][0] if names_for_key.get(k) else k      # the actual product name
        if _same_family(k, x, vocab_info) or _same_family(display, x, vocab_info):
            ev.dropped_same_family += 1
            continue
        p_f = n_f / len(scope_free) if scope_free else 0.0
        p_w = n_w / len(scope_with) if scope_with else 0.0
        if classes and (_conflicts(k, classes, req.restriction_mode) or _conflicts(display, classes, req.restriction_mode)):
            ev.dropped_restriction += 1
            continue
        y_roles = roles.roles_of(display, k)       # the product name first ("nutritional yeast"), then its group ("yeast")
        covered, missing, note = roles.compare(x_roles, y_roles)
        if x_roles and y_roles and not covered:
            ev.dropped_role += 1
            continue
        lift_note = ""
        if n_alt == 0 and p_f - p_w <= 0:
            # Not more common where the ingredient is absent. That is no evidence by itself, but a role-compatible ingredient
            # that does occur in similar recipes without it is still worth showing as a LEAD, clearly labelled as weak.
            if not (covered and n_f >= 2):
                continue
            lift_note = (f"also common in similar recipes that keep {x.name} ({p_w:.0%}), so its presence is weak evidence "
                         "that it replaces it")
        n_alt_similar = alt_nb.get(k, 0)
        tier = ("direct" if n_alt_similar >= req.min_support else
                "variant" if n_close >= req.min_support else "indirect")
        if n_alt and n_alt_similar < req.min_support:
            lift_note = "; ".join(x_ for x_ in (lift_note, f"listed as an alternative in {n_alt} recipes overall but only "
                                                 f"{n_alt_similar} among the recipes most similar to this one") if x_)
        evidence = [EvidenceRecipe(id=r, title=titles.get(r), line=line, kind="alternative_listed") for r, line in alt_ex[:2]]
        for rid, sim, text in sorted(support.get(k, []), key=lambda t: -t[1])[:3]:
            evidence.append(EvidenceRecipe(id=rid, title=titles.get(rid), similarity=round(sim, 3), line=text, kind="variant_without"))
        ev.candidates.append(SubstituteCandidate(
            substitute=display, tier=tier, n_supporting=n_f, prevalence_without=round(p_f, 3), prevalence_with=round(p_w, 3),
            n_alternative_lines=n_alt, n_alternative_similar=n_alt_similar, roles_covered=covered, roles_missing=missing,
            role_note="; ".join(x for x in (note, lift_note) if x), evidence=evidence))
    rank = {"direct": 0, "variant": 1, "indirect": 2}
    def fits(c: SubstituteCandidate) -> bool:
        f = _fit(c)
        return c.tier == "direct" or (f is not None and f >= MIN_FIT)

    ev.candidates.sort(key=lambda c: (0 if fits(c) else 1, rank[c.tier], 0 if c.roles_covered else 1,
                                      -(c.prevalence_without - c.prevalence_with), -c.n_supporting,
                                      -c.n_alternative_lines, -len(c.roles_covered), c.substitute))
    ev.candidates = ev.candidates[:6]
    for rid, sim, _rank in sorted(scope_free, key=lambda n: -n[1])[:3]:
        ev.omission.append(EvidenceRecipe(id=rid, title=titles.get(rid), similarity=round(sim, 3),
                                          line=f"(a similar recipe that has no {x.name})", kind="omits"))
    return ev


def _fit(c: SubstituteCandidate) -> float | None:
    n = len(c.roles_covered) + len(c.roles_missing)
    return len(c.roles_covered) / n if n else None


def pick(ev: Evidence, verdict: str) -> SubstituteCandidate | None:
    """Apply a substitute only on supported/weak evidence, and never one whose role is unknown on indirect evidence alone."""
    if verdict == "supported":
        return ev.candidates[0]
    if verdict != "weak":
        return None
    for c in ev.candidates:
        f = _fit(c)
        lift = c.prevalence_without - c.prevalence_with
        support = c.n_supporting + c.n_alternative_similar      # alternatives found in OTHER kinds of dish do not count
        if f is not None and f >= MIN_FIT and support >= MIN_EVIDENCE_RECIPES and (lift > 0 or c.tier == "direct"):
            return c
    return None


def verdict_for(ev: Evidence) -> tuple[str, str]:
    where = f"{ev.n_without} {'closest' if ev.scope == 'closest' else 'similar'} recipes without it"
    if not ev.candidates:
        if ev.n_without >= 3:
            return "none", f"no substitute stands out: {where} exist, but none shares a common replacement"
        return "none", "the dataset has too few similar recipes without this ingredient to suggest anything"
    best = ev.candidates[0]
    lift = best.prevalence_without - best.prevalence_with
    fit = _fit(best)
    if fit is not None and fit < MIN_FIT and best.tier in ("direct", "variant"):
        return "weak", (f"{best.substitute} is used in similar recipes without it, but it covers only "
                        f"{', '.join(best.roles_covered) or 'none'} of the jobs the original does "
                        f"(missing: {', '.join(best.roles_missing)})")
    if best.tier == "direct":
        return "supported", (f"{best.n_alternative_lines} recipes list {best.substitute} as an alternative of it, "
                             "including similar recipes")
    if best.tier == "variant" and best.prevalence_without >= 0.25 and lift >= 0.15:
        return "supported", (f"{best.n_supporting} of the {where} use {best.substitute} ({best.prevalence_without:.0%}) against "
                             f"{best.prevalence_with:.0%} of those that keep it")
    if best.tier == "variant" and best.prevalence_without >= 0.25:
        return "weak", (f"{best.substitute} is in {best.prevalence_without:.0%} of the {where}, but also in "
                        f"{best.prevalence_with:.0%} of those that keep it, so it looks like an ordinary ingredient of this "
                        "kind of dish, not a replacement")
    if ev.n_without >= 8 and best.prevalence_without < 0.2:
        return "conflicting", (f"the {where} use many different things; the most common, {best.substitute}, "
                               f"appears in only {best.prevalence_without:.0%}")
    return "weak", (f"only {best.n_supporting} of the {where} use {best.substitute}"
                    + (f" and {best.n_alternative_lines} list it as an alternative" if best.n_alternative_lines else ""))


# ------------------------------------------------------------------------------------------------ assembly
def _pattern_for(item: Item) -> re.Pattern[str]:
    words = sorted({w for w in (item.name, *item.name.split()[-1:]) if w}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es)?\b", re.IGNORECASE)


def _replacement_line(target: Target, cand: SubstituteCandidate) -> tuple[str, EvidenceRecipe | None]:
    """Amount policy: copy it from the closest evidence recipe; for a listed alternative keep the original amount (1:1)."""
    variants = [e for e in cand.evidence if e.kind == "variant_without"]
    if cand.tier == "direct" or not variants:
        src = next((e for e in cand.evidence if e.kind == "alternative_listed"), None)
        head = target.item.text.split(",")[0]
        swapped = _pattern_for(target.item).sub(cand.substitute, head, count=1)
        if swapped != head:
            return swapped, src
        qty = (_fmt(target.item.qty) + (" " + target.item.unit if target.item.unit else "")) if target.item.qty else ""
        return (f"{qty} {cand.substitute}".strip(), src)
    best = variants[0]
    return best.line, best


def _restriction_check(db: Database, lines: list[str], instructions: str, classes: list[str], mode: str) -> list[RestrictionCheck]:
    tax = get_taxonomy()
    out = []
    per_line = []
    for text in lines:
        flags: dict[str, str] = {}
        for p in parse_ingredient_line(text):
            if p.status == "ok":
                for c, s in tax.classify_item(p.name, p.alternatives).items():
                    flags[c] = s if flags.get(c) != "yes" else "yes"
        for c, s in tax.classify_name(normalize_for_scan(text)).items():
            flags.setdefault(c, "maybe")
        per_line.append((text, flags))
    instr = {c: snip for c, snip in scan_instructions(instructions)}
    for cls in classes:
        yes = [t for t, f in per_line if f.get(cls) == "yes"]
        maybe = [t for t, f in per_line if f.get(cls) == "maybe"]
        mention = instr.get(cls)
        if yes:
            out.append(RestrictionCheck(restriction=cls, status="still_present", lines=yes))
        elif maybe or mention:
            out.append(RestrictionCheck(restriction=cls, status="possible",
                                        lines=maybe + ([f"(instructions) {mention}"] if mention else [])))
        else:
            out.append(RestrictionCheck(restriction=cls, status="no_term_found"))
    return out


def suggest_substitution(req: SubstitutionRequest, db: Database | None = None, index=None) -> SubstitutionResult:
    db = db or get_db()
    if not req.remove and not req.exclude_classes:
        raise ValueError("Give at least one ingredient to remove or one restriction class to avoid.")
    rec = db.query("SELECT title, instructions, dup_group_id FROM recipes WHERE id = ?", [req.recipe_id])
    if not rec:
        raise KeyError(f"unknown recipe id {req.recipe_id}")
    title, instructions, dup_group = rec[0]
    resolver = TermResolver(db)
    items = _load_items(db, req.recipe_id)
    targets = _resolve_targets(db, req.recipe_id, items, req, resolver)
    base_keys = {_ykey(i.name, i.parent) for i in items}
    original_lines = {}
    for i in items:
        original_lines.setdefault(i.position, (i.text, i.original))

    if not targets:
        checks = _restriction_check(db, [t for t, _ in original_lines.values()], instructions or "",
                                    list(req.exclude_classes), req.restriction_mode)
        return _result(req, title, "adapted", "No ingredient line of this recipe matches the request, so nothing needed to change.",
                       [], [AdaptedLine(position=p, text=t, changed=False) for p, (t, _) in sorted(original_lines.items())],
                       instructions or "", [], [], checks, [], False, db, 0, 0, 0)

    index = index or get_index(db)
    complete = [r[0] for r in db.query("SELECT id FROM recipes WHERE is_complete")]
    found = index.similar(req.recipe_id, k=req.neighbours + 30, allowed_ids=complete)
    dup_ids = {r[0] for r in db.query("SELECT id FROM recipes WHERE dup_group_id = ?", [dup_group])} if dup_group else set()
    neighbours = [(h.id, float(h.dense_score or 0.0), n) for n, h in enumerate((h for h in found if h.id not in dup_ids), start=1)]
    neighbours = neighbours[:req.neighbours]
    seen_ids = {n[0] for n in neighbours} | dup_ids
    same_dish = _same_dish(db, index, req.recipe_id, title, seen_ids)
    # a same-dish recipe counts as a CLOSE variant even if embeddings ranked it far away
    neighbours += [(i, sim, CLOSE_RANK) for i, sim in same_dish]
    n_same_dish = len(same_dish)
    ids = [n[0] for n in neighbours]
    nb_items = _neighbour_items(db, ids)
    titles = {r[0]: r[1] for r in db.query(f"SELECT id, title FROM recipes WHERE id IN ({', '.join('?' for _ in ids)})", ids)}
    if req.exclude_classes:
        ph = ", ".join("?" for _ in req.exclude_classes)
        st = "" if req.restriction_mode == "strict" else " AND status = 'yes'"
        bad = {r[0] for r in db.query(f'SELECT DISTINCT recipe_id FROM recipe_flags WHERE "class" IN ({ph}){st} AND recipe_id IN '
                                      f"({', '.join('?' for _ in ids)})", [*req.exclude_classes, *ids])}
        restricted_ok = set(ids) - bad
    else:
        restricted_ok = set(ids)
    vocab_info = {r[0]: (r[1] or "", r[2] or "") for r in db.query(
        "SELECT name, base, parent FROM ingredient_vocab WHERE n_lines >= ?", [VOCAB_MIN_LINES])}

    adapted_targets: list[TargetAdaptation] = []
    changes: list[Change] = []
    replaced: dict[int, str | None] = {}
    levels: dict[int, str] = {}
    effects: list[str] = []
    issues: list[str] = []
    total_without = 0
    for t in targets:
        ev = build_evidence(db, t, base_keys, neighbours, nb_items, titles, restricted_ok, req, vocab_info)
        verdict, reason = verdict_for(ev)
        total_without = max(total_without, ev.n_without)
        chosen = pick(ev, verdict)
        new_line, source = (None, None)
        if chosen:
            new_line, source = _replacement_line(t, chosen)
        note_bits = []
        if ev.dropped_restriction:
            note_bits.append(f"{ev.dropped_restriction} candidate(s) dropped because they fall under the excluded class")
        if ev.dropped_same_family:
            note_bits.append(f"{ev.dropped_same_family} candidate(s) dropped because they are the same ingredient family")
        if ev.dropped_role:
            note_bits.append(f"{ev.dropped_role} candidate(s) dropped because they cannot do the same job (role check)")
        if note_bits:
            reason += " (" + "; ".join(note_bits) + ")"
        adapted_targets.append(TargetAdaptation(
            line=IngredientLine(position=t.item.position, text=t.item.text, original=t.item.original), ingredient=t.item.name,
            why=t.why, n_similar_without=ev.n_without, n_similar_with=ev.n_with, verdict=verdict, verdict_reason=reason,
            candidates=ev.candidates, chosen=chosen.substitute if chosen else None, replacement_line=new_line,
            replacement_source=source, omission_evidence=ev.omission if not chosen else []))
        if chosen:
            replaced[t.item.position] = new_line
            levels[t.item.position] = verdict
            changes.append(Change(mention_words(t.item.name), chosen.substitute))
            effects.append(_effect_text(t.item, chosen))
        else:
            replaced.setdefault(t.item.position, None)
            changes.append(Change(mention_words(t.item.name), None))
            if req.remove and any(r for r in req.remove) and "you asked to remove" in t.why:
                effects.append(_effect_text(t.item, None))
            issues.append(f"No supported substitute for '{t.item.name}' ({t.why}); it stays in the recipe." if
                          "flagged" in t.why and "you asked" not in t.why else
                          f"'{t.item.name}' was removed WITHOUT a supported replacement.")

    adapted_lines = []
    for pos, (text, orig) in sorted(original_lines.items()):
        if pos in replaced:
            new = replaced[pos]
            target_flagged_only = any(t.item.position == pos and "you asked to remove" not in t.why for t in targets)
            if new is None and target_flagged_only and not any(t.item.position == pos and "you asked to remove" in t.why for t in targets):
                adapted_lines.append(AdaptedLine(position=pos, text=text, changed=False))   # restriction line with no substitute stays
            elif new is not None:
                adapted_lines.append(AdaptedLine(position=pos, text=new, changed=True, original=text,
                                                 evidence_level=levels.get(pos)))
            else:       # new is None and the user asked for removal: the line is dropped, and SAID so (it used to vanish silently)
                adapted_lines.append(AdaptedLine(position=pos, text="", changed=True, original=text))
        else:
            adapted_lines.append(AdaptedLine(position=pos, text=text, changed=False))
    keep_original = [c for c, t in zip(changes, targets) if c.replacement is None and "you asked to remove" not in t.why]
    active = [c for c in changes if c not in keep_original]
    revised, edits = rewrite_instructions(instructions or "", active)
    checks = _restriction_check(db, [l.text for l in adapted_lines], revised, list(req.exclude_classes), req.restriction_mode)
    for c in checks:
        if c.status == "still_present":
            issues.append(f"The adapted recipe still contains {c.restriction}: " + "; ".join(c.lines[:4]))
        elif c.status == "possible":
            issues.append(f"The adapted recipe may still contain {c.restriction}: " + "; ".join(c.lines[:4]))
    verdicts = [t.verdict for t in adapted_targets]
    n_ok = sum(v in ("supported", "weak") for v in verdicts)
    still = any(c.status == "still_present" for c in checks)
    if n_ok == 0:
        status = "unsupported"
    elif any(v != "supported" for v in verdicts) or still:
        status = "partial"
    else:
        status = "adapted"
    external = any(v in ("weak", "conflicting", "none") for v in verdicts)
    summary = _summary(status, adapted_targets)
    return _result(req, title, status, summary, adapted_targets, adapted_lines, revised, edits, effects, checks, issues, external,
                   db, len(neighbours), total_without, n_same_dish)


def _effect_text(item: Item, cand: SubstituteCandidate | None) -> str:
    x_roles = roles.roles_of(item.name, item.base, item.parent)
    if cand is None:
        lost = x_roles
        if not lost:
            return f"PREDICTION: removing {item.name} changes the dish; its role is not documented, so the effect is unknown."
        bits = "; ".join(roles.loss_effect(r) for r in lost if roles.loss_effect(r))
        return (f"PREDICTION (general culinary knowledge, not verified): {item.name} usually {', '.join(roles.role_label(r) for r in lost[:3])}. "
                f"Without it: {bits}.")
    y_roles = roles.roles_of(cand.substitute)
    adds = roles.adds_text(cand.substitute)
    lost = [r for r in x_roles if y_roles and r not in y_roles]
    s = f"PREDICTION (general culinary knowledge, not verified): replacing {item.name} with {cand.substitute}"
    if adds:
        s += f": {cand.substitute} {adds}"
    if lost:
        s += ". Not fully covered: " + "; ".join(roles.loss_effect(r) for r in lost[:3] if roles.loss_effect(r))
    elif not x_roles or not y_roles:
        s += ". Its effect on texture and taste is not documented here"
    return s.rstrip(".") + "."


def _summary(status: str, targets: list[TargetAdaptation]) -> str:
    def tail(t) -> str:
        if t.chosen:
            return f" -> {t.chosen}"
        removed = "you asked to remove" in t.why
        if t.candidates:
            return " (leads only, no replacement applied; the line was removed)" if removed else " (leads only, nothing applied)"
        return " (no replacement; the line was removed)" if removed else ""
    parts = [f"{t.ingredient}: {t.verdict}" + tail(t) for t in targets]
    head = {"adapted": "Adapted, every change is supported by similar recipes.",
            "partial": "Partly adapted: at least one change rests on weak evidence or could not be made.",
            "unsupported": "No supported adaptation."}[status]
    return head + " " + "; ".join(parts) + "."


def _result(req, title, status, summary, targets, adapted, revised, edits, effects, checks, issues, external, db,
            n_neighbours, n_without, n_same_dish=0) -> SubstitutionResult:
    total = db.scalar("SELECT count(*) FROM recipes")
    complete = db.scalar("SELECT count(*) FROM recipes WHERE is_complete")
    comp = Computation(
        tool="suggest_substitution", description=f"Adapting recipe {req.recipe_id}", population=f"complete recipes ({complete:,})",
        filters=req.model_dump(), funnel=[FunnelStep(label="recipes in the dataset", remaining=int(total)),
                                         FunnelStep(label="complete recipes", remaining=int(complete)),
                                         FunnelStep(label="most similar recipes examined (embedding neighbours + same-dish titles)", remaining=n_neighbours),
                                         FunnelStep(label="of those, recipes with the same dish name", remaining=n_same_dish),
                                         FunnelStep(label="of those, without the ingredient (largest over targets)", remaining=n_without)],
        n_analyzed=n_neighbours, denominator=int(complete),
        method=("1) find the lines to change; 2) take the most similar recipes by embedding similarity plus every recipe with the same dish name in its title; 3) split them into those that "
                "use the ingredient and those that do not (and, if a restriction was requested, that satisfy it); 4) a substitute's "
                "support is the share of the 'without' group that contains it, compared with the 'with' group; recipes' own 'or' "
                "alternatives count as direct evidence; 5) role knowledge only checks compatibility and writes predictions. "
                "No language model is involved."),
        sql=("-- alternatives listed by recipes\nSELECT alt, count(DISTINCT recipe_id) FROM (SELECT recipe_id, unnest(alternatives) alt "
             "FROM ingredients_parsed WHERE name = :ingredient) GROUP BY alt;\n-- neighbours: exact cosine search over "
             f"{n_neighbours} recipes"), notes=[EVIDENCE_POLICY])
    return SubstitutionResult(
        status=status, summary=summary, recipe_id=req.recipe_id, recipe_title=title, targets=targets, adapted_ingredients=adapted,
        revised_instructions=revised, instruction_edits=edits, predicted_effects=effects, restriction_check=checks,
        remaining_issues=issues, external_search_suggested=external, evidence_policy=EVIDENCE_POLICY, disclaimer=DISCLAIMER,
        computation=comp)
