"""Stage 2 of ingestion: parse every ingredient line and classify restrictions.

Reads  recipe_ingredients            (created by stage 1)
Writes ingredients_parsed            one row per parsed item (a line like "salt and pepper" yields two)
       ingredient_flags              one row per (item, restriction class) hit, status yes|maybe
       recipe_flags                  one row per (recipe, class): yes if any line is a definite hit, else maybe
       ingredient_vocab              one row per canonical name with line/recipe counts
and    data/ingredient_report.json   coverage numbers and the audit lists used to tune the lexicon/taxonomy
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from collections import Counter

import duckdb

from recipebot.domain.ingredient import normalize_for_scan, parse_cached
from recipebot.domain.instruction_scan import scan_instructions
from recipebot.domain.versioning import lexicon_fingerprint
from recipebot.domain.taxonomy import get_taxonomy
from recipebot.settings import Settings, get_settings

log = logging.getLogger("recipebot.ingredients")

# Last-resort net for the most dangerous allergens: scraped text sometimes fuses words ("8 cupspeanut oil",
# "tablespoonsesame seeds"), which defeats whole-word matching. Substring hits become "maybe" (never "yes").
_SUBSTRING_NET = {
    "peanut": ("peanut", "groundnut"),
    "sesame": ("sesame", "tahini"),
    "tree_nut": ("almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia"),
}
_FREE_RE = re.compile(r"\b(?:peanut|nut|tree nut|sesame)[ -]free\b")


def run(cfg: Settings | None = None) -> dict:
    cfg = cfg or get_settings()
    t0 = time.time()
    tax = get_taxonomy()
    con = duckdb.connect(str(cfg.db_path))
    rows = con.execute(
        "SELECT recipe_id, position, text, group_label FROM recipe_ingredients ORDER BY recipe_id, position"
    ).fetchall()
    log.info("parsing %d ingredient lines", len(rows))

    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    parsed_nd, flags_nd = cfg.tmp_dir / "parsed.ndjson", cfg.tmp_dir / "flags.ndjson"
    flag_cache: dict[tuple, dict[str, str]] = {}
    line_cache: dict[str, dict[str, str]] = {}
    status_count: Counter = Counter()
    n_items = 0
    with parsed_nd.open("w", encoding="utf-8") as fp, flags_nd.open("w", encoding="utf-8") as ff:
        for recipe_id, position, text, group in rows:
            items = parse_cached(text)
            # Recall net: a restricted word anywhere in the line (e.g. the 3rd option in "canola, safflower or
            # peanut oil") raises at least "maybe" for classes the item-level parse did not flag.
            line_flags = line_cache.get(text)
            if line_flags is None:
                line_flags = line_cache[text] = tax.classify_name(normalize_for_scan(text))
            item_classes: set[str] = set()
            for idx, p in enumerate(items):
                n_items += 1
                status_count[p.status] += 1
                fp.write(json.dumps({
                    "recipe_id": recipe_id, "position": position, "item_idx": idx, "status": p.status,
                    "name": p.name or None, "base": p.base or None, "form": p.form, "parent": p.parent or None,
                    "qty_min": p.qty_min, "qty_max": p.qty_max, "unit": p.unit, "unit_kind": p.unit_kind,
                    "qty_base_min": p.qty_base_min, "qty_base_max": p.qty_base_max,
                    "optional": p.optional, "alternatives": p.alternatives, "alt_source": p.alt_source,
                    "prep": p.prep, "descriptors": p.descriptors, "group_label": group,
                }, ensure_ascii=False) + "\n")
                if p.status != "ok":
                    continue
                key = (p.name, tuple(p.alternatives))
                flags = flag_cache.get(key)
                if flags is None:
                    flags = flag_cache[key] = tax.classify_item(p.name, p.alternatives)
                for cls, st in flags.items():
                    item_classes.add(cls)
                    ff.write(json.dumps({
                        "recipe_id": recipe_id, "position": position, "item_idx": idx,
                        "class": cls, "status": st, "via": "name", "name": p.name, "optional": p.optional,
                    }) + "\n")
            lowered = text.lower()
            if not _FREE_RE.search(lowered):
                for cls, roots in _SUBSTRING_NET.items():
                    if cls not in line_flags and any(r in lowered for r in roots):
                        line_flags = {**line_flags, cls: "maybe"}
            for cls in line_flags:
                if cls not in item_classes and items:
                    ff.write(json.dumps({
                        "recipe_id": recipe_id, "position": position, "item_idx": 0,
                        "class": cls, "status": "maybe", "via": "text", "name": items[0].name or None,
                        "optional": items[0].optional,
                    }) + "\n")
    log.info("parsed in %.1fs", time.time() - t0)

    # Recall net 3: ingredients that only the instructions mention ("brush with melted butter").
    # Only classes that the ingredient lines did NOT already flag for that recipe are added, always as "maybe".
    already: dict[str, set[str]] = {}
    with flags_nd.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            already.setdefault(d["recipe_id"], set()).add(d["class"])
    instr_nd = cfg.tmp_dir / "instruction_flags.ndjson"
    # The RAW scan (every class an instruction text mentions) takes about 4 minutes and depends only on the taxonomy
    # rules and the instruction texts, so it is cached under exactly that key. Which of those hits are "new" (not
    # already flagged by the ingredient lines) is decided afterwards, because that part changes with every label edit.
    key = hashlib.sha256()
    key.update((cfg.config_dir / "taxonomy.yaml").read_bytes())
    key.update(str(con.execute("SELECT md5(string_agg(id || ':' || instructions, '|' ORDER BY id)) "
                               "FROM recipes WHERE has_instructions").fetchone()[0]).encode())
    raw_cache = cfg.tmp_dir.parent / "cache" / f"instruction_scan_{key.hexdigest()[:16]}.ndjson"
    if not raw_cache.exists():
        raw_cache.parent.mkdir(parents=True, exist_ok=True)
        with raw_cache.open("w", encoding="utf-8") as fr:
            for recipe_id, instructions in con.execute(
                    "SELECT id, instructions FROM recipes WHERE has_instructions").fetchall():
                for cls, snippet in scan_instructions(instructions, tax):
                    fr.write(json.dumps({"recipe_id": recipe_id, "class": cls, "snippet": snippet},
                                        ensure_ascii=False) + "\n")
        log.info("instruction scan computed (%.1fs)", time.time() - t0)
    else:
        log.info("instruction scan: reused cache")
    n_instr = 0
    with raw_cache.open("r", encoding="utf-8") as fr, instr_nd.open("w", encoding="utf-8") as fi:
        for line in fr:
            d = json.loads(line)
            if d["class"] not in already.get(d["recipe_id"], ()):
                n_instr += 1
                fi.write(json.dumps({"recipe_id": d["recipe_id"], "class": d["class"], "status": "maybe",
                                     "snippet": d["snippet"]}, ensure_ascii=False) + "\n")
    log.info("instruction scan: %d extra flags (%.1fs)", n_instr, time.time() - t0)

    for tbl in ("ingredients_parsed", "ingredient_flags", "instruction_flags", "recipe_flags", "ingredient_vocab",
                "build_meta"):
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    con.execute(f"""
        CREATE TABLE ingredients_parsed AS SELECT * FROM read_json('{parsed_nd.as_posix()}',
          format='newline_delimited', columns={{recipe_id:'VARCHAR', position:'INTEGER', item_idx:'INTEGER',
          status:'VARCHAR', name:'VARCHAR', base:'VARCHAR', form:'VARCHAR', parent:'VARCHAR',
          qty_min:'DOUBLE', qty_max:'DOUBLE', unit:'VARCHAR', unit_kind:'VARCHAR', qty_base_min:'DOUBLE',
          qty_base_max:'DOUBLE', optional:'BOOLEAN', alternatives:'VARCHAR[]', alt_source:'VARCHAR',
          prep:'VARCHAR', descriptors:'VARCHAR[]', group_label:'VARCHAR'}})""")
    con.execute(f"""
        CREATE TABLE ingredient_flags AS SELECT * FROM read_json('{flags_nd.as_posix()}',
          format='newline_delimited', columns={{recipe_id:'VARCHAR', position:'INTEGER', item_idx:'INTEGER',
          "class":'VARCHAR', status:'VARCHAR', via:'VARCHAR', name:'VARCHAR', optional:'BOOLEAN'}})""")
    con.execute(f"""
        CREATE TABLE instruction_flags AS SELECT * FROM read_json('{instr_nd.as_posix()}',
          format='newline_delimited', columns={{recipe_id:'VARCHAR', "class":'VARCHAR', status:'VARCHAR',
          snippet:'VARCHAR'}})""")
    con.execute("""
        CREATE TABLE recipe_flags AS
        SELECT recipe_id, "class",
               CASE WHEN bool_or(status = 'yes') THEN 'yes' ELSE 'maybe' END AS status,
               count(*) AS n_lines, list(DISTINCT src ORDER BY src) AS sources
        FROM (SELECT recipe_id, "class", status, 'ingredients' AS src FROM ingredient_flags
              UNION ALL SELECT recipe_id, "class", status, 'instructions' AS src FROM instruction_flags)
        GROUP BY recipe_id, "class" """)
    con.execute("CREATE TABLE build_meta(key VARCHAR, value VARCHAR)")
    con.execute("INSERT INTO build_meta VALUES ('lexicon_fingerprint', ?)", [lexicon_fingerprint()])
    con.execute("""
        CREATE TABLE ingredient_vocab AS
        SELECT name, any_value(base) AS base, any_value(form) AS form, any_value(parent) AS parent,
               count(*) AS n_lines, count(DISTINCT recipe_id) AS n_recipes
        FROM ingredients_parsed WHERE status = 'ok' AND name IS NOT NULL GROUP BY name""")
    con.execute("CREATE INDEX ix_parsed_recipe ON ingredients_parsed(recipe_id)")
    con.execute("CREATE INDEX ix_parsed_name ON ingredients_parsed(name)")
    con.execute("CREATE INDEX ix_flags_recipe ON ingredient_flags(recipe_id)")
    con.execute('CREATE INDEX ix_recipe_flags ON recipe_flags("class", status)')

    q = con.execute
    ok_lines = q("SELECT count(*) FROM ingredients_parsed WHERE status='ok'").fetchone()[0]
    vocab = q("SELECT count(*), sum(n_lines) FROM ingredient_vocab").fetchone()
    cover = {}
    for k in (100, 500, 1000, 5000):
        cover[f"top_{k}_names_cover_pct"] = round(q(
            f"SELECT sum(n_lines) FROM (SELECT n_lines FROM ingredient_vocab ORDER BY n_lines DESC LIMIT {k})"
        ).fetchone()[0] / vocab[1] * 100, 1)
    per_class = {
        c: {"recipes_yes": n_yes, "recipes_maybe": n_maybe}
        for c, n_yes, n_maybe in q("""
            SELECT "class", count(*) FILTER (WHERE status='yes'), count(*) FILTER (WHERE status='maybe')
            FROM recipe_flags GROUP BY 1 ORDER BY 1""").fetchall()
    }
    report = {
        "source_lines": len(rows),
        "parsed_items": n_items,
        "status_counts": dict(status_count),
        "ok_items": ok_lines,
        "distinct_names": vocab[0],
        "names_seen_once": q("SELECT count(*) FROM ingredient_vocab WHERE n_lines = 1").fetchone()[0],
        "names_with_5plus_words": q(
            "SELECT count(*) FROM ingredient_vocab WHERE len(string_split(name, ' ')) >= 5").fetchone()[0],
        "items_without_quantity_pct": round(q(
            "SELECT 100.0 * count(*) FILTER (WHERE qty_min IS NULL) / count(*) FROM ingredients_parsed "
            "WHERE status='ok'").fetchone()[0], 1),
        "items_with_base_quantity_pct": round(q(
            "SELECT 100.0 * count(*) FILTER (WHERE qty_base_min IS NOT NULL) / count(*) FROM ingredients_parsed "
            "WHERE status='ok'").fetchone()[0], 1),
        "items_with_alternatives": q(
            "SELECT count(*) FROM ingredients_parsed WHERE len(alternatives) > 0").fetchone()[0],
        "vocabulary_coverage": cover,
        "recipes_flagged_by_class": per_class,
        "recipes_total": q("SELECT count(*) FROM recipes").fetchone()[0],
        "instruction_only_flags": n_instr,
        "lexicon_fingerprint": lexicon_fingerprint(),
        "taxonomy_classes": tax.classes,
        "seconds": round(time.time() - t0, 1),
    }
    con.close()
    parsed_nd.unlink()
    flags_nd.unlink()
    instr_nd.unlink()
    out = cfg.report_path.with_name("ingredient_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("report written to %s", out)
    return report
