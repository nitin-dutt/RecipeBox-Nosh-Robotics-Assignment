"""Offline ingestion: raw JSON (x3) -> cleaned JSON (x3) -> DuckDB + data-quality report.

Run:  uv run python -m recipebot.ingestion
Deterministic: same raw files -> byte-identical cleaned files (hashes are in the report).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

from recipebot.ingestion.cleaning import CleanRecord, clean_record
from recipebot.settings import Settings, get_settings

log = logging.getLogger("recipebot.ingestion")

SOURCES = {  # source code -> (raw file name, human name)
    "ar": ("recipes_raw_nosource_ar.json", "Allrecipes"),
    "epi": ("recipes_raw_nosource_epi.json", "Epicurious"),
    "fn": ("recipes_raw_nosource_fn.json", "Food Network"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dup_key(rec: CleanRecord) -> tuple | None:
    """Exact duplicate = same cleaned title and same ordered cleaned ingredient lines."""
    if not (rec.has_title and rec.has_ingredients):
        return None
    return (rec.title.lower(), tuple(i.lower() for i in rec.ingredients))  # type: ignore[union-attr]


def clean_all(cfg: Settings) -> tuple[dict[str, list[CleanRecord]], dict]:
    """Phase 1: load + clean every source. Returns records per source and per-source stats."""
    records: dict[str, list[CleanRecord]] = {}
    stats: dict[str, dict] = {}
    for src, (fname, human) in SOURCES.items():
        path = cfg.dataset_dir / fname
        t0 = time.time()
        raw = json.loads(path.read_text(encoding="utf-8"))
        rules: Counter = Counter()
        out: list[CleanRecord] = []
        empty = 0
        for rid, rec in raw.items():
            if not rec:  # {} placeholders
                empty += 1
                continue
            out.append(clean_record(rid, src, rec, lambda r: rules.update([r])))
        records[src] = out
        stats[src] = {
            "name": human,
            "file": fname,
            "sha256": sha256(path),
            "raw_keys": len(raw),
            "empty_records_dropped": empty,
            "records_kept": len(out),
            "rule_hits": dict(sorted(rules.items())),
            "missing": {
                "no_title": sum(not r.has_title for r in out),
                "no_ingredients": sum(not r.has_ingredients for r in out),
                "no_instructions": sum(not r.has_instructions for r in out),
                "incomplete": sum(not r.is_complete for r in out),
            },
        }
        log.info("%s: %d raw -> %d kept (%.1fs)", src, len(raw), len(out), time.time() - t0)
        del raw
    return records, stats


def assign_duplicates(records: dict[str, list[CleanRecord]]) -> dict[str, tuple[str | None, int]]:
    groups: dict[tuple, list[str]] = defaultdict(list)
    for recs in records.values():
        for r in recs:
            k = dup_key(r)
            if k is not None:
                groups[k].append(r.id)
    info: dict[str, tuple[str | None, int]] = {}
    for key, ids in groups.items():
        if len(ids) > 1:
            gid = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:12]
            for rid in ids:
                info[rid] = (gid, len(ids))
    return info


def write_clean_json(cfg: Settings, records, dups) -> dict[str, str]:
    cfg.clean_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for src, recs in records.items():
        path = cfg.clean_dir / f"recipes_clean_{src}.json"
        body = {r.id: r.to_json(*dups.get(r.id, (None, 1))) for r in recs}
        path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        hashes[src] = sha256(path)
        log.info("wrote %s", path)
    return hashes


def build_duckdb(cfg: Settings, records, dups) -> dict:
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    rec_nd, ing_nd = cfg.tmp_dir / "recipes.ndjson", cfg.tmp_dir / "ingredients.ndjson"
    with rec_nd.open("w", encoding="utf-8") as fr, ing_nd.open("w", encoding="utf-8") as fi:
        for recs in records.values():
            for r in recs:
                gid, gsize = dups.get(r.id, (None, 1))
                fr.write(json.dumps({
                    "id": r.id, "source": r.source, "title": r.title, "title_raw": r.title_raw,
                    "instructions": r.instructions, "instructions_raw": r.instructions_raw,
                    "n_ingredients": len(r.ingredients),
                    "n_steps_paragraphs": r.n_steps_paragraphs,
                    "n_steps_sentences": r.n_steps_sentences,
                    "picture_link": r.picture_link,
                    "has_title": r.has_title, "has_ingredients": r.has_ingredients,
                    "has_instructions": r.has_instructions, "is_complete": r.is_complete,
                    "dup_group_id": gid, "dup_group_size": gsize, "flags": r.flags,
                }, ensure_ascii=False) + "\n")
                for pos, (text, raw, group) in enumerate(
                    zip(r.ingredients, r.ingredient_raws, r.ingredient_groups), start=1
                ):
                    fi.write(json.dumps({
                        "recipe_id": r.id, "position": pos, "text": text,
                        "text_raw": raw, "group_label": group,
                    }, ensure_ascii=False) + "\n")

    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.db_path.exists():
        cfg.db_path.unlink()
    con = duckdb.connect(str(cfg.db_path))
    con.execute(f"""
        CREATE TABLE recipes AS SELECT * FROM read_json('{rec_nd.as_posix()}', format='newline_delimited',
          columns={{id:'VARCHAR', source:'VARCHAR', title:'VARCHAR', title_raw:'VARCHAR',
                    instructions:'VARCHAR', instructions_raw:'VARCHAR', n_ingredients:'INTEGER',
                    n_steps_paragraphs:'INTEGER', n_steps_sentences:'INTEGER', picture_link:'VARCHAR',
                    has_title:'BOOLEAN', has_ingredients:'BOOLEAN', has_instructions:'BOOLEAN',
                    is_complete:'BOOLEAN', dup_group_id:'VARCHAR', dup_group_size:'INTEGER',
                    flags:'VARCHAR[]'}})
    """)
    con.execute(f"""
        CREATE TABLE recipe_ingredients AS SELECT * FROM read_json('{ing_nd.as_posix()}',
          format='newline_delimited',
          columns={{recipe_id:'VARCHAR', position:'INTEGER', text:'VARCHAR', text_raw:'VARCHAR',
                    group_label:'VARCHAR'}})
    """)
    con.execute("CREATE UNIQUE INDEX ux_recipes_id ON recipes(id)")
    con.execute("CREATE INDEX ix_ing_recipe ON recipe_ingredients(recipe_id)")

    q = con.execute
    checks = {
        "recipes_rows": q("SELECT count(*) FROM recipes").fetchone()[0],
        "ingredient_rows": q("SELECT count(*) FROM recipe_ingredients").fetchone()[0],
        "distinct_recipe_ids": q("SELECT count(DISTINCT id) FROM recipes").fetchone()[0],
        "orphan_ingredient_rows": q(
            "SELECT count(*) FROM recipe_ingredients i LEFT JOIN recipes r ON r.id=i.recipe_id "
            "WHERE r.id IS NULL").fetchone()[0],
        "complete_recipes": q("SELECT count(*) FROM recipes WHERE is_complete").fetchone()[0],
        "recipes_in_dup_groups": q("SELECT count(*) FROM recipes WHERE dup_group_id IS NOT NULL").fetchone()[0],
        "dup_groups": q("SELECT count(DISTINCT dup_group_id) FROM recipes").fetchone()[0],
        "median_ingredients_complete": q(
            "SELECT median(n_ingredients) FROM recipes WHERE is_complete").fetchone()[0],
        "median_steps_sentences_complete": q(
            "SELECT median(n_steps_sentences) FROM recipes WHERE is_complete").fetchone()[0],
        "by_source": {s: n for s, n in q("SELECT source, count(*) FROM recipes GROUP BY 1 ORDER BY 1").fetchall()},
    }
    con.close()
    rec_nd.unlink()
    ing_nd.unlink()
    return checks


def residual_scan(records) -> dict:
    """Things we deliberately did not guess at, so the report shows what is left."""
    c: Counter = Counter()
    for recs in records.values():
        for r in recs:
            texts = [r.title or "", r.instructions, *r.ingredients]
            blob = "\n".join(texts)
            if "�" in blob:
                c["replacement_char_U+FFFD"] += 1
            if "¿" in blob:
                c["unresolved_inverted_question_mark"] += 1
            if "ADVERTISEMENT" in blob:
                c["advertisement_still_present"] += 1
            if "&" in blob and any(
                m for m in __import__("re").findall(r"&[#\w]+;", blob)
            ):
                c["entity_like_token_left"] += 1
    return dict(c)


def run(cfg: Settings | None = None) -> dict:
    cfg = cfg or get_settings()
    t0 = time.time()
    records, per_source = clean_all(cfg)
    dups = assign_duplicates(records)
    clean_hashes = write_clean_json(cfg, records, dups)
    db = build_duckdb(cfg, records, dups)
    for src in per_source:
        per_source[src]["clean_file_sha256"] = clean_hashes[src]

    raw_total = sum(s["raw_keys"] for s in per_source.values())
    empty_total = sum(s["empty_records_dropped"] for s in per_source.values())
    report = {
        "reconciliation": {
            "raw_keys_total": raw_total,
            "empty_records_dropped": empty_total,
            "records_kept": raw_total - empty_total,
            "records_in_database": db["recipes_rows"],
            "balanced": raw_total - empty_total == db["recipes_rows"] == db["distinct_recipe_ids"],
        },
        "sources": per_source,
        "database": db,
        "residual_after_cleaning": residual_scan(records),
        "definitions": {
            "is_complete": "has a title, at least one ingredient line and non-empty instructions",
            "n_steps_paragraphs": "non-empty instruction lines after cleaning",
            "n_steps_sentences": "sentences after cleaning (split on . ! ? followed by a capital/digit)",
            "dup_group": "same cleaned title AND same ordered cleaned ingredient lines; never deleted",
        },
        "seconds": round(time.time() - t0, 1),
    }
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("report written to %s", cfg.report_path)
    return report
