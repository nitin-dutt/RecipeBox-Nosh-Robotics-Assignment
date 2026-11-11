"""Command line for the ingredient labeling job (all steps are resumable and budgeted for a free Gemini tier).

  pilot         label the gold-set names (a few calls) and compare rules / LLM / merged against the gold set
  top           label the N most frequent unlabelled ingredient names, at most --max-calls API calls this run
  merge         compare rules vs LLM labels: write the raise-only overlay and the human review queue
  apply-review  record the decisions typed into the review CSV and regenerate the overrides file

  uv run python -m recipebot.labeling.run pilot --max-calls 6
  uv run python -m recipebot.labeling.run top --n 600 --max-calls 15
"""
from __future__ import annotations

import argparse
import json
import logging

from recipebot.domain.taxonomy import get_taxonomy
from recipebot.labeling import evaluate, llm_labeler, merge
from recipebot.labeling.suggest import suggest
from recipebot.labeling.schema import NameLabel
from recipebot.llm.gemini import GeminiClient
from recipebot.settings import ROOT, get_settings
from recipebot.store.db import Database

LABELS = ROOT / "data" / "labels" / "llm_labels.jsonl"
GOLD = ROOT / "eval" / "gold_ingredients.yaml"
OVERLAY = ROOT / "config" / "labels" / "ingredient_labels.json"
DECISIONS = ROOT / "config" / "labels" / "review_decisions.json"
OVERRIDES = ROOT / "config" / "ingredient_overrides.yaml"
REVIEW_CSV = ROOT / "data" / "review" / "ingredient_review.csv"


def client(max_calls: int, interval: float) -> GeminiClient:
    cfg = get_settings()
    return GeminiClient(cfg.gemini_api_key, cfg.gemini_model, cfg.llm_cache_dir, max_calls=max_calls, min_interval_s=interval)


def vocab(db: Database) -> tuple[dict[str, int], dict[str, str]]:
    rows = db.query("SELECT name, n_lines, parent FROM ingredient_vocab")
    return {n: c for n, c, _ in rows}, {n: p for n, _, p in rows}


def rule_preds(names: list[str], rule_parent: dict[str, str]) -> dict[str, evaluate.Pred]:
    tax = get_taxonomy()
    return {n: evaluate.Pred(True, rule_parent.get(n, n), tax.classify_name(n, rules_only=True)) for n in names}


def final_preds(names: list[str], rule_parent: dict[str, str]) -> dict[str, evaluate.Pred]:
    """What the running system actually does: rules + reviewed LLM overlay + human decisions."""
    tax = get_taxonomy()
    return {n: evaluate.Pred(True, rule_parent.get(n, n), tax.classify_name(n)) for n in names}


def llm_preds(labels: dict[str, NameLabel]) -> dict[str, evaluate.Pred]:
    return {n: evaluate.Pred(l.is_ingredient, l.parent, l.class_map() if l.is_ingredient else {}) for n, l in labels.items()}


def merged_preds(rules: dict[str, evaluate.Pred], llm: dict[str, evaluate.Pred],
                 n_lines: dict[str, int] | None = None) -> dict[str, evaluate.Pred]:
    """rules + LLM under the policy: raise to "yes" always; raise to "maybe" only for rare names (common ones go to
    human review and are NOT applied until accepted). The parent stays the rules' parent."""
    out = {}
    for n, r in rules.items():
        classes = dict(r.classes)
        if n in llm:
            for c, lvl in llm[n].classes.items():
                held_back = lvl == "maybe" and n_lines is not None and n_lines.get(n, 0) >= merge.COMMON_LINES
                if evaluate.RANK[lvl] > evaluate.RANK[classes.get(c)] and not held_back:
                    classes[c] = lvl
        out[n] = evaluate.Pred(r.is_ingredient, r.parent, classes)
    return out


def print_table(results: dict[str, dict]) -> None:
    keys = ["recall_any_flag", "recall_definite_flagged", "recall_definite_exact", "precision_of_flags", "false_alarm_pairs",
            "parent_accuracy", "is_ingredient_accuracy"]
    print(f"{'':28}" + "".join(f"{m:>12}" for m in results))
    for k in keys:
        print(f"{k:28}" + "".join(f"{str(r[k]):>12}" for r in results.values()))


def cmd_pilot(args) -> None:
    db = Database(get_settings().db_path)
    gold = evaluate.load_gold(GOLD)
    names = list(gold)
    c = client(args.max_calls, args.interval)
    got = llm_labeler.label_names(c, names, batch_size=args.batch)
    llm_labeler.save_labels(LABELS, got, c.model)
    print("API:", c.stats(), "| labelled", len(got), "of", len(names))
    n_lines, rule_parent = vocab(db)
    rules, llm = rule_preds(names, rule_parent), llm_preds(got)
    results = {"rules": evaluate.score(rules, gold), "llm": evaluate.score(llm, gold),
               "merged_all": evaluate.score(merged_preds(rules, llm), gold),
               "merged_policy": evaluate.score(merged_preds(rules, llm, n_lines), gold),
               "final_system": evaluate.score(final_preds(names, rule_parent), gold)}
    print_table(results)
    for m in ("rules", "llm", "merged_policy", "final_system"):
        print(f"\n[{m}] definite allergens NOT flagged:", results[m]["missed_definite"][:12])
        print(f"[{m}] any gold flag missed:", results[m]["missed_any"][:8])
    (ROOT / "eval" / "ingredient_labeling_pilot.json").write_text(
        json.dumps({k: {x: y for x, y in v.items()} for k, v in results.items()}, indent=1), encoding="utf-8")


def cmd_top(args) -> None:
    db = Database(get_settings().db_path)
    n_lines, _ = vocab(db)
    have = llm_labeler.load_labels(LABELS)
    todo = [n for n, _ in sorted(n_lines.items(), key=lambda kv: -kv[1]) if n not in have][: args.n]
    print(f"{len(have)} names already labelled; labelling the next {len(todo)} (most frequent first)")
    c = client(args.max_calls, args.interval)
    got = llm_labeler.label_names(c, todo, batch_size=args.batch)
    llm_labeler.save_labels(LABELS, got, c.model)
    covered = sum(n_lines[n] for n in {**have, **got})
    print("API:", c.stats(), "| newly labelled", len(got), "| total labelled", len(have) + len(got),
          f"| lines covered {covered / sum(n_lines.values()) * 100:.1f}%")


def cmd_merge(args) -> None:
    db = Database(get_settings().db_path)
    n_lines, rule_parent = vocab(db)
    labels = llm_labeler.load_labels(LABELS)
    cmp = merge.compare(labels, get_taxonomy(), rule_parent, n_lines)
    total = merge.write_overlay(OVERLAY, cmp, get_settings().gemini_model)
    rows = merge.write_review_queue(REVIEW_CSV, cmp, DECISIONS, suggest=suggest)
    print(f"{len(labels)} labelled names | agree with rules: {cmp.agree} | LLM RAISES applied: {len(cmp.raises)} names "
          f"(overlay now {total}) | review queue: {rows} rows -> {REVIEW_CSV}")
    print(f"  review queue: common-name 'maybe' raises {len(cmp.raises_review)} | lowers (rules keep their flag) "
          f"{len(cmp.lowers)} | not-ingredient {len(cmp.not_ingredient)} | LLM parents are not used")


def cmd_apply(args) -> None:
    print(merge.apply_review(REVIEW_CSV, DECISIONS, OVERRIDES, args.reviewer))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("pilot", cmd_pilot), ("top", cmd_top)):
        p = sub.add_parser(name)
        p.add_argument("--max-calls", type=int, default=6)
        p.add_argument("--batch", type=int, default=50)
        p.add_argument("--interval", type=float, default=4.0, help="minimum seconds between API calls")
        if name == "top":
            p.add_argument("--n", type=int, default=600)
        p.set_defaults(fn=fn)
    sub.add_parser("merge").set_defaults(fn=cmd_merge)
    p = sub.add_parser("apply-review")
    p.add_argument("--reviewer", default="reviewer")
    p.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
