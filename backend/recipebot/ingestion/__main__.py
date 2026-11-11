"""uv run python -m recipebot.ingestion          -> stage 1 (clean + load) then stage 2 (parse ingredients)
uv run python -m recipebot.ingestion --stage2   -> only re-run ingredient parsing/classification (fast)"""
import json
import logging
import sys

from recipebot.ingestion import ingredient_stage, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
if "--stage2" not in sys.argv:
    print(json.dumps(pipeline.run()["reconciliation"], indent=2))
report = ingredient_stage.run()
print(json.dumps({k: report[k] for k in ("source_lines", "parsed_items", "status_counts", "distinct_names",
                                         "vocabulary_coverage", "recipes_flagged_by_class")}, indent=2))
