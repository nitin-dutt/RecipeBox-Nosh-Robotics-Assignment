"""Ask Gemini to label ingredient names, in validated, cached batches.

The model is an offline labeler: it runs once over the vocabulary, its answers are stored, and the chatbot never calls
it at question time. Anything it returns is validated; names it skips or garbles are reported, never guessed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from recipebot.labeling.schema import CLASSES, NameLabel
from recipebot.llm.gemini import BudgetExceeded, DailyQuotaExhausted, GeminiClient
from recipebot.settings import ROOT

log = logging.getLogger("recipebot.labeling")
PROMPT_PATH = ROOT / "prompts" / "ingredient_labeler.md"


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _valid_batch(names: list[str]):
    expected = set(names)

    def check(parsed: object) -> bool:
        if not isinstance(parsed, list):
            return False
        try:
            labels = [NameLabel.model_validate(x) for x in parsed]
        except Exception:
            return False
        return {x.name for x in labels} == expected and len(labels) == len(names)
    return check


def label_batch(client: GeminiClient, names: list[str]) -> tuple[dict[str, NameLabel], list[str]]:
    """-> (labels by name, names that could not be labelled in this call)."""
    parsed, _cached = client.generate_json(system_prompt(), json.dumps(names, ensure_ascii=False), list[NameLabel],
                                           accept=_valid_batch(names))
    got: dict[str, NameLabel] = {}
    if isinstance(parsed, list):
        for x in parsed:
            try:
                lab = NameLabel.model_validate(x)
            except Exception:
                continue
            if lab.name in names and all(f.cls in CLASSES for f in lab.classes):
                got[lab.name] = lab
    return got, [n for n in names if n not in got]


def label_names(client: GeminiClient, names: list[str], batch_size: int = 50) -> dict[str, NameLabel]:
    """Label as many names as the budget allows. Stops cleanly on daily quota or the per-run call cap."""
    out: dict[str, NameLabel] = {}
    queue = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    while queue:
        batch = queue.pop(0)
        try:
            got, missing = label_batch(client, batch)
        except (DailyQuotaExhausted, BudgetExceeded) as e:
            log.warning("stopping: %s", e)
            break
        out.update(got)
        if missing and len(batch) > 5:           # an incomplete answer: retry the missing names in a smaller batch
            half = max(5, len(missing) // 2)
            queue = [missing[i:i + half] for i in range(0, len(missing), half)] + queue
        elif missing:
            log.warning("could not label: %s", missing)
        log.info("labelled %d names so far (%d calls, %d cached)", len(out), client.calls, client.cache_hits)
    return out


# ------------------------------------------------------------------------------------------ storage
def load_labels(path: Path) -> dict[str, NameLabel]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["label"]["name"]] = NameLabel.model_validate(d["label"])
    return out


def save_labels(path: Path, labels: dict[str, NameLabel], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_labels(path)
    existing.update(labels)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as f:
        for name in sorted(existing):
            f.write(json.dumps({"model": model, "saved": stamp, "label": existing[name].model_dump()},
                               ensure_ascii=False) + "\n")
