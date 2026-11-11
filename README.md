# Recipe Box chatbot

A conversational assistant over the **Recipe Box** dataset (about 125,000 recipes). It searches recipes, answers factual and
statistical questions, matches recipes to what is in your fridge, and adapts a recipe to a restriction using evidence from similar
recipes. Goal: **answers you can check**. Every recipe, ID, count and citation comes from a database query, never from the LLM,
which only turns a message into a small validated plan. What the data cannot answer (calories, time, servings, ratings) is refused.

## Run it (one command)

I automated the whole setup so you do not have to: no manual environment, dataset download, database build, index build, `.env`
editing or starting two servers. You need only **Python 3.11+**, **Node.js 18+**, internet access and about 3 GB of disk.

```bash
python setup_and_run.py        # Windows: py setup_and_run.py
```

It creates the virtualenv and installs `requirements.txt`, downloads the dataset (53 MB), builds the DuckDB database and the search
index, installs the web packages, creates `.env` from `.env.example` and asks for a Gemini API key (hidden input; Enter skips it),
then starts the API (`:8000`) and the UI and opens <http://127.0.0.1:5173>. Ctrl+C stops both.

- **First run takes about 15 minutes** (up to ~25 on a slow CPU), mostly the one-time search-index build (~8 min, embeds all
  125k recipes locally). You are not left guessing: it prints live progress, for example `chunk 12: 8192 docs, 190 docs/s, ~1.5 min left`.
  The build is resumable, and later runs skip every finished step and start in seconds.
- **API key:** a free key from <https://aistudio.google.com/apikey> is enough. Without one the bot still runs on a rule-based
  planner, but follow-ups and loose phrasing work much better with the model. Building the data needs no key.
- Options: `--no-start` (set up only), `--skip-key`, `--reset-key`.

<details><summary>Manual steps (what the script does)</summary>

```bash
python -m venv .venv && . .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install --no-deps -e .
# put the three recipes_raw_nosource_*.json files from https://eightportions.com/recipes_raw.zip in Dataset/
python -m recipebot.ingestion                         # clean, load DuckDB, parse ingredients, flag restrictions
python -m recipebot.retrieval.build                   # hybrid search index
cp .env.example .env                                  # optionally set GEMINI_API_KEY
python -m uvicorn recipebot.api.app:app --port 8000
npm --prefix web install && npm --prefix web run dev  # http://127.0.0.1:5173
```
</details>

## What to try

These are the conversations I ran against the real model (all in `eval/conversations.yaml`), so they are known to work.

| Task | Say this (in order) |
|---|---|
| 1. Search, follow-up, statistics | "Show me penne pasta recipes" → "I can't have dairy" → "Open the second one" · "What percentage of recipes use both garlic and onion?" · "Which 5 recipes with tomato have the fewest steps?" |
| 2. Ingredient matching | "I have eggs, onions, tomatoes and rice. What can I cook? No peanuts, at most 6 steps." → "Exclude eggs too" |
| 3. Substitution | "Show me banana bread recipes" → "Make the first one egg-free" |
| Allergy follow-up | "Show me fruit salad recipes" → "I'm allergic to grapes" → "What about ones with strawberries?" |
| Scaling | "Show me chili recipes" → "Open the second one" → "Double it" |
| Vegetarian switch | turn **Vegetarian only** on, then "Show me biryani recipes" → "Give me chicken biryani recipe" |
| Refusal | after banana bread: "How many calories does it have?" (refused, with what it can answer instead) |

Every answer has an **Evidence** dropdown: recipes used, filters in force, how many recipes were analysed, the method and the SQL.

## How it works

```
message ─► PLAN (LLM or rules) ─► GUARDS (code) ─► TOOLS (SQL / index) ─► grounded answer
```

- **Structured first.** Counts, filters and exclusions are SQL over a parsed ingredient table, so "no dairy" is a hard filter.
  Retrieval (BM25 + a small embedding model, fused with RRF, exact FAISS `IndexFlatIP`) only *ranks* recipes that already passed it.
- **RAG vs GraphRAG:** vector search cannot count or enforce negation, and the brief needs aggregates over the full filtered set.
  The recipe-to-ingredient table already is a graph, so SQL joins answer graph-style queries deterministically.
- **The LLM never writes facts.** It returns a `TurnPlan`, applied as a diff on session state (no history re-prompting). Code
  guards re-check stated allergies, step limits, scale factors and the vegetarian switch.
- **Conservative restrictions.** Each ingredient line is *yes* or *maybe* per class (dairy, egg, peanut, gluten...); strict mode
  excludes both. Results never claim a recipe is safe: only ingredient text is checked.
- **Substitution is evidence, not a guess.** Among the most similar recipes, those without the ingredient show what they use
  instead. Verdicts: *supported*, *weak*, *conflicting*, *none*; weak evidence is shown as leads and nothing is applied. What an
  ingredient does in a dish is curated knowledge and labelled a **prediction**. Web results appear only as labelled links and
  never influence the choice.

## Data

124,647 non-empty recipes (122,935 complete) and about 1.27M ingredient lines. Cleaning removes the "ADVERTISEMENT" artefact and
repeated instruction text and keeps the original next to the cleaned text; each line is parsed into name, quantity, unit, form and
alternatives. **Images are not included:** the files replace every `picture_link` with a hashed token.

## Tests and evaluation

```bash
pip install pytest && pytest        # 568 backend tests (some use the built database)
npm --prefix web test               # 28 UI tests
python eval/run_eval.py det         # deterministic eval, no model calls
```

| Measure | Result |
|---|---|
| Deterministic cases (scripted plans, independent checks) | **114 / 114**, 305 checks, 48 multi-turn |
| Real-model conversations (Gemini planner, real tools) | **41 / 41** turns, 14 conversations |
| Restriction-leak audit (raw-text regexes, not our parser) | 1 of 6,750 results, a regex false alarm |
| Live latency per turn | median 1.8 s, p90 2.7 s |
| One-shot LLM, no tools (contrast) | 0 of 10 quoted recipe IDs exist; it did not say the data has no nutrition |

The checks are written independently of the system, and some deliberately script a wrong plan to prove the guards repair it. I wrote
both the system and the cases, and the real-model set is small and uses one model, so a pass is not proof of no bugs.
Details and per-case results: `eval/report.md`.

## Layout

```
backend/recipebot/  ingestion/ domain/ tools/ retrieval/ substitution/ orchestrator/ api/ llm/
web/                React + TypeScript + Vite UI        config/   taxonomy, groups, roles, ingredient labels
prompts/            planner and labeller prompts        eval/     cases, conversations, independent checks, report
tests/              backend tests                       setup_and_run.py, requirements.txt
```

Env vars (`.env`): `GEMINI_API_KEY`, `RECIPEBOT_GEMINI_MODEL` (default `gemini-3.5-flash-lite`), `RECIPEBOT_WEB_SEARCH` (`true`),
`RECIPEBOT_NARRATE_WITH_LLM` (`false`), `RECIPEBOT_DATASET_DIR`, `RECIPEBOT_DB_PATH`. The key is never logged; model calls are
cached, paced, capped, and time out after 10 s into the rule-based planner.

## Known limitations

- Substitution evidence is often weak (a swap seen in 1 of 7 similar recipes is a lead, not proof); rewritten instructions are word
  replacement that a person must read before cooking.
- No servings in the data, so scaling is by factor ("double it"); times and pan sizes inside instructions are not scaled.
- Restriction checks read ingredient text only; they are not an allergy guarantee.
- The ingredient-label gold set was written by me and used to tune the rules (in-sample). Role knowledge covers ~70 ingredients.
- Mostly US recipe sites. Not built: cross-encoder reranker, CLI chat, accounts, persistent sessions (they live in server memory).

## AI tool disclosure

- **Claude Code** was used throughout as a coding assistant: data profiling, cleaning pipeline, parser, tools, orchestrator, API, UI,
  tests, evaluation harness, and drafting this README.
- **Architecture:** the assistant proposed options and trade-offs (structured-first retrieval with DuckDB as source of truth, hybrid
  ranking, the model only producing a validated plan). I reviewed each major decision, questioned it, chose between alternatives and
  approved it. Data findings were checked by running code against the dataset; bugs found along the way were
  fixed with regression tests.
- **Gemini (`gemini-3.5-flash-lite`, free tier):** *offline, once* to label ~3,200 ingredient names (validated, raise-only merge,
  human-reviewed); *at run time* one call per turn to produce the plan. Never a source of facts, numbers, IDs or citations.
- **Web search (`ddgs`)** only when substitution evidence is weak; shown as labelled links, never sent to the model.
- The React/FastAPI scaffolding and the safe text renderer were adapted from my earlier project, MindSearch; its secrets and
  personal files were not copied.
