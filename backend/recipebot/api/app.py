"""HTTP API: POST /chat streams Server-Sent Events. Event types (a superset of the MindSearch protocol):

  active_filters  the filters now in force (shown as chips)
  token           the answer text (one event; the answer is checked before it is sent, so it is not streamed word by word)
  sources         [{n, recipe_id, title, source, evidence}]  the [n] citations used in the answer
  computation     {filters, funnel, n_analyzed, denominator, method, sql}  "how this was calculated"
  error           a plain-language problem
  done            {intent, status}
"""
from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from recipebot.orchestrator.bot import Chatbot, TurnResult, build_chatbot


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    vegetarian: bool | None = None       # the vegetarian-only toggle; None leaves the session setting as it was
    taste: dict[str, str] | None = None  # {"sweet": "low|medium|high", "salt": ..., "spice": ...}; invalid values are dropped


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def events(result: TurnResult, session_id: str):
    yield sse("session", {"session_id": session_id})
    yield sse("active_filters", result.active_filters)
    yield sse("sources", [{"n": s.n, "recipe_id": s.recipe_id, "title": s.title, "source": s.source,
                           "evidence": s.evidence, "kind": s.kind, "view": s.view} for s in result.sources])
    if result.computation:
        c = result.computation
        yield sse("computation", {k: c.get(k) for k in ("tool", "description", "population", "filters", "funnel",
                                                         "n_analyzed", "denominator", "method", "sql", "notes")})
    if result.web:
        yield sse("web", result.web)
    yield sse("token", {"text": result.answer})
    yield sse("done", {"intent": result.intent, "status": result.status, "notes": result.notes, "suggest": result.suggest})


def create_app(bot: Chatbot | None = None) -> FastAPI:
    holder: dict = {"bot": bot}
    app = FastAPI(title="Recipe Box chatbot", version="0.1", lifespan=lambda a: lifespan(a))
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    def get_bot() -> Chatbot:
        if holder["bot"] is None:
            holder["bot"] = build_chatbot()
        return holder["bot"]

    @asynccontextmanager
    async def lifespan(_app):
        if bot is None:       # a real server (tests pass their own bot): load the index in the background before the first question
            threading.Thread(target=lambda: get_bot().warm_up(), daemon=True).start()
        yield

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat")
    def chat(req: ChatRequest):
        sid = req.session_id or uuid.uuid4().hex
        try:
            result = get_bot().turn(sid, req.message, req.vegetarian, req.taste)
        except Exception as e:                                       # never leak a stack trace to the browser
            def err():
                yield sse("error", {"message": "Something went wrong on the server. Please try again."})
                yield sse("done", {"intent": "error", "status": "error"})
            return StreamingResponse(err(), media_type="text/event-stream")
        return StreamingResponse(events(result, sid), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/recipe")
    def get_recipe(id: str, scale: float = 1.0, sweet: str | None = None, salt: str | None = None,
                   spice: str | None = None):          # a query parameter, not a path part: real recipe IDs contain "/" and "."
        """Full stored recipe for the recipe window. `scale` multiplies the amounts; sweet/salt/spice (low|medium|high)
        add the taste box: how much to reduce or increase the sweet, salty and hot ingredients."""
        from recipebot.tools.amounts import scale_lines
        from recipebot.tools.recipe import get_recipe as fetch
        from recipebot.tools.taste import adjust_for_taste, box_to_dict
        rec = fetch(id)
        if rec is None:
            raise HTTPException(status_code=404, detail="No recipe with that ID.")
        if not 0.1 <= scale <= 20:
            raise HTTPException(status_code=422, detail="scale must be between 0.1 and 20.")
        lines = scale_lines([i.text for i in rec.ingredients], scale)
        taste = adjust_for_taste([s.text for s in lines], {"sweet": sweet, "salt": salt, "spice": spice})
        return {"id": rec.id, "title": rec.title, "source": rec.source, "scale": scale,
                "ingredients": [{"text": s.text, "original": i.original, "scaled": s.scaled}
                                for s, i in zip(lines, rec.ingredients)],
                "instructions": rec.instructions, "steps": rec.n_steps_sentences, "taste": box_to_dict(taste)}

    @app.post("/reset")
    def reset(session_id: str):
        get_bot().sessions.reset(session_id)
        return {"status": "reset"}

    return app


app = create_app()
