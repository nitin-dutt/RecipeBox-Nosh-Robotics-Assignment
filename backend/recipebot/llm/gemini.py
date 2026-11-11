"""Gemini client built for a FREE TIER: cached, paced, capped, and it stops cleanly when the daily quota is gone.

* Every request is cached on disk by a hash of (model, system prompt, user prompt, schema). A repeated request never
  costs a call, and an interrupted job resumes exactly where it stopped.
* Calls are paced (`min_interval_s`) to stay under the per-minute limit; on a per-minute 429 the pace is slowed down.
* A hard cap per run (`max_calls`) so a bug or a typo cannot burn the daily budget.
* A DAILY quota error raises `DailyQuotaExhausted` instead of retrying: the caller saves progress and stops.
* The API key is a pydantic SecretStr and is never logged or put in an exception message.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import SecretStr

log = logging.getLogger("recipebot.llm")


class DailyQuotaExhausted(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class LLMUnavailable(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: SecretStr | None, model: str, cache_dir: Path, max_calls: int = 50,
                 min_interval_s: float = 4.0, max_retries: int = 4, request_timeout_s: float | None = None,
                 max_wait_s: float | None = None, thinking_level: str | None = None) -> None:
        """`request_timeout_s` and `max_wait_s` are for INTERACTIVE use (the chat): a slow or rate-limited call gives up
        quickly (LLMUnavailable -> the rule-based planner answers) instead of making the user wait 30 s. Batch jobs
        (labelling) leave both None and keep waiting.

        `thinking_level="minimal"` turns off most of the model's hidden reasoning: the chat planner only fills a small form,
        and this cut its call from about 6 s to about 1.5 s in our test. Labelling keeps the default (it needs the reasoning)."""
        if api_key is None or not api_key.get_secret_value():
            raise LLMUnavailable("GEMINI_API_KEY is not set (put it in .env).")
        self._key = api_key
        self.model, self.cache_dir = model, cache_dir
        self.max_calls, self.min_interval_s, self.max_retries = max_calls, min_interval_s, max_retries
        # the API rejects a deadline under 10 s with a 400, which would silently push every turn onto the rule-based fallback
        self.request_timeout_s = max(10.0, request_timeout_s) if request_timeout_s else None
        self.max_wait_s, self.thinking_level = max_wait_s, thinking_level
        self.calls = self.cache_hits = 0
        self.tokens_in = self.tokens_out = 0
        self._last_call = 0.0
        self._client = None
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    def _cache_path(self, system: str, user: str, schema_name: str) -> Path:
        h = hashlib.sha256("\x1f".join([self.model, system, user, schema_name, self.thinking_level or ""]).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h[:32]}.json"

    def _sdk(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._key.get_secret_value())
        return self._client

    def _scrub(self, text: str) -> str:
        return text.replace(self._key.get_secret_value(), "***")

    # ------------------------------------------------------------------ call
    def generate_json(self, system: str, user: str, schema, max_output_tokens: int = 8192,
                      accept: Callable[[object], bool] | None = None) -> tuple[object, bool]:
        """-> (parsed JSON, was_cached). `schema` is a pydantic model or list[model] used for structured output.

        `accept` validates a fresh response; a response it rejects is returned but NOT cached, so a bad answer is
        never served from the cache later."""
        schema_name = getattr(schema, "__name__", None) or str(schema)
        path = self._cache_path(system, user, schema_name)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"], True
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call cap of {self.max_calls} reached for this run")
        from google.genai import errors, types
        extra = {}
        if self.request_timeout_s:
            extra["http_options"] = types.HttpOptions(timeout=int(self.request_timeout_s * 1000))
        if self.thinking_level:
            extra["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0, max_output_tokens=max_output_tokens,
            response_mime_type="application/json", response_schema=schema, **extra)
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            self.calls += 1
            try:
                resp = self._sdk().models.generate_content(model=self.model, contents=user, config=config)
            except errors.APIError as e:
                msg = self._scrub(str(e))
                code = getattr(e, "code", None)
                if code == 429:
                    if re.search(r"PerDay|per day|daily", msg, re.IGNORECASE):
                        raise DailyQuotaExhausted("Gemini daily quota reached; progress is cached, resume tomorrow.") from None
                    delay = float((re.search(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"]([\d.]+)s", msg) or [None, 30, 30])[1]
                                  or 30) + 2
                    if self.max_wait_s is not None and delay > self.max_wait_s:
                        raise LLMUnavailable(f"Gemini is rate-limited (would wait {delay:.0f}s); using the fallback") from None
                    self.min_interval_s = min(self.min_interval_s * 1.5, 30.0)
                    log.warning("per-minute limit hit; waiting %.0fs and slowing to one call per %.1fs", delay, self.min_interval_s)
                    time.sleep(delay)
                    continue
                if code in (500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(1 if self.max_wait_s is not None else 5 * (attempt + 1))
                    continue
                raise LLMUnavailable(f"Gemini error {code}: {msg[:300]}") from None
            except (LLMUnavailable, DailyQuotaExhausted):
                raise
            except Exception as e:                      # a timeout or a dropped connection: never hang the chat on it
                raise LLMUnavailable(f"Gemini request failed ({type(e).__name__})") from None
            if resp.usage_metadata:
                self.tokens_in += resp.usage_metadata.prompt_token_count or 0
                self.tokens_out += resp.usage_metadata.candidates_token_count or 0
            text = (resp.text or "").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                if attempt < self.max_retries:
                    continue
                raise LLMUnavailable("Gemini returned text that is not valid JSON") from None
            if accept is None or accept(parsed):
                path.write_text(json.dumps({"model": self.model, "response": parsed}, ensure_ascii=False), encoding="utf-8")
            return parsed, False
        raise LLMUnavailable("Gemini did not answer after retries")

    def stats(self) -> dict:
        return {"model": self.model, "api_calls": self.calls, "cache_hits": self.cache_hits,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}
