"""Central configuration. Values come from environment variables (prefix RECIPEBOT_) or .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECIPEBOT_", env_file=ROOT / ".env", extra="ignore")

    dataset_dir: Path = Path("Dataset")
    config_dir: Path = Path("config")
    clean_dir: Path = Path("data/clean")
    tmp_dir: Path = Path("data/tmp")
    index_dir: Path = Path("data/index")
    llm_cache_dir: Path = Path("data/llm_cache")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-3.5-flash-lite"
    narrate_with_llm: bool = False        # the optional LLM lead-in sentence: one extra model call per turn, so off by default
    web_search: bool = True               # web results only when the dataset's evidence for a substitution is weak
    models_dir: Path = Path("data/models")
    db_path: Path = Path("data/recipes.duckdb")
    report_path: Path = Path("data/data_quality_report.json")

    @field_validator("dataset_dir", "config_dir", "clean_dir", "tmp_dir", "index_dir", "llm_cache_dir", "models_dir", "db_path", "report_path")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else ROOT / v


@lru_cache
def get_settings() -> Settings:
    return Settings()
