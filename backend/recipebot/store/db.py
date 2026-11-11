"""Read-only DuckDB access. One connection per database path; each call uses its own cursor (thread-safe)."""
from __future__ import annotations

import threading
from pathlib import Path

import duckdb

from recipebot.settings import get_settings


class Database:
    def __init__(self, path: Path | str | None = None, *, connection: duckdb.DuckDBPyConnection | None = None):
        self.path = Path(path) if path else get_settings().db_path
        self._lock = threading.Lock()
        self._con = connection or duckdb.connect(str(self.path), read_only=True)

    def query(self, sql: str, params: list | tuple = ()) -> list[tuple]:
        with self._lock:
            cur = self._con.cursor()
            try:
                return cur.execute(sql, list(params)).fetchall()
            finally:
                cur.close()

    def scalar(self, sql: str, params: list | tuple = ()):
        rows = self.query(sql, params)
        return rows[0][0] if rows else None

    def close(self) -> None:
        self._con.close()


_default: Database | None = None


class StaleDataError(RuntimeError):
    pass


def verify_lexicon(db: Database) -> None:
    """Refuse to answer from data that was built with a different lexicon/taxonomy than the one on disk."""
    from recipebot.domain.versioning import lexicon_fingerprint
    try:
        stored = db.scalar("SELECT value FROM build_meta WHERE key = 'lexicon_fingerprint'")
    except Exception:          # no build_meta table (hand-built test databases)
        return
    if stored is not None and stored != lexicon_fingerprint():
        raise StaleDataError(
            "config/ (ingredients, taxonomy, groups or reviewed labels) changed after the database was built. "
            "Rebuild: uv run python -m recipebot.ingestion --stage2   (then rebuild the retrieval index)")


def get_db() -> Database:
    global _default
    if _default is None:
        _default = Database()
        verify_lexicon(_default)
    return _default


def set_db(db: Database | None) -> None:
    """Test hook: swap the process-wide database."""
    global _default
    _default = db
