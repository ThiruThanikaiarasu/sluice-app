"""SQLite connection and schema lifecycle."""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sluice.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Drop and rebuild the database from schema.sql."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = connect(path)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    return conn


def scenario_of(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'scenario'").fetchone()
    return row["value"] if row else None
