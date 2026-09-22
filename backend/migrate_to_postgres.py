"""
One-time migration: copy rows from the legacy predictions.db (SQLite) into Postgres.

Usage:
    python backend/migrate_to_postgres.py [path/to/predictions.db]

Requires the same DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD (or DATABASE_URL)
env vars as the app itself - see .env.example.
"""
from __future__ import annotations
import sys
import sqlite3
from pathlib import Path

from sqlalchemy import text

try:
    from backend.db import engine
    from backend.main import ensure_db
except ImportError:
    from db import engine
    from main import ensure_db

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent / "predictions.db"

INSERT_SQL = text("""
    INSERT INTO predictions (
        season, week, game_id,
        home_team, away_team,
        predicted_winner,
        home_win_prob, away_win_prob, confidence,
        model_name, created_at, actual_winner
    )
    VALUES (
        :season, :week, :game_id,
        :home_team, :away_team,
        :predicted_winner,
        :home_win_prob, :away_win_prob, :confidence,
        :model_name, :created_at, :actual_winner
    )
    ON CONFLICT (season, week, game_id) DO NOTHING;
""")


def migrate(sqlite_path: Path) -> None:
    if not sqlite_path.exists():
        print(f"No sqlite db found at {sqlite_path}")
        sys.exit(1)

    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions;").fetchall()]

    if not rows:
        print("Nothing to migrate - source table is empty.")
        return

    for r in rows:
        r.pop("id", None)  # let Postgres assign fresh ids via SERIAL

    ensure_db()

    with engine.begin() as pg_conn:
        pg_conn.execute(INSERT_SQL, rows)
        total = pg_conn.execute(text("SELECT COUNT(*) FROM predictions;")).scalar()

    print(f"Attempted {len(rows)} rows from {sqlite_path}. Postgres table now has {total} rows total "
          f"(duplicates on season/week/game_id were skipped).")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SQLITE_PATH
    migrate(path)
