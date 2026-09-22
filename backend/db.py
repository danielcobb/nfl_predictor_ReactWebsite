"""Postgres connection setup, shared by main.py, app.py, and evaluate_accuracy.py."""
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "nfl_predictor")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


engine: Engine = create_engine(_database_url(), pool_pre_ping=True)
