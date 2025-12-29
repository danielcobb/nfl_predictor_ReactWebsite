import pandas as pd
import nflreadpy as nfl
import numpy as np
import joblib
import sqlite3
import json
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# (keep your existing functions above here)
# get_team_stats, get_schedule, build_rolling_features,
# _latest_team_stats, create_game_features, train_models, make_predictions
# -----------------------------


# -----------------------------
# Option A: Read-only DB runtime
# -----------------------------

# Put the DB somewhere stable. Since Render needs app.py/main.py at repo root,
# this points to: <repo_root>/data/predictions.sqlite
_DB_PATH = Path(__file__).resolve().parent / "data" / "predictions.sqlite"


def _load_predictions_from_db(season: int, week: int) -> list[dict] | None:
    """
    Read-only: returns cached predictions if present, else None.
    """
    if not _DB_PATH.exists():
        return None

    with sqlite3.connect(_DB_PATH) as conn:
        row = conn.execute(
            "SELECT data FROM predictions WHERE season = ? AND week = ?",
            (season, week),
        ).fetchone()

    if not row:
        return None

    return json.loads(row[0])


def predict_week(week: int, season: int = 2025) -> list[dict]:
    """
    Runtime entrypoint used by FastAPI.
    Option A: only reads from the precomputed SQLite DB.
    """
    preds = _load_predictions_from_db(season, week)
    return preds or []


# -----------------------------
# Builder: run locally to generate the DB
# -----------------------------

def build_predictions_db(
    season: int = 2025,
    weeks: list[int] | None = None,
    lookback_years: int = 2,
) -> None:
    """
    Build (or rebuild) the SQLite predictions DB locally.
    This DOES train models + compute predictions and writes them into SQLite.

    - season: target season to predict (e.g., 2025)
    - weeks: list of weeks to precompute. If None, computes 1..18
    - lookback_years: how many prior seasons to include for training/features
    """
    if weeks is None:
        weeks = list(range(1, 19))

    # Ensure folder exists
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create schema fresh each time (simple + reliable)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS predictions")
        conn.execute(
            """
            CREATE TABLE predictions (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (season, week)
            )
            """
        )
        conn.commit()

    seasons = [season - i for i in range(lookback_years, 0, -1)] + [season]

    print(f"Loading stats/schedules for seasons: {seasons}")
    team_stats = get_team_stats(seasons)
    schedules = pd.concat([get_schedule(y) for y in seasons], ignore_index=True)

    team_stats = build_rolling_features(team_stats)
    games = create_game_features(schedules, team_stats)

    # Precompute once: use the same feature list per training slice
    # We'll compute per-week training sets, because your training cutoff changes by week.
    for wk in weeks:
        print(f"Building predictions for {season} week {wk}...")

        train_games = games[
            (games["result"].notna()) &
            ((games["season"] < season) | ((games["season"] == season) & (games["week"] < wk)))
        ].copy()

        predict_games = games[
            (games["season"] == season) & (games["week"] == wk)
        ].copy()

        if train_games.empty or predict_games.empty:
            preds = []
        else:
            features = [
                c for c in train_games.columns
                if c not in ["game_id", "season", "week", "home_team", "away_team", "result", "game_date"]
            ]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_games[features].fillna(0))
            y_train = train_games["result"]

            models = train_models(X_train, y_train)

            X_pred = scaler.transform(
                predict_games.reindex(columns=features, fill_value=0).fillna(0)
            )

            preds = make_predictions(models, X_pred, predict_games)

        # Write results
        payload = json.dumps(preds)
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions (season, week, created_at, data)
                VALUES (?, ?, ?, ?)
                """,
                (season, wk, datetime.utcnow().isoformat(), payload),
            )
            conn.commit()

    print(f"Done. Wrote DB to: {_DB_PATH}")


if __name__ == "__main__":
    # Local build example:
    # build_predictions_db(season=2025, weeks=list(range(1, 19)))
    # Then test reads:
    # print(predict_week(1, 2025))
    pass
