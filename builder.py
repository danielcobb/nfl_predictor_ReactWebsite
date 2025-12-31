"""
builder.py
----------
- Loads NFL data (nflreadpy)
- Builds rolling features + game features
- Trains ensemble models
- Generates predictions for selected weeks
- Writes predictions into a SQLite DB as JSON blobs

Run locally (NOT on Render):
    python builder.py --season 2025 --weeks 1-18

Notes:
- This script is meant to be run manually (or via CI) to precompute predictions.
- Your FastAPI backend should only READ from the SQLite DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import nflreadpy as nfl

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Paths
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "python_files" / "predictions.sqlite"


# -----------------------------
# Data loading
# -----------------------------

def get_team_stats(seasons: list[int]) -> pd.DataFrame:
    team_stats = nfl.load_team_stats(seasons, summary_level="week")
    df = team_stats.to_pandas()
    df = df.sort_values(["season", "team", "week"]).reset_index(drop=True)
    return df


def get_schedule(season: int) -> pd.DataFrame:
    sch = nfl.load_schedules(season)
    df = sch.to_pandas()

    if "gameday" in df.columns and "game_date" not in df.columns:
        df["game_date"] = pd.to_datetime(df["gameday"])

    return df


# -----------------------------
# Feature engineering
# -----------------------------

STAT_COLUMNS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "passing_first_downs", "passing_epa", "passing_cpoe",
    "sacks_suffered", "sack_yards_lost",
    "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_epa",
    "receiving_yards", "receiving_tds", "receiving_first_downs", "receiving_epa",
    "def_sacks", "def_sack_yards", "def_qb_hits",
    "def_interceptions", "def_interception_yards",
    "def_pass_defended", "def_tds",
    "def_tackles_for_loss", "def_tackles_for_loss_yards",
    "def_fumbles_forced", "def_fumbles",
    "special_teams_tds", "fg_made", "fg_att", "pat_made",
    "penalties", "penalty_yards",
    "fumble_recovery_opp", "fumble_recovery_own",
]

KEY_STATS = [
    "passing_yards", "rushing_yards", "passing_tds", "rushing_tds",
    "passing_epa", "rushing_epa", "receiving_epa",
    "passing_interceptions", "rushing_fumbles_lost",
    "sacks_suffered", "penalties",
    "def_sacks", "def_interceptions", "def_tds",
    "def_tackles_for_loss", "def_fumbles_forced",
]


def build_rolling_features(team_stats: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Builds rolling features within each (season, team). Uses shift(1) to avoid leakage.
    """
    rolling_parts: list[pd.DataFrame] = []

    for (season, team), group in team_stats.groupby(["season", "team"]):
        group = group.sort_values("week").copy()

        for col in STAT_COLUMNS:
            if col in group.columns:
                group[f"{col}_roll_{window}"] = (
                    group[col].shift(1).rolling(window=window, min_periods=1).mean()
                )

        rolling_parts.append(group)

    return pd.concat(rolling_parts, ignore_index=True)


def _latest_team_stats(team_stats: pd.DataFrame, team: str, season: int, week: int) -> pd.DataFrame:
    """
    Fetch the most recent row of rolling stats for `team` before `week` in `season`.
    If none exists (early season), fall back to the last available row from prior seasons.
    """
    current = team_stats[
        (team_stats["team"] == team) &
        (team_stats["season"] == season) &
        (team_stats["week"] < week)
    ].tail(1)

    if not current.empty:
        return current

    previous = team_stats[
        (team_stats["team"] == team) &
        (team_stats["season"] < season)
    ].sort_values(["season", "week"]).tail(1)

    return previous


def create_game_features(schedule: pd.DataFrame, team_stats: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    games: list[dict] = []

    for _, game in schedule.iterrows():
        # Played?
        if pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
            result = None
        else:
            result = 1 if game["home_score"] > game["away_score"] else 0

        season = int(game["season"])
        week = int(game["week"])
        home = str(game["home_team"])
        away = str(game["away_team"])

        home_stats = _latest_team_stats(team_stats, home, season, week)
        away_stats = _latest_team_stats(team_stats, away, season, week)

        if home_stats.empty or away_stats.empty:
            continue

        row: dict = {
            "game_id": f"{season}_{week}_{home}_{away}",
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "result": result,
            "game_date": None if pd.isna(game.get("game_date")) else str(game.get("game_date")),
        }

        # rolling columns
        roll_suffix = f"_roll_{window}"

        for col in home_stats.columns:
            if col.endswith(roll_suffix):
                row[f"home_{col}"] = float(home_stats[col].values[0]) if pd.notna(home_stats[col].values[0]) else 0.0

        for col in away_stats.columns:
            if col.endswith(roll_suffix):
                row[f"away_{col}"] = float(away_stats[col].values[0]) if pd.notna(away_stats[col].values[0]) else 0.0

        # differentials
        for stat in KEY_STATS:
            h = f"home_{stat}{roll_suffix}"
            a = f"away_{stat}{roll_suffix}"
            if h in row and a in row:
                row[f"{stat}_diff"] = float(row[h] - row[a])

        games.append(row)

    return pd.DataFrame(games)


# -----------------------------
# Modeling
# -----------------------------

def train_models(X: np.ndarray, y: pd.Series) -> dict:
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    trained: dict = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        model.fit(X, y)
        trained[name] = {"model": model, "cv_score": float(scores.mean())}

    return trained


def make_predictions(models: dict, X: np.ndarray, games: pd.DataFrame) -> list[dict]:
    preds: list[dict] = []

    weights = [m["cv_score"] for m in models.values()]

    for i in range(len(games)):
        game = games.iloc[i]
        probs: list[float] = []

        for m in models.values():
            probs.append(float(m["model"].predict_proba(X[i:i+1])[0][1]))

        p = float(np.average(probs, weights=weights))

        home = str(game["home_team"])
        away = str(game["away_team"])

        preds.append({
            "game_id": str(game["game_id"]),
            "season": int(game["season"]),
            "week": int(game["week"]),
            "home_team": home,
            "away_team": away,
            "predicted_winner": home if p > 0.5 else away,
            "home_win_probability": p,
            "confidence": float(abs(p - 0.5) * 2),
            "game_date": None if pd.isna(game.get("game_date")) else str(game.get("game_date")),
        })

    return preds


# -----------------------------
# SQLite writer (Option A)
# -----------------------------

def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
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


def upsert_week(db_path: Path, season: int, week: int, preds: list[dict]) -> None:
    payload = json.dumps(preds)
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions (season, week, created_at, data)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(season, week) DO UPDATE SET
                created_at = excluded.created_at,
                data = excluded.data
            """,
            (season, week, created_at, payload),
        )
        conn.commit()


# -----------------------------
# Build pipeline
# -----------------------------

def build_db(
    season: int,
    weeks: list[int],
    lookback_years: int = 2,
    window: int = 4,
    db_path: Path = DB_PATH,
) -> None:
    """
    End-to-end: loads data, builds features once, then trains/predicts per week and writes DB.
    """
    init_db(db_path)

    seasons = [season - i for i in range(lookback_years, 0, -1)] + [season]
    print(f"Seasons: {seasons}")
    print("Loading team stats...")
    team_stats = get_team_stats(seasons)
    print("Loading schedules...")
    schedules = pd.concat([get_schedule(y) for y in seasons], ignore_index=True)

    print("Building rolling features...")
    team_stats = build_rolling_features(team_stats, window=window)

    print("Creating game-level features...")
    games = create_game_features(schedules, team_stats, window=window)

    if games.empty:
        raise RuntimeError("No game features were produced. Check source data columns.")

    for wk in weeks:
        print(f"Computing predictions: season {season}, week {wk} ...")

        train_games = games[
            (games["result"].notna()) &
            ((games["season"] < season) | ((games["season"] == season) & (games["week"] < wk)))
        ].copy()

        predict_games = games[
            (games["season"] == season) & (games["week"] == wk)
        ].copy()

        if train_games.empty or predict_games.empty:
            preds: list[dict] = []
            upsert_week(db_path, season, wk, preds)
            continue

        feature_cols = [
            c for c in train_games.columns
            if c not in ["game_id", "season", "week", "home_team", "away_team", "result", "game_date"]
        ]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_games[feature_cols].fillna(0))
        y_train = train_games["result"]

        models = train_models(X_train, y_train)

        X_pred = scaler.transform(
            predict_games.reindex(columns=feature_cols, fill_value=0).fillna(0)
        )

        preds = make_predictions(models, X_pred, predict_games)
        upsert_week(db_path, season, wk, preds)

    print(f"Done. Wrote SQLite DB to: {db_path}")


# -----------------------------
# CLI
# -----------------------------

def _parse_weeks(raw: str) -> list[int]:
    """
    Accepts:
      "1-18" or "1,2,3" or "5"
    """
    raw = raw.strip()
    if "-" in raw:
        a, b = raw.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(raw)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictions.sqlite (Option A, offline).")
    parser.add_argument("--season", type=int, default=2025, help="Target season (default: 2025)")
    parser.add_argument("--weeks", type=str, default="1-18", help="Weeks e.g. '1-18' or '16' or '1,2,3'")
    parser.add_argument("--lookback-years", type=int, default=2, help="How many prior seasons to include (default: 2)")
    parser.add_argument("--window", type=int, default=4, help="Rolling window size (default: 4)")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH), help="Output SQLite path (default: ./data/predictions.sqlite)")
    args = parser.parse_args()

    weeks = _parse_weeks(args.weeks)
    if any(w < 1 or w > 18 for w in weeks):
        raise ValueError("Weeks must be between 1 and 18.")

    build_db(
        season=args.season,
        weeks=weeks,
        lookback_years=args.lookback_years,
        window=args.window,
        db_path=Path(args.db_path),
    )


if __name__ == "__main__":
    main()
