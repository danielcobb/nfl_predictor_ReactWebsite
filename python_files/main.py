import pandas as pd
import nflreadpy as nfl
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


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

def build_rolling_features(team_stats: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    stat_columns = [
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

    rolling_stats = []

    for (season, team), group in team_stats.groupby(["season", "team"]):
        group = group.sort_values("week").copy()

        for col in stat_columns:
            if col in group.columns:
                group[f"{col}_roll_{window}"] = (
                    group[col].shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )

        rolling_stats.append(group)

    return pd.concat(rolling_stats, ignore_index=True)


def create_game_features(schedule: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    games = []

    for _, game in schedule.iterrows():
        if pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
            result = None
        else:
            result = 1 if game["home_score"] > game["away_score"] else 0

        home_stats = team_stats[
            (team_stats["team"] == game["home_team"]) &
            (team_stats["season"] == game["season"]) &
            (team_stats["week"] < game["week"])
        ].tail(1)

        away_stats = team_stats[
            (team_stats["team"] == game["away_team"]) &
            (team_stats["season"] == game["season"]) &
            (team_stats["week"] < game["week"])
        ].tail(1)

        if home_stats.empty or away_stats.empty:
            continue

        row = {
            "game_id": f"{game['season']}_{game['week']}_{game['home_team']}_{game['away_team']}",
            "season": game["season"],
            "week": game["week"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "result": result,
            "game_date": game.get("game_date"),
        }

        for col in home_stats.columns:
            if col.endswith("_roll_4"):
                row[f"home_{col}"] = home_stats[col].values[0]

        for col in away_stats.columns:
            if col.endswith("_roll_4"):
                row[f"away_{col}"] = away_stats[col].values[0]

        key_stats = [
            "passing_yards", "rushing_yards", "passing_tds", "rushing_tds",
            "passing_epa", "rushing_epa", "receiving_epa",
            "passing_interceptions", "rushing_fumbles_lost",
            "sacks_suffered", "penalties",
            "def_sacks", "def_interceptions", "def_tds",
            "def_tackles_for_loss", "def_fumbles_forced",
        ]

        for stat in key_stats:
            h = f"home_{stat}_roll_4"
            a = f"away_{stat}_roll_4"
            if h in row and a in row:
                row[f"{stat}_diff"] = row[h] - row[a]

        games.append(row)

    return pd.DataFrame(games)


# -----------------------------
# Modeling
# -----------------------------

def train_models(X, y):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    trained = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        model.fit(X, y)
        trained[name] = {"model": model, "cv_score": scores.mean()}

    return trained


def make_predictions(models, X, games):
    preds = []

    for i in range(len(games)):
        game = games.iloc[i]
        probs = []

        for m in models.values():
            probs.append(m["model"].predict_proba(X[i:i+1])[0][1])

        weights = [m["cv_score"] for m in models.values()]
        p = float(np.average(probs, weights=weights))  # force Python float

        preds.append({
            "game_id": str(game["game_id"]),
            "season": int(game["season"]),
            "week": int(game["week"]),
            "home_team": str(game["home_team"]),
            "away_team": str(game["away_team"]),
            "predicted_winner": str(game["home_team"] if p > 0.5 else game["away_team"]),
            "home_win_probability": p,
            "confidence": float(abs(p - 0.5) * 2),
            "game_date": None if pd.isna(game.get("game_date")) else str(game.get("game_date")),
        })

    return preds



# -----------------------------
# Public API entrypoint
# -----------------------------

def predict_week(week: int, season: int = 2025) -> list[dict]:
    team_stats = get_team_stats([2023, 2024, 2025])

    schedules = pd.concat(
        [get_schedule(y) for y in [2023, 2024, 2025]],
        ignore_index=True
    )

    team_stats = build_rolling_features(team_stats)
    games = create_game_features(schedules, team_stats)

    train_games = games[
        (games["result"].notna()) &
        ((games["season"] < season) | ((games["season"] == season) & (games["week"] < week)))
    ]

    predict_games = games[
        (games["season"] == season) & (games["week"] == week)
    ]

    if predict_games.empty or train_games.empty:
        return []

    features = [c for c in train_games.columns if c not in
                ["game_id", "season", "week", "home_team", "away_team", "result", "game_date"]]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_games[features].fillna(0))
    y_train = train_games["result"]

    models = train_models(X_train, y_train)

    X_pred = scaler.transform(predict_games[features].fillna(0))

    return make_predictions(models, X_pred, predict_games)


# -----------------------------
# Local test
# -----------------------------

if __name__ == "__main__":
    results = predict_week(16)
    for g in results:
        print(f"{g['away_team']} @ {g['home_team']} → {g['predicted_winner']} "
              f"({g['home_win_probability']:.1%}, conf {g['confidence']:.1%})")
