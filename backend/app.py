from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from backend.main import load_predictions

app = FastAPI(title="NFL Game Predictor API")

DB_PATH = str(Path(__file__).resolve().parent / "predictions.db")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
        "https://danielcobb.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PLAYOFF_ROUND_NAMES = {
    19: "Wild Card",
    20: "Divisional Round",
    21: "Conference Championships",
    22: "Super Bowl",
}

@app.get('/predictions')
def get_predictions(week: int = Query(..., ge=1, le=22, description="NFL week (1-22; 19-22 are playoff rounds)"),
                    season: int = Query(2025, description="NFL season year")):
    """
    Return predictions for a selected NFL week or playoff round.
    Weeks 19-22 correspond to Wild Card, Divisional, Conference Championships, and Super Bowl.
    """

    raw_predictions = load_predictions(DB_PATH, season=season, week=week)
    predictions = []
    for row in raw_predictions:
        home_prob = row.get("home_win_prob")
        if home_prob is not None:
            home_prob = float(home_prob)
        if home_prob is None:
            confidence = None
        else:
            confidence = abs(float(home_prob) - 0.5) * 2
        predictions.append({
            "game_id": row.get("game_id"),
            "season": row.get("season"),
            "week": row.get("week"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "predicted_winner": row.get("predicted_winner"),
            "home_win_prob": home_prob,
            "confidence": confidence,
        })

    if not predictions:
        round_name = PLAYOFF_ROUND_NAMES.get(week, f"Week {week}")
        raise HTTPException(status_code=404, detail=f"No predictions available for {round_name}, season {season}")

    round_name = PLAYOFF_ROUND_NAMES.get(week)

    return {
        "season": season,
        "week": week,
        "round_name": round_name,
        "num_games": len(predictions),
        "predictions": predictions,
    }
