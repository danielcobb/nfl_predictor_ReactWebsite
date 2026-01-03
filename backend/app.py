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


@app.get('/predictions')
def get_predictions(week: int = Query(..., ge=1, le=18, description="NFL week (1-18)"), 
                    season: int = Query(2025, description="NFL season year")):
    """
    Return predictions for a selected NFL week
    """

    predictions = load_predictions(DB_PATH, season=season, week=week)

    if not predictions:
        raise HTTPException(status_code=404, detail=f"No predictions available for week {week}, season {season}")
    
    return {
        "season" : season,
        "week" : week,
        "num_games" : len(predictions),
        "predictions" : predictions,
    }
