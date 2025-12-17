from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from main import predict_week

app = FastAPI(title="NFL Game Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
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

    predictions = predict_week(week=week, season=season)

    if not predictions:
        raise HTTPException(status_code=404, detail=f"No predictions available for week {week}, season {season}")
    
    return {
        "season" : season,
        "week" : week,
        "num_games" : len(predictions),
        "predictions" : predictions,
    }