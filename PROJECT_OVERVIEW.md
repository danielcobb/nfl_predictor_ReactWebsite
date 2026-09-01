# NFL Predictor — Project Overview

A full-stack web app that predicts NFL game outcomes using a machine-learning
ensemble trained on historical team performance data. Users pick a season and
week/playoff round in a React frontend; a FastAPI backend serves pre-computed
predictions from a SQLite database.

Repo: `nfl_predictor_ReactWebsite`
Live frontend: https://danielcobb.github.io (GitHub Pages)
Live backend: https://nfl-predictor-reactwebsite.onrender.com (Render)

---

## 1. High-level architecture

```
┌─────────────────────────┐        HTTPS GET /predictions        ┌──────────────────────────┐
│   React + TS frontend   │ ────────────────────────────────────▶│   FastAPI backend (app.py)│
│  (nfl-predictor-react-  │                                       │   Render web service      │
│   app/, Vite, GH Pages) │ ◀──────────────────────────────────── │                            │
└─────────────────────────┘        JSON predictions               └───────────┬──────────────┘
                                                                                │ reads
                                                                                ▼
                                                                    ┌──────────────────────┐
                                                                    │  predictions.db        │
                                                                    │  (SQLite, committed    │
                                                                    │   to the repo)         │
                                                                    └──────────┬────────────┘
                                                                               ▲ writes
                                                                               │ (offline / manual run)
                                                                    ┌──────────┴────────────┐
                                                                    │  main.py                │
                                                                    │  training + prediction  │
                                                                    │  pipeline (run locally) │
                                                                    └──────────┬────────────┘
                                                                               │ pulls historical
                                                                               │ team stats & schedules
                                                                               ▼
                                                                    ┌──────────────────────┐
                                                                    │  nflreadpy             │
                                                                    │  (nflverse data)        │
                                                                    └──────────────────────┘
```

**Key design point:** predictions are *not* computed on request. `main.py` is
run manually (or via a script) to train models and write predictions into
`predictions.db`, which is committed to the repo. The deployed FastAPI service
only ever *reads* that database — it never trains a model or hits `nflreadpy`
at request time. This keeps the API fast and avoids Render cold-start /
timeout issues with a heavyweight ML pipeline.

---

## 2. Backend (`backend/`)

### 2.1 `main.py` — data, features, training, prediction

- **Data source:** [`nflreadpy`](https://github.com/nflverse/nflreadpy), a
  Python client for the nflverse data releases (schedules, weekly team stats).
- **`get_team_stats(seasons)`** — loads weekly team stats per season,
  skipping any season that has no published data yet (e.g. a season that
  hasn't started) instead of failing. This matters for predicting the
  *current* season before any games have been played.
- **`get_schedule(season)`** — loads the season's game schedule (future games
  included, with null scores).
- **`build_rolling_features(team_stats, window=4)`** — for ~30 offensive/
  defensive/special-teams stats, computes a trailing 4-game rolling average
  per team (shifted by 1 so a team's upcoming-game features never leak the
  result of that same game).
- **`create_game_features(schedule, team_stats)`** — joins each scheduled
  game to the *most recent* rolling stats available for the home/away team
  at that point in time (falling back to the team's last known stats from a
  prior season if the current season has no history yet, e.g. week 1).
  Produces per-stat home/away differential features (`*_diff`).
- **`train_models(X, y)`** — trains four classifiers with 5-fold CV:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - SVM (RBF kernel, probability output)
- **`make_predictions(models, X, games)`** — ensembles the four models'
  win-probability outputs, weighted by each model's cross-validation
  accuracy score, into a single home-win probability. `confidence` is
  derived as `abs(prob - 0.5) * 2` (0 = coin flip, 1 = certain).
- **`predict_week(week, season)`** — orchestrates the above: trains on all
  completed games from `[season-2, season-1, season]` up to (but not
  including) the target week, then predicts that week's games. Trained
  model bundles are cached to disk (`model_cache/models_{season}_week_{week}.joblib`
  via `joblib`) so re-running the same week doesn't retrain from scratch.
- **`save_predictions()` / `load_predictions()`** — SQLite persistence layer
  (see schema below). Upserts on `(season, week, game_id)`, so re-running a
  week updates existing rows rather than duplicating them.
- **`__main__` block** — the manual entry point: loops weeks 1–22 for a
  given season, predicts, and saves to `predictions.db`. This is what you
  run (`python backend/main.py`) to refresh predictions for a new week or
  season. Playoff weeks (19–22) return no predictions until the playoff
  bracket exists in the schedule data.

### 2.2 `evaluate_accuracy.py` — grading past predictions

Standalone script: compares stored predictions against actual final scores
(pulled fresh from `nflreadpy`) and prints a per-game and per-week accuracy
report. `--backfill` also writes the actual winner back into the
`actual_winner` column in `predictions.db` for later analysis.

```
python backend/evaluate_accuracy.py 2025 --backfill
```

### 2.3 `app.py` — the API

FastAPI app with CORS restricted to the known frontend origins
(`localhost:5173`, `127.0.0.1:5173`, `danielcobb.github.io`).

**`GET /predictions`**
| Param | Type | Notes |
|---|---|---|
| `week` | int, required, 1–22 | 1–18 regular season, 19–22 playoff rounds |
| `season` | int, default 2026 | NFL season year |

Response:
```json
{
  "season": 2026,
  "week": 1,
  "round_name": null,
  "num_games": 16,
  "predictions": [
    {
      "game_id": "2026_1_CAR_CHI",
      "season": 2026,
      "week": 1,
      "home_team": "CAR",
      "away_team": "CHI",
      "predicted_winner": "CHI",
      "home_win_prob": 0.264,
      "confidence": 0.472
    }
  ]
}
```
Weeks 19–22 map to `round_name` values Wild Card / Divisional Round /
Conference Championships / Super Bowl. Returns `404` if no predictions exist
for the requested season/week (e.g. a future week not yet run through
`main.py`, or a playoff round that hasn't been played into yet).

### 2.4 Database schema (`predictions.db`)

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    home_win_prob REAL,
    away_win_prob REAL,
    confidence REAL,
    model_name TEXT,
    created_at TEXT NOT NULL,
    actual_winner TEXT,
    UNIQUE(season, week, game_id)
);
```

The DB file itself (and the `model_cache/*.joblib` bundles) are committed to
the repo — there's no separate "prediction generation" deploy step; whoever
runs `main.py` locally and commits the updated `predictions.db` is
effectively "publishing" that week's predictions.

---

## 3. Frontend (`nfl-predictor-react-app/`)

React 19 + TypeScript + Vite, deployed to GitHub Pages via `npm run deploy`
(`gh-pages -d dist`).

- **`App.tsx`** — top-level state: selected season/week, fetched games,
  loading/error state, "sort by confidence" toggle. Fetches
  `${API_BASE}/predictions?week=&season=` whenever week or season changes.
  `API_BASE` is hardcoded to the Render URL.
- **`components/Header.tsx`** — static branding header (SVG football icon,
  wordmark).
- **`components/SelectMenu.tsx`** — season dropdown (`SEASONS` constant,
  currently `[2022, 2023, 2024, 2025, 2026]`) and week dropdown (weeks 1–18
  plus the four playoff rounds as an `<optgroup>`).
- **`components/GameList.tsx`** — renders a grid of game cards: team logos,
  predicted winner callout, a confidence badge (Low/Med/High thresholds at
  50%/70%), and a home/away win-probability bar. Shows skeleton loading
  cards while fetching and an empty state before a week is selected.
- **`components/TeamBadge.tsx`** / **`teamLogos.tsx`** — team abbreviation →
  logo image asset mapping.
- **`types.tsx`** — shared `GamePrediction` / `PredictionsResponse` types
  matching the API response shape exactly.

No routing, no global state library, no backend calls other than the one
`/predictions` endpoint — deliberately simple.

---

## 4. Deployment

| Layer | Host | How |
|---|---|---|
| Frontend | GitHub Pages (`danielcobb.github.io`) | `npm run deploy` → builds and pushes `dist/` via `gh-pages` |
| Backend | Render | Runs `uvicorn` serving `app.py`; reads the committed `predictions.db` |
| Data refresh | Manual, local | Run `python backend/main.py` locally, commit the updated `predictions.db` + `model_cache/`, push |

There is currently no CI/CD or scheduled job that automatically regenerates
predictions — someone has to run the script and push the updated DB.

---

## 5. Known limitations

- **No automatic weekly refresh.** Predictions for a new week only appear
  after a human runs `main.py` and commits the DB. If that doesn't happen
  before kickoff, the API 404s for that week.
- **No live/in-progress game handling.** The model only knows completed
  games (via rolling stats) and the schedule; it doesn't account for
  injuries, weather, betting lines, or roster changes.
- **Small, static ensemble.** Four scikit-learn models trained on rolling
  4-game stat averages — no deep learning, no player-level data, no
  opponent-adjusted (SOS) metrics.
- **Model cache can go stale.** `model_cache/models_{season}_week_{week}.joblib`
  is reused if present, so retraining only happens on cache miss — if the
  underlying training data changes (e.g. corrected stats), the cache needs
  to be manually cleared to pick it up.
- **Single SQLite file as the datastore.** Fine at this scale, but it's
  committed binary data in git, which will make the repo history grow and
  makes concurrent writes from multiple contributors awkward.
- **No auth, no user accounts, no personalization** — the app is read-only
  and stateless from the user's perspective.
- **No tests.** No automated test suite on either the Python or TypeScript
  side.

---

## 6. Ideas for future improvements / features

**Automation & ops**
- Scheduled job (GitHub Actions cron, Render cron job, etc.) to run
  `main.py` weekly and auto-commit/push the refreshed `predictions.db`,
  removing the manual step.
- Move off committing a binary SQLite file to git — host the DB on a small
  managed Postgres/SQLite-on-Turso/Litestream setup, or regenerate it as a
  build artifact instead of versioning it.
- CI: lint + typecheck (`eslint`, `tsc -b`) and a basic backend smoke test
  (`predict_week` returns non-empty for a known week) on every PR.

**Modeling quality**
- Add strength-of-schedule / opponent-adjusted rolling stats instead of raw
  rolling averages.
- Incorporate Vegas betting lines (spread/moneyline) as a feature or as a
  baseline to beat.
- Track and expose each individual model's live accuracy (not just the
  ensemble) to see which models are pulling their weight.
- Add injury report / starting QB data — currently the biggest blind spot
  for a stats-only model.
- Publish `evaluate_accuracy.py` results somewhere visible (a `/accuracy`
  API endpoint + a page in the frontend) instead of only running it
  ad hoc locally.
- Experiment with a proper time-series/backtesting harness instead of
  standard k-fold CV, since games are temporally ordered and a naive CV
  split can leak future information across folds within a season.

**Product / frontend features**
- Historical accuracy dashboard: show the model's track record by
  season/week, and let users see predicted-vs-actual for completed games.
- Team-detail or matchup-detail view: click a card to see the underlying
  stat differentials that drove the prediction.
- "My picks" mode: let a user make their own picks per week and compare
  against the model's picks and final results (would need lightweight
  auth/local storage).
- Push/email notifications when new weekly predictions are published.
- Mobile-friendly polish / PWA installability.
- Live score overlay for in-progress/completed games instead of only
  pre-game predictions.

**Backend/API**
- Add a `/teams` or `/schedule` endpoint so the frontend isn't limited to
  exactly what's precomputed — e.g. show upcoming schedule even for weeks
  without predictions yet.
- Add response caching headers now that data is effectively static per
  deploy, to reduce load on Render's free tier.
- Rate limiting / basic abuse protection if traffic grows.
