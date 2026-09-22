# syntax=docker/dockerfile:1

# ASSUMPTION: backend/ has no its own Python version pin. python_files/.python-version
# and python_files/pyproject.toml (requires-python >=3.13) are the only version pins
# anywhere in the repo, so 3.13 is used here too.
FROM python:3.13-slim

# backend/app.py does `from backend.main import load_predictions` - the app is the
# module `backend.app:app`, resolved relative to the repo root, exactly like on EC2
# (see backend/app.py's WorkingDirectory note). So the repo root is the working
# directory here, and the `backend` package sits directly under it, not the other
# way around.
WORKDIR /app

# Install deps in their own layer so this is only re-run when requirements.txt changes.
# EC2 used a checked-in venv at backend/.venv; the image installs into the base
# interpreter fresh instead (see .dockerignore).
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code. Predictions now live in Postgres (connection configured via env vars /
# docker-compose.yml's env_file, see backend/db.py), so there's no sqlite file to
# exclude here anymore. backend/model_cache/*.joblib IS copied in (it's committed,
# pre-trained model data) and then layered under a named volume in compose so the
# cache still survives rebuilds once the app writes new entries.
COPY backend/ backend/

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
