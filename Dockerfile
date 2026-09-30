# Serving image: only the API, the model artifacts and the reference data.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/reference/ data/reference/
COPY reports/player_scores.csv reports/player_scores.csv

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

# Cloud Run injects $PORT
CMD ["sh", "-c", "uvicorn xpass.api:app --host 0.0.0.0 --port ${PORT}"]
