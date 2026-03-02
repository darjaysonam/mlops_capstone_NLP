#!/bin/bash

# ─── Run Full ML Pipeline ─────────────────────────────────────────────────

echo "Starting MLflow..."
mlflow server --host 0.0.0.0 --port 5000 &

echo "Training model..."
python src/training/trainer.py

echo "Starting API..."
uvicorn src.serving.api:app --reload