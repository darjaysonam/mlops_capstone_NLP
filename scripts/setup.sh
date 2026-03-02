#!/bin/bash

# ─── MLOps Capstone Setup Script ─────────────────────────────────────────

echo "Creating virtual environment..."
python -m venv venv

echo "Activating environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"