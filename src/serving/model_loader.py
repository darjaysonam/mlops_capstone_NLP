"""
Loads latest model from MLflow registry
"""

import mlflow.pytorch


def load_model():
    model = mlflow.pytorch.load_model(
        "models:/ChestXray-Classification/Production"
    )
    model.eval()
    return model