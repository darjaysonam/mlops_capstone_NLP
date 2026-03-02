"""
tests/test_models.py - Unit tests for all models, data pipeline, and API
Run with: pytest tests/ -v
"""

import torch
from src.models.cnn_model import ChestXrayCNN


def test_cnn_forward():
    model = ChestXrayCNN(num_classes=14)
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)

    assert output.shape[1] == 14