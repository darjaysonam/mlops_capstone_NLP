"""
ANN Model Definition (Phase 2 - 2.5)

Fully parameterized to allow:
- Different activation functions
- Optional BatchNorm
- Optional Dropout
"""

import torch
import torch.nn as nn


class ANNClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1=256,
        hidden2=128,
        activation="relu",
        use_batchnorm=True,
        dropout=0.5,
    ):
        super(ANNClassifier, self).__init__()

        # -------------------------
        # Activation Selection
        # -------------------------
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "leakyrelu":
            act = nn.LeakyReLU(0.1)
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

        layers = []

        # Layer 1
        layers.append(nn.Linear(input_dim, hidden1))
        layers.append(act)

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden1))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Layer 2
        layers.append(nn.Linear(hidden1, hidden2))
        layers.append(act)

        # Output layer (Binary classification)
        layers.append(nn.Linear(hidden2, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
