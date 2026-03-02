"""
Model Inference Service
Includes improved negation handling
"""

import os
import re

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# MultiLabelNLPClassifier
# -------------------------------------------------


class MultiLabelNLPClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MultiLabelNLPClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------
# Model Service
# -------------------------------------------------


class ModelService:

    def __init__(self):

        self.labels = [
            "atelectasis",
            "pneumonia",
            "mass",
            "cardiomegaly",
            "effusion",
            "pleural thickening",
            "pneumothorax",
            "consolidation",
        ]

        self.threshold = 0.4

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        MODEL_PATH = os.path.join(BASE_DIR, "best_multilabel_nlp.pth")

        self.model = MultiLabelNLPClassifier(input_dim=384, num_labels=len(self.labels))

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )

        self.model.eval()

    # -------------------------------------------------
    # Improved Negation Handling
    # -------------------------------------------------

    def apply_negation_rule(self, text, predictions):

        text_lower = text.lower()

        NEGATION_TERMS = [
            "no",
            "without",
            "absent",
            "negative for",
            "no evidence of",
            "free of",
        ]

        for neg in NEGATION_TERMS:

            if neg in text_lower:

                # Capture phrase after negation word
                pattern = rf"{neg}\s+([^\.]+)"
                matches = re.findall(pattern, text_lower)

                for match in matches:

                    for pred in predictions:

                        label = pred["label"]

                        # If label appears in negated phrase
                        if label in match:
                            pred["predicted"] = False
                            pred["probability"] = 0.0

        return predictions

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(self, text):

        embedding = self.embedder.encode([text], convert_to_tensor=True)

        with torch.no_grad():
            outputs = self.model(embedding)
            probs = torch.sigmoid(outputs).numpy()[0]

        predictions = []

        for label, prob in zip(self.labels, probs):
            predictions.append(
                {
                    "label": label,
                    "probability": float(prob),
                    "predicted": bool(prob >= self.threshold),
                }
            )

        predictions = self.apply_negation_rule(text, predictions)

        return predictions
