"""
Evaluation Script

- Loads trained MiniLM + MLP classifier
- Predicts diseases from new narrative
- Optional comparison with zero-shot model
"""

import numpy as np
import torch

from src.nlp.models import (MiniLMEmbedder, MultiLabelNLPClassifier,
                            load_zero_shot)

# -------------------------------------------------
# Target Labels (Must Match training.py)
# -------------------------------------------------

TARGET_LABELS = [
    "atelectasis",
    "pneumonia",
    "mass",
    "cardiomegaly",
    "effusion",
    "pleural thickening",
    "pneumothorax",
    "consolidation",
]


# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------


def load_trained_model():

    embedder = MiniLMEmbedder()

    # Get embedding dimension
    dummy = embedder.encode(["test"])
    input_dim = dummy.shape[1]

    model = MultiLabelNLPClassifier(input_dim=input_dim, num_labels=len(TARGET_LABELS))

    model.load_state_dict(torch.load("best_multilabel_nlp.pth"))
    model.eval()

    return embedder, model


# -------------------------------------------------
# Predict Using Trained Model
# -------------------------------------------------


def predict_with_trained_model(text):

    embedder, model = load_trained_model()

    embedding = embedder.encode([text])

    with torch.no_grad():
        logits = model(embedding)
        probs = torch.sigmoid(logits).numpy()[0]

    print("\n🧠 Trained Model Predictions:")
    for label, prob in zip(TARGET_LABELS, probs):
        print(f"{label}: {prob:.4f}")

    print("\nPredicted Diseases (threshold=0.4):")
    for label, prob in zip(TARGET_LABELS, probs):
        if prob > 0.4:
            print(f"✔ {label}")


# -------------------------------------------------
# Compare with Zero-Shot
# -------------------------------------------------


def compare_with_zero_shot(text):

    classifier = load_zero_shot()

    result = classifier(text, TARGET_LABELS, multi_label=True)

    print("\n🤖 Zero-Shot Predictions:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label}: {score:.4f}")


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    test_text = """
    There is right lower lobe opacity with pleural fluid
    and mild enlargement of the cardiac silhouette.
    """

    print("Input Narrative:")
    print(test_text)

    predict_with_trained_model(test_text)

    print("\n--------------------------------------")
    compare_with_zero_shot(test_text)
