"""
NLP Models - Multi-Label Disease Classification
"""

import os
os.environ["HF_HOME"] = "D:/hf_cache"

import torch
import torch.nn as nn
from transformers import pipeline
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# MiniLM Embedder
# -------------------------------------------------

class MiniLMEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings


# -------------------------------------------------
# Multi-Label Classifier
# -------------------------------------------------

class MultiLabelNLPClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MultiLabelNLPClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------
# Zero-Shot Classifier
# -------------------------------------------------

def load_zero_shot():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )


# -------------------------------------------------
# Text Generator
# -------------------------------------------------

def load_text_generator():
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )