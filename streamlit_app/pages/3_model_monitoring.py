import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import sys
import torch
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------
# Setup
# ---------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.nlp.inference import ModelService
from src.nlp.training import create_multilabel_data

st.title("📈 Model Performance & Explainability Dashboard")

model_service = ModelService()

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

DATA_PATH = "../data/processed/cleaned_data.csv"

st.subheader("📂 Load Evaluation Dataset")

texts, labels = create_multilabel_data(DATA_PATH)

eval_sample_size = st.slider(
    "Evaluation Sample Size (Performance Metrics)",
    min_value=50,
    max_value=min(500, len(texts)),
    value=100,
)

texts_eval = texts[:eval_sample_size]
labels_eval = labels[:eval_sample_size]

# ---------------------------------------------------
# Generate Predictions
# ---------------------------------------------------

st.subheader("🔄 Generating Predictions...")

embeddings = model_service.embedder.encode(texts_eval, convert_to_tensor=True)

with torch.no_grad():
    outputs = model_service.model(embeddings)
    probs = torch.sigmoid(outputs).numpy()

preds = (probs >= model_service.threshold).astype(int)

label_names = model_service.labels

# ---------------------------------------------------
# 1️⃣ Label-wise Performance Chart
# ---------------------------------------------------

st.subheader("📊 Label-wise Positive Counts")

true_counts = labels_eval.sum(axis=0)
pred_counts = preds.sum(axis=0)

fig1, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(label_names))

ax1.bar(x - 0.2, true_counts, width=0.4, label="True")
ax1.bar(x + 0.2, pred_counts, width=0.4, label="Predicted")

ax1.set_xticks(x)
ax1.set_xticklabels(label_names, rotation=45, ha="right")
ax1.legend()

st.pyplot(fig1)

# ---------------------------------------------------
# 2️⃣ Confusion Matrix (Per Label)
# ---------------------------------------------------

st.subheader("🧩 Confusion Matrix")

selected_label = st.selectbox("Select Label", label_names)
label_idx = label_names.index(selected_label)

cm = confusion_matrix(labels_eval[:, label_idx], preds[:, label_idx])

fig2, ax2 = plt.subplots()
im = ax2.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i, j], ha="center", va="center")

ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")
ax2.set_title(f"Confusion Matrix - {selected_label}")

st.pyplot(fig2)

# ---------------------------------------------------
# 3️⃣ Probability Distribution
# ---------------------------------------------------

st.subheader("📈 Probability Distribution")

fig3, ax3 = plt.subplots()
ax3.hist(probs[:, label_idx], bins=30)
ax3.set_xlabel("Predicted Probability")
ax3.set_ylabel("Frequency")
ax3.set_title(f"Probability Distribution - {selected_label}")

st.pyplot(fig3)

# ---------------------------------------------------
# 4️⃣ SHAP Explainability (Small Sample)
# ---------------------------------------------------

st.subheader("🔍 SHAP Explainability")

st.info("SHAP is computationally expensive. It runs on a small sample (10 texts).")

if st.button("Run SHAP Explanation"):

    shap_sample_size = 10
    shap_texts = texts[:shap_sample_size]

    shap_embeddings = model_service.embedder.encode(
        shap_texts, convert_to_tensor=True
    ).numpy()

    shap_label = st.selectbox(
        "Select Label for SHAP", label_names, key="shap_label_select"
    )

    shap_label_idx = label_names.index(shap_label)

    def model_forward(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = model_service.model(x_tensor)
            probs = torch.sigmoid(outputs).numpy()
        return probs[:, shap_label_idx]

    background = shap_embeddings[:5]

    explainer = shap.KernelExplainer(model_forward, background)

    shap_values = explainer.shap_values(shap_embeddings)

    st.subheader(f"SHAP Summary Plot - {shap_label}")

    fig4 = plt.figure()
    shap.summary_plot(shap_values, shap_embeddings, show=False)
    st.pyplot(fig4)

    st.subheader(f"SHAP Feature Importance (Bar) - {shap_label}")

    fig5 = plt.figure()
    shap.summary_plot(shap_values, shap_embeddings, plot_type="bar", show=False)
    st.pyplot(fig5)
