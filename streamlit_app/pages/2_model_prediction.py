import streamlit as st
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.nlp.inference import ModelService

st.title("🤖 Disease Prediction")

model_service = ModelService()

text_input = st.text_area("Enter Radiology Narrative", height=200)

threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.4)

if st.button("Predict"):

    if text_input.strip() == "":
        st.warning("Please enter text.")
    else:
        predictions = model_service.predict(text_input)

        filtered = [p for p in predictions if p["probability"] >= threshold]

        st.subheader("Prediction Results")

        if filtered:
            st.table(filtered)
        else:
            st.info("No disease predicted above threshold.")
