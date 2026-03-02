import streamlit as st

st.set_page_config(page_title="Radiology ML Dashboard", layout="wide")

st.title("🧠 Radiology Disease Prediction System")

st.markdown("""
Welcome to the Streamlit Interactive Dashboard.

Use the left sidebar to navigate:

• 📊 EDA  
• 🤖 Model Prediction  
• 📈 Model Monitoring  
""")

st.info("This dashboard demonstrates NLP-based multi-label radiology classification.")
