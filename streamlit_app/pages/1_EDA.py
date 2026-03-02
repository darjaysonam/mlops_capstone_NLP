import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("📊 Exploratory Data Analysis")

DATA_PATH = "../data/processed/cleaned_data.csv"

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Total Records")
st.write(len(df))

# Simple word count distribution
df["length"] = df["org_caption"].astype(str).apply(len)

st.subheader("Report Length Distribution")

fig, ax = plt.subplots()
ax.hist(df["length"], bins=30)
ax.set_xlabel("Character Length")
ax.set_ylabel("Frequency")

st.pyplot(fig)
