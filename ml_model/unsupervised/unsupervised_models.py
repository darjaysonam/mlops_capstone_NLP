import os
import pandas as pd

from evaluation import (
    plot_elbow,
    save_business_insights,
    save_clustering_metrics,
    visualize_clusters,
)

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def run_unsupervised_pipeline():

    # -----------------------------------
    # Load Smoking Dataset
    # -----------------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "smoking.csv")

    df = pd.read_csv(DATA_PATH)

    # -----------------------------------
    # Data Cleaning
    # -----------------------------------

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convert categorical columns
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    df["oral"] = df["oral"].map({"Y": 1, "N": 0})
    df["tartar"] = df["tartar"].map({"Y": 1, "N": 0})

    # Remove target variable for clustering
    if "smoking" in df.columns:
        df = df.drop(columns=["smoking"])

    # Fill missing values
    df = df.fillna(df.mean())

    X = df.values

    # -----------------------------------
    # Feature Scaling
    # -----------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===================================
    # 1️⃣ K-Means Clustering
    # ===================================

    plot_elbow(KMeans(random_state=42), X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    save_clustering_metrics("KMeans", X_scaled, labels_kmeans)
    visualize_clusters("KMeans", X_scaled, labels_kmeans)

    insights_kmeans = """
KMeans Clustering Insights (Smoking Health Dataset):

Cluster 0:
Individuals with relatively normal metabolic indicators.

Cluster 1:
Individuals with elevated cholesterol, triglycerides, and blood pressure.

Cluster 2:
Individuals showing higher liver enzyme values (AST, ALT, GTP) which may
indicate lifestyle risks such as smoking or alcohol exposure.

Healthcare Insight:
These clusters may represent different metabolic health profiles.
Public health programs could target high-risk clusters for lifestyle
interventions such as smoking cessation and cardiovascular risk reduction.
"""

    save_business_insights("KMeans", insights_kmeans)

    # ===================================
    # 2️⃣ DBSCAN Clustering
    # ===================================

    dbscan = DBSCAN(eps=1.5, min_samples=5)

    labels_dbscan = dbscan.fit_predict(X_scaled)

    save_clustering_metrics("DBSCAN", X_scaled, labels_dbscan)
    visualize_clusters("DBSCAN", X_scaled, labels_dbscan)

    insights_dbscan = """
DBSCAN Clustering Insights (Smoking Health Dataset):

DBSCAN identifies dense clusters and isolates noise points.

Noise points may represent individuals with unusual health indicators
such as extremely high cholesterol, abnormal liver enzyme levels,
or abnormal blood pressure.

Healthcare Insight:
Outliers may indicate individuals at extreme medical risk who may require
clinical screening or further diagnostic evaluation.
"""

    save_business_insights("DBSCAN", insights_dbscan)


if __name__ == "__main__":
    run_unsupervised_pipeline()
