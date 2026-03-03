from evaluation import (
    plot_elbow,
    save_business_insights,
    save_clustering_metrics,
    visualize_clusters,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def run_unsupervised_pipeline():

    # -----------------------------------
    # Example dataset (Replace with yours)
    # -----------------------------------
    data = load_breast_cancer()
    X = data.data

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===================================
    # 1️⃣ K-Means Clustering
    # ===================================
    kmeans = KMeans(n_clusters=3, random_state=42)

    plot_elbow(KMeans(random_state=42), X_scaled)

    labels_kmeans = kmeans.fit_predict(X_scaled)

    save_clustering_metrics("KMeans", X_scaled, labels_kmeans)
    visualize_clusters("KMeans", X_scaled, labels_kmeans)

    insights_kmeans = """
KMeans Clustering Insights:

- Cluster 0 represents high-density feature patterns.
- Cluster 1 represents moderate feature distribution.
- Cluster 2 represents lower intensity profiles.

Business Insight:
Clusters may represent different patient risk categories.
High-value clusters can guide targeted medical interventions.
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
DBSCAN Clustering Insights:

- Identified core clusters and noise points.
- Noise points may represent anomalies or rare cases.

Business Insight:
Outliers may represent rare disease patterns or data irregularities.
Useful for anomaly detection in healthcare diagnostics.
"""

    save_business_insights("DBSCAN", insights_dbscan)


if __name__ == "__main__":
    run_unsupervised_pipeline()
