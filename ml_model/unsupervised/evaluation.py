import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# --------------------------------------------
# Save clustering metrics
# --------------------------------------------
def save_clustering_metrics(model_name, X, labels):

    metrics = {}

    if len(set(labels)) > 1:
        metrics["silhouette_score"] = silhouette_score(X, labels)
        metrics["davies_bouldin_index"] = davies_bouldin_score(X, labels)
    else:
        metrics["silhouette_score"] = None
        metrics["davies_bouldin_index"] = None

    with open(os.path.join(REPORTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{model_name} metrics:", metrics)

    return metrics


# --------------------------------------------
# Elbow Method
# --------------------------------------------
def plot_elbow(kmeans_model, X, max_k=10):

    inertias = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans_model.set_params(n_clusters=k)
        kmeans_model.fit(X)
        inertias.append(kmeans_model.inertia_)

    plt.figure()
    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "elbow_method.png"))
    plt.close()


# --------------------------------------------
# PCA Visualization
# --------------------------------------------
def visualize_clusters(model_name, X, labels):

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels, palette="Set2", legend="full"
    )
    plt.title(f"{model_name} - PCA Cluster Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name}_pca_clusters.png"))
    plt.close()


# --------------------------------------------
# Save Business Insights
# --------------------------------------------
def save_business_insights(model_name, insights_text):

    with open(
        os.path.join(REPORTS_DIR, f"{model_name}_business_insights.txt"), "w"
    ) as f:
        f.write(insights_text)
