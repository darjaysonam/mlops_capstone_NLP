import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


class FeatureSelector:

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    # ----------------------------------------
    # Mutual Information
    # ----------------------------------------
    def mutual_information(self):

        mi = mutual_info_classif(self.X, self.y)

        mi_df = pd.Series(mi, index=self.X.columns).sort_values(ascending=False)

        mi_df.to_csv(os.path.join(REPORT_DIR, "mutual_information_scores.csv"))

        return mi_df

    # ----------------------------------------
    # Random Forest Feature Importance
    # ----------------------------------------
    def tree_importance(self):

        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

        model.fit(self.X, self.y)

        importances = pd.Series(
            model.feature_importances_, index=self.X.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))

        importances.head(15).plot(kind="barh")

        plt.title("Top Feature Importance (Random Forest)")

        plt.tight_layout()

        plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"))

        plt.close()

        return importances