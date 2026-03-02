import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

REPORT_DIR = "ml_model/reports/eda"
os.makedirs(REPORT_DIR, exist_ok=True)


class FeatureSelector:

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    # ----------------------------------------
    # 1. Mutual Information
    # ----------------------------------------
    def mutual_information(self):
        mi = mutual_info_classif(self.X, self.y)

        mi_df = pd.Series(mi, index=self.X.columns)
        mi_df.sort_values(ascending=False, inplace=True)

        return mi_df

    # ----------------------------------------
    # 2. Tree-Based Feature Importance
    # ----------------------------------------
    def tree_importance(self):
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X, self.y)

        importances = pd.Series(
            model.feature_importances_,
            index=self.X.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        importances.head(15).plot(kind="barh")
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/feature_importance.png")
        plt.close()

        return importances