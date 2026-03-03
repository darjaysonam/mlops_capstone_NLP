import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

REPORT_DIR = "ml_model/reports/eda"
os.makedirs(REPORT_DIR, exist_ok=True)


class EDAAnalyzer:

    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column

    # ----------------------------------------
    # 1. Statistical Summary
    # ----------------------------------------
    def statistical_summary(self):
        numeric_df = self.df.select_dtypes(include=np.number)

        summary = pd.DataFrame(
            {
                "mean": numeric_df.mean(),
                "median": numeric_df.median(),
                "std": numeric_df.std(),
                "variance": numeric_df.var(),
                "min": numeric_df.min(),
                "max": numeric_df.max(),
                "skewness": numeric_df.apply(skew),
                "kurtosis": numeric_df.apply(kurtosis),
            }
        )

        summary.to_csv(f"{REPORT_DIR}/summary_statistics.csv")
        return summary

    # ----------------------------------------
    # 2. Missing Values
    # ----------------------------------------
    def analyze_missing(self):
        missing = self.df.isnull().sum().to_frame("missing_count")
        missing["percentage"] = missing["missing_count"] / len(self.df) * 100
        missing.to_csv(f"{REPORT_DIR}/missing_values.csv")
        return missing

    # ----------------------------------------
    # 3. Correlation Matrix
    # ----------------------------------------
    def correlation_analysis(self):
        numeric_df = self.df.select_dtypes(include=np.number)

        corr = numeric_df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/correlation_matrix.png")
        plt.close()

        return corr

    # ----------------------------------------
    # 4. Feature Distributions
    # ----------------------------------------
    def feature_distributions(self):
        numeric_df = self.df.select_dtypes(include=np.number)

        numeric_df.hist(figsize=(14, 12), bins=20)
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/feature_distributions.png")
        plt.close()

    # ----------------------------------------
    # 5. Boxplots (Outliers Detection)
    # ----------------------------------------
    def boxplot_analysis(self):
        numeric_df = self.df.select_dtypes(include=np.number)

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=numeric_df)
        plt.xticks(rotation=90)
        plt.title("Boxplot for Outlier Detection")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/boxplots.png")
        plt.close()

    # ----------------------------------------
    # 6. Class Imbalance
    # ----------------------------------------
    def class_distribution(self):
        if self.target_column not in self.df.columns:
            return

        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.df[self.target_column])
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/class_distribution.png")
        plt.close()
