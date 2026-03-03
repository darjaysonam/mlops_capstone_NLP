import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ===============================
# PATH CONFIGURATION
# ===============================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target"  # change if needed


# ===============================
# LOAD DATA
# ===============================

def load_data():
    if not DATA_PATH.exists():
        print(f"\n❌ Dataset NOT found at:\n{DATA_PATH}")
        print("\nPlace your dataset at:")
        print(f"{BASE_DIR / 'data'}")
        print("And name it: dataset.csv")
        sys.exit()

    print("\n✅ Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    return df


# ===============================
# BASIC STATISTICS
# ===============================

def statistical_summary(df):
    print("\nGenerating statistical summary...")

    summary = df.describe(include="all").transpose()
    summary["skewness"] = df.select_dtypes(include=np.number).skew()
    summary["kurtosis"] = df.select_dtypes(include=np.number).kurt()

    summary.to_csv(REPORT_DIR / "statistical_summary.csv")


# ===============================
# DATA QUALITY CHECKS
# ===============================

def data_quality_report(df):
    print("Checking data quality...")

    report = pd.DataFrame({
        "missing_values": df.isnull().sum(),
        "missing_%": (df.isnull().sum() / len(df)) * 100,
        "unique_values": df.nunique(),
        "dtype": df.dtypes
    })

    report.to_csv(REPORT_DIR / "data_quality_report.csv")

    duplicates = df.duplicated().sum()

    with open(REPORT_DIR / "data_quality_notes.txt", "w") as f:
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Duplicate rows: {duplicates}\n")


# ===============================
# FEATURE DISTRIBUTIONS
# ===============================

def numeric_distributions(df):
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        print("No numeric columns found. Skipping numeric distribution plots.")
        return

    for col in numeric_df.columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{col}_distribution.png")
        plt.close()


# ===============================
# CORRELATION MATRIX
# ===============================

def correlation_matrix(df):
    print("Generating correlation matrix...")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_matrix.png")
    plt.close()


# ===============================
# TEXT ANALYSIS (NLP Support)
# ===============================

def text_analysis(df):
    text_cols = df.select_dtypes(include="object").columns

    if len(text_cols) == 0:
        print("No text columns detected.")
        return

    print("Performing text analysis...")

    for col in text_cols:
        text_lengths = df[col].astype(str).apply(len)

        plt.figure()
        sns.histplot(text_lengths, bins=30)
        plt.title(f"Text Length Distribution - {col}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{col}_text_length.png")
        plt.close()


# ===============================
# CLASS IMBALANCE
# ===============================

def class_distribution(df):
    if TARGET_COLUMN not in df.columns:
        print(f"Target column '{TARGET_COLUMN}' not found.")
        return

    print("Generating class distribution...")

    plt.figure()
    df[TARGET_COLUMN].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png")
    plt.close()


# ===============================
# FEATURE IMPORTANCE (Mutual Info)
# ===============================

def feature_importance(df):
    if TARGET_COLUMN not in df.columns:
        return

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        print("No numeric features for feature importance.")
        return

    X = numeric_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    if X.empty:
        return

    y = df[TARGET_COLUMN]

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    mi = mutual_info_classif(X.fillna(0), y)
    importance = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    importance.to_csv(REPORT_DIR / "feature_importance.csv")

    plt.figure()
    importance.head(15).plot(kind="barh")
    plt.title("Top Feature Importance (Mutual Info)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png")
    plt.close()


# ===============================
# MAIN
# ===============================

def main():
    print("\n========== STARTING EDA ==========")
    print("Dataset path:", DATA_PATH)
    print("Target column:", TARGET_COLUMN)

    df = load_data()

    statistical_summary(df)
    data_quality_report(df)
    numeric_distributions(df)
    correlation_matrix(df)
    text_analysis(df)
    class_distribution(df)
    feature_importance(df)

    print("\n✅ EDA Completed Successfully.")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()