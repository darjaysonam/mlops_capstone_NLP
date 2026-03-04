import pandas as pd
import os

from eda_analysis import EDAAnalyzer
from feature_selection import FeatureSelector
from preprocessing import DataPreprocessor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "smoking.csv")

TARGET_COLUMN = "smoking"


def main():

    print("\n===== Starting EDA Pipeline =====")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # --------------------------------
    # Preprocessing
    # --------------------------------

    preprocessor = DataPreprocessor(df, TARGET_COLUMN)

    df = preprocessor.remove_duplicates()

    df = preprocessor.impute_missing()

    df = preprocessor.encode_categorical()

    # --------------------------------
    # EDA Analysis
    # --------------------------------

    eda = EDAAnalyzer(df, TARGET_COLUMN)

    eda.statistical_summary()

    eda.analyze_missing()

    eda.correlation_analysis()

    eda.feature_distributions()

    eda.boxplot_analysis()

    eda.class_distribution()

    # --------------------------------
    # Feature Selection
    # --------------------------------

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    selector = FeatureSelector(X, y)

    selector.mutual_information()

    selector.tree_importance()

    print("\nEDA completed successfully.")

    print("\nReports saved in: ml_model/eda/reports")


if __name__ == "__main__":
    main()
