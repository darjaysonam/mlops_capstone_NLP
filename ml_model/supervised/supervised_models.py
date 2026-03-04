import os
import pandas as pd

from evaluation import evaluate_model

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def run_supervised_pipeline():

    print("\nLoading dataset...")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "smoking.csv")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    df["oral"] = df["oral"].map({"Y": 1, "N": 0})
    df["tartar"] = df["tartar"].map({"Y": 1, "N": 0})

    df = df.fillna(df.mean())

    # -----------------------------
    # Features and Target
    # -----------------------------
    X = df.drop(columns=["smoking"])
    y = df["smoking"]

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ==================================================
    # 1️⃣ Logistic Regression (Linear Model)
    # ==================================================
    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    evaluate_model(
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Logistic_Regression",
    )

    # ==================================================
    # 2️⃣ Random Forest (Tree Model)
    # ==================================================
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    evaluate_model(
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Random_Forest",
    )

    # ==================================================
    # 3️⃣ Support Vector Machine (Fast Linear SVM)
    # ==================================================
    svm_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=5000)),
        ]
    )

    evaluate_model(
        svm_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Linear_SVM",
    )

    # ==================================================
    # 4️⃣ Ensemble Model (Stacking)
    # ==================================================
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ("gb", GradientBoostingClassifier()),
    ]

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        n_jobs=-1,
    )

    evaluate_model(
        stacking_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Stacking_Ensemble",
    )


if __name__ == "__main__":
    run_supervised_pipeline()
