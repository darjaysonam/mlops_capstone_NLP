import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import cross_val_score, learning_curve

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):

    print(f"\n===== Evaluating {model_name} =====")

    # -------------------------
    # Train model
    # -------------------------
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -------------------------
    # Metrics
    # -------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    roc_auc = None

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        pass

    # -------------------------
    # Cross Validation (faster)
    # -------------------------
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)

    cv_mean = cv_scores.mean()

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "cv_mean": float(cv_mean),
    }

    print(metrics_dict)

    # -------------------------
    # Save metrics
    # -------------------------
    with open(os.path.join(REPORTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # -------------------------
    # Classification report
    # -------------------------
    report = classification_report(y_test, y_pred)

    with open(
        os.path.join(REPORTS_DIR, f"{model_name}_classification_report.txt"), "w"
    ) as f:
        f.write(report)

    # -------------------------
    # Confusion matrix
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name}_confusion_matrix.png"))

    plt.close()

    # -------------------------
    # Learning Curve (optimized)
    # -------------------------
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=3,
        n_jobs=-1,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    )

    plt.figure()

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train Score")

    plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")

    plt.title(f"{model_name} Learning Curve")

    plt.xlabel("Training Samples")

    plt.ylabel("Score")

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name}_learning_curve.png"))

    plt.close()

    return metrics_dict
