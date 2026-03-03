import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, learning_curve

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):

    print(f"\n===== Evaluating {model_name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities (for ROC-AUC)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    # ------------------------
    # Metrics
    # ------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            pass

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "cv_mean": cv_mean,
    }

    print(metrics_dict)

    # ------------------------
    # Save Metrics JSON
    # ------------------------
    with open(os.path.join(REPORTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # ------------------------
    # Save Classification Report
    # ------------------------
    report = classification_report(y_test, y_pred)
    with open(
        os.path.join(REPORTS_DIR, f"{model_name}_classification_report.txt"), "w"
    ) as f:
        f.write(report)

    # ------------------------
    # Confusion Matrix
    # ------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # ------------------------
    # Learning Curve
    # ------------------------
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1
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
