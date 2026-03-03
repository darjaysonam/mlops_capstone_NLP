from evaluation import evaluate_model
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def run_supervised_pipeline():

    # -----------------------------------
    # Example dataset (replace with yours)
    # -----------------------------------
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------
    # 1. Linear Model
    # -----------------------------------
    logistic_model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )

    evaluate_model(
        logistic_model, X_train, X_test, y_train, y_test, "Logistic_Regression"
    )

    # -----------------------------------
    # 2. Tree-Based Model
    # -----------------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random_Forest")

    # -----------------------------------
    # 3. Support Vector Machine
    # -----------------------------------
    svm_model = Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]
    )

    evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM")

    # -----------------------------------
    # 4. Ensemble (Stacking)
    # -----------------------------------
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=50)),
        ("gb", GradientBoostingClassifier()),
    ]

    stacking_model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )

    evaluate_model(
        stacking_model, X_train, X_test, y_train, y_test, "Stacking_Ensemble"
    )


if __name__ == "__main__":
    run_supervised_pipeline()
