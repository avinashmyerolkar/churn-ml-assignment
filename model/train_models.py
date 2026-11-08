"""
Train 5 classifiers on the Telco Customer Churn dataset and save them.
Run from the project root:  python model/train_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Churn"
SEED = 5068        # last digits of my student ID -> my own train/test split
TEST_SIZE = 0.20

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor(feature_cols):
    categorical_cols = [c for c in feature_cols if c not in NUMERIC_COLS]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, categorical_cols),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=SEED),
    }


def positive_scores(model, X):
    """Probability of churn if available, else the SVM decision score (for AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    print(f"scikit-learn version: {sklearn.__version__}")
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Dataset shape: {df.shape} | features: {X.shape[1]} | churn rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    rows = []
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X.columns)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        scores = evaluate(y_test, pipe.predict(X_test), positive_scores(pipe, X_test))
        rows.append({"Model": name, **scores})
        joblib.dump(pipe, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    test_df = X_test.copy()
    test_df[TARGET] = y_test.map({1: "Yes", 0: "No"})
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    print("\n" + metrics.to_string(index=False))
    print(f"\nSaved 5 models to model/ and test_data.csv ({len(test_df)} rows)")


if __name__ == "__main__":
    main()