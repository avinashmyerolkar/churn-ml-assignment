"""
Telco Customer Churn - model evaluation app (ML Assignment 2).
Upload the hold-out test CSV, pick one of the five saved models, and see how it
performs. Models are loaded from model/*.joblib and are NOT retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "Churn"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
HIGHLIGHT = "#C8553D"
BASE = "#1F5F8B"

st.set_page_config(page_title="Churn model evaluator", page_icon="📉", layout="wide")


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "test_data.csv")


def required_features(model):
    return list(model.named_steps["prep"].feature_names_in_)


def clean_input(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def encode_target(series):
    return series.astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def score(model, X, y):
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score) if y.nunique() > 1 else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_score


def plot_confusion(y, y_pred):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stays", "Churns"], yticklabels=["Stays", "Churns"],
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc(y, y_score, label):
    fpr, tpr, _ = roc_curve(y, y_score)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, color=BASE, lw=2, label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv",
                                help="Must contain the 19 customer features and the 'Churn' column.")
    st.download_button("Download sample test_data.csv",
                       data=load_sample().to_csv(index=False).encode(),
                       file_name="test_data.csv", mime="text/csv")
    st.caption("Models were trained offline on 80% of the Telco Customer Churn "
               "dataset. This app only loads them and predicts.")

# ---------- header ----------
st.title("Will this customer leave?")
st.write("Evaluate five pre-trained classifiers that predict telecom customer churn. "
         "Upload the held-out test file, choose a model, and compare results.")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to start. You can download the sample "
            "file there if you don't have one.")
    st.stop()

# ---------- data checks ----------
raw = pd.read_csv(uploaded)
data = clean_input(raw)
model = load_model(MODEL_FILES[model_name])
features = required_features(model)

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {', '.join(missing)}. "
             "Add them and upload again.")
    st.stop()

y = encode_target(data[TARGET])
if y.isna().any():
    st.error("The 'Churn' column must contain only Yes/No (or 1/0) values.")
    st.stop()
y = y.astype(int)
X = data[features]

c1, c2, c3 = st.columns(3)
c1.metric("Customers in file", f"{len(data):,}")
c2.metric("Actual churners", f"{int(y.sum()):,}")
c3.metric("Churn rate", f"{y.mean():.1%}")

with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(20), use_container_width=True)

tab_one, tab_all = st.tabs([model_name, "Compare all models"])

# ---------- single model ----------
with tab_one:
    metrics, y_pred, y_score = score(model, X, y)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion(y, y_pred))
    with right:
        st.subheader("ROC curve")
        st.pyplot(plot_roc(y, y_score, f"AUC = {metrics['AUC']:.3f}"))

    st.subheader("Classification report")
    report = classification_report(y, y_pred, target_names=["Stays (0)", "Churns (1)"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    out = raw.copy()
    out["Predicted_Churn"] = np.where(y_pred == 1, "Yes", "No")
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
                       mime="text/csv")

# ---------- all models ----------
with tab_all:
    rows = []
    for name, fname in MODEL_FILES.items():
        m, _, _ = score(load_model(fname), X, y)
        rows.append({"Model": name, **m})
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#CFE3F1"),
                 use_container_width=True)

    metric_pick = st.radio("Chart metric", list(table.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ordered = table[metric_pick].sort_values()
    colors = [HIGHLIGHT if n == ordered.idxmax() else BASE for n in ordered.index]
    ax.barh(ordered.index, ordered.values, color=colors)
    for i, v in enumerate(ordered.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(1.0, ordered.max() + 0.1))
    ax.set_xlabel(metric_pick)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)