"""Streamlit front-end: evaluate the pre-trained shopper-intent classifiers on an uploaded test CSV."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
TARGET = "Revenue"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="ShopperSense – Purchase Intent Classifier", page_icon="🛒", layout="wide")


@st.cache_resource
def load_pipeline(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


def clean_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    """Bring raw UCI-format columns (TRUE/FALSE strings) into the numeric form the pipelines expect."""
    df = df.copy()
    for col in ("Weekend", TARGET):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.upper().map({"TRUE": 1, "FALSE": 0})
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def evaluate(pipe, X, y) -> tuple[dict, pd.Series, pd.Series]:
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return scores, y_pred, y_prob


# ---------------- Sidebar ----------------
st.sidebar.title("🛒 ShopperSense")
st.sidebar.caption("M.Tech AIML · ML Assignment 2")
chosen_model = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
show_all = st.sidebar.checkbox("Also compare all 5 models", value=True)
st.sidebar.markdown("---")
sample_path = APP_DIR / "test_data.csv"
if sample_path.exists():
    st.sidebar.download_button("⬇️ Download sample test_data.csv", sample_path.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
st.sidebar.info("Models were trained offline; uploading data here only **evaluates** them — no retraining.")

# ---------------- Main ----------------
st.title("Will this visitor buy? 🛍️")
st.write(
    "Predicting purchase intent from e-commerce browsing sessions "
    "(UCI *Online Shoppers Purchasing Intention* dataset). Upload a test CSV containing the 17 "
    f"session features and the `{TARGET}` column."
)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.warning("Upload a CSV to begin — you can grab the sample `test_data.csv` from the sidebar.")
    st.stop()

data = clean_uploaded(pd.read_csv(uploaded))
if TARGET not in data.columns:
    st.error(f"The uploaded file must contain the target column `{TARGET}`.")
    st.stop()

X_test, y_test = data.drop(columns=[TARGET]), data[TARGET]
expected = list(load_pipeline(MODEL_FILES[chosen_model]).feature_names_in_)
missing = [c for c in expected if c not in X_test.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()
X_test = X_test[expected]

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Features", len(expected))
c3.metric("Purchase rate", f"{y_test.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), width="stretch")

# ---- selected model
st.header(f"📊 {chosen_model}")
pipe = load_pipeline(MODEL_FILES[chosen_model])
scores, y_pred, y_prob = evaluate(pipe, X_test, y_test)

cols = st.columns(6)
for col, (name, val) in zip(cols, scores.items()):
    col.metric(name, f"{val:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBu", cbar=False, ax=ax,
                xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("ROC curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
    ax2.plot(fpr, tpr, color="#2b6cb0", lw=2, label=f"AUC = {scores['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], ls="--", color="grey")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

st.subheader("Classification report")
report = classification_report(y_test, y_pred, target_names=["No purchase", "Purchase"], output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---- all models
if show_all:
    st.header("🏁 All models on this test data")
    table = []
    for name, fname in MODEL_FILES.items():
        s, _, _ = evaluate(load_pipeline(fname), X_test, y_test)
        table.append({"Model": name, **s})
    comp = pd.DataFrame(table).set_index("Model").round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6f6d5"), width="stretch")
    best = comp["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    comp[["Accuracy", "AUC", "F1", "MCC"]].plot(kind="bar", ax=ax3, rot=0, colormap="viridis")
    ax3.set_ylim(0, 1)
    ax3.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    st.pyplot(fig3)
