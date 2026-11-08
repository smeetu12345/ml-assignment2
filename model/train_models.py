"""
Train 5 classifiers on the UCI Online Shoppers Purchasing Intention dataset.

Run from the project root:
    python model/train_models.py

Outputs (written to model/):
    - <model_name>.joblib   : full sklearn Pipeline (preprocessing + classifier)
    - metrics_summary.csv   : evaluation metrics of every model on the hold-out test set
Also writes test_data.csv (raw hold-out rows incl. target) to the project root.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
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
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Revenue"
SEED = 26

CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
BOOLEAN = ["Weekend"]
NUMERIC = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]


def load_shopper_sessions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Weekend"] = df["Weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("bool", "passthrough", BOOLEAN),
        ]
    )


def candidate_classifiers() -> dict:
    # class_weight="balanced" is used where supported because only ~15% of sessions end in a purchase
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        # larger var_smoothing keeps the near-constant one-hot columns from dominating the likelihood
        "Naive Bayes": GaussianNB(var_smoothing=0.1),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                      class_weight="balanced", random_state=SEED),
    }


def score_model(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def file_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main():
    sessions = load_shopper_sessions(RAW_CSV)
    print(f"Dataset after de-duplication: {sessions.shape[0]} rows, {sessions.shape[1] - 1} features")
    print("Class balance:\n", sessions[TARGET].value_counts(normalize=True).round(3))

    features = sessions.drop(columns=[TARGET])
    labels = sessions[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    # held-out rows (raw form + target) — this is what gets uploaded to the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, clf in candidate_classifiers().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        result = {"Model": name, **score_model(y_test, y_pred, y_prob)}
        rows.append(result)
        joblib.dump(pipe, MODEL_DIR / f"{file_slug(name)}.joblib")
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in result.items() if k != "Model"))

    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
