import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


MODEL_READY_DIR = "data/processed/model_ready"

# Binary depression label (PHQ_Binary). Swap to the _regression files for PHQ_Score.
TARGET_TYPE = "binary"
TARGET_COLUMN = "PHQ_Binary"

ID_COLUMN = "participant_id"

# Fixed so results are reproducible.
SEED = 42


def load_split(split_name):
    """Load X / y for one split and return them row-aligned (X already matches y)."""
    X = pd.read_csv(os.path.join(MODEL_READY_DIR, f"X_{split_name}.csv"))
    y = pd.read_csv(os.path.join(MODEL_READY_DIR, f"y_{split_name}_{TARGET_TYPE}.csv"))

    # 1. Drop participant_id so it is NEVER scaled or used as a feature.
    #    (It is numeric, so a scaler/model would otherwise treat it as signal.)
    X = X.drop(columns=[ID_COLUMN])

    y = y[TARGET_COLUMN].astype(int).values
    return X, y


def build_pipeline(feature_columns):
    """
    StandardScaler on linguistic + S-BERT columns, TF-IDF passed through untouched.

    - Linguistic features live on wildly different scales (word_count in the
      thousands vs. rates in [0, 1]) -> they need standardizing.
    - S-BERT dimensions are L2-normalized per vector but each dimension still
      benefits from being centered/scaled across participants -> standardize.
    - TF-IDF is already L2-normalized by the vectorizer, so we leave it as-is
      ("passthrough") rather than re-scaling sparse, mostly-zero columns.
    """
    tfidf_cols = [c for c in feature_columns if c.startswith("tfidf_")]
    sbert_cols = [c for c in feature_columns if c.startswith("sbert_")]
    linguistic_cols = [
        c for c in feature_columns
        if not c.startswith("tfidf_") and not c.startswith("sbert_")
    ]

    print(f"  linguistic cols (scaled):  {len(linguistic_cols)}")
    print(f"  sbert cols      (scaled):  {len(sbert_cols)}")
    print(f"  tfidf cols  (passthrough): {len(tfidf_cols)}")

    preprocess = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), linguistic_cols + sbert_cols),
            ("tfidf_passthrough", "passthrough", tfidf_cols),
        ],
        remainder="drop",
    )

    # class_weight="balanced" handles the ~28% positive-class imbalance so the
    # model does not collapse to predicting "not depressed" for everyone.
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
    )

    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def evaluate(model, X, y, split_name):
    """Print a metric block appropriate for an imbalanced binary task."""
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    print(f"\n── {split_name} ──────────────────────────────")
    print(f"  accuracy : {accuracy_score(y, y_pred):.3f}")
    print(f"  F1 (pos) : {f1_score(y, y_pred):.3f}")
    print(f"  ROC-AUC  : {roc_auc_score(y, y_score):.3f}")
    print(f"  PR-AUC   : {average_precision_score(y, y_score):.3f}")
    print("  confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(confusion_matrix(y, y_pred))
    print("  classification report:")
    print(classification_report(y, y_pred, digits=3))


def main():
    np.random.seed(SEED)

    # 1. Load splits (participant_id already dropped inside load_split).
    X_train, y_train = load_split("train")
    X_dev, y_dev = load_split("dev")
    X_test, y_test = load_split("test")

    print("X_train:", X_train.shape, "| positives:", int(y_train.sum()))
    print("X_dev:  ", X_dev.shape, "| positives:", int(y_dev.sum()))
    print("X_test: ", X_test.shape, "| positives:", int(y_test.sum()))

    # 2. Build the scaling + classifier pipeline.
    print("\nFeature blocks:")
    model = build_pipeline(list(X_train.columns))

    # 3. Fit on TRAIN ONLY. The scaler's mean/std are learned here and reused
    #    for dev/test via transform inside the pipeline -> no leakage.
    model.fit(X_train, y_train)
    print("\nModel fitted on train (scaler statistics learned from train only).")

    # 4. Evaluate. Use dev for model selection; report test once at the end.
    evaluate(model, X_train, y_train, "TRAIN (sanity)")
    evaluate(model, X_dev, y_dev, "DEV (model selection)")
    evaluate(model, X_test, y_test, "TEST (final)")

    print("\nDone. Baseline LogisticRegression (balanced) trained and evaluated.")


if __name__ == "__main__":
    main()
