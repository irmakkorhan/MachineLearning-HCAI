import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


MODEL_READY_DIR = "data/processed/model_ready"

TARGET_TYPE = "binary"
TARGET_COLUMN = "PHQ_Binary"
ID_COLUMN = "participant_id"

SEED = 42

# L2 strength grid (smaller C = stronger regularization) to fight the
# overfitting seen in 07 (perfect train, ~0.75 test ROC-AUC).
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# Expected column counts per family. If naming ever drifts (e.g. sbert_ becomes
# emb_), the residual "linguistic" bucket would silently swell — this guard
# turns that into a loud failure instead of nonsense results.
EXPECTED_COUNTS = {"linguistic": 17, "tfidf": 500, "sbert": 384}

# Selection metric: PR-AUC (average_precision) is threshold-free AND focused on
# the positive (depressed) class, so it is consistent with the F1 we report,
# unlike ROC-AUC. C is chosen by cross-validated PR-AUC on TRAIN only.
CV_SCORING = "average_precision"
N_FOLDS = 5


def load_split(split_name):
    """Load X / y for one split; drop participant_id so it is never a feature."""
    X = pd.read_csv(os.path.join(MODEL_READY_DIR, f"X_{split_name}.csv"))
    y = pd.read_csv(os.path.join(MODEL_READY_DIR, f"y_{split_name}_{TARGET_TYPE}.csv"))
    X = X.drop(columns=[ID_COLUMN])
    y = y[TARGET_COLUMN].astype(int).values
    return X, y


def split_columns(columns):
    """Return (linguistic, tfidf, sbert) column-name lists, with a count guard."""
    tfidf = [c for c in columns if c.startswith("tfidf_")]
    sbert = [c for c in columns if c.startswith("sbert_")]
    linguistic = [c for c in columns
                  if not c.startswith("tfidf_") and not c.startswith("sbert_")]

    detected = {"linguistic": len(linguistic), "tfidf": len(tfidf), "sbert": len(sbert)}
    if detected != EXPECTED_COUNTS:
        raise ValueError(
            "Feature-family column counts do not match expected naming.\n"
            f"  expected: {EXPECTED_COUNTS}\n"
            f"  detected: {detected}\n"
            "Check the sbert_/tfidf_ prefixes — unknown columns are being "
            "misfiled as linguistic."
        )
    return linguistic, tfidf, sbert


def build_pipeline(family, all_columns, C):
    """
    Scale-appropriate pipeline for one feature family (or combined).
    Rule (same as 07): StandardScaler on linguistic + S-BERT, TF-IDF passthrough.
    """
    linguistic, tfidf, sbert = split_columns(all_columns)

    if family == "linguistic":
        scale_cols, pass_cols = linguistic, []
    elif family == "tfidf":
        scale_cols, pass_cols = [], tfidf
    elif family == "sbert":
        scale_cols, pass_cols = sbert, []
    elif family == "combined":
        scale_cols, pass_cols = linguistic + sbert, tfidf
    else:
        raise ValueError(f"Unknown family: {family}")

    preprocess = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("passthrough", "passthrough", pass_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        C=C, class_weight="balanced", max_iter=5000, random_state=SEED,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def select_C(family, X_train, y_train, all_columns):
    """Pick C by cross-validated PR-AUC on TRAIN (stratified k-fold)."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best = None
    for C in C_GRID:
        model = build_pipeline(family, all_columns, C)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=CV_SCORING)
        mean, std = scores.mean(), scores.std()
        if best is None or mean > best["mean"]:
            best = {"C": C, "mean": mean, "std": std}
    return best


def tune_threshold(model, X_dev, y_dev):
    """Pick the probability threshold on DEV that maximizes positive-class F1."""
    proba = model.predict_proba(X_dev)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_dev, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def report(model, X, y, threshold, label):
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "split": label,
        "acc": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc": roc_auc_score(y, proba),
        "pr": average_precision_score(y, proba),
    }


def main():
    np.random.seed(SEED)

    X_train, y_train = load_split("train")
    X_dev, y_dev = load_split("dev")
    X_test, y_test = load_split("test")

    all_columns = list(X_train.columns)
    linguistic, tfidf, sbert = split_columns(all_columns)  # also runs the guard
    print(f"Feature families OK: linguistic={len(linguistic)}, "
          f"tfidf={len(tfidf)}, sbert={len(sbert)}")
    print(f"C selection: {N_FOLDS}-fold CV on train, scoring={CV_SCORING}")
    print(f"Threshold: tuned on dev for max F1\n")

    families = ["linguistic", "tfidf", "sbert", "combined"]
    rows = []

    for family in families:
        # 1. Select C by cross-validated PR-AUC on train (consistent with F1 goal).
        sel = select_C(family, X_train, y_train, all_columns)

        # 2. Refit the chosen C on the full train set.
        model = build_pipeline(family, all_columns, sel["C"]).fit(X_train, y_train)

        # 3. Tune the decision threshold on dev for F1 (not the naive 0.5).
        thr, dev_f1 = tune_threshold(model, X_dev, y_dev)

        # 4. Report test at that fixed threshold.
        test = report(model, X_test, y_test, thr, "test")

        rows.append({
            "family": family,
            "best_C": sel["C"],
            "cv_pr_mean": sel["mean"],
            "cv_pr_std": sel["std"],
            "dev_f1@thr": dev_f1,
            "threshold": thr,
            "test_f1": test["f1"],
            "test_pr": test["pr"],
            "test_roc": test["roc"],
            "test_acc": test["acc"],
        })
        print(f"{family:11s} | C={sel['C']:<6} "
              f"| CV PR-AUC={sel['mean']:.3f}±{sel['std']:.3f} "
              f"| thr={thr:.2f} dev_f1={dev_f1:.3f} "
              f"| test: f1={test['f1']:.3f} pr={test['pr']:.3f} "
              f"roc={test['roc']:.3f} acc={test['acc']:.3f}")

    summary = pd.DataFrame(rows)
    print("\n=== Ablation summary (sorted by test F1) ===")
    print(summary.sort_values("test_f1", ascending=False)
                 .round(3).to_string(index=False))

    best = summary.sort_values("test_f1", ascending=False).iloc[0]
    print(f"\nBest family by test F1: {best['family']} "
          f"(C={best['best_C']}, thr={best['threshold']:.2f}, "
          f"test F1={best['test_f1']:.3f}, test ROC-AUC={best['test_roc']:.3f})")
    print("\nDone. CV-tuned regularization + threshold + single-family ablation complete.")


if __name__ == "__main__":
    main()
