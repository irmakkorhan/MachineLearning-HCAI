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
# overfitting seen before.
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# Expected column counts per family. If naming ever changes
# turns that into a loud failure instead of nonsense results
EXPECTED_COUNTS = {"linguistic": 17, "tfidf": 500, "sbert": 384}

# Selection metric: PR-AUC (average_precision) is threshold-free AND focused on
# the positive (depressed) class, so it is consistent with the F1 we report,
# unlike ROC-AUC. C is chosen by cross-validated PR-AUC on TRAIN only
CV_SCORING = "average_precision"
N_FOLDS = 5


RESULTS_FILE = "data/processed/threshold_cv_results.csv"


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
     StandardScaler on linguistic + S-BERT, TF-IDF passthrough.
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


def cv_threshold(family, X_train, y_train, all_columns, C):
    """
    Pick the decision threshold by cross-validation on TRAIN.

    Uses the SAME 5-fold StratifiedKFold as select_C. For each fold: fit the
    pipeline with the chosen C on the fold's training portion only, predict
    probabilities on the fold's held-out portion, and take the threshold in
    np.linspace(0.05, 0.95, 91) that maximizes positive-class F1 on that held-out
    portion. The threshold is therefore never fit on data the model trained on
    within that fold. Returns the 5 per-fold thresholds and their mean.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X = X_train.reset_index(drop=True)
    y = np.asarray(y_train)
    grid = np.linspace(0.05, 0.95, 91)

    fold_thresholds = []
    for train_idx, holdout_idx in cv.split(X, y):
        model = build_pipeline(family, all_columns, C)
        model.fit(X.iloc[train_idx], y[train_idx])

        proba = model.predict_proba(X.iloc[holdout_idx])[:, 1]
        y_holdout = y[holdout_idx]

        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_holdout, (proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        fold_thresholds.append(best_t)

    fold_thresholds = np.array(fold_thresholds)
    return fold_thresholds, fold_thresholds.mean()


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

    # Dev is intentionally NOT loaded: in this version the threshold is chosen by
    # cross-validation on train, so dev plays no role in selection.
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    all_columns = list(X_train.columns)
    linguistic, tfidf, sbert = split_columns(all_columns)  # also runs the guard
    print(f"Feature families OK: linguistic={len(linguistic)}, "
          f"tfidf={len(tfidf)}, sbert={len(sbert)}")
    print(f"C selection: {N_FOLDS}-fold CV on train, scoring={CV_SCORING}")
    print(f"Threshold: {N_FOLDS}-fold CV on train (mean of per-fold F1-optimal "
          f"thresholds)\n")

    families = ["linguistic", "tfidf", "sbert", "combined"]
    rows = []

    for family in families:
        # Select C by cross-validated PR-AUC on train
        sel = select_C(family, X_train, y_train, all_columns)

        # Refit the chosen C on the full train set
        model = build_pipeline(family, all_columns, sel["C"]).fit(X_train, y_train)

        # Choose the threshold by CV on train (per-fold held-out F1) mean them.
        fold_thrs, thr = cv_threshold(family, X_train, y_train, all_columns, sel["C"])
        print(f"{family:11s} | per-fold thresholds: "
              f"{np.round(fold_thrs, 2).tolist()} -> mean={thr:.3f}")

        # Report test at the CV-mean threshold
        test = report(model, X_test, y_test, thr, "test")

        rows.append({
            "family": family,
            "best_C": sel["C"],
            "cv_pr_mean": sel["mean"],
            "cv_pr_std": sel["std"],
            "threshold": thr,
            "thr_cv_std": fold_thrs.std(),
            "test_f1": test["f1"],
            "test_pr": test["pr"],
            "test_roc": test["roc"],
            "test_acc": test["acc"],
            "method": "cv_train",
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved summary to: {RESULTS_FILE}")

    # Sort by cv_pr_mean 
    print("\n=== Threshold-by-CV summary (sorted by cv_pr_mean) ===")
    print(summary.sort_values("cv_pr_mean", ascending=False)
                 .round(3).to_string(index=False))

    best = summary.sort_values("cv_pr_mean", ascending=False).iloc[0]
    print(f"\nBest family by CV PR-AUC: {best['family']} "
          f"(C={best['best_C']}, thr={best['threshold']:.2f}, "
          f"test F1={best['test_f1']:.3f}, test ROC-AUC={best['test_roc']:.3f})")
    print("\nDone. CV-on-train threshold + single-family ablation complete.")


if __name__ == "__main__":
    main()
