import pandas as pd
import os


FEATURES_FILE = "data/processed/set3_combined_features.csv"

TRAIN_LABELS_FILE = "data/raw/labels/train_split_Depression_AVEC2017.csv"
DEV_LABELS_FILE = "data/raw/labels/dev_split_Depression_AVEC2017.csv"
TEST_LABELS_FILE = "data/raw/labels/full_test_split.csv"

OUTPUT_DIR = "data/processed/model_ready"


def standardize_label_columns(df):
    """
    Make column names consistent across train/dev/test files.
    Some files use PHQ8_Binary, some use PHQ_Binary.
    Some files use PHQ8_Score, some use PHQ_Score.
    """

    df = df.copy()

    if "Participant_ID" in df.columns:
        df = df.rename(columns={"Participant_ID": "participant_id"})

    if "participant_ID" in df.columns:
        df = df.rename(columns={"participant_ID": "participant_id"})

    if "PHQ8_Binary" in df.columns:
        df = df.rename(columns={"PHQ8_Binary": "PHQ_Binary"})

    if "PHQ8_Score" in df.columns:
        df = df.rename(columns={"PHQ8_Score": "PHQ_Score"})

    df["participant_id"] = df["participant_id"].astype(str)

    return df


def merge_features_with_labels(features, labels, split_name):
    """
    Merge Set 3 features with labels by participant_id.
    """

    merged = pd.merge(
        labels,
        features,
        on="participant_id",
        how="inner"
    )

    print(f"\n{split_name} merged shape:", merged.shape)

    expected_n = labels.shape[0]
    actual_n = merged.shape[0]

    if expected_n != actual_n:
        print(f"WARNING: {split_name} expected {expected_n} participants, but merged {actual_n}.")
        missing = set(labels["participant_id"]) - set(merged["participant_id"])
        print("Missing participant IDs:", sorted(list(missing))[:20])

    return merged


def save_X_y(merged, split_name, target_type="binary"):
    """
    Save X and y separately.

    target_type='binary' uses PHQ_Binary.
    target_type='regression' uses PHQ_Score.
    """

    if target_type == "binary":
        target_column = "PHQ_Binary"
    elif target_type == "regression":
        target_column = "PHQ_Score"
    else:
        raise ValueError("target_type must be either 'binary' or 'regression'.")

    # y = target
    y = merged[["participant_id", target_column]].copy()

    # X = features only
    columns_to_remove = [
        "PHQ_Binary",
        "PHQ_Score",
        "Gender",
        "PHQ8_NoInterest",
        "PHQ8_Depressed",
        "PHQ8_Sleep",
        "PHQ8_Tired",
        "PHQ8_Appetite",
        "PHQ8_Failure",
        "PHQ8_Concentrating",
        "PHQ8_Moving",
    ]

    X = merged.drop(
        columns=[col for col in columns_to_remove if col in merged.columns]
    )

    X_path = os.path.join(OUTPUT_DIR, f"X_{split_name}.csv")
    y_path = os.path.join(OUTPUT_DIR, f"y_{split_name}_{target_type}.csv")

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

    print(f"Saved {X_path}")
    print(f"Saved {y_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Set 3 features
    features = pd.read_csv(FEATURES_FILE)
    features["participant_id"] = features["participant_id"].astype(str)

    print("Features shape:", features.shape)

    # Load label files
    train_labels = standardize_label_columns(pd.read_csv(TRAIN_LABELS_FILE))
    dev_labels = standardize_label_columns(pd.read_csv(DEV_LABELS_FILE))
    test_labels = standardize_label_columns(pd.read_csv(TEST_LABELS_FILE))

    print("Train labels shape:", train_labels.shape)
    print("Dev labels shape:", dev_labels.shape)
    print("Test labels shape:", test_labels.shape)

    # Merge features with labels
    train_data = merge_features_with_labels(features, train_labels, "train")
    dev_data = merge_features_with_labels(features, dev_labels, "dev")
    test_data = merge_features_with_labels(features, test_labels, "test")

    # Save binary classification X/y files
    save_X_y(train_data, "train", target_type="binary")
    save_X_y(dev_data, "dev", target_type="binary")
    save_X_y(test_data, "test", target_type="binary")

    # Optional: also save regression X/y files
    save_X_y(train_data, "train", target_type="regression")
    save_X_y(dev_data, "dev", target_type="regression")
    save_X_y(test_data, "test", target_type="regression")

    print("\nDone. Official train/dev/test splits are ready.")


if __name__ == "__main__":
    main()