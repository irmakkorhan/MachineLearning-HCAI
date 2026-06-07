import pandas as pd


SET1_FILE = "data/processed/set1_linguistic_features.csv"
SET2_FILE = "data/processed/set2_tfidf_features.csv"
SBERT_FILE = "data/processed/set3_sbert_features.csv"

OUTPUT_FILE = "data/processed/set3_combined_features.csv"

# All three sets describe the SAME 189 participants, just with different
# feature families. We expect a perfect 1-to-1 alignment on participant_id.
# An inner join is the safe choice here: if every set really does contain the
# same 189 participants, inner / outer / left all return the identical 189 rows.
# But if one set is silently missing a participant,
# inner join drops the unmatched rows instead of inventing NaNs. That makes the
# row-count assertion below catch the misalignment loudly, so we never save a
# combined matrix where some rows are padded with NaNs for a whole feature
# family which would quietly corrupt any model trained on it
EXPECTED_ROWS = 189
N_SBERT = 384


def main():
    # Load all three feature sets
    set1 = pd.read_csv(SET1_FILE)
    set2 = pd.read_csv(SET2_FILE)
    sbert = pd.read_csv(SBERT_FILE)

    print("Set 1 (linguistic) shape:", set1.shape)
    print("Set 2 (tfidf) shape:", set2.shape)
    print("S-BERT shape:", sbert.shape)

    # Cast participant_id to string in each set so the merge keys line up
    for df in (set1, set2, sbert):
        df["participant_id"] = df["participant_id"].astype(str)

    #  Merge pairwise on participant_id with how="inner"
    combined = pd.merge(set1, set2, on="participant_id", how="inner")
    combined = pd.merge(combined, sbert, on="participant_id", how="inner")

    # Assert the final row count is exactly 189 if not, report which
    #    participant_ids are missing from which set, then raise an error.
    if combined.shape[0] != EXPECTED_ROWS:
        ids1 = set(set1["participant_id"])
        ids2 = set(set2["participant_id"])
        ids3 = set(sbert["participant_id"])
        all_ids = ids1 | ids2 | ids3

        print("\nRow count mismatch! "
              f"Expected {EXPECTED_ROWS}, got {combined.shape[0]}.")
        print("Missing from Set 1 (linguistic):", sorted(all_ids - ids1))
        print("Missing from Set 2 (tfidf):     ", sorted(all_ids - ids2))
        print("Missing from S-BERT:            ", sorted(all_ids - ids3))
        raise ValueError(
            f"Combined row count {combined.shape[0]} != {EXPECTED_ROWS}; "
            "refusing to save a misaligned feature matrix."
        )

    # Assert no duplicate column names (other than participant_id, which is
    #    the shared merge key and is expected to appear once in the result).
    cols = [c for c in combined.columns if c != "participant_id"]
    duplicates = [c for c in set(cols) if cols.count(c) > 1]
    if duplicates:
        raise ValueError(f"Duplicate column names after merge: {duplicates}")

    # Print shapes and a per-family column breakdown.
    feature_cols = cols  # everything except participant_id
    n_tfidf = sum(1 for c in feature_cols if c.startswith("tfidf_"))
    n_sbert = sum(1 for c in feature_cols if c.startswith("sbert_"))
    n_linguistic = len(feature_cols) - n_tfidf - n_sbert

    # Sanity check that the S-BERT block is the expected 384 dims.
    assert n_sbert == N_SBERT, f"Expected {N_SBERT} sbert cols, got {n_sbert}"

    print("\nColumn family breakdown:")
    print(f"  linguistic (Set 1): {n_linguistic}")
    print(f"  tfidf_     (Set 2): {n_tfidf}")
    print(f"  sbert_     (S-BERT): {n_sbert}")
    print(f"  total features:     {len(feature_cols)} (+1 participant_id)")

    # Save the combined matrix.
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nSet 3 combined features created successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Combined shape:", combined.shape)


if __name__ == "__main__":
    main()
