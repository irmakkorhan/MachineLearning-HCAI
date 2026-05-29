import os
import sys
import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

INPUT_FILE  = "/Users/atillaarslan/Downloads/data/processed/participant_rows_clean.csv"
OUTPUT_FILE = "/Users/atillaarslan/Downloads/data/processed/set_sbert_features.csv"

# "all-mpnet-base-v2" (768-dim) is higher quality but ~3× slower
MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64
SEED = 42

# L2-normalise so cosine similarity == dot product; benefits linear classifiers and kNN downstream
NORMALIZE = True

EXPECTED_PARTICIPANTS = 189


# ── Helpers ────────────────────────────────────────────────────────────────────

def install_if_missing():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("sentence-transformers not found – installing …")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "sentence-transformers"]
        )


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 0. Install dependency if needed
    install_if_missing()

    import torch
    from sentence_transformers import SentenceTransformer

    device = get_device()

    print("=" * 60)
    print("05_extract_sbert_features.py")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Device     : {device}")
    print(f"  Normalize  : {NORMALIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Input      : {INPUT_FILE}")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 60)

    # 1. Load input file
    if not os.path.exists(INPUT_FILE):
        print(f"[FAIL] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"\n[PASS] Loaded input file.")
    print(f"  Shape: {df.shape}")
    print(df.head(3))

    # 2. Clean columns
    df["participant_id"] = df["participant_id"].astype(str)
    df["value_clean"]    = df["value_clean"].fillna("").astype(str)

    mask_valid = df["value_clean"].str.strip() != ""
    n_dropped  = (~mask_valid).sum()
    df_valid   = df[mask_valid].copy().reset_index(drop=True)

    all_participants = df["participant_id"].unique().tolist()
    print(f"\n[Step 2] {len(df_valid)} usable utterance rows "
          f"({n_dropped} empty/NaN rows skipped).")

    # 3. Report per-participant utterance counts
    valid_counts = df_valid.groupby("participant_id").size()

    print(f"\n[Step 3] Utterances per participant:")
    print(f"  min  : {valid_counts.min()}")
    print(f"  max  : {valid_counts.max()}")
    print(f"  mean : {valid_counts.mean():.1f}")

    zero_utt = [p for p in all_participants if p not in valid_counts.index]
    if zero_utt:
        print(f"  [WARNING] {len(zero_utt)} participant(s) with 0 usable utterances: {zero_utt}")
    else:
        print("  All participants have at least 1 usable utterance.")

    # 4. Load model
    print(f"\n[Step 4] Loading model '{MODEL_NAME}' on {device} …")
    torch.manual_seed(SEED)
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.eval()
    print(f"  Model loaded. Max sequence length: {model.max_seq_length}")

    # 5. Embed every utterance in batches (progress bar via tqdm inside encode)
    sentences = df_valid["value_clean"].tolist()
    print(f"\n[Step 5] Embedding {len(sentences)} utterances in batches of {BATCH_SIZE} …")

    with torch.no_grad():
        embeddings = model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    d = embeddings.shape[1]
    print(f"[Step 5] Done. Embedding matrix: {embeddings.shape}  (d = {d})")

    # 6. Mean-pool utterance embeddings per participant
    print(f"\n[Step 6] Mean-pooling per participant …")
    df_valid["_idx"] = np.arange(len(df_valid))

    participant_vectors = {}
    for pid, group in df_valid.groupby("participant_id"):
        idx = group["_idx"].values
        participant_vectors[pid] = embeddings[idx].mean(axis=0)

    # Participants with zero usable utterances get a zero vector rather than being dropped
    zero_vec = np.zeros(d, dtype=np.float32)
    for pid in zero_utt:
        participant_vectors[pid] = zero_vec.copy()
        print(f"  [WARNING] Zero vector assigned to participant {pid} (no usable utterances).")

    print(f"  Pooled vectors for {len(participant_vectors)} participants.")

    # 7. Build output DataFrame
    print(f"\n[Step 7] Building output DataFrame …")
    emb_cols = [f"sbert_{i}" for i in range(d)]

    rows = []
    for pid in sorted(participant_vectors.keys()):
        rows.append([pid] + participant_vectors[pid].tolist())

    out_df = pd.DataFrame(rows, columns=["participant_id"] + emb_cols)
    print(f"  Output shape: {out_df.shape}")
    print(out_df.iloc[:3, :6])  # first 3 rows, first 6 columns for readability

    # 8. Sanity checks
    print("\n── Sanity checks ──────────────────────────────────────────────")

    n_out_rows     = len(out_df)
    n_unique_parts = out_df["participant_id"].nunique()
    if n_out_rows == n_unique_parts:
        print(f"[PASS] Output rows ({n_out_rows}) == unique participants ({n_unique_parts}).")
    else:
        print(f"[FAIL] Output rows ({n_out_rows}) != unique participants ({n_unique_parts})!")

    if n_out_rows == EXPECTED_PARTICIPANTS:
        print(f"[PASS] Participant count matches expected ({EXPECTED_PARTICIPANTS}).")
    else:
        print(f"[WARNING] Participant count {n_out_rows} != expected {EXPECTED_PARTICIPANTS}. "
              f"Reconcile with set1/set2.")

    actual_sbert_cols = [c for c in out_df.columns if c.startswith("sbert_")]
    if len(actual_sbert_cols) == d:
        print(f"[PASS] Embedding dimensionality d = {d} confirmed across all output columns.")
    else:
        print(f"[FAIL] Expected {d} sbert columns, found {len(actual_sbert_cols)}!")

    emb_matrix = out_df[emb_cols].values
    has_nan = np.isnan(emb_matrix).any()
    has_inf = np.isinf(emb_matrix).any()
    if not has_nan and not has_inf:
        print("[PASS] No NaN or Inf values in embedding matrix.")
    else:
        if has_nan:
            print("[FAIL] NaN values detected in embedding matrix!")
        if has_inf:
            print("[FAIL] Inf values detected in embedding matrix!")

    print("── End of sanity checks ───────────────────────────────────────")

    # 9. Save output
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Step 9] Saved to: {OUTPUT_FILE}")

    # Summary
    print(f"\nDone: {n_out_rows} participants × {d}-dim SBERT embeddings → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
