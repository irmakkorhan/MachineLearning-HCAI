# S-BERT Feature Set — Validation Summary

## Setup
- **Model:** all-MiniLM-L6-v2 (384-dim sentence embeddings)
- **Method:** embed each utterance → mean-pool per participant (L2-normalized)
- **Output:** `set_sbert_features.csv` — 189 participants × 384 features

## Sanity checks — all clear
- Shape 189 × 385 → matches set1 / set2 row counts
- 189 unique IDs, 0 duplicates, 0 missing vectors
- No NaN / inf; zero-vector fallback never triggered

## Statistical shape — looks "flat" but is healthy
- Mean pairwise cosine ≈ 0.90, small per-dim variance
- Cause: BERT **anisotropy** + **mean-pooling** over long interviews
- Not a collapse: PCA needs **62 components for 90%** of variance → signal is spread across dimensions, not lost

## Action items
- **Standardize** S-BERT + linguistic features before modeling (fit on train only)
- Leave TF-IDF unscaled (already L2-normalized)
- Minor: cast `participant_id` to string for consistency

## Conclusion
Extraction is correct and pipeline-ready. No re-run needed — only downstream scaling remains.
