"""
precompute.py — Pre-computation script that generates all artefacts in /models/.

Run this ONCE before the ranking pipeline:
    python precompute.py --input_dir ./input --models_dir ./models

This script:
  1. Streams candidates.jsonl and generates BGE embeddings (768d) in batches
  2. Builds and saves the FAISS index
  3. Builds and pickles the BM25 index
  4. Trains and saves an XGBoost ranker using a synthetic rubric
  5. Saves the Cross-Encoder locally via save_pretrained()
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import time
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils import log, log_memory, ts, record_timing, save_timings, progress_bar
from preprocessor import build_text_blob, extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute all ranking model artefacts")
    parser.add_argument("--input_dir", default="./input")
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--embed_model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--cross_encoder_name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--max_candidates", type=int, default=0, help="0 = process all")
    return parser.parse_args()


def stream_jsonl(input_dir: str):
    """Stream candidates JSONL or JSONL.GZ line by line."""
    gz = os.path.join(input_dir, "candidates.jsonl.gz")
    plain = os.path.join(input_dir, "candidates.jsonl")
    if os.path.exists(gz):
        path, opener, mode = gz, gzip.open, "rt"
    elif os.path.exists(plain):
        path, opener, mode = plain, open, "r"
    else:
        raise FileNotFoundError(f"No candidates file in {input_dir}")

    log("PRECOMPUTE", f"Streaming from {path}...")
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def count_candidates(input_dir: str, max_candidates: int) -> int:
    """Count candidate rows so progress and ETA are based on the real input size."""
    gz = os.path.join(input_dir, "candidates.jsonl.gz")
    plain = os.path.join(input_dir, "candidates.jsonl")
    if os.path.exists(gz):
        path, opener, mode = gz, gzip.open, "rt"
    elif os.path.exists(plain):
        path, opener, mode = plain, open, "r"
    else:
        raise FileNotFoundError(f"No candidates file in {input_dir}")

    log("PRECOMPUTE", f"Counting candidates in {path} for progress ETA...")
    t0 = time.perf_counter()
    total = 0
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
                if max_candidates > 0 and total >= max_candidates:
                    break
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"Found {total:,} candidate rows in {elapsed:.2f}s")
    return total


def format_duration(seconds: float) -> str:
    """Format seconds as a compact ETA string."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def build_embeddings_and_corpus(
    input_dir: str,
    models_dir: str,
    embed_model,
    batch_size: int,
    max_candidates: int,
) -> tuple:
    """
    Stream candidates, build text blobs, extract features, generate embeddings in batches.
    Returns (all_ids, all_embeddings, all_features, all_text_blobs).
    """
    log("PRECOMPUTE", "Starting candidate streaming and embedding generation...")
    log_memory()

    all_ids: List[str] = []
    all_blobs: List[str] = []
    all_features: List[Dict] = []

    batch_texts: List[str] = []
    batch_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []

    batch_num = 0
    total_processed = 0
    total_candidates = count_candidates(input_dir, max_candidates)
    embed_start = time.perf_counter()

    for idx, candidate in enumerate(stream_jsonl(input_dir)):
        if max_candidates > 0 and idx >= max_candidates:
            break

        cid = candidate.get("candidate_id") or candidate.get("id") or f"CAND_{idx:06d}"
        blob = build_text_blob(candidate)
        feats = extract_features(candidate, blob)

        all_ids.append(cid)
        all_blobs.append(blob)
        all_features.append(feats)

        batch_ids.append(cid)
        batch_texts.append(blob[:512])
        total_processed += 1

        if len(batch_texts) >= batch_size:
            batch_num += 1
            t0 = time.perf_counter()
            vecs = embed_model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            elapsed = time.perf_counter() - t0

            all_embeddings.append(vecs)
            bar = progress_bar(total_processed, total_candidates)
            elapsed_total = time.perf_counter() - embed_start
            rows_per_sec = total_processed / elapsed_total if elapsed_total > 0 else 0.0
            remaining = total_candidates - total_processed
            eta = format_duration(remaining / rows_per_sec) if rows_per_sec > 0 else "unknown"
            print(
                f"[{ts()}] [PRECOMPUTE] Batch {batch_num} completed in {elapsed:.3f}s | {bar} | "
                f"{rows_per_sec:.1f} rows/s | ETA {eta}",
                flush=True,
            )
            batch_texts = []
            batch_ids = []

            if batch_num % 20 == 0:
                log_memory()

    if batch_texts:
        batch_num += 1
        t0 = time.perf_counter()
        vecs = embed_model.encode(
            batch_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        elapsed = time.perf_counter() - t0
        all_embeddings.append(vecs)
        print(
            f"[{ts()}] [PRECOMPUTE] Final batch {batch_num} completed in {elapsed:.3f}s | "
            f"{progress_bar(total_processed, total_candidates)} | ETA 0s",
            flush=True,
        )

    embeddings_matrix = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 768), dtype=np.float32)
    log("PRECOMPUTE", f"Embedding generation complete — {len(all_ids):,} candidates, matrix shape {embeddings_matrix.shape}")
    log_memory()
    return all_ids, embeddings_matrix, all_features, all_blobs


def build_and_save_faiss(embeddings: np.ndarray, models_dir: str) -> None:
    """Build FAISS flat IP index and save to disk."""
    import faiss

    log("PRECOMPUTE", f"Building FAISS index (dim={embeddings.shape[1]}, n={embeddings.shape[0]:,})...")
    t0 = time.perf_counter()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"FAISS index built in {elapsed:.3f}s — {index.ntotal:,} vectors")

    index_path = os.path.join(models_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    size_mb = os.path.getsize(index_path) / (1024 ** 2)
    log("PRECOMPUTE", f"FAISS index saved to {index_path} ({size_mb:.1f} MB)")

    emb_path = os.path.join(models_dir, "candidate_embeddings.npy")
    np.save(emb_path, embeddings)
    log("PRECOMPUTE", f"Embeddings saved to {emb_path}")


def build_and_save_bm25(text_blobs: List[str], models_dir: str) -> None:
    """Build BM25 index over all candidate text blobs and pickle it."""
    from rank_bm25 import BM25Okapi

    log("PRECOMPUTE", f"Tokenizing {len(text_blobs):,} candidate blobs for BM25...")
    t0 = time.perf_counter()
    tokenized = [blob.lower().split() for blob in text_blobs]
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"Tokenization done in {elapsed:.2f}s")

    log("PRECOMPUTE", "Building BM25Okapi index...")
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"BM25 index built in {elapsed:.2f}s")

    index_path = os.path.join(models_dir, "bm25_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(index_path) / (1024 ** 2)
    log("PRECOMPUTE", f"BM25 index saved to {index_path} ({size_mb:.1f} MB)")


def _synthetic_rubric_score(features: Dict) -> float:
    """
    Generate a synthetic ground-truth relevance score using a domain rubric.
    Used to train the XGBoost model when no real labels exist.
    """
    score = 0.0
    exp = min(features.get("total_experience_years", 0.0), 20.0)
    score += exp * 0.06

    score += features.get("expert_skill_count", 0.0) * 0.08
    score -= features.get("expert_skill_zero_years_count", 0.0) * 0.25
    score -= features.get("company_age_anomaly", 0.0) * 0.4
    score += features.get("has_github_link", 0.0) * 0.12
    score += features.get("response_rate", 0.5) * 0.1
    notice = features.get("notice_period_days", 30.0)
    score -= max(0, (notice - 30) / 90) * 0.1
    score += features.get("is_pune_noida_bangalore", 0.0) * 0.05
    score += features.get("production_evidence", 0.0) * 0.15
    score += features.get("profile_completeness", 0.5) * 0.08
    score += features.get("education_level", 0.0) * 0.03
    score -= features.get("honeypot_flag_count", 0.0) * 0.3

    score = np.clip(score, 0.0, 1.0)
    return float(score)


def train_and_save_xgboost(
    features_list: List[Dict],
    models_dir: str,
) -> None:
    """Train XGBoost on synthetic rubric scores and save model + feature names."""
    import xgboost as xgb

    ce_placeholder = np.full(len(features_list), 0.5, dtype=np.float32)
    rr_placeholder = np.full(len(features_list), 0.5, dtype=np.float32)

    feature_names = list(features_list[0].keys()) + ["cross_encoder_score", "rough_retrieval_score"]

    log("PRECOMPUTE", f"Building feature matrix ({len(features_list):,} x {len(feature_names)})...")
    X_rows = []
    y_labels = []
    for i, feats in enumerate(features_list):
        enriched = dict(feats)
        enriched["cross_encoder_score"] = float(ce_placeholder[i])
        enriched["rough_retrieval_score"] = float(rr_placeholder[i])
        row = [enriched.get(fn, 0.0) for fn in feature_names]
        X_rows.append(row)
        y_labels.append(_synthetic_rubric_score(feats))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)
    log("PRECOMPUTE", f"Feature matrix shape: {X.shape} | Label range: [{y.min():.3f}, {y.max():.3f}]")

    log("PRECOMPUTE", "Training XGBoost regressor...")
    t0 = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        device="cpu",
        random_state=42,
    )
    model.fit(X, y, verbose=False)
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"XGBoost training complete in {elapsed:.2f}s")

    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)
    log("PRECOMPUTE", f"XGBoost model saved to {model_path}")
    log("PRECOMPUTE", f"Feature names saved to {features_path}: {feature_names}")


def save_cross_encoder(cross_encoder_name: str, models_dir: str) -> None:
    """Download and save Cross-Encoder locally for offline use."""
    from sentence_transformers import CrossEncoder

    save_path = os.path.join(models_dir, "cross_encoder")
    log("PRECOMPUTE", f"Downloading Cross-Encoder '{cross_encoder_name}'...")
    t0 = time.perf_counter()
    model = CrossEncoder(cross_encoder_name, device="cpu")
    model.save(save_path)
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"Cross-Encoder saved to {save_path} in {elapsed:.2f}s")


def save_embedding_model(embed_model_name: str, models_dir: str) -> Optional[Any]:
    """Download and save embedding model locally for offline use."""
    from sentence_transformers import SentenceTransformer

    save_path = os.path.join(models_dir, os.path.basename(embed_model_name))
    if os.path.isdir(save_path):
        log("PRECOMPUTE", f"Embedding model already saved at {save_path}, skipping download")
        return None
    log("PRECOMPUTE", f"Downloading embedding model '{embed_model_name}'...")
    t0 = time.perf_counter()
    model = SentenceTransformer(embed_model_name, device="cpu")
    model.save(save_path)
    elapsed = time.perf_counter() - t0
    log("PRECOMPUTE", f"Embedding model saved to {save_path} in {elapsed:.2f}s")
    return model


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    pipeline_start = time.perf_counter()

    print(f"\n{'='*60}", flush=True)
    print(f"[{ts()}] PRECOMPUTE — Generating all model artefacts", flush=True)
    print(f"[{ts()}] Input: {args.input_dir} | Models: {args.models_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    log("PRECOMPUTE", "STEP 0: Downloading and saving embedding model for offline use...")
    embed_model = save_embedding_model(args.embed_model_name, args.models_dir)
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        bge_path = os.path.join(args.models_dir, os.path.basename(args.embed_model_name))
        embed_model = SentenceTransformer(bge_path, device="cpu")

    log("PRECOMPUTE", "STEP 1: Saving Cross-Encoder for offline use...")
    ce_path = os.path.join(args.models_dir, "cross_encoder")
    if not os.path.isdir(ce_path):
        save_cross_encoder(args.cross_encoder_name, args.models_dir)
    else:
        log("PRECOMPUTE", f"Cross-Encoder already at {ce_path}, skipping")

    log("PRECOMPUTE", "STEP 2: Streaming candidates and generating embeddings...")
    all_ids, embeddings, all_features, all_blobs = build_embeddings_and_corpus(
        args.input_dir,
        args.models_dir,
        embed_model,
        batch_size=args.embed_batch_size,
        max_candidates=args.max_candidates,
    )

    log("PRECOMPUTE", "STEP 3: Building and saving FAISS index...")
    build_and_save_faiss(embeddings, args.models_dir)
    log_memory()

    log("PRECOMPUTE", "STEP 4: Building and saving BM25 index...")
    build_and_save_bm25(all_blobs, args.models_dir)
    log_memory()

    log("PRECOMPUTE", "STEP 5: Training and saving XGBoost ranker...")
    train_and_save_xgboost(all_features, args.models_dir)
    log_memory()

    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n{'='*60}", flush=True)
    log("PRECOMPUTE", f"All artefacts generated in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log("PRECOMPUTE", f"Models directory: {os.path.abspath(args.models_dir)}")
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
