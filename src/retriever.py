"""
retriever.py — FAISS + BM25 hybrid recall (100k → Top-500).
"""

import os
import pickle
import time
import numpy as np
from typing import Dict, List, Tuple, Any

from utils import log, log_memory, start_timer, stop_timer, record_timing, ts, get_file_size_mb


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def load_embedding_model(models_dir: str):
    """
    Load the BGE embedding model from local disk (no network calls).
    Falls back to all-MiniLM-L6-v2 if BGE is not present.
    """
    from sentence_transformers import SentenceTransformer

    bge_path = os.path.join(models_dir, "bge-base-en-v1.5")
    minilm_path = os.path.join(models_dir, "all-MiniLM-L6-v2")

    if os.path.isdir(bge_path):
        log("RETRIEVER", f"Loading BGE embedding model from {bge_path}...")
        t0 = time.perf_counter()
        model = SentenceTransformer(bge_path, device="cpu")
        elapsed = time.perf_counter() - t0
        record_timing("load_embedding_model", elapsed)
        log("RETRIEVER", f"BGE model loaded in {elapsed:.3f}s")
        return model
    elif os.path.isdir(minilm_path):
        print(
            f"\n[{ts()}] [WARN] [RETRIEVER] BGE model not found at {bge_path}. "
            "Falling back to all-MiniLM-L6-v2 — QUALITY WILL BE REDUCED.\n",
            flush=True,
        )
        t0 = time.perf_counter()
        model = SentenceTransformer(minilm_path, device="cpu")
        elapsed = time.perf_counter() - t0
        record_timing("load_embedding_model", elapsed)
        log("RETRIEVER", f"Fallback model loaded in {elapsed:.3f}s")
        return model
    else:
        print(
            f"\n[{ts()}] [WARN] [RETRIEVER] No local embedding model found in {models_dir}. "
            "Downloading all-MiniLM-L6-v2 as last-resort fallback.\n",
            flush=True,
        )
        t0 = time.perf_counter()
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        elapsed = time.perf_counter() - t0
        record_timing("load_embedding_model", elapsed)
        log("RETRIEVER", f"Downloaded fallback model in {elapsed:.3f}s")
        return model


def embed_job_description(model, jd_text: str) -> np.ndarray:
    """Run one forward pass on the JD text and return a (1, D) float32 array."""
    log("RETRIEVER", "Embedding job description via BGE model...")
    t0 = time.perf_counter()
    vec = model.encode(
        [jd_text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    record_timing("embed_jd", elapsed)
    log("RETRIEVER", f"JD embedding done in {elapsed:.4f}s — shape {vec.shape}")
    return vec.astype(np.float32)


def faiss_search(
    models_dir: str,
    jd_vec: np.ndarray,
    top_k: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FAISS index from disk (with memory mapping) and search top_k candidates.
    Returns (indices, similarities).
    """
    import faiss

    index_path = os.path.join(models_dir, "faiss_index.bin")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"[RETRIEVER] faiss_index.bin not found at {index_path}. "
            "Run precompute.py first."
        )

    size_mb = get_file_size_mb(index_path)
    log("RETRIEVER", f"Loading FAISS index from {index_path} (size: {size_mb:.1f} MB)...")
    t0 = time.perf_counter()
    index = faiss.read_index(index_path)
    elapsed = time.perf_counter() - t0
    record_timing("load_faiss", elapsed)
    log("RETRIEVER", f"FAISS index loaded in {elapsed:.3f}s — {index.ntotal:,} vectors, dim={index.d}")
    log_memory()

    log("RETRIEVER", f"Running FAISS dense search for top {top_k}...")
    t0 = time.perf_counter()
    similarities, indices = index.search(jd_vec, top_k)
    elapsed = time.perf_counter() - t0
    record_timing("faiss_search", elapsed)
    log("RETRIEVER", f"FAISS search done in {elapsed:.4f}s — retrieved {len(indices[0])} candidates")

    return indices[0], similarities[0]


def bm25_search(
    models_dir: str,
    jd_text: str,
    top_k: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load BM25 index and retrieve top_k candidates by sparse matching.
    Returns (indices, bm25_scores).
    """
    index_path = os.path.join(models_dir, "bm25_index.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"[RETRIEVER] bm25_index.pkl not found at {index_path}. "
            "Run precompute.py first."
        )

    size_mb = get_file_size_mb(index_path)
    log("RETRIEVER", f"Loading BM25 index from {index_path} (size: {size_mb:.1f} MB)...")
    t0 = time.perf_counter()
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    elapsed = time.perf_counter() - t0
    record_timing("load_bm25", elapsed)
    log("RETRIEVER", f"BM25 index loaded in {elapsed:.3f}s")
    log_memory()

    log("RETRIEVER", "Tokenizing JD and computing BM25 scores across all candidates...")
    tokens = jd_text.lower().split()
    t0 = time.perf_counter()
    scores = np.array(bm25.get_scores(tokens), dtype=np.float32)
    elapsed = time.perf_counter() - t0
    record_timing("bm25_score", elapsed)
    log("RETRIEVER", f"BM25 scoring done in {elapsed:.4f}s — scored {len(scores):,} candidates")

    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def hybrid_merge(
    dense_indices: np.ndarray,
    dense_sims: np.ndarray,
    sparse_indices: np.ndarray,
    sparse_scores: np.ndarray,
    top_k: int = 500,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Tuple[int, float]]:
    """
    Merge FAISS and BM25 results into a single ranked list.
    Returns list of (global_index, rough_score) sorted descending.
    """
    log("RETRIEVER", "Merging dense and sparse results into hybrid ranked list...")
    t0 = time.perf_counter()

    norm_dense = _normalize(dense_sims)
    norm_sparse = _normalize(sparse_scores)

    score_map: Dict[int, float] = {}

    for rank, (idx, sim) in enumerate(zip(dense_indices, norm_dense)):
        if idx < 0:
            continue
        score_map[int(idx)] = dense_weight * float(sim)

    for rank, (idx, bm) in enumerate(zip(sparse_indices, norm_sparse)):
        if idx < 0:
            continue
        existing = score_map.get(int(idx), 0.0)
        score_map[int(idx)] = existing + sparse_weight * float(bm)

    merged = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
    elapsed = time.perf_counter() - t0
    record_timing("hybrid_merge", elapsed)

    unique_total = len(score_map)
    log(
        "RETRIEVER",
        f"Merge done in {elapsed:.4f}s — {unique_total} unique candidates, "
        f"reduction ratio {100_000 / max(unique_total, 1):.1f}x → keeping top {len(merged)}",
    )
    return merged


def run_retrieval(
    models_dir: str,
    jd_text: str,
    top_k: int = 500,
) -> Tuple[List[int], List[float], Any]:
    """
    Full hybrid retrieval pipeline. Returns (top_indices, rough_scores, embedding_model).
    """
    start_timer("retrieval_total")

    embed_model = load_embedding_model(models_dir)
    jd_vec = embed_job_description(embed_model, jd_text)

    dense_idx, dense_sim = faiss_search(models_dir, jd_vec, top_k=top_k)
    sparse_idx, sparse_scores = bm25_search(models_dir, jd_text, top_k=top_k)

    merged = hybrid_merge(dense_idx, dense_sim, sparse_idx, sparse_scores, top_k=top_k)

    stop_timer("retrieval_total", f"top {len(merged)} candidates selected")
    log_memory()

    indices = [item[0] for item in merged]
    scores = [item[1] for item in merged]
    return indices, scores, embed_model
