"""
reranker.py — Cross-Encoder deep re-ranking (Top-500 → Top-200).
"""

import os
import time
import numpy as np
from typing import List, Tuple

from utils import log, log_memory, record_timing, start_timer, stop_timer, ts, get_file_size_mb


def load_cross_encoder(models_dir: str):
    """
    Load the Cross-Encoder from local disk. No network calls.
    Falls back to a smaller model if the primary is missing.
    """
    from sentence_transformers import CrossEncoder

    primary_path = os.path.join(models_dir, "cross_encoder")
    fallback_path = os.path.join(models_dir, "cross_encoder_fallback")

    if os.path.isdir(primary_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(primary_path)
            for f in fn
        ) / (1024 ** 2)
        log("RERANKER", f"Loading Cross-Encoder from {primary_path} (~{size_mb:.1f} MB)...")
        t0 = time.perf_counter()
        model = CrossEncoder(primary_path, device="cpu", max_length=512)
        elapsed = time.perf_counter() - t0
        record_timing("load_cross_encoder", elapsed)
        log("RERANKER", f"Cross-Encoder loaded in {elapsed:.3f}s")
        return model
    elif os.path.isdir(fallback_path):
        print(
            f"\n[{ts()}] [WARN] [RERANKER] Primary cross-encoder not found. "
            f"Using fallback from {fallback_path}.\n",
            flush=True,
        )
        t0 = time.perf_counter()
        model = CrossEncoder(fallback_path, device="cpu", max_length=512)
        elapsed = time.perf_counter() - t0
        record_timing("load_cross_encoder", elapsed)
        log("RERANKER", f"Fallback Cross-Encoder loaded in {elapsed:.3f}s")
        return model
    else:
        print(
            f"\n[{ts()}] [WARN] [RERANKER] No local cross-encoder found in {models_dir}. "
            "Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 as fallback.\n",
            flush=True,
        )
        t0 = time.perf_counter()
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
        elapsed = time.perf_counter() - t0
        record_timing("load_cross_encoder", elapsed)
        log("RERANKER", f"Downloaded fallback Cross-Encoder in {elapsed:.3f}s")
        return model


def rerank_candidates(
    cross_encoder,
    jd_text: str,
    candidate_ids: List[str],
    text_blobs: List[str],
    top_k: int = 200,
    batch_size: int = 32,
) -> List[Tuple[str, float]]:
    """
    Score all candidates with the Cross-Encoder in batches.
    Returns list of (candidate_id, relevance_score) sorted descending, top_k only.
    """
    start_timer("reranker_total")
    n = len(candidate_ids)
    log("RERANKER", f"Starting Cross-Encoder scoring of {n} candidates in batches of {batch_size}...")
    log_memory()

    pairs = [(jd_text[:1000], blob[:1000]) for blob in text_blobs]
    all_scores = np.zeros(n, dtype=np.float32)
    num_batches = (n + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_pairs = pairs[start:end]

        t0 = time.perf_counter()
        raw_scores = cross_encoder.predict(batch_pairs, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        record_timing(f"cross_encoder_batch_{batch_idx + 1}", elapsed)

        all_scores[start:end] = raw_scores
        print(
            f"[{ts()}] [RERANKER] Batch {batch_idx + 1}/{num_batches} completed in {elapsed:.4f}s "
            f"({end - start} pairs)",
            flush=True,
        )

    relevance_scores = _normalize_scores(all_scores)

    top_indices = np.argsort(relevance_scores)[::-1][:top_k]
    results = [
        (candidate_ids[i], float(relevance_scores[i]))
        for i in top_indices
    ]

    stop_timer("reranker_total", f"top {len(results)} candidates selected")
    log("RERANKER", f"Score stats — min: {relevance_scores.min():.4f}, max: {relevance_scores.max():.4f}, "
        f"mean: {relevance_scores.mean():.4f}")
    log_memory()
    return results


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize cross-encoder logits into a relative [0, 1] rerank score."""
    scores = np.asarray(scores, dtype=np.float32)
    mn = float(np.nanmin(scores))
    mx = float(np.nanmax(scores))
    if np.isclose(mx, mn):
        return np.ones_like(scores, dtype=np.float32)
    return np.clip((scores - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)
