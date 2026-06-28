"""
main.py — Orchestrates the full CPU-only candidate ranking pipeline.

Usage:
    python main.py --input_dir ./input --output_dir ./output --models_dir ./models
"""

import argparse
import json
import os
import sys
import time
import csv
import numpy as np
from typing import Any, Dict, List, Tuple

# Ensure src/ is in the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils import log, log_memory, start_timer, stop_timer, record_timing, save_timings, ts
from loader import load_job_description, stream_candidates
from preprocessor import preprocess_candidate
from retriever import run_retrieval
from reranker import load_cross_encoder, rerank_candidates
from scorer import load_xgboost_model, build_feature_matrix, run_xgboost_scoring, get_feature_contributions
from explainer import generate_all_reasoning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only offline candidate ranking engine"
    )
    parser.add_argument("--input_dir", default="./input", help="Path to input folder")
    parser.add_argument("--output_dir", default="./output", help="Path to output folder")
    parser.add_argument("--models_dir", default="./models", help="Path to pre-computed models folder")
    parser.add_argument("--top_k", type=int, default=100, help="Final number of candidates to output")
    parser.add_argument("--retrieval_k", type=int, default=500, help="Candidates to retrieve (FAISS+BM25)")
    parser.add_argument("--rerank_k", type=int, default=200, help="Candidates to pass to XGBoost")
    parser.add_argument("--batch_size", type=int, default=32, help="Cross-encoder batch size")
    parser.add_argument("--verbose_preprocess", action="store_true", help="Log every candidate during preprocessing")
    return parser.parse_args()


def validate_dirs(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.input_dir):
        log("MAIN", f"Input directory not found: {args.input_dir}", level="ERROR")
        sys.exit(1)
    if not os.path.isdir(args.models_dir):
        log("MAIN", f"Models directory not found: {args.models_dir}", level="WARN")
    os.makedirs(args.output_dir, exist_ok=True)
    log("MAIN", f"Directories validated — input: {args.input_dir}, output: {args.output_dir}, models: {args.models_dir}")


def step1_load_inputs(args: argparse.Namespace) -> str:
    print(f"\n{'='*60}", flush=True)
    log("STEP 1", "INPUT LOADING — Reading job description")
    print(f"{'='*60}\n", flush=True)
    start_timer("step1_load_inputs")

    jd_text = load_job_description(args.input_dir)

    if len(jd_text.strip()) < 200:
        raise ValueError(
            f"Job description text is too short ({len(jd_text)} characters). "
            "Check that job_description.md is not empty and markdown stripping is not over-aggressive."
        )
    log("STEP 1", f"JD length: {len(jd_text)} characters, {len(jd_text.split())} words")

    stop_timer("step1_load_inputs", f"JD loaded ({len(jd_text.split())} words)")
    log_memory()
    return jd_text


def step2_preprocess_all(
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, str], Dict[str, Dict], Dict[str, Dict]]:
    """
    STEP 2: Stream and preprocess all candidates.
    Returns (candidate_ids_in_order, text_blobs, features_map, profiles_map).
    """
    print(f"\n{'='*60}", flush=True)
    log("STEP 2", "PREPROCESSING — Building text blobs and extracting features for all candidates")
    print(f"{'='*60}\n", flush=True)
    start_timer("step2_preprocess")

    all_ids: List[str] = []
    text_blobs: Dict[str, str] = {}
    features_map: Dict[str, Dict] = {}
    profiles_map: Dict[str, Dict] = {}

    for idx, candidate in enumerate(stream_candidates(args.input_dir)):
        cid = candidate.get("candidate_id") or candidate.get("id") or f"CAND_{idx:06d}"
        profiles_map[cid] = candidate  # Store raw profile for reasoning
        cid, feats, blob = preprocess_candidate(
            candidate, idx, verbose=args.verbose_preprocess
        )
        all_ids.append(cid)
        text_blobs[cid] = blob
        features_map[cid] = feats

        if not args.verbose_preprocess and idx % 10000 == 0 and idx > 0:
            log("STEP 2", f"Preprocessed {idx:,} candidates so far...")
            log_memory()

    stop_timer("step2_preprocess", f"{len(all_ids):,} candidates preprocessed")
    log_memory()
    return all_ids, text_blobs, features_map, profiles_map


def step3_retrieval(
    args: argparse.Namespace,
    jd_text: str,
) -> Tuple[List[int], Dict[int, float]]:
    print(f"\n{'='*60}", flush=True)
    log("STEP 3", "HYBRID RETRIEVAL — FAISS + BM25 (100k → Top-500)")
    print(f"{'='*60}\n", flush=True)

    top_indices, rough_scores, _embed_model = run_retrieval(
        models_dir=args.models_dir,
        jd_text=jd_text,
        top_k=args.retrieval_k,
    )

    rough_score_map = dict(zip(top_indices, rough_scores))
    return top_indices, rough_score_map


def step4_reranking(
    args: argparse.Namespace,
    jd_text: str,
    top_global_indices: List[int],
    all_ids: List[str],
    text_blobs: Dict[str, str],
) -> Dict[str, float]:
    print(f"\n{'='*60}", flush=True)
    log("STEP 4", "CROSS-ENCODER RERANKING — Deep re-ranking (Top-500 → Top-200)")
    print(f"{'='*60}\n", flush=True)

    retrieved_ids = [all_ids[i] for i in top_global_indices if i < len(all_ids)]
    retrieved_blobs = [text_blobs.get(cid, "") for cid in retrieved_ids]

    log("STEP 4", f"Loading Cross-Encoder model...")
    cross_encoder = load_cross_encoder(args.models_dir)

    ranked = rerank_candidates(
        cross_encoder=cross_encoder,
        jd_text=jd_text,
        candidate_ids=retrieved_ids,
        text_blobs=retrieved_blobs,
        top_k=args.rerank_k,
        batch_size=args.batch_size,
    )

    ce_scores = {cid: score for cid, score in ranked}
    log("STEP 4", f"Cross-Encoder selected top {len(ce_scores)} candidates")
    return ce_scores


def step5_xgboost(
    args: argparse.Namespace,
    ce_scores: Dict[str, float],
    features_map: Dict[str, Dict],
    rough_score_by_index: Dict[int, float],
    all_ids: List[str],
) -> List[Tuple[str, float, Dict]]:
    print(f"\n{'='*60}", flush=True)
    log("STEP 5", "DETERMINISTIC SCORING — Final ranking (Top-200 → Top-100)")
    print(f"{'='*60}\n", flush=True)

    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    rough_scores_by_id = {
        cid: rough_score_by_index.get(id_to_idx.get(cid, -1), 0.0)
        for cid in ce_scores
    }

    top_100 = run_xgboost_scoring(
        models_dir=args.models_dir,
        candidate_ids=list(ce_scores.keys()),
        features_map=features_map,
        cross_encoder_scores=ce_scores,
        rough_scores=rough_scores_by_id,
        top_k=args.top_k,
    )

    return top_100


def step6_explainability(
    args: argparse.Namespace,
    top_100: List[Tuple[str, float, Dict]],
    ce_scores: Dict[str, float],
    features_map: Dict[str, Dict],
    profiles_map: Dict[str, Dict],
) -> Dict[str, str]:
    print(f"\n{'='*60}", flush=True)
    log("STEP 6", "EXPLAINABILITY — Generating reasoning strings")
    print(f"{'='*60}\n", flush=True)

    try:
        import xgboost as xgb
        import pickle
        model_path = os.path.join(args.models_dir, "xgboost_model.pkl")
        features_path = os.path.join(args.models_dir, "feature_names.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)

        top_ids = [t[0] for t in top_100]
        rough_scores = {cid: 0.0 for cid in top_ids}
        X = build_feature_matrix(
            top_ids, features_map, ce_scores,
            rough_scores,
            feature_names,
        )
        contribs = get_feature_contributions(model, X, feature_names)
    except Exception as e:
        log("STEP 6", f"Could not compute SHAP contributions: {e} — using heuristic fallback", level="WARN")
        feature_names = []
        contribs = None

    reasoning_map = generate_all_reasoning(
        top_candidates=top_100,
        cross_encoder_scores=ce_scores,
        feature_names=feature_names,
        contributions=contribs,
        candidate_profiles=profiles_map,
    )
    return reasoning_map


def step7_write_outputs(
    args: argparse.Namespace,
    top_100: List[Tuple[str, float, Dict]],
    reasoning_map: Dict[str, str],
    ce_scores: Dict[str, float],
) -> None:
    print(f"\n{'='*60}", flush=True)
    log("STEP 7", "OUTPUT WRITING — submission.csv + ranking_log.json")
    print(f"{'='*60}\n", flush=True)
    start_timer("step7_write_outputs")

    submission_path = os.path.join(args.output_dir, "submission.csv")
    log_path = os.path.join(args.output_dir, "ranking_log.json")

    scores = [s for _, s, _ in top_100]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:
            log("STEP 7", f"Score monotonicity violation at rank {i + 1} — fixing", level="WARN")
            scores[i] = scores[i - 1]

    with open(submission_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, ((cid, score, _feats), adj_score) in enumerate(zip(top_100, scores), start=1):
            reasoning = reasoning_map.get(cid, "No reasoning available.")
            writer.writerow([cid, rank, f"{adj_score:.6f}", reasoning])

    log("STEP 7", f"submission.csv written — {len(top_100)} rows at {submission_path}")

    ranking_log = []
    for rank, (cid, score, feats) in enumerate(top_100, start=1):
        ranking_log.append({
            "rank": rank,
            "candidate_id": cid,
            "composite_score": score,
            "cross_encoder_score": ce_scores.get(cid, 0.0),
            "features": {k: round(v, 6) for k, v in feats.items()},
            "reasoning": reasoning_map.get(cid, ""),
        })

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(ranking_log, f, indent=2)

    log("STEP 7", f"ranking_log.json written — {len(ranking_log)} entries at {log_path}")
    stop_timer("step7_write_outputs")


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"[{ts()}] CANDIDATE RANKING ENGINE — CPU-ONLY OFFLINE MODE", flush=True)
    print(f"[{ts()}] Input: {args.input_dir} | Output: {args.output_dir} | Models: {args.models_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    pipeline_start = time.perf_counter()
    validate_dirs(args)
    log_memory()

    jd_text = step1_load_inputs(args)
    all_ids, text_blobs, features_map, profiles_map = step2_preprocess_all(args)
    top_global_indices, rough_score_by_index = step3_retrieval(args, jd_text)
    ce_scores = step4_reranking(args, jd_text, top_global_indices, all_ids, text_blobs)
    top_100 = step5_xgboost(args, ce_scores, features_map, rough_score_by_index, all_ids)
    reasoning_map = step6_explainability(args, top_100, ce_scores, features_map, profiles_map)
    step7_write_outputs(args, top_100, reasoning_map, ce_scores)

    total_elapsed = time.perf_counter() - pipeline_start
    record_timing("pipeline_total", total_elapsed)
    save_timings(args.output_dir)

    print(f"\n{'='*60}", flush=True)
    log("MAIN", f"Pipeline complete in {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")
    log("MAIN", f"Results written to {args.output_dir}/")
    print(f"{'='*60}\n", flush=True)
    log_memory()


if __name__ == "__main__":
    main()
