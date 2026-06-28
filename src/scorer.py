"""
scorer.py — Deterministic recruiter-fit scoring with JD‑aligned weights.
Now heavily weights:
  - India location (visa)
  - Evidence of shipping ranking/search systems
  - Product‑company experience
  - Penalises big‑tech‑only and consulting‑only
"""

import os
import pickle
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from utils import log, log_memory, record_timing, start_timer, stop_timer, ts


# ---------------------------------------------------------------------------
# XGBoost model loading (optional)
# ---------------------------------------------------------------------------

def load_xgboost_model(models_dir: str):
    """Load XGBoost model and feature names from disk. Returns (model, feature_names)."""
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[SCORER] xgboost_model.pkl not found at {model_path}. "
            "Run precompute.py first."
        )

    log("SCORER", f"Loading XGBoost model from {model_path}...")
    t0 = time.perf_counter()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    elapsed = time.perf_counter() - t0
    record_timing("load_xgboost", elapsed)
    log("SCORER", f"XGBoost model loaded in {elapsed:.4f}s")

    feature_names: List[str] = []
    if os.path.exists(features_path):
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
        log("SCORER", f"Feature names loaded: {feature_names}")

    return model, feature_names


def build_feature_matrix(
    candidate_ids: List[str],
    features_map: Dict[str, Dict[str, float]],
    cross_encoder_scores: Dict[str, float],
    rough_scores: Dict[str, float],
    feature_names: List[str],
) -> np.ndarray:
    """Build (N × D) feature matrix for XGBoost inference."""
    rows = []
    for cid in candidate_ids:
        feats = dict(features_map.get(cid, {}))
        feats["cross_encoder_score"] = cross_encoder_scores.get(cid, 0.0)
        feats["rough_retrieval_score"] = rough_scores.get(cid, 0.0)
        if feature_names:
            row = [feats.get(fn, 0.0) for fn in feature_names]
        else:
            row = list(feats.values())
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def get_feature_contributions(model, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Per-feature SHAP-like contributions using XGBoost pred_contribs."""
    try:
        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=feature_names if feature_names else None)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        return contribs[:, :-1]
    except Exception as e:
        log("SCORER", f"pred_contribs unavailable ({e}), returning zeros", level="WARN")
        return np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)


# ---------------------------------------------------------------------------
# Main scoring dispatcher
# ---------------------------------------------------------------------------

def run_xgboost_scoring(
    models_dir: str,
    candidate_ids: List[str],
    features_map: Dict[str, Dict[str, float]],
    cross_encoder_scores: Dict[str, float],
    rough_scores: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Full scoring pipeline. Falls back to deterministic scoring if no XGBoost model.
    Returns list of (candidate_id, composite_score, features) sorted desc, top_k only.
    """
    return run_deterministic_recruiter_scoring(
        candidate_ids=candidate_ids,
        features_map=features_map,
        cross_encoder_scores=cross_encoder_scores,
        rough_scores=rough_scores,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


def _cap(value: float, cap: float) -> float:
    """Normalise value to [0, 1] relative to cap."""
    if cap <= 0:
        return 0.0
    return _clamp(float(value) / cap)


def _normalize_mapping(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a dict of scores."""
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float64)
    mn = float(np.nanmin(values))
    mx = float(np.nanmax(values))
    if np.isclose(mx, mn):
        return {cid: 1.0 for cid in scores}
    return {
        cid: float(np.clip((float(s) - mn) / (mx - mn), 0.0, 1.0))
        for cid, s in scores.items()
    }


def _recruiter_fit_score(features: Dict[str, float]) -> float:
    """
    Compute a deterministic recruiter-fit composite score.
    Revised to align with JD: heavy emphasis on location, shipping evidence,
    and product-company experience; lower weight on plain skill counts.
    """
    # ---- Semantic match (25%) ----
    ce = features.get("cross_encoder_score", 0.0)
    rough = features.get("rough_retrieval_score", 0.0)
    semantic = 0.70 * ce + 0.30 * rough

    # ---- Skill depth (20%) ----
    expert = _cap(features.get("expert_skill_count", 0.0), 6.0)
    advanced = _cap(features.get("advanced_skill_count", 0.0), 8.0)
    intermediate = _cap(features.get("intermediate_skill_count", 0.0), 8.0)
    total_score = _cap(features.get("total_skill_score", 0.0), 12.0)
    verified = features.get("verified_skill_score", 0.0)
    skill_depth = (
        0.30 * expert
        + 0.30 * advanced
        + 0.15 * intermediate
        + 0.15 * total_score
        + 0.10 * verified
    )

    # ---- Experience (15%) ----
    exp_years = features.get("total_experience_years", 0.0)
    avg_tenure = features.get("avg_tenure_per_job", 0.0)
    seniority = _cap(features.get("title_seniority", 2.0), 5.0)
    jobs_count = features.get("jobs_count", 0.0)

    exp_score = _cap(exp_years, 10.0)
    # Sweet spot 6-8 years -> extra boost
    if 6.0 <= exp_years <= 8.0:
        exp_score = min(1.0, exp_score * 1.2)
    if 0.0 < exp_years < 1.5:
        exp_score *= 0.5

    tenure_score = _cap(avg_tenure, 3.0)
    tenure_stddev = features.get("tenure_stddev", 0.0)
    stability = 1.0 - _cap(tenure_stddev, 3.0) * 0.3

    experience = (
        0.50 * exp_score
        + 0.20 * tenure_score
        + 0.20 * seniority
        + 0.10 * stability
    )

    # ---- Production / shipping evidence (20%) ----
    ranking_evidence = features.get("ranking_evidence_score", 0.0)
    has_shipped = features.get("has_shipped_ranking_system", 0.0)
    has_product_exp = features.get("has_product_company_experience", 0.0)
    production = features.get("production_evidence", 0.0)
    completeness = features.get("profile_completeness", 0.5)
    response_rate = features.get("response_rate", 0.0)
    github = features.get("has_github_link", 0.0)
    endorse = _cap(np.log1p(features.get("endorsement_count", 0.0)), np.log1p(200.0))
    edu_level = _cap(features.get("education_level", 0.0), 4.0)
    edu_tier = features.get("education_tier_score", 0.25)
    interview_rate = features.get("interview_completion_rate", 0.0)

    # New weighting: ranking evidence and shipping are the core
    proof_quality = (
        0.40 * ranking_evidence
        + 0.20 * has_shipped
        + 0.10 * production
        + 0.05 * completeness
        + 0.05 * response_rate
        + 0.05 * github
        + 0.05 * endorse
        + 0.05 * edu_level
        + 0.05 * edu_tier
        + 0.05 * interview_rate
    )

    # ---- Logistics & Location (20%) ----
    is_india = features.get("is_india_based", 0.0)
    location_score = features.get("is_target_location", 0.0)
    # If not India, location score becomes almost zero
    if is_india < 0.5:
        location_score = 0.0
    else:
        location_score = location_score  # already 0-1

    notice_days = features.get("notice_period_days", 30.0)
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 1.0 - ((notice_days - 30) / 30.0) * 0.5
    else:
        notice_score = max(0.1, 0.5 - (notice_days - 60) / 120.0)

    open_to_work = features.get("open_to_work", 0.0)

    logistics = (
        0.50 * location_score
        + 0.35 * notice_score
        + 0.15 * open_to_work
    )

    # ---- Company type penalties (applied after composite) ----
    is_big_tech = features.get("is_big_tech", 0.0)
    is_consulting = features.get("is_consulting", 0.0)
    has_product_exp = features.get("has_product_company_experience", 0.0)

    big_tech_penalty = 0.0
    if is_big_tech and not has_product_exp:
        big_tech_penalty = 0.4   # big tech-only → heavy penalty
    elif is_big_tech and has_product_exp:
        big_tech_penalty = 0.1   # still a small penalty

    consulting_penalty = 0.0
    if is_consulting and not has_product_exp:
        consulting_penalty = 0.5   # consulting-only → heavy penalty
    elif is_consulting and has_product_exp:
        consulting_penalty = 0.2

    # ---- Composite ----
    score = (
        0.25 * semantic
        + 0.20 * skill_depth
        + 0.15 * experience
        + 0.20 * proof_quality
        + 0.20 * logistics
    )

    # Apply penalties
    score -= (big_tech_penalty + consulting_penalty)

    # ---- Standard honeypot penalties ----
    honeypot = features.get("honeypot_flag_count", 0.0)
    expert_zero = features.get("expert_skill_zero_years_count", 0.0)
    company_anom = features.get("company_age_anomaly", 0.0)

    penalty = (
        0.08 * _cap(honeypot, 3.0)
        + 0.06 * _cap(expert_zero, 3.0)
        + 0.06 * company_anom
        + 0.02 * _cap(features.get("beginner_skill_count", 0.0), 8.0)
    )

    return float(np.clip(score - penalty, 0.0, 1.0))


def run_deterministic_recruiter_scoring(
    candidate_ids: List[str],
    features_map: Dict[str, Dict[str, float]],
    cross_encoder_scores: Dict[str, float],
    rough_scores: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Deterministic scoring pass. Uses normalised cross-encoder and retrieval scores
    then applies weighted feature formula.
    """
    start_timer("xgboost_total")
    log("SCORER", f"Deterministic scoring for {len(candidate_ids)} candidates...")
    t0 = time.perf_counter()

    norm_ce = _normalize_mapping(cross_encoder_scores)
    norm_rough = _normalize_mapping(rough_scores)

    scored: List[Tuple[str, float, Dict[str, float]]] = []
    for cid in candidate_ids:
        enriched = dict(features_map.get(cid, {}))
        enriched["cross_encoder_score"] = norm_ce.get(cid, 0.0)
        enriched["rough_retrieval_score"] = norm_rough.get(cid, 0.0)
        s = _recruiter_fit_score(enriched)
        scored.append((cid, s, enriched))

    elapsed = time.perf_counter() - t0
    record_timing("deterministic_scoring", elapsed)

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    final_scores = np.array([s[1] for s in top], dtype=np.float32)
    log(
        "SCORER",
        f"Scoring done in {elapsed:.4f}s — top-{top_k} "
        f"mean={final_scores.mean():.4f}, "
        f"min={final_scores.min():.4f}, "
        f"max={final_scores.max():.4f}",
    )
    stop_timer("xgboost_total", f"top {len(top)} selected")
    log_memory()
    return top


# ---------------------------------------------------------------------------
# Legacy XGBoost model path (kept for reference / future use)
# ---------------------------------------------------------------------------

def _run_xgboost_model_scoring(
    models_dir: str,
    candidate_ids: List[str],
    features_map: Dict[str, Dict[str, float]],
    cross_encoder_scores: Dict[str, float],
    rough_scores: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """XGBoost model inference path (disabled until a real model is trained)."""
    start_timer("xgboost_total")
    model, feature_names = load_xgboost_model(models_dir)

    if not feature_names:
        feature_names = _infer_feature_names(features_map, candidate_ids)
        feature_names += ["cross_encoder_score", "rough_retrieval_score"]

    X = build_feature_matrix(
        candidate_ids, features_map, cross_encoder_scores, rough_scores, feature_names
    )
    log("SCORER", f"Feature matrix shape: {X.shape}")

    try:
        raw_scores = model.predict_proba(X)[:, 1]
    except AttributeError:
        raw_scores = model.predict(X)

    raw_scores = _normalise_to_unit(raw_scores)

    penalised: List[Tuple[str, float, Dict[str, float]]] = []
    for i, cid in enumerate(candidate_ids):
        feats = features_map.get(cid, {})
        s = float(raw_scores[i])
        # Soft penalty for honeypot flags
        hf = int(feats.get("honeypot_flag_count", 0))
        if hf >= 2:
            s *= max(0.05, 1.0 - hf * 0.25)
        penalised.append((cid, s, feats))

    penalised.sort(key=lambda x: x[1], reverse=True)
    stop_timer("xgboost_total", f"top {top_k} selected")
    log_memory()
    return penalised[:top_k]


def _normalise_to_unit(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mn, mx = float(np.nanmin(scores)), float(np.nanmax(scores))
    if mn >= 0.0 and mx <= 1.0:
        return scores
    if np.isclose(mx, mn):
        return np.clip(scores, 0.0, 1.0)
    return np.clip((scores - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def _infer_feature_names(features_map: Dict, candidate_ids: List[str]) -> List[str]:
    for cid in candidate_ids:
        feats = features_map.get(cid, {})
        if feats:
            return list(feats.keys())
    return []