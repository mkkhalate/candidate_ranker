"""
scorer.py — Deterministic recruiter-fit scoring with strict JD alignment.

This version:
  - Filters candidates to 5–9 years experience.
  - Applies penalties for title‑chasers, framework‑only, CV/speech primary,
    low shipping evidence, non‑India, and consulting‑only.
  - Includes new features (external_validation, eval_framework).
"""

import os
import pickle
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from utils import log, log_memory, record_timing, start_timer, stop_timer, ts

# ---------------------------------------------------------------------------
# XGBoost model loading (kept for future use)
# ---------------------------------------------------------------------------

def load_xgboost_model(models_dir: str):
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[SCORER] xgboost_model.pkl not found at {model_path}."
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
    try:
        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=feature_names if feature_names else None)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        return contribs[:, :-1]
    except Exception as e:
        log("SCORER", f"pred_contribs unavailable ({e}), returning zeros", level="WARN")
        return np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def run_xgboost_scoring(
    models_dir: str,
    candidate_ids: List[str],
    features_map: Dict[str, Dict[str, float]],
    cross_encoder_scores: Dict[str, float],
    rough_scores: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[str, float, Dict[str, float]]]:
    return run_deterministic_recruiter_scoring(
        candidate_ids=candidate_ids,
        features_map=features_map,
        cross_encoder_scores=cross_encoder_scores,
        rough_scores=rough_scores,
        top_k=top_k,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))

def _cap(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return _clamp(float(value) / cap)

def _normalize_mapping(scores: Dict[str, float]) -> Dict[str, float]:
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

# ---------------------------------------------------------------------------
# Core recruiter fit score (with all penalties)
# ---------------------------------------------------------------------------

def _recruiter_fit_score(features: Dict[str, float]) -> float:
    # Semantic match (25%)
    ce = features.get("cross_encoder_score", 0.0)
    rough = features.get("rough_retrieval_score", 0.0)
    semantic = 0.70 * ce + 0.30 * rough

    # Skill depth (20%)
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

    # Experience (15%) — but we already filtered to 5‑9, so this is just bonus
    exp_years = features.get("total_experience_years", 0.0)
    avg_tenure = features.get("avg_tenure_per_job", 0.0)
    seniority = _cap(features.get("title_seniority", 2.0), 5.0)

    # Since we already enforce 5‑9, exp_score is 1.0 for all remaining
    if 5.0 <= exp_years <= 9.0:
        exp_score = 1.0
    else:
        exp_score = max(0.0, 1.0 - abs(exp_years - 7.0) * 0.2)

    tenure_score = _cap(avg_tenure, 3.0)
    tenure_stddev = features.get("tenure_stddev", 0.0)
    stability = 1.0 - _cap(tenure_stddev, 3.0) * 0.3

    experience = (
        0.50 * exp_score
        + 0.20 * tenure_score
        + 0.20 * seniority
        + 0.10 * stability
    )

    # Production / shipping evidence (20%)
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

    external_val = features.get("external_validation_score", 0.0)
    eval_framework = features.get("eval_framework_experience", 0.0)

    proof_quality = (
        0.50 * ranking_evidence
        + 0.25 * has_shipped
        + 0.05 * production
        + 0.05 * has_product_exp
        + 0.03 * external_val
        + 0.02 * eval_framework
        + 0.05 * completeness
        + 0.05 * response_rate
        + 0.05 * github
        + 0.05 * endorse
        + 0.05 * edu_level
        + 0.05 * edu_tier
        + 0.05 * interview_rate
    )

    # Logistics (20%)
    is_india = features.get("is_india_based", 0.0)
    location_score = features.get("is_target_location", 0.0)
    if is_india < 0.5:
        location_score = 0.0

    notice_days = features.get("notice_period_days", 30.0)
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 1.0 - ((notice_days - 30) / 30.0) * 0.5
    elif notice_days <= 90:
        notice_score = 0.5 - ((notice_days - 60) / 30.0) * 0.3
    else:
        notice_score = max(0.05, 0.2 - (notice_days - 90) / 200.0)

    open_to_work = features.get("open_to_work", 0.0)

    logistics = (
        0.50 * location_score
        + 0.35 * notice_score
        + 0.15 * open_to_work
    )

    # Composite before penalties
    score = (
        0.25 * semantic
        + 0.20 * skill_depth
        + 0.15 * experience
        + 0.20 * proof_quality
        + 0.20 * logistics
    )

    # ---- Explicit penalties ----
    penalty = 0.0

    if features.get("is_title_chaser", 0) > 0.5:
        penalty += 0.30

    if features.get("is_framework_enthusiast", 0) > 0.5:
        penalty += 0.35

    if features.get("is_cv_speech_primary", 0) > 0.5:
        penalty += 0.50

    # No shipped ranking and low evidence
    if features.get("has_shipped_ranking_system", 0) < 0.5 and features.get("ranking_evidence_score", 0) < 0.3:
        penalty += 0.20

    if features.get("is_india_based", 0) < 0.5:
        penalty += 0.15

    # Consulting penalty (with product exp pardon)
    is_consulting = features.get("is_consulting", 0)
    has_product = features.get("has_product_company_experience", 0)
    if is_consulting > 0.5 and has_product < 0.5:
        penalty = max(penalty, 0.60)
    elif is_consulting > 0.5 and has_product >= 0.5:
        penalty = max(penalty, 0.20)

    score = max(0.0, score - penalty)

    # Integrity penalties (unchanged)
    honeypot = features.get("honeypot_flag_count", 0.0)
    expert_zero = features.get("expert_skill_zero_years_count", 0.0)
    company_anom = features.get("company_age_anomaly", 0.0)

    integrity_penalty = (
        0.08 * _cap(honeypot, 3.0)
        + 0.06 * _cap(expert_zero, 3.0)
        + 0.06 * company_anom
        + 0.02 * _cap(features.get("beginner_skill_count", 0.0), 8.0)
    )
    score = max(0.0, score - integrity_penalty)

    return float(np.clip(score, 0.0, 1.0))

# ---------------------------------------------------------------------------
# Deterministic scoring with experience filter
# ---------------------------------------------------------------------------

def run_deterministic_recruiter_scoring(
    candidate_ids: List[str],
    features_map: Dict[str, Dict[str, float]],
    cross_encoder_scores: Dict[str, float],
    rough_scores: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[str, float, Dict[str, float]]]:
    start_timer("xgboost_total")
    log("SCORER", f"Deterministic scoring for {len(candidate_ids)} candidates...")

    # ---- Filter by experience (5‑9 years) ----
    filtered_ids = []
    for cid in candidate_ids:
        exp = features_map.get(cid, {}).get("total_experience_years", 0.0)
        if 5.0 <= exp <= 9.0:
            filtered_ids.append(cid)
        else:
            log("SCORER", f"Excluding {cid} (experience {exp:.1f} outside 5‑9)", level="DEBUG")

    if not filtered_ids:
        log("SCORER", "No candidates in experience range. Returning empty.", level="WARN")
        return []

    log("SCORER", f"Retained {len(filtered_ids)} candidates within 5‑9 years.")

    # ---- Normalise scores ----
    norm_ce = _normalize_mapping({cid: cross_encoder_scores.get(cid, 0.0) for cid in filtered_ids})
    norm_rough = _normalize_mapping({cid: rough_scores.get(cid, 0.0) for cid in filtered_ids})

    scored: List[Tuple[str, float, Dict[str, float]]] = []
    t0 = time.perf_counter()
    for cid in filtered_ids:
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
        f"mean={final_scores.mean():.4f}, min={final_scores.min():.4f}, max={final_scores.max():.4f}",
    )
    stop_timer("xgboost_total", f"top {len(top)} selected")
    log_memory()
    return top