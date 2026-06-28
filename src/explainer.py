"""
explainer.py — Generates per-candidate strengths/weaknesses via feature contributions.
Now includes specific skills, company, title, and JD-aligned signals.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from utils import log, ts


FEATURE_LABELS = {
    "total_experience_years": "years of relevant experience",
    "avg_tenure_per_job": "average job tenure",
    "expert_skill_count": "expert-level skills",
    "advanced_skill_count": "advanced-level skills",
    "expert_skill_zero_years_count": "expert skills claimed with zero usage (red flag)",
    "company_age_anomaly": "implausible experience claim",
    "has_github_link": "GitHub presence",
    "response_rate": "recruiter response rate",
    "notice_period_days": "notice period",
    "is_target_location": "preferred location match",
    "production_evidence": "production deployment evidence",
    "total_skills": "total skill count",
    "intermediate_skill_count": "intermediate-level skills",
    "beginner_skill_count": "beginner-level skills",
    "education_level": "education level",
    "education_tier_score": "institution prestige",
    "open_to_work": "open-to-work signal",
    "profile_completeness": "profile completeness",
    "endorsement_count": "professional endorsements",
    "jobs_count": "number of past roles",
    "honeypot_flag_count": "honeypot/integrity flags",
    "text_blob_word_count": "profile richness",
    "cross_encoder_score": "semantic match",
    "rough_retrieval_score": "keyword match",
    "title_seniority": "seniority level",
    "verified_skill_score": "verified skill assessments",
    "role_experience_alignment": "title vs skill alignment",
    # JD-aligned labels
    "is_india_based": "India-based",
    "is_big_tech": "currently at Big Tech",
    "is_consulting": "currently at consulting firm",
    "has_product_company_experience": "product-company experience",
    "ranking_evidence_score": "evidence of ranking system work",
    "has_shipped_ranking_system": "shipped ranking/search system",
    # NEW
    "external_validation_score": "external validation (papers, talks, open-source)",
    "eval_framework_experience": "evaluation framework experience (NDCG, MRR, A/B tests)",
    "is_title_chaser": "title-chaser pattern (red flag)",
    "is_framework_enthusiast": "framework enthusiast (red flag)",
    "is_cv_speech_primary": "CV/speech primary without NLP/IR (red flag)",
}

NEGATIVE_FEATURES = {
    "expert_skill_zero_years_count",
    "company_age_anomaly",
    "honeypot_flag_count",
    "notice_period_days",
    "beginner_skill_count",
    "is_consulting",
    "is_title_chaser",           # NEW
    "is_framework_enthusiast",   # NEW
    "is_cv_speech_primary",      # NEW
    # is_big_tech deliberately excluded
}


def _feature_value_str(feature_name: str, value: float) -> str:
    if feature_name == "notice_period_days":
        return f"{int(value)}-day notice period"
    if feature_name in ("cross_encoder_score", "response_rate", "profile_completeness", "verified_skill_score",
                        "external_validation_score"):
        return f"{value:.2f}"
    if feature_name in ("total_experience_years", "avg_tenure_per_job"):
        return f"{value:.1f} years"
    if feature_name == "has_github_link":
        return "GitHub profile found" if value > 0 else "no GitHub link"
    if feature_name == "is_target_location":
        return "location matches preference" if value > 0 else ""
    if feature_name == "production_evidence":
        return f"production evidence score {value:.2f}"
    if feature_name == "open_to_work":
        return "actively looking" if value > 0 else ""
    if feature_name == "education_level":
        levels = {0: "no degree", 1: "diploma", 2: "bachelor's", 3: "master's", 4: "PhD"}
        return levels.get(int(value), "degree")
    if feature_name == "education_tier_score":
        tiers = {1.0: "Tier 1", 0.75: "Tier 2", 0.5: "Tier 3", 0.25: "Tier 4"}
        return tiers.get(value, "")
    if feature_name == "role_experience_alignment":
        return "strong title-skill alignment" if value >= 0.8 else "potential title-skill mismatch" if value < 0.3 else ""
    if feature_name == "is_india_based":
        return "India-based" if value > 0 else "location outside India (visa risk)"
    if feature_name == "has_product_company_experience":
        return "product-company experience" if value > 0 else "no product-company experience"
    if feature_name == "has_shipped_ranking_system":
        return "shipped ranking/search system" if value > 0 else ""
    if feature_name == "ranking_evidence_score":
        return f"ranking evidence {value:.2f}"
    if feature_name == "eval_framework_experience":
        return "evaluation framework experience" if value > 0 else ""
    if feature_name == "is_title_chaser":
        return "title-chaser pattern (penalty)" if value > 0 else ""
    if feature_name == "is_framework_enthusiast":
        return "framework enthusiast (penalty)" if value > 0 else ""
    if feature_name == "is_cv_speech_primary":
        return "CV/speech primary without NLP/IR (penalty)" if value > 0 else ""
    return f"{value:.2f}"


def _extract_top_skills(candidate_profile: Dict) -> List[str]:
    if not candidate_profile:
        return []
    skills = candidate_profile.get("skills", [])
    if not skills:
        return []
    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    sorted_skills = sorted(
        [s for s in skills if isinstance(s, dict) and s.get("name")],
        key=lambda x: prof_order.get(x.get("proficiency", "").lower(), 0),
        reverse=True
    )
    return [s.get("name") for s in sorted_skills[:3] if s.get("name")]


def generate_reasoning(
    candidate_id: str,
    features: Dict[str, float],
    contribs: Optional[np.ndarray],
    feature_names: List[str],
    cross_encoder_score: float,
    composite_score: float,
    rank: int,
    candidate_profile: Optional[Dict] = None,
) -> str:
    """
    Generate a human-readable reasoning string for one candidate.
    Uses feature contributions plus specific profile details.
    """
    parts = []

    # Extract rich details from raw profile
    company = ""
    title = ""
    if candidate_profile:
        profile = candidate_profile.get("profile", {})
        title = profile.get("current_title", "")
        company = profile.get("current_company", "")

    # Opening: title + company
    if title and company:
        parts.append(f"{title} at {company}")
    elif title:
        parts.append(f"{title}")
    else:
        parts.append(f"Candidate")

    # Semantic match
    ce = cross_encoder_score
    parts.append(f"semantic match: {ce:.2f}")

    # Specific top skills
    top_skills = _extract_top_skills(candidate_profile)
    if top_skills:
        parts.append(f"strong skills in {', '.join(top_skills)}")
    else:
        exp_cnt = int(features.get("expert_skill_count", 0))
        adv_cnt = int(features.get("advanced_skill_count", 0))
        if exp_cnt > 0 or adv_cnt > 0:
            parts.append(f"{exp_cnt + adv_cnt} expert/advanced skills")

    # JD-aligned evidence: ranking system shipping
    if features.get("has_shipped_ranking_system", 0) > 0:
        parts.append("shipped a ranking/search system")
    elif features.get("ranking_evidence_score", 0) > 0.3:
        parts.append("strong evidence of ranking system work")

    # Product-company experience
    if features.get("has_product_company_experience", 0) > 0:
        parts.append("product-company experience")

    # External validation
    if features.get("external_validation_score", 0) > 0.5:
        parts.append("has external validation (papers/talks/open-source)")

    # Feature contributions (top positives)
    if contribs is not None and len(contribs) == len(feature_names):
        contrib_pairs = list(zip(feature_names, contribs))
    else:
        contrib_pairs = list(features.items())
        contrib_pairs = [(k, v) for k, v in contrib_pairs]

    positive = [
        (k, v, features.get(k, 0.0))
        for k, v in contrib_pairs
        if k not in NEGATIVE_FEATURES and v > 0
    ]
    positive.sort(key=lambda x: x[1], reverse=True)

    for feat_name, _, feat_val in positive[:2]:
        if feat_name in ("cross_encoder_score", "rough_retrieval_score"):
            continue
        label = FEATURE_LABELS.get(feat_name, feat_name.replace("_", " "))
        val_str = _feature_value_str(feat_name, feat_val)
        if val_str:
            parts.append(f"boosted by {val_str} ({label})")

    # Experience years
    exp_yrs = features.get("total_experience_years", 0.0)
    if exp_yrs > 0 and "total_experience_years" not in [p[0] for p in positive[:2]]:
        parts.append(f"{exp_yrs:.1f} years of experience")

    # Education
    edu_level = int(features.get("education_level", 0))
    edu_tier = features.get("education_tier_score", 0)
    if edu_level >= 2:
        level_str = "bachelor's" if edu_level == 2 else "master's" if edu_level == 3 else "PhD"
        tier_str = " (Tier 1)" if edu_tier >= 0.9 else ""
        parts.append(f"{level_str} degree{tier_str}")

    # Role alignment
    role_match = features.get("role_experience_alignment", 0.5)
    if role_match < 0.3:
        parts.append("Warning: title may not align with technical skills")
    elif role_match >= 0.8:
        parts.append("strong title-skill alignment")

    # Concerns / penalties
    concern_parts = []
    negative = [
        (k, v, features.get(k, 0.0))
        for k, v in contrib_pairs
        if k in NEGATIVE_FEATURES and (v > 0 or features.get(k, 0) > 0)
    ]
    negative.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat_name, _, feat_val in negative[:3]:  # include more items
        if feat_name == "notice_period_days" and feat_val > 30:
            concern_parts.append(f"{int(feat_val)}-day notice period")
        elif feat_name == "expert_skill_zero_years_count" and feat_val > 0:
            concern_parts.append(f"{int(feat_val)} expert skill(s) claimed with zero usage")
        elif feat_name == "honeypot_flag_count" and feat_val > 0:
            concern_parts.append(f"{int(feat_val)} integrity flag(s) detected")
        elif feat_name == "company_age_anomaly" and feat_val > 0:
            concern_parts.append("experience timeline anomaly detected")
        elif feat_name == "is_consulting" and feat_val > 0:
            if features.get("has_product_company_experience", 0) > 0:
                concern_parts.append("currently at consulting firm (pardoned with product exp)")
            else:
                concern_parts.append("currently at consulting firm (penalty applied)")
        elif feat_name == "is_title_chaser" and feat_val > 0:
            concern_parts.append("title-chaser pattern detected (penalty)")
        elif feat_name == "is_framework_enthusiast" and feat_val > 0:
            concern_parts.append("framework enthusiast (penalty)")
        elif feat_name == "is_cv_speech_primary" and feat_val > 0:
            concern_parts.append("CV/speech primary without NLP/IR (penalty)")

    if concern_parts:
        parts.append("Concern: " + "; ".join(concern_parts))

    # Rank / score summary
    if rank == 1:
        parts.append("Top pick")
    elif rank <= 10:
        parts.append(f"Ranked #{rank}")
    else:
        parts.append(f"Score: {composite_score:.4f}")

    return ". ".join(parts) + "."


def generate_all_reasoning(
    top_candidates: List[Tuple[str, float, Dict[str, float]]],
    cross_encoder_scores: Dict[str, float],
    feature_names: List[str],
    contributions: Optional[np.ndarray],
    candidate_profiles: Optional[Dict[str, Dict]] = None,
) -> Dict[str, str]:
    """
    Generate reasoning strings for all top-100 candidates.
    """
    log("EXPLAINER", f"Generating reasoning for {len(top_candidates)} candidates...")
    result = {}

    for rank, (cid, score, features) in enumerate(top_candidates, start=1):
        ce_score = cross_encoder_scores.get(cid, 0.0)
        contribs = contributions[rank - 1] if contributions is not None and rank - 1 < len(contributions) else None
        profile = candidate_profiles.get(cid) if candidate_profiles else None

        reasoning = generate_reasoning(
            candidate_id=cid,
            features=features,
            contribs=contribs,
            feature_names=feature_names,
            cross_encoder_score=ce_score,
            composite_score=score,
            rank=rank,
            candidate_profile=profile,
        )
        result[cid] = reasoning

        if rank % 10 == 0 or rank == 1:
            print(
                f"[{ts()}] [EXPLAINER] Generated reasoning for rank {rank}/{len(top_candidates)}: {cid}",
                flush=True,
            )

    log("EXPLAINER", f"Reasoning generation complete for {len(result)} candidates")
    return result