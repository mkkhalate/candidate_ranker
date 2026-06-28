"""
preprocessor.py — Builds text blobs and extracts 30+ structural features.
Now includes:
  - is_india_based, is_big_tech, is_consulting
  - has_product_company_experience
  - ranking_evidence_score, has_shipped_ranking_system
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils import log

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRODUCTION_KEYWORDS_STRONG = {
    "shipped", "launched", "deployed", "released", "scaled", "migrated",
    "architected", "productionized",
}
PRODUCTION_KEYWORDS_MEDIUM = {
    "built", "delivered", "owned", "drove", "led", "designed", "implemented",
    "developed", "created", "established",
}

LOCATION_BOOST_CITIES = {"pune", "noida", "bangalore", "bengaluru", "hyderabad", "chennai"}

SENIORITY_KEYWORDS = {
    "staff": 5, "principal": 5, "distinguished": 5, "fellow": 5,
    "lead": 4, "senior": 3, "sr.": 3, "sr ": 3,
    "mid": 2, "junior": 1, "jr.": 1, "jr ": 1, "associate": 1, "trainee": 1, "intern": 0,
}

# Degree normalisation
_DEGREE_MAP: List[Tuple[int, List[str]]] = [
    (4, ["phd", "ph d", "doctorate", "doctor of"]),
    (3, ["mtech", "m tech", "me ", "m e ", "ms ", "m s ", "msc", "m sc", "mba",
         "m ba", "master", "postgrad", "pgd", "pgdm"]),
    (2, ["btech", "b tech", "be ", "b e ", "bs ", "b s ", "bsc", "b sc",
         "bachelor", "beng", "b eng", "basc", "b asc"]),
    (1, ["diploma", "associate", "certificate", "polytechnic"]),
]

TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50, "tier_4": 0.25, "": 0.25}

PROFICIENCY_SCORES = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2, "novice": 0.15}

HANDS_ON_KEYWORDS = {
    "engineer", "developer", "architect", "scientist", "programmer", "coder",
    "implemented", "coded", "debugged", "repository", "pipeline", "api", "sdk",
}

# ---- New JD-aligned constants ----
BIG_TECH_KEYWORDS = {
    "google", "meta", "facebook", "microsoft", "amazon", "apple", "netflix",
    "linkedin", "uber", "airbnb", "salesforce", "adobe", "oracle", "ibm"
}
CONSULTING_KEYWORDS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "lti", "mphasis", "hexaware", "virtusa"
}
RANKING_EVIDENCE_KEYWORDS = {
    "ranking", "retrieval", "search", "recommendation", "relevance",
    "ndcg", "mrr", "map", "learning to rank", "ltr", "evaluation",
    "offline evaluation", "online ab test", "ab test"
}
SHIP_EVIDENCE_KEYWORDS = {
    "shipped", "built", "owned", "led", "deployed", "launched", "scaled",
    "production", "real users", "production system"
}

# ---------------------------------------------------------------------------
# Degree parsing
# ---------------------------------------------------------------------------

def _parse_degree_level(raw_degree: str) -> int:
    if not raw_degree:
        return 0
    norm = raw_degree.lower().replace(".", " ").replace("-", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    norm_spaced = norm + " "
    for level, patterns in _DEGREE_MAP:
        for pat in patterns:
            if pat in norm_spaced:
                return level
    return 0


def _parse_institution_tier(edu_dict: Dict) -> float:
    tier = str(edu_dict.get("tier", "")).lower().strip()
    return TIER_SCORES.get(tier, 0.25)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%m/%Y", "%B %Y", "%b %Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    m = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    if m:
        try:
            return datetime(int(m.group()), 6, 1)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Experience computation
# ---------------------------------------------------------------------------

def _compute_experience_years(employer_history: List[Dict]) -> Tuple[float, float, float, float]:
    if not employer_history:
        return 0.0, 0.0, 0.0, 0.0

    intervals: List[Tuple[datetime, datetime]] = []
    tenures: List[float] = []
    desc_lengths: List[int] = []

    now = datetime.now()
    for job in employer_history:
        start = _parse_date(job.get("start_date") or job.get("from"))
        end_raw = job.get("end_date") or job.get("to") or job.get("current")
        if end_raw in (True, "Present", "present", "current", "Current", None, ""):
            end = now
        else:
            end = _parse_date(str(end_raw))
        if start is None:
            continue
        if end is None:
            end = now
        if end < start:
            continue
        intervals.append((start, end))
        tenures.append((end - start).days / 365.25)
        desc = job.get("description", "")
        desc_lengths.append(len(desc.split()))

    if not intervals:
        return 0.0, 0.0, 0.0, 0.0

    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    total = sum((e - s).days / 365.25 for s, e in merged)
    avg = sum(tenures) / len(tenures) if tenures else 0.0

    if len(tenures) > 1:
        mean_t = sum(tenures) / len(tenures)
        stddev = (sum((t - mean_t) ** 2 for t in tenures) / len(tenures)) ** 0.5
    else:
        stddev = 0.0

    avg_desc_len = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0.0
    density = min(1.0, avg_desc_len / 100.0)

    return round(total, 2), round(avg, 2), round(stddev, 2), round(density, 4)


# ---------------------------------------------------------------------------
# Skill analysis
# ---------------------------------------------------------------------------

def _analyse_skills(skills: List) -> Dict[str, Any]:
    expert_count = 0
    advanced_count = 0
    intermediate_count = 0
    beginner_count = 0
    expert_zero_years = 0
    total_score = 0.0
    weighted_endorse = 0.0

    for s in skills:
        if not isinstance(s, dict):
            continue
        prof = str(s.get("proficiency", "")).lower().strip()
        months = s.get("duration_months", None)
        endorse = float(s.get("endorsements", 0) or 0)
        prof_score = PROFICIENCY_SCORES.get(prof, 0.1)
        total_score += prof_score
        weighted_endorse += prof_score * endorse

        if prof == "expert":
            expert_count += 1
            try:
                if months is not None and float(months) == 0:
                    expert_zero_years += 1
            except (ValueError, TypeError):
                pass
        elif prof == "advanced":
            advanced_count += 1
        elif prof == "intermediate":
            intermediate_count += 1
        elif prof in ("beginner", "novice"):
            beginner_count += 1

    return {
        "expert_count": expert_count,
        "advanced_count": advanced_count,
        "intermediate_count": intermediate_count,
        "beginner_count": beginner_count,
        "expert_zero_years_count": expert_zero_years,
        "total_skill_score": total_score,
        "weighted_endorsements": weighted_endorse,
        "total_skills": expert_count + advanced_count + intermediate_count + beginner_count,
    }


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def _company_age_anomaly(employer_history: List[Dict]) -> int:
    now_year = datetime.now().year
    for job in employer_history:
        founding = job.get("company_founding_year") or job.get("founded")
        if founding:
            try:
                founding_yr = int(founding)
                company_age = now_year - founding_yr
                start = _parse_date(job.get("start_date") or job.get("from"))
                if start:
                    yrs_at_company = (datetime.now() - start).days / 365.25
                    if yrs_at_company > company_age + 1:
                        return 1
            except (ValueError, TypeError):
                pass
    return 0


def _has_github_link(candidate: Dict[str, Any]) -> int:
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    url_fields = [
        profile.get("github_url", ""), profile.get("github", ""),
        profile.get("portfolio_url", ""), profile.get("website", ""),
        candidate.get("github_url", ""), candidate.get("github", ""),
    ]
    for field in url_fields:
        if field and "github.com" in str(field).lower():
            return 1
    score = signals.get("github_activity_score", -1)
    if score is not None:
        try:
            if float(score) > 0:
                return 1
        except (TypeError, ValueError):
            pass
    return 0


def _get_signal(signals: Dict, key: str, default: Any = 0) -> Any:
    if not signals:
        return default
    val = signals.get(key, default)
    return val if val is not None else default


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def _is_target_location(profile: Dict[str, Any], signals: Dict[str, Any]) -> int:
    location = str(profile.get("location", "") or "").lower()
    for city in LOCATION_BOOST_CITIES:
        if city in location:
            return 1
    reloc_raw = signals.get("willing_to_relocate", None) or profile.get("relocation_preference", None)
    if reloc_raw is not None:
        if isinstance(reloc_raw, bool):
            if reloc_raw:
                return 1
        else:
            reloc_str = str(reloc_raw).lower().strip()
            if reloc_str in ("true", "yes", "anywhere", "open"):
                return 1
    return 0


# ---------------------------------------------------------------------------
# Production evidence
# ---------------------------------------------------------------------------

def _production_evidence_score(text_blob: str) -> float:
    lower = text_blob.lower()
    score = 0.0
    for kw in PRODUCTION_KEYWORDS_STRONG:
        if kw in lower:
            score += 0.4
    for kw in PRODUCTION_KEYWORDS_MEDIUM:
        if kw in lower:
            score += 0.15
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

def _title_seniority(title: str) -> int:
    if not title:
        return 2
    lower = title.lower()
    for keyword, level in sorted(SENIORITY_KEYWORDS.items(), key=lambda x: -x[1]):
        if keyword in lower:
            return level
    return 2


# ---------------------------------------------------------------------------
# Skill assessment
# ---------------------------------------------------------------------------

def _skill_assessment_score(signals: Dict) -> float:
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments or not isinstance(assessments, dict):
        return 0.0
    scores = [v for v in assessments.values() if isinstance(v, (int, float)) and v >= 0]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores) / 100.0, 4)


# ---------------------------------------------------------------------------
# Title vs Skill Mismatch
# ---------------------------------------------------------------------------

def _role_experience_alignment(title: str, skills: List) -> float:
    if not title or not skills:
        return 0.5
    title_lower = title.lower()
    has_hands_on_title = any(kw in title_lower for kw in HANDS_ON_KEYWORDS)
    tech_skill_count = 0
    for s in skills:
        if not isinstance(s, dict):
            continue
        prof = str(s.get("proficiency", "")).lower()
        if prof in ("expert", "advanced"):
            tech_skill_count += 1
    if not has_hands_on_title and tech_skill_count >= 3:
        return 0.2
    if has_hands_on_title and tech_skill_count >= 2:
        return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# NEW: Company-type helpers (Big Tech, Consulting, Product experience)
# ---------------------------------------------------------------------------

def _is_big_tech(company: str) -> int:
    if not company:
        return 0
    c = company.lower()
    return 1 if any(kw in c for kw in BIG_TECH_KEYWORDS) else 0


def _is_consulting(company: str) -> int:
    if not company:
        return 0
    c = company.lower()
    return 1 if any(kw in c for kw in CONSULTING_KEYWORDS) else 0


def _has_product_company_experience(employer_history: List[Dict]) -> int:
    """
    Returns 1 if the candidate has at least one job at a non‑consulting, non‑big‑tech product company.
    """
    for job in employer_history:
        company = job.get("company", "")
        if not company:
            continue
        c = company.lower()
        is_consulting = any(kw in c for kw in CONSULTING_KEYWORDS)
        is_big = any(kw in c for kw in BIG_TECH_KEYWORDS)
        if not is_consulting and not is_big:
            return 1
    return 0


def _ranking_evidence_score(employer_history: List[Dict]) -> float:
    """
    Score based on how strongly the candidate's job descriptions mention
    ranking/search/recommendation systems and production shipping.
    """
    score = 0.0
    for job in employer_history:
        desc = job.get("description", "").lower()
        rank_terms = sum(1 for kw in RANKING_EVIDENCE_KEYWORDS if kw in desc)
        ship_terms = sum(1 for kw in SHIP_EVIDENCE_KEYWORDS if kw in desc)
        job_score = min(0.5, (rank_terms * 0.05 + ship_terms * 0.08))
        score += job_score
    return min(1.0, score)


def _has_shipped_ranking_system(employer_history: List[Dict]) -> int:
    """
    Returns 1 if any job description contains both ranking/search terms AND shipping terms.
    """
    for job in employer_history:
        desc = job.get("description", "").lower()
        has_rank = any(kw in desc for kw in RANKING_EVIDENCE_KEYWORDS)
        has_ship = any(kw in desc for kw in SHIP_EVIDENCE_KEYWORDS)
        if has_rank and has_ship:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Text blob
# ---------------------------------------------------------------------------

def build_text_blob(candidate: Dict[str, Any], max_words: int = 600) -> str:
    parts: List[str] = []
    profile = candidate.get("profile", {}) or {}
    if not isinstance(profile, dict):
        profile = {}

    title = profile.get("current_title") or candidate.get("current_title", "")
    if title:
        parts.append(f"Current role: {title}.")

    headline = profile.get("headline", "") or candidate.get("headline", "")
    if headline:
        parts.append(headline)

    career_history = candidate.get("career_history") or candidate.get("employer_history", [])
    for job in (career_history or [])[:5]:
        role = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        if role or company:
            parts.append(f"Role: {role} at {company}.")
        if desc:
            parts.append(desc[:300])

    skills = candidate.get("skills", []) or []
    skill_strings = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        months = s.get("duration_months")
        if not name:
            continue
        entry = name
        if prof:
            entry += f" ({prof}"
            if months:
                entry += f", {months}m"
            entry += ")"
        skill_strings.append(entry)
    if skill_strings:
        parts.append("Skills: " + ", ".join(skill_strings) + ".")

    education = candidate.get("education", []) or []
    for edu in education[:3]:
        if not isinstance(edu, dict):
            continue
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        institution = edu.get("institution", "")
        if degree or field:
            edu_str = f"Education: {degree}"
            if field:
                edu_str += f" in {field}"
            if institution:
                edu_str += f" from {institution}"
            parts.append(edu_str + ".")

    summary = profile.get("summary", "") or candidate.get("summary", "") or candidate.get("about", "")
    if summary:
        parts.append(summary[:400])

    blob = " ".join(parts)
    words = blob.split()
    if len(words) > max_words:
        blob = " ".join(words[:max_words])
    return blob.strip()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    candidate: Dict[str, Any],
    text_blob: str,
) -> Dict[str, float]:
    profile = candidate.get("profile", {}) or {}
    if not isinstance(profile, dict):
        profile = {}

    employer_history = candidate.get("career_history") or candidate.get("employer_history") or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    if not isinstance(signals, dict):
        signals = {}

    # --- Experience ---
    total_exp, avg_tenure, tenure_stddev, desc_density = _compute_experience_years(employer_history)
    reported_exp = profile.get("years_of_experience")
    if reported_exp is not None:
        try:
            reported_exp = float(reported_exp)
            if abs(reported_exp - total_exp) <= 3:
                total_exp = reported_exp
        except (TypeError, ValueError):
            pass

    # --- Skills ---
    skill_stats = _analyse_skills(skills)

    # --- Integrity signals ---
    company_anomaly = _company_age_anomaly(employer_history)
    has_github = _has_github_link(candidate)

    # --- Recruiter / engagement signals ---
    response_rate = float(_get_signal(signals, "recruiter_response_rate",
                                      _get_signal(signals, "response_rate", 0.0)))
    notice_raw = _get_signal(signals, "notice_period_days", 30)
    try:
        notice_period = float(notice_raw)
    except (TypeError, ValueError):
        notice_period = 30.0

    open_to_work = float(bool(_get_signal(signals, "open_to_work_flag",
                                          _get_signal(signals, "open_to_work", False))))
    profile_completeness = float(_get_signal(signals, "profile_completeness_score",
                                             _get_signal(signals, "profile_completeness", 50.0)))
    if profile_completeness > 1.0:
        profile_completeness /= 100.0

    endorsements = float(_get_signal(signals, "endorsements_received",
                                     _get_signal(signals, "endorsement_count", 0)))
    interview_completion = float(_get_signal(signals, "interview_completion_rate", 0.0))
    offer_acceptance = float(_get_signal(signals, "offer_acceptance_rate", 0.0))
    if offer_acceptance < 0:
        offer_acceptance = 0.0
    if interview_completion < 0:
        interview_completion = 0.0

    # --- Location ---
    is_target_loc = _is_target_location(profile, signals)

    # --- Production evidence ---
    prod_score = _production_evidence_score(text_blob)

    # --- Education ---
    edu_level = 0
    best_tier_score = 0.0
    for edu in education:
        if not isinstance(edu, dict):
            continue
        deg = edu.get("degree", "")
        lvl = _parse_degree_level(deg)
        edu_level = max(edu_level, lvl)
        tier_s = _parse_institution_tier(edu)
        best_tier_score = max(best_tier_score, tier_s)

    # --- Seniority ---
    current_title = profile.get("current_title") or candidate.get("current_title", "")
    title_seniority = _title_seniority(current_title)
    role_match = _role_experience_alignment(current_title, skills)

    # --- Skill assessment ---
    verified_skill_score = _skill_assessment_score(signals)

    # --- Honeypot ---
    es = skill_stats
    total_skills = es["total_skills"]
    expert_zero = es["expert_zero_years_count"]
    expert_ratio_flag = 1 if (es["expert_count"] > total_skills * 0.9 and total_skills > 5) else 0
    honeypot_flag_count = int(expert_zero > 0) + int(company_anomaly) + expert_ratio_flag

    jobs_count = len(employer_history)

    # --- NEW: JD‑aligned features ---
    country = profile.get("country", "").lower()
    is_india = 1.0 if country == "india" else 0.0

    current_company = profile.get("current_company", "")
    is_big_tech = float(_is_big_tech(current_company))
    is_consulting = float(_is_consulting(current_company))
    has_product_exp = float(_has_product_company_experience(employer_history))
    ranking_evidence = _ranking_evidence_score(employer_history)
    has_shipped_ranking = float(_has_shipped_ranking_system(employer_history))

    return {
        # Experience
        "total_experience_years": float(total_exp),
        "avg_tenure_per_job": float(avg_tenure),
        "tenure_stddev": float(tenure_stddev),
        "jobs_count": float(jobs_count),
        "career_description_density": float(desc_density),

        # Skills
        "expert_skill_count": float(es["expert_count"]),
        "advanced_skill_count": float(es["advanced_count"]),
        "intermediate_skill_count": float(es["intermediate_count"]),
        "beginner_skill_count": float(es["beginner_count"]),
        "total_skill_score": float(es["total_skill_score"]),
        "total_skills": float(total_skills),

        # Integrity
        "expert_skill_zero_years_count": float(es["expert_zero_years_count"]),
        "company_age_anomaly": float(company_anomaly),
        "honeypot_flag_count": float(honeypot_flag_count),

        # Profile quality
        "has_github_link": float(has_github),
        "profile_completeness": float(profile_completeness),
        "endorsement_count": float(endorsements),
        "weighted_endorsements": float(es["weighted_endorsements"]),
        "verified_skill_score": float(verified_skill_score),
        "text_blob_word_count": float(len(text_blob.split())),

        # Engagement
        "response_rate": float(response_rate),
        "notice_period_days": float(notice_period),
        "open_to_work": float(open_to_work),
        "interview_completion_rate": float(interview_completion),
        "offer_acceptance_rate": float(offer_acceptance),

        # Education
        "education_level": float(edu_level),
        "education_tier_score": float(best_tier_score),

        # Title & seniority
        "title_seniority": float(title_seniority),
        "role_experience_alignment": float(role_match),

        # Location
        "is_target_location": float(is_target_loc),

        # Production evidence
        "production_evidence": float(prod_score),

        # ---- NEW FEATURES ----
        "is_india_based": float(is_india),
        "is_big_tech": float(is_big_tech),
        "is_consulting": float(is_consulting),
        "has_product_company_experience": float(has_product_exp),
        "ranking_evidence_score": float(ranking_evidence),
        "has_shipped_ranking_system": float(has_shipped_ranking),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_candidate(
    candidate: Dict[str, Any],
    idx: int,
    verbose: bool = False,
) -> Tuple[str, Dict[str, float], str]:
    t0 = time.perf_counter()
    cid = (
        candidate.get("candidate_id")
        or candidate.get("id")
        or f"CAND_{idx:06d}"
    )
    text_blob = build_text_blob(candidate)
    features = extract_features(candidate, text_blob)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(
            f"[PREPROCESSOR] {cid}: edu={features['education_level']:.0f}, "
            f"exp={features['total_experience_years']:.1f}y, "
            f"expert={features['expert_skill_count']:.0f}, "
            f"advanced={features['advanced_skill_count']:.0f}, "
            f"match={features['role_experience_alignment']:.2f} "
            f"({elapsed*1000:.1f}ms)",
            flush=True,
        )

    return cid, features, text_blob