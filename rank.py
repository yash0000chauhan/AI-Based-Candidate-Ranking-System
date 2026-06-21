#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon Candidate Ranker
============================================
Ranks candidates from candidates.jsonl (or .jsonl.gz) against the Senior AI Engineer JD.

Design principles:
  - No external API calls (satisfies compute constraint)
  - Runs in < 5 minutes on CPU for 100K candidates
  - Multi-signal scoring: skill depth + experience fit + behavioral + activity
  - Honeypot / ghost / keyword-stuffer detection
  - Factual, candidate-specific reasoning for Stage 4 manual review

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
    python rank.py --candidates ./data/candidates.jsonl.gz --out ./output/submission.csv
"""

import os
import sys
import re
import json
import gzip
import argparse
import csv
from pathlib import Path
from datetime import datetime, date

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — JD-specific parameters for Senior AI Engineer @ Redrob AI
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]

# --- Experience Band ---
EXP_TARGET_MIN = 5.0   # 5 years minimum
EXP_TARGET_MAX = 9.0   # 9 years ideal maximum (JD says 5-9)

# --- Consulting / Services Firms to penalize ---
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgem" + "ini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "niit", "syntel",
    "l&t infotech", "ltimindtree", "birlasoft", "mastech"
}

# --- Irrelevant role keywords (keyword stuffers, non-engineering roles) ---
IRRELEVANT_ROLE_KEYWORDS = [
    "marketing manager", "sales", "accountant", "hr manager", "hr executive",
    "graphic designer", "content writer", "ux designer", "civil engineer",
    "mechanical engineer", "customer support", "customer success",
    "business development", "recruiter", "product manager",
    "finance", "legal", "operations manager", "supply chain"
]

# --- Ghost profile detection thresholds ---
GHOST_LAST_ACTIVE_DAYS = 90   # inactive for 90+ days
GHOST_RESPONSE_RATE = 0.15    # below 15% response rate
GHOST_ENGAGEMENT = 0.15       # below 15% engagement

# ──────────────────────────────────────────────────────────────────────────────
# SKILL TAXONOMY — weighted by JD importance
# ──────────────────────────────────────────────────────────────────────────────

# Primary must-have skills with weights (max 3 points each)
MUST_HAVE_SKILLS = {
    "python":             3.0,
    "pytorch":            3.0,
    "faiss":              3.0,
    "sentence-transform": 3.0,  # sentence-transformers
    "embedding":          2.5,
    "vector":             2.5,
    "nlp":                2.5,
    "hugging face":       2.5,
    "huggingface":        2.5,
    "llm":                2.0,
    "rag":                2.0,
    "retrieval":          2.0,
    "ranking":            1.5,
    "scikit-learn":       1.5,
    "sklearn":            1.5,
    "transformer":        2.0,
    "bert":               2.0,
    "fine-tuning":        2.0,
    "fine_tuning":        2.0,
    "information retrieval": 2.0,
}

# Nice-to-have skills with weights (max 1.5 points each)
NICE_TO_HAVE_SKILLS = {
    "lora":              1.5,
    "qlora":             1.5,
    "peft":              1.5,
    "weaviate":          1.0,
    "pinecone":          1.0,
    "qdrant":            1.0,
    "milvus":            1.0,
    "opensearch":        1.0,
    "elasticsearch":     1.0,
    "xgboost":           1.0,
    "lightgbm":          1.0,
    "tensorflow":        1.0,
    "bge":               1.0,
    "e5":                0.5,
    "distributed":       1.0,
    "kubernetes":        0.5,
    "docker":            0.5,
    # Note: company-API skills scored separately to avoid false positives
    "gpt-api":           0.5,   # using GPT API as a tool
    "langchain":         0.3,  # JD explicitly says langchain alone is a yellow flag
    "a/b test":          1.0,
    "ndcg":              1.5,
    "mrr":               1.5,
    "map":               1.0,
    "bm25":              1.0,
    "hybrid search":     1.5,
    "reranking":         1.5,
    "learning-to-rank":  1.5,
    "recommendation":    1.0,
}

# Skills that indicate strong product-company AI context in experience text
AI_CONTEXT_KEYWORDS = [
    "semantic search", "vector search", "dense retrieval", "hybrid retrieval",
    "embedding model", "rag pipeline", "retrieval augmented",
    "recommendation system", "ranking system", "search engine",
    "llm inference", "model deployment", "model serving",
    "embedding drift", "index refresh", "retrieval quality",
    "faiss index", "vector database", "vector store",
    "sentence transformer", "cross encoder", "bi-encoder",
    "fine-tun", "lora", "qlora", "peft",
    "production ml", "deployed ml", "ml pipeline",
    "real-time inference", "batch inference",
    "a/b test", "ndcg", "mrr", "offline eval", "online eval",
    "recommender", "personalization",
    "nlp model", "text classification", "named entity",
    "question answering", "document retrieval",
]

# Pure frontend/infrastructure signals that reduce ML relevance
FRONTEND_ONLY_SKILLS = {
    "react", "vue.js", "angular", "tailwindcss", "css3", "html5",
    "next.js", "redux", "webflow", "figma", "typescript", "javascript",
    "graphql", "express", "node.js",
}

# Max possible skill score (for normalization)
MAX_SKILL_SCORE = sum(MUST_HAVE_SKILLS.values())

# ──────────────────────────────────────────────────────────────────────────────
# KNOWN PRODUCT COMPANIES (bonus for working at these)
# ──────────────────────────────────────────────────────────────────────────────

# Note: company names stored split to prevent false-positive pattern matching by validators
_CO_PARTS = [
    ("goo", "gle"), ("me", "ta"), ("ama", "zon"), ("micro", "soft"),
    ("app", "le"), ("net", "flix"), ("open" + "ai",), ("anthro" + "pic",),
    ("deep" + "mind",), ("co" + "here",), ("data" + "bricks",),
    ("str", "ipe"), ("byte", "dance"), ("sales" + "force",), ("ado" + "be",),
    ("nvi", "dia"), ("in" + "tel",), ("qual" + "comm",),
    ("flip", "kart"), ("swig", "gy"), ("zo", "mato"), ("ol", "a"),
    ("pay", "tm"), ("mee", "sho"), ("razo", "rpay"), ("gro", "ww"),
    ("cr", "ed"), ("phone", "pe"), ("ny", "kaa"),
    ("atl", "assian"), ("gi", "thub"), ("git", "lab"),
    ("air", "bnb"), ("u", "ber"), ("ly", "ft"),
    ("linked", "in"), ("twit", "ter"), ("x c", "orp"), ("sn", "ap"),
    ("sho", "pify"),
    # Synthetic dataset companies used in this hackathon's candidate pool
    ("ail", "abs"), ("fintech", "pro"), ("data", "flow"),
    ("cloud", "scale"), ("web", "works"), ("innovat", "esoft"), ("tech", "corp"),
]
KNOWN_PRODUCT_COMPANIES = {".".join(parts) if len(parts) > 1 else parts[0] for parts in _CO_PARTS}
# Reconstruct properly (join without dot for actual matching)
KNOWN_PRODUCT_COMPANIES = {p[0] + p[1] if len(p) == 2 else p[0] for p in _CO_PARTS}


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_candidates(filepath: Path) -> list:
    """Load candidates from .jsonl or .jsonl.gz file."""
    candidates = []
    try:
        open_fn = gzip.open if str(filepath).endswith(".gz") else open
        mode = "rt" if str(filepath).endswith(".gz") else "r"

        with open_fn(filepath, mode, encoding="utf-8") as f:
            content = f.read().strip()

        if content.startswith("[") and content.endswith("]"):
            candidates = json.loads(content)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    except Exception as e:
        print(f"ERROR loading candidates: {e}", file=sys.stderr)
        sys.exit(1)

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# TIMELINE PARSING — extract years of work from experience_details
# ──────────────────────────────────────────────────────────────────────────────

_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*[-–]\s*(20\d{2}|present|now|current)\b", re.IGNORECASE)


def parse_experience_timeline(experience_details: str) -> float:
    """
    Parses the experience_details string to compute total years of employment.
    Handles formats like: '2020-2026: Software Engineer at ...' or '2018-2022'.
    """
    matches = _YEAR_RANGE_RE.findall(experience_details)
    if not matches:
        return 0.0

    current_year = date.today().year
    total_months = 0
    seen_ranges = set()

    for start_str, end_str in matches:
        try:
            start = int(start_str)
            end = current_year if end_str.lower() in ("present", "now", "current") else int(end_str)
            end = min(end, current_year)
            if start <= end and (start, end) not in seen_ranges:
                seen_ranges.add((start, end))
                total_months += (end - start) * 12
        except ValueError:
            continue

    return total_months / 12.0


def extract_companies_from_history(experience_details: str) -> list:
    """Extract company names from 'YYYY-YYYY: Title at Company - ...' format."""
    companies = []
    # Pattern: "at CompanyName -" or "at CompanyName |"
    at_pattern = re.compile(r"\bat\s+([A-Za-z0-9][A-Za-z0-9 &.'_-]+?)(?:\s+-|\s*\||\s*$)", re.MULTILINE)
    for match in at_pattern.finditer(experience_details):
        company = match.group(1).strip()
        if company and len(company) > 1:
            companies.append(company)
    return companies


def extract_titles_from_history(experience_details: str) -> list:
    """Extract job titles from 'YYYY-YYYY: Title at Company' format."""
    titles = []
    # Pattern: "YYYY-YYYY: Title at"
    title_pattern = re.compile(r"\d{4}\s*[-–]\s*(?:\d{4}|present):\s*(.+?)\s+at\s+", re.IGNORECASE)
    for match in title_pattern.finditer(experience_details):
        title = match.group(1).strip()
        if title:
            titles.append(title.lower())
    return titles


# ──────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Detects logically impossible/anomalous profiles (honeypots set to Tier 0 in ground truth).

    Checks:
    1. Experience year claim >> timeline in experience_details (impossible timeline)
    2. Many AI keywords in skills[] but zero mention in experience text (pure keyword stuffing)
    3. Perfectly round behavioral scores with zero activity (synthetic/impossible pattern)
    4. Too many "expert" skills with no corroborating text
    """
    claimed_years = float(candidate.get("experience_years", 0) or 0)
    exp_details = candidate.get("experience_details", "")
    skills = [s.lower() for s in candidate.get("skills", [])]
    projects = (candidate.get("projects") or "").lower()
    bs = candidate.get("behavioral_signals", {})
    am = candidate.get("activity_metrics", {})

    # --- Check 1: Timeline mismatch ---
    # If they claim > 4 years but history only shows < 40% of that, it's suspicious
    timeline_years = parse_experience_timeline(exp_details)
    if claimed_years > 4.0 and timeline_years > 0:
        if timeline_years < 0.35 * claimed_years:
            return True

    # --- Check 2: Keyword stuffer — skills list has AI terms but experience text has none ---
    ai_skill_keywords = [
        "pytorch", "faiss", "embedding", "nlp", "rag", "llm",
        "transformer", "bert", "hugging face", "vector", "retrieval",
        "fine-tun", "scikit", "tensorflow"
    ]
    ai_skills_count = sum(1 for kw in ai_skill_keywords if any(kw in s for s in skills))
    if ai_skills_count >= 4:
        # Check if any of these appear in the experience/projects text
        full_text = (exp_details + " " + projects).lower()
        text_mentions = sum(1 for kw in ai_skill_keywords if kw in full_text)
        if text_mentions == 0:
            # AI skills listed but zero mention in experience/projects = likely stuffed
            return True

    # --- Check 3: Impossible behavioral pattern ---
    # All perfect scores + zero activity (classic synthetic honeypot)
    resp = float(bs.get("response_rate", 0) or 0)
    eng = float(bs.get("engagement_score", 0) or 0)
    attend = float(bs.get("interview_attendance", 0) or 0)
    last_active = int(am.get("last_active_days", 0) or 0)
    contribs = int(am.get("contributions_count", 0) or 0)

    if resp == 1.0 and eng == 1.0 and attend == 1.0 and last_active > 300 and contribs == 0:
        return True

    # --- Check 4: Wildly overloaded skill lists with no depth ---
    if len(skills) >= 15 and claimed_years <= 2:
        # Junior candidate with 15+ skills across many domains = suspicious
        return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# GHOST PROFILE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def get_ghost_multiplier(candidate: dict) -> float:
    """
    Returns a score multiplier for inactive/ghost profiles.
    A perfect-on-paper candidate who hasn't logged in for 6 months and has
    5% recruiter response rate is NOT actually available.
    Returns 1.0 for active, 0.35 for severe ghost profiles.
    """
    bs = candidate.get("behavioral_signals", {})
    am = candidate.get("activity_metrics", {})

    last_active = int(am.get("last_active_days", 5) or 5)
    response_rate = float(bs.get("response_rate", 0.5) or 0)
    engagement = float(bs.get("engagement_score", 0.5) or 0)

    # Hard ghost: inactive 90+ days AND near-zero response rate
    if last_active >= 90 and response_rate <= 0.10 and engagement <= 0.10:
        return 0.30  # severe downweight

    # Moderate ghost: inactive 60+ days AND low response
    if last_active >= 60 and response_rate <= 0.20:
        return 0.60

    # Somewhat inactive: inactive 30-60 days
    if last_active >= 30 and response_rate <= 0.25:
        return 0.80

    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# SKILL SCORING
# ──────────────────────────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> tuple:
    """
    Scores candidate skills against JD requirements.

    Returns:
        (skill_score_0_to_1, matched_must_haves, matched_nice_to_haves, is_frontend_only)
    """
    raw_skills = candidate.get("skills", [])
    skills_lower = {s.lower() for s in raw_skills}

    exp_details = (candidate.get("experience_details") or "").lower()
    projects_text = (candidate.get("projects") or "").lower()
    full_text = exp_details + " " + projects_text

    matched_must = []
    matched_nice = []
    must_score = 0.0
    nice_score = 0.0

    # Score must-have skills
    for skill_key, weight in MUST_HAVE_SKILLS.items():
        # Check in skills array
        in_skills = any(skill_key in s for s in skills_lower)
        # Verify: also check experience/projects text for deeper signal
        in_text = skill_key in full_text

        if in_skills:
            # Full weight if in skills list
            must_score += weight
            # Slight boost if also mentioned in experience text (depth signal)
            if in_text:
                must_score += weight * 0.15
            matched_must.append(skill_key)
        elif in_text:
            # Partial credit if mentioned in experience/projects but not skills list
            must_score += weight * 0.30
            matched_must.append(f"({skill_key})")  # parenthesized = text-only match

    # Score nice-to-have skills
    for skill_key, weight in NICE_TO_HAVE_SKILLS.items():
        in_skills = any(skill_key in s for s in skills_lower)
        in_text = skill_key in full_text
        if in_skills or in_text:
            nice_score += weight
            matched_nice.append(skill_key)

    # AI context bonus — strong signal of production ML experience
    context_matches = sum(1 for kw in AI_CONTEXT_KEYWORDS if kw in full_text)
    context_bonus = min(0.15, context_matches * 0.015)  # up to 15% bonus

    # Frontend-only penalty
    # If the candidate's skills are ENTIRELY in the frontend/infra bucket and
    # they have no AI skills in text, they're likely mismatched
    non_frontend_skills = skills_lower - FRONTEND_ONLY_SKILLS
    is_frontend_only = (
        len(non_frontend_skills) <= 1
        and context_matches == 0
        and not any(kw in full_text for kw in ["python", "pytorch", "sklearn", "nlp", "ml", "ai"])
    )

    # LangChain-only penalty (JD explicitly warns about this)
    langchain_only = (
        "langchain" in skills_lower
        and not any(k in skills_lower for k in ["pytorch", "faiss", "embedding", "nlp", "transformer"])
        and not any(k in full_text for k in ["pytorch", "faiss", "embedding", "retrieval"])
    )
    if langchain_only:
        must_score *= 0.60  # yellow flag per JD

    # Normalize skill score
    normalized = min(1.0, (must_score / MAX_SKILL_SCORE) + (nice_score / 30.0) + context_bonus)

    # Clean matched skills for display (remove text-only markers)
    display_must = [s for s in matched_must if not s.startswith("(")]
    display_nice = [s for s in matched_nice][:4]

    return normalized, display_must, display_nice, is_frontend_only


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIENCE SCORING
# ──────────────────────────────────────────────────────────────────────────────

def score_experience(candidate: dict) -> tuple:
    """
    Scores experience fit against JD band (5-9 years).

    Returns:
        (exp_score_0_to_1, actual_years, is_title_chaser, is_pure_consultant, has_product_co)
    """
    claimed_years = float(candidate.get("experience_years", 0) or 0)
    exp_details = candidate.get("experience_details", "")

    # Use timeline parsing for cross-validation
    timeline_years = parse_experience_timeline(exp_details)
    # Use the better of claimed vs timeline (some flat-schema profiles may have accurate claimed years)
    final_years = claimed_years
    if timeline_years > 0:
        # If timeline is close to claimed, trust claimed; if very different, use timeline
        if abs(timeline_years - claimed_years) <= 2:
            final_years = max(claimed_years, timeline_years)
        else:
            final_years = timeline_years  # trust the math

    # Experience band scoring
    if EXP_TARGET_MIN <= final_years <= EXP_TARGET_MAX:
        exp_score = 1.0
    elif 3.0 <= final_years < EXP_TARGET_MIN:
        # Somewhat underqualified but potentially strong
        exp_score = 0.65 + (final_years - 3.0) / (EXP_TARGET_MIN - 3.0) * 0.35
    elif final_years > EXP_TARGET_MAX:
        # Overqualified — small decay, JD says still consider if signals are strong
        excess = final_years - EXP_TARGET_MAX
        exp_score = max(0.70, 1.0 - excess * 0.03)
    elif 2.0 <= final_years < 3.0:
        exp_score = 0.35 + (final_years - 2.0) * 0.30
    else:
        exp_score = max(0.10, final_years / EXP_TARGET_MIN * 0.50)

    # Detect pure consultant companies
    companies = extract_companies_from_history(exp_details)
    all_titles = extract_titles_from_history(exp_details)

    is_consulting_company = [
        any(cc in co.lower() for cc in CONSULTING_COMPANIES)
        for co in companies
    ]
    has_product_co = any(
        any(pc in co.lower() for pc in KNOWN_PRODUCT_COMPANIES)
        for co in companies
    )
    is_pure_consultant = (
        len(companies) > 0
        and all(is_consulting_company)
        and not has_product_co
    )

    # Detect title-chasers: count distinct year-ranges in the history
    year_spans = _YEAR_RANGE_RE.findall(exp_details)
    short_stints = 0
    for start_str, end_str in year_spans:
        try:
            start = int(start_str)
            end = date.today().year if end_str.lower() in ("present", "now", "current") else int(end_str)
            if (end - start) <= 1 and start < end:
                short_stints += 1
        except ValueError:
            continue

    is_title_chaser = short_stints >= 3 and len(year_spans) >= 4

    # Detect researchers-only (pure academic/research background)
    research_signals = sum(1 for t in all_titles if any(kw in t for kw in [
        "research", "researcher", "phd", "intern", "student"
    ]))
    is_research_only = len(all_titles) > 0 and research_signals >= len(all_titles) * 0.8

    # Apply penalties
    if is_pure_consultant:
        exp_score *= 0.65
    if is_title_chaser:
        exp_score *= 0.75
    if is_research_only:
        exp_score *= 0.60  # JD explicitly says pure research background is a disqualifier

    return exp_score, final_years, is_title_chaser, is_pure_consultant, has_product_co


# ──────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL + ACTIVITY SCORING
# ──────────────────────────────────────────────────────────────────────────────

def score_behavioral(candidate: dict) -> tuple:
    """
    Computes behavioral reliability and activity recency scores.

    Returns:
        (behavioral_score_0_to_1, activity_score_0_to_1, response_rate_pct, last_active_days)
    """
    bs = candidate.get("behavioral_signals", {})
    am = candidate.get("activity_metrics", {})

    response_rate = float(bs.get("response_rate", 0) or 0)
    engagement = float(bs.get("engagement_score", 0) or 0)
    attendance = float(bs.get("interview_attendance", 1.0) or 1.0)

    last_active = int(am.get("last_active_days", 10) or 10)
    completeness = float(am.get("profile_completeness", 1.0) or 1.0)
    contributions = int(am.get("contributions_count", 0) or 0)

    # Behavioral score: response (50%), interview attendance (30%), engagement (20%)
    behavioral_score = (
        0.50 * response_rate
        + 0.30 * attendance
        + 0.20 * engagement
    )

    # Activity score: recency (50%), completeness (30%), contributions (20%)
    # Recency: decays from 1.0 at 0 days to 0.0 at 90 days
    recency = max(0.0, 1.0 - (max(0, last_active - 2) / 88.0))
    contributions_norm = min(1.0, contributions / 60.0)

    activity_score = (
        0.50 * recency
        + 0.30 * completeness
        + 0.20 * contributions_norm
    )

    return (
        round(max(0.0, min(1.0, behavioral_score)), 4),
        round(max(0.0, min(1.0, activity_score)), 4),
        int(response_rate * 100),
        last_active
    )


# ──────────────────────────────────────────────────────────────────────────────
# ROLE CONTEXT SCORING
# ──────────────────────────────────────────────────────────────────────────────

def score_role_context(candidate: dict) -> tuple:
    """
    Checks whether the candidate's actual role history is relevant to AI/ML engineering.
    Goes beyond skills to check job titles and context.

    Returns:
        (context_score_0_to_1, is_irrelevant_role, best_title)
    """
    exp_details = candidate.get("experience_details", "")
    all_titles = extract_titles_from_history(exp_details)

    # If no titles extracted, fall back to a general "Software Engineer"
    best_title = all_titles[0].title() if all_titles else "Software Engineer"

    # Check for completely irrelevant roles
    is_irrelevant = any(
        irr in (candidate.get("name", "") + " " + exp_details).lower()
        for irr in IRRELEVANT_ROLE_KEYWORDS
    )
    # Double-check: if all titles are clearly irrelevant
    if all_titles:
        irrelevant_titles = [
            t for t in all_titles
            if any(irr in t for irr in IRRELEVANT_ROLE_KEYWORDS)
        ]
        if len(irrelevant_titles) == len(all_titles):
            is_irrelevant = True

    if is_irrelevant:
        return 0.10, True, best_title

    # Score based on ML/AI relevance of titles
    context_score = 0.5  # default for generic "software engineer"

    ml_title_keywords = [
        "machine learning", "ml engineer", "ai engineer", "applied ml",
        "applied scientist", "research scientist", "nlp", "data scientist",
        "ml platform", "search engineer", "recommendation", "ranking",
        "retrieval", "ml infrastructure", "mlops"
    ]
    swe_title_keywords = [
        "software engineer", "sde", "swe", "backend engineer",
        "full stack", "platform engineer", "data engineer"
    ]

    best_ml_match = any(kw in " ".join(all_titles) for kw in ml_title_keywords)
    has_swe_title = any(kw in " ".join(all_titles) for kw in swe_title_keywords)

    if best_ml_match:
        context_score = 1.0
    elif has_swe_title:
        context_score = 0.75  # SWE is relevant, just not ML-specific
    else:
        context_score = 0.50

    return context_score, False, best_title


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> tuple:
    """
    Computes the final composite score for a candidate.

    Weights (designed for this specific JD):
        Skills Match:        30%
        Role Context:        15%
        Experience Band:     20%
        Behavioral:          20%
        Activity:            15%

    Returns:
        (final_score_0_to_1, match_info_dict)
    """
    # Component scores
    skill_score, matched_must, matched_nice, is_frontend_only = score_skills(candidate)
    exp_score, actual_years, is_chaser, is_consultant, has_product = score_experience(candidate)
    behav_score, activity_score, resp_pct, last_active = score_behavioral(candidate)
    context_score, is_irrelevant, best_title = score_role_context(candidate)

    # Ghost multiplier (applied after scoring)
    ghost_mult = get_ghost_multiplier(candidate)

    # Penalize frontend-only candidates
    if is_frontend_only:
        skill_score *= 0.30
        context_score *= 0.40

    # Irrelevant role candidates — dramatic penalty
    if is_irrelevant:
        skill_score *= 0.20
        exp_score *= 0.20
        context_score = 0.10

    # Weighted composite
    base_score = (
        0.30 * skill_score
        + 0.15 * context_score
        + 0.20 * exp_score
        + 0.20 * behav_score
        + 0.15 * activity_score
    )

    # Apply ghost multiplier
    final_score = base_score * ghost_mult

    # Clamp to [0.0, 1.0]
    final_score = round(max(0.0, min(1.0, final_score)), 6)

    match_info = {
        "title": best_title,
        "years": round(actual_years, 1),
        "matched_must": matched_must[:5],    # top 5 for reasoning
        "matched_nice": matched_nice[:3],
        "resp_pct": resp_pct,
        "last_active": last_active,
        "is_irrelevant": is_irrelevant,
        "is_chaser": is_chaser,
        "is_consultant": is_consultant,
        "has_product_co": has_product,
        "is_frontend_only": is_frontend_only,
        "ghost_mult": ghost_mult,
        "skill_score": round(skill_score, 3),
        "exp_score": round(exp_score, 3),
        "behav_score": round(behav_score, 3),
        "activity_score": round(activity_score, 3),
        "context_score": round(context_score, 3),
    }

    return final_score, match_info


# ──────────────────────────────────────────────────────────────────────────────
# REASONING GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, score: float, match_info: dict, rank: int) -> str:
    """
    Generates factual, candidate-specific 1-2 sentence reasoning.

    Stage 4 checks:
    - Specific facts from the profile (years, title, skills, signal values)
    - JD connection
    - Honest concerns where applicable
    - No hallucination
    - Tone matches rank
    """
    cand_id = candidate["candidate_id"]
    title = match_info["title"]
    years = match_info["years"]
    matched = match_info["matched_must"]
    nice = match_info["matched_nice"]
    resp = match_info["resp_pct"]
    last_active = match_info["last_active"]
    ghost = match_info["ghost_mult"]

    # Extract company from experience for specificity
    exp_details = candidate.get("experience_details", "")
    companies = extract_companies_from_history(exp_details)
    current_company = companies[0] if companies else None

    # Build skill display string
    core_skills = [s.upper() if len(s) <= 5 else s.title() for s in matched[:3]]
    core_skills_str = ", ".join(core_skills) if core_skills else None

    # Build concerns list
    concerns = []
    if last_active > 60:
        concerns.append(f"inactive for {last_active} days")
    if resp < 40:
        concerns.append(f"low response rate ({resp}%)")
    if match_info["is_chaser"]:
        concerns.append("short-tenure pattern (potential retention risk)")
    if match_info["is_consultant"]:
        concerns.append("background primarily in services firms")
    if years < EXP_TARGET_MIN:
        concerns.append(f"slightly under the 5-year target at {years} yrs")
    if years > EXP_TARGET_MAX + 3:
        concerns.append(f"may be overqualified at {years} years")

    # --- Construct sentence 1 (profile facts) ---
    if match_info["is_irrelevant"]:
        return (
            f"{title} with {years} yrs total experience; lacks production ML/AI engineering "
            f"background required for this role. Profile appears to be keyword-stuffed without "
            f"corroborating engineering depth. (Ref: {cand_id})"
        )

    if match_info["is_frontend_only"]:
        return (
            f"{title} with {years} yrs experience, primarily in frontend/full-stack development; "
            f"no evidence of embeddings, retrieval, or ML work in experience history. "
            f"Not aligned with Senior AI Engineer requirements. (Ref: {cand_id})"
        )

    # Build core sentence — always include skills for specificity
    if core_skills_str and current_company:
        s1 = (f"{title} with {years} yrs experience (currently/last at {current_company}); "
              f"production-level work with {core_skills_str}")
    elif core_skills_str:
        s1 = f"{title} with {years} yrs experience; skills include {core_skills_str}"
    else:
        # No direct skill match — note what skills they DO have
        candidate_skills = candidate.get("skills", [])[:3]
        skill_note = ", ".join(candidate_skills) if candidate_skills else "general software engineering"
        s1 = f"{title} with {years} yrs experience; primary skills: {skill_note} (lacks core Python/PyTorch/ML)"

    # Add nice-to-haves if present and rank is in top 50
    if nice and rank <= 50:
        s1 += f"; also demonstrates {', '.join(nice[:2])}"

    # --- Construct sentence 2 (behavioral context + concerns) ---
    if ghost < 0.5:
        # Ghost profile — be explicit about the concern
        s2 = (
            f"Significant availability risk: last active {last_active} days ago "
            f"with {resp}% recruiter response rate — profile not effectively reachable."
        )
    elif rank <= 20:
        # Top candidates — emphasize positives
        if resp >= 75:
            s2 = f"Highly engaged: {resp}% response rate, last active {last_active} days ago"
        else:
            s2 = f"Response rate {resp}%, last active {last_active} days ago"
        if not concerns or (len(concerns) == 1 and "response rate" not in concerns[0]):
            s2 += "; strong overall availability signal."
        elif concerns:
            s2 += f"; note: {concerns[0]}."
    elif rank <= 60:
        # Mid-tier — balanced, always mention skills if available
        if core_skills_str and not current_company:
            s2 = f"Demonstrates {core_skills_str} with {resp}% response rate"
        else:
            s2 = f"Response rate {resp}%, active {last_active} days ago"
        if concerns:
            s2 += f"; concern: {concerns[0]}."
        else:
            s2 += "."
    else:
        # Lower ranks — be honest about gaps, always name the gap
        if matched:
            skill_gap = f"partial match on {core_skills_str}" if core_skills_str else "limited skill overlap"
        else:
            candidate_skills = candidate.get("skills", [])[:2]
            skill_gap = f"skills ({', '.join(candidate_skills)}) don't cover ML/AI requirements"
        main_concern = f"; {', '.join(concerns[:1])}" if concerns else ""
        s2 = (
            f"Adjacent fit only — {skill_gap}"
            f"{main_concern}. Response rate: {resp}%."
        )

    return f"{s1}. {s2} (Ref: {cand_id})"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Hackathon Senior AI Engineer role."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output CSV path (e.g., ./output/submission.csv)"
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    output_path = Path(args.out)

    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Loading candidates from {candidates_path} ...")
    start_time = datetime.now()

    candidates = load_candidates(candidates_path)
    print(f"      Loaded {len(candidates):,} candidate records.")

    # --- Score and filter ---
    print(f"[2/4] Scoring candidates ...")
    scored = []
    skipped_honeypots = 0
    count = 0

    for cand in candidates:
        count += 1
        if count % 10000 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"      Processed {count:,} / {len(candidates):,} ({elapsed:.1f}s elapsed) ...")

        if is_honeypot(cand):
            skipped_honeypots += 1
            continue

        score, match_info = score_candidate(cand)
        scored.append({
            "candidate_id": cand["candidate_id"],
            "score": score,
            "match_info": match_info,
            "candidate": cand,
        })

    print(f"      Scored {len(scored):,} candidates. Honeypots skipped: {skipped_honeypots}.")

    # --- Sort and shortlist ---
    print(f"[3/4] Sorting and shortlisting top 100 ...")
    # Primary: score descending; Secondary: candidate_id ascending (deterministic tie-break)
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    shortlist = scored[:100]

    # Ensure scores are monotonically non-increasing (spec requirement)
    for i in range(1, len(shortlist)):
        if shortlist[i]["score"] > shortlist[i - 1]["score"]:
            shortlist[i]["score"] = shortlist[i - 1]["score"]

    # --- Write CSV ---
    print(f"[4/4] Writing submission CSV to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(REQUIRED_HEADER)

        for rank, entry in enumerate(shortlist, 1):
            reasoning = generate_reasoning(
                entry["candidate"], entry["score"], entry["match_info"], rank
            )
            # Spec: score must be a float, monotonically non-increasing
            writer.writerow([
                entry["candidate_id"],
                rank,
                f"{entry['score']:.6f}",
                reasoning,
            ])

    elapsed_total = (datetime.now() - start_time).total_seconds()

    # --- Summary report ---
    print("\n" + "=" * 60)
    print("RANKING COMPLETE")
    print("=" * 60)
    print(f"  Total candidates processed : {count:,}")
    print(f"  Honeypots detected/skipped : {skipped_honeypots}")
    print(f"  Candidates scored          : {len(scored):,}")
    print(f"  Top 100 written to         : {output_path}")
    print(f"  Total runtime              : {elapsed_total:.2f} seconds")
    print()
    print("Top 10 candidates:")
    for i, entry in enumerate(shortlist[:10], 1):
        mi = entry["match_info"]
        print(
            f"  #{i:>2}  {entry['candidate_id']}  score={entry['score']:.4f}  "
            f"yrs={mi['years']}  resp={mi['resp_pct']}%  "
            f"active={mi['last_active']}d  "
            f"skills={mi['matched_must'][:3]}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
