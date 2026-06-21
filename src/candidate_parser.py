import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

class CandidateParser:
    @staticmethod
    def _is_flat_schema(record: Dict[str, Any]) -> bool:
        """
        Detects if a record is already in the flat/standardized schema
        (has top-level 'skills', 'experience_years', 'behavioral_signals', 'activity_metrics')
        vs the nested schema (has 'profile', 'redrob_signals', 'career_history').
        """
        return (
            "experience_years" in record or
            "behavioral_signals" in record or
            "activity_metrics" in record
        )

    @staticmethod
    def _parse_skills(record: Dict[str, Any]) -> List[str]:
        """
        Normalizes skills to a flat list of strings.
        Handles:
          - list of strings: ["Python", "PyTorch"]
          - list of dicts:   [{"name": "Python", "level": "Expert"}, ...]
          - nested under 'skills' key at top level
        """
        raw = record.get("skills", [])
        result = []
        for s in raw:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, dict):
                name = s.get("name") or s.get("skill") or ""
                if name:
                    result.append(name)
        return result

    @staticmethod
    def load_candidates(file_path: Path = None) -> List[Dict[str, Any]]:
        """
        Loads candidate profiles from a JSONL or JSON array file.
        Automatically detects and handles two schemas:
          1. Flat schema  — fields at top level (skills as strings)
          2. Nested schema — 'profile', 'redrob_signals', 'career_history' wrappers
        """
        if file_path is None:
            file_path = DATA_DIR / "candidates.jsonl"

        candidates = []
        if not file_path.exists():
            logger.warning(f"Candidates file {file_path} not found. Returning empty list.")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Support both JSON array and JSONL (one object per line)
            if content.startswith("[") and content.endswith("]"):
                raw_candidates = json.loads(content)
            else:
                raw_candidates = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        raw_candidates.append(json.loads(line))

            for rc in raw_candidates:
                try:
                    if CandidateParser._is_flat_schema(rc):
                        # ── FLAT SCHEMA (already standardized) ──────────────────
                        skills_names = CandidateParser._parse_skills(rc)

                        candidates.append({
                            "candidate_id": rc["candidate_id"],
                            "name": rc.get("name") or "Anonymized Candidate",
                            "skills": skills_names,
                            "experience_years": float(rc.get("experience_years", 0) or 0),
                            "experience_details": rc.get("experience_details", ""),
                            "projects": rc.get("projects", ""),
                            "behavioral_signals": {
                                "response_rate":        float(rc.get("behavioral_signals", {}).get("response_rate", 0.0) or 0.0),
                                "engagement_score":     float(rc.get("behavioral_signals", {}).get("engagement_score", 0.0) or 0.0),
                                "interview_attendance": float(rc.get("behavioral_signals", {}).get("interview_attendance", 1.0) or 1.0),
                            },
                            "activity_metrics": {
                                "last_active_days":     int(rc.get("activity_metrics", {}).get("last_active_days", 5) or 5),
                                "profile_completeness": float(rc.get("activity_metrics", {}).get("profile_completeness", 1.0) or 1.0),
                                "contributions_count":  int(rc.get("activity_metrics", {}).get("contributions_count", 0) or 0),
                            },
                        })

                    else:
                        # ── NESTED SCHEMA (profile / redrob_signals / career_history) ──
                        profile  = rc.get("profile", {})
                        signals  = rc.get("redrob_signals", {})
                        history  = rc.get("career_history", [])

                        # Compute total experience
                        total_months = sum(job.get("duration_months") or 0 for job in history)
                        exp_years = max(
                            float(profile.get("years_of_experience") or 0.0),
                            total_months / 12.0
                        )

                        # Build experience details string
                        job_strings = []
                        for job in history:
                            start   = job.get("start_date") or ""
                            end     = job.get("end_date")   or "Present"
                            company = job.get("company")    or ""
                            title   = job.get("title")      or ""
                            desc    = job.get("description") or ""
                            job_strings.append(f"{start} - {end}: {title} at {company} - {desc}")
                        exp_details = " | ".join(job_strings)

                        skills_names = CandidateParser._parse_skills(rc)

                        # Determine last_active_days from signal date
                        last_active_date_str = signals.get("last_active_date", "")
                        if last_active_date_str:
                            try:
                                last_active_days = (
                                    datetime.now() - datetime.strptime(last_active_date_str, "%Y-%m-%d")
                                ).days
                            except ValueError:
                                last_active_days = 5
                        else:
                            last_active_days = 5

                        candidates.append({
                            "candidate_id": rc["candidate_id"],
                            "name": profile.get("anonymized_name") or "Anonymized Candidate",
                            "skills": skills_names,
                            "experience_years": round(exp_years, 1),
                            "experience_details": exp_details,
                            "projects": "",
                            "behavioral_signals": {
                                "response_rate":        float(signals.get("recruiter_response_rate") or 0.0),
                                "engagement_score":     float(signals.get("profile_completeness_score", 100.0) or 100.0) / 100.0,
                                "interview_attendance": float(signals.get("interview_completion_rate") or 1.0),
                            },
                            "activity_metrics": {
                                "last_active_days":     last_active_days,
                                "profile_completeness": float(signals.get("profile_completeness_score", 100.0) or 100.0) / 100.0,
                                "contributions_count":  int(max(0, signals.get("github_activity_score") or 0)),
                            },
                        })

                except Exception as record_err:
                    logger.warning(f"Skipping record due to error: {record_err}", exc_info=True)
                    continue

            logger.info(f"Loaded and standardized {len(candidates)} candidates from {file_path}")

        except Exception as e:
            logger.error(f"Error loading candidates file: {e}", exc_info=True)

        return candidates

    @staticmethod
    def construct_embedding_text(candidate: Dict[str, Any]) -> str:
        """
        Constructs a rich contextual description of a candidate to generate semantic embeddings.
        """
        skills_str   = ", ".join(candidate.get("skills", []))
        exp_years    = candidate.get("experience_years", 0)
        exp_details  = candidate.get("experience_details", "")
        projects     = candidate.get("projects", "")

        text  = "Candidate Profile:\n"
        text += f"- Total Experience: {exp_years} years\n"
        text += f"- Tech Skills: {skills_str}\n"
        text += f"- Professional Experience: {exp_details}\n"
        text += f"- Project Highlights: {projects}\n"
        return text

    @staticmethod
    def anonymize(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymizes candidate data to reduce demographic bias during manual and LLM screening.
        Replaces candidate name with candidate_id and strips potential gender-identifying cues.
        """
        anon_cand = candidate.copy()
        anon_cand["name"] = f"Candidate {candidate.get('candidate_id', 'Unknown')}"
        details = anon_cand.get("experience_details", "")
        details = details.replace(" he ",  " they ").replace(" she ",  " they ")
        details = details.replace(" He ",  " They ").replace(" She ",  " They ")
        details = details.replace(" his ", " their ").replace(" her ", " their ")
        details = details.replace(" His ", " Their ").replace(" Her ", " Their ")
        anon_cand["experience_details"] = details
        return anon_cand
