import logging
from typing import Dict, Any
from src.config import (
    WEIGHT_SEMANTIC, WEIGHT_EXPERIENCE, WEIGHT_BEHAVIORAL, WEIGHT_ACTIVITY,
    COLD_START_WEIGHT_SEMANTIC, COLD_START_WEIGHT_EXPERIENCE
)

logger = logging.getLogger(__name__)

class HybridScorer:
    @staticmethod
    def calculate_experience_score(candidate_years: float, required_years: float) -> float:
        """
        Calculates a non-linear experience alignment score.
        - Under-qualified: Linear/quadratic penalty down to 0.0.
        - Optimal range: [required_years, required_years + 4] -> 1.0.
        - Over-qualified: Gradual decay down to 0.70 to represent hiring/cost risk,
          without disqualifying the candidate.
        """
        if required_years <= 0:
            return 1.0
            
        if candidate_years < required_years:
            # Underqualified penalty
            return float((candidate_years / required_years) ** 1.2)
        elif required_years <= candidate_years <= (required_years + 4):
            # Perfect fit
            return 1.0
        else:
            # Overqualified decay
            excess_years = candidate_years - (required_years + 4)
            score = 1.0 - (0.03 * excess_years)
            return float(max(0.70, score))

    @staticmethod
    def calculate_behavioral_score(signals: Dict[str, Any]) -> float:
        """
        Computes behavioral reliability score from platform engagement signals:
        - response_rate (50%)
        - interview_attendance (30%)
        - engagement_score (20%)
        """
        if not signals:
            return 0.0
            
        response_rate = float(signals.get("response_rate", 0.0))
        interview_attendance = float(signals.get("interview_attendance", 1.0)) # Default to 1.0 if not yet interviewed
        engagement_score = float(signals.get("engagement_score", 0.0))
        
        score = (0.50 * response_rate) + (0.30 * interview_attendance) + (0.20 * engagement_score)
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def calculate_activity_score(metrics: Dict[str, Any]) -> float:
        """
        Computes activity score to measure recency and pipeline readiness:
        - last_active_days: Recency factor, decays to 0 over 60 days (50%)
        - profile_completeness (30%)
        - contributions_count: Cap at 50 commits/contributions (20%)
        """
        if not metrics:
            return 0.0
            
        last_active = float(metrics.get("last_active_days", 90))
        completeness = float(metrics.get("profile_completeness", 0.0))
        contributions = float(metrics.get("contributions_count", 0))
        
        # Recency decay (perfect score if active within 0-3 days, decays to 0 after 60 days)
        recency_score = 1.0 - min(1.0, max(0.0, last_active - 3) / 57.0)
        
        # Contributions scaling
        contributions_score = min(1.0, contributions / 50.0)
        
        score = (0.50 * recency_score) + (0.30 * completeness) + (0.20 * contributions_score)
        return float(max(0.0, min(1.0, score)))

    def score_candidate(self, candidate: Dict[str, Any], semantic_score: float, required_years: float, handle_cold_start: bool = True) -> Dict[str, Any]:
        """
        Combines components into a single hybrid score, checking for cold-start conditions.
        Returns a dictionary with the final score and sub-scores.
        """
        cand_id = candidate.get("candidate_id")
        cand_years = float(candidate.get("experience_years", 0))
        
        # 1. Experience Fit
        exp_score = self.calculate_experience_score(cand_years, required_years)
        
        # Check if candidate is a cold-start profile (i.e. has no behavioral log/activity records)
        activity_metrics = candidate.get("activity_metrics", {})
        behavioral_signals = candidate.get("behavioral_signals", {})
        
        is_cold = (
            activity_metrics.get("last_active_days", 999) > 365 and 
            behavioral_signals.get("response_rate", 0.0) == 0.0 and 
            activity_metrics.get("contributions_count", 0) == 0
        )
        
        if is_cold and handle_cold_start:
            # Cold-start path: re-allocate weight to semantic alignment and experience
            behavior_score = 0.0
            activity_score = 0.0
            
            final_score = (COLD_START_WEIGHT_SEMANTIC * semantic_score) + (COLD_START_WEIGHT_EXPERIENCE * exp_score)
            
            logger.info(f"Candidate {cand_id} evaluated under Cold-Start conditions (Final: {final_score:.4f})")
        else:
            # Standard hybrid scoring
            behavior_score = self.calculate_behavioral_score(behavioral_signals)
            activity_score = self.calculate_activity_score(activity_metrics)
            
            final_score = (
                (WEIGHT_SEMANTIC * semantic_score) +
                (WEIGHT_EXPERIENCE * exp_score) +
                (WEIGHT_BEHAVIORAL * behavior_score) +
                (WEIGHT_ACTIVITY * activity_score)
            )
            
        return {
            "candidate_id": cand_id,
            "name": candidate.get("name"),
            "semantic_score": round(semantic_score, 4),
            "experience_score": round(exp_score, 4),
            "behavior_score": round(behavior_score, 4),
            "activity_score": round(activity_score, 4),
            "hybrid_score": round(final_score, 4),
            "is_cold_start": is_cold
        }
