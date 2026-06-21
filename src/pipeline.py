import csv
import logging
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import concurrent.futures
from src.config import TOP_N_RETRIEVAL, OUTPUT_DIR
from src.job_parser import JobParser
from src.candidate_parser import CandidateParser
from src.embedder import Embedder
from src.retriever import FAISSRetriever
from src.scorer import HybridScorer
from src.reranker import LLMReranker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class CandidateRankingPipeline:
    def __init__(self, provider: str = None):
        self.job_parser = JobParser(provider=provider) if provider else JobParser()
        self.embedder = Embedder()
        self.scorer = HybridScorer()
        self.reranker = LLMReranker(provider=provider) if provider else LLMReranker()

    def run(self, jd_text: str, candidates_file: Path = None, output_csv: Path = None) -> List[Dict[str, Any]]:
        """
        Executes the entire ranking and shortlisting pipeline.
        Returns the final ranked list of candidates.
        """
        if output_csv is None:
            output_csv = OUTPUT_DIR / "shortlist.csv"
            
        # 1. Job Description Parsing
        logger.info("Step 1: Parsing Job Description...")
        parsed_jd = self.job_parser.parse(jd_text)
        logger.info(f"Parsed Job: {parsed_jd.get('role_name')} (Required Exp: {parsed_jd.get('experience_required')} years)")
        logger.info(f"Must-have skills: {parsed_jd.get('must_have_skills')}")

        # 2. Loading Candidates
        logger.info("Step 2: Loading Candidates...")
        candidates = CandidateParser.load_candidates(candidates_file)
        if not candidates:
            raise ValueError("No candidates found in the dataset.")

        # 3. Generating Embeddings for Candidates
        logger.info("Step 3: Generating embeddings for candidates...")
        cand_texts = [CandidateParser.construct_embedding_text(c) for c in candidates]
        cand_embeddings = self.embedder.get_embeddings(cand_texts)
        cand_ids = [c["candidate_id"] for c in candidates]

        # 4. Semantic Retrieval via FAISS
        logger.info("Step 4: Indexing candidates and running FAISS semantic retrieval...")
        dimension = cand_embeddings.shape[1]
        retriever = FAISSRetriever(dimension)
        retriever.add_candidates(cand_embeddings, cand_ids)
        
        # Embed Job Description
        jd_embedding_text = f"Role: {parsed_jd.get('role_name')} | Core Skills: {', '.join(parsed_jd.get('must_have_skills', []))} | Nice to have: {', '.join(parsed_jd.get('nice_to_have_skills', []))} | Responsibilities: {', '.join(parsed_jd.get('responsibilities', []))}"
        jd_vector = self.embedder.get_embedding(jd_embedding_text)
        
        # Retrieve top candidates semantically
        retrieved_results = retriever.retrieve(jd_vector, top_n=TOP_N_RETRIEVAL)
        retrieved_ids = [res[0] for res in retrieved_results]
        retrieved_semantic_scores = {res[0]: res[1] for res in retrieved_results}
        
        # Filter candidate objects
        retrieved_candidates = [c for c in candidates if c["candidate_id"] in retrieved_ids]

        # 5. Hybrid Scoring
        logger.info("Step 5: Computing hybrid scores...")
        hybrid_results = []
        for candidate in retrieved_candidates:
            cand_id = candidate["candidate_id"]
            semantic_score = retrieved_semantic_scores[cand_id]
            req_years = parsed_jd.get("experience_required", 3)
            
            scores = self.scorer.score_candidate(candidate, semantic_score, req_years)
            hybrid_results.append((candidate, scores))
            
        # Sort candidates by hybrid score descending to prepare for LLM reranking
        hybrid_results.sort(key=lambda x: x[1]["hybrid_score"], reverse=True)

        # 6. LLM-Based Re-ranking & Explainability
        logger.info("Step 6: Executing LLM Re-ranking on candidates...")
        
        def process_candidate(item):
            candidate, scores = item
            cand_id = candidate["candidate_id"]
            logger.info(f"Reranking candidate {cand_id} ({candidate.get('name')})...")
            
            # Run deep LLM analysis
            llm_evaluation = self.reranker.rerank_candidate(candidate, parsed_jd)
            
            # Combine Hybrid Score and LLM Score
            hybrid_score_scaled = scores["hybrid_score"] * 100.0
            llm_score = float(llm_evaluation.get("llm_score", 50))
            
            # Weighted combined score: 50% Hybrid, 50% LLM
            combined_score = (0.50 * hybrid_score_scaled) + (0.50 * llm_score)
            
            return {
                "candidate_id": cand_id,
                "name": candidate.get("name"),
                "experience_years": candidate.get("experience_years"),
                "skills": candidate.get("skills", []),
                "hybrid_score_scaled": round(hybrid_score_scaled, 2),
                "semantic_score": round(scores["semantic_score"] * 100.0, 2),
                "experience_score": round(scores["experience_score"] * 100.0, 2),
                "behavior_score": round(scores["behavior_score"] * 100.0, 2),
                "activity_score": round(scores["activity_score"] * 100.0, 2),
                "llm_score": llm_score,
                "score": round(combined_score, 2),
                "recommendation": llm_evaluation.get("recommendation", "Hire"),
                "explanation": llm_evaluation.get("explanation", ""),
                "strengths": llm_evaluation.get("strengths", []),
                "gaps": llm_evaluation.get("gaps", []),
                "skill_gap_analysis": llm_evaluation.get("skill_gap_analysis", {}),
                "confidence_score": llm_evaluation.get("confidence_score", 0.8),
                "is_cold_start": scores["is_cold_start"]
            }

        # Run concurrently with up to 10 threads to avoid Render timeout (which occurs ~60s)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            final_ranked_list = list(executor.map(process_candidate, hybrid_results))

        # 7. Final Sort
        logger.info("Step 7: Sorting and exporting final ranked list...")
        final_ranked_list.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank field
        for rank, cand in enumerate(final_ranked_list, 1):
            cand["rank"] = rank

        # 8. Export to CSV
        self.export_csv(final_ranked_list, output_csv)
        
        return final_ranked_list

    def export_csv(self, ranked_list: List[Dict[str, Any]], filepath: Path):
        """
        Exports the ranked shortlist to a CSV file.
        CSV columns: candidate_id | rank | score | explanation
        We also add secondary fields for completeness.
        """
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["candidate_id", "rank", "score", "explanation", "name", "recommendation", "strengths", "gaps"])
                
                for cand in ranked_list:
                    strengths_str = " | ".join(cand.get("strengths", []))
                    gaps_str = " | ".join(cand.get("gaps", []))
                    
                    writer.writerow([
                        cand["candidate_id"],
                        cand["rank"],
                        cand["score"],
                        cand["explanation"],
                        cand["name"],
                        cand["recommendation"],
                        strengths_str,
                        gaps_str
                    ])
            logger.info(f"Successfully exported ranked shortlist to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")

if __name__ == "__main__":
    # Test script
    sample_jd = """
    We are looking for a Senior Machine Learning Engineer with 5+ years of experience.
    Must have skills: Python, PyTorch, FAISS, scikit-learn, and NLP.
    Nice to have skills: AWS, Docker, Kubernetes.
    Responsibilities include building and scaling semantic search vector databases and deploying transformer models.
    We need a self-starter with strong communication skills who has startup experience.
    """
    pipeline = CandidateRankingPipeline()
    results = pipeline.run(sample_jd)
    print("\n--- TOP 3 RANKED CANDIDATES ---")
    for r in results[:3]:
        print(f"Rank {r['rank']}: {r['candidate_id']} - {r['name']} | Score: {r['score']} | Rec: {r['recommendation']}")
