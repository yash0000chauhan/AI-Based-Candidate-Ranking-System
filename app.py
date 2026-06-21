import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Ensure project root is in system path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.pipeline import CandidateRankingPipeline
from src.config import OUTPUT_DIR, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AURA - AI Candidate Ranking System",
    description="Hackathon-ready AI-powered recruitment shortlisting and explainability engine.",
    version="1.0.0"
)

# In-memory store for the last run results (to serve CSV downloads easily)
LATEST_RUN_PATH = OUTPUT_DIR / "shortlist.csv"

# Predefined Job Description Presets
JOB_PRESETS = {
    "ml_engineer": """We are seeking a Senior Machine Learning Engineer with 5+ years of experience to join our Core AI team.
    
Must-have skills: Python, PyTorch, FAISS, scikit-learn, NLP, LLMs.
Nice-to-have skills: AWS, Docker, Kubernetes, LangChain.

Responsibilities:
- Build and optimize large-scale semantic search vector databases.
- Develop, evaluate, and deploy Transformer-based NLP models.
- Collaborate with backend engineers to integrate AI APIs into high-throughput SaaS pipelines.
- Research and implement state-of-the-art retrieval-augmented generation (RAG) architectures.

We value high agency, strong communication skills, and candidates who thrive in fast-paced startup environments.""",

    "frontend_developer": """We are looking for a Senior Frontend Engineer (Next.js & React) with 4+ years of professional experience.

Must-have skills: React, TypeScript, Next.js, TailwindCSS, Redux.
Nice-to-have skills: Vue.js, GraphQL, Webflow, Docker.

Responsibilities:
- Build beautiful, performant, and responsive web user interfaces.
- Transition legacy UI elements to a modern reusable component library.
- Optimize site performance, targeting 90+ lighthouse scores and excellent Core Web Vitals.
- Collaborate with product designers to implement interactive data dashboards and charts.

We are looking for detail-oriented self-starters with a strong passion for product design and rich micro-animations.""",

    "backend_engineer": """Join our infrastructure team as a Lead Backend Software Engineer. We require 6+ years of backend development experience.

Must-have skills: Python, Node.js, FastAPI, PostgreSQL, Redis.
Nice-to-have skills: Go, GraphQL, AWS, Docker, CI/CD.

Responsibilities:
- Architect high-throughput, low-latency RESTful APIs using FastAPI and Node.js.
- Optimize PostgreSQL database schemas, write complex indexes, and resolve N+1 queries.
- Build event-driven background job queues using Redis and Celery.
- Maintain and deploy services using Docker containerization and Kubernetes.

Requirements: Experience scaling applications to millions of active users and passion for high engineering standards.""",

    "fullstack_developer": """We are hiring a Full Stack Engineer with 3+ years of experience.

Must-have skills: React, Node.js, TypeScript, PostgreSQL, TailwindCSS.
Nice-to-have skills: Next.js, MongoDB, Express, AWS, CI/CD.

Responsibilities:
- Own product features end-to-end: from database schema to backend endpoints and frontend interfaces.
- Write clean, modular, and well-tested code across the stack.
- Set up automated testing and CI/CD deployment pipelines using GitHub Actions.
- Optimize database queries and frontend bundles for speed and performance.

Excellent team collaboration and problem-solving skills are essential."""
}

class RankRequest(BaseModel):
    jd_text: str
    openai_key: Optional[str] = ""
    gemini_key: Optional[str] = ""
    weight_semantic: float = 0.40
    weight_experience: float = 0.20
    weight_behavioral: float = 0.25
    weight_activity: float = 0.15
    anonymize: bool = False
    handle_cold_start: bool = True

@app.get("/api/presets")
def get_presets():
    """Returns the pre-configured job description templates."""
    return JOB_PRESETS

@app.post("/api/rank")
async def run_ranking_pipeline(payload: RankRequest):
    """
    Triggers the AI Ranking and retrieval pipeline.
    Optionally configures API keys for OpenAI / Gemini on the fly.
    """
    if not payload.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    # Configure API keys temporarily if provided
    original_openai = os.environ.get("OPENAI_API_KEY")
    original_gemini = os.environ.get("GEMINI_API_KEY")

    if payload.openai_key:
        os.environ["OPENAI_API_KEY"] = payload.openai_key
    if payload.gemini_key:
        os.environ["GEMINI_API_KEY"] = payload.gemini_key

    # Overwrite weights in config module dynamically
    import src.config as cfg
    cfg.WEIGHT_SEMANTIC = payload.weight_semantic
    cfg.WEIGHT_EXPERIENCE = payload.weight_experience
    cfg.WEIGHT_BEHAVIORAL = payload.weight_behavioral
    cfg.WEIGHT_ACTIVITY = payload.weight_activity

    # Decide provider based on keys
    provider = "mock"
    if payload.openai_key or os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
        cfg.DEFAULT_LLM_PROVIDER = "openai"
        cfg.LLM_MODEL = "gpt-4o-mini"
    elif payload.gemini_key or os.environ.get("GEMINI_API_KEY"):
        provider = "gemini"
        cfg.DEFAULT_LLM_PROVIDER = "gemini"
        cfg.LLM_MODEL = "gemini-1.5-flash"
    else:
        provider = "mock"
        cfg.DEFAULT_LLM_PROVIDER = "mock"
        cfg.LLM_MODEL = "mock-evaluator"

    try:
        pipeline = CandidateRankingPipeline(provider=provider)
        
        # Override candidates database if anonymize is enabled
        candidates_file = DATA_DIR / "candidates.jsonl"
        
        # If anonymization is active, we'll temporarily create anonymized candidate dataset
        if payload.anonymize:
            from src.candidate_parser import CandidateParser
            original_candidates = CandidateParser.load_candidates(candidates_file)
            anon_candidates = [CandidateParser.anonymize(c) for c in original_candidates]
            
            # Save anonymized data
            temp_candidates_file = DATA_DIR / "candidates_anon.jsonl"
            import json
            with open(temp_candidates_file, "w", encoding="utf-8") as f:
                for cand in anon_candidates:
                    f.write(json.dumps(cand) + "\n")
            
            candidates_path = temp_candidates_file
        else:
            candidates_path = candidates_file

        results = pipeline.run(
            jd_text=payload.jd_text,
            candidates_file=candidates_path,
            output_csv=LATEST_RUN_PATH
        )
        
        # Clean up temp file if created
        if payload.anonymize and temp_candidates_file.exists():
            try:
                temp_candidates_file.unlink()
            except Exception:
                pass

        return {
            "status": "success",
            "provider_used": provider,
            "candidates": results,
            "parsed_jd": pipeline.job_parser.parse(payload.jd_text) # Show parsed JD variables in UI
        }
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Restore original keys
        if original_openai is not None:
            os.environ["OPENAI_API_KEY"] = original_openai
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            
        if original_gemini is not None:
            os.environ["GEMINI_API_KEY"] = original_gemini
        elif "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

@app.get("/api/download")
def download_shortlist():
    """Serves the latest ranked candidates CSV for download."""
    if not LATEST_RUN_PATH.exists():
        raise HTTPException(status_code=404, detail="Shortlist CSV not found. Please run the pipeline first.")
    return FileResponse(
        path=LATEST_RUN_PATH,
        media_type="text/csv",
        filename="aura_ranked_shortlist.csv"
    )

# Serve Frontend SPA
@app.get("/")
def serve_index():
    templates_dir = Path(__file__).resolve().parent / "templates"
    index_file = templates_dir / "index.html"
    
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    # Create templates dir if not exists
    templates_dir = Path(__file__).resolve().parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Run development server
    uvicorn.run(app, host="127.0.0.1", port=8000)
