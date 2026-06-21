import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they do not exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Scoring Weights
WEIGHT_SEMANTIC = 0.40
WEIGHT_EXPERIENCE = 0.20
WEIGHT_BEHAVIORAL = 0.25
WEIGHT_ACTIVITY = 0.15

# Embedding Config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, runs locally

# LLM Config
# Supports 'openai', 'gemini', or 'mock' (default if no keys are provided)
if OPENAI_API_KEY:
    DEFAULT_LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o-mini"
elif GEMINI_API_KEY:
    DEFAULT_LLM_PROVIDER = "gemini"
    LLM_MODEL = "gemini-1.5-flash"
else:
    DEFAULT_LLM_PROVIDER = "mock"
    LLM_MODEL = "mock-evaluator"

# Pipeline Config
TOP_N_RETRIEVAL = 15  # Number of candidates to retrieve via vector search for re-ranking
FINAL_SHORTLIST_SIZE = 10  # Number of candidates in final display

# Cold Start Weights (re-allocates weight if a candidate has no behavioral/activity records)
# If cold start is active, we distribute behavioral and activity weights to semantic and experience
COLD_START_WEIGHT_SEMANTIC = 0.60
COLD_START_WEIGHT_EXPERIENCE = 0.40
