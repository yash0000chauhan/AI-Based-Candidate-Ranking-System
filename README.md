# 🌌 AURA: AI-Powered Candidate Ranking & Shortlisting System

AURA is a next-generation recruiting system designed to replace outdated keyword-based applicant tracking systems (ATS). By combining **semantic vector search**, **heuristic hybrid scoring (experience + behavior + activity)**, and **LLM re-ranking (GPT-4o / Gemini)**, AURA evaluates candidates holistically and provides natural language explanations for its recommendations—just like an expert human recruiter.

---

## 🚀 Key Features

- **Deep Job Understanding**: Uses an LLM to extract must-have skills, secondary skills, experience targets, responsibilities, culture fit criteria, and hidden hiring signals from raw text.
- **Holistic Candidate Representation**: Maps skills, experience details, projects, behavioral signals (response rate, engagement), and activity metrics (recency, git contributions) into a unified profile.
- **Semantic Retrieval (FAISS)**: Uses SentenceTransformers (`all-MiniLM-L6-v2`) to embed profiles and run high-speed cosine similarity search in a FAISS index.
- **Multi-Signal Hybrid Scorer**: Combines semantic alignment (40%), experience fit (20%), behavioral reliability (25%), and pipeline readiness (15%) using a custom weighting formula.
- **Experience Alignment Curve**: Evaluates years of experience using a non-linear bell curve/trapezoid that penalizes underqualification and decays slowly for overqualification (representing cost/retention risks).
- **LLM Re-Ranking (RAG)**: Processes the top FAISS candidates through deep LLM reasoning to evaluate skill depth, project complexity, and career progression, returning a final Match Rating (0-100).
- **Explainability Layer**: Generates natural language recruiter summaries, bulleted lists of strengths, warning gaps, and structured recommendations.
- **Bias Reduction (Anonymize)**: Toggles on-the-fly candidate anonymization to mask names and gender pronouns from both the LLM prompts and the user interface.
- **Cold-Start Handling**: Detects fresh or passive profiles with no behavioral logs and automatically redistributes weights (60% Semantic, 40% Experience) to ensure fair scoring.
- **Interactive UI Dashboard**: A premium, glassmorphic dark-mode web application featuring preset job templates, auto-balancing weight sliders, live pipeline execution console, and detailed candidate fit analysis cards.

---

## 🏗️ System Architecture & Scoring Formulas

### 1. Hybrid Scoring Formula
Each candidate's hybrid score ($S_{\text{hybrid}}$) is computed as follows:

$$S_{\text{hybrid}} = w_s \cdot S_{\text{semantic}} + w_e \cdot S_{\text{experience}} + w_b \cdot S_{\text{behavior}} + w_a \cdot S_{\text{activity}}$$

Where the default weights are:
- $w_s = 0.40$ (Semantic Vector Similarity)
- $w_e = 0.20$ (Experience Fit Score)
- $w_b = 0.25$ (Behavioral Score)
- $w_a = 0.15$ (Activity Score)

### 2. Experience Fit Function ($S_{\text{experience}}$)
Calculated relative to required job years ($Y_{\text{req}}$):
- **Under-qualified** ($Y < Y_{\text{req}}$): 
  $$S_{\text{experience}} = \left(\frac{Y}{Y_{\text{req}}}\right)^{1.2}$$
- **Optimal Range** ($Y_{\text{req}} \le Y \le Y_{\text{req}} + 4$): 
  $$S_{\text{experience}} = 1.0$$
- **Over-qualified** ($Y > Y_{\text{req}} + 4$):
  $$S_{\text{experience}} = \max\left(0.70, 1.0 - 0.03 \cdot (Y - [Y_{\text{req}} + 4])\right)$$

### 3. Behavioral Reliability ($S_{\text{behavior}}$)
$$S_{\text{behavior}} = 0.50 \cdot \text{response\_rate} + 0.30 \cdot \text{interview\_attendance} + 0.20 \cdot \text{engagement\_score}$$

### 4. Pipeline Activity ($S_{\text{activity}}$)
$$S_{\text{activity}} = 0.50 \cdot \text{recency\_score} + 0.30 \cdot \text{profile\_completeness} + 0.20 \cdot \min\left(1.0, \frac{\text{contributions}}{50}\right)$$
*(where $\text{recency\_score}$ decays linearly to 0 over 60 inactive days)*

### 5. Final Combined Score
Once candidates are retrieved and re-ranked, the final Match Rating ($S_{\text{final}}$) merges the hybrid heuristics and LLM evaluation:

$$S_{\text{final}} = 0.50 \cdot (S_{\text{hybrid}} \cdot 100) + 0.50 \cdot S_{\text{llm\_rerank}}$$

---

## 📁 Project Structure

```
ai-powered-resume-shortlisting/
│
├── data/
│   └── candidates.jsonl         # Generated candidate database (50 profiles)
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Weights, directory paths, models config
│   ├── generator.py             # Rich synthetic candidate profile generator
│   ├── job_parser.py            # Extracts structured details from JDs (LLM/Regex)
│   ├── candidate_parser.py      # Standardizes candidate formatting & anonymization
│   ├── embedder.py              # SentenceTransformers vector generation
│   ├── retriever.py             # FAISS indexing and Cosine similarity retrieval
│   ├── scorer.py                # Multi-signal hybrid scoring algorithms
│   ├── reranker.py              # LLM re-ranking reasoning (OpenAI/Gemini/Mock)
│   └── pipeline.py              # Pipeline orchestrator & CSV exporter
│
├── templates/
│   └── index.html               # Premium glassmorphic dashboard SPA UI
│
├── output/
│   └── shortlist.csv            # Generated ranked candidate list output
│
├── app.py                       # FastAPI server exposing REST APIs & UI
├── requirements.txt             # Python packages list
└── README.md                    # System documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository and navigate inside:
```bash
cd "ai powerd resume sortlisting"
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set up your LLM API Keys in a `.env` file:
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```
*Note: If no API keys are provided, AURA automatically defaults to **Local Heuristic Mode**, which uses local SentenceTransformers and simulated LLM prompts. The dashboard is 100% functional without internet connectivity or API credits.*

---

## 🚀 Running the System

### 1. Generate the Candidate Database:
```bash
python -m src.generator
```
This generates 50 diverse candidates (React devs, ML Engineers, DevOps, etc.) in `data/candidates.jsonl`.

### 2. Start the FastAPI Web Server:
```bash
python app.py
```
Uvicorn will boot up the server at `http://127.0.0.1:8000`.

### 3. Open the Dashboard:
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Click on any of the Job Description presets (e.g., **ML Eng**, **Backend**, **Frontend**) to load a sample job description.
- Adjust weights or toggle anonymization.
- Click **Parse & Rank Candidates**.
- Inspect candidate score breakdowns, click **Analyze Fit** to see the recruiter evaluation modal, and click **Export Shortlist (CSV)** to download the results.

---

## 📊 API Reference

### `POST /api/rank`
Triggers the search and re-ranking pipeline.
- **Request Body**:
  ```json
  {
    "jd_text": "We need a Python developer with 3+ years experience...",
    "openai_key": "optional-key-override",
    "gemini_key": "optional-key-override",
    "weight_semantic": 0.40,
    "weight_experience": 0.20,
    "weight_behavioral": 0.25,
    "weight_activity": 0.15,
    "anonymize": false,
    "handle_cold_start": true
  }
  ```
- **Response**: Returns a JSON object containing the parsed job description specifications and a sorted list of the Top 15 candidates including rankings, sub-scores, strengths, gaps, and recruiter notes.

### `GET /api/presets`
Returns predefined job description templates.

### `GET /api/download`
Downloads the CSV file representing the shortlist generated during the last run. Columns include:
`candidate_id | rank | score | explanation | name | recommendation | strengths | gaps`
