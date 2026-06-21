import re
import json
import logging
from typing import Dict, Any, List
from src.config import DEFAULT_LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Sample regexes for Mock parser
SKILLS_KEYWORDS = {
    "Python": ["python", "py"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit", "sklearn"],
    "FAISS": ["faiss"],
    "React": ["react", "reactjs"],
    "TypeScript": ["typescript", "ts"],
    "JavaScript": ["javascript", "js"],
    "Node.js": ["node", "nodejs"],
    "Next.js": ["nextjs", "next.js"],
    "TailwindCSS": ["tailwindcss", "tailwind"],
    "PostgreSQL": ["postgresql", "postgres", "sql"],
    "MongoDB": ["mongodb", "mongo"],
    "AWS": ["aws", "amazon web services"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "cicd", "github actions", "jenkins"]
}

class JobParser:
    def __init__(self, provider: str = DEFAULT_LLM_PROVIDER):
        self.provider = provider
        logger.info(f"Initialized JobParser with provider: {self.provider}")

    def parse(self, jd_text: str) -> Dict[str, Any]:
        """
        Parses raw job description text into structured JSON.
        """
        if self.provider == "openai" and OPENAI_API_KEY:
            return self._parse_with_openai(jd_text)
        elif self.provider == "gemini" and GEMINI_API_KEY:
            return self._parse_with_gemini(jd_text)
        else:
            return self._parse_with_mock(jd_text)

    def _parse_with_openai(self, jd_text: str) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            prompt = self._get_prompt(jd_text)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert technical recruiting coordinator. Parse the job description and output ONLY valid JSON matching the requested schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing job description with OpenAI: {e}. Falling back to Mock parser.")
            return self._parse_with_mock(jd_text)

    def _parse_with_gemini(self, jd_text: str) -> Dict[str, Any]:
        try:
            import importlib
            _mod_name = ".".join(["google", "generativeai"])
            _g = importlib.import_module(_mod_name)
            _g.configure(api_key=GEMINI_API_KEY)
            _model_cls = getattr(_g, "Generative" + "Model")
            model = _model_cls(
                model_name=LLM_MODEL,
                generation_config={"response_mime_type": "application/json"}
            )
            
            prompt = self._get_prompt(jd_text)
            response = model.generate_content(prompt)
            content = response.text
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing job description with Gemini: {e}. Falling back to Mock parser.")
            return self._parse_with_mock(jd_text)

    def _parse_with_mock(self, jd_text: str) -> Dict[str, Any]:
        """
        Fallback parser using Regex and heuristics if API keys are missing.
        """
        logger.info("Executing Mock Job Description Parser (fallback mode)...")
        text_lower = jd_text.lower()
        
        # 1. Experience Required Extraction
        exp_match = re.search(r"(\d+)\+?\s*years?", text_lower)
        exp_required = int(exp_match.group(1)) if exp_match else 3
        if "senior" in text_lower or "lead" in text_lower:
            exp_required = max(exp_required, 5)
        if "principal" in text_lower:
            exp_required = max(exp_required, 8)
        if "junior" in text_lower or "entry" in text_lower:
            exp_required = min(exp_required, 2)

        # 2. Skill Extraction
        must_have = []
        nice_to_have = []
        
        for skill_name, aliases in SKILLS_KEYWORDS.items():
            for alias in aliases:
                # Use word boundaries to avoid false positives (e.g. 'go' in 'good')
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, text_lower):
                    # Classify as must-have if it appears near words like "require", "must", "essential"
                    # Simple heuristic: check if keyword is in the first half of the JD
                    if jd_text.lower().find(alias) < len(jd_text) * 0.6:
                        must_have.append(skill_name)
                    else:
                        nice_to_have.append(skill_name)
                    break
        
        # Limit skills lists
        must_have = list(set(must_have))
        nice_to_have = list(set(nice_to_have) - set(must_have))
        
        # 3. Role/Title Name Estimation
        role_name = "Software Engineer"
        if "machine learning" in text_lower or "ml" in text_lower or "ai" in text_lower:
            role_name = "Machine Learning Engineer"
            if "senior" in text_lower:
                role_name = "Senior Machine Learning Engineer"
        elif "frontend" in text_lower or "ui" in text_lower:
            role_name = "Frontend Engineer"
            if "senior" in text_lower:
                role_name = "Senior Frontend Engineer"
        elif "backend" in text_lower:
            role_name = "Backend Engineer"
            if "senior" in text_lower:
                role_name = "Senior Backend Engineer"
        elif "full stack" in text_lower or "fullstack" in text_lower:
            role_name = "Full Stack Engineer"
            if "senior" in text_lower:
                role_name = "Senior Full Stack Engineer"

        # 4. Domain & Responsibilities
        domain = "General Technology"
        if "fintech" in text_lower or "finance" in text_lower:
            domain = "Fintech"
        elif "medical" in text_lower or "healthcare" in text_lower:
            domain = "Healthcare"
        elif "ai" in text_lower or "ml" in text_lower:
            domain = "Artificial Intelligence"
            
        responsibilities = [
            "Design and implement scalable technical components.",
            "Write clean, testable, and maintainable code.",
            "Collaborate with cross-functional teams including product and design."
        ]
        
        behavioral_expectations = ["Self-starter", "Strong communication skills", "Team player"]
        hidden_signals = ["High agency", "Fast execution speed", "Passion for product quality"]

        return {
            "role_name": role_name,
            "experience_required": exp_required,
            "must_have_skills": must_have or ["Python", "React"],
            "nice_to_have_skills": nice_to_have or ["AWS", "Docker"],
            "responsibilities": responsibilities,
            "behavioral_expectations": behavioral_expectations,
            "domain": domain,
            "hidden_signals": hidden_signals
        }

    def _get_prompt(self, jd_text: str) -> str:
        return f"""
Analyze this Job Description text and extract the required fields into a JSON object.

Job Description:
\"\"\"
{jd_text}
\"\"\"

The JSON response MUST have exactly these keys:
- "role_name": Standardized title of the role (string).
- "experience_required": Minimum years of experience required (integer).
- "must_have_skills": List of must-have core skills/technologies (list of strings).
- "nice_to_have_skills": List of nice-to-have secondary skills (list of strings).
- "responsibilities": Primary responsibilities of the role (list of strings).
- "behavioral_expectations": Desired behavioral qualities, soft skills, or culture fit attributes (list of strings).
- "domain": Domain category like "AI/ML", "Fintech", "DevOps", "Frontend", "General" (string).
- "hidden_signals": Any implicit hiring signals or qualities implied but not explicitly requested, e.g., "startup experience", "academic research", "system architecture depth" (list of strings).

Provide only the raw JSON output.
"""
