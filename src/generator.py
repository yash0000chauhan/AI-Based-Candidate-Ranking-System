import json
import random
from pathlib import Path
from src.config import DATA_DIR

# Seed for reproducibility
random.seed(42)

NAMES = [
    "Alex Rivera", "Priya Sharma", "David Chen", "Sarah Jenkins", "Michael Mbanefo",
    "Elena Rostova", "Yuki Tanaka", "Carlos Gomez", "Amina Diop", "James Wilson",
    "Emily Taylor", "Arjun Patel", "Sofia Bianchi", "Lucas Silva", "Fatima Al-Sayed",
    "John Doe", "Jane Smith", "Robert Johnson", "Patricia Williams", "Thomas Brown",
    "Barbara Davis", "William Miller", "Linda Wilson", "Richard Moore", "Elizabeth Taylor",
    "Joseph Anderson", "Susan Thomas", "Charles Jackson", "Jessica White", "Christopher Harris",
    "Karen Martin", "Matthew Thompson", "Nancy Garcia", "Daniel Martinez", "Lisa Robinson",
    "Paul Clark", "Betty Rodriguez", "Mark Lewis", "Margaret Lee", "Donald Walker",
    "Sandra Hall", "George Allen", "Ashley Young", "Kenneth Hernandez", "Dorothy King",
    "Steven Wright", "Kimberly Lopez", "Edward Hill", "Emily Scott", "Ronald Green"
]

SKILLS_POOL = {
    "frontend": ["React", "TypeScript", "JavaScript", "HTML5", "CSS3", "Next.js", "Redux", "TailwindCSS", "Vue.js", "Angular", "Webflow"],
    "backend": ["Python", "Node.js", "Go", "Java", "Django", "FastAPI", "Express", "PostgreSQL", "MongoDB", "Redis", "GraphQL", "Spring Boot"],
    "ml_ai": ["Python", "PyTorch", "TensorFlow", "scikit-learn", "FAISS", "Hugging Face", "NLP", "LLMs", "Pandas", "NumPy", "OpenCV", "LangChain"],
    "devops": ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform", "GitHub Actions", "Linux", "GCP", "Ansible", "Prometheus"],
    "fullstack": ["React", "Node.js", "TypeScript", "PostgreSQL", "Express", "MongoDB", "Redux", "GraphQL", "TailwindCSS", "Next.js"]
}

COMPANIES = ["TechCorp", "InnovateSoft", "DataFlow", "Stripe", "ByteDance", "CloudScale", "AILabs", "WebWorks", "FintechPro"]

ROLE_TEMPLATES = [
    {
        "type": "ml_ai",
        "title": "Machine Learning Engineer",
        "primary_skills": ["Python", "PyTorch", "scikit-learn", "Pandas"],
        "experience_templates": [
            "Developed and deployed deep learning models for image recognition, improving classification accuracy by 15%. Worked on data preprocessing pipelines using Pandas and NumPy.",
            "Built NLP classification models for customer sentiment analysis using Hugging Face and PyTorch. Configured FAISS indexes for real-time document search.",
            "Led a team to scale generative AI pipelines using LangChain and LLMs. Optimised inference speed using TensorRT.",
            "Researched and implemented anomaly detection models for transaction fraud detection using scikit-learn and XGBoost."
        ],
        "project_templates": [
            "SemanticSearch: Built a semantic search engine over 10M records using FAISS and SentenceTransformers, reducing search latency to under 50ms.",
            "DocQA: Implemented a document question-answering system using llama-index and OpenAI APIs with a customized retrieval-augmented generation (RAG) pipeline.",
            "RecommenderSystem: Built a collaborative filtering recommendation engine using PyTorch that boosted click-through rates by 22%."
        ]
    },
    {
        "type": "frontend",
        "title": "Senior Frontend Developer",
        "primary_skills": ["React", "TypeScript", "Next.js", "TailwindCSS"],
        "experience_templates": [
            "Rebuilt the core customer portal using React and TypeScript, improving Core Web Vitals scores by 40%. Implemented complex state management using Redux Toolkit.",
            "Designed and developed a reusable component library used by 5 product teams, styling components using CSS modules and TailwindCSS.",
            "Architected a Next.js server-side rendered application, integrating with a headless CMS and optimizing SEO elements.",
            "Developed interactive dashboards with rich data visualizations using D3.js and React."
        ],
        "project_templates": [
            "UI-Kit: Built an open-source accessible React component library conforming to WAI-ARIA standards, garnering 2k+ stars on GitHub.",
            "SaaS-Dashboard: Created a real-time monitoring dashboard for server metrics using Next.js, Tailwind, and WebSockets.",
            "E-Commerce-Frontend: Implemented a highly optimized e-commerce frontend with instant search, client-side caching, and responsive design."
        ]
    },
    {
        "type": "backend",
        "title": "Backend Software Engineer",
        "primary_skills": ["Python", "Node.js", "FastAPI", "PostgreSQL"],
        "experience_templates": [
            "Designed high-throughput RESTful APIs using FastAPI and Node.js. Maintained database migrations and performance optimizations in PostgreSQL.",
            "Created microservices orchestration layer, reducing response times by 30% through caching with Redis and async task queues with Celery.",
            "Migrated a monolithic legacy application to a clean hexagonal architecture using Django and PostgreSQL.",
            "Integrated multiple third-party payment gateways (Stripe, PayPal) and ensured PCI-DSS compliance."
        ],
        "project_templates": [
            "API-Gateway: Designed an API gateway with rate-limiting, authentication, and request routing in Go, handling 10k requests/sec.",
            "DataSync-Service: Created an event-driven background synchronisation service processing millions of messages daily using Kafka and FastAPI.",
            "GraphQL-Server: Developed a flexible GraphQL API for client applications, optimizing SQL queries and resolving N+1 database queries."
        ]
    },
    {
        "type": "fullstack",
        "title": "Full Stack Engineer",
        "primary_skills": ["React", "Node.js", "TypeScript", "PostgreSQL"],
        "experience_templates": [
            "Owned end-to-end features for a high-growth SaaS platform. Written backend Node.js endpoints and implemented matching React user interfaces.",
            "Developed interactive web portals using React, Express, and PostgreSQL. Set up robust CI/CD pipelines using GitHub Actions.",
            "Integrated frontend React forms with backend Django services, ensuring input sanitisation and strict validation.",
            "Built a collaborative document editing tool using React, Node.js, WebSockets, and operational transformation algorithms."
        ],
        "project_templates": [
            "CollaborateApp: A full-stack Trello clone with drag-and-drop interfaces, real-time updates, and PostgreSQL database.",
            "InvoiceManager: A billing app for freelancers supporting PDF generation, automated email reminders, and Stripe checkouts.",
            "JobBoard: Built a niche job search portal with full-text search, user auth, and email notifications."
        ]
    }
]

def generate_candidates(count=200):
    candidates = []
    
    for i in range(count):
        candidate_id = f"CAND_{i+1:07d}"
        name = NAMES[i % len(NAMES)]
        
        # Pick a career profile type
        profile_template = random.choice(ROLE_TEMPLATES)
        p_type = profile_template["type"]
        
        # Determine experience years (skewed between 1 and 15)
        experience_years = random.randint(1, 15)
        
        # Select skills based on profile type
        skills = list(profile_template["primary_skills"])
        # Add random skills from the pool
        secondary_skill_pool = SKILLS_POOL[p_type] + SKILLS_POOL[random.choice(list(SKILLS_POOL.keys()))]
        # Remove duplicates
        secondary_skill_pool = list(set(secondary_skill_pool) - set(skills))
        extra_skills = random.sample(secondary_skill_pool, k=random.randint(2, 5))
        skills.extend(extra_skills)
        
        # Generate experience details based on years
        exp_entries = []
        num_jobs = max(1, experience_years // 3)
        current_year = 2026
        
        for j in range(num_jobs):
            job_years = max(1, experience_years // num_jobs)
            start_yr = current_year - job_years
            company = random.choice(COMPANIES)
            desc = random.choice(profile_template["experience_templates"])
            exp_entries.append(f"{start_yr}-{current_year}: Software Engineer at {company} - {desc}")
            current_year = start_yr
            
        experience_details = " | ".join(exp_entries)
        
        # Select 1-2 projects
        projects_count = random.randint(1, 2)
        candidate_projects = random.sample(profile_template["project_templates"], k=projects_count)
        projects_str = " | ".join(candidate_projects)
        
        # Behavioral signals (with some variation)
        # We also create a few "problematic" or "passive" candidates to test scoring
        candidate_behavior_profile = random.choice(["active", "passive", "unresponsive", "cold_start"])
        
        if candidate_behavior_profile == "active":
            response_rate = round(random.uniform(0.85, 1.0), 2)
            engagement_score = round(random.uniform(0.80, 1.0), 2)
            interview_attendance = round(random.uniform(0.90, 1.0), 2)
            last_active_days = random.randint(1, 7)
            profile_completeness = round(random.uniform(0.90, 1.0), 2)
            contributions_count = random.randint(30, 150)
        elif candidate_behavior_profile == "passive":
            response_rate = round(random.uniform(0.60, 0.85), 2)
            engagement_score = round(random.uniform(0.40, 0.70), 2)
            interview_attendance = round(random.uniform(0.80, 0.95), 2)
            last_active_days = random.randint(8, 30)
            profile_completeness = round(random.uniform(0.75, 0.90), 2)
            contributions_count = random.randint(5, 50)
        elif candidate_behavior_profile == "unresponsive":
            response_rate = round(random.uniform(0.10, 0.40), 2)
            engagement_score = round(random.uniform(0.10, 0.30), 2)
            interview_attendance = round(random.uniform(0.40, 0.70), 2)
            last_active_days = random.randint(31, 90)
            profile_completeness = round(random.uniform(0.50, 0.75), 2)
            contributions_count = random.randint(0, 10)
        else:  # cold_start
            response_rate = 0.0
            engagement_score = 0.0
            interview_attendance = 1.0  # default
            last_active_days = 999  # very old or never active
            profile_completeness = round(random.uniform(0.60, 0.80), 2)
            contributions_count = 0
            
        candidate_data = {
            "candidate_id": candidate_id,
            "name": name,
            "skills": skills,
            "experience_years": experience_years,
            "experience_details": experience_details,
            "projects": projects_str,
            "behavioral_signals": {
                "response_rate": response_rate,
                "engagement_score": engagement_score,
                "interview_attendance": interview_attendance
            },
            "activity_metrics": {
                "last_active_days": last_active_days,
                "profile_completeness": profile_completeness,
                "contributions_count": contributions_count
            }
        }
        candidates.append(candidate_data)
        
    return candidates

def generate_and_save_data():
    candidates = generate_candidates(200)
    output_path = DATA_DIR / "candidates.jsonl"
    with open(output_path, "w") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate) + "\n")
    print(f"Generated 200 candidates and saved to {output_path}")

if __name__ == "__main__":
    generate_and_save_data()
