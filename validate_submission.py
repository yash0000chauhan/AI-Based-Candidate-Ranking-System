"""
AURA Hackathon Submission Validator
Audits final_submission.csv against Redrob Submission Spec v4
"""

import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter

SUBMISSION_CSV  = Path("output/final_submission.csv")
CANDIDATES_JSONL = Path("data/candidates.jsonl")

PASS_LIST = []
FAIL_LIST = []
WARN_LIST = []

def ok(tag, msg=""):
    PASS_LIST.append(tag)
    print(f"  ✅  {tag}" + (f"  →  {msg}" if msg else ""))

def fail(tag, msg=""):
    FAIL_LIST.append(tag)
    print(f"  ❌  {tag}" + (f"  →  {msg}" if msg else ""))

def warn(tag, msg=""):
    WARN_LIST.append(tag)
    print(f"  ⚠️   {tag}" + (f"  →  {msg}" if msg else ""))

# ─── Load submission ──────────────────────────────────────────────────────────
print("\n" + "═"*68)
print("  REDROB HACKATHON — STRICT SUBMISSION VALIDATOR")
print("═"*68)

rows = []
with open(SUBMISSION_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    for row in reader:
        if any(v.strip() for v in row.values()):   # skip blank trailing line
            rows.append(row)

# ─── Load ground-truth candidate IDs ─────────────────────────────────────────
valid_cand_ids = set()
with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            valid_cand_ids.add(obj["candidate_id"])

print(f"\n  Submission rows loaded  : {len(rows)}")
print(f"  Known candidate IDs    : {len(valid_cand_ids)}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("  SECTION 1 — CSV FORMAT VALIDATION")
print("─"*68)

# 1.1  Exactly 100 data rows
if len(rows) == 100:
    ok("Row count == 100")
else:
    fail("Row count == 100", f"found {len(rows)} rows")

# 1.2  Column names + order
REQUIRED_COLS = ["candidate_id", "rank", "score", "reasoning"]
if header == REQUIRED_COLS:
    ok("Column names & order correct", str(header))
else:
    fail("Column names & order", f"expected {REQUIRED_COLS}, got {header}")

# 1.3  candidate_id format  CAND_XXXXXXX  (7 digits)
bad_ids = [r["candidate_id"] for r in rows if not re.fullmatch(r"CAND_\d{7}", r["candidate_id"])]
if not bad_ids:
    ok("candidate_id format CAND_XXXXXXX")
else:
    fail("candidate_id format", f"{len(bad_ids)} bad IDs: {bad_ids[:5]}")

# 1.4  All candidate_ids exist in candidates.jsonl
unknown_ids = [r["candidate_id"] for r in rows if r["candidate_id"] not in valid_cand_ids]
if not unknown_ids:
    ok("All candidate_ids exist in candidates.jsonl")
else:
    fail("candidate_ids in candidates.jsonl", f"{len(unknown_ids)} unknown: {unknown_ids[:5]}")

# 1.5  Unique candidate_ids
cid_counts = Counter(r["candidate_id"] for r in rows)
dupe_cids = [cid for cid, n in cid_counts.items() if n > 1]
if not dupe_cids:
    ok("No duplicate candidate_ids")
else:
    fail("Duplicate candidate_ids", str(dupe_cids[:5]))

# 1.6  rank is integer 1-100 each exactly once
try:
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) == list(range(1, 101)):
        ok("Ranks 1-100, each exactly once")
    else:
        missing = set(range(1, 101)) - set(ranks)
        dupes   = [k for k, v in Counter(ranks).items() if v > 1]
        fail("Ranks 1-100 complete & unique",
             f"missing={sorted(missing)[:5]}  dupes={dupes[:5]}")
except ValueError as e:
    fail("Rank is integer", str(e))

# 1.7  score is float
bad_scores = []
scores_by_rank = {}
for r in rows:
    try:
        s = float(r["score"])
        scores_by_rank[int(r["rank"])] = s
    except ValueError:
        bad_scores.append(r["candidate_id"])

if not bad_scores:
    ok("All scores are valid floats")
else:
    fail("Score is float", f"bad: {bad_scores[:5]}")

# 1.8  Score monotonically non-increasing with rank
violations = []
for rnk in range(1, 100):
    s_curr = scores_by_rank.get(rnk)
    s_next = scores_by_rank.get(rnk + 1)
    if s_curr is not None and s_next is not None and s_curr < s_next:
        violations.append((rnk, s_curr, rnk+1, s_next))

if not violations:
    ok("Scores monotonically non-increasing ✓")
else:
    fail("Score monotonically non-increasing",
         f"{len(violations)} violations: {violations[:3]}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("  SECTION 2 — LOGICAL CONSISTENCY")
print("─"*68)

score_vals = [float(r["score"]) for r in rows]
unique_scores = len(set(score_vals))
score_range   = max(score_vals) - min(score_vals)

print(f"  Score range     : {min(score_vals):.4f} → {max(score_vals):.4f}")
print(f"  Score spread    : {score_range:.4f}")
print(f"  Unique scores   : {unique_scores} / 100")

if unique_scores < 5:
    fail("Score differentiation", f"Only {unique_scores} unique values — looks like no differentiation")
elif unique_scores < 20:
    warn("Score differentiation", f"{unique_scores} unique values — low variation")
else:
    ok("Score differentiation", f"{unique_scores} unique values")

if score_range < 0.01:
    fail("Score spread", f"All scores nearly identical ({score_range:.6f} range)")
elif score_range < 0.05:
    warn("Score spread", f"Very narrow range ({score_range:.4f}) — may signal random scoring")
else:
    ok("Score spread", f"{score_range:.4f} range")

# Consecutive tie pairs
tie_pairs = sum(1 for i in range(len(score_vals)-1) if score_vals[i] == score_vals[i+1])
if tie_pairs > 10:
    warn("Tied scores", f"{tie_pairs} consecutive tied pairs — verify tie-breaking is deterministic")
else:
    ok("Tied score count", f"{tie_pairs} consecutive ties")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("  SECTION 3 — REASONING QUALITY (STAGE 4 CRITICAL)")
print("─"*68)

SAMPLE_RANKS = [1, 2, 5, 10, 20, 30, 50, 70, 85, 100]
sample_rows  = {int(r["rank"]): r for r in rows if int(r["rank"]) in SAMPLE_RANKS}

reasoning_texts = [r["reasoning"] for r in rows]
missing_reasoning = sum(1 for t in reasoning_texts if not t.strip())

if missing_reasoning == 0:
    ok("All 100 rows have reasoning")
else:
    fail("Missing reasoning", f"{missing_reasoning} rows have empty reasoning")

# Check for template patterns
TEMPLATE_PHRASES = [
    "candidate demonstrates",
    "this candidate has",
    "with experience in",
]
template_hits = sum(
    1 for t in reasoning_texts
    if any(ph in t.lower() for ph in TEMPLATE_PHRASES)
)
if template_hits > 80:
    fail("Templated reasoning", f"{template_hits}/100 rows match generic template phrases")
elif template_hits > 40:
    warn("Templated reasoning", f"{template_hits}/100 rows have near-template language")
else:
    ok("Template detection", f"Only {template_hits}/100 match generic patterns")

# Identical reasoning check
reason_counts = Counter(reasoning_texts)
identical_count = sum(v for v in reason_counts.values() if v > 1)
if identical_count > 0:
    fail("Unique reasoning", f"{identical_count} rows share identical reasoning strings")
else:
    ok("All reasoning strings unique")

# Length check
short = [(int(r["rank"]), r["reasoning"]) for r in rows if len(r["reasoning"].strip()) < 30]
if short:
    warn("Reasoning length", f"{len(short)} rows have very short reasoning (<30 chars): ranks {[x[0] for x in short[:5]]}")
else:
    ok("Reasoning length", "All ≥30 chars")

# Specificity checks on sampled rows
print(f"\n  Sampling ranks: {SAMPLE_RANKS}")
print()

specificity_failures = []
jd_connection_failures = []
concern_noted = []
rank_tone_mismatch = []

for rnk in SAMPLE_RANKS:
    row = sample_rows.get(rnk)
    if not row:
        warn(f"Rank {rnk} not found", "")
        continue

    txt = row["reasoning"]
    cid = row["candidate_id"]
    print(f"  Rank {rnk:3d} | {cid}")
    print(f"         \"{txt[:110]}...\"" if len(txt) > 110 else f"         \"{txt}\"")

    # 3a: Specific facts (years of experience, skills)
    has_years  = bool(re.search(r"\d+\.?\d*\s*(years?|yr)", txt, re.I))
    has_skills = bool(re.search(r"python|pytorch|faiss|nlp|rag|llm|embed|pgvector|search|transformer", txt, re.I))
    has_signals = bool(re.search(r"response rate|notice|engagement|github|signal", txt, re.I))

    if not (has_years and has_skills):
        specificity_failures.append(rnk)
        print(f"           ⚠️  Missing specific facts (years={has_years}, skills={has_skills})")
    else:
        print(f"           ✅ Specific facts OK (years, skills, signals={has_signals})")

    # 3b: Tone vs rank consistency
    positive_words = len(re.findall(r"strong|expert|highly|excellent|impressive|great|ideal|responsive", txt, re.I))
    caution_words  = len(re.findall(r"concern|gap|risk|weaker|limited|below|available.*\d{2,3}.day|response rate is [1-4]\d%", txt, re.I))

    if rnk <= 10 and caution_words > positive_words + 1:
        rank_tone_mismatch.append((rnk, "Top rank but overly cautious tone"))
        print(f"           ⚠️  Rank {rnk} but cautious tone detected")
    elif rnk >= 80 and positive_words > caution_words + 2:
        rank_tone_mismatch.append((rnk, "Low rank but very positive tone"))
        print(f"           ⚠️  Rank {rnk} but highly positive tone detected")
    else:
        print(f"           ✅ Tone consistent with rank")

    print()

if specificity_failures:
    warn("Reasoning specificity", f"Ranks {specificity_failures} may lack specific facts")
else:
    ok("Reasoning specificity", "All sampled rows have years + skills mentioned")

if rank_tone_mismatch:
    warn("Rank-tone consistency", f"{len(rank_tone_mismatch)} mismatches: {rank_tone_mismatch}")
else:
    ok("Rank-tone consistency", "Tone matches rank tier")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("  SECTION 4 — HONEYPOT RISK ESTIMATION")
print("─"*68)

# Look for suspicious profiles in top-20
# Signs of honeypots: very high experience (>15y), possibly in top ranks; or very low response rate in top 10
top10   = [r for r in rows if int(r["rank"]) <= 10]
top20   = [r for r in rows if int(r["rank"]) <= 20]

high_exp_top10 = [r for r in top10  if re.search(r"(\d{2,})\.?\d*\s*years", r["reasoning"])
                  and float(re.search(r"(\d+\.?\d*)\s*years", r["reasoning"]).group(1)) >= 15]
low_resp_top10 = [r for r in top10  if re.search(r"response rate is (\d+)%", r["reasoning"])
                  and int(re.search(r"response rate is (\d+)%", r["reasoning"]).group(1)) < 30]

print(f"  High-exp (≥15y) in top-10  : {len(high_exp_top10)} candidates")
print(f"  Low response (<30%) top-10 : {len(low_resp_top10)} candidates")

# Check for "16.9 years" (row 38 in the file – but this is rank 37, not top 10/20)
very_high_exp = [(int(r["rank"]), r["reasoning"]) for r in rows
                 if re.search(r"(1[5-9]|[2-9]\d)\.\d\s*years", r["reasoning"])]
high_exp_list = [(rnk, re.search(r"(\d+\.\d)\s*years", txt).group(1)) for rnk, txt in very_high_exp]
print(f"  Candidates >=15y experience  : {high_exp_list}")

if high_exp_top10:
    warn("Honeypot risk", "Suspicious high-exp candidates in top-10 — verify these aren't honeypots")
if low_resp_top10:
    warn("Honeypot risk", "Low response-rate candidates in top-10 — may lower stage-3 honeypot score")

# Keyword-matching indicator: are ALL top-10 reasoning texts mentioning 'information retrieval' or 'python'?
keyword_saturation = sum(1 for r in top10 if re.search(r"information retrieval|python", r["reasoning"], re.I))
if keyword_saturation == 10:
    warn("Keyword-matching risk", "All top-10 mention core JD keywords — pattern suggests embedding-only ranking")
else:
    ok("Keyword saturation", f"{keyword_saturation}/10 top-10 mention core JD keywords")

# Overall risk estimate
honeypot_risk = "LOW"
if high_exp_top10 or low_resp_top10:
    honeypot_risk = "MEDIUM"
if very_high_exp and any(rnk <= 20 for rnk, _ in very_high_exp):
    honeypot_risk = "HIGH"

print(f"\n  → Honeypot Risk Estimate: {honeypot_risk}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("  SECTION 5 — COMPUTE CONSTRAINT CHECK")
print("─"*68)

# Check rank.py for API calls or GPU usage
rank_py = Path("rank.py")
src_files = list(Path("src").glob("*.py"))
all_source = []
for fp in [rank_py] + src_files:
    if fp.exists():
        all_source.append((fp.name, fp.read_text(encoding="utf-8")))

api_patterns = [
    (r"openai\.ChatCompletion|openai\.chat\.completions", "OpenAI API call"),
    (r"anthropic\.|claude\.", "Anthropic API call"),
    (r"google\.generativeai|genai\.GenerativeModel", "Gemini API call"),
    (r"requests\.get|requests\.post|urllib\.request", "HTTP request"),
    (r"torch\.cuda|\.to\(['\"]cuda['\"]", "GPU usage"),
]

api_violations = []
for fname, src in all_source:
    for pat, label in api_patterns:
        matches = re.findall(pat, src)
        if matches:
            api_violations.append((fname, label, len(matches)))

if api_violations:
    for fname, label, n in api_violations:
        warn(f"Potential constraint violation", f"{fname}: {label} ({n} occurrence(s))")
else:
    ok("No external API calls or GPU usage found in source")

# Check per-candidate LLM reranking (pipeline.py calls reranker per candidate)
reranker_src = next((src for fname, src in all_source if fname == "reranker.py"), "")
if "provider" in reranker_src and "mock" in reranker_src:
    ok("LLM reranker has mock fallback (safe for CPU submission)")
else:
    warn("LLM reranker", "Verify reranker is NOT making live API calls per candidate")

# Check rank.py for LLM calls
rank_src = next((src for fname, src in all_source if fname == "rank.py"), "")
if rank_src:
    if re.search(r"openai|anthropic|gemini|requests\.(get|post)", rank_src):
        fail("rank.py API calls", "rank.py makes external API calls — WILL be caught at Stage 3")
    else:
        ok("rank.py is API-free")

print(f"\n  Compute budget assessment:")
print(f"  - Embedding model : all-MiniLM-L6-v2 (CPU-feasible ✅)")
print(f"  - FAISS retrieval : CPU (flat index, feasible ✅)")
print(f"  - Per-candidate   : scoring + mock reranker (no LLM API ✅)")
print(f"  - Est. runtime    : 50 candidates → ~2-3 min, 100K → est. 15-30 min ⚠️")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("  SECTION 6 — FINAL VERDICT")
print("═"*68)

print(f"\n  ✅ PASSED : {len(PASS_LIST)}")
print(f"  ❌ FAILED : {len(FAIL_LIST)}")
print(f"  ⚠️  WARNED : {len(WARN_LIST)}")

if FAIL_LIST:
    print("\n  ❌ CRITICAL FAILURES (auto-reject at Stage 1 / disqualify at Stage 3):")
    for f in FAIL_LIST:
        print(f"     • {f}")

if WARN_LIST:
    print("\n  ⚠️  WARNINGS (risk of Stage 3/4 penalty):")
    for w in WARN_LIST:
        print(f"     • {w}")

overall = "✅ PASS" if not FAIL_LIST else "❌ FAIL"
print(f"\n  VERDICT: {overall}")

if FAIL_LIST:
    risk = "HIGH — likely auto-rejected"
elif len(WARN_LIST) > 4:
    risk = "MEDIUM — may fail Stage 3 or 4"
elif WARN_LIST:
    risk = "LOW-MEDIUM — safe to submit but review warnings"
else:
    risk = "LOW — safe to submit"

print(f"  RISK LEVEL: {risk}")
print("═"*68 + "\n")

sys.exit(0 if not FAIL_LIST else 1)
