import requests
import json
import sys

BASE = "http://127.0.0.1:8000"
PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} -- {detail}")

print("=" * 60)
print("  AURA API Test Suite")
print("=" * 60)

# ── Test 1: GET / ────────────────────────────────────────────
print("\n[1] GET /  (Frontend)")
r = requests.get(f"{BASE}/")
check("Status 200", r.status_code == 200, r.status_code)
check("Returns HTML", "text/html" in r.headers.get("content-type", ""), r.headers.get("content-type"))

# ── Test 2: GET /api/presets ──────────────────────────────────
print("\n[2] GET /api/presets")
r = requests.get(f"{BASE}/api/presets")
presets = r.json()
check("Status 200", r.status_code == 200, r.status_code)
check("Has ml_engineer",         "ml_engineer"         in presets)
check("Has frontend_developer",  "frontend_developer"  in presets)
check("Has backend_engineer",    "backend_engineer"    in presets)
check("Has fullstack_developer", "fullstack_developer" in presets)

# ── Test 3: POST /api/rank ────────────────────────────────────
print("\n[3] POST /api/rank  (ml_engineer, mock mode)")
payload = {
    "jd_text": presets["ml_engineer"],
    "weight_semantic":   0.40,
    "weight_experience": 0.20,
    "weight_behavioral": 0.25,
    "weight_activity":   0.15,
}
r = requests.post(f"{BASE}/api/rank", json=payload, timeout=120)
check("Status 200", r.status_code == 200, r.text[:200])

if r.status_code == 200:
    data = r.json()
    cands = data.get("candidates", [])
    check("Provider = mock",         data.get("provider_used") == "mock", data.get("provider_used"))
    check("Candidates returned > 0", len(cands) > 0, len(cands))
    check("parsed_jd present",       "parsed_jd" in data)
    if cands:
        top = cands[0]
        check("Top has rank=1",       top.get("rank") == 1,           top.get("rank"))
        check("Top has score field",  isinstance(top.get("score"), (int, float)))
        check("Top has name",         bool(top.get("name")))
        check("Top has recommendation", bool(top.get("recommendation")))
        check("Top has explanation",  bool(top.get("explanation")))
        check("Top has strengths",    isinstance(top.get("strengths"), list))
        check("Top has gaps",         isinstance(top.get("gaps"), list))
        print(f"\n  Top 3 Candidates:")
        for c in cands[:3]:
            print(f"    Rank {c['rank']}: {c['name']} | Score={c['score']} | {c['recommendation']}")

# ── Test 4: POST /api/rank – frontend_developer preset ────────
print("\n[4] POST /api/rank  (frontend_developer, mock mode)")
payload2 = {
    "jd_text": presets["frontend_developer"],
    "weight_semantic":   0.40,
    "weight_experience": 0.20,
    "weight_behavioral": 0.25,
    "weight_activity":   0.15,
}
r2 = requests.post(f"{BASE}/api/rank", json=payload2, timeout=120)
check("Status 200", r2.status_code == 200, r2.text[:200])
if r2.ok:
    d2 = r2.json()
    check("Returns candidates", len(d2.get("candidates", [])) > 0)

# ── Test 5: POST /api/rank – validation (empty JD) ─────────────
print("\n[5] POST /api/rank  (empty JD – should 400)")
r3 = requests.post(f"{BASE}/api/rank", json={"jd_text": "   "}, timeout=30)
check("Status 400 for empty JD", r3.status_code == 400, r3.status_code)

# ── Test 6: GET /api/download ─────────────────────────────────
print("\n[6] GET /api/download  (CSV)")
r4 = requests.get(f"{BASE}/api/download")
check("Status 200", r4.status_code == 200, r4.status_code)
check("CSV content-type", "text/csv" in r4.headers.get("content-type", ""))
lines = r4.text.strip().splitlines()
check("Has header row",  lines[0].startswith("candidate_id") if lines else False)
check("Has data rows",   len(lines) > 1, len(lines))
print(f"  CSV rows (incl. header): {len(lines)}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED tests:")
    for f in FAIL:
        print(f"    - {f}")
else:
    print("  All tests passed!")
print("=" * 60)
sys.exit(0 if not FAIL else 1)
