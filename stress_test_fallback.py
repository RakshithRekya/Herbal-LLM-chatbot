"""
Stress test: exhaust Groq rate limit → observe fallback chain progression.
Run from project root: python stress_test_fallback.py
"""

import logging
import time
import sys
import os

# ── Enable verbose logging so fallback messages appear ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.model import _fallback_invoke
from langchain_core.messages import HumanMessage

# ── Config ──────────────────────────────────────────────────────────────────
BURST_SIZE   = 15    # requests fired rapidly to exhaust Groq (~30 RPM free tier)
DELAY        = 0.0   # seconds between requests (fast enough to hit rate limit)
TEST_PROMPT  = "What herb is good for sleep? One sentence answer. " + ("context filler " * 500)

# ── Run ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🔥 FALLBACK CHAIN STRESS TEST")
print(f"   Firing {BURST_SIZE} requests with {DELAY}s delay")
print(f"   Watch for ⚠️  (provider failed) and ✅ (provider used)")
print("="*60 + "\n")

provider_tally = {}

for i in range(1, BURST_SIZE + 1):
    print(f"--- Request {i}/{BURST_SIZE} ---")
    try:
        response = _fallback_invoke([HumanMessage(content=TEST_PROMPT)])
        # Tally which provider responded (logged inside _fallback_invoke)
        print(f"   Response snippet: {response.content[:80]}...")
    except RuntimeError as e:
        print(f"❌ ALL PROVIDERS FAILED: {e}")
        break
    
    time.sleep(DELAY)

print("\n" + "="*60)
print("✅ Stress test complete. Review logs above for fallback progression.")
print("="*60)