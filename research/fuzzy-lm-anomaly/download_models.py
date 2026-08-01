"""Pre-download candidate models for the cross-architecture comparison.

Tries a few plausible HF ids per model, since naming varies (LFM2 vs LFM2.5,
-it vs base). Reports which resolved so the capture runs can be wired up.
"""
import sys, traceback
from huggingface_hub import snapshot_download

CANDIDATES = {
    "gemma3-270m": ["google/gemma-3-270m-it", "google/gemma-3-270m"],
    "lfm2.5-350m": ["LiquidAI/LFM2.5-350M", "LiquidAI/LFM2-350M",
                    "LiquidAI/LFM2-350M-Instruct"],
}

resolved = {}
for nick, ids in CANDIDATES.items():
    for mid in ids:
        try:
            p = snapshot_download(mid, allow_patterns=["*.json", "*.safetensors",
                                                       "*.model", "*.txt"])
            print(f"OK   {nick:14s} -> {mid}")
            resolved[nick] = mid
            break
        except Exception as e:
            print(f"FAIL {nick:14s} -> {mid}: {type(e).__name__}: {str(e)[:120]}")
    else:
        print(f"NONE {nick}: no candidate resolved")
print("\nRESOLVED:", resolved)
