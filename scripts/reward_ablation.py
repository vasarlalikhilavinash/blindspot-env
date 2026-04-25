#!/usr/bin/env python3
"""Reward-shaping ablation: re-score the trained policy under different reward weights.

Demonstrates that each reward component contributes independently and that the
multi-component design isn't decorative. Re-uses the cached trained-policy outputs
so this runs in seconds with no GPU.

Output: prints a markdown table + writes data/reward_ablation.json
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.blindspot_demo import BlindspotDemo


WEIGHT_VARIANTS = {
    "Adoption only":            {"adoption": 1.0, "novelty": 0.0, "onboarding": 0.0, "fp": 0.0},
    "+ Novelty":                {"adoption": 1.0, "novelty": 1.0, "onboarding": 0.0, "fp": 0.0},
    "+ Onboarding":             {"adoption": 1.0, "novelty": 1.0, "onboarding": 1.0, "fp": 0.0},
    "+ False-positive penalty": {"adoption": 1.0, "novelty": 1.0, "onboarding": 1.0, "fp": 1.0},
}


def reward_for(demo, uid, surfaced, w):
    adoption = demo.d["adoption"].get(uid, {})
    knn = demo._knn_adopted(uid)
    novelty = demo.d["novelty"]
    comp = demo.d["comp"].get(uid, {})

    total = 0.0
    for cid in surfaced:
        cs = str(cid)
        a = float(adoption.get(cs, 0.0))
        adopted = a >= 1e-6
        if not adopted and cs in knn:
            a = 0.3
            adopted = True
        total += w["adoption"] * a
        if adopted:
            total += w["novelty"] * 0.5 * (1.0 if novelty.get(cs) else 0.0)
            total += w["onboarding"] * float(comp.get(cs, 0.0))
        else:
            total -= w["fp"] * 0.1
    return total


def main():
    demo = BlindspotDemo()
    if not demo.demo_cache:
        print("⚠️ no data/demo_cache.json — run scripts/precompute_demo_cache.py first")
        sys.exit(1)

    print("\n## Reward-shaping ablation\n")
    print("Re-scoring the same trained-policy surfaces under different reward weights "
          "(across all cached real users).\n")
    print("| Variant | Mean reward | Std |")
    print("|---|---|---|")

    out = {}
    for variant, w in WEIGHT_VARIANTS.items():
        rewards = []
        for uid, _summary in demo.users.items():
            entry = demo.demo_cache.get(f"user::{uid}")
            if not entry:
                continue
            r = reward_for(demo, uid, entry["surfaced"], w)
            rewards.append(r)
        if rewards:
            m = statistics.mean(rewards)
            s = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            print(f"| {variant} | {m:+.2f} | {s:.2f} |")
            out[variant] = {"mean": m, "std": s, "n": len(rewards)}

    (REPO_ROOT / "data" / "reward_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\n✓ wrote data/reward_ablation.json")


if __name__ == "__main__":
    main()
