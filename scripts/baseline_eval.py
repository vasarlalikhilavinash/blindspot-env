#!/usr/bin/env python3
"""Real-data calibration: run all baselines × N seeds × all users.

Replaces the synthetic-seed numbers in the README with measured results
from the actual 17-user dataset.

Output: prints a markdown table + writes data/baseline_calibration.json
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.blindspot_demo import BlindspotDemo
import random

SEEDS = [0, 1, 2, 3, 4]


def oracle_policy(demo, uid, candidates, k=3):
    """Upper-bound policy: pick the 3 candidates with the highest reward signal."""
    adoption = demo.d["adoption"].get(uid, {})
    knn = demo._knn_adopted(uid)
    novelty = demo.d["novelty"]
    comp = demo.d["comp"].get(uid, {})

    def s(c):
        cs = str(c)
        a = float(adoption.get(cs, 0.0))
        if a < 1e-6 and cs in knn:
            a = 0.3
        if a < 1e-6:
            return -0.1  # FP penalty
        bonus = 0.5 * (1.0 if novelty.get(cs) else 0.0) + float(comp.get(cs, 0.0))
        return a + bonus
    return sorted(candidates, key=lambda c: -s(c))[:k]


def main():
    demo = BlindspotDemo()
    splits = json.load(open(REPO_ROOT / "data" / "user_splits.json"))
    all_users = list(demo.users.keys())
    train_users = splits["train"]
    test_users = splits["test"]

    POLICIES = {
        "Random":          lambda demo, profile, cands: demo.policy_random(profile, cands)[0],
        "Trending":        lambda demo, profile, cands: demo.policy_trending(profile, cands)[0],
        "Dense Retrieval": lambda demo, profile, cands: demo.policy_dense_retrieval(profile, cands)[0],
        "Blindspot proxy": lambda demo, profile, cands: demo.policy_blindspot(profile, cands)[0],
        "Oracle (upper-bound)": lambda demo, profile, cands: oracle_policy(demo, profile["matched_user_id"], cands),
    }

    results = {}  # name -> list of per-(user,seed) totals

    for name, fn in POLICIES.items():
        rewards = []
        for uid in all_users:
            profile = {
                "matched_user_id": uid,
                "matched_summary": demo.users[uid][:300],
                "match_similarity": 1.0,
                "shared_keywords": [],
                "query_vec": demo.vocab.transform(demo.users[uid]),
            }
            base_candidates = [str(c) for c in demo.d["pool"].get(uid, [])][:50]
            if not base_candidates:
                base_candidates = demo.build_candidates(profile)
            for seed in SEEDS:
                rng = random.Random(seed)
                cands = base_candidates[:]
                rng.shuffle(cands)
                # For Random policy we need the seed to take effect
                if name == "Random":
                    surfaced = rng.sample(cands, 3)
                else:
                    surfaced = fn(demo, profile, cands)
                r = demo._reward_for(uid, surfaced)["total"]
                rewards.append(r)
        results[name] = rewards
        print(f"  {name:25s}  mean={statistics.mean(rewards):+.3f}  std={statistics.stdev(rewards):.3f}")

    print("\n## Real-data calibration (5 seeds × 17 users)\n")
    print("| Policy | Mean total | Std |")
    print("|---|---|---|")
    for name in POLICIES:
        rewards = results[name]
        m, s = statistics.mean(rewards), statistics.stdev(rewards)
        print(f"| {name} | {m:+.2f} | {s:.2f} |")

    out = {
        name: {"mean": statistics.mean(rewards), "std": statistics.stdev(rewards),
               "n": len(rewards)}
        for name, rewards in results.items()
    }
    out["_meta"] = {"seeds": SEEDS, "users": all_users,
                    "train_users": train_users, "test_users": test_users}
    (REPO_ROOT / "data" / "baseline_calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\n✓ wrote data/baseline_calibration.json")


if __name__ == "__main__":
    main()
