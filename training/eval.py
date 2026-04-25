#!/usr/bin/env python3
"""Eval harness — produces the comparison table & plots in plots/.

Reads checkpoint reward histories (jsonl) and runs all baselines + the
SFT/GRPO checkpoints against the local env.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from server.blindspot_environment import BlindspotEnvironment
from baselines import random_baseline, trending_baseline, dense_retrieval_baseline


POLICIES = {
    "random": random_baseline.run_episode,
    "trending": trending_baseline.run_episode,
    "dense_retrieval": dense_retrieval_baseline.run_episode,
}


def evaluate(name, fn, env, users, seeds):
    totals = []
    for uid in users:
        for s in range(seeds):
            br = fn(env, uid, seed=s)
            totals.append(br.get("total", 0.0))
    return {
        "name": name,
        "n": len(totals),
        "mean": statistics.mean(totals),
        "std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "min": min(totals),
        "max": max(totals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(REPO_ROOT / "plots" / "eval_results.json"))
    args = ap.parse_args()

    env = BlindspotEnvironment()
    users = env._data.user_ids  # noqa: SLF001
    results = []
    for name, fn in POLICIES.items():
        r = evaluate(name, fn, env, users, args.seeds)
        print(f"{r['name']:>20s}  mean={r['mean']:+.3f} std={r['std']:.3f}  "
              f"min={r['min']:+.3f}  max={r['max']:+.3f}  (n={r['n']})")
        results.append(r)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
