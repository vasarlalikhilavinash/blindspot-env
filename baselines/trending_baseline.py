#!/usr/bin/env python3
"""Trending baseline: always surface the top 10 trending concepts.

This is the strawman every "be first to know" feed reader implements.
Blindspot is designed to BEAT this baseline because trending concepts get
the novelty penalty (they're not unknown-unknowns).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import BlindspotAction
from server.blindspot_environment import BlindspotEnvironment


def run_episode(env: BlindspotEnvironment, user_id: str, seed: int = 0) -> dict:
    obs = env.reset(seed=seed, user_id=user_id)
    catalog = env._data.concept_catalog  # noqa: SLF001
    # Score each candidate by trending+growth_signal — pick top 10
    scored = []
    for c in obs.candidate_concepts:
        rec = catalog.get(c.concept_id, {})
        score = (1.0 if rec.get("is_trending") else 0.0) + float(rec.get("growth_signal", 0.0))
        scored.append((score, c.concept_id))
    scored.sort(reverse=True)

    for _, cid in scored[:10]:
        obs = env.step(BlindspotAction(type="surface", concept_id=cid))
        if obs.done:
            break
    if not obs.done:
        obs = env.step(BlindspotAction(type="stop"))
    return obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()

    env = BlindspotEnvironment()
    users = env._data.user_ids  # noqa: SLF001
    totals = []
    for uid in users:
        for s in range(args.seed_start, args.seed_start + args.n_seeds):
            br = run_episode(env, uid, seed=s)
            totals.append(br.get("total", 0.0))
    print(f"trending_baseline: n={len(totals)} mean={statistics.mean(totals):.4f} "
          f"stdev={statistics.stdev(totals):.4f} min={min(totals):.3f} max={max(totals):.3f}")


if __name__ == "__main__":
    main()
