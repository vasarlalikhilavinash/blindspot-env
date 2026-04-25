#!/usr/bin/env python3
"""Dense-retrieval baseline: surface concepts whose titles are most similar
to the user_summary under a sentence-transformer encoder.

This is the strongest non-RL baseline — it captures relevance via
semantic similarity but cannot reason about novelty, onboarding quality,
or budget allocation.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import BlindspotAction
from server.blindspot_environment import BlindspotEnvironment


def _hash_embed(text: str, dim: int = 256) -> "list[float]":
    """Cheap deterministic hash-based bag-of-words embedding (no deps)."""
    import hashlib
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _cos(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def run_episode(env: BlindspotEnvironment, user_id: str, seed: int = 0) -> dict:
    obs = env.reset(seed=seed, user_id=user_id)
    user_vec = _hash_embed(obs.user_summary)
    scored = []
    for c in obs.candidate_concepts:
        cand_vec = _hash_embed(c.title + " " + c.one_liner)
        scored.append((_cos(user_vec, cand_vec), c.concept_id))
    scored.sort(reverse=True)

    # Inspect top 8 (uses some inspect budget for credibility), surface top 10
    for _, cid in scored[:8]:
        obs = env.step(BlindspotAction(type="inspect", concept_id=cid))
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
    print(f"dense_retrieval_baseline: n={len(totals)} mean={statistics.mean(totals):.4f} "
          f"stdev={statistics.stdev(totals):.4f} min={min(totals):.3f} max={max(totals):.3f}")


if __name__ == "__main__":
    main()
