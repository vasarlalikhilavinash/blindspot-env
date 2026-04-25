#!/usr/bin/env python3
"""Stage 04 — Build per-user candidate pools (50 concepts each).

Pool composition (per README):
    15 relevant + adopted     ← positives
    15 relevant + not adopted ← hard negatives
    10 trending distractors   ← popularity bait
    10 random noise           ← easy negatives

"Relevant" = top-cosine-similarity between user_summary embedding and
concept title embedding. "Adopted" = appears in user.papers_post_T text.
"Trending at T" = HuggingFace / arXiv trending feed snapshot at T (here:
top 100 concepts by mention_count in the 90 days leading up to T).

Outputs:
    scripts/_cache/concept_pool_per_user.json
    scripts/_cache/user_summaries.json
    scripts/_cache/trending_at_T.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
USERS_IN = CACHE_DIR / "users.json"
CONCEPTS_IN = CACHE_DIR / "concepts.json"
POOL_OUT = CACHE_DIR / "concept_pool_per_user.json"
SUMMARIES_OUT = CACHE_DIR / "user_summaries.json"
TRENDING_OUT = CACHE_DIR / "trending_at_T.json"

POOL_SIZE = 50
N_POSITIVES = 15
N_HARD_NEG = 15
N_TRENDING = 10
N_NOISE = 10


def user_summary_text(user) -> str:
    titles = [p.get("title", "") for p in user.get("papers_pre_T", [])][:30]
    return f"Researcher {user['name']}. Recent paper titles: " + " | ".join(titles)


def adopted_concepts(user, concepts) -> Set[int]:
    text = " ".join(p.get("title", "") + " " + (p.get("abstract") or "")
                    for p in user.get("papers_post_T", [])).lower()
    out = set()
    for c in concepts:
        if re.search(r"\b" + re.escape(c["title"]) + r"\b", text):
            out.add(c["concept_id"])
    return out


def main():
    users = json.loads(USERS_IN.read_text())
    concepts = json.loads(CONCEPTS_IN.read_text())
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    concept_titles = [c["title"] for c in concepts]
    concept_emb = enc.encode(concept_titles, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    summaries = {}
    pools = {}
    for u in users:
        uid = u["author_id"]
        summary = user_summary_text(u)
        summaries[uid] = summary
        u_emb = enc.encode([summary], normalize_embeddings=True)[0]
        sims = concept_emb @ u_emb  # cosine since normalized
        ranked = np.argsort(-sims).tolist()

        adopted = adopted_concepts(u, concepts)
        positives, hard_negs = [], []
        for idx in ranked:
            cid = concepts[idx]["concept_id"]
            if cid in adopted and len(positives) < N_POSITIVES:
                positives.append(cid)
            elif cid not in adopted and len(hard_negs) < N_HARD_NEG and sims[idx] > 0.30:
                hard_negs.append(cid)
            if len(positives) >= N_POSITIVES and len(hard_negs) >= N_HARD_NEG:
                break

        pools[uid] = {
            "positives": positives,
            "hard_negatives": hard_negs,
            "adopted_set": sorted(adopted),
        }

    # Trending @ T = top mention_count concepts
    trending = sorted(concepts, key=lambda c: c.get("mention_count", 0), reverse=True)[:200]
    trending_ids = [c["concept_id"] for c in trending]

    # Now stitch pools to 50 each — adding trending + noise
    rng = np.random.default_rng(42)
    final_pools: Dict[str, List[int]] = {}
    structured: Dict[str, Dict[str, List[int]]] = {}
    all_ids = [c["concept_id"] for c in concepts]
    for uid, p in pools.items():
        used = set(p["positives"]) | set(p["hard_negatives"])
        trend_pick = [c for c in trending_ids if c not in used][:N_TRENDING]
        used |= set(trend_pick)
        remaining = [c for c in all_ids if c not in used]
        rng.shuffle(remaining)
        noise_pick = remaining[:N_NOISE]
        pool = list(p["positives"]) + list(p["hard_negatives"]) + trend_pick + noise_pick
        rng.shuffle(pool)
        final_pools[uid] = pool[:POOL_SIZE]
        structured[uid] = {
            "positives": p["positives"],
            "hard_negatives": p["hard_negatives"],
            "trending": trend_pick,
            "noise": noise_pick,
            "adopted_set": p["adopted_set"],
            "final_pool": final_pools[uid],
        }

    POOL_OUT.write_text(json.dumps(final_pools, indent=2))
    (CACHE_DIR / "pool_structured.json").write_text(json.dumps(structured, indent=2))
    SUMMARIES_OUT.write_text(json.dumps(summaries, indent=2))
    TRENDING_OUT.write_text(json.dumps(trending_ids, indent=2))
    print(f"Wrote pools for {len(final_pools)} users; {len(trending_ids)} trending @ T.")


if __name__ == "__main__":
    main()
