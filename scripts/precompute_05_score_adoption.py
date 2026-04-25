#!/usr/bin/env python3
"""Stage 05 — Score adoption per (user, concept).

Rules:
    1.0  if concept appears in user's post-T papers (self-adoption)
    0.3  partial credit if ≥2 of user's 5-NN researchers self-adopted

Outputs:
    scripts/_cache/ground_truth_adoption.json   {user_id: {concept_id_str: float}}
    scripts/_cache/knn_users.json               {user_id: [neighbor_user_id, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
USERS_IN = CACHE_DIR / "users.json"
STRUCT_IN = CACHE_DIR / "pool_structured.json"
POOL_IN = CACHE_DIR / "concept_pool_per_user.json"
SUMMARIES_IN = CACHE_DIR / "user_summaries.json"
ADOPTION_OUT = CACHE_DIR / "ground_truth_adoption.json"
KNN_OUT = CACHE_DIR / "knn_users.json"

K = 5
NEIGHBOR_THRESHOLD = 2


def main():
    users = json.loads(USERS_IN.read_text())
    structured = json.loads(STRUCT_IN.read_text())
    flat_pools = json.loads(POOL_IN.read_text())
    summaries = json.loads(SUMMARIES_IN.read_text())

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    uid_order = list(summaries.keys())
    user_emb = enc.encode([summaries[u] for u in uid_order], normalize_embeddings=True)
    sim = user_emb @ user_emb.T

    knn = {}
    for i, uid in enumerate(uid_order):
        order = np.argsort(-sim[i]).tolist()
        neighbors = [uid_order[j] for j in order if uid_order[j] != uid][:K]
        knn[uid] = neighbors
    KNN_OUT.write_text(json.dumps(knn, indent=2))

    # 1. Self-adoption from structured pool
    adoption: Dict[str, Dict[str, float]] = {}
    for uid, p in structured.items():
        self_adopted = set(p.get("adopted_set", []))
        scored = {}
        for cid in p.get("positives", []) + p.get("hard_negatives", []):
            if cid in self_adopted:
                scored[str(cid)] = 1.0
        adoption[uid] = scored

    # 2. k-NN partial credit (0.3) over all final-pool concepts
    for uid, pool_ids in flat_pools.items():
        for cid in pool_ids:
            cs = str(cid)
            if cs in adoption.get(uid, {}):
                continue
            hits = sum(
                1 for n in knn.get(uid, [])
                if cs in adoption.get(n, {}) and adoption[n][cs] >= 1.0
            )
            if hits >= NEIGHBOR_THRESHOLD:
                adoption.setdefault(uid, {})[cs] = 0.3

    ADOPTION_OUT.write_text(json.dumps(adoption, indent=2))
    print(f"Wrote adoption for {len(adoption)} users.")


if __name__ == "__main__":
    main()
