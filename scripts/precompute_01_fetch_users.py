#!/usr/bin/env python3
"""Stage 01 — Fetch ~50 ML researchers from Semantic Scholar.

Output:
    scripts/_cache/users.json   [{"author_id", "name", "papers_pre_T", "papers_post_T"}]

Splits each author's papers at T = 2025-09-01.

Requires `requests`. Optional `SEMANTIC_SCHOLAR_API_KEY` for higher rate limits.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

T_DATE = datetime(2025, 9, 1)
N_AUTHORS = 50
CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
OUT = CACHE_DIR / "users.json"

SEED_QUERIES = [
    "large language model alignment",
    "reinforcement learning from human feedback",
    "retrieval augmented generation",
    "speculative decoding",
    "vision language model",
]

S2_API = "https://api.semanticscholar.org/graph/v1"
HEADERS = {}
if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
    HEADERS["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]


def s2_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(5):
        r = requests.get(f"{S2_API}{path}", params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Repeated 429 from Semantic Scholar on {path}")


def fetch_top_authors() -> List[Dict[str, Any]]:
    """Return top N_AUTHORS distinct authors across SEED_QUERIES."""
    seen: Dict[str, Dict[str, Any]] = {}
    for q in SEED_QUERIES:
        data = s2_get(
            "/paper/search",
            {"query": q, "limit": 100, "fields": "authors.authorId,authors.name,year"},
        )
        for paper in data.get("data", []):
            for author in paper.get("authors", []) or []:
                aid = author.get("authorId")
                if aid and aid not in seen:
                    seen[aid] = {"author_id": aid, "name": author.get("name", "")}
            if len(seen) >= N_AUTHORS * 4:
                break
    # Keep top N_AUTHORS by an arbitrary stable order (dict insertion order)
    return list(seen.values())[: N_AUTHORS * 2]


def fetch_author_papers(author_id: str) -> List[Dict[str, Any]]:
    data = s2_get(
        f"/author/{author_id}/papers",
        {"limit": 200, "fields": "title,year,publicationDate,abstract,externalIds"},
    )
    return data.get("data", []) or []


def split_papers(papers: List[Dict[str, Any]]):
    pre, post = [], []
    for p in papers:
        date = p.get("publicationDate")
        if not date:
            year = p.get("year")
            if not year:
                continue
            date = f"{year}-01-01"
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue
        (pre if d < T_DATE else post).append(p)
    return pre, post


def main():
    candidates = fetch_top_authors()
    keepers = []
    for cand in candidates:
        try:
            papers = fetch_author_papers(cand["author_id"])
        except Exception as e:
            print(f"  skip {cand['name']}: {e}")
            continue
        pre, post = split_papers(papers)
        if len(pre) >= 8 and len(post) >= 3:  # need both halves
            cand["papers_pre_T"] = pre
            cand["papers_post_T"] = post
            keepers.append(cand)
            print(f"  ✓ {cand['name']}  pre={len(pre)} post={len(post)}")
        if len(keepers) >= N_AUTHORS:
            break
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(keepers, f, indent=2)
    print(f"Wrote {len(keepers)} users to {OUT}")


if __name__ == "__main__":
    main()
