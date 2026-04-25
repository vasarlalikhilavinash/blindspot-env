#!/usr/bin/env python3
"""Stage 02 — Fetch arXiv corpus (cs.LG / cs.CL / cs.AI 2024-2025).

Streams the arXiv OAI-PMH feed via `feedparser` and caches abstracts.
Output:
    scripts/_cache/corpus.jsonl  [{"arxiv_id","title","abstract","year","categories"}]
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode

import feedparser

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
OUT = CACHE_DIR / "corpus.jsonl"

CATEGORIES = ["cs.LG", "cs.CL", "cs.AI"]
N_PER_CATEGORY = 2000  # ~6k total
ARXIV_API = "http://export.arxiv.org/api/query?"


def fetch_category(category: str, max_results: int) -> int:
    written = 0
    page = 100
    with OUT.open("a", encoding="utf-8") as f:
        for start in range(0, max_results, page):
            url = ARXIV_API + urlencode({
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": page,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            })
            feed = feedparser.parse(url)
            if not feed.entries:
                break
            for entry in feed.entries:
                arxiv_id = entry.id.split("/abs/")[-1].split("v")[0]
                rec = {
                    "arxiv_id": arxiv_id,
                    "title": (entry.title or "").strip().replace("\n", " "),
                    "abstract": (entry.summary or "").strip().replace("\n", " "),
                    "year": entry.published[:4] if hasattr(entry, "published") else "",
                    "categories": [t.term for t in getattr(entry, "tags", [])],
                }
                f.write(json.dumps(rec) + "\n")
                written += 1
            time.sleep(3)  # arXiv rate limit
    return written


def main():
    if OUT.exists():
        OUT.unlink()
    total = 0
    for cat in CATEGORIES:
        n = fetch_category(cat, N_PER_CATEGORY)
        print(f"  {cat}: {n} papers")
        total += n
    print(f"Total: {total} papers → {OUT}")


if __name__ == "__main__":
    main()
