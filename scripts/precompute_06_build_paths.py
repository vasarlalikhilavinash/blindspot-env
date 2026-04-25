#!/usr/bin/env python3
"""Stage 06 — Build a 5-paper canonical reading path per (in-pool) concept.

Strategy (corpus-grounded, no network):
    Each concept already carries ``mention_papers`` — the arXiv IDs of
    papers in our 6k-paper corpus that mention the concept's title. We
    look those papers up in ``corpus.jsonl``, dedupe, sort foundational →
    recent (by year ascending), and keep the first 5.

    This avoids slow Semantic Scholar citation BFS (which gets aggressively
    rate-limited even with an API key) and produces a fully reproducible
    path drawn from data we control.

Output:
    scripts/_cache/reading_paths.json   {concept_id_str: [{title, arxiv_id, year}]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CONCEPTS_IN = CACHE_DIR / "concepts.json"
CORPUS_IN = CACHE_DIR / "corpus.jsonl"
POOL_IN = CACHE_DIR / "concept_pool_per_user.json"
OUT = CACHE_DIR / "reading_paths.json"

PATH_LEN = 5


def load_corpus() -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    with CORPUS_IN.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ax = rec.get("arxiv_id")
            if ax:
                by_id[ax] = rec
    return by_id


def main():
    concepts = json.loads(CONCEPTS_IN.read_text())
    corpus = load_corpus()
    print(f"Corpus: {len(corpus)} papers indexed")

    in_pool = set()
    if POOL_IN.exists():
        for ids in json.loads(POOL_IN.read_text()).values():
            in_pool.update(int(x) for x in ids)
        concepts = [c for c in concepts if c["concept_id"] in in_pool]
        print(f"Restricted to {len(concepts)} in-pool concepts.")

    paths: Dict[str, List[dict]] = {}
    for c in concepts:
        cid = c["concept_id"]
        seen_titles: set[str] = set()
        items: List[dict] = []
        for ax in c.get("mention_papers", []):
            paper = corpus.get(ax)
            if not paper:
                continue
            title = (paper.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            items.append({
                "title": title,
                "arxiv_id": ax,
                "year": str(paper.get("year") or ""),
            })
        # Sort foundational → recent
        items.sort(key=lambda r: r["year"])
        paths[str(cid)] = items[:PATH_LEN]

    OUT.write_text(json.dumps(paths, indent=2))
    n_with_path = sum(1 for v in paths.values() if v)
    print(f"Wrote {len(paths)} reading paths ({n_with_path} non-empty).")


if __name__ == "__main__":
    main()
