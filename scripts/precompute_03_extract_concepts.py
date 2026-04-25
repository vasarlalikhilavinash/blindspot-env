#!/usr/bin/env python3
"""Stage 03 — Extract ~5k candidate concepts via KeyBERT over the corpus.

Output:
    scripts/_cache/concepts.json  [{"concept_id","title","one_liner","mention_papers":[arxiv_ids]}]
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from keybert import KeyBERT  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
IN = CACHE_DIR / "corpus.jsonl"
OUT = CACHE_DIR / "concepts.json"

MAX_CONCEPTS = 5000
KEYPHRASE_NGRAM_RANGE = (2, 4)
TOP_N_PER_DOC = 8


def main():
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=encoder)

    phrase_counts: Counter[str] = Counter()
    phrase_papers: Dict[str, List[str]] = {}

    with IN.open("r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            text = paper["title"] + ". " + paper["abstract"]
            try:
                kws = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
                    stop_words="english",
                    top_n=TOP_N_PER_DOC,
                )
            except Exception:
                continue
            for phrase, _ in kws:
                ph = phrase.lower().strip()
                phrase_counts[ph] += 1
                phrase_papers.setdefault(ph, []).append(paper["arxiv_id"])

    # Filter: ≥3 mentions, ≤500 mentions (drop boilerplate)
    candidates = [
        (ph, c) for ph, c in phrase_counts.items() if 3 <= c <= 500
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:MAX_CONCEPTS]

    out = []
    for cid, (phrase, count) in enumerate(candidates):
        out.append({
            "concept_id": cid,
            "title": phrase,
            "one_liner": f"{phrase} (mentioned in {count} papers across the corpus).",
            "mention_papers": phrase_papers[phrase][:50],
            "mention_count": count,
        })
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} concepts to {OUT}")


if __name__ == "__main__":
    main()
