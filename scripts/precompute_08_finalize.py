#!/usr/bin/env python3
"""Stage 08 — Finalize: convert scripts/_cache/ artifacts to data/ schema.

Maps the precompute outputs onto the runtime contract expected by
`server/data_loader.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "scripts" / "_cache"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)


def cp(src: str, dst: str, transform=None):
    s = CACHE / src
    if not s.exists():
        print(f"  [skip] {src} (missing)")
        return
    payload = json.loads(s.read_text())
    if transform:
        payload = transform(payload)
    (OUT / dst).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"  wrote data/{dst}")


def to_concept_catalog(concepts):
    out = {}
    for c in concepts:
        cid = str(c["concept_id"])
        out[cid] = {
            "title": c["title"],
            "one_liner": c["one_liner"],
            "abstract_summary": c.get("abstract_summary", c["one_liner"]),
            "growth_signal": min(1.0, c.get("mention_count", 1) / 100.0),
            "is_trending": False,  # set in trending pass
        }
    return out


def main():
    cp("user_summaries.json", "user_summaries.json")
    cp("concept_pool_per_user.json", "concept_pool_per_user.json",
       transform=lambda p: (
           {uid: (v if isinstance(v, list)
                   else list(set(v.get("positives", []) + v.get("hard_negatives", []))))
            for uid, v in p.items()}
       ))
    cp("ground_truth_adoption.json", "ground_truth_adoption.json")
    cp("knn_users.json", "knn_users.json")
    cp("comprehension_scores.json", "comprehension_scores.json")
    cp("reading_paths.json", "reading_paths.json")

    # Catalog + novelty (derived from trending list)
    concepts = json.loads((CACHE / "concepts.json").read_text())
    catalog = to_concept_catalog(concepts)
    trending_ids = set(json.loads((CACHE / "trending_at_T.json").read_text()))
    for cid in catalog:
        catalog[cid]["is_trending"] = int(cid) in trending_ids
    (OUT / "concept_catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True))
    novelty = {cid: (not v["is_trending"]) for cid, v in catalog.items()}
    (OUT / "novelty_flags.json").write_text(json.dumps(novelty, indent=2, sort_keys=True))
    print("  wrote data/concept_catalog.json + data/novelty_flags.json")
    print("Done.")


if __name__ == "__main__":
    main()
