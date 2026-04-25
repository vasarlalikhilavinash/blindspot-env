"""Pre-compute pipeline for the Blindspot environment.

Each script in this directory is one stage of the offline pipeline shown in
the README diagram. They write artifacts under `data/` that the runtime env
loads via `server/data_loader.py`.

Run order (or just `bash run_all_precompute.sh`):

    01_fetch_users.py          — Semantic Scholar API: pull 50 ML researchers
    02_fetch_corpus.py         — arXiv: cs.LG / cs.CL / cs.AI 2024-2025
    03_extract_concepts.py     — KeyBERT over corpus → ~5k candidate concepts
    04_build_pools.py          — per-user candidate pools (50 each, mixed)
    05_score_adoption.py       — ground-truth adoption from post-T artifacts
    06_build_paths.py          — citation-BFS reading paths (5 papers each)
    07_score_comprehension.py  — two-judge comprehension (GPT + Claude)

All scripts are idempotent and cache intermediates under `scripts/_cache/`.

For local dev / CI, prefer:  python scripts/build_synthetic_seed.py
"""
