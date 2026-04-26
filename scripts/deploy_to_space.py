#!/usr/bin/env python3
"""Deploy the Gradio demo to a Hugging Face Space.

Uploads the minimum file set needed for the Space to run:
    spaces/app.py                    → app.py            (Space entrypoint)
    spaces/README.md                 → README.md         (Space metadata frontmatter)
    spaces/requirements.txt          → requirements.txt
    scripts/blindspot_demo.py        → scripts/blindspot_demo.py
    scripts/precompute_demo_cache.py → scripts/precompute_demo_cache.py
    data/*.json                      → data/*.json

In particular, the Space expects both cache variants when available:
    data/demo_cache.json
    data/demo_cache_pretrain.json

Usage (set HF_TOKEN with write access first):
    HF_TOKEN=hf_xxx python scripts/deploy_to_space.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_ID = os.environ.get("HF_SPACE_ID", "Vasarlaavinash/blindspot-demo")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not TOKEN:
    sys.exit("HF_TOKEN env var not set")

from huggingface_hub import HfApi, create_repo  # type: ignore

api = HfApi(token=TOKEN)

EXPECTED_CACHE_FILES = ["demo_cache.json", "demo_cache_pretrain.json"]

print(f"creating Space {SPACE_ID} (if missing) ...")
create_repo(SPACE_ID, repo_type="space", space_sdk="gradio",
            exist_ok=True, token=TOKEN, private=False)

# Mapping: local path → path inside the Space repo
uploads = [
    ("spaces/app.py",          "app.py"),
    ("spaces/README.md",       "README.md"),
    ("spaces/requirements.txt","requirements.txt"),
    ("scripts/blindspot_demo.py",        "scripts/blindspot_demo.py"),
    ("scripts/precompute_demo_cache.py", "scripts/precompute_demo_cache.py"),
]
# All JSON data files
for jf in (REPO_ROOT / "data").glob("*.json"):
    uploads.append((f"data/{jf.name}", f"data/{jf.name}"))

for cache_name in EXPECTED_CACHE_FILES:
    if not (REPO_ROOT / "data" / cache_name).exists():
        print(f"warning: data/{cache_name} not found; Space will fall back to non-cached behavior")

# Empty __init__.py so 'scripts' is importable as a package
init_marker = REPO_ROOT / "scripts" / "__init__.py"
if not init_marker.exists():
    init_marker.write_text("")
uploads.append(("scripts/__init__.py", "scripts/__init__.py"))

print(f"\nuploading {len(uploads)} files to {SPACE_ID} ...")
for local, remote in uploads:
    src = REPO_ROOT / local
    if not src.exists():
        print(f"  skip {local} (not found)")
        continue
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=remote,
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message=f"deploy: {remote}",
    )
    print(f"  ✓ {local} → {remote}")

print(f"\n✓ Space deployed: https://huggingface.co/spaces/{SPACE_ID}")
print("  (build takes ~2 min the first time)")
