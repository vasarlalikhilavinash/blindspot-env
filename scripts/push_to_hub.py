#!/usr/bin/env python3
"""Push the trained LoRA adapter to the Hugging Face Hub.

Usage (in Colab, after training):
    HF_TOKEN=hf_xxx python scripts/push_to_hub.py

Or set HF_REPO_ID env var (default: "vasarlalikhilavinash/blindspot-qwen25-7b-grpo").
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTER = REPO_ROOT / os.environ.get("ADAPTER_PATH", "training/checkpoints/grpo")
REPO_ID = os.environ.get("HF_REPO_ID", "vasarlalikhilavinash/blindspot-qwen25-7b-grpo")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not ADAPTER.is_dir():
    sys.exit(f"adapter dir not found: {ADAPTER}")
if not TOKEN:
    sys.exit("HF_TOKEN env var not set")

from huggingface_hub import HfApi, create_repo  # type: ignore

api = HfApi(token=TOKEN)
print(f"creating repo {REPO_ID} (if missing)...")
create_repo(REPO_ID, exist_ok=True, token=TOKEN, private=False)

print(f"uploading {ADAPTER} → {REPO_ID} ...")
api.upload_folder(
    folder_path=str(ADAPTER),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Blindspot GRPO LoRA adapter",
)

# Upload the model card as README so it shows on the Hub page
card = REPO_ROOT / "training" / "MODEL_CARD.md"
if card.exists():
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )
    print("✓ uploaded model card")

print(f"✓ pushed: https://huggingface.co/{REPO_ID}")
