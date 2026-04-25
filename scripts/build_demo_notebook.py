#!/usr/bin/env python3
"""Generate notebooks/03_demo.ipynb.

The notebook intentionally reuses the exact Gradio UI in spaces/app.py so the
Colab demo and the Hugging Face Space stay in sync.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CELLS = []


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    })


def code(text: str):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    })


md(
    """
# Blindspot · Live Judge Demo

This notebook launches the exact same UI shipped in `spaces/app.py`.

It includes:

- 5 policy panels: Random, Trending, Dense Retrieval, Blindspot pre-training, Blindspot RL
- real-user mode with held-out ground-truth rewards
- persona mode
- paragraph mode with nearest-neighbor fallback
- before/after focus toggle
- concept catalog browser
"""
)

md("## 1. Setup")

code(
    """
%%bash
pip install -q --upgrade gradio openai requests numpy
git clone https://github.com/vasarlalikhilavinash/blindspot-env || (cd blindspot-env && git pull)
"""
)

md("## 2. Optional secrets")

code(
    """
import os

try:
    from google.colab import userdata
    for key in ["OPENAI_API_KEY", "HF_INFERENCE_ENDPOINT", "HF_TOKEN"]:
        try:
            value = userdata.get(key)
            if value:
                os.environ[key] = value
        except Exception:
            pass
except Exception:
    pass

print({
    "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
    "HF_INFERENCE_ENDPOINT": bool(os.environ.get("HF_INFERENCE_ENDPOINT")),
    "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
})
"""
)

md("## 3. Launch the same app used in the Hugging Face Space")

code(
    """
import sys
sys.path.insert(0, 'blindspot-env')

from spaces.app import demo_engine, ui

print(f"users={len(demo_engine.users)}  concepts={len(demo_engine.catalog)}")
print("Launching the exact Space UI with real-user, persona, paragraph, catalog, and about tabs...")

ui.launch(share=True, debug=False)
"""
)


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = REPO_ROOT / "notebooks" / "03_demo.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(CELLS)} cells")