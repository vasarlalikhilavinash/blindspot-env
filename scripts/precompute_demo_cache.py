#!/usr/bin/env python3
"""Run the trained policy on every cacheable input and save responses.

This makes the hosted Space serve trained-policy results with ZERO GPU at
request time — the GPU is only needed once, here. Output: data/demo_cache.json

Cacheable inputs:
  - All 17 real users  (key = "user::<user_id>")
  - The 3 hard-coded personas in PERSONAS  (key = "persona::<name>")

Run AFTER training on the same Colab session that has the trained adapter loaded.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from unsloth import FastLanguageModel  # type: ignore

from scripts.blindspot_demo import BlindspotDemo

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "training/checkpoints/grpo")
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-9B-bnb-4bit")
OUT = REPO_ROOT / "data" / "demo_cache.json"


# Same personas as notebooks/03_demo.ipynb — keep these in sync.
PERSONAS = {
    "healthcare_ai_lead": (
        "I lead an applied AI team at a healthcare company. We build clinical decision-support "
        "agents using LLMs and RAG over electronic health records. Lately I worry about "
        "hallucination in long-form medical answers and whether our QA evaluations are even "
        "meaningful without expensive doctor reviews."
    ),
    "fintech_ml_engineer": (
        "I am a senior ML engineer at a fintech, building retrieval-augmented LLM systems for "
        "compliance Q&A. I work with embeddings, vector search, and prompt engineering. I want "
        "to know what I am missing — agentic patterns? new fine-tuning tricks? something else?"
    ),
    "bio_ai_founder": (
        "I am a founder building protein-design LLMs for early-stage drug discovery. We "
        "fine-tune on assay data and use RAG over molecular databases. I am drowning in arxiv "
        "and feel like I am always 6 months behind on alignment / evaluation / safety methods."
    ),
}


def main():
    print(f"loading {BASE_MODEL} + adapter {ADAPTER_PATH} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=4096 + 128, load_in_4bit=True
    )
    if Path(ADAPTER_PATH).is_dir():
        model.load_adapter(ADAPTER_PATH)
        print("✓ adapter loaded")
    else:
        print(f"⚠️ adapter dir not found at {ADAPTER_PATH} — caching base-model responses")
    FastLanguageModel.for_inference(model)

    def llm_generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.0)
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)

    demo = BlindspotDemo(llm_generate=llm_generate)

    cache: dict = {}

    # 1. All 17 real users
    print(f"\nrunning trained policy on {len(demo.users)} real users...")
    for i, uid in enumerate(demo.users.keys()):
        try:
            r = demo.compare_all(user_id=uid)
            blind = r["policies"]["Blindspot RL"]
            cache[f"user::{uid}"] = {
                "surfaced": blind["surfaced"],
                "reasoning": blind["meta"].get("reasoning", "")[:600],
                "reward_total": blind["reward"]["total"],
            }
            print(f"  [{i+1}/{len(demo.users)}] user {uid}: surfaced {blind['surfaced']} "
                  f"(reward {blind['reward']['total']:+.2f})")
        except Exception as e:
            print(f"  [{i+1}] user {uid} FAILED: {e}")

    # 2. The 3 personas
    print(f"\nrunning trained policy on {len(PERSONAS)} personas...")
    for name, paragraph in PERSONAS.items():
        try:
            r = demo.compare_all(paragraph=paragraph, persona_key=name)
            blind = r["policies"]["Blindspot RL"]
            cache[f"persona::{name}"] = {
                "surfaced": blind["surfaced"],
                "reasoning": blind["meta"].get("reasoning", "")[:600],
                "reward_total": blind["reward"]["total"],
                "matched_user": r["profile"]["matched_user_id"],
            }
            print(f"  persona {name}: surfaced {blind['surfaced']} "
                  f"(reward {blind['reward']['total']:+.2f})")
        except Exception as e:
            print(f"  persona {name} FAILED: {e}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(cache, indent=2))
    print(f"\n✓ wrote {OUT} with {len(cache)} entries")


if __name__ == "__main__":
    main()
