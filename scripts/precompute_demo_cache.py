#!/usr/bin/env python3
"""Run the trained policy on every cacheable input and save responses.

This makes the hosted Space serve trained-policy results with ZERO GPU at
request time — the GPU is only needed once, here.

Outputs TWO caches so the Before/After toggle on the demo shows the real lift:
  - data/demo_cache.json          (LoRA adapter ON  — trained Blindspot policy)
  - data/demo_cache_pretrain.json (LoRA adapter OFF — base Qwen, no Blindspot reward)

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
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
OUT_TRAINED  = REPO_ROOT / "data" / "demo_cache.json"
OUT_PRETRAIN = REPO_ROOT / "data" / "demo_cache_pretrain.json"


# Same personas as notebooks/03_demo.ipynb — keep these in sync.
PERSONAS = {
    "llm_agents_researcher": (
        "I am a researcher at a frontier-model lab working on LLM agents — tool use, "
        "long-horizon planning, multi-agent coordination, and RL fine-tuning of large "
        "language models. I read arxiv daily but feel like I'm missing important work "
        "outside the agents/RL bubble that would change how I design my training loops."
    ),
    "diffusion_phd_student": (
        "I am a PhD student working on diffusion models for image and video generation. "
        "I focus on architectures, sampling efficiency, and conditional generation. I've "
        "been heads-down on diffusion for two years and worry I'm missing important "
        "developments in alternative generative paradigms or in evaluation methodology."
    ),
    "ml_infra_engineer": (
        "I am an ML infrastructure engineer at a 50-person AI startup. I build training "
        "and inference platforms — distributed training, KV-cache management, model "
        "serving, observability. I want to know which research papers will actually "
        "change my infra stack in the next 6 months, not which ones go viral on Twitter."
    ),
}


def main():
    print(f"loading {BASE_MODEL} + adapter {ADAPTER_PATH} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=4096 + 128, load_in_4bit=True
    )
    has_adapter = Path(ADAPTER_PATH).is_dir()
    if has_adapter:
        model.load_adapter(ADAPTER_PATH)
        print("✓ adapter loaded")
    else:
        print(f"⚠️ adapter dir not found at {ADAPTER_PATH} — caching base-model responses only")
    FastLanguageModel.for_inference(model)

    def llm_generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.0)
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)

    def build_cache(label: str) -> dict:
        demo = BlindspotDemo(llm_generate=llm_generate)
        cache: dict = {}
        # 1. All real users
        print(f"\n[{label}] running on {len(demo.users)} real users...")
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
        # 2. Personas
        print(f"\n[{label}] running on {len(PERSONAS)} personas...")
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
        return cache

    OUT_TRAINED.parent.mkdir(exist_ok=True)

    # Pass 1 — TRAINED policy (adapter ON, if present)
    trained_cache = build_cache("TRAINED")
    OUT_TRAINED.write_text(json.dumps(trained_cache, indent=2))
    print(f"\n✓ wrote {OUT_TRAINED} with {len(trained_cache)} entries")

    # Pass 2 — PRE-TRAINING policy (adapter OFF) for the Before/After toggle
    if has_adapter:
        try:
            print("\n=== switching to PRE-TRAINING (disabling adapter) ===")
            model.disable_adapters()
            pretrain_cache = build_cache("PRETRAIN")
            OUT_PRETRAIN.write_text(json.dumps(pretrain_cache, indent=2))
            print(f"\n✓ wrote {OUT_PRETRAIN} with {len(pretrain_cache)} entries")
        except Exception as e:
            print(f"⚠️ pre-training cache step failed: {e}")
        finally:
            try:
                model.enable_adapters()
            except Exception:
                pass
    else:
        print("(skipping pre-training cache — no adapter to disable)")


if __name__ == "__main__":
    main()
