#!/usr/bin/env python3
"""SFT fine-tune on Blindspot expert traces using TRL SFTTrainer + Unsloth.

Train unsloth/Qwen2.5-1.5B-Instruct on data/sft_traces.jsonl for 1 epoch.
Saves adapter to ./blindspot-sft-final and pushes to HF Hub.

Usage (on Colab A100 after running generate_sft_traces.py):
    python training/sft_train.py
    # or with custom HF token:
    HF_TOKEN=hf_xxx python training/sft_train.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HUB_REPO = "Vasarlaavinash/blindspot-sft-1.5b"
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-1.5B-Instruct")
DATA_FILE = str(REPO_ROOT / "data" / "sft_traces.jsonl")
OUTPUT_DIR = "./blindspot-sft"
FINAL_DIR = "./blindspot-sft-final"
PLOTS_DIR = REPO_ROOT / "plots"


def main():
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATA_FILE} ...")
    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"  {len(ds)} traces loaded")

    print(f"Loading {BASE_MODEL} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def format_trace(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(format_trace, remove_columns=ds.column_names)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        max_seq_length=2048,
        logging_steps=10,
        save_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_config,
    )

    FastLanguageModel.for_training(model)
    print(f"\nStarting SFT: 1 epoch on {len(ds)} examples ...")
    trainer.train()
    print("✓ Training complete")

    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"✓ Adapter saved to {FINAL_DIR}")

    log_history = trainer.state.log_history
    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    if steps:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, losses, color="#3377cc", lw=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("SFT Training Loss — Blindspot Qwen2.5-1.5B")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "sft_loss.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Loss plot saved to plots/sft_loss.png")
        import shutil
        state_path = Path(OUTPUT_DIR) / "trainer_state.json"
        if state_path.exists():
            shutil.copy(state_path, PLOTS_DIR / "trainer_state.json")

    if HF_TOKEN:
        print(f"Pushing adapter to {HUB_REPO} ...")
        trainer.model.push_to_hub(HUB_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(HUB_REPO, token=HF_TOKEN)
        print(f"✓ Pushed to https://huggingface.co/{HUB_REPO}")
    else:
        print("(skipping Hub push — set HF_TOKEN env var to enable)")


if __name__ == "__main__":
    main()
