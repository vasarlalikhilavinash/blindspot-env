#!/usr/bin/env python3
"""SFT fine-tune Qwen3.5-9B on Blindspot oracle traces.

Two backends supported, picked by --backend:

  mlx          : Apple Silicon (M-series) via mlx-lm  — recommended on M5 Pro 48GB
                 https://github.com/ml-explore/mlx-examples
  transformers : NVIDIA CUDA via transformers + peft (LoRA, QLoRA)

The training data is `training/sft_traces.jsonl` produced by
`generate_sft_traces.py`. Output adapter is saved to `training/checkpoints/sft/`.

Example (Apple Silicon):
    python -m mlx_lm.lora \
        --model unsloth/Qwen3.5-9B-bnb-4bit \
        --train --data training/sft_traces.jsonl \
        --batch-size 1 --iters 600 \
        --adapter-path training/checkpoints/sft

Example (CUDA):
    python training/sft_train.py --backend transformers \
        --base-model unsloth/Qwen3.5-9B-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "training" / "sft_traces.jsonl"
CKPT = REPO_ROOT / "training" / "checkpoints" / "sft"


def run_mlx(args):
    CKPT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.base_model,
        "--train",
        "--data", str(DATA.parent),  # mlx_lm expects a directory containing train.jsonl
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--adapter-path", str(CKPT),
        "--learning-rate", str(args.lr),
    ]
    # mlx_lm reads train.jsonl/valid.jsonl in --data dir
    train_dst = DATA.parent / "train.jsonl"
    if not train_dst.exists():
        train_dst.symlink_to(DATA.name)
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def run_transformers(args):
    """LoRA SFT via transformers + peft + trl SFTTrainer."""
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    rows = []
    with DATA.open() as f:
        for line in f:
            rec = json.loads(line)
            rows.append({"messages": rec["messages"]})
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    cfg = SFTConfig(
        output_dir=str(CKPT),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        gradient_accumulation_steps=4,
        bf16=True,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(str(CKPT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    ap.add_argument("--base-model", default="unsloth/Qwen3.5-9B-bnb-4bit")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    if not DATA.exists():
        sys.exit(f"Missing {DATA}. Run training/generate_sft_traces.py first.")

    if args.backend == "mlx":
        run_mlx(args)
    else:
        run_transformers(args)


if __name__ == "__main__":
    main()
