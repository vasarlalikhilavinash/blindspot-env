#!/usr/bin/env python3
"""Generate final_comparison.png and sft_loss.png from eval_results.json.

Usage:
    python training/plot_results.py
    # Reads:  plots/eval_results.json, plots/trainer_state.json
    # Writes: plots/final_comparison.png, plots/sft_loss.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "plots"


def load_eval_results() -> dict:
    path = PLOTS_DIR / "eval_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run training/eval.py first.")
        sys.exit(1)
    return json.loads(path.read_text())


def plot_comparison(results: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(results.keys())
    means, stds = [], []
    for name in labels:
        rewards = results[name]
        if rewards:
            m = sum(rewards) / len(rewards)
            s = math.sqrt(sum((r - m) ** 2 for r in rewards) / len(rewards))
        else:
            m, s = 0.0, 0.0
        means.append(m)
        stds.append(s)

    x = np.arange(len(labels))
    colors = ["#e05c5c", "#f0a500", "#3377cc"]
    colors = colors[: len(labels)] + ["#3377cc"] * max(0, len(labels) - 3)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=0.8, error_kw={"ecolor": "#333"})

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace("_", "\n") for lbl in labels], fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Blindspot — Policy Comparison\n(mean ± std, 13 train users)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.03,
            f"{mean:+.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    out = PLOTS_DIR / "final_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out}")


def plot_sft_loss() -> None:
    import matplotlib.pyplot as plt

    state_path = PLOTS_DIR / "trainer_state.json"
    if not state_path.exists():
        print(f"(trainer_state.json not found, skipping sft_loss.png)")
        return

    state = json.loads(state_path.read_text())
    history = state.get("log_history", [])
    steps = [e["step"] for e in history if "loss" in e]
    losses = [e["loss"] for e in history if "loss" in e]
    if not steps:
        print("(no loss entries in trainer_state.json, skipping sft_loss.png)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, color="#3377cc", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Training Loss — Blindspot Qwen2.5-1.5B")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "sft_loss.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_eval_results()

    print("=== Eval Results ===")
    for name, rewards in results.items():
        if rewards:
            m = sum(rewards) / len(rewards)
            s = math.sqrt(sum((r - m) ** 2 for r in rewards) / len(rewards))
            print(f"  {name:12s}: {m:+.3f} ± {s:.3f}  (n={len(rewards)})")
        else:
            print(f"  {name:12s}: no data")

    plot_comparison(results)
    plot_sft_loss()


if __name__ == "__main__":
    main()
