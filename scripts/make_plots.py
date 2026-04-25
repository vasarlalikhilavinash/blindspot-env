#!/usr/bin/env python3
"""Generate the publication plots in plots/.

Plots:
    plots/baseline_comparison.png     bar chart of mean reward by policy
    plots/reward_decomposition.png    stacked-bar of components
    plots/learning_curve_demo.png     synthetic GRPO learning curve mock-up
                                      (real curve replaces this after training)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

from server.blindspot_environment import BlindspotEnvironment
from baselines import random_baseline, trending_baseline, dense_retrieval_baseline
from models import BlindspotAction


PLOTS = REPO_ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def oracle_episode(env, user_id, seed):
    """Upper-bound: surface concepts ranked by true reward contribution."""
    obs = env.reset(seed=seed, user_id=user_id)
    data = env._data  # noqa: SLF001
    adoption = data.adoption.get(user_id, {})
    comprehension = data.comprehension.get(user_id, {})
    scored = []
    for c in obs.candidate_concepts:
        cid = c.concept_id
        a = float(adoption.get(cid, 0.0))
        n = 0.5 if (a > 0 and bool(data.novelty.get(cid, False))) else 0.0
        o = float(comprehension.get(cid, 0.0)) if a > 0 else 0.0
        fp = -0.1 if a == 0.0 else 0.0
        scored.append((a + n + o + fp, cid))
    scored.sort(reverse=True)
    for _, cid in scored[:10]:
        obs = env.step(BlindspotAction(type="surface", concept_id=cid))
        if obs.done:
            break
    if not obs.done:
        obs = env.step(BlindspotAction(type="stop"))
    return obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}


def evaluate(name, fn, env, users, seeds):
    rows = []
    for uid in users:
        for s in range(seeds):
            br = fn(env, uid, seed=s)
            rows.append(br)
    return {
        "name": name,
        "total": statistics.mean(b.get("total", 0.0) for b in rows),
        "adoption": statistics.mean(b.get("adoption", 0.0) for b in rows),
        "novelty": statistics.mean(b.get("novelty", 0.0) for b in rows),
        "onboarding": statistics.mean(b.get("onboarding", 0.0) for b in rows),
        "efficiency": statistics.mean(b.get("efficiency", 0.0) for b in rows),
        "false_positive": statistics.mean(b.get("false_positive", 0.0) for b in rows),
        "std": statistics.stdev(b.get("total", 0.0) for b in rows) if len(rows) > 1 else 0.0,
    }


def main():
    env = BlindspotEnvironment()
    users = env._data.user_ids  # noqa: SLF001
    seeds = 10

    policies = [
        ("Random", random_baseline.run_episode),
        ("Trending", trending_baseline.run_episode),
        ("Dense Retrieval", dense_retrieval_baseline.run_episode),
        ("Oracle (upper bound)", oracle_episode),
    ]
    results = [evaluate(name, fn, env, users, seeds) for name, fn in policies]

    # ---- Plot 1: bar chart of mean reward ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r["name"] for r in results]
    totals = [r["total"] for r in results]
    stds = [r["std"] for r in results]
    colors = ["#888888", "#cc7733", "#3377cc", "#22aa66"]
    ax.bar(names, totals, yerr=stds, capsize=6, color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Blindspot baselines vs oracle (synthetic seed data)")
    for i, t in enumerate(totals):
        ax.text(i, t + 0.2, f"{t:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS / "baseline_comparison.png", dpi=150)
    plt.close()

    # ---- Plot 2: reward decomposition stacked bar ----
    fig, ax = plt.subplots(figsize=(8, 4.8))
    components = ["adoption", "novelty", "onboarding", "efficiency", "false_positive"]
    bottoms_pos = [0.0] * len(results)
    bottoms_neg = [0.0] * len(results)
    palette = {
        "adoption": "#22aa66", "novelty": "#3377cc", "onboarding": "#9966cc",
        "efficiency": "#888888", "false_positive": "#cc3333",
    }
    for comp in components:
        vals = [r[comp] for r in results]
        bottoms = [bottoms_pos[i] if v >= 0 else bottoms_neg[i] for i, v in enumerate(vals)]
        ax.bar(names, vals, bottom=bottoms, label=comp.replace("_", " "), color=palette[comp])
        for i, v in enumerate(vals):
            if v >= 0:
                bottoms_pos[i] += v
            else:
                bottoms_neg[i] += v
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean reward (per-component)")
    ax.set_title("Reward decomposition by policy")
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS / "reward_decomposition.png", dpi=150)
    plt.close()

    # ---- Plot 3: synthetic learning curve placeholder ----
    # This is replaced by the actual reward+loss curve produced by
    # GRPO training (training/checkpoints/grpo/trainer_state.json).
    grpo_log = REPO_ROOT / "training" / "checkpoints" / "grpo" / "trainer_state.json"
    if grpo_log.exists():
        state = json.loads(grpo_log.read_text())
        steps = [h["step"] for h in state.get("log_history", []) if "reward" in h]
        rewards = [h["reward"] for h in state.get("log_history", []) if "reward" in h]
        losses = [h["loss"] for h in state.get("log_history", []) if "loss" in h]
        steps_loss = [h["step"] for h in state.get("log_history", []) if "loss" in h]
    else:
        # Synthetic mock-up — clearly labelled
        import math
        steps = list(range(0, 401, 20))
        rewards = [results[0]["total"] + (results[3]["total"] - results[0]["total"]) * (1 - math.exp(-s/120)) for s in steps]
        steps_loss = steps
        losses = [1.5 * math.exp(-s/100) + 0.1 for s in steps]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, rewards, color="#22aa66", linewidth=2)
    axes[0].axhline(results[0]["total"], color="#888", linestyle="--", label="random")
    axes[0].axhline(results[1]["total"], color="#cc7733", linestyle="--", label="trending")
    axes[0].axhline(results[3]["total"], color="#22aa66", linestyle=":", label="oracle ceiling")
    axes[0].set_xlabel("GRPO step"); axes[0].set_ylabel("mean episode reward")
    axes[0].set_title("Reward curve" + (" (placeholder)" if not grpo_log.exists() else ""))
    axes[0].legend(fontsize=9)
    axes[1].plot(steps_loss, losses, color="#3377cc", linewidth=2)
    axes[1].set_xlabel("GRPO step"); axes[1].set_ylabel("loss")
    axes[1].set_title("Loss curve" + (" (placeholder)" if not grpo_log.exists() else ""))
    plt.tight_layout()
    plt.savefig(PLOTS / "learning_curve.png", dpi=150)
    plt.close()

    (PLOTS / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote plots to {PLOTS}/")
    for r in results:
        print(f"  {r['name']:>22s}  total={r['total']:+.3f}  std={r['std']:.3f}")


if __name__ == "__main__":
    main()
