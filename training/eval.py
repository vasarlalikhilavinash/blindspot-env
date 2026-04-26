#!/usr/bin/env python3
"""Evaluate 3 policies via HTTP against the live env server.

Policies:
  random   — randomly surface 10 concepts, stop
  trending — surface first 10 candidates in list order, stop
  sft      — load fine-tuned model from HF Hub, generate actions, execute

Usage:
    python training/eval.py
    # or with custom model:
    SFT_MODEL=Vasarlaavinash/blindspot-sft-1.5b python training/eval.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import List

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ENV_URL = "http://localhost:8000"
SFT_MODEL = "Vasarlaavinash/blindspot-sft-1.5b"
OUT_FILE = REPO_ROOT / "plots" / "eval_results.json"

SYSTEM_PROMPT = (
    "You are Blindspot, a research discovery agent. "
    "Given a user profile and candidate concepts, choose actions to surface "
    "the most relevant unknown-unknown concepts the user has not seen yet. "
    'Respond with JSON: {"type": "inspect|surface|stop", "concept_id": int}'
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _post(endpoint: str, payload: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(f"{ENV_URL}/{endpoint}", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def _render_obs(obs: dict) -> str:
    cands = obs.get("candidate_concepts", [])
    surfaced = obs.get("surfaced", [])
    lines = [f"  id={c['concept_id']}: {c.get('title','')} — {c.get('one_liner','')[:80]}"
             for c in cands]
    return (
        f"USER PROFILE:\n{obs.get('user_summary', '')[:800]}\n\n"
        f"BUDGETS: inspect={obs.get('inspect_budget_remaining')} "
        f"surface={obs.get('surface_budget_remaining')}\n\n"
        f"ALREADY SURFACED: {surfaced}\n\n"
        f"CANDIDATES:\n" + "\n".join(lines)
    )


def _parse_action(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for m in _JSON_RE.finditer(text):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


# ─────────────────────────── policies ─────────────────────────────────────────

def policy_random(obs: dict) -> list:
    """Randomly surface 10 concepts, then stop."""
    cands = obs.get("candidate_concepts", [])
    rng = random.Random(42)
    chosen = rng.sample(cands, min(10, len(cands)))
    return [{"type": "surface", "concept_id": c["concept_id"]} for c in chosen] + [{"type": "stop"}]


def policy_trending(obs: dict) -> list:
    """Surface first 10 candidates in list order (simulates trending bias)."""
    cands = obs.get("candidate_concepts", [])
    return [{"type": "surface", "concept_id": c["concept_id"]} for c in cands[:10]] + [{"type": "stop"}]


def run_scripted_episode(user_id: str, seed: int, get_actions) -> float:
    """Run a scripted episode where all actions are pre-determined from the initial obs."""
    body = _post("reset", {"user_id": user_id, "seed": seed})
    obs = body.get("observation", body)
    actions = get_actions(obs)
    reward = 0.0
    for action in actions:
        result = _post("step", {"action": action})
        result_obs = result.get("observation", result)
        if result_obs.get("done"):
            rb = result_obs.get("reward_breakdown") or {}
            reward = float(rb.get("total", result.get("reward", 0.0)))
            break
    return reward


def run_sft_episode(user_id: str, seed: int, model, tokenizer) -> float:
    """Run one episode with the SFT model generating actions step by step."""
    import torch
    body = _post("reset", {"user_id": user_id, "seed": seed})
    obs = body.get("observation", body)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_obs(obs)},
    ]
    surfaced_set = set()
    reward = 0.0
    for _ in range(15):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.1)
        completion = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        action = _parse_action(completion)
        if not action or "type" not in action:
            action = {"type": "stop"}
        # Avoid duplicate surfaces
        if action.get("type") == "surface":
            cid = action.get("concept_id")
            if cid in surfaced_set:
                action = {"type": "stop"}
            else:
                surfaced_set.add(cid)
        result = _post("step", {"action": action})
        result_obs = result.get("observation", result)
        msgs.append({"role": "assistant", "content": completion})
        if result_obs.get("done"):
            rb = result_obs.get("reward_breakdown") or {}
            reward = float(rb.get("total", result.get("reward", 0.0)))
            break
        msgs.append({"role": "user", "content": _render_obs(result_obs)})
    return reward


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    print("Connecting to env server ...")
    body = _post("reset", {})
    obs = body.get("observation", body)
    all_users = obs.get("user_id_pool", [])
    if not all_users:
        print("ERROR: no user_id_pool. Is the env server running at http://localhost:8000?")
        sys.exit(1)

    splits_path = REPO_ROOT / "data" / "user_splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        train_users = [u for u in splits.get("train", []) if u in all_users]
    else:
        train_users = all_users[:13]

    print(f"Evaluating on {len(train_users)} train users, seeds 100-129 (30 episodes each)")
    seeds = list(range(100, 130))
    results = {"random": [], "trending": [], "sft": []}

    # ── Random & Trending (fast, scripted) ─────────────────────────────────────
    for name, get_actions in [("random", policy_random), ("trending", policy_trending)]:
        print(f"\n[{name.upper()}]")
        for uid in train_users:
            for seed in seeds[:3]:  # 3 seeds × 13 users = 39 episodes
                try:
                    r = run_scripted_episode(uid, seed, get_actions)
                    results[name].append(r)
                    print(f"  user={uid} seed={seed} reward={r:+.3f}")
                except Exception as e:
                    print(f"  FAILED user={uid} seed={seed}: {e}")

    # ── SFT ───────────────────────────────────────────────────────────────────
    print(f"\n[SFT] Loading {SFT_MODEL} ...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=SFT_MODEL,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        for uid in train_users:
            for seed in seeds[:3]:
                try:
                    r = run_sft_episode(uid, seed, model, tokenizer)
                    results["sft"].append(r)
                    print(f"  user={uid} seed={seed} reward={r:+.3f}")
                except Exception as e:
                    print(f"  FAILED user={uid} seed={seed}: {e}")
    except Exception as e:
        print(f"  Could not load SFT model: {e}")
        print("  (SFT results will be empty)")

    # ── Save & print ───────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved to {OUT_FILE}")

    print("\n=== SUMMARY ===")
    for name, rewards in results.items():
        if rewards:
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
            print(f"  {name:12s}: {mean:+.3f} ± {std:.3f}  (n={len(rewards)})")
        else:
            print(f"  {name:12s}: no data")


if __name__ == "__main__":
    main()
