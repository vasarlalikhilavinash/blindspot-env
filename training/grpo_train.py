#!/usr/bin/env python3
"""GRPO fine-tune (TRL + Unsloth) for the Blindspot environment.

Designed to run on a single HF A100 (80 GB) — uses Unsloth bf16 LoRA
and TRL's GRPOTrainer. The reward function calls a *running*
Blindspot env over HTTP so policy rollouts are pure lookups.

Usage:
    # 1. Boot the env in another terminal:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # 2. Run training:
    python training/grpo_train.py --base-model unsloth/Qwen3.5-9B

The same script powers `notebooks/02_training.ipynb` (Colab-runnable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import requests

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000").rstrip("/")
SYSTEM_PROMPT = (
    "You are a research-onboarding assistant. Surface concepts a user will likely adopt.\n"
    "Each turn respond with EXACTLY ONE JSON command — no extra text:\n"
    "  {\"type\": \"surface\", \"concept_id\": <id>}  — recommend a concept to the user\n"
    "  {\"type\": \"inspect\", \"concept_id\": <id>}  — read concept details (uses inspect budget)\n"
    "  {\"type\": \"stop\"}                            — end the session\n\n"
    "CRITICAL RULES — violation causes negative reward:\n"
    "1. concept_id MUST be one of the integer ids listed under CANDIDATES in the observation\n"
    "2. NEVER surface or inspect the same concept_id more than once per session\n"
    "3. Choose concepts most relevant to this user's research profile\n"
    "4. Surface at least 3 different concepts before stopping"
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(text: str):
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


def render_obs(obs) -> str:
    cands = obs.get("candidate_concepts", [])
    surf = obs.get("surfaced", [])
    insp = obs.get("inspected", {})
    lines = []
    for c in cands:
        cid = c["concept_id"]
        marker = " [S]" if cid in surf else (" [I]" if str(cid) in insp else "")
        lines.append(f"  id={cid}: {c['title']}{marker}")
    return (
        f"USER:\n{obs.get('user_summary','')[:800]}\n\n"
        f"BUDGETS i={obs.get('inspect_budget_remaining')} s={obs.get('surface_budget_remaining')}\n\n"
        f"SURFACED: {surf}\n\nCANDIDATES:\n" + "\n".join(lines)
    )


def rollout(generator, prompt_msgs, max_steps=30) -> float:
    """Roll out a single episode using the model's `generate` callable.

    `generator(messages) -> str`  — one response per call.
    Returns the final episode total reward.
    """
    # Reset to a deterministic user via /reset
    r = requests.post(f"{ENV_URL}/reset", json={}, timeout=30); r.raise_for_status()
    obs = r.json().get("observation", r.json())
    msgs = list(prompt_msgs) + [{"role": "user", "content": render_obs(obs)}]
    for _ in range(max_steps):
        text = generator(msgs)
        action = parse_action(text)
        if not action or "type" not in action:
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "Reply with EXACTLY one JSON command."})
            continue
        r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30); r.raise_for_status()
        result = r.json()
        new_obs = result.get("observation", result)
        if result.get("done"):
            br = (new_obs or {}).get("reward_breakdown") or {}
            return float(br.get("total", result.get("reward", 0.0) or 0.0))
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user", "content": render_obs(new_obs)})
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="unsloth/Qwen3.5-9B")
    ap.add_argument("--sft-adapter", default="training/checkpoints/sft")
    ap.add_argument("--output", default="training/checkpoints/grpo")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--num-generations", type=int, default=8,
                    help="GRPO group size — number of completions per prompt for relative ranking.")
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--rollout-step-limit", type=int, default=8)
    ap.add_argument("--fallback-base-model", default="unsloth/Qwen3.5-4B")
    args = ap.parse_args()

    # Heavy imports here so --help works without GPU
    from unsloth import FastLanguageModel  # type: ignore
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # type: ignore
    import torch

    if "qwen3_5" not in CONFIG_MAPPING:
        raise RuntimeError(
            "Qwen3.5 requires Transformers v5. Install transformers>=5.2.0 "
            "or git+https://github.com/huggingface/transformers.git, then restart Python."
        )

    def load_base_model(model_name: str):
        return FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.max_prompt_length + args.max_completion_length,
            load_in_4bit=False,
            dtype=torch.bfloat16,
            fast_inference=False,
        )

    try:
        model, tokenizer = load_base_model(args.base_model)
        print(f"Loaded base model: {args.base_model}")
    except (RuntimeError, OSError) as exc:
        message = str(exc)
        missing_config = "No config file found" in message or "is not a local folder" in message
        unsupported_arch = "qwen3_5" in message and ("does not support" in message or "not recognize" in message)
        if unsupported_arch:
            raise RuntimeError(
                "Qwen3.5 requires Transformers v5. Install transformers>=5.2.0 "
                "or git+https://github.com/huggingface/transformers.git, then restart Python."
            ) from exc
        if not missing_config or args.base_model == args.fallback_base_model:
            raise
        print(f"Base model could not be loaded: {args.base_model}")
        print(f"First error line: {message.splitlines()[0]}")
        args.base_model = args.fallback_base_model
        print(f"Falling back to: {args.base_model}")
        model, tokenizer = load_base_model(args.base_model)

    if args.sft_adapter and Path(args.sft_adapter).exists():
        model.load_adapter(args.sft_adapter)
    FastLanguageModel.for_training(model)

    def post_env(endpoint: str, payload: dict):
        resp = requests.post(f"{ENV_URL}/{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        obs = body.get("observation", body) or {}
        return body, obs

    def generate_completion(messages) -> str:
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                inputs,
                max_new_tokens=args.max_completion_length,
                do_sample=False,
                temperature=0.0,
            )
        return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)

    def run_episode(user_id, seed, first_completion=None, first_action=None):
        payload = {}
        if user_id is not None:
            payload["user_id"] = user_id
        if seed is not None:
            payload["seed"] = seed

        _, obs = post_env("reset", payload)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_obs(obs)},
        ]

        invalid_streak = 0
        for step_idx in range(args.rollout_step_limit):
            if step_idx == 0 and first_completion is not None:
                text = first_completion
                action = first_action or parse_action(first_completion)
            else:
                text = generate_completion(messages)
                action = parse_action(text)

            if not action or "type" not in action:
                if step_idx == 0:
                    return -0.05
                messages.append({"role": "assistant", "content": text or ""})
                messages.append({"role": "user", "content": "Reply with EXACTLY one JSON command."})
                invalid_streak += 1
                if invalid_streak >= 2:
                    break
                continue

            result, obs = post_env("step", {"action": action})
            invalid_streak = 0
            messages.append({"role": "assistant", "content": text or ""})
            if result.get("done") or obs.get("done"):
                breakdown = obs.get("reward_breakdown") or {}
                return float(breakdown.get("total", result.get("reward", 0.0) or 0.0))
            messages.append({"role": "user", "content": render_obs(obs)})

        result, obs = post_env("step", {"action": {"type": "stop"}})
        breakdown = obs.get("reward_breakdown") or {}
        return float(breakdown.get("total", result.get("reward", 0.0) or 0.0))

    # Build a tiny "prompt" dataset — each row is a fresh reset; the
    # reward function below resets the env to the SAME user_id+seed
    # before evaluating the action so the prompt context matches the
    # candidate pool the action targets.
    n_prompts = 256
    rows = []
    import random as _rng_mod
    _rng = _rng_mod.Random(0)
    # discover available users
    r0 = requests.post(f"{ENV_URL}/reset", json={}); r0.raise_for_status()
    user_pool = (r0.json().get("observation", r0.json()) or {}).get("user_id_pool", [])
    if not user_pool:
        raise RuntimeError("Env returned empty user_id_pool")

    split_path = REPO_ROOT / "data" / "user_splits.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_users = [uid for uid in split.get("train", []) if uid in user_pool]
    else:
        train_users = list(user_pool)
    if not train_users:
        raise RuntimeError("No training users available for prompt sampling")

    for i in range(n_prompts):
        uid = _rng.choice(train_users)
        seed = _rng.randrange(1_000_000)
        r = requests.post(f"{ENV_URL}/reset", json={"user_id": uid, "seed": seed}); r.raise_for_status()
        obs = r.json().get("observation", r.json())
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_obs(obs)},
            ],
            "user_id": uid,
            "seed": seed,
        })
    ds = Dataset.from_list(rows)

    # -------------------- reward function --------------------
    def reward_fn(prompts, completions, user_id=None, seed=None, **kwargs) -> List[float]:
        """Score each GRPO completion by continuing a short OpenEnv episode.

        The first completion is the candidate action from GRPO. After that,
        the current policy keeps interacting through reset/step until stop,
        done, or the rollout limit. This matches the session-level task.
        """
        rewards = []
        uids = user_id if isinstance(user_id, list) else [user_id] * len(completions)
        seeds = seed if isinstance(seed, list) else [seed] * len(completions)
        for prompt_msgs, completion, uid, sd in zip(prompts, completions, uids, seeds):
            text = completion if isinstance(completion, str) else completion[-1].get("content", "")
            action = parse_action(text)
            if not action or "type" not in action:
                rewards.append(-0.05)
                continue
            try:
                rewards.append(run_episode(uid, sd, first_completion=text, first_action=action))
            except Exception:
                rewards.append(-0.1)
        return rewards

    cfg = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
