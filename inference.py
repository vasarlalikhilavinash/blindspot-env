#!/usr/bin/env python3
"""Baseline inference for the Blindspot environment.

Uses an OpenAI-compatible LLM to drive a full agent loop against a
running Blindspot server (default: http://localhost:8000).

Required env vars:
    API_BASE_URL   - LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     - Model identifier (default: gpt-4o-mini)
    HF_TOKEN       - API key
    ENV_URL        - Blindspot server URL (default: http://localhost:8000)
    USERS          - Comma-separated list of user_ids to evaluate (default: all)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000").rstrip("/")
USERS = [u for u in os.environ.get("USERS", "").split(",") if u]
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
DEBUG = os.environ.get("DEBUG", "false").lower() in ("true", "1")

MAX_STEPS = 30


SYSTEM_PROMPT = """\
You are an expert research-onboarding assistant. Your job is to surface concepts
that a researcher SHOULD be tracking but currently isn't.

Each turn, respond with EXACTLY ONE JSON command — no other text:

  {"type": "inspect", "concept_id": <int>}   # peek at reading path + abstract
  {"type": "surface", "concept_id": <int>}   # commit a recommendation (locked-in)
  {"type": "stop"}                             # end episode now

Strategy:
1. Read the user_summary carefully — what sub-fields do they work in?
2. Inspect promising concepts (budget: 15) to confirm relevance.
3. Surface up to 10 concepts that:
   - match the user's research area (highest reward signal)
   - are NOT already trending (novelty bonus)
   - have a coherent reading path (onboarding bonus)
4. AVOID surfacing trending bait or noise — false positives are penalized.
5. Stop early if you've found your best candidates.
"""


_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return _client


# ---------------------------------------------------------------------------
# HTTP transport — direct REST against /reset and /step
# ---------------------------------------------------------------------------


def env_reset(user_id: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if user_id is not None:
        payload["user_id"] = user_id
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{ENV_URL}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state() -> Dict[str, Any]:
    r = requests.get(f"{ENV_URL}/state", timeout=10)
    r.raise_for_status()
    return r.json()


def env_users() -> List[str]:
    payload = env_reset()  # any reset returns user_id_pool
    obs = payload.get("observation", payload)
    return obs.get("user_id_pool", [])


# ---------------------------------------------------------------------------
# Observation rendering
# ---------------------------------------------------------------------------


def render_obs_for_llm(obs: Dict[str, Any]) -> str:
    cands = obs.get("candidate_concepts", [])
    inspected = obs.get("inspected", {})
    surfaced = obs.get("surfaced", [])

    cand_lines = []
    for c in cands:
        cid = c["concept_id"]
        marker = ""
        if cid in surfaced:
            marker = " [SURFACED]"
        elif str(cid) in inspected:
            marker = " [INSPECTED]"
        cand_lines.append(f"  - id={cid}: {c['title']} | {c['one_liner']}{marker}")

    insp_lines = []
    for cid_str, det in inspected.items():
        papers = det.get("top_papers", [])
        paper_str = "; ".join(p.get("title", "") for p in papers[:3])
        insp_lines.append(
            f"  [{cid_str}] {det['title']} (trending={det.get('is_trending')}, "
            f"growth={det.get('growth_signal'):.2f})\n"
            f"    summary: {det.get('abstract_summary','')[:200]}\n"
            f"    papers: {paper_str}"
        )

    return (
        f"USER PROFILE:\n{obs.get('user_summary','')}\n\n"
        f"BUDGETS: inspect={obs.get('inspect_budget_remaining')}, "
        f"surface={obs.get('surface_budget_remaining')}\n\n"
        f"SURFACED SO FAR: {surfaced}\n\n"
        f"INSPECTED DETAILS:\n" + ("\n".join(insp_lines) if insp_lines else "  (none yet)") + "\n\n"
        f"CANDIDATE CONCEPTS:\n" + "\n".join(cand_lines)
    )


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first JSON-ish blob
    for m in _JSON_RE.finditer(text):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------


def run_episode(user_id: str, seed: int = 0) -> Dict[str, Any]:
    reset_payload = env_reset(user_id=user_id, seed=seed)
    obs = reset_payload.get("observation", reset_payload)
    print(f"\n{'='*60}\nUSER: {obs.get('user_id')}\n{'='*60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_obs_for_llm(obs)},
    ]

    client = get_client()
    cumulative = 0.0
    final = None
    for step in range(MAX_STEPS):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            print(f"  LLM error: {e}", file=sys.stderr)
            break
        text = resp.choices[0].message.content or ""
        action = parse_action(text)
        if not action or "type" not in action:
            print(f"  step {step}: failed to parse action: {text[:120]!r}")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Reply with EXACTLY one JSON command. No prose."})
            continue

        if DEBUG:
            print(f"  step {step}: {action}")

        result = env_step(action)
        new_obs = result.get("observation", result)
        reward = result.get("reward", 0.0) or 0.0
        done = result.get("done", False)
        cumulative += reward

        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": render_obs_for_llm(new_obs)})

        if done:
            final = new_obs
            break

    breakdown = (final or {}).get("reward_breakdown") or {}
    print(
        f"  RESULT: total={breakdown.get('total', cumulative):.3f} "
        f"adoption={breakdown.get('adoption',0):.3f} "
        f"novelty={breakdown.get('novelty',0):.3f} "
        f"onboarding={breakdown.get('onboarding',0):.3f} "
        f"efficiency={breakdown.get('efficiency',0):.3f} "
        f"false_positive={breakdown.get('false_positive',0):.3f}"
    )
    return {
        "user_id": user_id,
        "cumulative_step_reward": cumulative,
        "breakdown": breakdown,
    }


def main():
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN / OPENAI_API_KEY not set — LLM calls will fail.", file=sys.stderr)

    users = USERS or env_users()
    print(f"Evaluating {len(users)} users with model={MODEL_NAME} against {ENV_URL}")
    results = []
    for uid in users:
        try:
            results.append(run_episode(uid))
        except Exception as e:
            print(f"Episode for {uid} failed: {e}", file=sys.stderr)

    if results:
        avg = sum(r["breakdown"].get("total", 0.0) for r in results) / len(results)
        print(f"\n=== AVG TOTAL REWARD across {len(results)} users: {avg:.3f} ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
