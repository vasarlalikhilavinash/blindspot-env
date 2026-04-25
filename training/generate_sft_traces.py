#!/usr/bin/env python3
"""Generate SFT traces by running an oracle policy and recording the
(observation, action) sequence in chat format.

The oracle has full information access — it knows which concepts are
adopted by the user, which are novel, and the comprehension scores —
so it produces near-optimal trajectories. We then optionally rewrite
each action with an LLM to produce CoT-style rationales (Ollama by
default to keep this local and free).

Output:
    training/sft_traces.jsonl   one trace per line, OpenAI chat format
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import BlindspotAction
from server.blindspot_environment import BlindspotEnvironment


SYSTEM_PROMPT = (
    "You are an expert research-onboarding assistant. Each turn respond with "
    "EXACTLY ONE JSON command: {\"type\": \"inspect\"|\"surface\"|\"stop\", \"concept_id\": <int|null>}. "
    "Maximize adoption + novelty + onboarding while avoiding false positives."
)


def render_obs(obs) -> str:
    """Compact renderer matching inference.py."""
    cands = obs.candidate_concepts
    insp = obs.inspected
    surf = list(obs.surfaced)
    cand_lines = []
    for c in cands[:50]:
        marker = ""
        if c.concept_id in surf:
            marker = " [SURFACED]"
        elif str(c.concept_id) in insp:
            marker = " [INSPECTED]"
        cand_lines.append(f"  - id={c.concept_id}: {c.title} | {c.one_liner}{marker}")
    return (
        f"USER PROFILE:\n{obs.user_summary}\n\n"
        f"BUDGETS: inspect={obs.inspect_budget_remaining}, "
        f"surface={obs.surface_budget_remaining}\n\n"
        f"SURFACED: {surf}\n\n"
        f"CANDIDATES:\n" + "\n".join(cand_lines)
    )


def oracle_actions(env: BlindspotEnvironment, obs) -> List[Dict[str, Any]]:
    """Generate the oracle action sequence for this episode.

    Strategy:
      1. Rank candidates by (adoption_score + 0.5*novelty + onboarding − 0.1*not_adopted)
      2. Inspect top 10 (cheap signal-gathering — fewer than budget=15)
      3. Surface top 10 by score
      4. Stop
    """
    data = env._data  # noqa: SLF001
    uid = obs.user_id
    adoption = data.adoption.get(uid, {})
    comprehension = data.comprehension.get(uid, {})

    scored = []
    for c in obs.candidate_concepts:
        cid = c.concept_id
        a = float(adoption.get(cid, 0.0))
        n = 0.5 if (a > 0 and bool(data.novelty.get(cid, False))) else 0.0
        o = float(comprehension.get(cid, 0.0)) if a > 0 else 0.0
        fp = -0.1 if a == 0.0 else 0.0
        scored.append((a + n + o + fp, cid))
    scored.sort(reverse=True)

    actions = []
    inspect_targets = [cid for _, cid in scored[:10]]
    surface_targets = [cid for _, cid in scored[:10]]
    for cid in inspect_targets:
        actions.append({"type": "inspect", "concept_id": cid})
    for cid in surface_targets:
        actions.append({"type": "surface", "concept_id": cid})
    actions.append({"type": "stop", "concept_id": None})
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "training" / "sft_traces.jsonl"))
    ap.add_argument("--seeds-per-user", type=int, default=3)
    ap.add_argument("--rewrite-with-ollama", action="store_true",
                    help="If set, ask Ollama (llama3.1:8b by default) to add a one-line rationale.")
    ap.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    args = ap.parse_args()

    env = BlindspotEnvironment()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traces = 0
    with out_path.open("w", encoding="utf-8") as f:
        for uid in env._data.user_ids:  # noqa: SLF001
            for seed in range(args.seeds_per_user):
                obs = env.reset(seed=seed, user_id=uid)
                actions = oracle_actions(env, obs)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_obs(obs)},
                ]
                for action in actions:
                    content = json.dumps({k: v for k, v in action.items() if v is not None})
                    if args.rewrite_with_ollama:
                        content = _rewrite_with_ollama(args.ollama_url, args.ollama_model, content)
                    messages.append({"role": "assistant", "content": content})
                    obs = env.step(BlindspotAction(**{k: v for k, v in action.items() if v is not None}))
                    if obs.done:
                        break
                    messages.append({"role": "user", "content": render_obs(obs)})
                f.write(json.dumps({
                    "user_id": uid,
                    "seed": seed,
                    "messages": messages,
                    "final_reward": (obs.reward_breakdown.total if obs.reward_breakdown else 0.0),
                }) + "\n")
                n_traces += 1
    print(f"Wrote {n_traces} traces to {out_path}")


def _rewrite_with_ollama(url, model, action_json: str) -> str:
    """Optionally wrap action in a one-line rationale via local Ollama."""
    try:
        import requests
        prompt = (
            "Rewrite this action as <reasoning>one-line reason</reasoning>"
            f"<action>{action_json}</action>. Keep the JSON inside <action> identical."
        )
        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", action_json).strip()
    except Exception:
        return action_json


if __name__ == "__main__":
    main()
