#!/usr/bin/env python3
"""Generate 256 SFT training traces using the Dense Retrieval+ heuristic.

Runs the best-measured policy (skip all inspects, surface top-10 by cosine
similarity to user_summary, then stop — +0.547 mean reward) against the live
env server for the 13 training users with seeds 0-19.

Output: data/sft_traces.jsonl  (one JSON object per line)
Each object: {"messages": [{"role": "system"|"user"|"assistant", "content": str}]}
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ENV_URL = "http://localhost:8000"
OUT_FILE = REPO_ROOT / "data" / "sft_traces.jsonl"

SYSTEM_PROMPT = (
    "You are Blindspot, a research discovery agent. "
    "Given a user profile and candidate concepts, choose actions to surface "
    "the most relevant unknown-unknown concepts the user hasn't seen yet. "
    'Respond with JSON: {"type": "inspect|surface|stop", "concept_id": int}'
)

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-_]{2,}")
STOPWORDS = set("""
a an and are as at be by for from has have he her his i in is it its of on or our
that the their them they this to was we were what when where which who will with you
your yours us my me but not no so if then than also can could should would may might
must just over under into out about more most less some any all each every other
""".split())


def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")
            if w.lower() not in STOPWORDS and len(w) > 2]


def _vec(tokens: List[str], vocab: dict) -> List[float]:
    v = [0.0] * len(vocab)
    for t in tokens:
        if t in vocab:
            v[vocab[t]] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cos(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _build_vocab(texts: List[str]) -> dict:
    df: dict = {}
    for text in texts:
        for w in set(_tokenize(text)):
            df[w] = df.get(w, 0) + 1
    # keep top-4000 by document frequency
    top = sorted(df.items(), key=lambda kv: -kv[1])[:4000]
    return {w: i for i, (w, _) in enumerate(top)}


def _post(endpoint: str, payload: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(f"{ENV_URL}/{endpoint}", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} for /{endpoint}: {e}")
            time.sleep(1)


def _render_obs(obs: dict) -> str:
    cands = obs.get("candidate_concepts", [])
    surfaced = obs.get("surfaced", [])
    inspected = obs.get("inspected", {})
    lines = []
    for c in cands:
        cid = c["concept_id"]
        marker = " [SURFACED]" if cid in surfaced else (" [INSPECTED]" if str(cid) in inspected else "")
        lines.append(f"  id={cid}: {c.get('title', '')} — {c.get('one_liner', '')[:80]}{marker}")
    return (
        f"USER PROFILE:\n{obs.get('user_summary', '')[:800]}\n\n"
        f"BUDGETS: inspect={obs.get('inspect_budget_remaining', '?')} "
        f"surface={obs.get('surface_budget_remaining', '?')}\n\n"
        f"ALREADY SURFACED: {surfaced}\n\n"
        f"CANDIDATES (50 concepts):\n" + "\n".join(lines)
    )


def run_dense_noinspect_episode(user_id: str, seed: int) -> Optional[dict]:
    """Run one Dense Retrieval+ episode. Returns the trace dict or None on failure."""
    try:
        # Reset
        body = _post("reset", {"user_id": user_id, "seed": seed})
        obs = body.get("observation", body)

        user_summary = obs.get("user_summary", "")
        cands = obs.get("candidate_concepts", [])
        if not cands:
            return None

        # Build vocab from user_summary + all candidate texts
        all_texts = [user_summary] + [
            f"{c.get('title', '')} {c.get('one_liner', '')}" for c in cands
        ]
        vocab = _build_vocab(all_texts)
        user_vec = _vec(_tokenize(user_summary), vocab)

        # Score candidates by cosine similarity
        scored = []
        for c in cands:
            text = f"{c.get('title', '')} {c.get('one_liner', '')}"
            cvec = _vec(_tokenize(text), vocab)
            scored.append((_cos(user_vec, cvec), c["concept_id"]))
        scored.sort(reverse=True)
        top10 = [cid for _, cid in scored[:10]]

        # Build assistant trace: surface top-10 then stop
        action_lines = []
        for cid in top10:
            action_lines.append(json.dumps({"type": "surface", "concept_id": cid}))
            step_body = _post("step", {"action": {"type": "surface", "concept_id": cid}})
            step_obs = step_body.get("observation", step_body)
            if step_obs.get("done"):
                break
        action_lines.append(json.dumps({"type": "stop"}))
        _post("step", {"action": {"type": "stop"}})

        user_turn = _render_obs(obs)
        assistant_turn = "\n".join(action_lines)

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": assistant_turn},
            ],
            "user_id": user_id,
            "seed": seed,
        }

    except Exception as e:
        print(f"  FAILED user={user_id} seed={seed}: {e}")
        return None


def main():
    # Get training users from env
    print("Connecting to env server...")
    body = _post("reset", {})
    obs = body.get("observation", body)
    all_users = obs.get("user_id_pool", [])
    if not all_users:
        print("ERROR: no user_id_pool in observation. Is the env server running?")
        sys.exit(1)

    # Load train split
    splits_path = REPO_ROOT / "data" / "user_splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        train_users = [u for u in splits.get("train", []) if u in all_users]
    else:
        train_users = all_users[:13]

    print(f"Training users: {len(train_users)} | All users: {len(all_users)}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    total = len(train_users) * 20  # seeds 0-19

    with open(OUT_FILE, "w") as f:
        i = 0
        for uid in train_users:
            for seed in range(20):
                i += 1
                print(f"[{i:3d}/{total}] user={uid} seed={seed} ...", end=" ", flush=True)
                trace = run_dense_noinspect_episode(uid, seed)
                if trace:
                    f.write(json.dumps(trace) + "\n")
                    traces.append(trace)
                    print("ok")
                else:
                    print("skip")

    print(f"\n✓ Wrote {len(traces)} traces to {OUT_FILE}")
    return traces


if __name__ == "__main__":
    main()
