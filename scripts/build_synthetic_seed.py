#!/usr/bin/env python3
"""Build a small deterministic synthetic dataset for Blindspot.

This generates all artifacts the runtime env needs WITHOUT any external
API calls — so a fresh checkout can boot the env in <2 seconds:

    python scripts/build_synthetic_seed.py

For the *real* hackathon dataset (50 Semantic Scholar researchers,
arXiv corpus, KeyBERT concept extraction, two-judge comprehension
scoring) run scripts/run_all_precompute.sh instead.

The synthetic data is realistic enough to:
  - exercise every code path in the env / rewards / client
  - support a meaningful demo run (random vs trained policy comparison)
  - calibrate the false-positive penalty empirically

It is NOT realistic enough to publish RL results from. The README's
results figures must be regenerated against the real precomputed data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# Concept catalog — 200 ML concepts grouped into 8 sub-domains
# ---------------------------------------------------------------------------

SUB_DOMAINS: Dict[str, List[str]] = {
    "alignment": [
        "Constitutional AI", "RLHF", "DPO", "KTO", "ORPO", "RLAIF", "Self-Rewarding",
        "Reward Hacking", "Sycophancy", "Deliberative Alignment", "Iterated Amplification",
        "Debate Protocols", "Process Reward Models", "Outcome Reward Models", "Rule-Based Rewards",
        "Inverse RL from Preferences", "Bradley-Terry Modeling", "Active Preference Learning",
        "Off-Policy Preference Optimization", "GRPO Variants", "PPO Variants",
        "Reward Shaping for LLMs", "Value-Based Alignment", "Robust Reward Modeling",
        "Cooperative Inverse RL",
    ],
    "efficiency": [
        "Speculative Decoding", "Eagle Decoding", "Medusa Heads", "Lookahead Decoding",
        "FlashAttention", "PagedAttention", "RingAttention", "Sparse MoE", "Granite MoE",
        "Quantization-Aware Training", "GPTQ", "AWQ", "SmoothQuant", "FP8 Training",
        "MX Formats", "Sliding Window Attention", "Grouped Query Attention", "Multi-Query Attention",
        "Multi-Latent Attention", "Mamba SSM", "Hyena", "Linear Attention",
        "KV Cache Compression", "H2O Cache Eviction", "StreamingLLM",
    ],
    "agents": [
        "ReAct", "Reflexion", "Toolformer", "Voyager", "AutoGPT", "Tree-of-Thoughts",
        "Graph-of-Thoughts", "Plan-and-Solve", "Self-Refine", "Constitutional Self-Critique",
        "OpenAgents", "Multi-Agent Debate", "Agent Forest", "AutoGen Patterns",
        "MCP Protocol", "Function Calling Bench", "Toolbench", "AgentBench",
        "Long-Horizon Planning", "Hierarchical Planning", "World Model Planning",
        "Latent Action Spaces", "Skill Library Composition", "Memory-Augmented Agents",
        "Episodic Memory Stores",
    ],
    "retrieval": [
        "ColBERTv2", "GritLM", "Matryoshka Embeddings", "ColPali", "Vision-RAG",
        "GraphRAG", "HippoRAG", "Late Interaction", "Sparse-Dense Hybrid",
        "BGE-M3", "E5-Mistral", "Reranker Fine-Tuning", "Cross-Encoder Distillation",
        "Long-Context RAG", "Iterative RAG", "Self-RAG", "CRAG",
        "Adaptive Retrieval", "Query Decomposition", "HyDE",
        "Knowledge Graph Augmentation", "Code-Augmented RAG", "Tool-Augmented RAG",
        "Multimodal Retrieval", "Citation-Conditioned Generation",
    ],
    "evaluation": [
        "LLM-as-a-Judge", "G-Eval", "PairRM", "Prometheus-2", "Chatbot Arena",
        "MT-Bench", "AlpacaEval", "Arena-Hard", "LiveCodeBench", "SWE-Bench",
        "GAIA Benchmark", "AgentBench Suite", "BFCL Function Calling",
        "Lost-in-the-Middle Eval", "Needle-in-a-Haystack", "RULER Long-Context",
        "Truthfulness Probes", "TruthfulQA-v2", "HaluEval", "FActScore",
        "Critique-Out-Loud", "Best-of-N Sampling", "Self-Consistency Voting",
        "Pairwise Preference Eval", "Process-Based Evaluation",
    ],
    "training": [
        "Curriculum Learning", "Data Mixing Laws", "Chinchilla Scaling",
        "Compute-Optimal Training", "muP Hyperparameter Transfer", "Adam-mini",
        "Lion Optimizer", "Sophia Optimizer", "Schedule-Free Adam",
        "Sequence Packing", "FlashPacking", "LoRA", "QLoRA", "DoRA", "PiSSA",
        "Galore", "ReFT", "BitFit", "Adapter Tuning",
        "Continual Pretraining", "Knowledge Distillation", "Distillation-from-RM",
        "On-Policy Distillation", "Sequence-Level Distillation", "Synthetic Pretraining",
    ],
    "multimodal": [
        "LLaVA-Next", "VILA", "Idefics3", "Pixtral", "Molmo",
        "Cambrian-1", "Eagle-X", "InternVL", "Qwen2-VL", "MiniGPT-4",
        "CLIP Beats Out-of-Domain", "SigLIP", "DINOv2", "MaskFeat",
        "Diffusion Transformers", "DiT", "Stable Diffusion 3", "FLUX",
        "Rectified Flow", "Flow Matching", "Consistency Models",
        "Video Diffusion", "Sora-Style Video", "I-JEPA", "V-JEPA",
    ],
    "safety": [
        "Jailbreak Robustness", "Prompt Injection Defense", "Indirect Prompt Injection",
        "Adversarial Suffix Attacks", "GCG Attack", "Many-Shot Jailbreaking",
        "Watermarking LLM Output", "Synthid", "Tree-Ring Watermarks",
        "Membership Inference Attacks", "Training Data Extraction", "Backdoor Attacks",
        "Data Poisoning", "Sleeper Agents", "Trojan Detection",
        "Constitutional Classifiers", "Llama Guard", "Granite Guardian",
        "Hallucination Detection", "Refusal Steering", "Activation Steering",
        "Representation Engineering", "Probe-Based Monitoring", "Interpretability Probes",
        "Dictionary Learning",
    ],
}


def build_concept_catalog():
    """Return ({concept_id: record}, [(domain, concept_id)] groups)."""
    catalog: Dict[int, Dict] = {}
    domain_groups: Dict[str, List[int]] = {}
    cid = 0
    for domain, names in SUB_DOMAINS.items():
        domain_groups[domain] = []
        for name in names:
            catalog[cid] = {
                "title": name,
                "one_liner": f"{name} — a {domain} technique published in the post-T research wave.",
                "abstract_summary": (
                    f"{name} is a recent {domain} method. The 5-paper reading path "
                    "starts from foundational antecedents and ends at the canonical "
                    "introduction. Empirical results show consistent gains on the "
                    "domain's standard benchmark suite versus prior baselines, with "
                    "the trade-off being increased compute or implementation complexity."
                ),
                "growth_signal": round(random.Random(SEED + cid).uniform(0.1, 0.95), 3),
                "is_trending": random.Random(SEED + cid * 7).random() < 0.50,  # 50% trending
            }
            domain_groups[domain].append(cid)
            cid += 1
    return catalog, domain_groups


# ---------------------------------------------------------------------------
# Synthetic users — each user "specializes" in 1-2 sub-domains
# ---------------------------------------------------------------------------

USER_PROFILES = [
    ("u_alice",   ["alignment", "training"]),
    ("u_bob",     ["efficiency", "training"]),
    ("u_carol",   ["agents", "retrieval"]),
    ("u_dave",    ["multimodal", "efficiency"]),
    ("u_eve",     ["safety", "alignment"]),
    ("u_frank",   ["evaluation", "agents"]),
    ("u_gina",    ["retrieval", "evaluation"]),
    ("u_hank",    ["training", "efficiency"]),
    ("u_iris",    ["multimodal", "retrieval"]),
    ("u_jack",    ["agents", "alignment"]),
    ("u_kate",    ["safety", "evaluation"]),
    ("u_leo",     ["alignment", "agents"]),
    ("u_mia",     ["efficiency", "multimodal"]),
    ("u_noah",    ["retrieval", "agents"]),
    ("u_olivia",  ["evaluation", "training"]),
    ("u_peter",   ["safety", "multimodal"]),
    ("u_quinn",   ["alignment", "evaluation"]),
    ("u_rachel",  ["training", "agents"]),
    ("u_sam",     ["multimodal", "alignment"]),
    ("u_tina",    ["efficiency", "safety"]),
]


def build_dataset():
    rng = random.Random(SEED)
    catalog, domain_groups = build_concept_catalog()
    all_ids = list(catalog.keys())

    user_summaries: Dict[str, str] = {}
    concept_pool: Dict[str, List[int]] = {}
    adoption: Dict[str, Dict[str, float]] = {}
    comprehension: Dict[str, Dict[str, float]] = {}
    knn_users: Dict[str, List[str]] = {}

    user_specs = {uid: doms for uid, doms in USER_PROFILES}

    for uid, doms in USER_PROFILES:
        user_summaries[uid] = (
            f"Researcher {uid[2:].title()} works primarily on {' and '.join(doms)}. "
            f"Their pre-T publications (2023-Sept 2025) cluster in those areas with "
            f"strong methodological emphasis. Citation graph shows recurring "
            f"co-authorship within these sub-domains."
        )

        # Build candidate pool — 50 concepts per spec:
        #   15 from user's domain (positives)
        #   15 from adjacent domains (hard negatives)
        #   10 trending across all (popularity bait)
        #   10 random noise
        own_domain_pool = []
        for d in doms:
            own_domain_pool.extend(domain_groups[d])
        own_domain_pool = list(set(own_domain_pool))
        rng.shuffle(own_domain_pool)

        positives = own_domain_pool[:15]
        hard_negs_pool = own_domain_pool[15:30] if len(own_domain_pool) >= 30 else own_domain_pool[15:]
        if len(hard_negs_pool) < 15:
            extras = [c for c in all_ids if c not in own_domain_pool]
            rng.shuffle(extras)
            hard_negs_pool = list(hard_negs_pool) + extras[: 15 - len(hard_negs_pool)]
        hard_negs = hard_negs_pool[:15]

        trending = [c for c in all_ids if catalog[c]["is_trending"]]
        rng.shuffle(trending)
        trending = trending[:10]

        used = set(positives) | set(hard_negs) | set(trending)
        noise = [c for c in all_ids if c not in used]
        rng.shuffle(noise)
        noise = noise[:10]

        pool = list(set(positives + hard_negs + trending + noise))
        rng.shuffle(pool)
        # If short due to overlaps, top up with random
        topup = [c for c in all_ids if c not in pool]
        rng.shuffle(topup)
        while len(pool) < 50:
            pool.append(topup.pop())
        pool = pool[:50]
        concept_pool[uid] = pool

        # Adoption labels: ~30% of positives genuinely adopted (research is hard);
        # ~5% of hard negatives get k-NN partial credit (0.3).
        adoption_u: Dict[str, float] = {}
        for cid in positives:
            if rng.random() < 0.30:
                adoption_u[str(cid)] = 1.0
        for cid in hard_negs:
            if rng.random() < 0.05:
                adoption_u[str(cid)] = 0.3
        adoption[uid] = adoption_u

        # Comprehension scores: only for adopted-or-positive concepts
        comp_u: Dict[str, float] = {}
        for cid_str in adoption_u:
            comp_u[cid_str] = round(rng.uniform(0.4, 0.95), 3)
        # plus a few "high quality reading path" concepts even if not adopted
        for cid in positives + hard_negs:
            if str(cid) not in comp_u and rng.random() < 0.15:
                comp_u[str(cid)] = round(rng.uniform(0.3, 0.7), 3)
        comprehension[uid] = comp_u

    # k-NN users by Jaccard overlap of domain specs
    for uid, doms in USER_PROFILES:
        scored = []
        for other_uid, other_doms in USER_PROFILES:
            if other_uid == uid:
                continue
            inter = len(set(doms) & set(other_doms))
            union = len(set(doms) | set(other_doms))
            scored.append((inter / union if union else 0.0, other_uid))
        scored.sort(reverse=True)
        knn_users[uid] = [u for _, u in scored[:5]]

    # Apply k-NN partial credit: if ≥2 of 5 nearest neighbors adopted a
    # concept, give the user 0.3 partial credit (if not already 1.0)
    for uid in user_summaries:
        for cid in concept_pool[uid]:
            cs = str(cid)
            if cs in adoption[uid]:
                continue
            neighbor_hits = sum(
                1 for n in knn_users[uid] if cs in adoption.get(n, {})
            )
            if neighbor_hits >= 3:
                adoption[uid][cs] = 0.3

    # Novelty flags = NOT trending
    novelty = {str(cid): (not catalog[cid]["is_trending"]) for cid in all_ids}

    # Reading paths — 5 mock papers per concept (deterministic)
    reading_paths: Dict[str, List[Dict[str, str]]] = {}
    for cid, rec in catalog.items():
        title = rec["title"]
        papers = [
            {"title": f"Foundations of {title}", "arxiv_id": f"2401.{cid:05d}", "year": "2024"},
            {"title": f"{title}: A Survey", "arxiv_id": f"2403.{cid:05d}", "year": "2024"},
            {"title": f"{title} in Practice", "arxiv_id": f"2406.{cid:05d}", "year": "2024"},
            {"title": f"Scaling {title}", "arxiv_id": f"2410.{cid:05d}", "year": "2024"},
            {"title": f"Canonical {title} Paper", "arxiv_id": f"2502.{cid:05d}", "year": "2025"},
        ]
        reading_paths[str(cid)] = papers

    # Concept catalog must be JSON-serializable with string keys
    catalog_json = {str(cid): rec for cid, rec in catalog.items()}

    # Persist all artifacts
    artifacts = {
        "user_summaries.json": user_summaries,
        "concept_pool_per_user.json": concept_pool,
        "concept_catalog.json": catalog_json,
        "ground_truth_adoption.json": adoption,
        "novelty_flags.json": novelty,
        "comprehension_scores.json": comprehension,
        "reading_paths.json": reading_paths,
        "knn_users.json": knn_users,
    }
    for name, payload in artifacts.items():
        with (DATA_DIR / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  wrote {name}  ({(DATA_DIR / name).stat().st_size / 1024:.1f} KB)")


def main():
    random.seed(SEED)
    print(f"Building synthetic Blindspot seed dataset under {DATA_DIR} ...")
    build_dataset()
    print("Done.")


if __name__ == "__main__":
    main()
