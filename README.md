# Blindspot

**An OpenEnv environment for unknown-unknowns discovery & contextual onboarding.**

> *"You can't search for what you don't know exists. Blindspot trains agents to surface the concepts you should be tracking — and then make you competent in them."*

OpenEnv Hackathon · April 2026 · Theme #3.2 (Personalized Tasks) + #2 (Long-Horizon Planning)

---

## The Problem

Every knowledge worker in a fast-moving field faces two compounding gaps that current AI tools structurally cannot solve:

1. **The vocabulary gap** — You don't know what concepts exist, so you can't query for them. ChatGPT, Perplexity, and Deep Research are *pull-based*: they require you to already know enough to ask. Nothing in the current stack handles **unknown unknowns**.
2. **The onboarding gap** — Even when a new concept reaches you, going from *"I've heard of it"* → *"I can use it"* takes hours of disorganized reading.

Result: people stay perpetually 6 months behind concepts already affecting their work.

---

## Why This Is a Real RL Problem

| Property | Why RL fits |
|---|---|
| Multi-step planning | Agent inspects candidates, weighs novelty vs. relevance, decides what to surface under a budget |
| Verifiable ground truth | Real users' actual concept adoption (terms in their post-T work) is the held-out label |
| Multiple independent rewards | Adoption + novelty + onboarding-quality + efficiency — hard to game |
| Sparse + delayed reward | Most candidates are noise; reward concentrates on a few correct surfaces |
| Hard baseline gap | Trending feeds and dense retrieval fail by construction (they need a query) |

---

## Pre-Compute Pipeline

All expensive computation happens **offline once**. The env at runtime is a fast lookup table — every `step` is a sub-millisecond dict lookup, so GRPO can do thousands of rollouts per training step at zero marginal cost.

```
                         Blindspot Pre-Compute Pipeline
                         ──────────────────────────────

  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │ Semantic Scholar │   │      arXiv       │   │  HF / GitHub     │
  │   ~50 ML users   │   │  cs.* 2024-2025  │   │  trending @ T    │
  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
           │                      │                      │
           ▼                      ▼                      ▼
  ┌────────────────────────────────────────────────────────────────┐
  │     [1] Split each user @ T = 2025-09-01                       │
  │         pre-T  →  user_profile, known_concepts (TF-IDF)        │
  │         post-T →  adopted_concepts (held-out ground truth)     │
  └────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────┐
  │     [2] Extract ~5k candidate concepts (KeyBERT on corpus)     │
  └────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────┐
  │     [3] Build per-user candidate pool (50 concepts each):      │
  │         15 relevant + adopted     ← positives                  │
  │         15 relevant + not adopted ← hard negatives             │
  │         10 trending distractors   ← popularity bait            │
  │         10 random noise           ← easy negatives             │
  └────────────────────────────┬───────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
  ┌──────────────────┐ ┌────────────────┐ ┌──────────────────┐
  │ [4a] Adoption    │ │ [4b] Novelty   │ │ [4c] Reading     │
  │ score per pair   │ │ flag (¬trending│ │ path (5 papers,  │
  │ (1.0 self,       │ │ at T)          │ │ citation BFS,    │
  │  0.3 k-NN user)  │ │                │ │ foundational→new)│
  └────────┬─────────┘ └────────┬───────┘ └────────┬─────────┘
           │                    │                  │
           │                    │                  ▼
           │                    │       ┌────────────────────────┐
           │                    │       │ [5] Comprehension score│
           │                    │       │ GPT-4 + Claude judges  │
           │                    │       │ answer QA w/ path vs   │
           │                    │       │ abstracts; lift = score│
           │                    │       │ Both must agree (κ≥.7) │
           │                    │       └────────────┬───────────┘
           │                    │                    │
           ▼                    ▼                    ▼
  ┌────────────────────────────────────────────────────────────────┐
  │     data/  (parquet + json, ~50 MB, ships in HF Space)         │
  │     ─────                                                      │
  │     ground_truth_adoption.parquet                              │
  │     novelty_flags.json                                         │
  │     reading_paths.json                                         │
  │     comprehension_scores.parquet                               │
  │     concept_pool_per_user.json                                 │
  │     user_summaries.json                                        │
  └────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────┐
  │              OpenEnv runtime (FastAPI, <1ms/step)              │
  │              reset · step · state  →  lookups only             │
  └────────────────────────────────────────────────────────────────┘
```

**False-positive calibration:** the false-positive penalty (`−0.1` per surfaced concept that is neither user-adopted nor adopted by ≥2 k-NN-similar users) is calibrated so that a uniformly random surface policy yields expected reward ≈ 0, ensuring the trained agent's reward gain reflects real discovery skill rather than baseline drift.

---

## Environment Spec

### Action Space (3 verbs)

```python
@dataclass
class BlindspotAction:
    type: Literal["inspect", "surface", "stop"]
    concept_id: int | None = None
```

### Observation

- `user_summary` (~200 tokens)
- `candidate_concepts`: 50 `ConceptCard`s (id, title, one-liner), shuffled per reset
- `inspected`: dict of `ConceptDetail` for concepts the agent has inspected
- `surfaced`: list of surfaced concept ids
- `inspect_budget_remaining` (max 15), `surface_budget_remaining` (max 10)

### Reward (4 independent components, all pre-computed lookups)

Let `adopted(c) = 1[adoption_score[c] ≥ 1e-6]` (i.e. the recommendation actually landed). Novelty and onboarding bonuses are **gated on adoption** so that "novel" or "well-onboarded" garbage cannot be reward-hacked.

| Component | Formula | Range | Purpose |
|---|---|---|---|
| Adoption | `+adoption_score[c]` per surfaced (1.0 self / 0.3 kNN partial) | [0, 1] | Did user actually adopt? |
| Novelty | `+0.5 × novelty_flags[c] × adopted(c)` per surfaced | [0, 0.5] | True unknown-unknown? |
| Onboarding | `+comprehension_scores[c] × adopted(c)` per surfaced | [0, 1] | Quality of reading path (only if adopted) |
| Efficiency | `−0.01 × inspect_count` | [−0.15, 0] | Penalize blind inspection |
| False-positive | `−0.1` per surfaced with no adoption signal | — | Discourage noise (calibrates random ≈ 0) |

Calibration on the synthetic seed (5 seeds × 20 users):

| Policy | Mean total | Std |
|---|---|---|
| Random | +1.05 | 1.79 |
| Trending-only | +0.86 | 1.63 |
| Dense retrieval | +0.62 | 1.39 |
| **Oracle (upper bound)** | **+8.77** | 2.98 |

→ ~7.7 reward of headroom for RL to capture; baselines are tightly clustered, the oracle is far away.

### Anti-hacking guards

- Hard caps on inspect (15) and surface (10) per episode
- Random shuffle of candidate order each reset
- Held-out test users not in training set
- Two judges (Claude + GPT) must agree for non-zero comprehension score

---

## Why Blindspot Is Trainable in 24h

| Concern | Status |
|---|---|
| Step latency | <1ms (pure lookups) |
| Step cost | $0 (no online API/LLM calls) |
| Cold-start reward | ~0.3 from random policy → RL bootstraps |
| Multiple independent rewards | 4 components → hard to game |
| Real ground truth | Actual user adoption, not synthetic |

---

## Repo Layout

```
blindspot-env/
├── README.md
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── models.py
├── server/
│   ├── app.py
│   ├── blindspot_environment.py
│   ├── data_loader.py
│   └── rewards.py
├── client/
│   └── blindspot_client.py
├── data/                          # produced by precompute pipeline
├── scripts/
│   ├── precompute_01_fetch_users.py
│   ├── precompute_02_fetch_corpus.py
│   ├── precompute_03_extract_concepts.py
│   ├── precompute_04_build_pools.py
│   ├── precompute_05_score_adoption.py
│   ├── precompute_06_build_paths.py
│   ├── precompute_07_score_comprehension.py
│   └── run_all_precompute.sh
├── baselines/
│   ├── random_baseline.py
│   ├── trending_baseline.py
│   └── dense_retrieval_baseline.py
├── training/
│   ├── generate_sft_traces.py
│   ├── sft_train.py
│   ├── grpo_train.py
│   └── eval.py
├── notebooks/
│   ├── 01_demo.ipynb
│   └── 02_training.ipynb
├── inference.py
└── plots/
```

---

## Judging Criteria Mapping

| Criterion | Weight | How Blindspot Scores |
|---|---|---|
| Environment Innovation | 40% | First RL env for unknown-unknowns discovery; novel multi-component reward; problem GPT-5.5/Deep Research structurally cannot solve |
| Storytelling | 30% | Universal pain; concrete before/after demo with real held-out user; reading-path drill-down |
| Reward Improvement | 20% | Random → trending → SFT → GRPO curves on Adoption-Recall and Comprehension-Lift |
| Pipeline | 10% | Standard OpenEnv + TRL + Unsloth + HF Space; Colab-runnable training notebook |

---

## Future Work

1. Causal counterfactual reward modeling (replace k-NN proxy)
2. Actionability as Stage 3 (track real downstream usage)
3. Human-calibrated reward critic (500+ expert labels, Cohen's κ)
4. Multi-session curriculum learning with persistent user state
5. Real-user closed-loop deployment with online RL
6. Cross-domain generalization (bio, econ, engineering)

---

## What Blindspot Explicitly Is NOT

- ❌ Not a news feed or "be first to know" tool
- ❌ Not a recommender over a known item set
- ❌ Not a Deep Research clone (those need a query; Blindspot generates it)
- ❌ Not a personalization layer over Perplexity
