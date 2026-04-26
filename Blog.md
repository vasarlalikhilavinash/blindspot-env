# Blindspot: Teaching LLMs to Surface What Researchers Don't Know They Need

**Team**: vasarlalikhilavinash  
**Track**: Theme #3.1 — Professional Tasks / World Modeling  
**OpenEnv Hackathon India 2026**

---

## The Problem: Unknown-Unknowns in Research Discovery

Every researcher has a blindspot.

You know what you know. You know what you're searching for. But the papers and concepts that could most change your work — the ones you've never heard of and therefore never searched for — stay invisible.

Search engines and RAG systems are pull-based: they help once you already know the query. Trending feeds surface what is popular, not what is relevant to *you*. Neither approach solves unknown-unknowns.

Blindspot is an RL environment that forces an agent to solve this exact problem: given a researcher's existing publication record and reading history, surface the concepts they should know about but don't yet — and optimize for whether they actually adopted those concepts afterward.

---

## The Environment

Blindspot implements the full OpenEnv interface against a dataset of **17 real ML researchers**, **1,168 candidate concepts**, **282 reading paths**, and **62 ground-truth adoption pairs** measured from post-timestamp research artifacts.

### What the Agent Sees

At every step the agent receives:

- A **user summary**: their research interests, recent papers, and expertise areas
- A **candidate pool** of 50 concepts drawn from their personal concept graph
- Remaining **inspect** and **surface** budgets
- Which concepts have already been inspected or surfaced this episode

### What the Agent Can Do

```python
{"type": "inspect", "concept_id": 42}   # reveal reading path, costs 1 inspect budget
{"type": "surface", "concept_id": 42}   # commit as a recommendation, costs 1 surface budget
{"type": "stop"}                         # end episode early
```

Surface budget = 10. Inspect budget = 15. The agent must decide which of 50 candidates to surface within budget — without knowing ground-truth adoption in advance.

### Why This Is Hard

- **Sparse reward**: the agent only finds out how good its choices were at episode end.
- **Partial observability**: inspecting reveals a reading path but not whether the user will adopt.
- **False-positive cost**: surfacing irrelevant concepts is penalized, so carpet-bombing fails.
- **Personalization**: the same concept may be high-value for one researcher and irrelevant for another.

---

## The Reward Signal

Episode reward is computed once at `stop` or budget exhaustion, via four components:

| Component | Signal | Note |
|---|---|---|
| **Adoption** | +adoption_score per concept | Ground-truth from post-T artifacts |
| **Novelty** | +0.5 per novel adopted concept | Not in trending feeds at time T |
| **Onboarding** | +comprehension_lift per adopted concept | LLM judge, κ ≥ 0.7 |
| **Efficiency** | −0.01 per inspect call | Small but accumulates |
| **False positive** | −0.1 per surfaced concept with zero adoption | Discourages noise |

The false-positive penalty is calibrated so that a **uniform random policy yields E[reward] ≈ 0**, confirmed empirically. This means any positive signal is real signal.

---

## Baseline Calibration (Real Data)

Before training, we measured four policies over 5 seeds × 17 users:

| Policy | Mean reward | Std |
|---|---:|---:|
| Random | +0.088 | ±1.40 |
| Trending | +0.212 | ±0.51 |
| **Dense Retrieval** | **+0.467** | ±1.20 |
| Oracle (upper bound) | +3.286 | ±3.59 |

Key observations:
- `Random ≈ 0` confirms the reward is not inflated.
- `Oracle − Dense Retrieval ≈ 2.8` — there is substantial headroom for a learned policy.
- Dense Retrieval is a strong baseline because the concept pool is already semantically related to each user; the agent needs to do better than semantic similarity alone.

![Baselines](plots/baseline_comparison.png)

The reward decomposition shows WHY dense retrieval scores well — it earns meaningful adoption and novelty rewards — while the false-positive penalty remains the dominant cost for all non-oracle policies:

![Reward decomposition](plots/reward_decomposition.png)

---

## Training Setup

We trained a **16-rank LoRA adapter** on top of `unsloth/Qwen3.5-9B` (bf16) using **TRL's GRPOTrainer** on a single NVIDIA H100 80GB.

**Training config:**
- 120 gradient steps × 4 rollouts/step = 480 reward queries
- Each reward query rolls out a full multi-step episode against the live OpenEnv server
- `max_completion_length = 96`, `learning_rate = 5e-6`, `gradient_accumulation = 4`
- 256 prompts from 13 training users; 4 users held out for evaluation

**System prompt engineering:**  
A key challenge we identified was policy collapse — the model defaulted to repeatedly surfacing `concept_id=0` or `concept_id=1`, yielding zero reward. We addressed this by:

1. Explicit instruction in the system prompt to choose concept_ids from the visible CANDIDATES list
2. In-episode duplicate surface penalty (−0.15) fed back into the episode reward
3. In-episode error messages redirecting the model when it attempted a re-surface
4. Sampled inner-rollout generation (temperature 0.8) to improve trajectory diversity

**Training infrastructure:**  
The reward function calls the live OpenEnv HTTP server (`/reset`, `/step`) for each rollout. This means training is grounded in the actual environment dynamics, not a static proxy.

---

## Results

Training ran end-to-end without errors. The training loss curve and reward decomposition over rollouts are saved at:

- `plots/training_reward_curve.png`
- `plots/training_component_curves.png`

Post-training evaluation runs 5 seeds × 17 users using the trained adapter:

- Results saved to `plots/comparison_with_trained.png`
- Per-user breakdown saved to `plots/per_user_reward.png`
- Component breakdown saved to `plots/decomposition_with_trained.png`

---

## What We Learned

**The hardest part was not the model — it was the reward signal density.**

GRPO requires within-group reward variance to compute a meaningful advantage. When the base model is strongly peaked on a single short output (`{"type": "surface", "concept_id": 1}`), all rollouts within a group produce identical trajectories and identical rewards — yielding zero gradient despite non-zero batch-level variance.

The fix requires ensuring diversity in the GRPO-generated completions themselves, not just in the inner episode steps. Approaches that work in practice include:
- Higher temperature on the GRPO-internal generation
- SFT warm-start on demonstration traces to diversify the initial policy
- Curriculum: start with shorter episodes and sparser prompts

This is a genuine research problem in multi-step RL with LLMs: the base model's strong priors can prevent the exploration needed for GRPO to obtain contrastive signal. Blindspot makes this problem crisp and reproducible, which is itself a contribution.

---

## Why Blindspot Is A Good RL Environment

- **Cheap**: pure lookup, sub-millisecond step, no GPU at inference time
- **Real**: 17 actual researchers, real adoption ground truth, not synthetic
- **Hard to game**: false-positive penalty cancels naive "surface everything" strategies
- **Personalized**: the same concept has different value for different users
- **Measurable**: held-out users provide uncontaminated evaluation
- **Extensible**: the concept catalog (1,168 entries) can grow; new users can be onboarded with their arXiv records

The gap between Dense Retrieval (+0.467) and Oracle (+3.286) represents a real, unsolved problem in research personalization. Blindspot turns that gap into a trainable RL objective.

---

## Limitations

- **Dataset size**: 17 users is enough to prove the environment shape but not to claim broad generalization.
- **Adoption proxy**: uses kNN backoff when direct adoption signal is absent.
- **Comprehension**: judge-assessed, not human-verified.
- **Demo**: cache-backed for stability, not a live online RL loop.

---

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/vasarlalikhilavinash/blindspot-env |
| HF Space (demo) | https://huggingface.co/spaces/vasarlalikhilavinash/blindspot-demo |
| Trained adapter | https://huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo |
| Training notebook (Colab) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/02_training.ipynb) |
| Demo notebook | https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/03_demo.ipynb |
