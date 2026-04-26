# Blindspot: An RL Environment for Unknown-Unknown Discovery — and a Diagnostic Testbed for Exploration Collapse

**Team:** Vasarla Avinash  
**Track:** Theme #3.1 — World Modeling (Professional Tasks)  
**OpenEnv Hackathon India 2026**

---

## The Idea

There's a category of knowledge gap that no search engine can fix. You can't search for a concept you've never heard of. You can't ask a RAG system to find papers outside your vocabulary. Trending feeds show what's popular, not what's relevant to your specific research direction.

These are unknown unknowns — things you'd care deeply about if you knew they existed, but which stay invisible precisely because you don't know to look.

Blindspot turns this into a reinforcement learning problem. Given a researcher's existing publication record and reading history, an agent has to surface the concepts they're missing — and the reward signal is whether the researcher actually adopted those concepts in their subsequent work. Along the way, we discovered that the environment has a second use: it reliably triggers and measures the exploration collapse that current RL methods are known to suffer from, making it a useful diagnostic alongside being a training environment.

---

## The Environment

The dataset behind Blindspot is fully real. Seventeen ML researchers, 1,168 candidate concepts drawn from the academic literature, 282 reading paths, and 62 ground-truth adoption events measured from post-timestamp research artifacts. Nothing is synthetic.

At each step, the agent receives a researcher's profile, a pool of 50 candidate concepts from their personal concept graph, and budget counters for `inspect` (up to 15) and `surface` (up to 10) actions. The agent can open a concept to see its reading path, commit to recommending it, or stop early. It has to decide which 10 of the 50 to recommend — without knowing in advance which ones the researcher will actually adopt.

```python
{"type": "inspect", "concept_id": 42}   # look closer, costs budget
{"type": "surface", "concept_id": 42}   # commit as recommendation
{"type": "stop"}                          # end the episode
```

The hard part is that reward is delayed until episode end, the inspect action reveals reading paths but not adoption likelihood, and the same concept can be high-value for one researcher and irrelevant for another. There's no shortcut.

---

## Reward Design

The reward is deliberately multi-component to capture what "good" actually means in this domain:

| Component | Signal |
|---|---|
| Adoption | +score per concept the researcher actually used later |
| Novelty | +0.5 per adopted concept that wasn't trending at time T |
| Onboarding | +comprehension lift per adopted concept (LLM judge, κ ≥ 0.7) |
| Efficiency | −0.01 per inspect call |
| False positive | −0.1 per surfaced concept with zero adoption |

The false-positive penalty does a lot of work here. It's calibrated so that a uniform random policy earns approximately zero reward — confirmed empirically across multiple seeds. That means any positive reward is real signal, not noise.

---

## Calibration Before Training

Before touching any model, we measured four policies on the real dataset (5 seeds × 17 users):

| Policy | Mean reward | Std |
|---|---:|---:|
| Random | +0.088 | ±1.40 |
| Trending | +0.212 | ±0.51 |
| Dense Retrieval | +0.467 | ±1.20 |
| Oracle (upper bound) | +3.286 | ±3.59 |

Random is near zero, which confirms the reward isn't inflated. Dense Retrieval does well because the candidate pool is already semantically filtered — it gets genuine adoption and novelty reward. But the gap between Dense Retrieval (+0.467) and Oracle (+3.286) is about 2.8 reward points. That's the gap a learned policy should close.

![Baseline comparison](plots/baseline_comparison.png)

![Reward decomposition](plots/reward_decomposition.png)

---

## Training

We trained a 16-rank LoRA adapter on top of `unsloth/Qwen2.5-1.5B-Instruct` (4-bit NF4, bf16) using TRL's SFTTrainer on a single H100. The goal was not to build the final policy — it was to prove the environment is learnable.

**Expert traces:** We generated 40 demonstration traces using Dense Retrieval+, our best heuristic (TF-IDF cosine similarity, surface top-10, no inspect calls). Each trace is a full chat-format conversation: system prompt → observation → action sequence. 40 traces is intentionally small. If a 1.5B model trained on 40 examples can cross the zero-reward threshold, that's a meaningful signal about the environment's structure.

**Config:** rank=16, alpha=16, 3 epochs, batch size 8, lr=2e-5, bf16. Loss went from 1.108 → 1.080 across 15 logged steps. The curve is flat, which makes sense — the model learned the action format and surfacing strategy within the first epoch. With only 40 traces, there's not much room for further loss reduction. No signs of overfitting.

![SFT training loss](plots/sft_loss.png)

**Infrastructure note:** One thing that cost us time — the OpenEnv HTTP server creates a fresh `BlindspotEnvironment` instance per request. Every `/reset` and `/step` call destroys episode state, so rewards always come back zero. The fix is to call `BlindspotEnvironment` directly in Python and keep one instance alive per episode. This is worth documenting for anyone else building multi-step evaluations on OpenEnv.

---

## Results

Evaluation ran over 13 training users × 10 seeds = 130 episodes per policy. Note: these episodes use seeds 100–109, a different shuffle than the calibration runs above (seeds 0–19). The false-positive penalty is more punishing with these shuffles — adopted concepts land outside the top positions in the candidate pool more often — so baselines are negative here. SFT still crosses zero, which is the meaningful result.

| Policy | Mean reward | Std |
|---|---:|---:|
| Random | −0.340 | ±0.854 |
| Trending | −0.355 | ±0.905 |
| **SFT — Qwen2.5-1.5B (ours)** | **+0.039** | ±0.453 |

![Policy comparison](plots/final_comparison.png)

SFT is the only policy with positive mean reward. The improvement over random is +0.380 and over trending is +0.394. A two-sample t-test (unequal variance) gives p = 0.03. The 95% confidence interval for SFT is [−0.04, +0.12], which lies entirely above the random mean of −0.34. The result is statistically significant.

To be precise about what this proves: this is a proof of learnability, not a production policy. An effect of +0.039 per episode is small in absolute terms — well below the Oracle ceiling of +3.286. The model learned the action format and the general strategy of selecting based on profile relevance, but has not learned fine-grained user–concept matching. The value of the number is not that it's large; it's that it's positive when the baselines are both negative, and the distance from those baselines (+0.38 above random) is what matters for a first-training run on 40 traces.

**Held-out users.** On the 4 researchers withheld from training (20 episodes), SFT achieved 0.00 ± 0.00. The improvement does not transfer to unseen users at this scale. This is expected — 17 researchers total is too small a dataset for generalization. Scaling the dataset is the primary future work, and the held-out result is the clearest argument for it.

The baselines being negative here (vs. positive in calibration) is expected — evaluation uses seeds 100–109, which produce different candidate shuffles than the calibration seeds. The false-positive penalty is unforgiving when adopted concepts land outside the top positions in a shuffled pool. SFT avoids the worst of this by reading the researcher profile and selecting based on relevance rather than list position.

**Note on Dense Retrieval.** The calibration table above shows Dense Retrieval at +0.467, which appears much higher than SFT's +0.039. Those numbers used different seeds (0–19 vs 100–109) and are not directly comparable. Dense Retrieval was not re-evaluated on seeds 100–109, so we cannot say whether it would score +1.2 or −0.5 on the same shuffles. Readers should treat the calibration and evaluation tables as separate measurements, not a head-to-head comparison.

---

## Why GRPO Didn't Work — And What the Logs Prove

Before SFT we ran GRPO directly on the base model for 150 steps (600 total rollouts). Not only did it not improve — the training dynamics show exactly why, step by step.

**The reward trace.** Across all 150 steps, the mean reward never left the band [−0.01, 0.00]. The first 10% of rollouts averaged −0.004. The last 10% averaged 0.000. Net gain over 90 minutes of H100 training: +0.004. That is not noise rounding — it is a flat line.

**The variance collapse.** GRPO computes advantages by ranking rollouts within a group. That ranking requires reward variance. At steps 5, 45, 50, 55, 65, 75, 85, 90, 105, 110, 125, 130, 140, 145, 150 the logged `reward_std` is exactly 0.000. When every rollout in the group earns the same reward, the advantage for every rollout is zero, the gradient is zero, and the weights do not move. The model cannot learn from a signal it cannot differentiate.

**The action collapse.** The qualitative output makes the mechanism visible. After training, the model produces this across every episode, every user, every seed:

```
{"type": "surface", "concept_id": 1}
{"type": "surface", "concept_id": 1}
{"type": "surface", "concept_id": 1}
... (8× then stop)
```

That is not a policy. It is a degenerate fixed point. The base model converged to the lowest-risk action — surface the first concept listed — before GRPO could introduce any useful pressure.

**Why the environment makes this worse.** Blindspot has three properties that compound the problem simultaneously: reward is delayed until episode end (no per-step signal to differentiate early actions), the false-positive penalty makes surface calls risky (the model learns to avoid variety), and personalization means the right answer varies by user (so copying one pattern across all rollouts is almost always wrong, but consistently wrong in a way that produces near-zero variance). The result is a policy that earns approximately −0.05 per invalid action or 0.0 for a stop, with very little spread across the group.

**This is a known failure mode, with published fixes.** The DAPO paper (Yu et al., arXiv:2503.14476, 2025) identifies entropy collapse as the central instability in GRPO-style training and shows that standard GRPO's clipping causes the policy to converge to low-entropy outputs before it can explore. Their proposed decoupled clipping directly targets this. Hybrid GRPO (Sane, arXiv:2502.01652, 2025) makes the same diagnosis from a variance perspective: purely empirical reward estimation in GRPO amplifies variance problems when rollouts are homogeneous, and they add bootstrapped value estimation to stabilize it. Both papers are essentially describing what we observed in Blindspot, in a real multi-step environment rather than a controlled math-reasoning testbed. The NeurIPS 2025 result from Wang et al. (arXiv:2506.01939) adds a token-level explanation: RLVR only meaningfully updates high-entropy tokens, which are the decision points. A base model that has collapsed to a fixed action sequence has no high-entropy tokens left to update, so RLVR stalls entirely.

**SFT warm-start is the fix.** After SFT the model produces varied action sequences — different concept IDs, different numbers of inspect calls, different stopping points — because it has learned the task structure. That diversity is exactly what GRPO needs to rank rollouts. RL from the SFT checkpoint is the immediate next step, and the training dynamics above make clear why the order matters.

The full GRPO training table and the repeated-action samples are visible in the [training notebook](https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/02_training.ipynb). The logs are not edited. Judges who want to verify the collapse can scroll to section 3 and read the per-step reward and reward_std columns directly.

---

## Limitations

The dataset is small. 17 researchers is enough to establish that the environment works and that the reward signal is learnable, but it's not enough to claim the policy generalizes broadly. Adoption uses a kNN backoff when direct signal is absent for a concept-user pair. Comprehension lift is measured with an LLM judge, not human evaluation. The demo is cache-backed, not a live RL loop.

---

## What Makes This a Good RL Environment — and a Useful Diagnostic

A few properties that make Blindspot worth training on:

- The reward is grounded in real behavior, not human annotation or proxy metrics. Adoption events come from actual post-timestamp research artifacts.
- The false-positive penalty prevents degenerate strategies. You can't just surface everything.
- The personalization requirement means the agent has to actually understand the researcher's profile, not just rank by popularity.
- Step time is sub-millisecond. No GPU required at episode time. Training is fast.
- The held-out test users (4 of 17) weren't touched during training, providing uncontaminated evaluation.
- The Oracle gap (~2.8 reward points above Dense Retrieval) gives a clear target for a learned policy.

The GRPO failure adds a further dimension. The environment reliably induces the reward-variance collapse that RL researchers have been studying in 2025–26, because it combines sparse reward, partial observability, and a high false-positive penalty in a way that traps an untrained model at a degenerate fixed point. That makes Blindspot useful not just as a training environment but as a testbed for evaluating whether a new RL algorithm can escape early-stage collapse — a property that synthetic environments rarely have.

---

## Try It

The live demo is at **https://huggingface.co/spaces/Vasarlaavinash/blindspot-demo**. Pick a real researcher from the dropdown, hit Run, and see side-by-side what the base model vs. the SFT-trained model recommends — with adoption verdicts for each concept.

---

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/vasarlalikhilavinash/blindspot-env |
| HF Space (demo) | https://huggingface.co/spaces/Vasarlaavinash/blindspot-demo |
| Trained adapter (SFT) | https://huggingface.co/Vasarlaavinash/blindspot-sft-1.5b |
| Training notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/02_training.ipynb) |
| Demo notebook | https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/03_demo.ipynb |
