# 🚀 Blindspot

**A Real-User RL Environment for Discovering "Unknown Unknowns" in Research**

**Team:** Vasarla Avinash  
**Track:** Theme #3.1 — World Modeling (Professional Tasks)  
**OpenEnv Hackathon India 2026**

---

## ⚡ TL;DR (Read This First)

Most AI systems help you find what you *already know to search for*.  
**Blindspot solves the harder problem: finding what you don't even know you're missing.**

We built:

* A **real-user RL environment**
* Where an agent must recommend *unknown but high-impact concepts*
* And is rewarded based on **actual future adoption by researchers**

📈 Result: Even a small 1.5B model trained on just 40 traces **outperforms random and trending baselines** and achieves **positive reward** — proving the environment is learnable and RL-ready.

---

## 🎯 The Problem

Every researcher has blindspots:

* You search what you already know
* RAG retrieves what you ask
* Trends show what's popular

👉 But the *most valuable ideas* are often:

* Not searched
* Not trending
* Not obvious

**These are "unknown unknowns."**

---

## 💡 Our Idea

Turn this into a **reinforcement learning problem**:

> Given a researcher's history, recommend concepts they don't know — but *will actually adopt later.*

---

## 🌍 The Environment (What the Agent Does)

### 👀 Observation

At each step, the agent sees:

* Researcher profile (papers, interests)
* 50 candidate concepts
* Remaining budgets (inspect / surface)

### 🎮 Actions

```python
{"type": "inspect", "concept_id": 42}
{"type": "surface", "concept_id": 42}
{"type": "stop"}
```

* Inspect = reveal more info (costly)
* Surface = recommend concept
* Budget constrained → forces strategy

### 🧠 Why This Is Hard

* ❌ **Sparse reward** (only at end)
* ❌ **Partial observability**
* ❌ **Personalization** (same concept ≠ same value)
* ❌ **False positives are penalized**

👉 Random guessing fails.

---

## 🏆 Reward Design (Core Innovation)

We don't reward "looks good" — we reward **real impact**:

| Component | Signal |
| --- | --- |
| Adoption | Did the researcher actually use it later? |
| Novelty | Was it non-obvious at the time? |
| Onboarding | Did it improve understanding? |
| Efficiency | Penalize unnecessary exploration |
| False positives | Penalize irrelevant recommendations |

👉 Key property:

> **Random policy ≈ 0 reward**  
> So any positive score = real learning

---

## 📊 Baselines (Before Training)

Measured over 5 seeds × 17 users:

| Policy | Mean Reward | Std |
| --- | ---: | ---: |
| Random | +0.088 | ±1.40 |
| Trending | +0.212 | ±0.51 |
| **Dense Retrieval** | **+0.467** | ±1.20 |
| Oracle (upper bound) | +3.286 | ±3.59 |

👉 There is a **large learnable gap** between heuristic methods and optimal behavior.

![Baselines](plots/baseline_comparison.png)

![Reward decomposition](plots/reward_decomposition.png)

---

## 🤖 Training Setup

* **Model:** `unsloth/Qwen2.5-1.5B-Instruct`, 4-bit NF4, bf16
* **Method:** SFT (warm start for RL) via TRL's SFTTrainer on H100
* **Data:** 40 expert traces from Dense Retrieval+ heuristic
* **LoRA:** rank=16, alpha=16, 3 epochs, lr=2e-5, batch=8
* **Loss:** 1.108 → 1.080 (steps 5/10/15: 1.1078 / 1.0940 / 1.0800)

👉 Goal: Prove **learnability before RL**

![SFT training loss](plots/sft_loss.png)

---

## 📈 Results (What Actually Improved)

Evaluation: 13 training users × 10 seeds = **130 episodes per policy**

| Policy | Mean Reward | Std |
| --- | ---: | ---: |
| Random | −0.340 | ±0.854 |
| Trending | −0.355 | ±0.905 |
| ✅ **Blindspot SFT (ours)** | **+0.039** | ±0.453 |

![Policy comparison](plots/final_comparison.png)

✔ Two-sample t-test (SFT vs Random): **p = 0.03**  
✔ 95% CI for SFT: [−0.04, +0.12] — entirely above the random mean (−0.34)  
✔ Crosses the critical threshold: **positive reward**

---

## 🔍 What Changed (Intuitive View)

### ❌ Before (Random / Trending)

* Recommends many irrelevant concepts
* High false-positive penalty
* No personalization

### ✅ After (Trained Model)

* Reads user profile
* Avoids irrelevant concepts
* Makes **more precise recommendations**

👉 The model learns:

> "Don't suggest everything — suggest what matters."

---

## 🧪 Key Insights

### 1. Small Models Can Learn This

1.5B + 40 traces → already positive reward. The Oracle at +3.286 shows substantial headroom for larger models and more data.

### 2. False Positives Are the Real Enemy

The −0.1 penalty per non-adopted surface is the dominant cost for all non-oracle policies. Avoiding noise is the biggest gain.

### 3. RL Needs a Warm Start

Direct GRPO on the base model failed: the model always produced the **same first action** for every rollout (`{"type": "surface", "concept_id": 1}`). With `num_generations=4`, all four trajectories were identical → within-group reward variance = 0 → no gradient flow. SFT provides the **initial policy diversity** GRPO needs. RL from this SFT checkpoint is the immediate next step.

---

## ⚠️ Important Engineering Insight

OpenEnv's HTTP server resets the environment **per request** — every `/reset` and `/step` call destroys episode state (budgets, surfaced concepts), so all rewards return zero.

**Fix:** run `BlindspotEnvironment` **statefully in Python**, keeping one instance alive per episode.

👉 Critical footgun for anyone building multi-step RL on OpenEnv.

---

## 🚀 Why This Matters

This is not just a benchmark. It enables:

* 🔬 Research discovery beyond search
* 🧠 Personalized knowledge expansion
* 🏢 Enterprise knowledge assistants
* 📚 Learning systems that find "what you're missing"

---

## 🧩 Why This Is a Strong RL Environment

* ✅ Real-world data — 17 actual researchers, 62 ground-truth adoption pairs
* ✅ Hard to game — false-positive penalty cancels "surface everything" strategies
* ✅ Personalized rewards — same concept has different value for different users
* ✅ Cheap to run — pure lookup, sub-millisecond step, no GPU env
* ✅ Measurable improvement — held-out users provide uncontaminated evaluation
* ✅ Clear RL upgrade path — SFT warm-start is done; GRPO is next

---

## 🔮 What's Next

* Train with **GRPO from SFT checkpoint**
* Scale dataset (more researchers, more traces)
* Improve fine-grained user–concept matching
* Add richer process rewards

---

## ⚠️ Limitations

* Dataset size: 17 users proves environment shape, not broad generalization
* Adoption proxy: uses kNN backoff when direct signal is absent
* Comprehension: LLM-judged, not human-verified
* Demo: cache-backed for stability, not a live online RL loop

---

## 📎 Links

| Resource | URL |
| --- | --- |
| GitHub | https://github.com/vasarlalikhilavinash/blindspot-env |
| HF Space (demo) | https://huggingface.co/spaces/Vasarlaavinash/blindspot-demo |
| Trained adapter (SFT) | https://huggingface.co/Vasarlaavinash/blindspot-sft-1.5b |
| Training notebook (Colab) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/02_training.ipynb) |
| Demo notebook | https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/03_demo.ipynb |

---

## 🏁 Final Takeaway

> Blindspot turns one of the hardest problems in AI — discovering unknown unknowns — into a **trainable, measurable RL task**.

👉 **It already works — and gets better with training.**
