# Blindspot

Blindspot is an OpenEnv environment for unknown-unknowns discovery: given a researcher's existing work, an agent must surface the AI/ML concepts they should be tracking but currently are not, then justify those recommendations with concrete reading paths.

## 🏆 Submission Notes

- Live demo: https://huggingface.co/spaces/vasarlalikhilavinash/blindspot-demo
- Trained adapter: https://huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo
- GitHub repo: https://github.com/vasarlalikhilavinash/blindspot-env
- Training notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/02_training.ipynb)
- Demo notebook: https://colab.research.google.com/github/vasarlalikhilavinash/blindspot-env/blob/main/notebooks/03_demo.ipynb
- Submission walkthrough: [docs/submission_walkthrough.md](docs/submission_walkthrough.md)
- Loom walkthrough: pending upload

## Why This Matters

Search, chat, and RAG systems are pull-based: they help once the user already knows what to ask. Blindspot targets the harder problem. It asks an agent to discover concepts that are both relevant and missing from the user's current vocabulary, then optimize for whether those concepts were later adopted and whether the recommended reading path actually improves comprehension.

That makes Blindspot a good RL environment:

- The agent must act under budgets (`inspect`, `surface`, `stop`), not just rank passively.
- Reward is delayed and partially sparse.
- There is a real baseline gap between popularity/retrieval heuristics and the true objective.
- The environment is cheap enough to train against at scale because runtime is pure lookup, not live LLM evaluation.

## What Ships In V1

Blindspot v1 is intentionally small but fully real-data grounded:

| Artifact | Value |
|---|---:|
| Real ML researchers | 17 |
| Train / held-out split | 13 / 4 |
| Candidate concept catalog | 1,168 |
| Reading paths | 282 |
| Adoption pairs | 62 |
| Comprehension pairs | 23 |

The held-out test users are the four users with the richest post-T adoption signal, stored in `data/user_splits.json`.

## Real-Data Calibration

Measured over 5 seeds × 17 users using the current real dataset:

| Policy | Mean reward | Std |
|---|---:|---:|
| Random | -0.01 | 0.56 |
| Trending | +1.11 | 0.46 |
| Dense Retrieval | +0.41 | 0.65 |
| Blindspot proxy | +1.99 | 1.09 |
| Oracle (upper bound) | +2.77 | 1.20 |

These numbers matter for the story:

- `Random ≈ 0` confirms the false-positive penalty is calibrated instead of reward-inflated.
- `Trending > Dense Retrieval` reflects how popularity can beat naive semantic similarity in a fast-moving field.
- `Oracle - Trending ≈ 1.66` shows there is still real headroom left for RL.
- The live demo now exposes a before/after view using a pre-training cache and a post-GRPO cache on the same queries.

The calibration script lives in `scripts/baseline_eval.py` and writes `data/baseline_calibration.json`.

## How The Demo Works

The public Hugging Face Space is designed to stay stable for the full judging window on a free CPU tier.

1. A real user, persona, or free-form paragraph is mapped onto the closest user profile in the dataset.
2. A 50-concept candidate pool is assembled.
3. Five policies are compared side by side: Random, Trending, Dense Retrieval, Blindspot pre-training, and Blindspot RL.
4. The Blindspot panels are served from two deterministic caches:
   - `data/demo_cache.json` for the trained GRPO policy
   - `data/demo_cache_pretrain.json` for the same base model with adapters disabled
5. Every surfaced concept shows its reading path, adoption verdict, comprehension lift, and latency.

Because the Space reads cached policy outputs, it does not need a GPU at request time. That keeps the demo cheap, deterministic, and much less likely to fail during judging.

## Training Recipe

Blindspot trains a LoRA adapter on top of `unsloth/Qwen3.5-9B` loaded in 4-bit mode by Unsloth.

1. Optionally attach an SFT warm-start adapter if `training/checkpoints/sft` exists.
2. Build GRPO prompts from the 13 training users in `data/user_splits.json`.
3. Run GRPO with 8 generations per prompt against the live OpenEnv server; each reward rolls out a short multi-step episode through `/reset` and `/step`.
4. Evaluate on the 4 held-out users and on the full 17-user set.
5. Save reward curves, reward-component plots, all-user summaries, and held-out summaries.
6. Precompute demo caches for both pre-training and post-training variants.
7. Push the adapter to the Hub and deploy the Space.

The main training notebook is `notebooks/02_training.ipynb`. The model card template is in `training/MODEL_CARD.md` and is uploaded as the Hub README by `scripts/push_to_hub.py`.

## OpenEnv Surface

Action space:

```python
{"type": "inspect", "concept_id": 42}
{"type": "surface", "concept_id": 42}
{"type": "stop"}
```

Observation highlights:

- `user_summary`
- `candidate_concepts`
- `inspected`
- `surfaced`
- `inspect_budget_remaining`
- `surface_budget_remaining`
- `reward_breakdown` on episode end

Server state now also exposes a `reasoning_log` with per-step action outcomes, which helps explain trajectories during debugging and review.

## OpenEnv Criteria Checklist

- **Environment manifest:** `openenv.yaml` declares the FastAPI app entry point.
- **API surface:** `server.app:app` exposes `/reset`, `/step`, `/state`, `/schema`, and `/ws` through `openenv.create_app`.
- **Typed action/observation/state:** `models.py` uses OpenEnv core types and Pydantic schemas.
- **Real RL training:** `notebooks/02_training.ipynb` runs TRL GRPO against the live HTTP environment.
- **Session-level reward:** GRPO completions are scored by multi-step OpenEnv rollouts, not static dataset scoring.
- **Held-out evidence:** the notebook reserves `data/user_splits.json` test users and writes `plots/summary_heldout_with_trained.json` after training.
- **Demo:** the Hugging Face Space gives judges a stable, visual replay of pre-training vs GRPO behavior.

## Quickstart

Run the environment locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run the conformance tests:

```bash
python -m pytest tests/test_openenv_compliance.py -v
```

Generate the judge-facing notebooks:

```bash
python scripts/build_training_notebook.py
python scripts/build_demo_notebook.py
```

Deploy the Space after precomputing caches:

```bash
HF_TOKEN=hf_xxx python scripts/precompute_demo_cache.py
HF_TOKEN=hf_xxx python scripts/deploy_to_space.py
```

## Repo Layout

```text
blindspot-env/
├── README.md
├── inference.py
├── models.py
├── notebooks/
│   ├── 02_training.ipynb
│   └── 03_demo.ipynb
├── scripts/
│   ├── baseline_eval.py
│   ├── blindspot_demo.py
│   ├── build_demo_notebook.py
│   ├── build_training_notebook.py
│   ├── deploy_to_space.py
│   ├── precompute_demo_cache.py
│   ├── push_to_hub.py
│   └── reward_ablation.py
├── server/
│   ├── app.py
│   ├── blindspot_environment.py
│   ├── data_loader.py
│   └── rewards.py
├── spaces/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── tests/
│   └── test_openenv_compliance.py
├── training/
│   ├── MODEL_CARD.md
│   ├── generate_sft_traces.py
│   ├── grpo_train.py
│   └── sft_train.py
└── data/
    ├── baseline_calibration.json
    ├── concept_catalog.json
    ├── user_splits.json
    └── ...
```

## Limitations & Honest Failures

- The dataset is still small: 17 users is enough to prove the environment shape, not enough to claim broad generalization.
- Adoption uses a kNN backoff when the exact user has no direct signal for a concept. That is a pragmatic proxy, not a causal estimate.
- Comprehension lift is measured with LLM judges, not humans.
- The public demo is cache-backed, so it is faithful and stable, but not a live online RL loop.
- The current paragraph mode uses nearest-neighbor lookup when the trained model is not actively loaded.

## Why This Is Still Strong For The Hackathon

Blindspot is not just another RAG wrapper or benchmark over known items. It turns a common human pain point into an OpenEnv-compatible RL environment with:

- a concrete, universal user story
- multi-component reward that is hard to game
- real held-out adoption signal
- a fast training loop
- a public demo that can stay online without paid GPU hosting

That combination is the core bet: a novel environment, a defensible reward, and a stable end-to-end demo.