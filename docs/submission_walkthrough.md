# Blindspot OpenEnv Submission Walkthrough

## Problem Statement

Researchers do not only need answers to questions they already know how to ask. They also need help discovering useful AI/ML concepts that are missing from their current vocabulary but are likely to matter for their future work. Blindspot turns that unknown-unknown discovery problem into an OpenEnv environment.

## Environment

Blindspot exposes a FastAPI OpenEnv server with a small action space:

- `inspect`: spend budget to reveal evidence about a candidate concept.
- `surface`: recommend a concept to the researcher.
- `stop`: end the episode and receive the final reward breakdown.

The observation contains the user summary, candidate concepts, inspected evidence, surfaced concepts, remaining budgets, and final reward breakdown. The server also exposes state with a reasoning log for replay and debugging.

## Reward

The reward is computed from real, precomputed data rather than live LLM judging:

- adoption: whether the user later adopted the concept in post-T work.
- novelty: bonus for relevant concepts that were not merely trending.
- onboarding: comprehension lift from the recommended reading path.
- efficiency: small inspect-cost penalty.
- false_positive: penalty for surfacing concepts with no adoption signal.

This makes the task cheap enough for RL while preserving delayed, session-level credit assignment.

## Training

The Colab notebook trains a bf16 LoRA adapter on `unsloth/Qwen3.5-9B` using TRL GRPO against the live OpenEnv HTTP server.

The current notebook is aligned to the OpenEnv criteria in these ways:

- It starts the actual environment server and verifies `/reset` before training.
- It samples prompts from the declared training split in `data/user_splits.json`.
- It scores GRPO completions by rolling out short multi-step episodes through `/step`.
- It saves reward curves and reward-component plots.
- It evaluates the trained policy on held-out users and on all users.
- It writes `plots/summary_with_trained.json` and `plots/summary_heldout_with_trained.json` after training.

## Demo

The Hugging Face Space provides a stable CPU demo using deterministic precomputed policy caches. It compares random, trending, dense retrieval, pre-training, and GRPO-trained behavior, then visualizes how the policy improves action by action.

## Final Evidence To Attach After Colab Finishes

Commit or link these generated artifacts after the run completes:

- `plots/training_reward_curve.png`
- `plots/training_component_curves.png`
- `plots/comparison_with_trained.png`
- `plots/decomposition_with_trained.png`
- `plots/per_user_reward.png`
- `plots/action_type_distribution.png`
- `plots/summary_with_trained.json`
- `plots/summary_heldout_with_trained.json`

These files provide the concrete training evidence judges can inspect without rerunning the GPU job.
