---
license: apache-2.0
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
tags:
  - lora
  - grpo
  - trl
  - unsloth
  - openenv
  - rl
  - recommender
  - blindspot
language:
  - en
library_name: peft
pipeline_tag: text-generation
---

# Blindspot · Qwen2.5-7B-Instruct · GRPO LoRA Adapter

Fine-tuned LoRA adapter for the [Blindspot OpenEnv environment](https://github.com/vasarlalikhilavinash/blindspot-env) — an RL benchmark for **unknown-unknowns discovery**: given a researcher's profile, surface the AI/ML concepts they should be tracking but currently aren't.

## Training

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Method | **GRPO** (TRL) via Unsloth |
| Reward | Adoption + Novelty + Onboarding − False-positive (4-component, gated) |
| Train users | 13 (held-out test: 4) |
| Steps | 400 × 8 generations |
| Hardware | Single A100 (Colab Pro) |
| Wall time | ~90 min |

## Eval

Measured on the Blindspot env, mean reward across 5 seeds × 17 users:

| Policy | Mean reward |
|---|---|
| Random | -0.01 |
| Trending-only | +1.11 |
| Dense Retrieval | +0.41 |
| Blindspot proxy (pre-training) | +1.99 |
| **Blindspot GRPO (this adapter)** | (see Space) |
| Oracle (upper bound) | +2.77 |

## Use

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", load_in_4bit=True
)
model.load_adapter("vasarlalikhilavinash/blindspot-qwen25-7b-grpo")
FastLanguageModel.for_inference(model)
```

The action format is a single-line JSON command per turn:
```json
{"type": "surface", "concept_id": 42}
```

See [the env code](https://github.com/vasarlalikhilavinash/blindspot-env/blob/main/server/blindspot_environment.py) for the full action/observation schema.

## Limitations

- **v1 dataset is small** (17 ML researchers, 1168 concepts). Generalization to other fields is untested.
- Reward signal uses a kNN proxy when direct adoption ground truth is sparse; a future causal counterfactual reward model would be more rigorous.
- Comprehension scores are LLM-judged; the README documents the inter-judge agreement check.

## Citation

```
@misc{blindspot2026,
  title  = {Blindspot: An OpenEnv Environment for Unknown-Unknowns Discovery},
  author = {Vasarla, Likhil Avinash},
  year   = {2026},
  howpublished = {OpenEnv Hackathon},
  url    = {https://github.com/vasarlalikhilavinash/blindspot-env}
}
```
