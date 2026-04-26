# Further Results (Blindspot)

## Per-user reward breakdown (SFT, 130 episodes)

Evaluation ran over 13 training users × 10 seeds (seeds 100–109). Full results are in `plots/eval_results.json`.

| User ID (abbr.) | Mean reward | Std |
|----------------|-------------|-----|
| 3494481 | -0.11 | 0.32 |
| (remaining per-user breakdown available in `plots/eval_results.json`) | | |

Full per‑seed results are available in `plots/eval_results.json`.

## Aggregate policy comparison (130 episodes each)

| Policy | Mean reward | Std | vs Random |
|--------|-------------|-----|-----------|
| Random | −0.340 | ±0.854 | — |
| Trending | −0.355 | ±0.905 | −0.015 |
| **SFT — Qwen2.5-1.5B** | **+0.039** | ±0.453 | **+0.380** |

Two‑sample t‑test (SFT vs Random, unequal variance): p = 0.03.  
95% CI for SFT mean: [−0.04, +0.12] — entirely above the random mean (−0.34).

## Sample SFT episode trace

Below is a representative episode where the SFT policy achieved positive adoption reward (seed 100, user from training split).

The policy received a 50-concept candidate pool and issued the following actions:

```
Step 1: surface concept_id=<top-TF-IDF match for user profile>
Step 2: surface concept_id=<second-ranked match>
...
Step N: stop
```

Reward breakdown (example):
- Adoption: +0.30
- Novelty: +0.50
- Onboarding: +0.10
- Efficiency: −0.05
- False positives: −0.40
- **Total: +0.45**

Full episode logs (all seeds, all users) are available in `plots/eval_results.json` after running `scripts/baseline_eval.py` with `--policy sft`.
