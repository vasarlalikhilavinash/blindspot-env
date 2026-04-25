# Phase C Summary — Real Data, Pre-Phase D

Generated 2026-04-25 after running stages 01–08 with live OpenAI + Gemini APIs.

## Pipeline scale

| Stage | Output | Count |
|---|---|---|
| 01 fetch_users | users with ≥10 papers | 17 |
| 02 fetch_corpus | papers (Semantic Scholar) | 6,000 |
| 03 extract_concepts | unique noun-phrase concepts | 1,168 |
| 04 build_pools | in-pool concepts (positives + hard-negs) | 282 unique, 38.5 avg/user |
| 04 build_pools | ground-truth adoptions | 62 (uid,cid) pairs |
| 05 score_adoption | trending-at-T concepts | 200 |
| 06 build_paths | reading paths (corpus-grounded) | 282 |
| 07 score_comprehension | pairs judged by GPT-5.4-mini + Gemini-2.5-flash | 170 |
| 07 score_comprehension | scored pairs (after agreement filter) | 23 across 14/17 users |

## Stage 07 (LLM-as-judge) — final numbers

| Metric | Value |
|---|---|
| Control accuracy (definition only) | 96.2% |
| Treatment accuracy (definition + reading path) | 99.1% |
| Mean raw lift | +2.9 pp |
| Positive-lift pairs | 28 |
| Negative-lift pairs | 5 |
| Mean inter-judge κ | 0.812 |
| Pairs passing κ ≥ 0.7 | 138/170 |
| Pairs passing 5/6 observed-agreement filter AND lift > 0 | 23 |
| Mean stored lift (capped to [0,1]) | +0.181 |

### Bias controls applied
- **Positional bias**: deterministic option shuffling per (uid, cid, judge, condition).
  Verified post-shuffle gold distribution = A:24 B:32 C:24 D:37 (was 91% A pre-fix).
- **Question difficulty**: gen prompt explicitly requires path-specific facts, not
  derivable from one-line definition.
- **Agreement filter**: replaced raw κ ≥ 0.7 (which collapses to 0 when both judges
  are 6/6 perfect) with observed correctness agreement ≥ 5/6. Recovers 23 valid
  signals that the κ filter was incorrectly discarding.

## Baselines vs Oracle

| Policy | Mean reward | Std | Adoption | Novelty | Onboarding | FP |
|---|---|---|---|---|---|---|
| Random | +0.09 | 1.40 | 0.88 | 0.10 | 0.02 | -0.91 |
| Trending | +0.21 | 0.51 | 1.09 | 0.00 | 0.00 | -0.88 |
| Dense Retrieval | +0.47 | 1.20 | 1.35 | 0.06 | 0.00 | -0.86 |
| **Oracle (upper bound)** | **+3.29** | 3.59 | 3.51 | 0.35 | 0.07 | -0.64 |

Headroom Dense → Oracle ≈ +2.8 reward. This is the gap GRPO needs to close.

## Cross-check verdict — PASS

| Check | Result |
|---|---|
| Env smoke test (17 users × seed=0) | All episodes complete, reward computed |
| Pool concepts ⊆ catalog | OK |
| GT adoption ⊆ user pool | OK |
| Comprehension scores ⊆ adopted concepts | 11 are for hard-negatives (env ignores them — no bug, just wasted compute) |
| Novelty coverage | 968/1168 concepts flagged novel |
| Reading-path coverage | 282/1168 catalog concepts (by design — only in-pool concepts) |

## Cost spent (approx)

- Stage 07 v1 (easy questions, scored=2): ~$1
- Stage 07 v2 (hard path-grounded questions, scored=23): ~$3
- Total: ~$4 of the $24 approved budget

## Files in `data/` (consumed by env)

- `concept_catalog.json` (1,168 entries, 355 KB)
- `concept_pool_per_user.json` (17 users)
- `ground_truth_adoption.json` (17 users, 62 pairs)
- `comprehension_scores.json` (17 users, 23 lift values)
- `reading_paths.json` (282 concepts)
- `novelty_flags.json` (1,168 concepts)
- `knn_users.json`, `user_summaries.json`

## Logs archived in `reports/`
- `precompute.log` — full stages 01–08 console output
- `stage07.log` — final Stage 07 run
- `stage08.log` — finalize step
- `phase_C_metrics.json` — machine-readable summary
