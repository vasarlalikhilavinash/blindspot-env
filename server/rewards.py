"""Reward computation for the Blindspot environment.

All reward components are pure lookups against pre-computed tables — no
network, no LLM calls. A `step()` invocation that triggers reward
computation completes in microseconds.

Components (all independent, see README for justification):
    adoption       : ground-truth user adoption (1.0 self / 0.3 k-NN partial)
    novelty        : +0.5 bonus per concept NOT in trending feeds at T
    onboarding     : pre-computed comprehension lift (judge-agreed) per concept
    efficiency     : −0.01 per inspect call (budget already caps at 15)
    false_positive : −0.1 per surfaced concept with zero adoption signal

The false-positive penalty is calibrated so that a uniformly random
surface policy yields E[reward] ≈ 0 — see notebooks/01_demo.ipynb for
the empirical calibration cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .data_loader import BlindspotData


# ---------------------------------------------------------------------------
# Tunable reward constants — DO NOT CHANGE without recalibrating false-positive
# ---------------------------------------------------------------------------

NOVELTY_BONUS = 0.5
INSPECT_PENALTY = 0.01
FALSE_POSITIVE_PENALTY = 0.1
FALSE_POSITIVE_THRESHOLD = 1e-6  # below this, treat adoption signal as "none"


@dataclass
class RewardResult:
    adoption: float = 0.0
    novelty: float = 0.0
    onboarding: float = 0.0
    efficiency: float = 0.0
    false_positive: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.adoption
            + self.novelty
            + self.onboarding
            + self.efficiency
            + self.false_positive
        )


def compute_episode_reward(
    data: BlindspotData,
    user_id: str,
    surfaced_concept_ids: Iterable[int],
    inspect_count: int,
) -> RewardResult:
    """Compute the full episode reward breakdown.

    Called once at episode end (on `stop` action or budget exhaustion).
    Pure function — only reads from `data`.
    """
    result = RewardResult()
    adoption_table = data.adoption.get(user_id, {})
    comprehension_table = data.comprehension.get(user_id, {})

    for cid in surfaced_concept_ids:
        # 1. Adoption — primary reward signal, ground truth from user's
        #    actual post-T research artifacts.
        adoption_score = float(adoption_table.get(cid, 0.0))
        adopted = adoption_score >= FALSE_POSITIVE_THRESHOLD
        result.adoption += adoption_score

        # 2. Novelty — +0.5 if NOT trending bait, GATED on adoption.
        #    Without the gate an agent could rack up novelty by surfacing
        #    obscure-but-irrelevant concepts. The gate enforces "novel AND
        #    actually-relevant" — true unknown-unknowns.
        is_novel = bool(data.novelty.get(cid, False))
        if adopted and is_novel:
            result.novelty += NOVELTY_BONUS

        # 3. Onboarding — pre-computed comprehension lift (Claude+GPT must
        #    agree, κ ≥ 0.7). Also gated on adoption: a great reading path
        #    for a concept the user doesn't care about contributes nothing.
        if adopted:
            result.onboarding += float(comprehension_table.get(cid, 0.0))

        # 4. False-positive — discourage noise-surfacing.
        if not adopted:
            result.false_positive -= FALSE_POSITIVE_PENALTY

    # 5. Efficiency — small per-inspect cost; budget cap is the hard limit.
    result.efficiency = -INSPECT_PENALTY * float(inspect_count)

    return result


# ---------------------------------------------------------------------------
# Tiny incremental shaping signal — applied per surface action so policy
# gradients aren't entirely sparse over the episode. Magnitude is
# intentionally << final reward so the dominant signal is end-of-episode.
# ---------------------------------------------------------------------------


def shaping_reward_for_surface(data: BlindspotData, user_id: str, concept_id: int) -> float:
    """Tiny incremental signal so SFT/GRPO has dense gradient.

    Returns a small fraction of what the final reward will contribute,
    so it does not dominate the calibrated episode reward.
    """
    adoption_table = data.adoption.get(user_id, {})
    score = float(adoption_table.get(concept_id, 0.0))
    return 0.1 * score  # 10% of the eventual adoption contribution
