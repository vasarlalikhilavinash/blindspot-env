"""Blindspot Environment — server-side implementation.

Implements the OpenEnv `Environment` interface with three actions:
  - inspect(concept_id) -> ConceptDetail revealed
  - surface(concept_id) -> commit concept as a recommendation
  - stop                 -> end episode, return final reward breakdown

Episode termination conditions:
  1. Agent emits 'stop'
  2. Surface budget exhausted (every surface auto-stops at 0)
  3. Step count exceeds inspect_budget + surface_budget + 1

All work is pure-lookup against pre-computed `BlindspotData`. No
network or LLM calls happen on the hot path.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ACTION_TYPES,
        DEFAULT_CANDIDATE_POOL_SIZE,
        DEFAULT_INSPECT_BUDGET,
        DEFAULT_SURFACE_BUDGET,
        BlindspotAction,
        BlindspotObservation,
        BlindspotState,
        ConceptCard,
        ConceptDetail,
        RewardBreakdown,
    )
except ImportError:  # when run as plain `server.app:app`
    from models import (
        ACTION_TYPES,
        DEFAULT_CANDIDATE_POOL_SIZE,
        DEFAULT_INSPECT_BUDGET,
        DEFAULT_SURFACE_BUDGET,
        BlindspotAction,
        BlindspotObservation,
        BlindspotState,
        ConceptCard,
        ConceptDetail,
        RewardBreakdown,
    )

try:
    from .data_loader import BlindspotData, load_data
    from .rewards import compute_episode_reward, shaping_reward_for_surface
except ImportError:
    from server.data_loader import BlindspotData, load_data
    from server.rewards import compute_episode_reward, shaping_reward_for_surface


class BlindspotEnvironment(
    Environment[BlindspotAction, BlindspotObservation, BlindspotState]
):
    """OpenEnv environment for unknown-unknowns discovery + onboarding."""

    def __init__(self) -> None:
        super().__init__()
        self._data: BlindspotData = load_data()

        # Per-episode state — initialized in reset()
        self._user_id: str = ""
        self._candidate_ids: List[int] = []
        self._inspected: Dict[int, ConceptDetail] = {}
        self._surfaced: List[int] = []
        self._inspect_count: int = 0
        self._surface_count: int = 0
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._episode_id: str = str(uuid4())
        self._max_steps: int = DEFAULT_INSPECT_BUDGET + DEFAULT_SURFACE_BUDGET + 2
        # Per-step reasoning log (interpretability hook). Each entry:
        #   {"step": int, "action": str, "concept_id": int|None, "reward": float, "note": str}
        # Populated by step(); exposed via state.reasoning_log when available.
        self._reasoning_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BlindspotObservation:
        """Start a new episode for `user_id` (or a deterministic default)."""
        rng = random.Random(seed)

        all_users = self._data.user_ids
        if not all_users:
            raise RuntimeError("No users in pre-computed data — run scripts/build_synthetic_seed.py.")

        if user_id and user_id in self._data.user_summaries:
            self._user_id = user_id
        else:
            # Pick deterministically based on seed for reproducibility
            self._user_id = all_users[rng.randint(0, len(all_users) - 1)]

        # Candidate pool — shuffled per reset to defeat positional hacking
        pool = list(self._data.concept_pool.get(self._user_id, []))
        rng.shuffle(pool)
        self._candidate_ids = pool[:DEFAULT_CANDIDATE_POOL_SIZE]

        self._inspected = {}
        self._surfaced = []
        self._inspect_count = 0
        self._surface_count = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_id = episode_id or str(uuid4())
        self._reasoning_log = []

        return self._make_observation(
            message=(
                f"=== Blindspot · user={self._user_id} ===\n"
                f"You have {len(self._candidate_ids)} candidate concepts. "
                f"Use 'inspect' to reveal a concept's reading path (budget {DEFAULT_INSPECT_BUDGET}), "
                f"'surface' to recommend it (budget {DEFAULT_SURFACE_BUDGET}), "
                f"or 'stop' to end the episode early.\n"
                f"User profile: {self._data.user_summaries[self._user_id][:200]}..."
            ),
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: BlindspotAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BlindspotObservation:
        """Execute one action."""
        if self._done:
            return self._make_observation(
                message="Episode is already done. Call reset() to start a new episode.",
                reward=0.0,
                done=True,
            )

        self._step_count += 1
        verb = action.type

        # Hard step cap as a safety belt (budget caps usually trigger first)
        if self._step_count > self._max_steps:
            obs = self._end_episode(
                message=(
                    f"Step limit ({self._max_steps}) exceeded — terminating episode."
                ),
            )
            self._record_reasoning(action, obs)
            return obs

        if verb == "inspect":
            obs = self._do_inspect(action.concept_id)
            self._record_reasoning(action, obs)
            return obs
        if verb == "surface":
            obs = self._do_surface(action.concept_id)
            self._record_reasoning(action, obs)
            return obs
        if verb == "stop":
            obs = self._end_episode(message="Agent emitted 'stop'.")
            self._record_reasoning(action, obs)
            return obs

        # Unknown verb — small penalty, episode continues
        obs = self._make_observation(
            message=(
                f"Unknown action type '{verb}'. Valid: {', '.join(ACTION_TYPES)}."
            ),
            reward=-0.01,
            done=False,
        )
        self._record_reasoning(action, obs)
        return obs

    @property
    def state(self) -> BlindspotState:
        return BlindspotState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            user_id=self._user_id,
            candidate_pool_size=len(self._candidate_ids),
            inspect_count=self._inspect_count,
            surface_count=self._surface_count,
            cumulative_reward=self._cumulative_reward,
            reasoning_log=list(self._reasoning_log),
        )

    def close(self) -> None:
        """Clean up resources. Nothing to do — pure in-memory env."""
        pass

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_inspect(self, concept_id: Optional[int]) -> BlindspotObservation:
        if concept_id is None:
            return self._make_observation(
                message="Error: 'inspect' requires concept_id.",
                reward=-0.01,
                done=False,
            )
        if concept_id not in self._candidate_ids:
            return self._make_observation(
                message=f"Error: concept_id {concept_id} not in candidate pool for this user.",
                reward=-0.01,
                done=False,
            )
        if self._inspect_count >= DEFAULT_INSPECT_BUDGET:
            return self._make_observation(
                message=(
                    f"Inspect budget exhausted ({DEFAULT_INSPECT_BUDGET}). "
                    f"You can still 'surface' or 'stop'."
                ),
                reward=-0.01,
                done=False,
            )
        if concept_id in self._inspected:
            return self._make_observation(
                message=f"Concept {concept_id} already inspected — no extra cost charged.",
                reward=0.0,
                done=False,
            )

        record = self._data.concept_catalog.get(concept_id)
        if record is None:
            return self._make_observation(
                message=f"Internal error: concept {concept_id} not found in catalog.",
                reward=-0.01,
                done=False,
            )

        detail = ConceptDetail(
            concept_id=concept_id,
            title=record["title"],
            abstract_summary=record.get("abstract_summary", ""),
            top_papers=self._data.reading_paths.get(concept_id, []),
            growth_signal=float(record.get("growth_signal", 0.0)),
            is_trending=bool(record.get("is_trending", False)),
        )
        self._inspected[concept_id] = detail
        self._inspect_count += 1

        return self._make_observation(
            message=(
                f"Inspected concept {concept_id}: '{detail.title}'. "
                f"{len(detail.top_papers)} papers in reading path. "
                f"Inspect budget: {DEFAULT_INSPECT_BUDGET - self._inspect_count} left."
            ),
            reward=0.0,
            done=False,
        )

    def _do_surface(self, concept_id: Optional[int]) -> BlindspotObservation:
        if concept_id is None:
            return self._make_observation(
                message="Error: 'surface' requires concept_id.",
                reward=-0.01,
                done=False,
            )
        if concept_id not in self._candidate_ids:
            return self._make_observation(
                message=f"Error: concept_id {concept_id} not in candidate pool.",
                reward=-0.01,
                done=False,
            )
        if concept_id in self._surfaced:
            return self._make_observation(
                message=f"Concept {concept_id} already surfaced — no double credit.",
                reward=-0.01,
                done=False,
            )
        if self._surface_count >= DEFAULT_SURFACE_BUDGET:
            # Auto-end the episode — surface budget is the binding constraint
            return self._end_episode(
                message=(
                    f"Surface budget exhausted ({DEFAULT_SURFACE_BUDGET}). Episode ends."
                ),
            )

        self._surfaced.append(concept_id)
        self._surface_count += 1
        shaping = shaping_reward_for_surface(self._data, self._user_id, concept_id)
        self._cumulative_reward += shaping

        msg = (
            f"Surfaced concept {concept_id}. Surface budget: "
            f"{DEFAULT_SURFACE_BUDGET - self._surface_count} left."
        )

        # If we just hit the cap, end the episode and return final reward
        if self._surface_count >= DEFAULT_SURFACE_BUDGET:
            return self._end_episode(
                message=msg + " (Final surface — closing episode.)",
                tail_shaping=shaping,
            )

        return self._make_observation(
            message=msg,
            reward=shaping,
            done=False,
        )

    def _end_episode(
        self,
        message: str,
        tail_shaping: float = 0.0,
    ) -> BlindspotObservation:
        breakdown = compute_episode_reward(
            self._data,
            self._user_id,
            self._surfaced,
            self._inspect_count,
        )
        # Subtract any shaping that was already credited so the cumulative
        # episode reward equals breakdown.total exactly (no double-counting).
        already_shaped = sum(
            shaping_reward_for_surface(self._data, self._user_id, cid)
            for cid in self._surfaced
        )
        final_step_reward = breakdown.total - already_shaped
        # `tail_shaping` was just added by the caller this very step;
        # we still want the reported reward this step to round-trip to total.
        # The caller already included tail_shaping in `already_shaped`, so no
        # extra subtraction needed.
        _ = tail_shaping  # currently unused — kept for API symmetry

        self._cumulative_reward = breakdown.total
        self._done = True

        return self._make_observation(
            message=(
                f"{message}\n"
                f"Episode complete · surfaced={len(self._surfaced)} "
                f"· inspected={self._inspect_count} "
                f"· total reward={breakdown.total:.4f} "
                f"(adoption={breakdown.adoption:.3f}, novelty={breakdown.novelty:.3f}, "
                f"onboarding={breakdown.onboarding:.3f}, efficiency={breakdown.efficiency:.3f}, "
                f"false_positive={breakdown.false_positive:.3f})"
            ),
            reward=final_step_reward,
            done=True,
            reward_breakdown=RewardBreakdown(
                adoption=breakdown.adoption,
                novelty=breakdown.novelty,
                onboarding=breakdown.onboarding,
                efficiency=breakdown.efficiency,
                false_positive=breakdown.false_positive,
                total=breakdown.total,
            ),
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _record_reasoning(
        self,
        action: BlindspotAction,
        obs: BlindspotObservation,
    ) -> None:
        self._reasoning_log.append(
            {
                "step": self._step_count,
                "action": action.type,
                "concept_id": action.concept_id,
                "reward": float(obs.reward),
                "done": bool(obs.done),
                "note": obs.message.splitlines()[0] if obs.message else "",
            }
        )

    def _make_observation(
        self,
        message: str,
        reward: float,
        done: bool,
        reward_breakdown: Optional[RewardBreakdown] = None,
    ) -> BlindspotObservation:
        candidates: List[ConceptCard] = []
        for cid in self._candidate_ids:
            rec = self._data.concept_catalog.get(cid)
            if rec is None:
                continue
            candidates.append(
                ConceptCard(
                    concept_id=cid,
                    title=rec.get("title", f"concept-{cid}"),
                    one_liner=rec.get("one_liner", ""),
                )
            )

        inspected_payload = {str(cid): det for cid, det in self._inspected.items()}

        return BlindspotObservation(
            done=done,
            reward=reward,
            message=message,
            user_summary=self._data.user_summaries.get(self._user_id, ""),
            user_id=self._user_id,
            candidate_concepts=candidates,
            inspected=inspected_payload,
            surfaced=list(self._surfaced),
            inspect_budget_remaining=max(0, DEFAULT_INSPECT_BUDGET - self._inspect_count),
            surface_budget_remaining=max(0, DEFAULT_SURFACE_BUDGET - self._surface_count),
            available_actions=list(ACTION_TYPES),
            user_id_pool=self._data.user_ids,
            step_number=self._step_count,
            max_steps=self._max_steps,
            reward_breakdown=reward_breakdown,
        )
