"""Pydantic types for the Blindspot environment.

These define the action / observation / state contract that flows over
the OpenEnv FastAPI surface. They mirror the conventions in
`incident-triage-env` for compatibility with the OpenEnv tooling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action verbs (deliberately small to keep RL credit assignment tractable)
# ---------------------------------------------------------------------------

ACTION_TYPES: List[str] = ["inspect", "surface", "stop"]

# Hard caps applied per episode. Make sure these match server defaults
# AND the values referenced in the README.
DEFAULT_INSPECT_BUDGET: int = 15
DEFAULT_SURFACE_BUDGET: int = 10
DEFAULT_CANDIDATE_POOL_SIZE: int = 50


class BlindspotAction(Action):
    """Action emitted by the agent each step.

    The action space is intentionally tiny — three verbs — so credit
    assignment under GRPO is feasible from a cold start.
    """

    type: Literal["inspect", "surface", "stop"] = Field(
        description=(
            "Verb to execute. 'inspect' reveals a ConceptDetail (consumes "
            "inspect budget). 'surface' commits a concept as a recommendation "
            "(consumes surface budget; locked-in for grading). 'stop' ends the "
            "episode early and triggers final reward computation."
        )
    )
    concept_id: Optional[int] = Field(
        default=None,
        description=(
            "Concept ID to inspect or surface. Required for 'inspect' and "
            "'surface'. Ignored for 'stop'."
        ),
    )


# ---------------------------------------------------------------------------
# Concept payloads
# ---------------------------------------------------------------------------


class ConceptCard(BaseModel):
    """Lightweight card shown in the candidate pool — cheap to serialize."""

    concept_id: int
    title: str
    one_liner: str = Field(
        description="One-sentence description (~30 tokens).",
    )


class ConceptDetail(BaseModel):
    """Full payload returned by an `inspect` action — only loaded on demand."""

    concept_id: int
    title: str
    abstract_summary: str = Field(
        description="3-5 sentence summary distilled from top papers."
    )
    top_papers: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {title, arxiv_id, year} for the canonical 5-paper reading path.",
    )
    growth_signal: float = Field(
        default=0.0,
        description=(
            "Normalized growth rate of the concept in the corpus over the "
            "12 months pre-T (0..1). Higher = faster growing."
        ),
    )
    is_trending: bool = Field(
        default=False,
        description="True if this concept appeared in trending feeds at T (NOT a true unknown-unknown).",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class RewardBreakdown(BaseModel):
    """Per-concept reward decomposition — populated only on episode end."""

    adoption: float = 0.0
    novelty: float = 0.0
    onboarding: float = 0.0
    efficiency: float = 0.0
    false_positive: float = 0.0
    total: float = 0.0


class BlindspotObservation(Observation):
    """Observation returned to the agent each step."""

    message: str = Field(default="", description="Human-readable feedback / status message.")
    user_summary: str = Field(
        default="",
        description="~200-token summary of the user's pre-T research profile.",
    )
    user_id: str = Field(default="", description="Stable user identifier.")
    candidate_concepts: List[ConceptCard] = Field(
        default_factory=list,
        description="The 50 (default) candidate concepts on offer for this episode.",
    )
    inspected: Dict[str, ConceptDetail] = Field(
        default_factory=dict,
        description="ConceptDetails revealed so far this episode, keyed by concept_id (str).",
    )
    surfaced: List[int] = Field(
        default_factory=list,
        description="Concept IDs the agent has committed to surface.",
    )
    inspect_budget_remaining: int = Field(default=DEFAULT_INSPECT_BUDGET)
    surface_budget_remaining: int = Field(default=DEFAULT_SURFACE_BUDGET)
    available_actions: List[str] = Field(
        default_factory=lambda: list(ACTION_TYPES),
        description="Valid action verbs.",
    )
    user_id_pool: List[str] = Field(
        default_factory=list,
        description="All available user IDs (returned on reset for convenience).",
    )
    step_number: int = Field(default=0)
    max_steps: int = Field(
        default=DEFAULT_INSPECT_BUDGET + DEFAULT_SURFACE_BUDGET + 2,
        description="Inspect budget + surface budget + 2 (final 'stop' + safety belt).",
    )
    reward_breakdown: Optional[RewardBreakdown] = Field(
        default=None,
        description="Final per-component reward breakdown — populated only when done.",
    )


class BlindspotState(State):
    """Server-side bookkeeping returned by GET /state."""

    user_id: str = Field(default="")
    candidate_pool_size: int = Field(default=0)
    inspect_count: int = Field(default=0)
    surface_count: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
