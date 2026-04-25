"""HTTP client for the Blindspot environment."""

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import BlindspotAction, BlindspotObservation, BlindspotState
except ImportError:
    from models import BlindspotAction, BlindspotObservation, BlindspotState


class BlindspotEnvClient(EnvClient[BlindspotAction, BlindspotObservation, BlindspotState]):
    """Client for connecting to a remote Blindspot environment."""

    def _step_payload(self, action: BlindspotAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[BlindspotObservation]:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=BlindspotObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> BlindspotState:
        return BlindspotState(**payload)
