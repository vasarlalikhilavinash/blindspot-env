"""FastAPI application for the Blindspot environment.

Endpoints (all produced by openenv.create_app):
    POST /reset
    POST /step
    GET  /state
    GET  /schema
    WS   /ws
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core[core]>=0.2.2 is required.") from e

try:
    from ..models import BlindspotAction, BlindspotObservation
    from .blindspot_environment import BlindspotEnvironment
except ImportError:
    from models import BlindspotAction, BlindspotObservation
    from server.blindspot_environment import BlindspotEnvironment


app = create_app(
    BlindspotEnvironment,
    BlindspotAction,
    BlindspotObservation,
    env_name="blindspot_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
