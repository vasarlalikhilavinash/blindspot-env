#!/usr/bin/env python3
"""OpenEnv conformance smoke tests for the Blindspot environment.

Asserts that the environment exposes the standard reset/step/state contract
and that observation/action shapes are well-formed. Run with:

    python -m pytest tests/test_openenv_compliance.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from server.blindspot_environment import BlindspotEnvironment
from models import BlindspotAction


@pytest.fixture
def env():
    e = BlindspotEnvironment()
    yield e
    e.close()


def test_reset_returns_well_formed_observation(env):
    obs = env.reset(seed=0)
    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.user_id != ""
    assert isinstance(obs.candidate_concepts, list)
    assert len(obs.candidate_concepts) > 0
    assert obs.inspect_budget_remaining > 0
    assert obs.surface_budget_remaining > 0
    assert "inspect" in obs.available_actions
    assert "surface" in obs.available_actions
    assert "stop" in obs.available_actions


def test_state_is_consistent_after_reset(env):
    env.reset(seed=42)
    s = env.state
    assert s.step_count == 0
    assert s.surface_count == 0
    assert s.inspect_count == 0
    assert s.cumulative_reward == 0.0


def test_inspect_then_surface_then_stop(env):
    obs = env.reset(seed=7)
    cid = obs.candidate_concepts[0].concept_id
    obs = env.step(BlindspotAction(type="inspect", concept_id=cid))
    assert obs.done is False
    assert str(cid) in obs.inspected
    obs = env.step(BlindspotAction(type="surface", concept_id=cid))
    assert obs.done is False
    assert cid in obs.surfaced
    obs = env.step(BlindspotAction(type="stop"))
    assert obs.done is True
    assert obs.reward_breakdown is not None
    assert len(env.state.reasoning_log) == 3
    assert env.state.reasoning_log[-1]["action"] == "stop"


def test_reset_is_deterministic_with_seed(env):
    obs1 = env.reset(seed=123)
    ids1 = [c.concept_id for c in obs1.candidate_concepts]
    obs2 = env.reset(seed=123)
    ids2 = [c.concept_id for c in obs2.candidate_concepts]
    assert ids1 == ids2, "reset(seed=N) must produce a reproducible candidate order"


def test_surface_budget_is_enforced(env):
    obs = env.reset(seed=0)
    cids = [c.concept_id for c in obs.candidate_concepts]
    surfaces_made = 0
    for cid in cids:
        if obs.done:
            break
        obs = env.step(BlindspotAction(type="surface", concept_id=cid))
        if not obs.done:
            surfaces_made += 1
    # Episode must terminate within the surface budget
    assert obs.done is True
    assert surfaces_made <= 10, "surface budget cap is 10"


def test_action_schema_rejects_unknown_verbs(env):
    # The pydantic action model should reject anything outside {inspect, surface, stop}
    with pytest.raises(Exception):
        BlindspotAction(type="bogus_verb")


def test_user_id_pool_matches_data(env):
    obs = env.reset(seed=0)
    assert len(obs.user_id_pool) >= 17, "expected at least 17 users in v1 dataset"
