"""Loads pre-computed Blindspot artifacts into in-memory dicts at startup.

The runtime environment is a pure lookup over these structures — no network,
no LLM calls, sub-millisecond `step()`. All expensive computation happened
once, offline, in `scripts/precompute_*.py`.

Artifacts (all under `data/`):
    user_summaries.json          : { user_id -> str }                    ~100 KB
    concept_pool_per_user.json   : { user_id -> [concept_id, ...] }      ~50 KB
    concept_catalog.json         : { concept_id (str) -> ConceptRecord } ~5 MB
    ground_truth_adoption.json   : { user_id -> { concept_id -> float } }
    novelty_flags.json           : { concept_id (str) -> bool }
    comprehension_scores.json    : { user_id -> { concept_id -> float } }
    reading_paths.json           : { concept_id (str) -> [paper, ...] }
    knn_users.json               : { user_id -> [neighbor_user_id, ...] }

If any artifact is missing, `build_synthetic_seed.py` is automatically
invoked to materialize a small deterministic dev dataset (so the env
boots even on a fresh checkout). In the production HF Space deployment,
artifacts are baked in at image build time and the synthetic fallback
is never triggered.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
DATA_DIR = Path(os.environ.get("BLINDSPOT_DATA_DIR", _REPO_ROOT / "data"))


_REQUIRED_FILES = [
    "user_summaries.json",
    "concept_pool_per_user.json",
    "concept_catalog.json",
    "ground_truth_adoption.json",
    "novelty_flags.json",
    "comprehension_scores.json",
    "reading_paths.json",
    "knn_users.json",
]


# ---------------------------------------------------------------------------
# Container holding the loaded data
# ---------------------------------------------------------------------------


@dataclass
class BlindspotData:
    """In-memory snapshot of all pre-computed artifacts."""

    user_summaries: Dict[str, str]
    concept_pool: Dict[str, List[int]]
    concept_catalog: Dict[int, Dict[str, Any]]  # concept_id -> {title, one_liner, abstract_summary, growth_signal, is_trending}
    adoption: Dict[str, Dict[int, float]]
    novelty: Dict[int, bool]
    comprehension: Dict[str, Dict[int, float]]
    reading_paths: Dict[int, List[Dict[str, str]]]
    knn_users: Dict[str, List[str]]

    @property
    def user_ids(self) -> List[str]:
        return sorted(self.user_summaries.keys())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _materialize_synthetic_if_missing() -> None:
    """If any required artifact is missing, run the synthetic seed builder."""
    missing = [f for f in _REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        return

    seed_script = _REPO_ROOT / "scripts" / "build_synthetic_seed.py"
    if not seed_script.exists():
        raise FileNotFoundError(
            f"Pre-computed data missing ({missing}) and seed script not found at {seed_script}."
        )
    print(
        f"[blindspot] Missing data files {missing} — running synthetic seed builder...",
        file=sys.stderr,
    )
    subprocess.check_call([sys.executable, str(seed_script)])


def _load_json(name: str) -> Any:
    with (DATA_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _intkey(d: Dict[str, Any]) -> Dict[int, Any]:
    """Convert string-keyed dict (JSON requirement) back to int-keyed."""
    return {int(k): v for k, v in d.items()}


def _intkey_nested(d: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[int, Any]]:
    return {user: _intkey(inner) for user, inner in d.items()}


_CACHE: Optional[BlindspotData] = None


def load_data(force_reload: bool = False) -> BlindspotData:
    """Idempotent loader. Subsequent calls return the cached instance."""
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    _materialize_synthetic_if_missing()

    user_summaries = _load_json("user_summaries.json")
    concept_pool = _load_json("concept_pool_per_user.json")
    concept_catalog_raw = _load_json("concept_catalog.json")
    adoption_raw = _load_json("ground_truth_adoption.json")
    novelty_raw = _load_json("novelty_flags.json")
    comprehension_raw = _load_json("comprehension_scores.json")
    reading_paths_raw = _load_json("reading_paths.json")
    knn_users = _load_json("knn_users.json")

    _CACHE = BlindspotData(
        user_summaries=user_summaries,
        concept_pool=concept_pool,
        concept_catalog=_intkey(concept_catalog_raw),
        adoption=_intkey_nested(adoption_raw),
        novelty=_intkey(novelty_raw),
        comprehension=_intkey_nested(comprehension_raw),
        reading_paths=_intkey(reading_paths_raw),
        knn_users=knn_users,
    )
    return _CACHE
