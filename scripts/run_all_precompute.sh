#!/usr/bin/env bash
# Run the full Blindspot pre-compute pipeline end-to-end.
# Outputs land in scripts/_cache/, then are converted to data/ by stage 08.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

# Stage 07 needs LLM judge keys (unless smoke-testing with BLINDSPOT_DRY_RUN=1)
if [[ "${BLINDSPOT_DRY_RUN:-0}" != "1" ]]; then
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY (https://platform.openai.com/api-keys) or export BLINDSPOT_DRY_RUN=1}"
  : "${GEMINI_API_KEY:?Set GEMINI_API_KEY (https://aistudio.google.com/apikey) or export BLINDSPOT_DRY_RUN=1}"
fi

echo "=== Stage 01: fetch users ==="
$PY scripts/precompute_01_fetch_users.py

echo "=== Stage 02: fetch corpus ==="
$PY scripts/precompute_02_fetch_corpus.py

echo "=== Stage 03: extract concepts ==="
$PY scripts/precompute_03_extract_concepts.py

echo "=== Stage 04: build pools ==="
$PY scripts/precompute_04_build_pools.py

echo "=== Stage 05: score adoption ==="
$PY scripts/precompute_05_score_adoption.py

echo "=== Stage 06: build reading paths ==="
$PY scripts/precompute_06_build_paths.py

echo "=== Stage 07: score comprehension ==="
$PY scripts/precompute_07_score_comprehension.py

echo "=== Stage 08: convert _cache/ → data/ ==="
$PY scripts/precompute_08_finalize.py

echo "=== Done. Artifacts in data/ ==="
ls -lh data/
