#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. The implementation lives in the Python package so apps
# can share one Slurm submission path.

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${SHARED_WORKER_POOL_PYTHON:-${SOURCE_DIR}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

exec "${python_bin}" -m shared_worker_pool.submit_slurm
