"""Command-line entry point for SharedWorkerPool."""

from __future__ import annotations

import sys

from shared_worker_pool.pool import main


if __name__ == "__main__":
    sys.exit(main())
