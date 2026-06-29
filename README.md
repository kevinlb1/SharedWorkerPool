# SharedWorkerPool

Reusable Slurm-backed worker-pool orchestration for apps that expose their own
HTTP control planes.

SharedWorkerPool owns the generic mechanics:

- a submit-host launcher that polls configured app status endpoints;
- fair shared Slurm demand calculation across apps;
- adaptive worker submission and backoff;
- local dispatch-state snapshots so every worker does not poll every app;
- worker retirement when the launcher requires a newer worker version;
- Slurm worker submission wrappers;
- node-local scratch and non-credential cache setup;
- delegation from a generic pool worker into app-specific worker modules.

Apps that use this package keep their own control-plane and job semantics. A
configured app must provide:

- a bearer-token control URL;
- a status endpoint compatible with one of the configured status modes;
- a Python module that can run one app-specific worker pass when called with
  `--control-url`, `--token-file`, `--worker-id`, `--scratch-root`, and `--once`.

CAIDA Concierge and CodingWorkspace are the first two app profiles, but this
package is intentionally app-agnostic.

## Install

For a submit-host checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Launcher

```bash
python -m shared_worker_pool launcher --config config/shared_worker_pool.json
```

## Worker

```bash
python -m shared_worker_pool worker \
  --config config/shared_worker_pool.json \
  --worker-id shared-manual-0 \
  --scratch-root /tmp/shared-worker-manual
```

## Slurm Submission

Use `scripts/submit_shared_slurm_workers.sh` from this repository or copy it
into an app deployment checkout. The script submits generic workers that run
`python -m shared_worker_pool worker`; app-specific work happens only after a
generic worker delegates to an app's configured worker module.

