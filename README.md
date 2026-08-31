# SharedWorkerPool

Reusable Slurm-backed worker-pool orchestration for apps that expose their own
HTTP control planes.

SharedWorkerPool owns the generic mechanics:

- a submit-host launcher that polls configured app status endpoints;
- fair shared Slurm demand calculation across apps;
- adaptive worker submission and backoff;
- local dispatch-state snapshots so every worker does not poll every app;
- optional, lease-coordinated persistent warm children for latency-sensitive apps;
- worker retirement when the launcher requires a newer worker version;
- Slurm worker submission wrappers;
- node-local scratch and non-credential cache setup;
- delegation from a generic pool worker into app-specific worker modules.

The runtime helpers also include `resource_admission()` for dense app workers
that decide whether to claim another unit of work. It covers configured
capacity, scratch disk headroom, system and allocation memory headroom, and
host load. Host-wide load averages are compared against
`max(allocated_cpus, visible_cpus)`, not just a small Slurm CPU allocation, so a
healthy 32-core node is not treated as overloaded merely because a shared worker
requested 2 CPUs. App repositories should adapt this shared decision to their
own nouns, such as CodingWorkspace turns or CAIDA jobs, instead of reimplementing
the admission math locally.

Apps that use this package keep their own control-plane and job semantics. A
configured app must provide:

- a bearer-token control URL;
- a status endpoint compatible with one of the configured status modes;
- a Python module that can run one app-specific worker pass when called with
  `--control-url`, `--token-file`, `--worker-id`, `--scratch-root`, and `--once`.

CAIDA Concierge and CodingWorkspace are the first two app profiles, but this
package is intentionally app-agnostic.

## Persistent warm children

An app profile can set `persistent_warm_workers` to a positive limit. While
that app's fresh control-plane status requests warm workers, generic pool jobs
atomically compete for that many slots in `warm_lease_state_path`. A winner
runs the app worker without `--once`, so its imports, runtime initialization,
and app-owned caches remain ready between units of work. All other generic jobs
continue through the normal fair one-shot dispatcher.

The parent pool worker heartbeats its lease while the child runs. Leases older
than `warm_lease_ttl_seconds` are recoverable after a dead Slurm job, and the
lease limit is additionally capped by the control plane's current
`warmPool.requestedWorkers`. A fresh lease owned by an older worker version is
kept until that child exits, preventing config rotation from starting a second
warm child beside an active old one. New claims are refused while an app is
draining or after `warmPool.secondsRemaining` expires. An already-running child
is never killed merely because the warm request expires; the app worker's own
idle timeout lets active work finish and provides the post-turn warm window.
If a login refresh arrives just as that idle timeout fires, the lease is
released and a replacement may briefly start instead of extending the same
process.

The lease state and lock must be on storage shared by the submit host and Slurm
jobs (normally the user's shared home filesystem). Recommended settings are a
5-second heartbeat and a 30-second TTL. Persistent children that are finishing
their idle tail remain accounted for in launcher demand so they cannot hide
capacity needed by another app.

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
