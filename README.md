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

The parent pool worker heartbeats its lease while the child runs. Lease updates
use an atomic directory lock rather than `flock`, because the shared home
filesystem used by Slurm may not provide POSIX advisory locks. Lock acquisition
is bounded by `warm_lease_lock_timeout_seconds`; an abandoned owner is
recoverable after `warm_lease_lock_stale_seconds`, and release verifies both the
owner token and directory inode so an old process cannot unlock a replacement.
Malformed, unreadable, or wrong-version lease state fails closed instead of
being silently replaced. Release and stale recovery also elect one inode-tied
transition owner before renaming the lock directory. A crashed transition is
not automatically superseded because portable NFS operations cannot safely
fence the paused actor; that rare case remains failed closed for an operator to
inspect after confirming no launcher or worker still owns it.

During release and stale recovery, the coordinator defers `SIGTERM` until the
transition finishes, then restores its previous signal mask. This lets the
launcher's scheduled shutdown complete after it detaches the lock directory.
Host failure or a forced kill during a transition still requires operator review.

The verified rename of that exact lock inode to a unique detached quarantine is
the unlock commit point. The first detached name ends in `.pending` and is
never eligible for automatic cleanup. Only after the releaser verifies the
moved inode does it rename the directory to an exact
`.ready` name. This second rename is a garbage-collection readiness handshake,
so a paused releaser cannot race a sweeper before its strict verification
completes.

Removing a ready detached directory is garbage collection, not coordination:
NFS may briefly leave a `.nfs*` file for an open unlinked record or report
`ENOTEMPTY` for an apparently empty directory. Cleanup retries for at most
`warm_lease_cleanup_timeout_seconds` (0.25 seconds by default), then leaves the
quarantine without failing the completed claim, heartbeat, read, or release. A
deferred cleanup also sweeps a small bounded set of exact UUID-named ready
quarantines older than `warm_lease_cleanup_defer_seconds` (1 second by default),
while successful operations avoid an extra shared-directory scan. Cleanup pins
and rechecks the expected directory inode, refuses symlinks and unexpected
entries, never unlinks `.nfs*`, and never examines the canonical lock path. A
sweeper captures that inode identity on its own NFS client; it never compares a
creator's client-local device number across hosts. Ready names are random and
never reused by cooperating processes. Portable NFS has no conditional
`rmdir`-by-inode, so adversarial same-account path replacement during the final
identity-check window is outside this trusted staging topology. Cleanup
diagnostics are best effort so a broken Slurm output stream cannot turn garbage
collection back into a transaction failure.

Release quarantines made by package 0.2.1 do not carry the readiness handshake
and are intentionally ignored by automatic cleanup. After all 0.2.1 actors are
gone, an operator may remove only those legacy directories independently
verified to be empty; an unmarked nonempty directory remains evidence to
inspect rather than time-fenced garbage.

Leases older than `warm_lease_ttl_seconds` are recoverable after a dead Slurm
job, and the lease limit is additionally capped by the control plane's current
`warmPool.requestedWorkers`. A fresh lease owned by an older worker version is
kept until that child exits, preventing config rotation from starting a second
warm child beside an active old one. Fresh leases continue to count as tail
capacity even if a new config disables persistence or removes the app. New
claims require finite, positive `warmPool.secondsRemaining` and are refused
while an app is draining or after the request expires. An already-running child
is never killed merely because the warm request expires; the app worker's own
idle timeout lets active work finish and provides the post-turn warm window.
If a login refresh arrives just as that idle timeout fires, the lease is
released and a replacement may briefly start instead of extending the same
process.

If a child loses its lease, or lease heartbeats repeatedly fail, the parent
sends `SIGUSR1` once and stops renewing. A compatible persistent app worker
must interpret that signal as a graceful drain: stop claiming new work, let
active work finish, and then exit. The shared parent waits and does not send
`SIGTERM` or `SIGKILL` merely because lease coordination failed. The
`warm_lease_max_heartbeat_failures` setting bounds how many consecutive failed
heartbeat intervals occur before that drain request. A failed signal system
call is retried; after the signal is delivered, stopping new claims is bounded
by signal handling but process exit still waits for any active turn. Normal
Slurm/parent teardown continues to use the regular termination path.

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

Pending Slurm array tasks count individually toward the pool limit. Workers
already waiting to start satisfy startup-burst demand, so an unavailable
partition does not cause another burst on every launcher poll.

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
