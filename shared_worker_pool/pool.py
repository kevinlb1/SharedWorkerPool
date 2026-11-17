"""Shared Slurm worker-pool supervisor.

The pool launcher owns one Slurm job namespace and submits generic pool workers.
Each pool worker polls configured app control planes, then delegates one unit of
work to that app's own worker module. This keeps app job semantics separate
while sharing an expensive cluster worker pool.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from shared_worker_pool.runtime import (
    LauncherRequestError,
    active_worker_summary,
    is_transient_control_error,
    json_request,
    launcher_log,
    live_slurm_job_count,
    read_token,
)


POOL_LAUNCHER_VERSION = "shared-worker-pool-20260830-nfs-warm-cleanup"
POOL_WORKER_VERSION = "shared-worker-pool-worker-20260830-nfs-warm-cleanup"
WARM_LEASE_STATE_VERSION = 1
WARM_LOCK_OWNER_VERSION = 1
MAX_WARM_LEASE_STATE_BYTES = 1024 * 1024
WARM_LOCK_CLEANUP_RETRY_SECONDS = 0.02
WARM_LOCK_QUARANTINE_ENTRY_LIMIT = 8
WARM_LOCK_QUARANTINE_SCAN_LIMIT = 64
WARM_LOCK_QUARANTINE_SWEEP_LIMIT = 8
WARM_LOCK_QUARANTINE_SWEEP_SECONDS = 0.05


class WarmLeaseError(RuntimeError):
    """Base class for persistent-warm lease coordination failures."""


class WarmLeaseStateError(WarmLeaseError):
    """Raised when persistent-warm state cannot be trusted."""


class WarmLeaseLockTimeout(WarmLeaseError):
    """Raised when the persistent-warm lock cannot be acquired promptly."""


@dataclass(frozen=True)
class PoolAppProfile:
    name: str
    control_url: str
    token_file: Path
    source_dir: Path
    python: str
    worker_module: str
    status_mode: str
    worker_capacity: int = 1
    worker_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    launcher_status_path: str = ""
    enabled: bool = True
    persistent_warm_workers: int = 0


@dataclass(frozen=True)
class PoolConfig:
    config_path: Path
    apps: tuple[PoolAppProfile, ...]
    launcher_id: str
    partition: str
    job_name: str
    submit_script: Path
    source_dir: Path
    slurm_user: str
    max_jobs: int
    max_submit_per_cycle: int
    min_idle_workers: int
    poll_seconds: float
    launcher_active_poll_seconds: float
    launcher_idle_poll_seconds: float
    launcher_idle_backoff_after_seconds: float
    worker_poll_seconds: float
    worker_idle_timeout_seconds: float
    worker_max_runtime_seconds: float
    worker_task_timeout_seconds: float
    dispatch_state_path: Path
    dispatch_state_ttl_seconds: float
    output_dir: Path
    time_limit: str
    cpus_per_task: int
    memory_mb: int
    auto_pull: bool = False
    adaptive_scaling_enabled: bool = False
    adaptive_start_jobs: int = 0
    adaptive_start_submit_per_cycle: int = 0
    adaptive_min_jobs: int = 0
    adaptive_min_submit_per_cycle: int = 0
    adaptive_step_jobs: int = 0
    adaptive_step_submit_per_cycle: int = 0
    adaptive_recover_cycles: int = 3
    adaptive_slow_status_seconds: float = 3.0
    adaptive_reset_when_idle: bool = True
    app_start_burst_workers: int = 0
    app_start_burst_overcommit: int = 0
    warm_lease_state_path: Path | None = None
    warm_lease_heartbeat_seconds: float = 5.0
    warm_lease_ttl_seconds: float = 30.0
    warm_lease_lock_timeout_seconds: float = 2.0
    warm_lease_lock_stale_seconds: float = 30.0
    warm_lease_max_heartbeat_failures: int = 3
    warm_lease_cleanup_timeout_seconds: float = 0.25
    warm_lease_cleanup_defer_seconds: float = 1.0


@dataclass(frozen=True)
class AppNeed:
    name: str
    queued_units: int
    running_units: int
    worker_capacity: int
    enabled: bool = True
    restart_drain_active: bool = False
    warm_requested_workers: int = 0
    warm_request_expires_at: float = 0.0
    registered_workers: int | None = None
    available_slots: int = 0

    @property
    def needed_workers(self) -> int:
        if not self.enabled or self.restart_drain_active or self.queued_units <= 0:
            return 0
        return math.ceil(max(0, self.queued_units - self.available_slots) / max(1, self.worker_capacity))

    @property
    def active_workers(self) -> int:
        if self.registered_workers is not None:
            return self.registered_workers
        return math.ceil(max(0, self.running_units) / max(1, self.worker_capacity))

    @property
    def target_workers(self) -> int:
        return self.target_workers_at()

    def target_workers_at(self, *, now: float | None = None) -> int:
        if not self.enabled or self.restart_drain_active:
            return 0
        return self.active_workers + max(self.needed_workers, self.effective_warm_requested_workers(now=now))

    def effective_warm_requested_workers(self, *, now: float | None = None) -> int:
        if not self.warm_request_is_active(now=now):
            return 0
        return max(0, self.warm_requested_workers)

    def warm_request_is_active(self, *, now: float | None = None) -> bool:
        if self.warm_requested_workers <= 0:
            return False
        try:
            expires_at = float(self.warm_request_expires_at)
        except (TypeError, ValueError):
            return False
        return math.isfinite(expires_at) and expires_at > (time.time() if now is None else now)


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise ValueError(f"expected string or list of strings, got {type(value).__name__}")


def load_pool_config(path: Path) -> PoolConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pool = raw.get("pool") if isinstance(raw.get("pool"), dict) else {}
    apps: list[PoolAppProfile] = []
    for item in raw.get("apps") or []:
        if not isinstance(item, dict):
            raise ValueError("each app profile must be an object")
        apps.append(
            PoolAppProfile(
                name=str(item["name"]),
                control_url=str(item["control_url"]),
                token_file=_expand_path(item["token_file"]),
                source_dir=_expand_path(item["source_dir"]),
                python=str(item.get("python") or "python3"),
                worker_module=str(item["worker_module"]),
                status_mode=str(item["status_mode"]),
                worker_capacity=max(1, int(item.get("worker_capacity") or 1)),
                worker_args=_string_tuple(item.get("worker_args")),
                env={str(key): str(value) for key, value in (item.get("env") or {}).items()},
                launcher_status_path=str(item.get("launcher_status_path") or ""),
                enabled=bool(item.get("enabled", True)),
                persistent_warm_workers=max(0, int(item.get("persistent_warm_workers") or 0)),
            )
        )
    if not apps:
        raise ValueError("shared worker pool config must contain at least one app profile")
    max_jobs = max(0, int(pool.get("max_jobs") or 400))
    max_submit_per_cycle = max(0, int(pool.get("max_submit_per_cycle") or 100))
    warm_lease_heartbeat_seconds = max(1.0, float(pool.get("warm_lease_heartbeat_seconds") or 5.0))
    warm_lease_ttl_seconds = max(
        warm_lease_heartbeat_seconds * 3.0,
        float(pool.get("warm_lease_ttl_seconds") or 30.0),
    )
    warm_lease_lock_timeout_seconds = float(pool.get("warm_lease_lock_timeout_seconds") or 2.0)
    warm_lease_lock_stale_seconds = float(pool.get("warm_lease_lock_stale_seconds") or 30.0)
    warm_lease_cleanup_timeout_seconds = float(pool.get("warm_lease_cleanup_timeout_seconds", 0.25))
    warm_lease_cleanup_defer_seconds = float(pool.get("warm_lease_cleanup_defer_seconds", 1.0))
    if not math.isfinite(warm_lease_lock_timeout_seconds) or warm_lease_lock_timeout_seconds <= 0:
        raise ValueError("warm_lease_lock_timeout_seconds must be finite and positive")
    if not math.isfinite(warm_lease_lock_stale_seconds) or warm_lease_lock_stale_seconds <= 0:
        raise ValueError("warm_lease_lock_stale_seconds must be finite and positive")
    if not math.isfinite(warm_lease_cleanup_timeout_seconds) or warm_lease_cleanup_timeout_seconds < 0:
        raise ValueError("warm_lease_cleanup_timeout_seconds must be finite and nonnegative")
    if not math.isfinite(warm_lease_cleanup_defer_seconds) or warm_lease_cleanup_defer_seconds < 0:
        raise ValueError("warm_lease_cleanup_defer_seconds must be finite and nonnegative")
    warm_lease_lock_stale_seconds = max(
        warm_lease_lock_stale_seconds,
        warm_lease_lock_timeout_seconds * 3.0,
    )
    return PoolConfig(
        config_path=path.resolve(),
        apps=tuple(apps),
        launcher_id=str(pool.get("launcher_id") or f"shared-launcher-{socket.gethostname()}"),
        partition=str(pool.get("partition") or "ada_cpu_long"),
        job_name=str(pool.get("job_name") or "shared-worker-auto"),
        submit_script=_expand_path(pool.get("submit_script") or "scripts/submit_shared_slurm_workers.sh"),
        source_dir=_expand_path(pool.get("source_dir") or Path.cwd()),
        slurm_user=str(pool.get("slurm_user") or os.environ.get("USER") or ""),
        max_jobs=max_jobs,
        max_submit_per_cycle=max_submit_per_cycle,
        min_idle_workers=max(0, int(pool.get("min_idle_workers") or 0)),
        poll_seconds=max(1.0, float(pool.get("poll_seconds") or 10.0)),
        launcher_active_poll_seconds=max(
            1.0,
            float(pool.get("launcher_active_poll_seconds") or pool.get("active_poll_seconds") or pool.get("poll_seconds") or 10.0),
        ),
        launcher_idle_poll_seconds=max(
            1.0,
            float(pool.get("launcher_idle_poll_seconds") or pool.get("idle_poll_seconds") or pool.get("poll_seconds") or 10.0),
        ),
        launcher_idle_backoff_after_seconds=max(
            0.0,
            float(pool.get("launcher_idle_backoff_after_seconds") or pool.get("idle_backoff_after_seconds") or 120.0),
        ),
        worker_poll_seconds=max(1.0, float(pool.get("worker_poll_seconds") or 3.0)),
        worker_idle_timeout_seconds=max(0.0, float(pool.get("worker_idle_timeout_seconds") or 300.0)),
        worker_max_runtime_seconds=max(0.0, float(pool.get("worker_max_runtime_seconds") or 0.0)),
        worker_task_timeout_seconds=max(0.0, float(pool.get("worker_task_timeout_seconds") or 0.0)),
        dispatch_state_path=_expand_path(pool.get("dispatch_state_path") or "~/shared-worker-pool/dispatch_state.json"),
        dispatch_state_ttl_seconds=max(1.0, float(pool.get("dispatch_state_ttl_seconds") or 30.0)),
        output_dir=_expand_path(pool.get("output_dir") or "~/shared-worker-pool/logs"),
        time_limit=str(pool.get("time_limit") or "12:00:00"),
        cpus_per_task=max(1, int(pool.get("cpus_per_task") or 2)),
        memory_mb=max(0, int(pool.get("memory_mb") or 0)),
        auto_pull=bool(pool.get("auto_pull", False)),
        adaptive_scaling_enabled=bool(pool.get("adaptive_scaling_enabled", False)),
        adaptive_start_jobs=max(0, int(pool.get("adaptive_start_jobs") or max_jobs)),
        adaptive_start_submit_per_cycle=max(
            0,
            int(pool.get("adaptive_start_submit_per_cycle") or max_submit_per_cycle),
        ),
        adaptive_min_jobs=max(0, int(pool.get("adaptive_min_jobs") or 0)),
        adaptive_min_submit_per_cycle=max(0, int(pool.get("adaptive_min_submit_per_cycle") or 0)),
        adaptive_step_jobs=max(1, int(pool.get("adaptive_step_jobs") or 16)),
        adaptive_step_submit_per_cycle=max(1, int(pool.get("adaptive_step_submit_per_cycle") or 8)),
        adaptive_recover_cycles=max(1, int(pool.get("adaptive_recover_cycles") or 3)),
        adaptive_slow_status_seconds=max(0.25, float(pool.get("adaptive_slow_status_seconds") or 3.0)),
        adaptive_reset_when_idle=bool(pool.get("adaptive_reset_when_idle", True)),
        app_start_burst_workers=max(0, int(pool.get("app_start_burst_workers") or 0)),
        app_start_burst_overcommit=max(0, int(pool.get("app_start_burst_overcommit") or 0)),
        warm_lease_state_path=_expand_path(
            pool.get("warm_lease_state_path")
            or Path(pool.get("dispatch_state_path") or "~/shared-worker-pool/dispatch_state.json").with_name(
                "warm_leases.json"
            )
        ),
        warm_lease_heartbeat_seconds=warm_lease_heartbeat_seconds,
        warm_lease_ttl_seconds=warm_lease_ttl_seconds,
        warm_lease_lock_timeout_seconds=warm_lease_lock_timeout_seconds,
        warm_lease_lock_stale_seconds=warm_lease_lock_stale_seconds,
        warm_lease_max_heartbeat_failures=max(
            1,
            int(pool.get("warm_lease_max_heartbeat_failures") or 3),
        ),
        warm_lease_cleanup_timeout_seconds=warm_lease_cleanup_timeout_seconds,
        warm_lease_cleanup_defer_seconds=warm_lease_cleanup_defer_seconds,
    )


@dataclass
class PoolScaleState:
    config: PoolConfig
    effective_max_jobs: int = field(init=False)
    effective_max_submit_per_cycle: int = field(init=False)
    healthy_cycles: int = 0
    last_control_plane_seconds: float = 0.0
    last_timing_breakdown: dict[str, float] = field(default_factory=dict)
    last_reason: str = "fixed"

    def __post_init__(self) -> None:
        if self.config.adaptive_scaling_enabled:
            self.effective_max_jobs = self._clamp_jobs(self.config.adaptive_start_jobs)
            self.effective_max_submit_per_cycle = self._clamp_submit(self.config.adaptive_start_submit_per_cycle)
            self.last_reason = "adaptive start"
        else:
            self.effective_max_jobs = self.config.max_jobs
            self.effective_max_submit_per_cycle = self.config.max_submit_per_cycle

    def _clamp_jobs(self, value: int) -> int:
        if self.config.max_jobs <= 0:
            return 0
        lower = min(self.config.max_jobs, max(0, self.config.adaptive_min_jobs))
        return max(lower, min(self.config.max_jobs, max(0, int(value or 0))))

    def _clamp_submit(self, value: int) -> int:
        if self.config.max_submit_per_cycle <= 0:
            return 0
        lower = min(self.config.max_submit_per_cycle, max(0, self.config.adaptive_min_submit_per_cycle))
        return max(lower, min(self.config.max_submit_per_cycle, max(0, int(value or 0))))

    def record_poll(
        self,
        *,
        active: bool,
        control_plane_seconds: float,
        transient_failure: bool = False,
        timings: dict[str, float] | None = None,
    ) -> None:
        self.last_control_plane_seconds = max(0.0, float(control_plane_seconds or 0.0))
        self.last_timing_breakdown = {
            str(key): round(max(0.0, float(value or 0.0)), 3)
            for key, value in (timings or {}).items()
        }
        if not self.config.adaptive_scaling_enabled:
            self.effective_max_jobs = self.config.max_jobs
            self.effective_max_submit_per_cycle = self.config.max_submit_per_cycle
            self.last_reason = "fixed"
            return
        if not active and self.config.adaptive_reset_when_idle:
            self.healthy_cycles = 0
            self.effective_max_jobs = self._clamp_jobs(self.config.adaptive_start_jobs)
            self.effective_max_submit_per_cycle = self._clamp_submit(self.config.adaptive_start_submit_per_cycle)
            self.last_reason = "idle reset"
            return
        if transient_failure or self.last_control_plane_seconds >= self.config.adaptive_slow_status_seconds:
            self.healthy_cycles = 0
            self.effective_max_jobs = self._clamp_jobs(max(self.config.adaptive_min_jobs, self.effective_max_jobs // 2))
            self.effective_max_submit_per_cycle = self._clamp_submit(
                max(self.config.adaptive_min_submit_per_cycle, self.effective_max_submit_per_cycle // 2)
            )
            self.last_reason = "backoff"
            return
        self.healthy_cycles += 1
        self.last_reason = f"healthy {self.healthy_cycles}/{self.config.adaptive_recover_cycles}"
        if self.healthy_cycles >= self.config.adaptive_recover_cycles:
            self.healthy_cycles = 0
            self.effective_max_jobs = self._clamp_jobs(self.effective_max_jobs + self.config.adaptive_step_jobs)
            self.effective_max_submit_per_cycle = self._clamp_submit(
                self.effective_max_submit_per_cycle + self.config.adaptive_step_submit_per_cycle
            )
            self.last_reason = "ramp up"

    def payload(self) -> dict[str, Any]:
        return {
            "enabled": self.config.adaptive_scaling_enabled,
            "effective_max_jobs": self.effective_max_jobs,
            "effective_max_submit_per_cycle": self.effective_max_submit_per_cycle,
            "hard_max_jobs": self.config.max_jobs,
            "hard_max_submit_per_cycle": self.config.max_submit_per_cycle,
            "control_plane_seconds": round(self.last_control_plane_seconds, 3),
            "timings": self.last_timing_breakdown,
            "reason": self.last_reason,
        }


def fetch_app_status(app: PoolAppProfile, *, launcher_id: str) -> dict[str, Any]:
    token = read_token(app.token_file)
    return json_request(
        app.control_url,
        token,
        "/api/worker/cluster/status",
        {
            "launcherId": launcher_id,
            "hostname": socket.gethostname(),
            "version": POOL_LAUNCHER_VERSION,
        },
        timeout_seconds=30,
        error_cls=LauncherRequestError,
    )


def app_need_from_status(app: PoolAppProfile, status: dict[str, Any]) -> AppNeed:
    mode = app.status_mode
    warm_pool = status.get("warmPool") if isinstance(status.get("warmPool"), dict) else {}
    try:
        warm_requested_workers = max(0, int(warm_pool.get("requestedWorkers") or 0))
    except (TypeError, ValueError, OverflowError):
        warm_requested_workers = 0
    if app.persistent_warm_workers:
        warm_requested_workers = min(warm_requested_workers, app.persistent_warm_workers)
    try:
        warm_seconds_remaining = float(warm_pool.get("secondsRemaining") or 0.0)
    except (TypeError, ValueError):
        warm_seconds_remaining = 0.0
    if not math.isfinite(warm_seconds_remaining) or warm_seconds_remaining <= 0:
        warm_requested_workers = 0
        warm_seconds_remaining = 0.0
    warm_request_expires_at = time.time() + warm_seconds_remaining if warm_requested_workers else 0.0
    if mode == "caida":
        jobs = status.get("jobs") if isinstance(status.get("jobs"), dict) else {}
        enabled = bool(status.get("clusterWorkersEnabled") or status.get("remoteWorkersEnabled"))
        workers = status.get("workers")
        shared_workers = []
        for worker in workers if isinstance(workers, list) else []:
            if not isinstance(worker, dict) or not str(worker.get("id", "")).startswith("shared-"):
                continue
            capacity = min(int(worker.get("capacity") or 0), int(status.get("claimLimit") or app.worker_capacity))
            if str(worker.get("message") or "").startswith("resource backpressure:"):
                capacity = min(capacity, int(worker.get("activeJobs") or 0))
            shared_workers.append({**worker, "capacity": capacity})
        observed = active_worker_summary({**status, "workers": shared_workers}, min_capacity=0)
        registered = observed["activeWorkers"]
        return AppNeed(
            name=app.name,
            queued_units=max(0, int(jobs.get("queuedUnclaimed") or 0)),
            running_units=max(0, int(jobs.get("runningTotal") or 0)),
            worker_capacity=max(1, min(app.worker_capacity, observed["capacity"] // registered))
            if registered else app.worker_capacity,
            registered_workers=registered or None,
            available_slots=observed["availableCapacity"],
            enabled=app.enabled and enabled,
            restart_drain_active=bool(status.get("restartDrainActive")),
            warm_requested_workers=warm_requested_workers,
            warm_request_expires_at=warm_request_expires_at,
        )
    if mode == "codingworkspace":
        turns = status.get("turns") if isinstance(status.get("turns"), dict) else {}
        return AppNeed(
            name=app.name,
            queued_units=max(0, int(turns.get("queuedUnclaimed") or 0)),
            running_units=max(0, int(turns.get("runningTotal") or turns.get("runningRemote") or 0)),
            worker_capacity=app.worker_capacity,
            enabled=app.enabled and bool(status.get("remoteWorkersEnabled")),
            restart_drain_active=bool(status.get("restartDrainActive")),
            warm_requested_workers=warm_requested_workers,
            warm_request_expires_at=warm_request_expires_at,
        )
    raise ValueError(f"unknown shared worker app status_mode: {mode}")


def desired_pool_submissions(
    needs: list[AppNeed],
    *,
    live_slurm_jobs: int,
    min_idle_workers: int,
    max_jobs: int,
    max_submit_per_cycle: int,
    app_start_burst_workers: int = 0,
    app_start_burst_overcommit: int = 0,
    persistent_warm_tail_workers: int = 0,
    now: float | None = None,
) -> int:
    checked_at = time.time() if now is None else float(now)
    active_needs = [need for need in needs if need.enabled and not need.restart_drain_active]
    needed_workers = sum(need.target_workers_at(now=checked_at) for need in active_needs)
    needed_workers += max(0, int(persistent_warm_tail_workers or 0))
    if any(need.queued_units or need.running_units for need in active_needs):
        needed_workers += max(0, int(min_idle_workers or 0))
    live_jobs = max(0, int(live_slurm_jobs or 0))
    submit_cap = max(0, int(max_submit_per_cycle or 0))
    missing_workers = max(0, needed_workers - live_jobs)
    normal_room = max(0, int(max_jobs or 0) - live_jobs)
    normal_submit = max(0, min(missing_workers, normal_room, submit_cap))

    startup_burst = max(0, int(app_start_burst_workers or 0))
    burst_overcommit = max(0, int(app_start_burst_overcommit or 0))
    if startup_burst <= 0 or burst_overcommit <= 0 or submit_cap <= 0:
        return normal_submit
    startup_needed = sum(
        min(need.needed_workers, startup_burst)
        for need in active_needs
        if need.queued_units > 0 and need.active_workers == 0
    )
    # Jobs waiting in Slurm or still registering can serve the new app. Do not
    # submit another startup burst every poll while those workers are pending.
    registered_workers = sum(max(0, need.active_workers) for need in needs)
    starting_workers = max(0, live_jobs - registered_workers)
    startup_needed = max(0, startup_needed - starting_workers)
    if startup_needed <= 0:
        return normal_submit
    burst_room = max(0, int(max_jobs or 0) + burst_overcommit - live_jobs)
    startup_submit = max(0, min(startup_needed, burst_room, submit_cap))
    return max(normal_submit, startup_submit)


def app_need_to_payload(need: AppNeed) -> dict[str, Any]:
    return {
        "name": need.name,
        "queued_units": need.queued_units,
        "running_units": need.running_units,
        "worker_capacity": need.worker_capacity,
        "enabled": need.enabled,
        "restart_drain_active": need.restart_drain_active,
        "warm_requested_workers": need.warm_requested_workers,
        "warm_request_expires_at": need.warm_request_expires_at,
        "registered_workers": need.registered_workers,
        "available_slots": need.available_slots,
        "needed_workers": need.needed_workers,
        "active_workers": need.active_workers,
        "target_workers": need.target_workers,
    }


def app_need_from_payload(payload: dict[str, Any]) -> AppNeed:
    try:
        warm_request_expires_at = float(payload.get("warm_request_expires_at") or 0.0)
    except (TypeError, ValueError):
        warm_request_expires_at = 0.0
    if not math.isfinite(warm_request_expires_at) or warm_request_expires_at <= 0:
        warm_request_expires_at = 0.0
    return AppNeed(
        name=str(payload.get("name") or ""),
        queued_units=max(0, int(payload.get("queued_units") or 0)),
        running_units=max(0, int(payload.get("running_units") or 0)),
        worker_capacity=max(1, int(payload.get("worker_capacity") or 1)),
        enabled=bool(payload.get("enabled", True)),
        restart_drain_active=bool(payload.get("restart_drain_active")),
        warm_requested_workers=max(0, int(payload.get("warm_requested_workers") or 0)),
        warm_request_expires_at=warm_request_expires_at,
        registered_workers=payload.get("registered_workers"),
        available_slots=max(0, int(payload.get("available_slots") or 0)),
    )


def dispatch_state_payload(
    config: PoolConfig,
    *,
    needs: list[AppNeed],
    live_slurm_jobs: int,
    desired_submissions: int,
    submitted_jobs: int,
    status: str,
    message: str = "",
    scale: PoolScaleState | None = None,
) -> dict[str, Any]:
    now = time.time()
    worker_version = pool_worker_config_version(config)
    scale_payload = scale.payload() if scale else {
        "enabled": False,
        "effective_max_jobs": config.max_jobs,
        "effective_max_submit_per_cycle": config.max_submit_per_cycle,
        "hard_max_jobs": config.max_jobs,
        "hard_max_submit_per_cycle": config.max_submit_per_cycle,
        "control_plane_seconds": 0.0,
        "reason": "fixed",
    }
    return {
        "version": POOL_LAUNCHER_VERSION,
        "launcher_version": POOL_LAUNCHER_VERSION,
        "worker_version": worker_version,
        "required_worker_version": worker_version,
        "launcher_id": config.launcher_id,
        "hostname": socket.gethostname(),
        "updated_at": now,
        "expires_at": now + config.dispatch_state_ttl_seconds,
        "ttl_seconds": config.dispatch_state_ttl_seconds,
        "status": status,
        "message": message,
        "live_slurm_jobs": max(0, int(live_slurm_jobs or 0)),
        "max_jobs": max(0, int(config.max_jobs or 0)),
        "max_submit_per_cycle": max(0, int(config.max_submit_per_cycle or 0)),
        "adaptive_scale": scale_payload,
        "desired_submissions": max(0, int(desired_submissions or 0)),
        "submitted_jobs": max(0, int(submitted_jobs or 0)),
        "apps": [app_need_to_payload(need) for need in needs],
    }


def write_dispatch_state(
    config: PoolConfig,
    *,
    needs: list[AppNeed],
    live_slurm_jobs: int,
    desired_submissions: int,
    submitted_jobs: int,
    status: str,
    message: str = "",
    scale: PoolScaleState | None = None,
) -> None:
    payload = dispatch_state_payload(
        config,
        needs=needs,
        live_slurm_jobs=live_slurm_jobs,
        desired_submissions=desired_submissions,
        submitted_jobs=submitted_jobs,
        status=status,
        message=message,
        scale=scale,
    )
    config.dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config.dispatch_state_path.with_name(f".{config.dispatch_state_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, config.dispatch_state_path)


def read_dispatch_state_payload(config: PoolConfig, *, allow_stale: bool = False) -> dict[str, Any] | None:
    try:
        raw = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        expires_at = float(raw.get("expires_at") or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(expires_at):
        return None
    if not allow_stale and expires_at < time.time():
        return None
    return raw


def read_dispatch_state_needs(config: PoolConfig, *, allow_stale: bool = False) -> list[AppNeed] | None:
    raw = read_dispatch_state_payload(config, allow_stale=allow_stale)
    if raw is None:
        return None
    apps = raw.get("apps")
    if not isinstance(apps, list):
        return None
    needs: list[AppNeed] = []
    for item in apps:
        if isinstance(item, dict):
            try:
                need = app_need_from_payload(item)
            except (TypeError, ValueError, OverflowError):
                return None
            if need.name:
                needs.append(need)
    return needs


def dispatch_required_worker_version(config: PoolConfig, *, allow_stale: bool = False) -> str | None:
    raw = read_dispatch_state_payload(config, allow_stale=allow_stale)
    if raw is None:
        return None
    value = raw.get("required_worker_version") or raw.get("worker_version")
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def warm_lease_state_path(config: PoolConfig) -> Path:
    if config.warm_lease_state_path is not None:
        return config.warm_lease_state_path
    return config.dispatch_state_path.with_name("warm_leases.json")


def warm_lease_lock_path(config: PoolConfig) -> Path:
    path = warm_lease_state_path(config)
    # A directory created with mkdir(2) is an atomic exclusion primitive on NFS.
    # Keep this distinct from the legacy flock file so a rolling upgrade cannot
    # mistake the old regular file for a corrupt directory lock.
    return path.with_suffix(path.suffix + ".lock.d")


@dataclass(frozen=True)
class _WarmLeaseLockOwnership:
    config: PoolConfig
    path: Path
    token: str
    device: int
    inode: int
    directory_fd: int


def _warm_state_error(path: Path, message: str, exc: BaseException | None = None) -> WarmLeaseStateError:
    error = WarmLeaseStateError(f"persistent warm lease state {path}: {message}")
    if exc is not None:
        error.__cause__ = exc
    return error


def _read_warm_leases_unlocked(config: PoolConfig) -> list[dict[str, Any]]:
    path = warm_lease_state_path(config)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise _warm_state_error(path, f"cannot open it ({exc})", exc)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _warm_state_error(path, "is not a regular file")
        if metadata.st_size > MAX_WARM_LEASE_STATE_BYTES:
            raise _warm_state_error(path, f"exceeds {MAX_WARM_LEASE_STATE_BYTES} bytes")
        chunks: list[bytes] = []
        remaining = MAX_WARM_LEASE_STATE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_WARM_LEASE_STATE_BYTES:
            raise _warm_state_error(path, f"exceeds {MAX_WARM_LEASE_STATE_BYTES} bytes")
    except OSError as exc:
        raise _warm_state_error(path, f"cannot read it ({exc})", exc)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _warm_state_error(path, "contains malformed JSON", exc)
    if not isinstance(payload, dict):
        raise _warm_state_error(path, "must contain a JSON object")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != WARM_LEASE_STATE_VERSION:
        raise _warm_state_error(
            path,
            f"has unsupported version {version!r}; expected {WARM_LEASE_STATE_VERSION}",
        )
    updated_at = payload.get("updated_at")
    if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)) or not math.isfinite(float(updated_at)):
        raise _warm_state_error(path, "has an invalid updated_at value")
    leases = payload.get("leases")
    if not isinstance(leases, list):
        raise _warm_state_error(path, "has a non-list leases value")
    normalized: list[dict[str, Any]] = []
    seen_workers: set[str] = set()
    for index, item in enumerate(leases):
        if not isinstance(item, dict):
            raise _warm_state_error(path, f"lease {index} is not an object")
        app_name = item.get("app")
        worker_id = item.get("worker_id")
        if not isinstance(app_name, str) or not app_name.strip():
            raise _warm_state_error(path, f"lease {index} has an invalid app")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise _warm_state_error(path, f"lease {index} has an invalid worker_id")
        app_name = app_name.strip()
        worker_id = worker_id.strip()
        if worker_id in seen_workers:
            raise _warm_state_error(path, f"contains duplicate worker_id {worker_id!r}")
        heartbeat_at = item.get("heartbeat_at")
        if (
            isinstance(heartbeat_at, bool)
            or not isinstance(heartbeat_at, (int, float))
            or not math.isfinite(float(heartbeat_at))
        ):
            raise _warm_state_error(path, f"lease {index} has an invalid heartbeat_at")
        copied = dict(item)
        copied["app"] = app_name
        copied["worker_id"] = worker_id
        copied["heartbeat_at"] = float(heartbeat_at)
        for field_name in ("claimed_at", "request_expires_at"):
            if field_name not in copied:
                continue
            value = copied[field_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise _warm_state_error(path, f"lease {index} has an invalid {field_name}")
            copied[field_name] = float(value)
        normalized.append(copied)
        seen_workers.add(worker_id)
    return normalized


def _current_warm_leases_unlocked(
    config: PoolConfig,
    *,
    now: float,
) -> list[dict[str, Any]]:
    current: list[dict[str, Any]] = []
    for lease in _read_warm_leases_unlocked(config):
        heartbeat_at = float(lease["heartbeat_at"])
        if heartbeat_at < now - config.warm_lease_ttl_seconds:
            continue
        if heartbeat_at > now + config.warm_lease_ttl_seconds:
            continue
        current.append(lease)
    return current


def _write_warm_leases_unlocked(
    config: PoolConfig,
    leases: list[dict[str, Any]],
    *,
    now: float,
) -> None:
    path = warm_lease_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise _warm_state_error(path, f"cannot inspect it before writing ({exc})", exc)
    else:
        if not stat.S_ISREG(existing.st_mode):
            raise _warm_state_error(path, "refuses to replace a non-regular file")
    encoded = json.dumps(
        {
            "version": WARM_LEASE_STATE_VERSION,
            "updated_at": now,
            "leases": leases,
        },
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_WARM_LEASE_STATE_BYTES:
        raise _warm_state_error(path, f"would exceed {MAX_WARM_LEASE_STATE_BYTES} bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    except OSError as exc:
        raise _warm_state_error(path, f"cannot update it ({exc})", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short write while updating persistent warm coordination state")
        offset += written


def _read_warm_lock_owner_fd(directory_fd: int, lock_path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    owner_path = lock_path / "owner.json"
    try:
        descriptor = os.open("owner.json", flags, dir_fd=directory_fd)
    except OSError as exc:
        raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} is unreadable or malformed") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
            raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} is not a valid regular file")
        data = os.read(descriptor, 16385)
        if len(data) > 16384:
            raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} is too large")
    except OSError as exc:
        raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} is unreadable or malformed") from exc
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} is unreadable or malformed") from exc
    if (
        not isinstance(payload, dict)
        or isinstance(payload.get("version"), bool)
        or payload.get("version") != WARM_LOCK_OWNER_VERSION
    ):
        raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} has an unsupported version")
    token = payload.get("token")
    created_at = payload.get("created_at")
    if not isinstance(token, str) or not token:
        raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} has an invalid token")
    if (
        isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not math.isfinite(float(created_at))
    ):
        raise WarmLeaseStateError(f"persistent warm lock owner {owner_path} has an invalid timestamp")
    return payload


def _write_warm_lock_record_fd(directory_fd: int, name: str, payload: dict[str, Any]) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        _write_all(descriptor, json.dumps(payload, sort_keys=True).encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _warm_transition_index(name: str) -> int | None:
    prefix = "transition."
    suffix = ".json"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    value = name[len(prefix) : -len(suffix)]
    if not value.isdigit():
        raise WarmLeaseStateError(f"persistent warm lock has malformed transition record {name!r}")
    return int(value)


def _claim_warm_lock_transition(
    config: PoolConfig,
    directory_fd: int,
    lock_path: Path,
    *,
    action: str,
    now: float,
) -> str | None:
    """Elect one release/recovery actor inside the observed lock inode.

    The transition record pins the directory so its winner can rename exactly
    that inode. We deliberately never supersede a transition automatically: a
    paused older actor cannot be fenced from a newer actor with portable NFS
    operations, and unsafe takeover could rename a replacement lock (ABA). A
    crashed transition therefore fails closed and requires operator cleanup.
    """

    for _attempt in range(8):
        try:
            names = os.listdir(directory_fd)
        except OSError as exc:
            raise WarmLeaseStateError(f"cannot inspect persistent warm lock {lock_path}: {exc}") from exc
        transitions: list[tuple[int, str]] = []
        for name in names:
            index = _warm_transition_index(name)
            if index is not None:
                transitions.append((index, name))
        if transitions:
            for _index, transition_name in transitions:
                try:
                    metadata = os.stat(transition_name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise WarmLeaseStateError(
                        f"cannot inspect persistent warm lock transition {lock_path / transition_name}: {exc}"
                    ) from exc
                if not stat.S_ISREG(metadata.st_mode):
                    raise WarmLeaseStateError(
                        f"persistent warm lock transition {lock_path / transition_name} is not a regular file"
                    )
            return None
        name = "transition.00000000000000000000.json"
        try:
            _write_warm_lock_record_fd(
                directory_fd,
                name,
                {
                    "version": WARM_LOCK_OWNER_VERSION,
                    "token": uuid.uuid4().hex,
                    "action": action,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_at": now,
                },
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise WarmLeaseStateError(f"cannot claim persistent warm lock transition {lock_path}: {exc}") from exc
        return name
    raise WarmLeaseStateError(f"persistent warm lock {lock_path} changed too often during transition election")


def _expected_warm_lock_cleanup_entry(name: str) -> bool:
    if name == "owner.json":
        return True
    try:
        return _warm_transition_index(name) is not None
    except WarmLeaseStateError:
        return False


def _warm_coordination_diagnostic(message: str) -> None:
    """Report noncritical coordination cleanup without creating a failure path."""

    try:
        launcher_log(message)
    except Exception:
        # stdout may itself be an unavailable Slurm/NFS-backed stream. Once a
        # lock inode is detached, even diagnostics must not undo the logical
        # success of the completed coordination transaction.
        pass


def _warm_lock_ready_quarantine(
    lock_path: Path,
    name: str,
) -> str | None:
    actions = {
        f".{lock_path.name}.released.": "release",
        f".{lock_path.name}.stale.": "recover",
    }
    for prefix, action in actions.items():
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        parts = suffix.split(".")
        if len(parts) != 2 or parts[1] != "ready":
            return None
        token, _ready = parts
        if len(token) != 32 or not all(character in "0123456789abcdef" for character in token):
            return None
        return action
    return None


def _ready_warm_lock_quarantine_path(
    lock_path: Path,
    *,
    action: str,
    token: str,
) -> Path:
    label = "released" if action == "release" else "stale"
    return lock_path.with_name(f".{lock_path.name}.{label}.{token}.ready")


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        return (-1, -1)
    if not stat.S_ISDIR(metadata.st_mode):
        return (-1, -1)
    return (metadata.st_dev, metadata.st_ino)


def _remove_recovered_warm_lock(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
    timeout_seconds: float = 0.0,
    log_deferred: bool = True,
) -> bool:
    """Best-effort cleanup after the canonical lock was safely detached.

    NFS can transiently return ENOTEMPTY for an apparently empty directory, or
    create a temporary `.nfs*` file when an unlinked record still has a remote
    open reference. Neither condition affects mutual exclusion once the lock
    inode has been renamed away from the canonical path. Retry for a bounded
    interval, then leave the quarantine for a later sweep without converting a
    successful lease transaction into a coordination failure.
    """

    if not path.name.endswith(".ready"):
        if log_deferred:
            _warm_coordination_diagnostic(
                f"persistent warm lock quarantine {path} has no GC-readiness handshake; "
                "leaving it untouched"
            )
        return False

    if expected_identity is None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return True
        except OSError as exc:
            if log_deferred:
                _warm_coordination_diagnostic(
                    f"deferred persistent warm lock cleanup for {path}: "
                    f"cannot inspect directory: {type(exc).__name__}: {exc}"
                )
            return False
        if not stat.S_ISDIR(metadata.st_mode):
            if log_deferred:
                _warm_coordination_diagnostic(
                    f"persistent warm lock quarantine {path} is no longer the expected directory; "
                    "leaving it untouched"
                )
            return False
        expected_identity = (metadata.st_dev, metadata.st_ino)

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    last_problem = "NFS directory state did not settle"
    while True:
        retry = False
        current_identity = _path_identity(path)
        if current_identity is None:
            return True
        if current_identity != expected_identity:
            if log_deferred:
                _warm_coordination_diagnostic(
                    f"persistent warm lock quarantine {path} changed identity; leaving its replacement untouched"
                )
            return False

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd: int | None = None
        try:
            directory_fd = os.open(path, flags)
            opened = os.fstat(directory_fd)
        except FileNotFoundError:
            return True
        except OSError as exc:
            last_problem = f"cannot open directory: {type(exc).__name__}: {exc}"
            retry = True
        else:
            if (opened.st_dev, opened.st_ino) != expected_identity:
                os.close(directory_fd)
                if log_deferred:
                    _warm_coordination_diagnostic(
                        f"persistent warm lock quarantine {path} changed while opening; "
                        "leaving its replacement untouched"
                    )
                return False
            entries: list[str] = []
            too_many_entries = False
            try:
                with os.scandir(directory_fd) as iterator:
                    for entry in iterator:
                        if len(entries) >= WARM_LOCK_QUARANTINE_ENTRY_LIMIT:
                            too_many_entries = True
                            break
                        entries.append(entry.name)
            except OSError as exc:
                last_problem = f"cannot list directory: {type(exc).__name__}: {exc}"
                retry = True
                entries = []
            if too_many_entries:
                os.close(directory_fd)
                if log_deferred:
                    _warm_coordination_diagnostic(
                        f"persistent warm lock quarantine {path} has more than "
                        f"{WARM_LOCK_QUARANTINE_ENTRY_LIMIT} entries; leaving it untouched"
                    )
                return False

            removable_entries: list[str] = []
            unexpected_entry: str | None = None
            for name in entries:
                try:
                    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    last_problem = f"cannot inspect {name}: {type(exc).__name__}: {exc}"
                    retry = True
                    continue
                if name.startswith(".nfs") and stat.S_ISREG(metadata.st_mode):
                    last_problem = f"waiting for transient NFS entry {name}"
                    retry = True
                    continue
                if not _expected_warm_lock_cleanup_entry(name) or not stat.S_ISREG(metadata.st_mode):
                    unexpected_entry = name
                    break
                removable_entries.append(name)
            if unexpected_entry is not None:
                os.close(directory_fd)
                if log_deferred:
                    _warm_coordination_diagnostic(
                        f"persistent warm lock quarantine {path} has unexpected entry "
                        f"{unexpected_entry!r}; leaving it untouched"
                    )
                return False

            for name in removable_entries:
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    last_problem = f"cannot unlink {name}: {type(exc).__name__}: {exc}"
                    retry = True

        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError as exc:
                last_problem = f"cannot close directory: {type(exc).__name__}: {exc}"
                retry = True

        if not retry:
            # rmdir is path-based, so revalidate the detached name immediately
            # before using it. All content access above is pinned to the opened
            # inode and cannot follow a swapped symlink or touch a replacement.
            current_identity = _path_identity(path)
            if current_identity is None:
                return True
            if current_identity != expected_identity:
                if log_deferred:
                    _warm_coordination_diagnostic(
                        f"persistent warm lock quarantine {path} changed before removal; "
                        "leaving its replacement untouched"
                    )
                return False
            try:
                path.rmdir()
            except FileNotFoundError:
                return True
            except OSError as exc:
                # In particular, NFS may report ENOTEMPTY even when the listing
                # above was empty. A subsequent pass or later acquisition can
                # remove the already-detached quarantine safely.
                last_problem = f"cannot remove directory: {type(exc).__name__}: {exc}"
            else:
                return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if log_deferred:
                _warm_coordination_diagnostic(
                    f"deferred persistent warm lock cleanup for {path}: {last_problem}"
                )
            return False
        time.sleep(min(WARM_LOCK_CLEANUP_RETRY_SECONDS, remaining))


def _cleanup_old_warm_lock_quarantines(config: PoolConfig) -> None:
    """Best-effort bounded sweep of settled detached lock quarantines."""

    lock_path = warm_lease_lock_path(config)
    checked_at = time.time()
    deadline = time.monotonic() + WARM_LOCK_QUARANTINE_SWEEP_SECONDS
    candidates: list[tuple[float, Path, tuple[int, int]]] = []
    scanned = 0
    try:
        children = os.scandir(lock_path.parent)
    except OSError as exc:
        _warm_coordination_diagnostic(
            f"could not inspect persistent warm lock quarantines under {lock_path.parent}: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    with children:
        for child in children:
            if time.monotonic() >= deadline or scanned >= WARM_LOCK_QUARANTINE_SCAN_LIMIT:
                break
            scanned += 1
            if _warm_lock_ready_quarantine(lock_path, child.name) is None:
                continue
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                continue
            if checked_at - float(metadata.st_mtime) < config.warm_lease_cleanup_defer_seconds:
                continue
            # st_dev is client-local for NFS mounts. Capture and use the
            # sweeper's own identity only; never encode/compare the creator's
            # device number across the submit host and Ada workers.
            candidates.append(
                (
                    float(metadata.st_mtime),
                    Path(child.path),
                    (metadata.st_dev, metadata.st_ino),
                )
            )
    for _modified_at, candidate, expected_identity in sorted(candidates)[
        :WARM_LOCK_QUARANTINE_SWEEP_LIMIT
    ]:
        if time.monotonic() >= deadline:
            break
        _remove_recovered_warm_lock(
            candidate,
            expected_identity=expected_identity,
            timeout_seconds=0.0,
            log_deferred=False,
        )


def _finish_warm_lock_quarantine_cleanup(
    config: PoolConfig,
    quarantine: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Keep all post-detach garbage collection outside the critical path."""

    removed = False
    try:
        removed = _remove_recovered_warm_lock(
            quarantine,
            expected_identity=expected_identity,
            timeout_seconds=config.warm_lease_cleanup_timeout_seconds,
        )
    except Exception as exc:
        # The verified rename above is the unlock linearization point. This is
        # deliberately broad so a cleanup defect cannot report a committed
        # state update as failed or mask the callback's own exception.
        _warm_coordination_diagnostic(
            f"deferred persistent warm lock cleanup for {quarantine}: "
            f"{type(exc).__name__}: {exc}"
        )
    if removed:
        return
    try:
        _cleanup_old_warm_lock_quarantines(config)
    except Exception as exc:
        _warm_coordination_diagnostic(
            f"deferred persistent warm lock quarantine sweep under {quarantine.parent}: "
            f"{type(exc).__name__}: {exc}"
        )


def _recover_stale_warm_lock(config: PoolConfig, lock_path: Path, *, now: float) -> bool:
    directory_fd: int | None = None
    try:
        metadata = lock_path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise WarmLeaseStateError(f"cannot inspect persistent warm lock {lock_path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise WarmLeaseStateError(f"persistent warm lock {lock_path} is not a directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(lock_path, flags)
        opened = os.fstat(directory_fd)
    except FileNotFoundError:
        if directory_fd is not None:
            os.close(directory_fd)
        return True
    except OSError as exc:
        if directory_fd is not None:
            os.close(directory_fd)
        raise WarmLeaseStateError(f"cannot open persistent warm lock {lock_path}: {exc}") from exc
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        os.close(directory_fd)
        return True
    quarantine_token = uuid.uuid4().hex
    pending_quarantine: Path | None = None
    try:
        newest_timestamp = float(metadata.st_mtime)
        try:
            owner = _read_warm_lock_owner_fd(directory_fd, lock_path)
        except WarmLeaseStateError:
            try:
                owner_metadata = os.stat("owner.json", dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                pass
            else:
                newest_timestamp = max(newest_timestamp, float(owner_metadata.st_mtime))
        else:
            newest_timestamp = max(newest_timestamp, float(owner["created_at"]))
        if now - newest_timestamp < config.warm_lease_lock_stale_seconds:
            return False

        transition = _claim_warm_lock_transition(
            config,
            directory_fd,
            lock_path,
            action="recover",
            now=now,
        )
        if transition is None:
            return False

        # The original owner may have been paused between mkdir and writing its
        # record. Once this transition pins the inode, re-read the owner. A
        # newly written owner cancels recovery; it will either continue under
        # the still-canonical inode or observe this transition and abort.
        try:
            refreshed_owner = _read_warm_lock_owner_fd(directory_fd, lock_path)
        except WarmLeaseStateError:
            refreshed_owner = None
        if refreshed_owner is not None and time.time() - float(refreshed_owner["created_at"]) < (
            config.warm_lease_lock_stale_seconds
        ):
            try:
                os.unlink(transition, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            return False

        try:
            current = lock_path.lstat()
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise WarmLeaseStateError(f"cannot recheck persistent warm lock {lock_path}: {exc}") from exc
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            return True

        pending_quarantine = lock_path.with_name(
            f".{lock_path.name}.stale.{quarantine_token}.pending"
        )
        try:
            # The inode-tied transition prevents any cooperating releaser or
            # reclaimer from renaming this path between the check and rename.
            os.rename(lock_path, pending_quarantine)
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise WarmLeaseStateError(f"cannot recover stale persistent warm lock {lock_path}: {exc}") from exc
    finally:
        try:
            os.close(directory_fd)
        except OSError as exc:
            _warm_coordination_diagnostic(
                f"could not close persistent warm lock descriptor for {lock_path}: "
                f"{type(exc).__name__}: {exc}"
            )
    assert pending_quarantine is not None
    moved = pending_quarantine.lstat()
    if (moved.st_dev, moved.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise WarmLeaseStateError(f"persistent warm lock {lock_path} changed during stale recovery")
    quarantine = _ready_warm_lock_quarantine_path(
        lock_path,
        action="recover",
        token=quarantine_token,
    )
    try:
        # This second rename is the GC-readiness handshake. Sweepers do not
        # recognize the pending name, so they cannot race the strict moved-
        # inode verification above. Failure leaves a harmless unsweepable
        # pending quarantine and cannot undo the canonical detach.
        os.rename(pending_quarantine, quarantine)
    except OSError as exc:
        _warm_coordination_diagnostic(
            f"deferred persistent warm lock cleanup for {pending_quarantine}: "
            f"could not publish GC readiness: {type(exc).__name__}: {exc}"
        )
        return True
    _finish_warm_lock_quarantine_cleanup(
        config,
        quarantine,
        expected_identity=(moved.st_dev, moved.st_ino),
    )
    return True


def _acquire_warm_lease_lock(config: PoolConfig) -> _WarmLeaseLockOwnership:
    lock_path = warm_lease_lock_path(config)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + config.warm_lease_lock_timeout_seconds
    token = uuid.uuid4().hex
    first_attempt = True
    while True:
        if not first_attempt and time.monotonic() >= deadline:
            raise WarmLeaseLockTimeout(
                f"timed out after {config.warm_lease_lock_timeout_seconds:g}s acquiring {lock_path}"
            )
        first_attempt = False
        try:
            os.mkdir(lock_path, 0o700)
        except FileExistsError:
            if _recover_stale_warm_lock(config, lock_path, now=time.time()):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WarmLeaseLockTimeout(
                    f"timed out after {config.warm_lease_lock_timeout_seconds:g}s acquiring {lock_path}"
                )
            time.sleep(min(0.05, remaining))
            continue
        except OSError as exc:
            raise WarmLeaseStateError(f"cannot create persistent warm lock {lock_path}: {exc}") from exc

        directory_fd: int | None = None
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            directory_fd = os.open(lock_path, flags)
            metadata = os.fstat(directory_fd)
            _write_warm_lock_record_fd(
                directory_fd,
                "owner.json",
                {
                    "version": WARM_LOCK_OWNER_VERSION,
                    "token": token,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_at": time.time(),
                },
            )
            current = lock_path.lstat()
            if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise WarmLeaseStateError(
                    f"persistent warm lock {lock_path} moved while its owner record was created"
                )
            transitions = [
                name
                for name in os.listdir(directory_fd)
                if _warm_transition_index(name) is not None
            ]
            if transitions:
                raise WarmLeaseStateError(
                    f"persistent warm lock {lock_path} entered recovery while its owner record was created"
                )
            current = lock_path.lstat()
            if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise WarmLeaseStateError(
                    f"persistent warm lock {lock_path} moved while acquisition completed"
                )
        except BaseException:
            try:
                if directory_fd is not None:
                    os.unlink("owner.json", dir_fd=directory_fd)
            except OSError:
                pass
            try:
                current = lock_path.lstat()
                if directory_fd is not None and (current.st_dev, current.st_ino) == (
                    os.fstat(directory_fd).st_dev,
                    os.fstat(directory_fd).st_ino,
                ):
                    lock_path.rmdir()
            except OSError:
                pass
            if directory_fd is not None:
                os.close(directory_fd)
            raise
        assert directory_fd is not None
        return _WarmLeaseLockOwnership(
            config=config,
            path=lock_path,
            token=token,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            directory_fd=directory_fd,
        )


def _release_warm_lease_lock(ownership: _WarmLeaseLockOwnership) -> None:
    quarantine_token = uuid.uuid4().hex
    pending_quarantine: Path | None = None
    try:
        metadata = os.fstat(ownership.directory_fd)
        if (metadata.st_dev, metadata.st_ino) != (ownership.device, ownership.inode):
            return
        try:
            owner = _read_warm_lock_owner_fd(ownership.directory_fd, ownership.path)
        except WarmLeaseStateError:
            return
        if owner["token"] != ownership.token:
            return
        transition = _claim_warm_lock_transition(
            ownership.config,
            ownership.directory_fd,
            ownership.path,
            action="release",
            now=time.time(),
        )
        if transition is None:
            return
        try:
            current = ownership.path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != (ownership.device, ownership.inode):
            return
        pending_quarantine = ownership.path.with_name(
            f".{ownership.path.name}.released.{quarantine_token}.pending"
        )
        os.rename(ownership.path, pending_quarantine)
        moved = pending_quarantine.lstat()
        if (moved.st_dev, moved.st_ino) != (ownership.device, ownership.inode):
            raise WarmLeaseStateError(f"persistent warm lock {ownership.path} changed during release")
    except OSError as exc:
        raise WarmLeaseStateError(f"cannot release persistent warm lock {ownership.path}: {exc}") from exc
    finally:
        try:
            os.close(ownership.directory_fd)
        except OSError as exc:
            _warm_coordination_diagnostic(
                f"could not close persistent warm lock descriptor for {ownership.path}: "
                f"{type(exc).__name__}: {exc}"
            )
    if pending_quarantine is not None:
        quarantine = _ready_warm_lock_quarantine_path(
            ownership.path,
            action="release",
            token=quarantine_token,
        )
        try:
            os.rename(pending_quarantine, quarantine)
        except OSError as exc:
            _warm_coordination_diagnostic(
                f"deferred persistent warm lock cleanup for {pending_quarantine}: "
                f"could not publish GC readiness: {type(exc).__name__}: {exc}"
            )
        else:
            _finish_warm_lock_quarantine_cleanup(
                ownership.config,
                quarantine,
                expected_identity=(ownership.device, ownership.inode),
            )


def _with_warm_lease_lock(config: PoolConfig, callback: Any) -> Any:
    try:
        ownership = _acquire_warm_lease_lock(config)
    except WarmLeaseError:
        raise
    except OSError as exc:
        raise WarmLeaseStateError(f"cannot acquire persistent warm coordination lock: {exc}") from exc
    def release() -> None:
        try:
            _release_warm_lease_lock(ownership)
        except WarmLeaseError:
            raise
        except OSError as exc:
            raise WarmLeaseStateError(f"cannot release persistent warm coordination lock: {exc}") from exc

    try:
        result = callback()
    except BaseException as callback_error:
        try:
            release()
        except Exception as release_error:
            # Preserve the operation's primary failure. The release failure is
            # still visible to operators and on the original exception without
            # replacing the reason the callback failed.
            callback_error.add_note(
                "persistent warm lock release also failed: "
                f"{type(release_error).__name__}: {release_error}"
            )
            _warm_coordination_diagnostic(
                "persistent warm lock release failed while preserving callback error: "
                f"{type(release_error).__name__}: {release_error}"
            )
        raise
    release()
    return result


def claim_persistent_warm_app(
    config: PoolConfig,
    needs: list[AppNeed],
    *,
    worker_id: str,
    now: float | None = None,
) -> PoolAppProfile | None:
    """Atomically reserve at most one persistent warm child for this pool worker."""

    checked_at = time.time() if now is None else float(now)
    need_by_name = {need.name: need for need in needs}
    eligible: list[tuple[PoolAppProfile, AppNeed, int]] = []
    for app in config.apps:
        if not app.enabled or app.persistent_warm_workers <= 0:
            continue
        need = need_by_name.get(app.name)
        if need is None or not need.enabled or need.restart_drain_active:
            continue
        if not need.warm_request_is_active(now=checked_at):
            continue
        limit = min(
            app.persistent_warm_workers,
            need.effective_warm_requested_workers(now=checked_at),
        )
        if limit > 0:
            eligible.append((app, need, limit))
    if not eligible:
        return None

    def claim() -> PoolAppProfile | None:
        leases = _current_warm_leases_unlocked(config, now=checked_at)
        existing = next((lease for lease in leases if lease["worker_id"] == worker_id), None)
        if existing is not None:
            app = next((item[0] for item in eligible if item[0].name == existing["app"]), None)
            if app is not None:
                existing["heartbeat_at"] = checked_at
                _write_warm_leases_unlocked(config, leases, now=checked_at)
                return app
            leases = [lease for lease in leases if lease["worker_id"] != worker_id]
            _write_warm_leases_unlocked(config, leases, now=checked_at)

        for app, need, limit in eligible:
            active = sum(1 for lease in leases if lease["app"] == app.name)
            if active >= limit:
                continue
            leases.append(
                {
                    "app": app.name,
                    "worker_id": worker_id,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "claimed_at": checked_at,
                    "heartbeat_at": checked_at,
                    "request_expires_at": need.warm_request_expires_at,
                    "worker_version": pool_worker_config_version(config),
                }
            )
            _write_warm_leases_unlocked(config, leases, now=checked_at)
            return app
        return None

    return _with_warm_lease_lock(config, claim)


def heartbeat_persistent_warm_lease(
    config: PoolConfig,
    *,
    app_name: str,
    worker_id: str,
    now: float | None = None,
) -> bool:
    checked_at = time.time() if now is None else float(now)

    def heartbeat() -> bool:
        leases = _current_warm_leases_unlocked(config, now=checked_at)
        for lease in leases:
            if lease["app"] == app_name and lease["worker_id"] == worker_id:
                lease["heartbeat_at"] = checked_at
                _write_warm_leases_unlocked(config, leases, now=checked_at)
                return True
        app = next(
            (
                profile
                for profile in config.apps
                if profile.name == app_name and profile.enabled and profile.persistent_warm_workers > 0
            ),
            None,
        )
        if app is None:
            return False
        if any(lease["worker_id"] == worker_id for lease in leases):
            return False
        if sum(1 for lease in leases if lease["app"] == app_name) >= app.persistent_warm_workers:
            return False
        leases.append(
            {
                "app": app_name,
                "worker_id": worker_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": checked_at,
                "heartbeat_at": checked_at,
                "request_expires_at": 0.0,
                "worker_version": pool_worker_config_version(config),
                "recovered": True,
            }
        )
        _write_warm_leases_unlocked(config, leases, now=checked_at)
        return True

    return bool(_with_warm_lease_lock(config, heartbeat))


def release_persistent_warm_lease(
    config: PoolConfig,
    *,
    app_name: str,
    worker_id: str,
    now: float | None = None,
) -> None:
    checked_at = time.time() if now is None else float(now)

    def release() -> None:
        leases = [
            lease
            for lease in _current_warm_leases_unlocked(config, now=checked_at)
            if not (lease["app"] == app_name and lease["worker_id"] == worker_id)
        ]
        _write_warm_leases_unlocked(config, leases, now=checked_at)

    _with_warm_lease_lock(config, release)


def active_persistent_warm_leases(
    config: PoolConfig,
    *,
    now: float | None = None,
) -> list[dict[str, Any]]:
    checked_at = time.time() if now is None else float(now)

    def read() -> list[dict[str, Any]]:
        leases = _current_warm_leases_unlocked(config, now=checked_at)
        _write_warm_leases_unlocked(config, leases, now=checked_at)
        return [dict(lease) for lease in leases]

    return _with_warm_lease_lock(config, read)


def pool_worker_config_version(config: PoolConfig) -> str:
    material = {
        "worker_poll_seconds": config.worker_poll_seconds,
        "worker_idle_timeout_seconds": config.worker_idle_timeout_seconds,
        "worker_max_runtime_seconds": config.worker_max_runtime_seconds,
        "worker_task_timeout_seconds": config.worker_task_timeout_seconds,
        "warm_lease_state_path": str(warm_lease_state_path(config)),
        "warm_lease_heartbeat_seconds": config.warm_lease_heartbeat_seconds,
        "warm_lease_ttl_seconds": config.warm_lease_ttl_seconds,
        "warm_lease_lock_timeout_seconds": config.warm_lease_lock_timeout_seconds,
        "warm_lease_lock_stale_seconds": config.warm_lease_lock_stale_seconds,
        "warm_lease_max_heartbeat_failures": config.warm_lease_max_heartbeat_failures,
        "warm_lease_cleanup_timeout_seconds": config.warm_lease_cleanup_timeout_seconds,
        "warm_lease_cleanup_defer_seconds": config.warm_lease_cleanup_defer_seconds,
        "apps": [
            {
                "name": app.name,
                "enabled": app.enabled,
                "control_url": app.control_url,
                "source_dir": str(app.source_dir),
                "python": app.python,
                "worker_module": app.worker_module,
                "worker_capacity": app.worker_capacity,
                "worker_args": list(app.worker_args),
                "env": dict(sorted(app.env.items())),
                "persistent_warm_workers": app.persistent_warm_workers,
            }
            for app in config.apps
        ],
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{POOL_WORKER_VERSION}-{digest}"


def pool_worker_should_retire(config: PoolConfig) -> bool:
    required_version = dispatch_required_worker_version(config)
    return bool(required_version and required_version != pool_worker_config_version(config))


def dispatch_state_has_activity(config: PoolConfig) -> bool:
    needs = read_dispatch_state_needs(config, allow_stale=True) or []
    now = time.time()
    return any(
        need.enabled and (need.queued_units or need.running_units or need.warm_request_is_active(now=now))
        for need in needs
    )


def report_launcher_status(
    app: PoolAppProfile,
    *,
    launcher_id: str,
    job_name: str,
    partition: str,
    max_jobs: int,
    max_submit_per_cycle: int,
    min_idle_workers: int,
    live_slurm_jobs: int,
    desired_submissions: int,
    submitted_jobs: int,
    status: str,
    message: str = "",
    last_error: str = "",
) -> None:
    if not app.launcher_status_path:
        return
    token = read_token(app.token_file)
    json_request(
        app.control_url,
        token,
        app.launcher_status_path,
        {
            "launcherId": launcher_id,
            "hostname": socket.gethostname(),
            "version": POOL_LAUNCHER_VERSION,
            "status": status,
            "partition": partition,
            "jobName": job_name,
            "maxJobs": max_jobs,
            "maxSubmitPerCycle": max_submit_per_cycle,
            "minIdleWorkers": min_idle_workers,
            "workerCapacity": app.worker_capacity,
            "liveSlurmJobs": live_slurm_jobs,
            "desiredSubmissions": desired_submissions,
            "submittedJobs": submitted_jobs,
            "message": message,
            "lastError": last_error,
        },
        timeout_seconds=30,
        error_cls=LauncherRequestError,
    )


def submit_pool_workers(config: PoolConfig, count: int, *, dry_run: bool = False) -> str:
    if dry_run:
        return f"dry-run would submit {count} shared worker job(s)"
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_DIR": str(config.source_dir),
            "CONFIG_FILE": str(config.config_path),
            "PARTITION": config.partition,
            "JOB_COUNT": str(count),
            "JOB_NAME": config.job_name,
            "TIME_LIMIT": config.time_limit,
            "CPUS_PER_TASK": str(config.cpus_per_task),
            "MEMORY_MB": str(config.memory_mb),
            "POLL_SECONDS": str(config.worker_poll_seconds),
            "OUTPUT_DIR": str(config.output_dir),
            "MAX_RUNTIME_SECONDS": str(config.worker_max_runtime_seconds),
            "IDLE_TIMEOUT_SECONDS": str(config.worker_idle_timeout_seconds),
            "TASK_TIMEOUT_SECONDS": str(config.worker_task_timeout_seconds),
        }
    )
    result = subprocess.run(
        [str(config.submit_script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=config.source_dir,
    )
    return result.stdout.strip()


def maybe_pull_source_dirs(config: PoolConfig) -> None:
    if not config.auto_pull:
        return
    seen: set[Path] = set()
    for source_dir in [config.source_dir, *(app.source_dir for app in config.apps)]:
        if source_dir in seen or not (source_dir / ".git").exists():
            continue
        seen.add(source_dir)
        try:
            result = subprocess.run(
                ["git", "-C", str(source_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
            text = " ".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            if text:
                launcher_log(f"autopull {source_dir}: {text}")
        except Exception as exc:
            launcher_log(f"autopull failed for {source_dir}: {type(exc).__name__}: {exc}")


def poll_launcher_once(config: PoolConfig, *, dry_run: bool = False, scale: PoolScaleState | None = None) -> int:
    maybe_pull_source_dirs(config)
    control_started_at = time.monotonic()
    timing_breakdown: dict[str, float] = {}
    needs: list[AppNeed] = []
    status_success_count = 0
    status_failure_count = 0
    for app in config.apps:
        if not app.enabled:
            needs.append(
                AppNeed(name=app.name, queued_units=0, running_units=0, worker_capacity=app.worker_capacity, enabled=False)
            )
            continue
        app_started_at = time.monotonic()
        try:
            status = fetch_app_status(app, launcher_id=config.launcher_id)
            app_seconds = time.monotonic() - app_started_at
            needs.append(app_need_from_status(app, status))
            timing_breakdown[app.name] = app_seconds
            status_success_count += 1
        except Exception as exc:
            if not is_transient_control_error(exc):
                raise
            app_seconds = time.monotonic() - app_started_at
            status_failure_count += 1
            timing_breakdown[app.name] = app_seconds
            needs.append(
                AppNeed(name=app.name, queued_units=0, running_units=0, worker_capacity=app.worker_capacity, enabled=False)
            )
            launcher_log(f"shared launcher skipping unavailable app {app.name}: {exc}")
    if status_failure_count and status_success_count == 0:
        raise LauncherRequestError("all enabled app status polls failed")
    squeue_started_at = time.monotonic()
    live_jobs = live_slurm_job_count(config.job_name, config.slurm_user)
    squeue_seconds = time.monotonic() - squeue_started_at
    control_plane_seconds = time.monotonic() - control_started_at
    active = any(need.enabled and (need.queued_units or need.running_units) for need in needs)
    if scale is not None:
        timing_breakdown["squeue"] = squeue_seconds
        scale.record_poll(
            active=active,
            control_plane_seconds=control_plane_seconds,
            transient_failure=bool(status_failure_count and status_success_count == 0),
            timings=timing_breakdown,
        )
        effective_max_jobs = scale.effective_max_jobs
        effective_max_submit_per_cycle = scale.effective_max_submit_per_cycle
    else:
        effective_max_jobs = config.max_jobs
        effective_max_submit_per_cycle = config.max_submit_per_cycle
    checked_at = time.time()
    persistent_warm_leases: list[dict[str, Any]] = []
    lease_state_failed = False
    try:
        # Read unconditionally: a fresh child from an older config remains a
        # live Slurm job even if persistence was disabled or its app was removed.
        persistent_warm_leases = active_persistent_warm_leases(config, now=checked_at)
    except Exception as exc:
        lease_state_failed = True
        launcher_log(f"shared launcher could not read persistent warm leases: {type(exc).__name__}: {exc}")
    persistent_app_limits = {
        app.name: app.persistent_warm_workers
        for app in config.apps
        if app.enabled and app.persistent_warm_workers > 0
    }
    current_persistent_requests = {
        need.name: min(
            persistent_app_limits.get(need.name, 0),
            need.effective_warm_requested_workers(now=checked_at),
        )
        for need in needs
        if need.enabled and not need.restart_drain_active
    }
    persistent_lease_counts: dict[str, int] = {}
    for lease in persistent_warm_leases:
        app_name = str(lease.get("app") or "")
        persistent_lease_counts[app_name] = persistent_lease_counts.get(app_name, 0) + 1
    matched_persistent_workers = sum(
        min(lease_count, current_persistent_requests.get(app_name, 0))
        for app_name, lease_count in persistent_lease_counts.items()
    )
    persistent_warm_tail_workers = min(
        max(0, live_jobs),
        max(0, len(persistent_warm_leases) - matched_persistent_workers),
    )
    submission_needs = needs
    if lease_state_failed:
        # Fail closed: do not submit new capacity solely for an uncoordinated
        # persistent request. Reserve plausible *existing* live slots as tail so
        # an unknown warm child cannot hide another app's ordinary demand.
        persistent_names = set(persistent_app_limits)
        submission_needs = [
            replace(need, warm_requested_workers=0, warm_request_expires_at=0.0)
            if need.name in persistent_names
            else need
            for need in needs
        ]
        possible_old_workers = sum(persistent_app_limits.values())
        if os.path.lexists(warm_lease_state_path(config)):
            possible_old_workers = max(1, possible_old_workers)
        persistent_warm_tail_workers = min(max(0, live_jobs), possible_old_workers)
    count = desired_pool_submissions(
        submission_needs,
        live_slurm_jobs=live_jobs,
        min_idle_workers=config.min_idle_workers,
        max_jobs=effective_max_jobs,
        max_submit_per_cycle=effective_max_submit_per_cycle,
        app_start_burst_workers=config.app_start_burst_workers,
        app_start_burst_overcommit=config.app_start_burst_overcommit,
        persistent_warm_tail_workers=persistent_warm_tail_workers,
        now=checked_at,
    )
    need_text = ", ".join(
        (
            f"{need.name}:queued={need.queued_units}:running={need.running_units}:"
            f"cap={need.worker_capacity}:active_workers={need.active_workers}:"
            f"needed={need.needed_workers}:warm={need.effective_warm_requested_workers(now=checked_at)}:"
            f"target={need.target_workers_at(now=checked_at)}"
        )
        for need in needs
    )
    if scale is not None and config.adaptive_scaling_enabled:
        scale_text = (
            f"adaptive={scale.effective_max_jobs}/{config.max_jobs} jobs "
            f"{scale.effective_max_submit_per_cycle}/{config.max_submit_per_cycle} submit "
            f"{scale.last_control_plane_seconds:.2f}s {scale.last_reason}"
        )
    else:
        scale_text = f"max={effective_max_jobs} submit_cap={effective_max_submit_per_cycle}"
    launcher_log(
        f"shared status live={live_jobs} persistent_warm={len(persistent_warm_leases)} "
        f"persistent_warm_tail={persistent_warm_tail_workers} "
        f"submit={count} {scale_text} {need_text}"
    )
    output = ""
    submitted = 0
    if count > 0:
        output = submit_pool_workers(config, count, dry_run=dry_run)
        submitted = 0 if dry_run else count
        launcher_log(output or f"submitted {count} shared worker job(s)")
    write_dispatch_state(
        config,
        needs=needs,
        live_slurm_jobs=live_jobs,
        desired_submissions=count,
        submitted_jobs=submitted,
        status="dry-run" if dry_run and count else ("submitted" if submitted else "idle"),
        message=output or "shared launcher poll completed",
        scale=scale,
    )
    for app in config.apps:
        try:
            report_launcher_status(
                app,
                launcher_id=config.launcher_id,
                job_name=config.job_name,
                partition=config.partition,
                max_jobs=config.max_jobs,
                max_submit_per_cycle=config.max_submit_per_cycle,
                min_idle_workers=config.min_idle_workers,
                live_slurm_jobs=live_jobs,
                desired_submissions=count,
                submitted_jobs=submitted,
                status="dry-run" if dry_run and count else ("submitted" if submitted else "idle"),
                message=(
                    f"{output or 'shared launcher poll completed'}; "
                    f"effective {effective_max_jobs} jobs/{effective_max_submit_per_cycle} submit"
                ),
            )
        except Exception as exc:
            if not is_transient_control_error(exc):
                raise
            launcher_log(f"could not report shared launcher status for {app.name}: {exc}")
    return count


def dispatch_state_lock_path(config: PoolConfig) -> Path:
    return config.dispatch_state_path.with_suffix(config.dispatch_state_path.suffix + ".lock")


def refresh_dispatch_state_from_worker(config: PoolConfig, *, worker_id: str) -> list[AppNeed] | None:
    lock_path = dispatch_state_lock_path(config)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return read_dispatch_state_needs(config)
        try:
            current = read_dispatch_state_needs(config)
            if current is not None:
                return current
            stale_needs = read_dispatch_state_needs(config, allow_stale=True)
            needs: list[AppNeed] = []
            status_success_count = 0
            status_failure_count = 0
            for app in config.apps:
                if not app.enabled:
                    needs.append(
                        AppNeed(
                            name=app.name,
                            queued_units=0,
                            running_units=0,
                            worker_capacity=app.worker_capacity,
                            enabled=False,
                        )
                    )
                    continue
                try:
                    status = fetch_app_status(app, launcher_id=f"{worker_id}-dispatch-refresh")
                    needs.append(app_need_from_status(app, status))
                    status_success_count += 1
                except Exception as exc:
                    if not is_transient_control_error(exc):
                        raise
                    status_failure_count += 1
                    launcher_log(f"worker {worker_id} could not refresh {app.name} dispatch status: {exc}")
                    needs.append(
                        AppNeed(
                            name=app.name,
                            queued_units=0,
                            running_units=0,
                            worker_capacity=app.worker_capacity,
                            enabled=False,
                        )
                    )
            if status_failure_count and status_success_count == 0:
                if stale_needs is not None:
                    launcher_log(f"worker {worker_id} preserving stale dispatch state after all app status polls failed")
                    return stale_needs
                raise LauncherRequestError("all enabled app status polls failed during worker dispatch refresh")
            try:
                live_jobs = live_slurm_job_count(config.job_name, config.slurm_user)
            except Exception as exc:
                if not is_transient_control_error(exc):
                    raise
                launcher_log(f"worker {worker_id} could not count Slurm jobs during dispatch refresh: {exc}")
                live_jobs = 0
            write_dispatch_state(
                config,
                needs=needs,
                live_slurm_jobs=live_jobs,
                desired_submissions=0,
                submitted_jobs=0,
                status="worker-refresh",
                message=f"dispatch state refreshed by {worker_id}",
            )
            return needs
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def candidate_apps_from_needs(
    config: PoolConfig,
    needs: list[AppNeed],
    *,
    worker_id: str,
) -> list[tuple[PoolAppProfile, AppNeed]]:
    app_by_name = {app.name: app for app in config.apps if app.enabled}
    candidates: list[tuple[PoolAppProfile, AppNeed]] = []
    for need in needs:
        app = app_by_name.get(need.name)
        if app is None:
            continue
        if need.enabled and not need.restart_drain_active and need.queued_units > 0:
            candidates.append((app, need))
    if not candidates:
        return []

    def tie_breaker(name: str) -> int:
        digest = hashlib.sha256(f"{worker_id}:{name}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    def priority(item: tuple[PoolAppProfile, AppNeed]) -> tuple[int, int, int, int, int, int]:
        _app, need = item
        active_workers = need.active_workers
        shortfall = max(0, need.needed_workers - active_workers)
        startup_rank = 0 if active_workers == 0 else 1
        return (
            startup_rank,
            active_workers,
            -shortfall,
            -need.needed_workers,
            -need.queued_units,
            tie_breaker(need.name),
        )

    return sorted(candidates, key=priority)


def candidate_apps(config: PoolConfig, *, worker_id: str) -> list[tuple[PoolAppProfile, AppNeed]]:
    needs = read_dispatch_state_needs(config)
    if needs is None:
        needs = refresh_dispatch_state_from_worker(config, worker_id=worker_id)
    if needs is None:
        return []
    return candidate_apps_from_needs(config, needs, worker_id=worker_id)


def app_python_command(app: PoolAppProfile) -> str:
    python = app.python
    if "/" in python:
        path = Path(python)
        if not path.is_absolute():
            path = app.source_dir / path
        return str(path)
    return python


def app_worker_command(
    app: PoolAppProfile,
    *,
    pool_worker_id: str,
    scratch_root: Path,
    once: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    app_scratch = scratch_root / app.name
    app_scratch.mkdir(parents=True, exist_ok=True)
    command = [
        app_python_command(app),
        "-m",
        app.worker_module,
        "--control-url",
        app.control_url,
        "--token-file",
        str(app.token_file),
        "--worker-id",
        f"{pool_worker_id}-{app.name}",
        "--scratch-root",
        str(app_scratch),
    ]
    if once:
        command.append("--once")
    command.extend(app.worker_args)
    env = os.environ.copy()
    env.update(app.env)
    return command, app_scratch, env


def run_app_worker_once(
    app: PoolAppProfile,
    *,
    pool_worker_id: str,
    scratch_root: Path,
    task_timeout_seconds: float,
) -> int:
    command, _app_scratch, env = app_worker_command(
        app,
        pool_worker_id=pool_worker_id,
        scratch_root=scratch_root,
        once=True,
    )
    launcher_log(f"pool worker {pool_worker_id} delegating to {app.name}")
    result = subprocess.run(
        command,
        cwd=app.source_dir,
        env=env,
        timeout=task_timeout_seconds or None,
    )
    return int(result.returncode or 0)


def _stop_persistent_child(process: subprocess.Popen[Any], *, worker_id: str, app_name: str) -> None:
    if process.poll() is not None:
        return
    launcher_log(f"pool worker {worker_id} stopping persistent {app_name} child")
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        launcher_log(f"pool worker {worker_id} killing unresponsive persistent {app_name} child")
        process.kill()
        process.wait(timeout=10)


def run_persistent_app_worker(
    config: PoolConfig,
    app: PoolAppProfile,
    *,
    pool_worker_id: str,
    scratch_root: Path,
) -> int:
    command, _app_scratch, env = app_worker_command(
        app,
        pool_worker_id=pool_worker_id,
        scratch_root=scratch_root,
        once=False,
    )
    launcher_log(f"pool worker {pool_worker_id} starting persistent warm {app.name} child")
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(command, cwd=app.source_dir, env=env)
        heartbeat_failures = 0
        drain_requested = False
        pending_drain_reason = ""
        drain_signal_failures = 0
        while True:
            try:
                return int(process.wait(timeout=config.warm_lease_heartbeat_seconds) or 0)
            except subprocess.TimeoutExpired:
                if drain_requested:
                    continue
                drain_reason = pending_drain_reason
                if not drain_reason:
                    try:
                        maintained = heartbeat_persistent_warm_lease(
                            config,
                            app_name=app.name,
                            worker_id=pool_worker_id,
                        )
                    except Exception as exc:
                        heartbeat_failures += 1
                        launcher_log(
                            f"pool worker {pool_worker_id} could not heartbeat its persistent {app.name} lease "
                            f"({heartbeat_failures}/{config.warm_lease_max_heartbeat_failures}): "
                            f"{type(exc).__name__}: {exc}"
                        )
                        if heartbeat_failures >= config.warm_lease_max_heartbeat_failures:
                            drain_reason = "lease heartbeat repeatedly failed"
                    else:
                        heartbeat_failures = 0
                        if not maintained:
                            drain_reason = "lease ownership was lost"
                if drain_reason:
                    if not pending_drain_reason:
                        launcher_log(
                            f"pool worker {pool_worker_id} requesting graceful drain of persistent {app.name} child: "
                            f"{drain_reason}"
                        )
                    pending_drain_reason = drain_reason
                    try:
                        process.send_signal(signal.SIGUSR1)
                    except ProcessLookupError:
                        drain_requested = True
                    except Exception as exc:
                        drain_signal_failures += 1
                        launcher_log(
                            f"pool worker {pool_worker_id} could not signal persistent {app.name} child to drain: "
                            f"{type(exc).__name__}: {exc}; will retry "
                            f"({drain_signal_failures} failed attempt(s))"
                        )
                    else:
                        # SIGUSR1 synchronously flips the compatible app worker's
                        # drain event. It stops new claims immediately; this
                        # parent then waits without a kill deadline so an active
                        # turn can finish safely before the child exits.
                        drain_requested = True
    except BaseException:
        if process is not None:
            _stop_persistent_child(process, worker_id=pool_worker_id, app_name=app.name)
        raise
    finally:
        try:
            release_persistent_warm_lease(
                config,
                app_name=app.name,
                worker_id=pool_worker_id,
            )
        except Exception as exc:
            launcher_log(f"pool worker {pool_worker_id} could not release its persistent {app.name} lease: {exc}")


def run_pool_worker(config: PoolConfig, *, worker_id: str, scratch_root: Path) -> int:
    scratch_root.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    last_work_at = started_at
    failures = 0
    next_warm_claim_at = started_at
    launcher_log(f"shared pool worker {worker_id} started")
    while True:
        if pool_worker_should_retire(config):
            required_version = dispatch_required_worker_version(config) or "unknown"
            launcher_log(
                (
                    f"shared pool worker {worker_id} retiring for worker version "
                    f"{required_version}; local version is {pool_worker_config_version(config)}"
                )
            )
            return 0
        if config.worker_max_runtime_seconds and time.monotonic() - started_at >= config.worker_max_runtime_seconds:
            launcher_log(f"shared pool worker {worker_id} max runtime reached")
            return 1 if failures else 0
        if config.worker_idle_timeout_seconds and time.monotonic() - last_work_at >= config.worker_idle_timeout_seconds:
            launcher_log(f"shared pool worker {worker_id} idle timeout reached")
            return 1 if failures else 0
        needs = read_dispatch_state_needs(config)
        if needs is None:
            needs = refresh_dispatch_state_from_worker(config, worker_id=worker_id)
        now = time.monotonic()
        warm_app = None
        if needs is not None and now >= next_warm_claim_at:
            try:
                warm_app = claim_persistent_warm_app(config, needs, worker_id=worker_id)
            except WarmLeaseError as exc:
                launcher_log(
                    f"pool worker {worker_id} could not coordinate a persistent warm child: "
                    f"{type(exc).__name__}: {exc}"
                )
                warm_app = None
            if warm_app is None:
                jitter = int(hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:4], 16) % 1000 / 1000.0
                next_warm_claim_at = now + config.warm_lease_heartbeat_seconds * (1.0 + jitter)
        if warm_app is not None:
            try:
                code = run_persistent_app_worker(
                    config,
                    warm_app,
                    pool_worker_id=worker_id,
                    scratch_root=scratch_root,
                )
            except Exception as exc:
                failures += 1
                launcher_log(
                    f"pool worker {worker_id} failed while warming {warm_app.name}: {type(exc).__name__}: {exc}"
                )
                time.sleep(config.worker_poll_seconds)
                continue
            last_work_at = time.monotonic()
            next_warm_claim_at = last_work_at
            if code:
                failures += 1
                launcher_log(f"pool worker {worker_id} persistent {warm_app.name} child exited with {code}")
                next_warm_claim_at = last_work_at + max(
                    config.worker_poll_seconds,
                    config.warm_lease_heartbeat_seconds,
                )
                time.sleep(config.worker_poll_seconds)
            continue
        candidates = candidate_apps_from_needs(config, needs or [], worker_id=worker_id)
        if not candidates:
            time.sleep(config.worker_poll_seconds)
            continue
        app, _need = candidates[0]
        try:
            code = run_app_worker_once(
                app,
                pool_worker_id=worker_id,
                scratch_root=scratch_root,
                task_timeout_seconds=config.worker_task_timeout_seconds,
            )
        except Exception as exc:
            failures += 1
            launcher_log(f"pool worker {worker_id} failed while running {app.name}: {type(exc).__name__}: {exc}")
            time.sleep(config.worker_poll_seconds)
            continue
        last_work_at = time.monotonic()
        if code:
            failures += 1
            launcher_log(f"pool worker {worker_id} app {app.name} exited with {code}")


def run_launcher_loop(config: PoolConfig, *, once: bool = False, dry_run: bool = False) -> int:
    launcher_log(f"starting shared launcher {config.launcher_id}")
    idle_started_at: float | None = None
    scale = PoolScaleState(config)
    while True:
        activity = False
        poll_started_at = time.monotonic()
        try:
            submitted = poll_launcher_once(config, dry_run=dry_run, scale=scale)
            activity = bool(submitted) or dispatch_state_has_activity(config)
        except Exception as exc:
            if not is_transient_control_error(exc):
                raise
            scale.record_poll(
                active=True,
                control_plane_seconds=time.monotonic() - poll_started_at,
                transient_failure=True,
            )
            launcher_log(f"shared launcher control plane temporarily unavailable: {exc}; retrying")
        if once:
            return 0
        if activity:
            idle_started_at = None
            sleep_seconds = config.launcher_active_poll_seconds
        else:
            now = time.monotonic()
            if idle_started_at is None:
                idle_started_at = now
            if now - idle_started_at < config.launcher_idle_backoff_after_seconds:
                sleep_seconds = config.launcher_active_poll_seconds
            else:
                sleep_seconds = config.launcher_idle_poll_seconds
        time.sleep(max(1.0, sleep_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a shared Slurm worker pool for configured apps.")
    parser.add_argument("mode", choices=("launcher", "worker"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-id", default=f"shared-worker-{socket.gethostname()}")
    parser.add_argument("--scratch-root", type=Path, default=Path(os.environ.get("SLURM_TMPDIR") or "/tmp") / "shared-worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_pool_config(args.config)
    if args.mode == "launcher":
        return run_launcher_loop(config, once=args.once, dry_run=args.dry_run)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate_worker(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_worker)
    try:
        return run_pool_worker(config, worker_id=args.worker_id, scratch_root=args.scratch_root)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    sys.exit(main())
