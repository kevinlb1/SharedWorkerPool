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
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared_worker_pool.runtime import (
    LauncherRequestError,
    is_transient_control_error,
    json_request,
    launcher_log,
    live_slurm_job_count,
    read_token,
)


POOL_LAUNCHER_VERSION = "shared-worker-pool-20260629-fairness"
POOL_WORKER_VERSION = "shared-worker-pool-worker-20260629-fairness"


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


@dataclass(frozen=True)
class AppNeed:
    name: str
    queued_units: int
    running_units: int
    worker_capacity: int
    enabled: bool = True
    restart_drain_active: bool = False
    warm_requested_workers: int = 0

    @property
    def needed_workers(self) -> int:
        if not self.enabled or self.restart_drain_active or self.queued_units <= 0:
            return 0
        return math.ceil(self.queued_units / max(1, self.worker_capacity))

    @property
    def active_workers(self) -> int:
        return math.ceil(max(0, self.running_units) / max(1, self.worker_capacity))

    @property
    def target_workers(self) -> int:
        if not self.enabled or self.restart_drain_active:
            return 0
        return self.active_workers + max(self.needed_workers, max(0, self.warm_requested_workers))


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
            )
        )
    if not apps:
        raise ValueError("shared worker pool config must contain at least one app profile")
    max_jobs = max(0, int(pool.get("max_jobs") or 400))
    max_submit_per_cycle = max(0, int(pool.get("max_submit_per_cycle") or 100))
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
    warm_requested_workers = max(0, int(warm_pool.get("requestedWorkers") or 0))
    if mode == "caida":
        jobs = status.get("jobs") if isinstance(status.get("jobs"), dict) else {}
        enabled = bool(status.get("clusterWorkersEnabled") or status.get("remoteWorkersEnabled"))
        return AppNeed(
            name=app.name,
            queued_units=max(0, int(jobs.get("queuedUnclaimed") or 0)),
            running_units=max(0, int(jobs.get("runningTotal") or 0)),
            worker_capacity=app.worker_capacity,
            enabled=app.enabled and enabled,
            restart_drain_active=bool(status.get("restartDrainActive")),
            warm_requested_workers=warm_requested_workers,
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
) -> int:
    active_needs = [need for need in needs if need.enabled and not need.restart_drain_active]
    needed_workers = sum(need.target_workers for need in active_needs)
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
        "needed_workers": need.needed_workers,
        "active_workers": need.active_workers,
        "target_workers": need.target_workers,
    }


def app_need_from_payload(payload: dict[str, Any]) -> AppNeed:
    return AppNeed(
        name=str(payload.get("name") or ""),
        queued_units=max(0, int(payload.get("queued_units") or 0)),
        running_units=max(0, int(payload.get("running_units") or 0)),
        worker_capacity=max(1, int(payload.get("worker_capacity") or 1)),
        enabled=bool(payload.get("enabled", True)),
        restart_drain_active=bool(payload.get("restart_drain_active")),
        warm_requested_workers=max(0, int(payload.get("warm_requested_workers") or 0)),
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
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        expires_at = float(raw.get("expires_at") or 0.0)
    except (TypeError, ValueError):
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
            need = app_need_from_payload(item)
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


def pool_worker_config_version(config: PoolConfig) -> str:
    material = {
        "worker_poll_seconds": config.worker_poll_seconds,
        "worker_idle_timeout_seconds": config.worker_idle_timeout_seconds,
        "worker_max_runtime_seconds": config.worker_max_runtime_seconds,
        "worker_task_timeout_seconds": config.worker_task_timeout_seconds,
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
    return any(need.enabled and (need.queued_units or need.running_units or need.warm_requested_workers) for need in needs)


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
    count = desired_pool_submissions(
        needs,
        live_slurm_jobs=live_jobs,
        min_idle_workers=config.min_idle_workers,
        max_jobs=effective_max_jobs,
        max_submit_per_cycle=effective_max_submit_per_cycle,
        app_start_burst_workers=config.app_start_burst_workers,
        app_start_burst_overcommit=config.app_start_burst_overcommit,
    )
    need_text = ", ".join(
        (
            f"{need.name}:queued={need.queued_units}:running={need.running_units}:"
            f"cap={need.worker_capacity}:active_workers={need.active_workers}:"
            f"needed={need.needed_workers}:warm={need.warm_requested_workers}:target={need.target_workers}"
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
    launcher_log(f"shared status live={live_jobs} submit={count} {scale_text} {need_text}")
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


def run_app_worker_once(
    app: PoolAppProfile,
    *,
    pool_worker_id: str,
    scratch_root: Path,
    task_timeout_seconds: float,
) -> int:
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
        "--once",
        *app.worker_args,
    ]
    env = os.environ.copy()
    env.update(app.env)
    launcher_log(f"pool worker {pool_worker_id} delegating to {app.name}")
    result = subprocess.run(
        command,
        cwd=app.source_dir,
        env=env,
        timeout=task_timeout_seconds or None,
    )
    return int(result.returncode or 0)


def run_pool_worker(config: PoolConfig, *, worker_id: str, scratch_root: Path) -> int:
    scratch_root.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    last_work_at = started_at
    failures = 0
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
        candidates = candidate_apps(config, worker_id=worker_id)
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
    return run_pool_worker(config, worker_id=args.worker_id, scratch_root=args.scratch_root)


if __name__ == "__main__":
    sys.exit(main())
