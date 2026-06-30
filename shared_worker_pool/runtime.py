"""Shared control-plane helpers for Slurm launchers and remote workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
SLURM_COMMAND_DIRS = (
    "/opt/slurm/bin",
    "/opt/slurm-25.11.6/bin",
    "/opt/slurm-24.11.5/bin",
)


class ControlPlaneRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LauncherRequestError(ControlPlaneRequestError):
    pass


class WorkerRequestError(ControlPlaneRequestError):
    pass


def timestamped_log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def launcher_log(message: str) -> None:
    timestamped_log(message)


def worker_log(message: str) -> None:
    timestamped_log(message)


def read_token(path: Path) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 24:
        raise RuntimeError(f"worker token in {path} is empty or too short")
    return token


def json_request(
    control_url: str,
    token: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 60,
    error_cls: type[ControlPlaneRequestError] = ControlPlaneRequestError,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{control_url.rstrip('/')}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise error_cls(f"{path} failed with HTTP {exc.code}: {text}", status_code=exc.code) from exc
    except URLError as exc:
        raise error_cls(f"{path} failed: {exc}") from exc


def is_transient_control_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if not isinstance(exc, ControlPlaneRequestError):
        return False
    return exc.status_code is None or exc.status_code in TRANSIENT_HTTP_STATUSES


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def worker_is_fresh(
    worker: dict[str, Any],
    *,
    now: datetime,
    heartbeat_ttl_seconds: int,
    status_fields: tuple[str, ...] = ("status",),
    stopped_statuses: set[str] | frozenset[str] = frozenset({"stopped"}),
    honor_is_fresh: bool = False,
) -> bool:
    status = ""
    for field in status_fields:
        value = worker.get(field)
        if value:
            status = str(value).lower()
            break
    if status in stopped_statuses:
        return False
    if honor_is_fresh and worker.get("isFresh") is False:
        return False
    if heartbeat_ttl_seconds <= 0:
        return True
    last_seen = parse_iso_datetime(worker.get("lastSeenAt"))
    if last_seen is None:
        return False
    return (now - last_seen).total_seconds() <= heartbeat_ttl_seconds


def active_worker_summary(
    status: dict[str, Any],
    *,
    now: datetime | None = None,
    active_field: str = "activeJobs",
    output_active_field: str = "activeJobs",
    capacity_field: str = "capacity",
    effective_capacity_field: str | None = None,
    available_capacity_field: str | None = None,
    min_capacity: int = 1,
    status_fields: tuple[str, ...] = ("status",),
    stopped_statuses: set[str] | frozenset[str] = frozenset({"stopped"}),
    honor_is_fresh: bool = False,
) -> dict[str, int]:
    now = now or datetime.now(UTC)
    heartbeat_ttl = int(status.get("heartbeatTtlSeconds") or 90)
    workers = status.get("workers") if isinstance(status.get("workers"), list) else []
    active_workers = 0
    capacity = 0
    active_work = 0
    available_capacity = 0
    for worker in workers:
        if not isinstance(worker, dict) or not worker_is_fresh(
            worker,
            now=now,
            heartbeat_ttl_seconds=heartbeat_ttl,
            status_fields=status_fields,
            stopped_statuses=stopped_statuses,
            honor_is_fresh=honor_is_fresh,
        ):
            continue
        raw_capacity = None
        if effective_capacity_field is not None:
            raw_capacity = worker.get(effective_capacity_field)
        if raw_capacity is None:
            raw_capacity = worker.get(capacity_field)
        worker_capacity = max(min_capacity, int(raw_capacity or 0))
        worker_active = max(0, int(worker.get(active_field) or 0))
        if available_capacity_field is not None and worker.get(available_capacity_field) is not None:
            worker_available = max(0, int(worker.get(available_capacity_field) or 0))
        else:
            worker_available = max(0, worker_capacity - worker_active)
        active_workers += 1
        capacity += worker_capacity
        active_work += worker_active
        available_capacity += worker_available
    return {
        "activeWorkers": active_workers,
        "capacity": capacity,
        output_active_field: active_work,
        "availableCapacity": available_capacity,
    }


def find_command(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in SLURM_COMMAND_DIRS:
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def live_slurm_job_count(job_name: str, user: str | None = None) -> int:
    squeue = find_command("squeue")
    if not squeue:
        raise RuntimeError("squeue not found; run the launcher on a Slurm submit host such as newcastle.cs.ubc.ca")
    command = [squeue, "-h", "-n", job_name, "-t", "PD,R,CF,CG", "-o", "%i"]
    if user:
        command[1:1] = ["-u", user]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    return len([line for line in result.stdout.splitlines() if line.strip()])


def host_load_cpu_basis(*, allocated_cpus: int, visible_cpus: int | None = None) -> int:
    """Return the CPU count that host load thresholds should be measured against.

    Slurm workers often request a small CPU allocation on a much larger shared
    host. The system load average is host-wide, so comparing it only to the
    requested allocation can make healthy nodes look overloaded.
    """
    allocated = max(1, int(allocated_cpus or 1))
    if visible_cpus is None:
        visible_cpus = os.cpu_count()
    visible = max(1, int(visible_cpus or allocated))
    return max(allocated, visible)


def host_load_is_high(
    load_1m: float | None,
    *,
    allocated_cpus: int,
    max_load_per_cpu: float,
    visible_cpus: int | None = None,
) -> bool:
    if load_1m is None or max_load_per_cpu <= 0:
        return False
    return float(load_1m) >= float(max_load_per_cpu) * host_load_cpu_basis(
        allocated_cpus=allocated_cpus,
        visible_cpus=visible_cpus,
    )


def free_disk_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(path).free)


def scratch_root_is_safe_to_remove(path: Path, *, allowed_prefixes: tuple[str, ...]) -> bool:
    resolved = path.expanduser().resolve()
    if not resolved.name.startswith(allowed_prefixes):
        return False
    if len(resolved.parts) < 3:
        return False
    return True


def cleanup_scratch_root(
    path: Path,
    *,
    preserve: bool = False,
    allowed_prefixes: tuple[str, ...],
) -> None:
    if preserve:
        return
    if not scratch_root_is_safe_to_remove(path, allowed_prefixes=allowed_prefixes):
        worker_log(f"not removing scratch root with unexpected name: {path}")
        return
    shutil.rmtree(path.expanduser().resolve(), ignore_errors=True)


def timeout_reached(started_at: float, max_runtime_seconds: float) -> bool:
    return bool(max_runtime_seconds and time.monotonic() - started_at >= max_runtime_seconds)


def idle_timeout_reached(last_claim_at: float, idle_timeout_seconds: float) -> bool:
    return bool(idle_timeout_seconds and time.monotonic() - last_claim_at >= idle_timeout_seconds)
