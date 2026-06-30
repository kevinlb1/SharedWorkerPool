"""Shared control-plane helpers for Slurm launchers and remote workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
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


PROCESS_CLOCK_TICKS = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
PAGE_SIZE_BYTES = os.sysconf("SC_PAGE_SIZE")


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


def env_int(name: str) -> int:
    try:
        return int(str(os.environ.get(name) or "").strip() or "0")
    except ValueError:
        return 0


def meminfo_bytes() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if not key or not parts:
                continue
            try:
                kib = int(parts[0])
            except ValueError:
                continue
            values[key] = kib * 1024
    except OSError:
        return {}
    return values


def effective_allocated_cpus(explicit_allocated_cpus: int | None = None) -> int:
    explicit = max(0, int(explicit_allocated_cpus or 0))
    if explicit:
        return explicit
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = env_int(name)
        if value > 0:
            return value
    return max(1, int(os.cpu_count() or 1))


def effective_allocated_memory_mb(
    explicit_allocated_memory_mb: int | None = None,
    *,
    allocated_cpus: int | None = None,
) -> int:
    explicit = max(0, int(explicit_allocated_memory_mb or 0))
    if explicit:
        return explicit
    mem_per_node = env_int("SLURM_MEM_PER_NODE")
    if mem_per_node > 0:
        return mem_per_node
    mem_per_cpu = env_int("SLURM_MEM_PER_CPU")
    if mem_per_cpu > 0:
        return mem_per_cpu * max(1, int(allocated_cpus or effective_allocated_cpus()))
    return 0


def current_load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def proc_snapshot() -> dict[int, tuple[int, int, int]]:
    snapshot: dict[int, tuple[int, int, int]] = {}
    proc_root = Path("/proc")
    if not proc_root.exists():
        return snapshot
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
            close_paren = stat_text.rfind(")")
            if close_paren < 0:
                continue
            fields = stat_text[close_paren + 2 :].split()
            ppid = int(fields[1])
            cpu_ticks = int(fields[11]) + int(fields[12])
            statm_fields = (entry / "statm").read_text(encoding="utf-8", errors="ignore").split()
            rss_bytes = int(statm_fields[1]) * PAGE_SIZE_BYTES if len(statm_fields) > 1 else 0
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError, OSError):
            continue
        snapshot[pid] = (ppid, rss_bytes, cpu_ticks)
    return snapshot


def process_forest_resources(root_pids: list[int]) -> dict[str, int]:
    roots = {pid for pid in root_pids if pid > 0}
    snapshot = proc_snapshot()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss, _cpu_ticks) in snapshot.items():
        children.setdefault(ppid, []).append(pid)
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    rss_bytes = 0
    cpu_ticks = 0
    process_count = 0
    for pid in seen:
        item = snapshot.get(pid)
        if item is None:
            continue
        _ppid, rss, ticks = item
        rss_bytes += rss
        cpu_ticks += ticks
        process_count += 1
    return {"rssBytes": rss_bytes, "cpuTicks": cpu_ticks, "processCount": process_count}


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


@dataclass(frozen=True)
class ResourceAdmission:
    allowed: bool
    reason: str
    advertised_capacity: int
    metrics: dict[str, Any]


def resource_admission(
    *,
    scratch_root: Path,
    configured_capacity: int,
    active_work: int,
    allocated_cpus: int | None = None,
    allocated_memory_mb: int | None = None,
    max_load_per_cpu: float = 0.0,
    min_free_memory_mb: float = 0.0,
    memory_reserve_per_work_mb: float = 0.0,
    min_free_disk_mb: float = 0.0,
    root_pids: list[int] | None = None,
    work_label: str = "job",
) -> ResourceAdmission:
    configured_capacity = max(1, int(configured_capacity or 1))
    active_work = max(0, int(active_work or 0))
    allocated_cpus = effective_allocated_cpus(allocated_cpus)
    visible_cpus = max(1, int(os.cpu_count() or allocated_cpus or 1))
    cpu_basis = host_load_cpu_basis(allocated_cpus=allocated_cpus, visible_cpus=visible_cpus)
    allocated_memory_mb = effective_allocated_memory_mb(
        allocated_memory_mb,
        allocated_cpus=allocated_cpus,
    )
    free_disk = free_disk_bytes(scratch_root)
    free_disk_mb = free_disk / (1024 * 1024)
    meminfo = meminfo_bytes()
    available_memory_bytes = meminfo.get("MemAvailable", 0)
    available_memory_mb = available_memory_bytes / (1024 * 1024) if available_memory_bytes else 0.0
    worker_resources = process_forest_resources(root_pids or [os.getpid()])
    worker_rss_mb = worker_resources["rssBytes"] / (1024 * 1024)
    allocation_free_memory_mb = (
        max(0.0, float(allocated_memory_mb) - worker_rss_mb) if allocated_memory_mb > 0 else None
    )
    load_average = current_load_average()
    load_1m = float(load_average[0]) if load_average is not None else None
    max_load_per_cpu = max(0.0, float(max_load_per_cpu or 0.0))
    min_free_memory_mb = max(0.0, float(min_free_memory_mb or 0.0))
    memory_reserve_per_work_mb = max(0.0, float(memory_reserve_per_work_mb or 0.0))
    min_free_disk_mb = max(0.0, float(min_free_disk_mb or 0.0))
    required_memory_headroom_mb = min_free_memory_mb + memory_reserve_per_work_mb

    metrics: dict[str, Any] = {
        "configuredCapacity": configured_capacity,
        "activeWork": active_work,
        "activeTurns": active_work,
        "allocatedCpus": allocated_cpus,
        "visibleCpus": visible_cpus,
        "hostLoadCpuBasis": cpu_basis,
        "allocatedMemoryMb": allocated_memory_mb,
        "freeDiskMb": free_disk_mb,
        "availableMemoryMb": available_memory_mb,
        "workerRssMb": worker_rss_mb,
        "workerProcessCount": worker_resources["processCount"],
        "allocationFreeMemoryMb": allocation_free_memory_mb,
        "load1m": load_1m,
        "maxLoadPerCpu": max_load_per_cpu,
        "minFreeMemoryMb": min_free_memory_mb,
        "memoryReservePerWorkMb": memory_reserve_per_work_mb,
        "memoryReservePerTurnMb": memory_reserve_per_work_mb,
        "minFreeDiskMb": min_free_disk_mb,
    }

    if active_work >= configured_capacity:
        return ResourceAdmission(
            allowed=False,
            reason=f"worker is at configured {work_label} capacity",
            advertised_capacity=configured_capacity,
            metrics=metrics,
        )

    reason = ""
    if min_free_disk_mb and free_disk_mb < min_free_disk_mb:
        reason = f"scratch disk headroom is low ({free_disk_mb:.0f} MiB free)"
    elif required_memory_headroom_mb and available_memory_mb and available_memory_mb < required_memory_headroom_mb:
        reason = f"system memory headroom is low ({available_memory_mb:.0f} MiB available)"
    elif (
        required_memory_headroom_mb
        and allocation_free_memory_mb is not None
        and allocation_free_memory_mb < required_memory_headroom_mb
    ):
        reason = f"worker allocation memory headroom is low ({allocation_free_memory_mb:.0f} MiB remaining)"
    elif host_load_is_high(
        load_1m,
        allocated_cpus=allocated_cpus,
        visible_cpus=visible_cpus,
        max_load_per_cpu=max_load_per_cpu,
    ):
        reason = (
            f"host load is high ({load_1m:.2f} over {cpu_basis} visible CPU(s); "
            f"allocation has {allocated_cpus} CPU(s))"
        )

    if reason:
        return ResourceAdmission(
            allowed=False,
            reason=reason,
            advertised_capacity=active_work,
            metrics=metrics,
        )
    return ResourceAdmission(
        allowed=True,
        reason="resource headroom is available",
        advertised_capacity=configured_capacity,
        metrics=metrics,
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
