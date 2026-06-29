"""Submit generic SharedWorkerPool Slurm workers.

This module replaces app-local shell implementations with one reusable
submission path. Configuration still comes from environment variables so app
deployment wrappers can stay thin.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SLURM_PATH = "${HOME}/.opencode/bin:/opt/slurm/bin:/opt/slurm-25.11.6/bin:/opt/slurm-24.11.5/bin:${PATH}"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _nonnegative_int(name: str, default: str) -> int:
    raw = _env(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise SystemExit(f"{name} must be a non-negative integer")
    return value


def _quote(value: str | Path) -> str:
    return repr(str(value))


def build_job_script(
    *,
    source_dir: Path,
    config_file: Path,
    job_name: str,
    partition: str,
    time_limit: str,
    cpus_per_task: int,
    memory_mb: int,
    output_dir: Path,
    install_if_missing: str,
    poll_seconds: str,
    max_runtime_seconds: str,
    idle_timeout_seconds: str,
    task_timeout_seconds: str,
    cache_ttl_seconds: str,
) -> str:
    memory_directive = f"#SBATCH --mem={memory_mb}M" if memory_mb else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
{memory_directive}
#SBATCH --output={output_dir}/%x-%A_%a.out

set -euo pipefail
export PATH="{SLURM_PATH}"

SOURCE_DIR={_quote(source_dir)}
CONFIG_FILE={_quote(config_file)}
INSTALL_IF_MISSING={_quote(install_if_missing)}
POLL_SECONDS={_quote(poll_seconds)}
MAX_RUNTIME_SECONDS={_quote(max_runtime_seconds)}
IDLE_TIMEOUT_SECONDS={_quote(idle_timeout_seconds)}
TASK_TIMEOUT_SECONDS={_quote(task_timeout_seconds)}
CACHE_TTL_SECONDS={_quote(cache_ttl_seconds)}

cd "${{SOURCE_DIR}}"

if [[ "${{INSTALL_IF_MISSING}}" == "1" && ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi

worker_host="${{HOSTNAME:-$(hostname)}}"
worker_host="${{worker_host%%.*}}"
worker_slot="${{SLURM_ARRAY_TASK_ID:-0}}"
worker_id="shared-${{SLURM_JOB_ID:-manual}}-${{worker_slot}}-${{worker_host}}"
scratch_parent="${{SLURM_TMPDIR:-${{TMPDIR:-/tmp}}}}"
scratch_root="${{scratch_parent}}/shared-worker-${{SLURM_JOB_ID:-manual}}-${{worker_slot}}"
cache_parent="${{SHARED_WORKER_CACHE_PARENT:-${{TMPDIR:-/tmp}}}}"
cache_root="${{SHARED_WORKER_CACHE_ROOT:-${{cache_parent}}/shared-worker-cache-${{USER:-unknown}}}}"
cache_ttl_minutes="$((CACHE_TTL_SECONDS / 60))"

mkdir -p "${{scratch_root}}"
mkdir -p "${{cache_root}}/pip" "${{cache_root}}/uv" "${{cache_root}}/npm" "${{cache_root}}/codingworkspace-git"
chmod 700 "${{cache_root}}" "${{cache_root}}/pip" "${{cache_root}}/uv" "${{cache_root}}/npm" "${{cache_root}}/codingworkspace-git" 2>/dev/null || true

case "${{cache_root}}" in
  /tmp/shared-worker-cache-*|"${{TMPDIR:-/tmp}}"/shared-worker-cache-*|"${{SLURM_TMPDIR:-/tmp}}"/shared-worker-cache-*)
    if [[ "${{cache_ttl_minutes}}" =~ ^[0-9]+$ && "${{cache_ttl_minutes}}" -gt 0 ]]; then
      find "${{cache_root}}" -mindepth 1 -maxdepth 2 -mmin "+${{cache_ttl_minutes}}" -exec rm -rf -- {{}} + 2>/dev/null || true
    fi
    ;;
  *)
    echo "Skipping shared cache cleanup for unexpected cache root: ${{cache_root}}" >&2
    ;;
esac
mkdir -p "${{cache_root}}/pip" "${{cache_root}}/uv" "${{cache_root}}/npm" "${{cache_root}}/codingworkspace-git"
chmod 700 "${{cache_root}}" "${{cache_root}}/pip" "${{cache_root}}/uv" "${{cache_root}}/npm" "${{cache_root}}/codingworkspace-git" 2>/dev/null || true

export SHARED_WORKER_CACHE_ROOT="${{cache_root}}"
export PIP_CACHE_DIR="${{cache_root}}/pip"
export UV_CACHE_DIR="${{cache_root}}/uv"
export npm_config_cache="${{cache_root}}/npm"
export SHARED_WORKER_GIT_CACHE_ROOT="${{cache_root}}/codingworkspace-git"
export CODINGWORKSPACE_GIT_CACHE_ROOT="${{cache_root}}/codingworkspace-git"

cleanup_scratch_root() {{
  case "${{scratch_root}}" in
    */shared-worker-*) rm -rf "${{scratch_root}}" ;;
  esac
}}
trap cleanup_scratch_root EXIT

echo "Starting shared worker ${{worker_id}} on $(hostname) at $(date -Is)"
.venv/bin/python -m shared_worker_pool worker \\
  --config "${{CONFIG_FILE}}" \\
  --worker-id "${{worker_id}}" \\
  --scratch-root "${{scratch_root}}"
"""


def main() -> int:
    os.environ["PATH"] = (
        f"{Path.home()}/.opencode/bin:/opt/slurm/bin:/opt/slurm-25.11.6/bin:"
        f"/opt/slurm-24.11.5/bin:{os.environ.get('PATH', '')}"
    )
    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch not found; run this on a Slurm submit host such as newcastle.cs.ubc.ca")

    source_dir = Path(_env("SOURCE_DIR", str(Path.cwd()))).expanduser().resolve()
    config_file = Path(_env("CONFIG_FILE", str(source_dir / "config/newcastle_shared_worker_pool.json"))).expanduser().resolve()
    partition = _env("PARTITION", "ada_cpu_short")
    job_count = _nonnegative_int("JOB_COUNT", "1")
    hard_max_job_count = _nonnegative_int("HARD_MAX_JOB_COUNT", "128")
    job_name = _env("JOB_NAME", "shared-worker")
    time_limit = _env("TIME_LIMIT", "02:30:00")
    cpus_per_task = _nonnegative_int("CPUS_PER_TASK", "2")
    memory_mb = _nonnegative_int("MEMORY_MB", "0")
    output_dir = Path(_env("OUTPUT_DIR", str(Path.home() / "shared-worker-pool/logs"))).expanduser().resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"SOURCE_DIR does not exist: {source_dir}")
    if not config_file.is_file():
        raise SystemExit(f"Shared worker pool config file not found: {config_file}")
    if hard_max_job_count and job_count > hard_max_job_count:
        print(f"Capping shared worker JOB_COUNT from {job_count} to {hard_max_job_count}", file=sys.stderr)
        job_count = hard_max_job_count

    output_dir.mkdir(parents=True, exist_ok=True)
    script = build_job_script(
        source_dir=source_dir,
        config_file=config_file,
        job_name=job_name,
        partition=partition,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        memory_mb=memory_mb,
        output_dir=output_dir,
        install_if_missing=_env("INSTALL_IF_MISSING", "1"),
        poll_seconds=_env("POLL_SECONDS", "3"),
        max_runtime_seconds=_env("MAX_RUNTIME_SECONDS", "0"),
        idle_timeout_seconds=_env("IDLE_TIMEOUT_SECONDS", "300"),
        task_timeout_seconds=_env("TASK_TIMEOUT_SECONDS", "0"),
        cache_ttl_seconds=_env("CACHE_TTL_SECONDS", "86400"),
    )

    job_file = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, prefix="shared-slurm-worker.", dir=os.environ.get("TMPDIR") or "/tmp") as file_obj:
            file_obj.write(script)
            job_file = Path(file_obj.name)
        command = ["sbatch", "--parsable"]
        if job_count != 1:
            command.append(f"--array=0-{job_count - 1}")
        command.append(str(job_file))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    finally:
        if job_file is not None:
            job_file.unlink(missing_ok=True)

    print(f"Submitted shared worker job {result.stdout.strip()}")
    print(f"Job name: {job_name}")
    print(f"Partition: {partition}")
    print(f"Count: {job_count}")
    print(f"Config: {config_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
