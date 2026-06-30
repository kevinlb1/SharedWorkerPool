from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from shared_worker_pool import runtime as runtime_module
from shared_worker_pool import (
    POOL_WORKER_VERSION,
    AppNeed,
    PoolAppProfile,
    PoolConfig,
    PoolScaleState,
    app_need_from_status,
    candidate_apps,
    desired_pool_submissions,
    dispatch_required_worker_version,
    host_load_cpu_basis,
    host_load_is_high,
    pool_worker_should_retire,
    resource_admission,
    write_dispatch_state,
)


def _config(root: Path, apps: tuple[PoolAppProfile, ...]) -> PoolConfig:
    return PoolConfig(
        config_path=root / "pool.json",
        apps=apps,
        launcher_id="test-launcher",
        partition="ada_cpu_long",
        job_name="shared-worker-auto",
        submit_script=root / "submit.sh",
        source_dir=root,
        slurm_user="kevinlb",
        max_jobs=32,
        max_submit_per_cycle=8,
        min_idle_workers=0,
        poll_seconds=10,
        launcher_active_poll_seconds=3,
        launcher_idle_poll_seconds=15,
        launcher_idle_backoff_after_seconds=120,
        worker_poll_seconds=3,
        worker_idle_timeout_seconds=300,
        worker_max_runtime_seconds=0,
        worker_task_timeout_seconds=0,
        dispatch_state_path=root / "dispatch_state.json",
        dispatch_state_ttl_seconds=20,
        output_dir=root / "logs",
        time_limit="12:00:00",
        cpus_per_task=2,
        memory_mb=6000,
        auto_pull=False,
    )


class SharedWorkerPoolTests(unittest.TestCase):
    def test_host_load_threshold_uses_visible_host_cpus(self) -> None:
        self.assertEqual(host_load_cpu_basis(allocated_cpus=2, visible_cpus=32), 32)
        self.assertFalse(
            host_load_is_high(
                24.0,
                allocated_cpus=2,
                visible_cpus=32,
                max_load_per_cpu=3.0,
            )
        )
        self.assertTrue(
            host_load_is_high(
                97.0,
                allocated_cpus=2,
                visible_cpus=32,
                max_load_per_cpu=3.0,
            )
        )

    def test_host_load_threshold_never_shrinks_below_allocation(self) -> None:
        self.assertEqual(host_load_cpu_basis(allocated_cpus=64, visible_cpus=32), 64)
        self.assertFalse(
            host_load_is_high(
                95.0,
                allocated_cpus=64,
                visible_cpus=32,
                max_load_per_cpu=1.5,
            )
        )
        self.assertTrue(
            host_load_is_high(
                96.0,
                allocated_cpus=64,
                visible_cpus=32,
                max_load_per_cpu=1.5,
            )
        )

    def test_resource_admission_preserves_capacity_when_full(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runtime_module, "free_disk_bytes", return_value=100 * 1024 * 1024 * 1024), \
                patch.object(runtime_module, "meminfo_bytes", return_value={"MemAvailable": 96 * 1024 * 1024 * 1024}), \
                patch.object(runtime_module, "process_forest_resources", return_value={"rssBytes": 1024, "cpuTicks": 0, "processCount": 1}), \
                patch.object(runtime_module, "current_load_average", return_value=(1.0, 1.0, 1.0)):
                admission = resource_admission(
                    scratch_root=root,
                    configured_capacity=2,
                    active_work=2,
                    allocated_cpus=16,
                    allocated_memory_mb=48000,
                    max_load_per_cpu=3.0,
                    min_free_memory_mb=4096.0,
                    memory_reserve_per_work_mb=1024.0,
                    min_free_disk_mb=0.0,
                    work_label="turn",
                )

            self.assertFalse(admission.allowed)
            self.assertEqual(admission.advertised_capacity, 2)
            self.assertIn("configured turn capacity", admission.reason)

    def test_resource_admission_can_advertise_zero_capacity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runtime_module, "free_disk_bytes", return_value=100 * 1024 * 1024 * 1024), \
                patch.object(runtime_module, "meminfo_bytes", return_value={"MemAvailable": 128 * 1024 * 1024}), \
                patch.object(runtime_module, "process_forest_resources", return_value={"rssBytes": 0, "cpuTicks": 0, "processCount": 1}), \
                patch.object(runtime_module, "current_load_average", return_value=(0.0, 0.0, 0.0)):
                admission = resource_admission(
                    scratch_root=root,
                    configured_capacity=8,
                    active_work=0,
                    allocated_cpus=16,
                    allocated_memory_mb=48000,
                    max_load_per_cpu=0.0,
                    min_free_memory_mb=4096.0,
                    memory_reserve_per_work_mb=1024.0,
                    min_free_disk_mb=0.0,
                )

            self.assertFalse(admission.allowed)
            self.assertEqual(admission.advertised_capacity, 0)
            self.assertIn("memory headroom", admission.reason)

    def test_resource_admission_uses_visible_host_cpus_for_load(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runtime_module, "free_disk_bytes", return_value=100 * 1024 * 1024 * 1024), \
                patch.object(runtime_module, "meminfo_bytes", return_value={"MemAvailable": 96 * 1024 * 1024 * 1024}), \
                patch.object(runtime_module, "process_forest_resources", return_value={"rssBytes": 0, "cpuTicks": 0, "processCount": 1}), \
                patch.object(runtime_module, "current_load_average", return_value=(24.0, 24.0, 24.0)), \
                patch.object(runtime_module.os, "cpu_count", return_value=32):
                admission = resource_admission(
                    scratch_root=root,
                    configured_capacity=4,
                    active_work=0,
                    allocated_cpus=2,
                    allocated_memory_mb=6000,
                    max_load_per_cpu=3.0,
                    min_free_memory_mb=4096.0,
                    memory_reserve_per_work_mb=1024.0,
                    min_free_disk_mb=0.0,
                )

            self.assertTrue(admission.allowed)
            self.assertEqual(admission.metrics["allocatedCpus"], 2)
            self.assertEqual(admission.metrics["visibleCpus"], 32)
            self.assertEqual(admission.metrics["hostLoadCpuBasis"], 32)

    def test_sums_app_needs_with_capacity(self) -> None:
        needs = [
            AppNeed(name="caida", queued_units=4, running_units=1, worker_capacity=1),
            AppNeed(name="codingworkspace", queued_units=64, running_units=0, worker_capacity=32),
        ]

        self.assertEqual(
            desired_pool_submissions(
                needs,
                live_slurm_jobs=2,
                min_idle_workers=5,
                max_jobs=400,
                max_submit_per_cycle=100,
            ),
            10,
        )

    def test_warm_request_submits_waiting_worker(self) -> None:
        needs = [
            AppNeed(
                name="codingworkspace",
                queued_units=0,
                running_units=0,
                worker_capacity=4,
                warm_requested_workers=1,
            ),
        ]

        self.assertEqual(
            desired_pool_submissions(
                needs,
                live_slurm_jobs=0,
                min_idle_workers=0,
                max_jobs=400,
                max_submit_per_cycle=100,
            ),
            1,
        )

    def test_warm_request_is_spare_capacity_beyond_running_work(self) -> None:
        needs = [
            AppNeed(
                name="codingworkspace",
                queued_units=0,
                running_units=4,
                worker_capacity=4,
                warm_requested_workers=1,
            ),
        ]

        self.assertEqual(
            desired_pool_submissions(
                needs,
                live_slurm_jobs=1,
                min_idle_workers=0,
                max_jobs=400,
                max_submit_per_cycle=100,
            ),
            1,
        )

    def test_reads_generic_app_status_modes(self) -> None:
        caida = PoolAppProfile(
            name="caida",
            control_url="https://example.test/CAIDA-Concierge",
            token_file=Path("/tmp/token"),
            source_dir=Path("/tmp/caida"),
            python="python3",
            worker_module="app.worker",
            status_mode="caida",
            worker_capacity=1,
        )
        codingworkspace = PoolAppProfile(
            name="codingworkspace",
            control_url="https://example.test/CodingWorkspace",
            token_file=Path("/tmp/token"),
            source_dir=Path("/tmp/cw"),
            python="python3",
            worker_module="codingworkspace.worker",
            status_mode="codingworkspace",
            worker_capacity=16,
        )

        self.assertEqual(
            app_need_from_status(
                caida,
                {
                    "clusterWorkersEnabled": True,
                    "jobs": {"queuedUnclaimed": 3, "runningTotal": 2},
                },
            ).needed_workers,
            3,
        )
        self.assertEqual(
            app_need_from_status(
                codingworkspace,
                {
                    "remoteWorkersEnabled": True,
                    "turns": {"queuedUnclaimed": 33, "runningRemote": 4},
                    "warmPool": {"requestedWorkers": 1, "secondsRemaining": 600},
                },
            ).needed_workers,
            3,
        )
        self.assertEqual(
            app_need_from_status(
                codingworkspace,
                {
                    "remoteWorkersEnabled": True,
                    "turns": {"queuedUnclaimed": 0, "runningRemote": 0},
                    "warmPool": {"requestedWorkers": 1, "secondsRemaining": 600},
                },
            ).target_workers,
            1,
        )

    def test_adaptive_scaling_ramps_and_backs_off(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            caida = PoolAppProfile(
                name="caida",
                control_url="https://example.test/CAIDA-Concierge",
                token_file=root / "caida-token",
                source_dir=root,
                python="python3",
                worker_module="app.worker",
                status_mode="caida",
                worker_capacity=4,
            )
            config = _config(root, (caida,))
            object.__setattr__(config, "max_jobs", 160)
            object.__setattr__(config, "max_submit_per_cycle", 64)
            object.__setattr__(config, "adaptive_scaling_enabled", True)
            object.__setattr__(config, "adaptive_start_jobs", 64)
            object.__setattr__(config, "adaptive_start_submit_per_cycle", 32)
            object.__setattr__(config, "adaptive_min_jobs", 32)
            object.__setattr__(config, "adaptive_min_submit_per_cycle", 8)
            object.__setattr__(config, "adaptive_step_jobs", 16)
            object.__setattr__(config, "adaptive_step_submit_per_cycle", 8)
            object.__setattr__(config, "adaptive_recover_cycles", 2)
            object.__setattr__(config, "adaptive_slow_status_seconds", 2.5)
            scale = PoolScaleState(config)

            scale.record_poll(active=True, control_plane_seconds=0.2, timings={"caida": 0.1, "squeue": 0.1})
            scale.record_poll(active=True, control_plane_seconds=0.2, timings={"caida": 0.08, "squeue": 0.12})
            self.assertEqual(scale.effective_max_jobs, 80)
            self.assertEqual(scale.effective_max_submit_per_cycle, 40)

            scale.record_poll(active=True, control_plane_seconds=3.0, timings={"caida": 0.2, "squeue": 2.8})
            self.assertEqual(scale.effective_max_jobs, 40)
            self.assertEqual(scale.effective_max_submit_per_cycle, 20)

    def test_worker_uses_dispatch_state_without_status_polling(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            caida = PoolAppProfile(
                name="caida",
                control_url="https://example.test/CAIDA-Concierge",
                token_file=root / "caida-token",
                source_dir=root,
                python="python3",
                worker_module="app.worker",
                status_mode="caida",
                worker_capacity=1,
            )
            config = _config(root, (caida,))
            write_dispatch_state(
                config,
                needs=[AppNeed(name="caida", queued_units=2, running_units=5, worker_capacity=1)],
                live_slurm_jobs=4,
                desired_submissions=0,
                submitted_jobs=0,
                status="idle",
            )

            with patch("shared_worker_pool.pool.fetch_app_status", side_effect=AssertionError("unexpected HTTP status poll")):
                candidates = candidate_apps(config, worker_id="worker-a")

            self.assertEqual([app.name for app, _need in candidates], ["caida"])
            self.assertEqual(candidates[0][1].queued_units, 2)

    def test_worker_retires_when_dispatch_requires_newer_version(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            caida = PoolAppProfile(
                name="caida",
                control_url="https://example.test/CAIDA-Concierge",
                token_file=root / "caida-token",
                source_dir=root,
                python="python3",
                worker_module="app.worker",
                status_mode="caida",
                worker_capacity=4,
            )
            config = _config(root, (caida,))
            write_dispatch_state(
                config,
                needs=[AppNeed(name="caida", queued_units=1, running_units=0, worker_capacity=4)],
                live_slurm_jobs=1,
                desired_submissions=0,
                submitted_jobs=0,
                status="idle",
            )
            payload = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["required_worker_version"], POOL_WORKER_VERSION)
            self.assertEqual(dispatch_required_worker_version(config), POOL_WORKER_VERSION)
            self.assertFalse(pool_worker_should_retire(config))

            payload["required_worker_version"] = "shared-worker-pool-worker-next"
            config.dispatch_state_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(pool_worker_should_retire(config))


if __name__ == "__main__":
    unittest.main()
