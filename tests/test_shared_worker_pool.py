from __future__ import annotations

import json
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from shared_worker_pool import runtime as runtime_module
from shared_worker_pool import pool as pool_module
from shared_worker_pool import (
    AppNeed,
    PoolAppProfile,
    PoolConfig,
    PoolScaleState,
    WarmLeaseLockTimeout,
    WarmLeaseStateError,
    active_persistent_warm_leases,
    app_need_from_status,
    candidate_apps,
    claim_persistent_warm_app,
    desired_pool_submissions,
    dispatch_required_worker_version,
    dispatch_state_has_activity,
    host_load_cpu_basis,
    host_load_is_high,
    heartbeat_persistent_warm_lease,
    pool_worker_config_version,
    pool_worker_should_retire,
    resource_admission,
    release_persistent_warm_lease,
    read_dispatch_state_needs,
    read_dispatch_state_payload,
    write_dispatch_state,
)
from shared_worker_pool.pool import (
    app_worker_command,
    poll_launcher_once,
    refresh_dispatch_state_from_worker,
    run_persistent_app_worker,
)
from shared_worker_pool.runtime import LauncherRequestError


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
                warm_request_expires_at=2000,
            ),
        ]

        self.assertEqual(
            desired_pool_submissions(
                needs,
                live_slurm_jobs=0,
                min_idle_workers=0,
                max_jobs=400,
                max_submit_per_cycle=100,
                now=1000,
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
                warm_request_expires_at=2000,
            ),
        ]

        self.assertEqual(
            desired_pool_submissions(
                needs,
                live_slurm_jobs=1,
                min_idle_workers=0,
                max_jobs=400,
                max_submit_per_cycle=100,
                now=1000,
            ),
            1,
        )

    def test_expiring_persistent_child_does_not_hide_capacity_needed_by_other_app(self) -> None:
        needs = [
            AppNeed(name="caida", queued_units=4, running_units=0, worker_capacity=4),
            AppNeed(name="codingworkspace", queued_units=0, running_units=0, worker_capacity=4),
        ]

        self.assertEqual(
            desired_pool_submissions(
                needs,
                live_slurm_jobs=1,
                min_idle_workers=0,
                max_jobs=400,
                max_submit_per_cycle=100,
                persistent_warm_tail_workers=1,
            ),
            1,
        )

    def test_persistent_warm_lease_assigns_exactly_one_worker_and_recovers_stale_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                worker_capacity=4,
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            need = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=4,
                warm_requested_workers=1,
                warm_request_expires_at=1005,
            )

            self.assertEqual(
                claim_persistent_warm_app(config, [need], worker_id="worker-a", now=1000),
                app,
            )
            self.assertIsNone(claim_persistent_warm_app(config, [need], worker_id="worker-b", now=1001))
            self.assertEqual(
                [(item["app"], item["worker_id"]) for item in active_persistent_warm_leases(config, now=1001)],
                [("codingworkspace", "worker-a")],
            )
            self.assertTrue(
                heartbeat_persistent_warm_lease(
                    config,
                    app_name=app.name,
                    worker_id="worker-a",
                    now=1010,
                )
            )
            renewed_need = replace(need, warm_request_expires_at=2000)
            self.assertIsNone(claim_persistent_warm_app(config, [renewed_need], worker_id="worker-b", now=1039))
            self.assertEqual(
                claim_persistent_warm_app(config, [renewed_need], worker_id="worker-b", now=1041),
                app,
            )
            release_persistent_warm_lease(
                config,
                app_name=app.name,
                worker_id="worker-b",
                now=1042,
            )
            self.assertEqual(active_persistent_warm_leases(config, now=1042), [])

    def test_concurrent_persistent_warm_claims_select_exactly_one_worker(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            need = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=2000,
            )
            barrier = threading.Barrier(12)

            def claim(index: int) -> str | None:
                barrier.wait(timeout=2)
                claimed = claim_persistent_warm_app(
                    config,
                    [need],
                    worker_id=f"worker-{index}",
                    now=1000,
                )
                return claimed.name if claimed else None

            with ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(claim, range(12)))

            self.assertEqual(results.count(app.name), 1)
            self.assertEqual(len(active_persistent_warm_leases(config, now=1000)), 1)

    def test_persistent_warm_lock_is_nfs_safe_bounded_and_recovers_stale_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = replace(
                _config(root, (app,)),
                warm_lease_lock_timeout_seconds=0.01,
                warm_lease_lock_stale_seconds=30,
            )
            ownership = pool_module._acquire_warm_lease_lock(config)
            with self.assertRaises(WarmLeaseLockTimeout):
                pool_module._acquire_warm_lease_lock(config)
            pool_module._release_warm_lease_lock(ownership)

            lock_path = pool_module.warm_lease_lock_path(config)
            lock_path.mkdir(mode=0o700)
            owner_path = lock_path / "owner.json"
            stale_at = time.time() - 120
            owner_path.write_text(
                json.dumps(
                    {
                        "version": pool_module.WARM_LOCK_OWNER_VERSION,
                        "token": "dead-owner",
                        "created_at": stale_at,
                    }
                ),
                encoding="utf-8",
            )
            owner_path.touch()
            pool_module.os.utime(owner_path, (stale_at, stale_at))
            pool_module.os.utime(lock_path, (stale_at, stale_at))
            need = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=time.time() + 60,
            )
            with patch("shared_worker_pool.pool.fcntl.flock", side_effect=OSError(37, "No locks available")) as flock:
                self.assertEqual(claim_persistent_warm_app(config, [need], worker_id="worker-a"), app)
            flock.assert_not_called()

    def test_old_lock_owner_cannot_release_replacement(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, ())
            old = pool_module._acquire_warm_lease_lock(config)
            moved = root / "old-lock"
            old.path.rename(moved)
            replacement = pool_module._acquire_warm_lease_lock(config)

            pool_module._release_warm_lease_lock(old)
            self.assertTrue(replacement.path.is_dir())
            self.assertEqual(
                json.loads((replacement.path / "owner.json").read_text(encoding="utf-8"))["token"],
                replacement.token,
            )

            pool_module._release_warm_lease_lock(replacement)
            pool_module._remove_recovered_warm_lock(moved)

    def test_stale_recovery_transition_pins_inode_against_owner_release(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                _config(root, ()),
                warm_lease_lock_timeout_seconds=0.5,
                warm_lease_lock_stale_seconds=0.1,
            )
            ownership = pool_module._acquire_warm_lease_lock(config)
            owner_path = ownership.path / "owner.json"
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            stale_at = time.time() - 1
            owner["created_at"] = stale_at
            owner_path.write_text(json.dumps(owner), encoding="utf-8")
            pool_module.os.utime(owner_path, (stale_at, stale_at))
            pool_module.os.utime(ownership.path, (stale_at, stale_at))

            rename_entered = threading.Event()
            allow_rename = threading.Event()
            recovery_errors: list[BaseException] = []
            real_rename = pool_module.os.rename

            def controlled_rename(source: object, destination: object) -> None:
                if Path(source) == ownership.path and threading.current_thread().name == "stale-recovery":
                    rename_entered.set()
                    if not allow_rename.wait(timeout=2):
                        raise TimeoutError("test did not release stale recovery")
                real_rename(source, destination)

            def recover() -> None:
                try:
                    pool_module._recover_stale_warm_lock(config, ownership.path, now=time.time())
                except BaseException as exc:
                    recovery_errors.append(exc)

            with patch.object(pool_module.os, "rename", side_effect=controlled_rename):
                thread = threading.Thread(target=recover, name="stale-recovery")
                thread.start()
                self.assertTrue(rename_entered.wait(timeout=2))
                # Recovery owns the inode-tied transition, so the old owner
                # cannot remove or rename a replacement while releasing.
                pool_module._release_warm_lease_lock(ownership)
                allow_rename.set()
                thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(recovery_errors, [])
            replacement = pool_module._acquire_warm_lease_lock(config)
            self.assertTrue(replacement.path.is_dir())
            pool_module._release_warm_lease_lock(replacement)

    def test_malformed_wrong_version_and_io_lease_state_fail_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            need = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=2000,
            )
            state_path = pool_module.warm_lease_state_path(config)
            for contents in (
                "{not-json",
                json.dumps({"version": 999, "updated_at": 1000, "leases": []}),
                json.dumps(
                    {
                        "version": pool_module.WARM_LEASE_STATE_VERSION,
                        "updated_at": 1000,
                        "leases": [{"app": app.name, "worker_id": "worker-a", "heartbeat_at": "bad"}],
                    }
                ),
            ):
                state_path.write_text(contents, encoding="utf-8")
                with self.assertRaises(WarmLeaseStateError):
                    claim_persistent_warm_app(config, [need], worker_id="worker-new", now=1000)
                self.assertEqual(state_path.read_text(encoding="utf-8"), contents)

            state_path.unlink()
            with patch.object(pool_module.os, "open", side_effect=OSError("simulated I/O failure")):
                with self.assertRaises(WarmLeaseStateError):
                    claim_persistent_warm_app(config, [need], worker_id="worker-new", now=1000)

    def test_missing_lease_heartbeat_does_not_exceed_app_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            need = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=2000,
            )
            self.assertIsNotNone(claim_persistent_warm_app(config, [need], worker_id="worker-b", now=1000))
            self.assertFalse(
                heartbeat_persistent_warm_lease(
                    config,
                    app_name=app.name,
                    worker_id="worker-a",
                    now=1001,
                )
            )
            self.assertEqual(
                [lease["worker_id"] for lease in active_persistent_warm_leases(config, now=1001)],
                ["worker-b"],
            )

    def test_missing_lease_heartbeat_rejects_worker_owned_by_another_app(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            apps = tuple(
                PoolAppProfile(
                    name=name,
                    control_url=f"https://example.test/{name}",
                    token_file=root / f"{name}-token",
                    source_dir=root,
                    python="python3",
                    worker_module=f"{name}.worker",
                    status_mode="codingworkspace",
                    persistent_warm_workers=1,
                )
                for name in ("app-a", "app-b")
            )
            config = _config(root, apps)
            need = AppNeed(
                name="app-a",
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=2000,
            )
            self.assertIsNotNone(claim_persistent_warm_app(config, [need], worker_id="worker-a", now=1000))
            self.assertFalse(
                heartbeat_persistent_warm_lease(
                    config,
                    app_name="app-b",
                    worker_id="worker-a",
                    now=1001,
                )
            )
            leases = active_persistent_warm_leases(config, now=1001)
            self.assertEqual([(lease["app"], lease["worker_id"]) for lease in leases], [("app-a", "worker-a")])

    def test_persistent_warm_lease_refuses_expired_or_draining_request(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            expired = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=999,
            )
            draining = replace(expired, warm_request_expires_at=2000, restart_drain_active=True)

            self.assertIsNone(claim_persistent_warm_app(config, [expired], worker_id="worker-a", now=1000))
            self.assertIsNone(claim_persistent_warm_app(config, [draining], worker_id="worker-a", now=1000))
            self.assertFalse(config.warm_lease_state_path and config.warm_lease_state_path.exists())

    def test_live_child_reasserts_lease_after_state_file_loss(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            need = AppNeed(
                name=app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=2000,
            )
            self.assertIsNotNone(claim_persistent_warm_app(config, [need], worker_id="worker-a", now=1000))
            (root / "warm_leases.json").unlink()

            self.assertTrue(
                heartbeat_persistent_warm_lease(
                    config,
                    app_name=app.name,
                    worker_id="worker-a",
                    now=1001,
                )
            )
            leases = active_persistent_warm_leases(config, now=1001)
            self.assertEqual([(lease["worker_id"], lease.get("recovered")) for lease in leases], [("worker-a", True)])

    def test_fresh_old_version_lease_blocks_duplicate_until_graceful_child_exit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                worker_args=("--poll-seconds", "1"),
                persistent_warm_workers=1,
            )
            old_config = _config(root, (old_app,))
            new_config = _config(root, (replace(old_app, worker_args=("--poll-seconds", "2")),))
            need = AppNeed(
                name=old_app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=2000,
            )

            self.assertEqual(
                claim_persistent_warm_app(old_config, [need], worker_id="old-worker", now=1000),
                old_app,
            )
            self.assertNotEqual(pool_worker_config_version(old_config), pool_worker_config_version(new_config))
            self.assertIsNone(
                claim_persistent_warm_app(new_config, [need], worker_id="new-worker", now=1001)
            )
            release_persistent_warm_lease(
                old_config,
                app_name=old_app.name,
                worker_id="old-worker",
                now=1002,
            )
            self.assertIsNotNone(
                claim_persistent_warm_app(new_config, [need], worker_id="new-worker", now=1003)
            )

    def test_persistent_child_command_omits_one_shot_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                worker_args=("--idle-timeout-seconds", "600"),
                persistent_warm_workers=1,
            )

            persistent, _scratch, _env = app_worker_command(
                app,
                pool_worker_id="worker-a",
                scratch_root=root / "scratch",
                once=False,
            )
            one_shot, _scratch, _env = app_worker_command(
                app,
                pool_worker_id="worker-a",
                scratch_root=root / "scratch",
                once=True,
            )

            self.assertNotIn("--once", persistent)
            self.assertIn("--once", one_shot)
            self.assertEqual(persistent[-2:], ["--idle-timeout-seconds", "600"])

    def test_persistent_parent_heartbeats_lease_while_child_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]

            with patch("shared_worker_pool.pool.subprocess.Popen", return_value=process), patch(
                "shared_worker_pool.pool.heartbeat_persistent_warm_lease",
                return_value=True,
            ) as heartbeat, patch("shared_worker_pool.pool.release_persistent_warm_lease") as release:
                code = run_persistent_app_worker(
                    config,
                    app,
                    pool_worker_id="worker-a",
                    scratch_root=root / "scratch",
                )

            self.assertEqual(code, 0)
            heartbeat.assert_called_once_with(config, app_name=app.name, worker_id="worker-a")
            release.assert_called_once_with(config, app_name=app.name, worker_id="worker-a")

    def test_persistent_parent_gracefully_drains_once_after_heartbeat_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = replace(_config(root, (app,)), warm_lease_max_heartbeat_failures=3)
            process = Mock()
            process.wait.side_effect = [
                subprocess.TimeoutExpired("worker", 5),
                subprocess.TimeoutExpired("worker", 5),
                subprocess.TimeoutExpired("worker", 5),
                subprocess.TimeoutExpired("worker", 5),
                0,
            ]

            with patch("shared_worker_pool.pool.subprocess.Popen", return_value=process), patch(
                "shared_worker_pool.pool.heartbeat_persistent_warm_lease",
                side_effect=WarmLeaseStateError("unavailable"),
            ) as heartbeat, patch("shared_worker_pool.pool.release_persistent_warm_lease"):
                code = run_persistent_app_worker(
                    config,
                    app,
                    pool_worker_id="worker-a",
                    scratch_root=root / "scratch",
                )

            self.assertEqual(code, 0)
            self.assertEqual(heartbeat.call_count, 3)
            process.send_signal.assert_called_once_with(signal.SIGUSR1)
            process.terminate.assert_not_called()
            process.kill.assert_not_called()

    def test_persistent_parent_gracefully_drains_immediately_after_lease_loss(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]

            with patch("shared_worker_pool.pool.subprocess.Popen", return_value=process), patch(
                "shared_worker_pool.pool.heartbeat_persistent_warm_lease",
                return_value=False,
            ) as heartbeat, patch("shared_worker_pool.pool.release_persistent_warm_lease"):
                code = run_persistent_app_worker(
                    config,
                    app,
                    pool_worker_id="worker-a",
                    scratch_root=root / "scratch",
                )

            self.assertEqual(code, 0)
            heartbeat.assert_called_once_with(config, app_name=app.name, worker_id="worker-a")
            process.send_signal.assert_called_once_with(signal.SIGUSR1)
            process.terminate.assert_not_called()

    def test_persistent_parent_retries_failed_drain_signal_without_killing_child(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            process = Mock()
            process.wait.side_effect = [
                subprocess.TimeoutExpired("worker", 5),
                subprocess.TimeoutExpired("worker", 5),
                0,
            ]
            process.send_signal.side_effect = [OSError("temporary signal failure"), None]

            with patch("shared_worker_pool.pool.subprocess.Popen", return_value=process), patch(
                "shared_worker_pool.pool.heartbeat_persistent_warm_lease",
                return_value=False,
            ) as heartbeat, patch("shared_worker_pool.pool.release_persistent_warm_lease"):
                code = run_persistent_app_worker(
                    config,
                    app,
                    pool_worker_id="worker-a",
                    scratch_root=root / "scratch",
                )

            self.assertEqual(code, 0)
            self.assertEqual(heartbeat.call_count, 1)
            self.assertEqual(process.send_signal.call_args_list, [unittest.mock.call(signal.SIGUSR1)] * 2)
            process.terminate.assert_not_called()

    def test_persistent_parent_cleans_up_child_on_supervisor_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            config = _config(root, (app,))
            process = Mock()
            process.wait.side_effect = [RuntimeError("supervisor failed"), 0]
            process.poll.return_value = None

            with patch("shared_worker_pool.pool.subprocess.Popen", return_value=process), patch(
                "shared_worker_pool.pool.release_persistent_warm_lease"
            ):
                with self.assertRaisesRegex(RuntimeError, "supervisor failed"):
                    run_persistent_app_worker(
                        config,
                        app,
                        pool_worker_id="worker-a",
                        scratch_root=root / "scratch",
                    )

            process.terminate.assert_called_once_with()

    def test_expired_warm_request_does_not_keep_launcher_in_active_poll_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
            )
            config = _config(root, (app,))
            with patch("shared_worker_pool.pool.time.time", return_value=1000):
                write_dispatch_state(
                    config,
                    needs=[
                        AppNeed(
                            name=app.name,
                            queued_units=0,
                            running_units=0,
                            worker_capacity=1,
                            warm_requested_workers=1,
                            warm_request_expires_at=999,
                        )
                    ],
                    live_slurm_jobs=1,
                    desired_submissions=0,
                    submitted_jobs=0,
                    status="idle",
                )
                self.assertFalse(dispatch_state_has_activity(config))

    def test_expired_warm_request_is_filtered_from_submission_demand(self) -> None:
        need = AppNeed(
            name="codingworkspace",
            queued_units=0,
            running_units=0,
            worker_capacity=1,
            warm_requested_workers=1,
            warm_request_expires_at=999,
        )
        self.assertEqual(
            desired_pool_submissions(
                [need],
                live_slurm_jobs=0,
                min_idle_workers=0,
                max_jobs=32,
                max_submit_per_cycle=8,
                now=1000,
            ),
            0,
        )

    def test_status_requires_finite_positive_warm_seconds(self) -> None:
        app = PoolAppProfile(
            name="codingworkspace",
            control_url="https://example.test/CodingWorkspace",
            token_file=Path("/tmp/token"),
            source_dir=Path("/tmp/cw"),
            python="python3",
            worker_module="codingworkspace.worker",
            status_mode="codingworkspace",
            persistent_warm_workers=1,
        )
        for seconds in (0, -1, float("nan"), float("inf"), "bad"):
            need = app_need_from_status(
                app,
                {
                    "remoteWorkersEnabled": True,
                    "turns": {},
                    "warmPool": {"requestedWorkers": 1, "secondsRemaining": seconds},
                },
            )
            self.assertEqual(need.warm_requested_workers, 0)
            self.assertFalse(need.warm_request_is_active())

    def test_malformed_dispatch_expiry_is_rejected_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
            )
            config = _config(root, (app,))
            write_dispatch_state(
                config,
                needs=[
                    AppNeed(
                        name=app.name,
                        queued_units=0,
                        running_units=0,
                        worker_capacity=1,
                        warm_requested_workers=1,
                        warm_request_expires_at=time.time() + 60,
                    )
                ],
                live_slurm_jobs=0,
                desired_submissions=0,
                submitted_jobs=0,
                status="idle",
            )
            original = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
            for expires_at in ("bad", float("nan"), float("inf")):
                payload = dict(original)
                payload["expires_at"] = expires_at
                config.dispatch_state_path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(read_dispatch_state_payload(config))

            original["apps"][0]["warm_request_expires_at"] = "bad"
            config.dispatch_state_path.write_text(json.dumps(original), encoding="utf-8")
            needs = read_dispatch_state_needs(config)
            self.assertIsNotNone(needs)
            assert needs is not None
            self.assertFalse(needs[0].warm_request_is_active())
            config.dispatch_state_path.write_bytes(b"\xff\xfe\xfd")
            self.assertIsNone(read_dispatch_state_payload(config))

    def test_launcher_counts_fresh_lease_after_persistent_app_is_removed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm_app = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "cw-token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                persistent_warm_workers=1,
            )
            old_config = _config(root, (warm_app,))
            now = time.time()
            warm_need = AppNeed(
                name=warm_app.name,
                queued_units=0,
                running_units=0,
                worker_capacity=1,
                warm_requested_workers=1,
                warm_request_expires_at=now + 60,
            )
            self.assertIsNotNone(
                claim_persistent_warm_app(old_config, [warm_need], worker_id="old-worker", now=now)
            )

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
            new_config = _config(root, (caida,))
            with patch(
                "shared_worker_pool.pool.fetch_app_status",
                return_value={
                    "clusterWorkersEnabled": True,
                    "jobs": {"queuedUnclaimed": 4, "runningTotal": 0},
                },
            ), patch("shared_worker_pool.pool.live_slurm_job_count", return_value=1):
                count = poll_launcher_once(new_config, dry_run=True)

            self.assertEqual(count, 1)

            with patch(
                "shared_worker_pool.pool.fetch_app_status",
                return_value={
                    "clusterWorkersEnabled": True,
                    "jobs": {"queuedUnclaimed": 0, "runningTotal": 0},
                },
            ), patch("shared_worker_pool.pool.live_slurm_job_count", return_value=0):
                self.assertEqual(poll_launcher_once(new_config, dry_run=True), 0)

    def test_launcher_lease_state_failure_uses_fail_closed_tail_accounting(self) -> None:
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
            codingworkspace = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "cw-token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                worker_capacity=4,
                persistent_warm_workers=1,
            )
            config = _config(root, (caida, codingworkspace))
            state_path = pool_module.warm_lease_state_path(config)
            state_path.write_text("{malformed", encoding="utf-8")

            def status(app: PoolAppProfile, *, launcher_id: str) -> dict[str, object]:
                if app.name == "caida":
                    return {
                        "clusterWorkersEnabled": True,
                        "jobs": {"queuedUnclaimed": 4, "runningTotal": 0},
                    }
                return {
                    "remoteWorkersEnabled": True,
                    "turns": {"queuedUnclaimed": 0, "runningTotal": 0},
                    "warmPool": {"requestedWorkers": 1, "secondsRemaining": 600},
                }

            with patch("shared_worker_pool.pool.fetch_app_status", side_effect=status), patch(
                "shared_worker_pool.pool.live_slurm_job_count",
                return_value=1,
            ):
                count = poll_launcher_once(config, dry_run=True)

            self.assertEqual(count, 1)
            self.assertEqual(state_path.read_text(encoding="utf-8"), "{malformed")

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
            persistent_warm_workers=1,
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
        with patch("shared_worker_pool.pool.time.time", return_value=1000):
            warm_need = app_need_from_status(
                codingworkspace,
                {
                    "remoteWorkersEnabled": True,
                    "turns": {"queuedUnclaimed": 0, "runningRemote": 0},
                    "warmPool": {"requestedWorkers": 4, "secondsRemaining": 600},
                },
            )
        self.assertEqual(warm_need.target_workers_at(now=1000), 1)
        self.assertEqual(warm_need.warm_requested_workers, 1)
        self.assertEqual(warm_need.warm_request_expires_at, 1600)

    def test_launcher_skips_unavailable_app_and_submits_for_healthy_app(self) -> None:
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
            codingworkspace = PoolAppProfile(
                name="codingworkspace",
                control_url="https://example.test/CodingWorkspace",
                token_file=root / "cw-token",
                source_dir=root,
                python="python3",
                worker_module="codingworkspace.worker",
                status_mode="codingworkspace",
                worker_capacity=4,
            )
            config = _config(root, (caida, codingworkspace))
            object.__setattr__(config, "adaptive_scaling_enabled", True)
            object.__setattr__(config, "adaptive_start_jobs", 16)
            object.__setattr__(config, "adaptive_start_submit_per_cycle", 8)
            object.__setattr__(config, "adaptive_min_jobs", 4)
            object.__setattr__(config, "adaptive_min_submit_per_cycle", 2)
            scale = PoolScaleState(config)

            def fake_status(app: PoolAppProfile, *, launcher_id: str) -> dict[str, object]:
                if app.name == "codingworkspace":
                    raise LauncherRequestError("coding workspace unavailable", status_code=500)
                return {
                    "clusterWorkersEnabled": True,
                    "jobs": {"queuedUnclaimed": 12, "runningTotal": 0},
                }

            with patch("shared_worker_pool.pool.fetch_app_status", side_effect=fake_status), \
                patch("shared_worker_pool.pool.live_slurm_job_count", return_value=0):
                count = poll_launcher_once(config, dry_run=True, scale=scale)

            self.assertEqual(count, 3)
            self.assertEqual(scale.effective_max_jobs, 16)
            self.assertEqual(scale.effective_max_submit_per_cycle, 8)
            self.assertEqual(scale.last_reason, "healthy 1/3")
            payload = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
            apps = {item["name"]: item for item in payload["apps"]}
            self.assertTrue(apps["caida"]["enabled"])
            self.assertEqual(apps["caida"]["queued_units"], 12)
            self.assertFalse(apps["codingworkspace"]["enabled"])

    def test_launcher_preserves_dispatch_state_when_all_apps_unavailable(self) -> None:
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
                needs=[AppNeed(name="caida", queued_units=8, running_units=4, worker_capacity=4)],
                live_slurm_jobs=2,
                desired_submissions=1,
                submitted_jobs=1,
                status="previous",
            )

            with patch(
                "shared_worker_pool.pool.fetch_app_status",
                side_effect=LauncherRequestError("caida unavailable", status_code=503),
            ), patch("shared_worker_pool.pool.live_slurm_job_count", side_effect=AssertionError("unexpected squeue")):
                with self.assertRaises(LauncherRequestError):
                    poll_launcher_once(config, dry_run=True)

            payload = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "previous")
            self.assertEqual(payload["apps"][0]["queued_units"], 8)

    def test_worker_refresh_preserves_stale_dispatch_when_all_apps_unavailable(self) -> None:
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
                needs=[AppNeed(name="caida", queued_units=8, running_units=4, worker_capacity=4)],
                live_slurm_jobs=2,
                desired_submissions=1,
                submitted_jobs=1,
                status="previous",
            )
            payload = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
            payload["expires_at"] = 0
            config.dispatch_state_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch(
                "shared_worker_pool.pool.fetch_app_status",
                side_effect=LauncherRequestError("caida unavailable", status_code=503),
            ), patch("shared_worker_pool.pool.live_slurm_job_count", side_effect=AssertionError("unexpected squeue")):
                needs = refresh_dispatch_state_from_worker(config, worker_id="worker-a")

            self.assertIsNotNone(needs)
            assert needs is not None
            self.assertEqual(needs[0].queued_units, 8)
            payload = json.loads(config.dispatch_state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "previous")
            self.assertEqual(payload["apps"][0]["queued_units"], 8)

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
            expected_version = pool_worker_config_version(config)
            self.assertEqual(payload["required_worker_version"], expected_version)
            self.assertEqual(dispatch_required_worker_version(config), expected_version)
            self.assertFalse(pool_worker_should_retire(config))

            payload["required_worker_version"] = "shared-worker-pool-worker-next"
            config.dispatch_state_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(pool_worker_should_retire(config))

    def test_worker_version_changes_with_delegated_capacity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = PoolAppProfile(
                name="caida",
                control_url="https://example.test/CAIDA-Concierge",
                token_file=root / "caida-token",
                source_dir=root,
                python="python3",
                worker_module="app.worker",
                status_mode="caida",
                worker_capacity=4,
            )
            old_config = _config(root, (app,))
            new_config = _config(root, (replace(app, worker_capacity=8),))

            self.assertNotEqual(
                pool_worker_config_version(old_config),
                pool_worker_config_version(new_config),
            )

            write_dispatch_state(
                new_config,
                needs=[AppNeed(name="caida", queued_units=1, running_units=0, worker_capacity=8)],
                live_slurm_jobs=1,
                desired_submissions=0,
                submitted_jobs=0,
                status="idle",
            )
            self.assertTrue(pool_worker_should_retire(old_config))
            self.assertFalse(pool_worker_should_retire(new_config))


if __name__ == "__main__":
    unittest.main()
