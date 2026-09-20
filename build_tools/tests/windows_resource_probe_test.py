# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import sys

import pytest

from build_tools import windows_resource_probe as probe


class FakeProvider:
    def __init__(self):
        self.closed = False
        self.tick = 0

    def capabilities(self, profile):
        return {
            "win32_process_tree": {"supported": profile != "basic"},
            "pdh_english_counters": {"supported": profile == "diagnostic"},
            "etw": {"supported": False, "reason": "not_collected_non_admin_probe"},
            "pmu": {"supported": False, "reason": "privilege_and_hardware_dependent"},
            "per_process_network": {
                "supported": False,
                "reason": "not_available_from_non_admin_win32_counters",
            },
        }

    def host_snapshot(self, storage_path, profile):
        self.tick += 1
        return {
            "cpu": {
                "logical_processors": 64,
                "processor_group_count": 2,
                "logical_processors_per_group": [32, 32],
                "numa_node_count": 2,
                "system_idle_100ns": self.tick * 100_000,
                "system_kernel_100ns": self.tick * 300_000,
                "system_user_100ns": self.tick * 200_000,
            },
            "memory": {
                "physical_total_bytes": 128 << 30,
                "physical_available_bytes": 64 << 30,
                "commit_total_bytes": 70 << 30,
                "commit_limit_bytes": 160 << 30,
            },
            "storage": {
                "scope": "selected_volume",
                "status": "available",
                "free_bytes": 400 << 30,
                "total_bytes": 800 << 30,
            },
            "network": {
                "scope": "host",
                "status": "available",
                "received_bytes": 1234,
                "sent_bytes": 5678,
            },
            "pdh": {"processor_queue_length": 3.0} if profile == "diagnostic" else None,
        }

    def process_snapshot(self, root_pid):
        return [
            {
                "identity": f"{root_pid}:1",
                "pid": root_pid,
                "category": "runtime",
                "threads": 2,
                "cpu_100ns": 50,
                "rss_bytes": 1000,
                "private_bytes": 1200,
                "page_faults": 4,
                "read_bytes": 10,
                "write_bytes": 20,
            },
            {
                "identity": f"{root_pid + 1}:2",
                "pid": root_pid + 1,
                "category": "compiler",
                "threads": 4,
                "cpu_100ns": 100,
                "rss_bytes": 2000,
                "private_bytes": 2500,
                "page_faults": 8,
                "read_bytes": 30,
                "write_bytes": 40,
            },
        ]

    def close(self):
        self.closed = True


def test_categorization_strips_exe_and_only_returns_allowlisted_category():
    assert probe.categorize_executable(r"C:\\secret\\clang-cl.EXE") == "compiler"
    assert probe.categorize_executable("unexpected-private-name.exe") == "other"


def test_aggregate_processes_contains_categories_not_names():
    provider = FakeProvider()
    aggregate = probe.aggregate_processes(provider.process_snapshot(100))
    assert aggregate["total"]["processes"] == 2
    assert aggregate["total"]["threads"] == 6
    assert aggregate["categories"]["compiler"]["rss_bytes"] == 2000
    assert "clang" not in json.dumps(aggregate).lower()


def test_fake_provider_end_to_end_and_privacy(tmp_path):
    secret = "do-not-record-this-command-value"
    args = probe.parse_args(
        [
            "--phase",
            "windows-python-build",
            "--detail-profile",
            "diagnostic",
            "--interval-seconds",
            "0.01",
            "--output-dir",
            str(tmp_path),
            "--storage-path",
            r"B:\\",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.03)",
            secret,
        ]
    )
    provider = FakeProvider()
    assert probe.run_probe(args, provider) == 0
    assert provider.closed
    summary = json.loads((tmp_path / "windows-python-build.summary.json").read_text())
    assert summary["platform"] == "windows"
    assert summary["detail_profile"] == "diagnostic"
    assert summary["capabilities"]["etw"]["supported"] is False
    assert summary["peaks"]["rss_bytes"] == 3000
    all_text = "\n".join(path.read_text() for path in tmp_path.iterdir())
    assert secret not in all_text
    assert str(tmp_path) not in all_text
    assert "unexpected-private-name" not in all_text.lower()


def test_basic_profile_omits_process_tree(tmp_path):
    args = probe.parse_args(
        [
            "--phase",
            "windows-package-upload",
            "--detail-profile",
            "basic",
            "--interval-seconds",
            "0.01",
            "--output-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
    )
    assert probe.run_probe(args, FakeProvider()) == 0
    record = json.loads(
        (tmp_path / "windows-package-upload.jsonl").read_text().splitlines()[0]
    )
    assert record["process_tree"] is None


def test_argument_validation():
    try:
        probe.parse_args(
            ["--phase", "contains spaces", "--output-dir", "x", "--", "cmd"]
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("unsafe phase label was accepted")


def test_process_intervals_use_identity_and_delta_counters():
    first = [
        {
            "identity": "42:100",
            "pid": 42,
            "category": "compiler",
            "threads": 3,
            "cpu_100ns": 20_000_000,
            "rss_bytes": 100,
            "private_bytes": 120,
            "page_faults": 5,
            "read_bytes": 10,
            "write_bytes": 20,
        }
    ]
    aggregate, previous = probe.process_interval(first, {}, 2.0)
    assert aggregate["interval"]["total"]["average_cores_used"] == 1.0
    assert aggregate["interval"]["processes_started"] == 1
    second = [dict(first[0], cpu_100ns=30_000_000, page_faults=7, read_bytes=50)]
    aggregate, _ = probe.process_interval(second, previous, 1.0)
    assert aggregate["interval"]["total"]["average_cores_used"] == 1.0
    assert aggregate["interval"]["categories"]["compiler"]["page_faults"] == 2
    assert aggregate["interval"]["categories"]["compiler"]["read_bytes"] == 40


def test_unavailable_process_memory_is_counted_not_reported_as_zero_peak():
    item = {
        "identity": "42:100",
        "pid": 42,
        "category": "other",
        "threads": 1,
        "cpu_100ns": 0,
        "memory_status": "unavailable",
        "rss_bytes": None,
        "private_bytes": None,
        "page_faults": None,
        "read_bytes": 0,
        "write_bytes": 0,
    }
    tree, _ = probe.process_interval([item], {}, 1.0)
    assert tree["total"]["memory_available_processes"] == 0
    assert tree["total"]["memory_unavailable_processes"] == 1
    sample = {
        "host": {"cpu": {"interval": {}}, "storage": {"status": "unavailable"}},
        "process_tree": tree,
    }
    summary = probe.build_summary(
        "test", "process", "start", "end", 1.0, 0, [sample], {}, [1.0], 0
    )
    assert summary["peaks"]["rss_bytes"] is None
    assert summary["peaks"]["private_bytes"] is None


def test_samples_have_baseline_and_final_boundaries(tmp_path):
    args = probe.parse_args(
        [
            "--phase",
            "windows-boundaries",
            "--interval-seconds",
            "0.01",
            "--output-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
    )
    assert probe.run_probe(args, FakeProvider()) == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "windows-boundaries.jsonl").read_text().splitlines()
    ]
    assert records[0]["sample_kind"] == "baseline"
    assert records[-1]["sample_kind"] == "final"
    assert records[0]["process_tree"] is None


def test_unwritable_telemetry_destination_preserves_child_exit(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    args = probe.parse_args(
        [
            "--phase",
            "windows-output-failure",
            "--interval-seconds",
            "0.01",
            "--output-dir",
            str(blocked),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ]
    )
    provider = FakeProvider()
    assert probe.run_probe(args, provider) == 7
    assert provider.closed


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 ABI smoke")
def test_real_windows_provider_abi_smoke():
    provider = probe.WindowsProvider()
    try:
        capabilities = provider.capabilities("diagnostic")
        snapshot = provider.host_snapshot(str(Path.cwd()), "diagnostic")
        processes = provider.process_snapshot(os.getpid())
        assert snapshot["cpu"]["logical_processors"] >= 1
        assert snapshot["memory"]["physical_total_bytes"] > 0
        assert snapshot["storage"]["status"] == "available"
        assert snapshot["network"]["scope"] == "host"
        assert capabilities["pmu"]["supported"] is False
        own = next(item for item in processes if item["pid"] == os.getpid())
        assert own["memory_status"] == "available"
        assert own["rss_bytes"] is not None and own["rss_bytes"] > 0
        assert own["private_bytes"] is not None and own["private_bytes"] > 0
        assert all(item["identity"].split(":", 1)[1] != "0" for item in processes)
    finally:
        provider.close()
