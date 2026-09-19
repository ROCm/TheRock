# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused contract tests for resource_probe.py."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import resource_probe as probe


SCRIPT = Path(__file__).parent.parent / "resource_probe.py"


def test_parser_helpers():
    assert probe.parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert probe.parse_cpu_list("3-1,bad,7") == [7]

    assert probe.parse_key_values("usage_usec 12\nbad nope\nnr_throttled 3\n") == {
        "usage_usec": 12,
        "nr_throttled": 3,
    }
    assert probe.parse_io_stat(
        "8:0 rbytes=10 wbytes=20 rios=1 wios=2\n"
        "8:16 rbytes=30 wbytes=40 rios=3 wios=4 dbytes=5 dios=6\n"
    ) == {
        "rbytes": 40,
        "wbytes": 60,
        "rios": 4,
        "wios": 6,
        "dbytes": 5,
        "dios": 6,
    }
    assert probe.parse_psi(
        "some avg10=1.25 avg60=0.50 avg300=0.10 total=42\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=7\n"
    ) == {
        "some": {"avg10": 1.25, "avg60": 0.5, "avg300": 0.1, "total": 42},
        "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 7},
    }


def test_cgroup_v2_discovery_and_capacity(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    (proc_root / "self").mkdir(parents=True)
    cgroup_mount = tmp_path / "sys" / "fs" / "cgroup"
    cgroup_dir = cgroup_mount / "actions" / "job"
    cgroup_dir.mkdir(parents=True)

    (proc_root / "self" / "cgroup").write_text("0::/actions/job\n")
    (proc_root / "self" / "mountinfo").write_text(
        f"29 23 0:26 / {cgroup_mount} rw,nosuid,nodev,noexec,relatime"
        " - cgroup2 cgroup rw\n"
    )
    (cgroup_dir / "cgroup.controllers").write_text("cpu io memory\n")
    (cgroup_dir / "cpu.max").write_text("200000 100000\n")
    (cgroup_dir / "cpuset.cpus.effective").write_text("0-3,8\n")
    (cgroup_dir / "memory.max").write_text("1073741824\n")
    (cgroup_dir / "memory.swap.max").write_text("max\n")

    assert probe.discover_cgroup_v2(proc_root) == cgroup_dir

    monkeypatch.setattr(probe.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(
        probe.os,
        "statvfs",
        lambda _path: type(
            "StatVfs",
            (),
            {"f_blocks": 100, "f_frsize": 4096},
        )(),
        raising=False,
    )

    capacity = probe.Collector(
        tmp_path, proc_root=proc_root, cgroup_root=cgroup_dir
    ).capacity()
    assert capacity["os_cpu_count"] == 96
    assert capacity["affinity_cpu_count"] == 8
    assert capacity["cpuset_cpu_count"] == 5
    assert capacity["cgroup_cpu_quota_usec"] == 200000
    assert capacity["cgroup_cpu_period_usec"] == 100000
    assert capacity["effective_cpu_count"] == 2.0
    assert capacity["cgroup_memory_limit_bytes"] == 1073741824
    assert capacity["cgroup_swap_limit_bytes"] is None
    assert capacity["filesystem_total_bytes"] == 409600


def _run_probe(tmp_path: Path, exit_code: int, *, secret: str | None = None):
    output_dir = tmp_path / "telemetry"
    child_program = f"raise SystemExit({exit_code})"
    if secret:
        child_program += f"  # {secret}"
    environment = os.environ.copy()
    if secret:
        environment["RESOURCE_PROBE_TEST_SECRET"] = secret
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "contract-test",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(output_dir),
            "--storage-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            child_program,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=15,
    )
    return completed, output_dir


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
@pytest.mark.parametrize("exit_code", [0, 17])
def test_child_exit_schema_and_summary(tmp_path, exit_code):
    completed, output_dir = _run_probe(tmp_path, exit_code)

    assert completed.returncode == exit_code
    records = [
        json.loads(line)
        for line in (output_dir / "contract-test.samples.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[0]["record_type"] == "metadata"
    assert records[0]["child_process_category"] == "runtime"
    assert any(record["record_type"] == "sample" for record in records)
    sample_records = [record for record in records if record["record_type"] == "sample"]
    assert sample_records[0]["sample_kind"] == "baseline"
    assert sample_records[0]["cpu"]["cores_used"] is None
    assert sample_records[0]["cpu"]["interval_ns"] is None
    assert sample_records[-1]["sample_kind"] == "final"
    assert records[-1]["record_type"] == "summary"
    assert {record["schema"] for record in records} == {probe.SCHEMA}
    assert {record["phase"] for record in records} == {"contract-test"}
    assert [record["sequence"] for record in records] == list(range(len(records)))

    summary = json.loads((output_dir / "contract-test.summary.json").read_text())
    assert summary == records[-1]
    assert summary["schema"] == probe.SCHEMA
    assert summary["record_type"] == "summary"
    assert summary["child_exit_code"] == exit_code
    assert summary["sample_count"] >= 1
    assert summary["probe_overhead"]["self_cpu_ns"] >= 0
    assert summary["probe_overhead"]["jsonl_bytes_before_summary"] > 0


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_telemetry_does_not_disclose_secret_command_or_workspace(tmp_path):
    secret = "DO_NOT_RECORD_7c22bba4"
    completed, output_dir = _run_probe(tmp_path, 0, secret=secret)

    assert completed.returncode == 0
    telemetry = "\n".join(
        path.read_text() for path in sorted(output_dir.glob("*.json*"))
    )
    assert secret not in telemetry
    assert "RESOURCE_PROBE_TEST_SECRET" not in telemetry
    assert str(tmp_path) not in telemetry
    assert child_program_marker(secret) not in telemetry


@pytest.mark.skipif(sys.platform != "linux", reason="prctl test targets Linux CI")
def test_workload_controlled_process_name_cannot_disclose_secret(tmp_path):
    secret = "PSECRET7c22"
    output_dir = tmp_path / "telemetry"
    child_program = (
        "import ctypes,time; "
        "libc=ctypes.CDLL(None); "
        f"libc.prctl(15, {secret.encode()!r}, 0, 0, 0); "
        "time.sleep(3.2)"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "process-name-secret",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(output_dir),
            "--storage-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            child_program,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 0
    telemetry = "\n".join(
        path.read_text() for path in sorted(output_dir.glob("*.json*"))
    )
    assert secret not in telemetry
    assert '"comm"' not in telemetry
    records = [
        json.loads(line)
        for line in (output_dir / "process-name-secret.samples.jsonl")
        .read_text()
        .splitlines()
    ]
    categories = {
        process["category"]
        for record in records
        if record.get("record_type") == "sample"
        and isinstance(record.get("processes"), dict)
        for process in record["processes"]["top"]
    }
    assert "other" in categories


def child_program_marker(secret: str) -> str:
    """Make the raw-argument assertion explicit without repeating CLI assembly."""
    return f"raise SystemExit(0)  # {secret}"


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_missing_command_returns_usage_error_without_outputs(tmp_path):
    output_dir = tmp_path / "telemetry"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "missing-command",
            "--output-dir",
            str(output_dir),
            "--",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 2
    assert "missing command" in completed.stderr
    assert not output_dir.exists()


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_nonexistent_child_returns_127_with_summary(tmp_path):
    output_dir = tmp_path / "telemetry"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "start-failure",
            "--output-dir",
            str(output_dir),
            "--",
            "resource-probe-command-that-does-not-exist",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 127
    summary = json.loads((output_dir / "start-failure.summary.json").read_text())
    assert summary["child_exit_code"] == 127
    metadata = json.loads(
        (output_dir / "start-failure.samples.jsonl").read_text().splitlines()[0]
    )
    assert metadata["child_start_error"] == "FileNotFoundError"


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_sigterm_is_forwarded_and_summarized(tmp_path):
    output_dir = tmp_path / "telemetry"
    child_ready = tmp_path / "child-ready"
    child_program = (
        "from pathlib import Path; import subprocess,time; "
        "descendant=subprocess.Popen(['sleep','30']); "
        f"Path({str(child_ready)!r}).write_text(str(descendant.pid)); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "signal-test",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(output_dir),
            "--storage-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            child_program,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while not child_ready.exists() and time.monotonic() < deadline:
        assert process.poll() is None
        time.sleep(0.05)

    assert child_ready.exists()
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 128 + signal.SIGTERM
    assert stdout == ""
    assert "exit=143" in stderr
    summary = json.loads((output_dir / "signal-test.summary.json").read_text())
    assert summary["child_exit_code"] == 143
    assert summary["child_signal"] == signal.SIGTERM
    assert summary["forwarded_signal"] == signal.SIGTERM
    descendant_pid = int(child_ready.read_text())
    deadline = time.monotonic() + 5
    descendant_state = None
    while time.monotonic() < deadline:
        try:
            stat_text = Path(f"/proc/{descendant_pid}/stat").read_text()
        except FileNotFoundError:
            descendant_state = None
            break
        close = stat_text.rfind(")")
        descendant_state = stat_text[close + 1 :].split()[0]
        if descendant_state == "Z":
            break
        time.sleep(0.05)
    assert descendant_state in (None, "Z")


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_signal_received_during_popen_is_replayed(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 424242

        def __init__(self):
            self.return_code = None

        def poll(self):
            return self.return_code

        def wait(self, timeout=None):
            del timeout
            return self.return_code

    fake_process = FakeProcess()
    forwarded = []

    def popen_with_signal(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return fake_process

    def record_forward(process, signum):
        forwarded.append((process, signum))
        process.return_code = -signum

    monkeypatch.setattr(probe.subprocess, "Popen", popen_with_signal)
    monkeypatch.setattr(probe, "_forward_signal", record_forward)
    args = probe.parse_args(
        [
            "--phase",
            "prelaunch-signal",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(tmp_path / "telemetry"),
            "--storage-path",
            str(tmp_path),
            "--",
            "fake-child",
        ]
    )

    assert probe.run_probe(args) == 128 + signal.SIGTERM
    assert forwarded == [(fake_process, signal.SIGTERM)]
    summary = json.loads(
        (tmp_path / "telemetry" / "prelaunch-signal.summary.json").read_text()
    )
    assert summary["forwarded_signal"] == signal.SIGTERM
    assert summary["child_signal"] == signal.SIGTERM


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_unwritable_output_does_not_change_child_result():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "unwritable-output",
            "--output-dir",
            "/proc/resource-probe-output-is-unwritable",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(17)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 17
    assert "telemetry disabled" in completed.stderr


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
@pytest.mark.parametrize("exit_code", [0, 17])
def test_late_telemetry_io_failure_preserves_child_result(
    tmp_path, monkeypatch, capsys, exit_code
):
    class FailingWriter:
        def __init__(self, _path):
            self.write_count = 0

        def write(self, _record):
            self.write_count += 1
            if self.write_count >= 2:
                raise OSError("simulated full filesystem")

        def close(self):
            raise OSError("simulated close failure")

    monkeypatch.setattr(probe, "JsonlWriter", FailingWriter)
    args = probe.parse_args(
        [
            "--phase",
            "late-io-failure",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(tmp_path / "telemetry"),
            "--storage-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({exit_code})",
        ]
    )

    assert probe.run_probe(args) == exit_code
    assert "telemetry disabled after write failure" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_telemetry_close_failure_preserves_child_result(tmp_path, monkeypatch, capsys):
    class CloseFailingWriter:
        def __init__(self, _path):
            pass

        def write(self, _record):
            pass

        def close(self):
            raise OSError("simulated close failure")

    monkeypatch.setattr(probe, "JsonlWriter", CloseFailingWriter)
    args = probe.parse_args(
        [
            "--phase",
            "close-failure",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(tmp_path / "telemetry"),
            "--storage-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(17)",
        ]
    )

    assert probe.run_probe(args) == 17
    assert "telemetry disabled after close failure" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="resource probe targets Linux CI")
def test_atomic_summary_failure_preserves_child_result(tmp_path, monkeypatch, capsys):
    def fail_summary(*_args, **_kwargs):
        raise OSError("simulated summary failure")

    monkeypatch.setattr(probe, "_write_atomic_json", fail_summary)
    args = probe.parse_args(
        [
            "--phase",
            "summary-failure",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(tmp_path / "telemetry"),
            "--storage-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(17)",
        ]
    )

    assert probe.run_probe(args) == 17
    assert "summary unavailable" in capsys.readouterr().err


def _summary_sample(monotonic_ns, usage_usec, cores_used, interval_ns):
    return {
        "monotonic_ns": monotonic_ns,
        "cpu": {
            "usage_usec": usage_usec,
            "user_usec": usage_usec,
            "system_usec": 0,
            "nr_throttled": 0,
            "throttled_usec": 0,
            "cores_used": cores_used,
            "interval_ns": interval_ns,
        },
        "memory": {
            "current_bytes": 100,
            "cgroup_lifetime_peak_bytes": 500,
            "events": {"oom": 0, "oom_kill": 0, "high": 0, "max": 0},
        },
        "io": {},
        "network": {},
        "psi": {
            "cpu": None,
            "memory": None,
            "io": None,
            "scope": {"cpu": "cgroup", "memory": "host", "io": "cgroup"},
        },
        "filesystem": {"available_bytes": 1000},
        "processes": None,
    }


def _make_summary(samples):
    return probe._summary(
        phase="accounting-test",
        sequence=4,
        start_utc="2026-01-01T00:00:00Z",
        start_ns=1_000_000_000,
        end_ns=3_500_000_000,
        samples=samples,
        child_code=0,
        child_signal=None,
        forwarded_signal=None,
        collection_durations=[1, 1, 1],
        missed_deadlines=0,
        collector_errors=0,
        self_cpu_ns=1,
        self_peak_rss_bytes=1,
        jsonl_bytes_before_summary=1,
    )


def test_summary_cpu_average_uses_phase_counter_delta_and_scopes_metrics():
    samples = [
        _summary_sample(1_000_000_000, 1_000_000, None, None),
        _summary_sample(3_000_000_000, 5_000_000, 2.0, 2_000_000_000),
        _summary_sample(3_500_000_000, 5_250_000, 0.5, 500_000_000),
    ]

    summary = _make_summary(samples)

    assert summary["cpu"]["average_cores_used"] == pytest.approx(1.7)
    assert summary["cpu"]["average_cores_used_source"] == "cgroup_usage_delta"
    assert summary["memory"]["cgroup_lifetime_peak_bytes"] == 500
    assert "cgroup_peak_bytes" not in summary["memory"]
    assert summary["psi"]["cpu"]["scope"] == "cgroup"
    assert summary["psi"]["memory"]["scope"] == "host"


def test_summary_cpu_average_has_duration_weighted_fallback():
    samples = [
        _summary_sample(1_000_000_000, None, None, None),
        _summary_sample(3_000_000_000, 5_000_000, 2.0, 2_000_000_000),
        _summary_sample(3_500_000_000, None, 0.5, 500_000_000),
    ]

    summary = _make_summary(samples)

    assert summary["cpu"]["average_cores_used"] == pytest.approx(1.7)
    assert (
        summary["cpu"]["average_cores_used_source"]
        == "interval_weighted_samples"
    )


def test_baseline_collection_precedes_child_launch(tmp_path, monkeypatch):
    events = []

    class FakeCollector:
        page_size = 4096
        cgroup_root = Path("/fake-cgroup")

        def __init__(self, _storage_path):
            pass

        def capacity(self):
            return {}

        def capabilities(self):
            return {}

        def sample(self, now_ns, child_pid, include_processes):
            events.append(("sample", child_pid, include_processes))
            sample_number = sum(event[0] == "sample" for event in events)
            return {
                "cpu": {
                    "usage_usec": sample_number,
                    "user_usec": sample_number,
                    "system_usec": 0,
                    "nr_throttled": 0,
                    "throttled_usec": 0,
                    "cores_used": None if sample_number == 1 else 0.0,
                    "interval_ns": None if sample_number == 1 else 1,
                },
                "memory": {
                    "current_bytes": 0,
                    "cgroup_lifetime_peak_bytes": 0,
                    "events": {},
                },
                "io": {},
                "network": {},
                "psi": {
                    "cpu": None,
                    "memory": None,
                    "io": None,
                    "scope": {"cpu": None, "memory": None, "io": None},
                },
                "filesystem": {"available_bytes": 0},
                "processes": None,
            }

    class FakeProcess:
        pid = 12345

        def poll(self):
            return 0

    def fake_popen(*_args, **_kwargs):
        events.append(("popen", None, None))
        return FakeProcess()

    monkeypatch.setattr(probe, "Collector", FakeCollector)
    monkeypatch.setattr(probe.subprocess, "Popen", fake_popen)
    args = probe.parse_args(
        [
            "--phase",
            "lifecycle-test",
            "--interval-seconds",
            "1",
            "--output-dir",
            str(tmp_path / "telemetry"),
            "--storage-path",
            str(tmp_path),
            "--",
            "fake-child",
        ]
    )

    assert probe.run_probe(args) == 0
    assert events[0] == ("sample", None, False)
    assert events[1][0] == "popen"
