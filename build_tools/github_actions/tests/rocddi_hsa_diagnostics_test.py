# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only tests for diagnostic isolation, retention and reservation cleanup."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rocddi_hsa_diagnostics as diagnostics
from workflow_utils import WORKFLOWS_DIR, load_workflow


def candidate_package(path):
    candidate = path / "candidate"
    (candidate / "lib").mkdir(parents=True)
    image = candidate / "lib/libhsa-runtime64.so.1"
    image.write_bytes(b"host-only fixture; never loaded")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (candidate / "HSA.sha256").write_text(f"{digest}  lib/libhsa-runtime64.so.1\n")
    (candidate / "SOURCE.json").write_text(
        json.dumps({"rocm_systems_commit": "b" * 40})
    )
    return candidate


def test_candidate_identity_uses_the_selected_manifest(tmp_path):
    candidate = candidate_package(tmp_path)
    assert (
        diagnostics.selected_image(candidate)
        == (candidate / "lib/libhsa-runtime64.so.1").resolve()
    )
    (candidate / "lib/libhsa-runtime64.so.1").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        diagnostics.selected_image(candidate)


def test_failed_shutdown_retains_runtime_and_callbacks():
    hsa = object.__new__(diagnostics.NativeHsa)
    hsa.initialized = True
    hsa.callbacks = [object()]
    hsa.call = lambda name: diagnostics.HSA_ERROR
    with pytest.raises(diagnostics.ShutdownError) as failure:
        hsa.shutdown()
    assert failure.value.owner is hsa
    assert failure.value.owner.callbacks == hsa.callbacks
    assert hsa.initialized


def test_child_signal_keeps_flushed_records_and_stderr(tmp_path):
    log = tmp_path / "child.jsonl"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, signal, sys; "
            "print(json.dumps({'kind': 'call_begin', 'function': 'faulting_touch'}), flush=True); "
            "print('before termination', file=sys.stderr, flush=True); "
            "os.kill(os.getpid(), signal.SIGTERM)"
        ),
    ]
    result = diagnostics.run_child(command, log)
    assert result["signal"] == signal.SIGTERM
    assert result["exit_code"] == -signal.SIGTERM
    assert result["status"] == "incomplete"
    assert result["records"] == [{"kind": "call_begin", "function": "faulting_touch"}]
    assert log.with_suffix(".stderr.txt").read_text() == "before termination\n"


def test_completed_child_retains_native_rejections(tmp_path):
    records = [
        {"kind": "call", "function": "hsa_call", "status": 0x1001},
        {"kind": "worker_result", "status": "unsupported"},
    ]
    result = diagnostics.run_child(
        [
            sys.executable,
            "-c",
            f"print({json.dumps(chr(10).join(json.dumps(row) for row in records))})",
        ],
        tmp_path / "child.jsonl",
    )
    assert result["exit_code"] == 0
    assert result["status"] == "unsupported"
    assert result["nonzero_hsa_statuses"] == records[:1]


def test_native_stdout_without_json_objects_is_retained(tmp_path):
    result = diagnostics.run_child(
        [sys.executable, "-c", "print('native output'); print('123')"],
        tmp_path / "child.jsonl",
    )
    assert result["status"] == "incomplete"
    assert [record["text"] for record in result["records"]] == ["native output", "123"]


def test_collection_keeps_running_after_native_child_failure(tmp_path, monkeypatch):
    candidate = candidate_package(tmp_path)
    calls = []

    def child(command, output):
        partial = json.loads((output.parent / "summary.json").read_text())
        assert partial["children"][-1]["status"] == "running"
        assert partial["children"][-1]["exit_code"] is None
        calls.append(command)
        code = -signal.SIGSEGV if len(calls) == 1 else 0
        return {
            "exit_code": code,
            "signal": -code if code < 0 else None,
            "status": "incomplete" if code else "observed",
            "records": [],
            "log": output.name,
        }

    monkeypatch.setattr(diagnostics, "run_child", child)
    assert diagnostics.collect(candidate, tmp_path / "output") == 1
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["source"]["rocm_systems_commit"] == "b" * 40
    assert summary["qualification"] == "not_assessed"
    assert len(summary["children"]) == 38
    assert summary["children"][0]["signal"] == signal.SIGSEGV
    assert summary["children"][-1]["phase"] == "reservation"
    assert summary["status"] == "errors"
    assert all(
        command[command.index("--candidate") + 1] == str(candidate) for command in calls
    )


def test_invalid_candidate_retains_failure_without_launching_native_code(tmp_path):
    candidate = candidate_package(tmp_path)
    (candidate / "HSA.sha256").write_text("invalid checksum\n")
    assert diagnostics.collect(candidate, tmp_path / "output") == 1
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["children"] == []
    assert "ValueError" in summary["error"]


class ReservationRuntime:
    """A dependency with real writable host storage, never an HSA DSO."""

    def __init__(self, *, reject=False, svm_status=0):
        self.storage = (ctypes.c_ubyte * os.sysconf("SC_PAGE_SIZE"))()
        self.calls = []
        self.reject = reject
        self.svm_status = svm_status
        self.attributes = []

    def bind(self, name, arguments):
        pass

    def agents(self):
        return [{"handle": 12345, "device_type": diagnostics.HSA_DEVICE_TYPE_GPU}]

    def call(self, name, *arguments, **context):
        self.calls.append((name, context))
        if "reserve" in name:
            self.reserve_size = arguments[1]
            if self.reject:
                return diagnostics.HSA_ERROR_INVALID_ARGUMENT
            ctypes.cast(arguments[0], ctypes.POINTER(ctypes.c_void_p))[0] = (
                ctypes.addressof(self.storage)
            )
        if name == "hsa_amd_svm_attributes_set":
            assert self.storage[0] == 0x35 and self.storage[3] == 0xA7
            self.logical_size = arguments[1]
            self.attributes = [(item.attribute, item.value) for item in arguments[2]]
            return self.svm_status
        return (
            diagnostics.HSA_ERROR_INVALID_ARGUMENT
            if context.get("purpose") == "wrong_size"
            else 0
        )


def reservation_case(**values):
    return {
        "flags": diagnostics.HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
        "size": 4,
        "alignment": 0,
        "free": "matching",
        "operation": "reserve",
    } | values


def test_wrong_size_free_is_followed_by_matching_cleanup():
    hsa = ReservationRuntime()
    assert (
        diagnostics.reservation(hsa, reservation_case(free="wrong_size")) == "observed"
    )
    assert hsa.calls[-2:] == [
        (
            "hsa_amd_vmem_address_free",
            {"purpose": "wrong_size", "size": os.sysconf("SC_PAGE_SIZE")},
        ),
        ("hsa_amd_vmem_address_free", {"purpose": "matching_cleanup", "size": 4}),
    ]


def test_subpage_svm_uses_real_enumerated_handle_after_cpu_access_and_cleans_up():
    hsa = ReservationRuntime(svm_status=diagnostics.HSA_ERROR_INVALID_ARGUMENT)
    assert diagnostics.reservation(hsa, reservation_case(operation="svm")) == "observed"
    assert hsa.reserve_size == hsa.logical_size == 4
    assert hsa.attributes == [(diagnostics.HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, 12345)]
    assert hsa.calls[-1] == (
        "hsa_amd_vmem_address_free",
        {"purpose": "matching_cleanup", "size": 4},
    )


def test_rejected_reservation_does_not_touch_memory_or_free():
    hsa = ReservationRuntime(reject=True)
    assert (
        diagnostics.reservation(hsa, reservation_case(operation="cpu_touch"))
        == "observed"
    )
    assert [name for name, _ in hsa.calls] == ["hsa_amd_vmem_address_reserve_align"]


def test_svm_page_control_retains_four_byte_logical_length():
    hsa = ReservationRuntime()
    case = next(
        case
        for case in diagnostics.reservation_cases(os.sysconf("SC_PAGE_SIZE"))
        if case["operation"] == "svm" and case.get("reserve_size")
    )
    assert diagnostics.reservation(hsa, case) == "observed"
    assert hsa.reserve_size == os.sysconf("SC_PAGE_SIZE")
    assert hsa.logical_size == 4


class ClockRuntime:
    """Supply native counter data without loading a runtime or using a device."""

    def __init__(self, status=0, gpu_count=1):
        self.library = self
        self.status = status
        self.gpu_count = gpu_count

    def agents(self):
        return [
            {"handle": 12345 + index, "device_type": diagnostics.HSA_DEVICE_TYPE_GPU}
            for index in range(self.gpu_count)
        ]

    def bind(self, name, arguments):
        pass

    def call(self, name, *arguments, **context):
        ctypes.cast(arguments[-1], ctypes.POINTER(ctypes.c_uint64))[0] = 100_000_000
        return 0

    def hsa_agent_get_info(self, agent, attribute, pointer):
        assert attribute == diagnostics.HSA_AMD_AGENT_INFO_CLOCK_COUNTERS
        counters = ctypes.cast(pointer, ctypes.POINTER(diagnostics.Counters)).contents
        counters.gpu_clock_counter = 11
        counters.cpu_clock_counter = 22
        counters.system_clock_counter = 33
        counters.system_clock_frequency = 44
        return self.status


def test_clock_batches_preserve_all_counters_and_raw_brackets(monkeypatch, capsys):
    current = [0]

    def clock(_):
        current[0] += 1
        return current[0]

    def sleep(seconds):
        current[0] += round(seconds * 1_000_000_000)

    monkeypatch.setattr(diagnostics.time, "clock_gettime_ns", clock)
    monkeypatch.setattr(diagnostics.time, "sleep", sleep)
    assert diagnostics.clocks(ClockRuntime(gpu_count=2)) == "observed"
    samples = [
        record
        for line in capsys.readouterr().out.splitlines()
        if (record := json.loads(line))["kind"] == "clock_sample"
    ]
    assert len(samples) == 2 * (128 + 21)
    assert {row["gpu_index"] for row in samples} == {0, 1}
    assert all(row["host_raw_before_ns"] < row["host_raw_after_ns"] for row in samples)
    assert all(
        row["native"]
        == {
            "gpu_clock_counter": 11,
            "cpu_clock_counter": 22,
            "system_clock_counter": 33,
            "system_clock_frequency": 44,
        }
        for row in samples
    )
    span = [row for row in samples if row["batch"] == "span" and row["gpu_index"] == 0]
    assert (
        span[-1]["host_raw_after_ns"] - span[0]["host_raw_before_ns"] >= 2_000_000_000
    )


def test_optional_clock_attribute_rejection_is_explicit(capsys):
    assert (
        diagnostics.clocks(ClockRuntime(status=diagnostics.HSA_ERROR_INVALID_ARGUMENT))
        == "unsupported"
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(
        row.get("status") == "unsupported"
        and row.get("hsa_status") == diagnostics.HSA_ERROR_INVALID_ARGUMENT
        for row in records
    )
    assert (
        next(row for row in records if row["kind"] == "clock_sample")["native"] is None
    )


def test_no_gpu_records_unsupported_without_clock_queries(capsys):
    assert diagnostics.clocks(ClockRuntime(gpu_count=0)) == "unsupported"
    assert json.loads(capsys.readouterr().out)["reason"] == "no enumerated GPU"


def test_diagnostics_follow_unchanged_hip_tests_and_require_verified_candidate():
    workflow = load_workflow(WORKFLOWS_DIR / "rocddi_dev_test.yml")
    steps = workflow["jobs"]["hip"]["steps"]
    hip = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run selected upstream HIP tests"
    )
    probe = steps[hip + 1]
    assert probe["name"] == "Capture candidate HSA observations"
    assert (
        probe["if"]
        == "${{ !cancelled() && steps.verify_candidate.outcome == 'success' }}"
    )
    assert probe["continue-on-error"] is True
    assert probe["timeout-minutes"] == 10
    assert not steps[hip].get("continue-on-error", False)
    upload = steps[hip + 2]
    assert upload["if"] == "${{ always() }}"
    assert "diagnostics/" in upload["with"]["path"]
