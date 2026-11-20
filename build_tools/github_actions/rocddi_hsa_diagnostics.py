#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Observe the selected candidate HSA image in isolated native processes.

This collector initializes HSA. Its observations complement the upstream HIP
tests and do not establish API conformance or GPU qualification. Every native
process has a flushed JSONL log and an independently retained exit status.
The candidate's SOURCE.json identifies the runtime build; ctypes declarations
mirror its public hsa.h and hsa_ext_amd.h API. SVM captures set access attributes
as CLR does for allocation initialization, without issuing prefetch work.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


HSA_SUCCESS = 0
HSA_ERROR = 0x1000
HSA_ERROR_INVALID_ARGUMENT = 0x1001
HSA_AGENT_INFO_NAME = 0
HSA_AGENT_INFO_DEVICE = 17
HSA_AMD_AGENT_INFO_UUID = 0xA011
HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = 0xA016
HSA_AMD_AGENT_INFO_CLOCK_COUNTERS = 0xA118
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = 3
HSA_DEVICE_TYPE_CPU = 0
HSA_DEVICE_TYPE_GPU = 1
HSA_AMD_VMEM_ADDRESS_NO_REGISTER = 1
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE = 0x200
ENVIRONMENT_KEYS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_XNACK",
    "HSA_ENABLE_SDMA",
    "HSA_ENABLE_HIGH_PRECISION_TIMESTAMPS",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)


class Agent(ctypes.Structure):
    """Public hsa_agent_t handle."""

    _fields_ = [("handle", ctypes.c_uint64)]


class Counters(ctypes.Structure):
    """Public hsa_amd_clock_counters_t, retaining all four counters."""

    _fields_ = [
        ("gpu_clock_counter", ctypes.c_uint64),
        ("cpu_clock_counter", ctypes.c_uint64),
        ("system_clock_counter", ctypes.c_uint64),
        ("system_clock_frequency", ctypes.c_uint64),
    ]


class SvmAttribute(ctypes.Structure):
    """Public hsa_amd_svm_attribute_pair_t."""

    _fields_ = [("attribute", ctypes.c_uint64), ("value", ctypes.c_uint64)]


class DlInfo(ctypes.Structure):
    """Linux dladdr result for identifying resolved native functions."""

    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("base_address", ctypes.c_void_p),
        ("symbol_name", ctypes.c_char_p),
        ("symbol_address", ctypes.c_void_p),
    ]


AGENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint32, Agent, ctypes.c_void_p)
EVENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p)


class ProbeError(RuntimeError):
    """A native diagnostic could not complete."""


class ShutdownError(ProbeError):
    """Keep callbacks and their DSO alive until the CLI terminates the process."""

    def __init__(self, status: int, owner: object):
        super().__init__(f"hsa_shut_down returned {hex(status)}")
        # Failed shutdown does not establish that asynchronous callbacks have quiesced.
        self.owner = owner


def emit(kind: str, **values: object) -> None:
    print(json.dumps({"kind": kind, **values}, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def selected_image(candidate: Path) -> Path:
    image = candidate / "lib/libhsa-runtime64.so.1"
    entries = (candidate / "HSA.sha256").read_text().splitlines()
    if len(entries) != 1:
        raise ValueError("HSA.sha256 must identify exactly one candidate HSA image")
    expected, name = entries[0].split(maxsplit=1)
    if name.lstrip("*") != "lib/libhsa-runtime64.so.1":
        raise ValueError("HSA.sha256 does not identify the candidate HSA image")
    actual = image_identity(image)
    if actual["sha256"] != expected:
        raise ValueError("candidate HSA SHA-256 does not match HSA.sha256")
    return image.resolve(strict=True)


def read_record(path: Path) -> None:
    try:
        emit("file", path=str(path), text=path.read_text())
    except OSError as error:
        emit("file", path=str(path), errno=error.errno, error=str(error))


def child_directories(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir(), key=lambda item: item.name)
    except OSError as error:
        emit("directory", path=str(path), errno=error.errno, error=str(error))
        return []


def capture_topology() -> None:
    emit("credentials", uid=os.getuid(), gid=os.getgid(), groups=os.getgroups())
    for device in [Path("/dev/kfd"), *child_directories(Path("/dev/dri"))]:
        try:
            information = device.stat()
            emit(
                "device_node",
                path=str(device),
                mode=oct(information.st_mode),
                uid=information.st_uid,
                gid=information.st_gid,
                major=os.major(information.st_rdev),
                minor=os.minor(information.st_rdev),
            )
        except OSError as error:
            emit("device_node", path=str(device), errno=error.errno, error=str(error))
    read_record(Path("/proc/self/limits"))
    root = Path("/sys/class/kfd/kfd/topology")
    read_record(root / "generation_id")
    for node in child_directories(root / "nodes"):
        if not node.name.isdecimal():
            continue
        for name in ("gpu_id", "name", "properties"):
            read_record(node / name)
        for collection in ("mem_banks", "caches", "io_links", "p2p_links"):
            for record in child_directories(node / collection):
                if record.name.isdecimal():
                    read_record(record / "properties")
    for render in child_directories(Path("/sys/class/drm")):
        if not render.name.startswith("renderD"):
            continue
        device = render / "device"
        for name in (
            "vendor",
            "device",
            "subsystem_vendor",
            "subsystem_device",
            "revision",
            "mem_info_vram_total",
            "mem_info_vis_vram_total",
        ):
            read_record(device / name)
        for name in ("major", "minor", "revision"):
            read_record(device / "ip_discovery/die/0/42/0" / name)
    read_record(root / "generation_id")


def loaded_runtime_images() -> list[dict]:
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            if "libhsa-runtime64" in fields[5] or "libamdhip64" in fields[5]:
                paths.add(Path(fields[5]))
    return [image_identity(path) for path in sorted(paths)]


class NativeHsa:
    """Own one initialized image and callbacks until synchronous shutdown."""

    def __init__(self, image: Path):
        self.image = image
        self.library = ctypes.CDLL(str(image), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        self.initialized = False
        self.callbacks = []
        process = ctypes.CDLL(None)
        self.dladdr = process.dladdr
        self.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        self.dladdr.restype = ctypes.c_int

    def bind(self, name: str, arguments: list) -> object:
        function = getattr(self.library, name)
        function.argtypes = arguments
        function.restype = ctypes.c_uint32
        information = DlInfo()
        if (
            not self.dladdr(
                ctypes.cast(function, ctypes.c_void_p), ctypes.byref(information)
            )
            or not information.filename
        ):
            raise ProbeError(f"dladdr could not identify {name}")
        path = Path(os.fsdecode(information.filename)).resolve(strict=True)
        emit("symbol", name=name, image=str(path))
        if not path.samefile(self.image):
            raise ProbeError(f"{name} resolved outside the selected HSA image")
        return function

    def call(self, name: str, *arguments: object, **context: object) -> int:
        emit("call_begin", function=name, **context)
        begin = time.monotonic_ns()
        status = int(getattr(self.library, name)(*arguments))
        emit(
            "call",
            function=name,
            status=status,
            status_hex=hex(status),
            duration_ns=time.monotonic_ns() - begin,
            **context,
        )
        return status

    def initialize(self) -> None:
        self.bind("hsa_init", [])
        self.bind("hsa_shut_down", [])
        emit("loaded_images", stage="before_init", images=loaded_runtime_images())
        status = self.call("hsa_init")
        self.initialized = status == HSA_SUCCESS
        emit("loaded_images", stage="after_init", images=loaded_runtime_images())
        if not self.initialized:
            raise ProbeError(f"hsa_init returned {hex(status)}")

    def shutdown(self) -> None:
        if self.initialized:
            status = self.call("hsa_shut_down")
            if status != HSA_SUCCESS:
                raise ShutdownError(status, self)
            self.initialized = False

    def agents(self) -> list[dict]:
        self.bind("hsa_iterate_agents", [AGENT_CALLBACK, ctypes.c_void_p])
        self.bind("hsa_agent_get_info", [Agent, ctypes.c_uint32, ctypes.c_void_p])
        handles = []
        callback_errors = []

        @AGENT_CALLBACK
        def visit(agent: Agent, _: object) -> int:
            try:
                handles.append(int(agent.handle))
                return HSA_SUCCESS
            except BaseException as error:
                callback_errors.append(error)
                return HSA_ERROR

        self.callbacks.append(visit)
        status = self.call("hsa_iterate_agents", visit, None)
        emit(
            "agent_callback",
            status=HSA_ERROR if callback_errors else HSA_SUCCESS,
            errors=[repr(error) for error in callback_errors],
        )
        agents = []
        for handle in handles:
            agent = Agent(handle)
            device = ctypes.c_uint32()
            name = ctypes.create_string_buffer(64)
            uuid = ctypes.create_string_buffer(32)
            type_status = self.call(
                "hsa_agent_get_info",
                agent,
                HSA_AGENT_INFO_DEVICE,
                ctypes.byref(device),
                handle=handle,
                attribute=HSA_AGENT_INFO_DEVICE,
            )
            name_status = self.call(
                "hsa_agent_get_info",
                agent,
                HSA_AGENT_INFO_NAME,
                name,
                handle=handle,
                attribute=HSA_AGENT_INFO_NAME,
            )
            uuid_status = self.call(
                "hsa_agent_get_info",
                agent,
                HSA_AMD_AGENT_INFO_UUID,
                uuid,
                handle=handle,
                attribute=HSA_AMD_AGENT_INFO_UUID,
            )
            agents.append(
                {
                    "handle": handle,
                    "device_type": device.value if type_status == 0 else None,
                    "type_status": type_status,
                    "name_status": name_status,
                    "name": name.value.decode(errors="replace"),
                    "uuid_status": uuid_status,
                    "uuid": uuid.value.decode(errors="replace"),
                }
            )
        emit(
            "agents",
            status=status,
            agents=agents,
            cpu_count=sum(row["device_type"] == HSA_DEVICE_TYPE_CPU for row in agents),
            gpu_count=sum(row["device_type"] == HSA_DEVICE_TYPE_GPU for row in agents),
        )
        if status != HSA_SUCCESS or callback_errors:
            raise ProbeError("agent enumeration failed")
        return agents


def startup(hsa: NativeHsa) -> str:
    agents = hsa.agents()
    try:
        hsa.bind(
            "hsa_amd_register_system_event_handler", [EVENT_CALLBACK, ctypes.c_void_p]
        )
    except AttributeError:
        emit("system_event_handler", status="unsupported", reason="symbol unavailable")
    else:

        @EVENT_CALLBACK
        def event(_: object, __: object) -> int:
            return HSA_ERROR

        hsa.callbacks.append(event)
        hsa.call("hsa_amd_register_system_event_handler", event, None)
    return (
        "observed"
        if any(row["device_type"] == HSA_DEVICE_TYPE_GPU for row in agents)
        else "no_gpu"
    )


def clocks(hsa: NativeHsa) -> str:
    gpus = [row for row in hsa.agents() if row["device_type"] == HSA_DEVICE_TYPE_GPU]
    if not gpus:
        emit("clocks", status="unsupported", reason="no enumerated GPU")
        return "unsupported"
    if not hasattr(time, "CLOCK_MONOTONIC_RAW"):
        emit("clocks", status="unsupported", reason="CLOCK_MONOTONIC_RAW unavailable")
        return "unsupported"
    hsa.bind("hsa_system_get_info", [ctypes.c_uint32, ctypes.c_void_p])
    system_rate = ctypes.c_uint64()
    hsa.call(
        "hsa_system_get_info",
        HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
        ctypes.byref(system_rate),
        attribute=HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
    )
    emit("system_frequency", hz=system_rate.value)
    outcomes = []
    for index, description in enumerate(gpus):
        agent = Agent(description["handle"])
        frequency = ctypes.c_uint64()
        frequency_status = hsa.call(
            "hsa_agent_get_info",
            agent,
            HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
            ctypes.byref(frequency),
            gpu_index=index,
            attribute=HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
        )
        emit(
            "gpu_frequency",
            gpu_index=index,
            status=frequency_status,
            hz=frequency.value,
        )
        # Retain every native counter and raw-host bracket; sampling changes no precision policy.
        for batch, count, spacing_ns in (("short", 128, 0), ("span", 21, 100_000_000)):
            emit(
                "clock_batch",
                gpu_index=index,
                batch=batch,
                count=count,
                spacing_ns=spacing_ns,
            )
            begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            for sequence in range(count):
                remaining = (
                    begin
                    + sequence * spacing_ns
                    - time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                )
                if remaining > 0:
                    time.sleep(remaining / 1_000_000_000)
                counters = Counters()
                before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                status = int(
                    hsa.library.hsa_agent_get_info(
                        agent, HSA_AMD_AGENT_INFO_CLOCK_COUNTERS, ctypes.byref(counters)
                    )
                )
                after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                emit(
                    "clock_sample",
                    gpu_index=index,
                    batch=batch,
                    sequence=sequence,
                    status=status,
                    host_raw_before_ns=before,
                    host_raw_after_ns=after,
                    native=(
                        {
                            name: int(getattr(counters, name))
                            for name, _ in Counters._fields_
                        }
                        if status == HSA_SUCCESS
                        else None
                    ),
                )
                if status != HSA_SUCCESS:
                    emit(
                        "clocks",
                        gpu_index=index,
                        status=(
                            "unsupported"
                            if status == HSA_ERROR_INVALID_ARGUMENT
                            else "error"
                        ),
                        hsa_status=status,
                        reason="clock counter attribute rejected",
                    )
                    break
            if status != HSA_SUCCESS:
                break
        outcomes.append(
            "observed"
            if status == HSA_SUCCESS
            else ("unsupported" if status == HSA_ERROR_INVALID_ARGUMENT else "error")
        )
    return (
        "error"
        if "error" in outcomes
        else ("unsupported" if "unsupported" in outcomes else "observed")
    )


def reservation_cases(page_size: int) -> list[dict]:
    cases = []
    for flags in (0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER):
        for size in (page_size, 4):
            for alignment in ("legacy", 0, page_size):
                for free in ("matching", "wrong_size"):
                    cases.append(
                        {
                            "flags": flags,
                            "size": size,
                            "alignment": alignment,
                            "free": free,
                            "operation": "reserve",
                        }
                    )
    # Touch and SVM probes have separate lifetimes so a CPU fault cannot hide free results.
    for operation in ("cpu_touch", "svm"):
        for size in (page_size, 4):
            for alignment in (0, page_size):
                cases.append(
                    {
                        "flags": HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
                        "size": size,
                        "alignment": alignment,
                        "free": "matching",
                        "operation": operation,
                    }
                )
        for alignment in (0, page_size):
            cases.append(
                {
                    "flags": HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
                    "size": 4,
                    "reserve_size": page_size,
                    "alignment": alignment,
                    "free": "matching",
                    "operation": operation,
                }
            )
    return cases


def reservation(hsa: NativeHsa, case: dict) -> str:
    emit("reservation_case", **case)
    reserve_size = case.get("reserve_size", case["size"])
    try:
        hsa.bind("hsa_amd_vmem_address_free", [ctypes.c_void_p, ctypes.c_size_t])
        if case["alignment"] == "legacy":
            function = "hsa_amd_vmem_address_reserve"
            hsa.bind(
                function,
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                ],
            )
            arguments = [reserve_size, 0, case["flags"]]
        else:
            function = "hsa_amd_vmem_address_reserve_align"
            hsa.bind(
                function,
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                ],
            )
            arguments = [reserve_size, 0, case["alignment"], case["flags"]]
    except AttributeError as error:
        emit("reservation", status="unsupported", error=str(error))
        return "unsupported"
    pointer = ctypes.c_void_p()
    status = hsa.call(function, ctypes.byref(pointer), *arguments)
    emit("reservation", status=status, address=pointer.value)
    if status != HSA_SUCCESS:
        return "observed"
    if not pointer.value:
        raise ProbeError("successful reservation returned a null address")
    try:
        if case["operation"] in ("cpu_touch", "svm"):
            # No mprotect or SVM calls precede the NO_REGISTER CPU access.
            emit("cpu_touch_begin", address=pointer.value, size=case["size"])
            memory = (ctypes.c_ubyte * case["size"]).from_address(pointer.value)
            memory[0], memory[-1] = 0x35, 0xA7
            emit("cpu_touch", first=memory[0], last=memory[-1], status="observed")
        if case["operation"] == "svm":
            gpus = [
                row for row in hsa.agents() if row["device_type"] == HSA_DEVICE_TYPE_GPU
            ]
            if not gpus:
                emit("svm", status="unsupported", reason="no enumerated GPU")
                return "unsupported"
            try:
                hsa.bind(
                    "hsa_amd_svm_attributes_set",
                    [
                        ctypes.c_void_p,
                        ctypes.c_size_t,
                        ctypes.POINTER(SvmAttribute),
                        ctypes.c_size_t,
                    ],
                )
            except AttributeError:
                emit("svm", status="unsupported", reason="symbol unavailable")
                return "unsupported"
            attributes = (SvmAttribute * len(gpus))(
                *[
                    SvmAttribute(HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, row["handle"])
                    for row in gpus
                ]
            )
            emit(
                "svm_attributes",
                attributes=[
                    {"attribute": item.attribute, "value": item.value}
                    for item in attributes
                ],
            )
            hsa.call(
                "hsa_amd_svm_attributes_set",
                pointer,
                case["size"],
                attributes,
                len(attributes),
            )
        if case["free"] == "wrong_size":
            page_size = os.sysconf("SC_PAGE_SIZE")
            wrong_size = (
                page_size if reserve_size < page_size else reserve_size + page_size
            )
            wrong_status = hsa.call(
                "hsa_amd_vmem_address_free",
                pointer,
                wrong_size,
                size=wrong_size,
                purpose="wrong_size",
            )
            if wrong_status == HSA_SUCCESS:
                pointer = ctypes.c_void_p()
    finally:
        if pointer.value:
            status = hsa.call(
                "hsa_amd_vmem_address_free",
                pointer,
                reserve_size,
                size=reserve_size,
                purpose="matching_cleanup",
            )
            if status != HSA_SUCCESS:
                raise ProbeError(f"matching reservation free returned {hex(status)}")
    return "observed"


def native_worker(arguments: argparse.Namespace) -> int:
    # A faulting touch must not leave a potentially large core file in the artifact.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    image = selected_image(arguments.candidate)
    emit("selected_image", **image_identity(image))
    emit(
        "abi_layout",
        agent_bytes=ctypes.sizeof(Agent),
        counter_bytes=ctypes.sizeof(Counters),
        counter_offsets={
            name: getattr(Counters, name).offset for name, _ in Counters._fields_
        },
        svm_attribute_bytes=ctypes.sizeof(SvmAttribute),
    )
    if (
        ctypes.sizeof(Agent) != 8
        or ctypes.sizeof(Counters) != 32
        or ctypes.sizeof(SvmAttribute) != 16
    ):
        raise ProbeError("ctypes layout does not match the public HSA structures")
    hsa = NativeHsa(image)
    try:
        hsa.initialize()
        if arguments.worker == "startup":
            status = startup(hsa)
        elif arguments.worker == "clocks":
            status = clocks(hsa)
        else:
            status = reservation(hsa, json.loads(arguments.case))
    finally:
        hsa.shutdown()
    emit("worker_result", status=status)
    return 1 if status == "error" else 0


def run_child(command: list[str], output: Path) -> dict:
    with output.open("x") as stream, output.with_suffix(".stderr.txt").open(
        "x"
    ) as errors:
        result = subprocess.run(command, stdout=stream, stderr=errors, check=False)
    records = []
    for line in output.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            records.append(
                {"kind": "invalid_record", "text": line, "error": str(error)}
            )
        else:
            records.append(
                record
                if isinstance(record, dict)
                else {
                    "kind": "invalid_record",
                    "text": line,
                    "error": "record is not a JSON object",
                }
            )
    outcome = next(
        (
            row["status"]
            for row in reversed(records)
            if row.get("kind") == "worker_result"
        ),
        "incomplete",
    )
    return {
        "log": output.name,
        "exit_code": result.returncode,
        "signal": -result.returncode if result.returncode < 0 else None,
        "status": outcome,
        "nonzero_hsa_statuses": [
            row
            for row in records
            if row.get("kind") in ("call", "clock_sample")
            and row.get("status") != HSA_SUCCESS
        ],
        "records": records,
    }


def collect(candidate: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "rocddi-hsa-observations-v1",
        "status": "collecting",
        "qualification": "not_assessed",
        "children": [],
    }
    summary_path = output / "summary.json"

    def save_summary() -> None:
        pending = summary_path.with_suffix(".pending")
        pending.write_text(json.dumps(summary, indent=2) + "\n")
        pending.replace(summary_path)

    save_summary()
    try:
        summary["source"] = json.loads((candidate / "SOURCE.json").read_text())
        summary["selected_image"] = image_identity(selected_image(candidate))
        with (output / "host.jsonl").open("x") as stream:
            host = {
                "kind": "host",
                "kernel": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "page_size": os.sysconf("SC_PAGE_SIZE"),
                "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
            }
            stream.write(json.dumps(host, sort_keys=True) + "\n")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--candidate",
            str(candidate),
        ]
        phases = [("startup", None), ("clocks", None)]
        phases.extend(
            ("reservation", case)
            for case in reservation_cases(os.sysconf("SC_PAGE_SIZE"))
        )
        for index, (phase, case) in enumerate(phases):
            child_command = command + ["--worker", phase]
            if case is not None:
                child_command += ["--case", json.dumps(case)]
            log = output / f"{index:02d}-{phase}.jsonl"
            child = {
                "phase": phase,
                "case": case,
                "log": log.name,
                "status": "running",
                "exit_code": None,
                "signal": None,
            }
            summary["children"].append(child)
            save_summary()
            result = run_child(child_command, log)
            records = result.pop("records")
            child.update(result)
            if phase == "startup":
                child["agent_summary"] = [
                    row
                    for row in records
                    if row.get("kind") in ("agents", "agent_callback")
                ]
            save_summary()
        summary["status"] = (
            "errors"
            if any(
                row["exit_code"] != 0 or row["status"] in ("error", "incomplete")
                for row in summary["children"]
            )
            else "observed"
        )
    except (OSError, ValueError) as error:
        summary["status"] = "error"
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        save_summary()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "observed" else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="verified HIP candidate SDK directory",
    )
    parser.add_argument(
        "--output", type=Path, help="new directory for retained diagnostic records"
    )
    parser.add_argument(
        "--worker", choices=("startup", "clocks", "reservation"), help=argparse.SUPPRESS
    )
    parser.add_argument("--case", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    arguments.candidate = arguments.candidate.resolve(strict=True)
    if arguments.worker:
        if arguments.worker == "startup":
            capture_topology()
        return native_worker(arguments)
    if arguments.output is None:
        parser.error("--output is required")
    return collect(arguments.candidate, arguments.output)


if __name__ == "__main__":
    try:
        result = main(sys.argv[1:])
    except ShutdownError as error:
        emit("worker_result", status="error", error=str(error))
        print(str(error), file=sys.stderr, flush=True)
        # The exception retains the callbacks and DSO; skip their destruction
        # because failed shutdown cannot guarantee callback quiescence.
        os._exit(1)
    sys.exit(result)
