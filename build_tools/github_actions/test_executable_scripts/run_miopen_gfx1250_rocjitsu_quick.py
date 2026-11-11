#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run only the installed MIOpen FFM quick suite on gfx1250 under rocjitsu.

Consumes TheRock's fetched prefix (OUTPUT_ARTIFACTS_DIR, THEROCK_BIN_DIR); never
builds MIOpen or CK. TEST_TYPE does not alter selection. The only local build is
the pinned rocjitsu host tool, with at most eight build workers. For a local
reproduction, pass both --rocjitsu-bin and --rocjitsu-config to reuse a real build.
"""

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROCJITSU_REPO = "https://github.com/ROCm/rocm-systems.git"


def run(cmd, **kw):
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def ensure_build_tools():
    """Install cmake/Ninja/build-essential/libdrm-dev/git if missing.

    Uses sudo rather than assuming the container runs as root: TheRock's
    no_rocm_image_ubuntu24_04 (and its variants) run as an unprivileged
    `tester` user with passwordless sudo configured, and some CI runner
    pools don't honor a container's --user 0:0 override.
    """
    if (
        all(shutil.which(t) for t in ("cmake", "ninja", "c++", "git"))
        and Path("/usr/include/libdrm").exists()
    ):
        return
    run(["sudo", "apt-get", "update"])
    run(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake",
            "ninja-build",
            "build-essential",
            "libdrm-dev",
            "git",
        ]
    )


def build_rocjitsu(workdir: Path, ref: str, jobs: int = None):
    """Sparse-clone rocjitsu @ ref and build the rocjitsu CLI launcher.

    The launcher (not a raw LD_PRELOAD of librocjitsu.so) is the supported entry point: it stands
    up the simulated KFD and injects the interposer into the child process it execs.

    Returns (rocjitsu_bin_path, rocjitsu_source_dir)."""
    src = workdir / "rocm-systems"
    if not src.exists():
        run(["git", "init", str(src)])
        run(["git", "-C", str(src), "remote", "add", "origin", ROCJITSU_REPO])
        run(["git", "-C", str(src), "config", "core.sparseCheckout", "true"])
        (src / ".git" / "info" / "sparse-checkout").write_text(
            "emulation/rocjitsu\nruntimes/api-headers\nshared/machine-readable-isa/isa\n"
        )
        run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", ref])
        run(["git", "-C", str(src), "checkout", "FETCH_HEAD"])
    build = workdir / "rocjitsu-build-ninja"
    run(
        [
            "cmake",
            "-S",
            str(src / "emulation" / "rocjitsu"),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            f"-DFETCHCONTENT_BASE_DIR={build / 'third_party'}",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "rocjitsu_bin",
            "rocjitsu_shared",
            "-j",
            str(jobs if jobs is not None else (os.cpu_count() or 4)),
        ]
    )
    rocjitsu_bin = build / "tools" / "rocjitsu" / "rocjitsu"
    if not rocjitsu_bin.exists():
        sys.exit(
            f"::error::rocjitsu CLI launcher not found at {rocjitsu_bin} after build"
        )
    return rocjitsu_bin, src


ROCJITSU_REF = "6f5d9658a4b13656e0c1a3719d86a320c9206dbe"
ROCJITSU_CONFIG = "gfx1250_mi455x_kmd_4gpu.json"
SUITE = "miopen_gtest_ffm-quick_suite"
TIMEOUT_SECONDS = 5400

# Probe the actual packaged HSA runtime in a child of the same launcher/config
# as the suite. Config text alone cannot prove the device exposed by the runtime.
# HSA_AGENT_INFO_NAME = 0, HSA_AGENT_INFO_DEVICE = 17, HSA_DEVICE_TYPE_GPU = 1.
RUNTIME_PROBE = r"""
import ctypes
import sys

class Agent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]

hsa = ctypes.CDLL(sys.argv[1])
callback_type = ctypes.CFUNCTYPE(ctypes.c_uint32, Agent, ctypes.c_void_p)
hsa.hsa_init.restype = ctypes.c_uint32
hsa.hsa_shut_down.restype = ctypes.c_uint32
hsa.hsa_agent_get_info.argtypes = [Agent, ctypes.c_int, ctypes.c_void_p]
hsa.hsa_agent_get_info.restype = ctypes.c_uint32
hsa.hsa_iterate_agents.argtypes = [callback_type, ctypes.c_void_p]
hsa.hsa_iterate_agents.restype = ctypes.c_uint32

def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed: HSA status {status}")

gpus = []
errors = []

@callback_type
def visit(agent, _):
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(64)
    status = hsa.hsa_agent_get_info(agent, 17, ctypes.byref(device))
    if status == 0:
        status = hsa.hsa_agent_get_info(agent, 0, name)
    if status != 0:
        errors.append(f"hsa_agent_get_info failed: HSA status {status}")
        return status
    decoded = name.value.decode("utf-8")
    print(f"HSA agent: {decoded}, device type: {device.value}", flush=True)
    if device.value == 1:
        gpus.append(decoded)
    return 0

check(hsa.hsa_init(), "hsa_init")
try:
    check(hsa.hsa_iterate_agents(visit, None), "hsa_iterate_agents")
    if errors or not gpus or any(name != "gfx1250" for name in gpus):
        raise RuntimeError(f"Expected gfx1250 GPU agents, observed {gpus}; {errors}")
    print("Runtime gfx1250 probe passed", flush=True)
finally:
    check(hsa.hsa_shut_down(), "hsa_shut_down")
"""


def suite_metadata(test_dir: Path, reports: Path):
    command = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--show-only=json-v1",
        "-R",
        f"^{SUITE}$",
        "-L",
        "^ffm-quick$",
    ]
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(
        command, check=False, text=True, capture_output=True, timeout=60
    )
    (reports / "ctest-metadata.json").write_text(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    if result.returncode:
        print(result.stdout, end="", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    tests = json.loads(result.stdout)["tests"]
    print(f"Installed CTest selection: {json.dumps(tests, indent=2)}", flush=True)
    if len(tests) != 1 or tests[0].get("name") != SUITE:
        raise RuntimeError(
            f"Expected exactly one installed {SUITE}, found {len(tests)}"
        )
    test = tests[0]
    command = test.get("command", [])
    if not command:
        raise RuntimeError(f"Installed {SUITE} has no executable command")
    properties = {prop["name"]: prop["value"] for prop in test.get("properties", [])}
    if properties.get("DISABLED"):
        raise RuntimeError(f"Installed {SUITE} is disabled")
    working_dir = Path(properties.get("WORKING_DIRECTORY", str(test_dir)))
    if not working_dir.is_absolute():
        working_dir = test_dir / working_dir
    if not working_dir.is_dir():
        raise RuntimeError(f"Installed suite working directory missing: {working_dir}")
    return command, properties, working_dir.resolve()


def suite_environment(properties):
    """Apply CTest's environment metadata before imposing run-local isolation."""
    env = os.environ.copy()
    for entry in properties.get("ENVIRONMENT", []):
        name, value = entry.split("=", 1)
        env[name] = value
    original = env.copy()
    for entry in properties.get("ENVIRONMENT_MODIFICATION", []):
        name, modification = entry.split("=", 1)
        operation, value = modification.split(":", 1)
        if operation == "reset":
            if name in original:
                env[name] = original[name]
            else:
                env.pop(name, None)
        elif operation == "set":
            env[name] = value
        elif operation == "unset":
            env.pop(name, None)
        elif operation in ("string_append", "string_prepend"):
            old = env.get(name, "")
            env[name] = old + value if operation == "string_append" else value + old
        elif operation in (
            "path_list_append",
            "path_list_prepend",
            "cmake_list_append",
            "cmake_list_prepend",
        ):
            separator = os.pathsep if operation.startswith("path_") else ";"
            old = env.get(name, "")
            parts = [old, value] if operation.endswith("append") else [value, old]
            env[name] = separator.join(part for part in parts if part)
        else:
            raise RuntimeError(f"Unsupported CTest environment modification: {entry}")
    return env


def run_logged(command, env, working_dir, log_path, timeout):
    """Stream and save output; bound the launcher and its child process group."""
    print(f"+ {' '.join(map(str, command))}", flush=True)
    print(f"Output: {log_path} (timeout {timeout}s)", flush=True)
    deadline = time.monotonic() + timeout
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=working_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def stream():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        try:
            status = process.wait(timeout=timeout)
            reader.join(timeout=max(deadline - time.monotonic(), 0))
            if reader.is_alive():
                raise subprocess.TimeoutExpired(command, timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join()
            raise RuntimeError(f"Command timed out after {timeout}s; see {log_path}")
        finally:
            process.stdout.close()
    if status:
        raise subprocess.CalledProcessError(status, command)


def check_report(report: Path):
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError(
            f"Quick suite did not produce a nonempty XML report: {report}"
        )
    root = ET.parse(report).getroot()
    cases = root.findall(".//testcase")
    if not any(
        case.get("status") != "notrun" and case.find("skipped") is None
        for case in cases
    ):
        raise RuntimeError(f"Quick suite report contains no executed tests: {report}")
    if root.findall(".//failure") or root.findall(".//error"):
        raise RuntimeError(f"Quick suite XML report contains test failures: {report}")
    print(f"Quick suite report: {len(cases)} test cases, {report}", flush=True)


def summarize_report(report: Path):
    try:
        root = ET.parse(report).getroot()
        cases = root.findall(".//testcase")
        failed = sum(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        )
        skipped = sum(case.find("skipped") is not None for case in cases)
        disabled = sum(case.get("status") == "notrun" for case in cases)
        passed = len(cases) - failed - skipped - disabled
        summary = f"{SUITE}: {passed} passed, {failed} failed, {skipped} skipped"
        if disabled:
            summary += f", {disabled} disabled/not run"
        if not cases:
            summary += " (FAILED: empty report)"
    except (OSError, ET.ParseError):
        summary = f"{SUITE}: failed to complete with a valid XML report"
    print(summary, flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as output:
            output.write(f"### MIOpen gfx1250 rocjitsu quick suite\n\n{summary}\n\n")


def available_cpu_count():
    """Respect CPU affinity and Linux cgroup quotas, not just host CPU count."""
    count = len(os.sched_getaffinity(0))
    # Unified cgroup v2 (current CI containers), or legacy v1 CPU controller.
    for quota_path, period_path in (
        (Path("/sys/fs/cgroup/cpu.max"), None),
        (
            Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        ),
    ):
        try:
            if period_path is None:
                quota, period = quota_path.read_text().split()
                if quota == "max":
                    continue
                quota, period = int(quota), int(period)
            else:
                quota = int(quota_path.read_text())
                period = int(period_path.read_text())
            if quota > 0 and period > 0:
                count = min(count, max(1, quota // period))
        except FileNotFoundError:
            continue
    return max(1, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rocjitsu-bin", type=Path, help="Real prebuilt rocjitsu launcher"
    )
    parser.add_argument(
        "--rocjitsu-config", type=Path, help="gfx1250 emulator configuration"
    )
    parser.add_argument(
        "--dispatch-threads",
        type=int,
        default=None,
        help="Positive dispatch workers (default: min(16, available CPU resources))",
    )
    args = parser.parse_args()
    cpu_count = available_cpu_count()
    dispatch_threads = args.dispatch_threads
    if dispatch_threads is None:
        dispatch_threads = min(16, cpu_count)
    elif not 1 <= dispatch_threads <= cpu_count:
        parser.error(f"--dispatch-threads must be between 1 and {cpu_count}")
    if bool(args.rocjitsu_bin) != bool(args.rocjitsu_config):
        parser.error("--rocjitsu-bin and --rocjitsu-config must be supplied together")

    dist = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "./build")).resolve()
    bin_dir = Path(os.environ.get("THEROCK_BIN_DIR", "./build/bin")).resolve()
    gtest = bin_dir / "miopen_gtest"
    test_dir = bin_dir / "MIOpen"
    if not gtest.is_file() or not os.access(gtest, os.X_OK):
        raise RuntimeError(f"Installed miopen_gtest missing or not executable: {gtest}")
    if not (test_dir / "CTestTestfile.cmake").is_file():
        raise RuntimeError(
            f"Installed CTest metadata missing: {test_dir / 'CTestTestfile.cmake'}"
        )
    plugins = list(dist.rglob("libMIOpenCKGroupedConv_gfx1250.so"))
    if not any(path.is_file() and path.stat().st_size for path in plugins):
        raise RuntimeError(
            f"libMIOpenCKGroupedConv_gfx1250.so missing or empty in {dist}"
        )
    print(f"gfx1250 CK plugins: {[str(path) for path in plugins]}", flush=True)
    hsa = dist / "lib" / "libhsa-runtime64.so.1"
    if not hsa.is_file():
        raise RuntimeError(f"Packaged HSA runtime missing: {hsa}")

    reports = Path("rocjitsu-work/gfx1250-quick").resolve()
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / f"{SUITE}.xml"
    # A failed launch must not accidentally validate an earlier successful run.
    report.unlink(missing_ok=True)
    command, properties, working_dir = suite_metadata(test_dir, reports)
    executable = Path(command[0])
    if not executable.is_absolute():
        executable = working_dir / executable
    if executable.resolve() != gtest.resolve():
        raise RuntimeError(
            f"Installed {SUITE} does not invoke the fetched miopen_gtest: {command[0]}"
        )

    with tempfile.TemporaryDirectory(prefix="miopen-gfx1250-rocjitsu-") as temp:
        work = Path(temp)
        if args.rocjitsu_bin:
            rocjitsu = args.rocjitsu_bin.resolve()
            config = args.rocjitsu_config.resolve()
        else:
            ensure_build_tools()
            build_work = reports / ROCJITSU_REF
            build_work.mkdir(exist_ok=True)
            rocjitsu, source = build_rocjitsu(
                build_work, ROCJITSU_REF, jobs=min(cpu_count, 8)
            )
            config = source / "emulation" / "rocjitsu" / "configs" / ROCJITSU_CONFIG
        if not rocjitsu.is_file() or not os.access(rocjitsu, os.X_OK):
            raise RuntimeError(
                f"rocjitsu launcher missing or not executable: {rocjitsu}"
            )
        if not config.is_file():
            raise RuntimeError(f"rocjitsu config missing: {config}")
        config_data = json.loads(config.read_text())
        vm = config_data.get("vm", {})
        if (
            vm.get("target", "gfx1250") != "gfx1250"
            or vm.get("gpu", {}).get("device", {}).get("gfx_target_version") != 120500
        ):
            raise RuntimeError(f"rocjitsu config does not target gfx1250: {config}")
        # Keep the pinned SKU's CU fields; this lane needs only one emulated GPU.
        config_data["vm"]["gpu"]["num_gpus"] = 1
        config_data["num_threads"] = 1
        config_data["async_helper_threads"] = 0
        config_data["cpu_dispatch_threads"] = dispatch_threads
        print(
            f"rocjitsu dispatch workers: {dispatch_threads} (CPU limit {cpu_count})",
            flush=True,
        )
        used_config = reports / "rocjitsu-config.json"
        used_config.write_text(json.dumps(config_data, indent=2))

        env = suite_environment(properties)
        env["ROCM_PATH"] = str(dist)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(dist / "lib"), str(dist / "lib64"), env.get("LD_LIBRARY_PATH", "")]
        )
        env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
        # Do not let an inherited override fake the runtime architecture check.
        env.pop("HSA_OVERRIDE_GFX_VERSION", None)
        env["HSA_DISABLE_GFX12_STRICT"] = "1"
        env.pop("GTEST_SHARD_STATUS_FILE", None)
        for name, directory in (
            ("MIOPEN_USER_DB_PATH", "db"),
            ("MIOPEN_CUSTOM_CACHE_DIR", "cache"),
            ("MIOPEN_VERIFY_CACHE_PATH", "verify-cache"),
            ("XDG_CACHE_HOME", "xdg-cache"),
            ("HOME", "home"),
        ):
            path = work / directory
            path.mkdir()
            env[name] = str(path)
        env["GTEST_TOTAL_SHARDS"] = "1"
        env["GTEST_SHARD_INDEX"] = "0"
        env["GTEST_REPEAT"] = "1"
        env["GTEST_FAIL_FAST"] = "0"
        env["GTEST_OUTPUT"] = f"xml:{report}"
        env["GTEST_COLOR"] = "no"
        env["MIOPEN_INVOKED_FROM_CTEST"] = "1"
        launcher = [str(rocjitsu), "--config", str(used_config), "--"]
        run_logged(
            launcher + [sys.executable, "-c", RUNTIME_PROBE, str(hsa)],
            env,
            working_dir,
            reports / "runtime-probe.log",
            120,
        )
        # Replay CTest's packaged command verbatim, not a reconstructed filter.
        # Its hardware TIMEOUT property is deliberately replaced for emulation.
        try:
            run_logged(
                launcher + command,
                env,
                working_dir,
                reports / f"{SUITE}.log",
                TIMEOUT_SECONDS,
            )
        finally:
            summarize_report(report)
        check_report(report)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, ET.ParseError) as error:
        sys.exit(f"::error::{error}")
    except subprocess.CalledProcessError as error:
        print(
            f"::error::Command failed with exit status {error.returncode}",
            file=sys.stderr,
        )
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except subprocess.TimeoutExpired as error:
        sys.exit(f"::error::Command timed out after {error.timeout}s")
