# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Wrapper for the rocgdb-corefile test job that collects environment
diagnostics before handing off to the real test_rocgdb.py."""

import os
import resource
import subprocess
import sys
from pathlib import Path

COREFILE_TESTS = [
    "gdb.rocm/simple.exp",
    "gdb.rocm/debugtrap.exp",
]

_ARTIFACTS_DIR = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "./build"))
_TEST_ROCGDB = _ARTIFACTS_DIR / "tests" / "rocgdb" / "test_rocgdb.py"


def _run(*cmd: str) -> str:
    return subprocess.run(list(cmd), capture_output=True, text=True).stdout.rstrip()


def _print_device_permissions() -> None:
    print("=== GPU device permissions ===", flush=True)

    kfd = Path("/dev/kfd")
    if kfd.exists():
        print(_run("ls", "-la", str(kfd)), flush=True)
    else:
        print("/dev/kfd not found", flush=True)

    dri = Path("/dev/dri")
    render_nodes = sorted(dri.glob("render*")) if dri.exists() else []
    if render_nodes:
        print(_run("ls", "-la", *[str(p) for p in render_nodes]), flush=True)
    else:
        print("No /dev/dri render nodes found", flush=True)


def _print_user_groups() -> None:
    print("=== Current user and groups ===", flush=True)
    print(_run("id"), flush=True)
    print("Groups:", _run("groups"), flush=True)


def _print_ptrace_scope() -> None:
    print("=== ptrace scope ===", flush=True)
    scope = Path("/proc/sys/kernel/yama/ptrace_scope")
    if scope.exists():
        print(f"ptrace_scope: {scope.read_text().rstrip()}", flush=True)
    else:
        print("/proc/sys/kernel/yama/ptrace_scope not found", flush=True)


def _print_core_dump_config() -> None:
    print("=== Core dump configuration ===", flush=True)
    pattern = Path("/proc/sys/kernel/core_pattern")
    if pattern.exists():
        print(f"core_pattern: {pattern.read_text().rstrip()}", flush=True)
    else:
        print("/proc/sys/kernel/core_pattern not found", flush=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    print(f"ulimit -c: soft={soft} hard={hard}", flush=True)


def _print_kfd_topology() -> None:
    print("=== KFD topology ===", flush=True)
    nodes_dir = Path("/sys/class/kfd/kfd/topology/nodes")
    if not nodes_dir.exists():
        print("KFD topology not found", flush=True)
        return
    for node in sorted(nodes_dir.iterdir()):
        props = node / "properties"
        if props.exists():
            print(f"-- node {node.name} --", flush=True)
            print(props.read_text().rstrip(), flush=True)


def _print_kernel_modules() -> None:
    print("=== Loaded amdgpu/kfd kernel modules ===", flush=True)
    lsmod = _run("lsmod")
    matches = [
        line for line in lsmod.splitlines() if any(k in line for k in ("amdgpu", "kfd"))
    ]
    print("\n".join(matches) if matches else "No amdgpu/kfd modules found", flush=True)


def _print_process_limits() -> None:
    print("=== Process limits (ulimit -a) ===", flush=True)
    print(_run("bash", "-c", "ulimit -a"), flush=True)


def _print_rocm_smi() -> None:
    print("=== rocm-smi ===", flush=True)
    rocm_smi = _ARTIFACTS_DIR / "bin" / "rocm-smi"
    if not rocm_smi.exists():
        print(f"rocm-smi not found at {rocm_smi}", flush=True)
        return
    print(_run(str(rocm_smi)), flush=True)


def main() -> None:
    for fn in (
        _print_device_permissions,
        _print_user_groups,
        _print_ptrace_scope,
        _print_core_dump_config,
        _print_kfd_topology,
        _print_kernel_modules,
        _print_process_limits,
        _print_rocm_smi,
    ):
        fn()
        print(flush=True)

    cmd = [sys.executable, str(_TEST_ROCGDB), "--tests"] + COREFILE_TESTS
    print(f"Running: {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
