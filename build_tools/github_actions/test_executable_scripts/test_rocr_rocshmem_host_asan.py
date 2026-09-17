#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the fail-closed CPU-only rocSHMEM and ROCr host-ASAN suites."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from host_asan_instrumentation import (
    native_host_asan_environment,
    require_direct_clang_asan,
)

COMPONENTS = {
    "rocshmem": (
        {
            "path": "bin/rocshmem_envvar_test",
            "count": 79,
            "sha256": (
                "5f517cf84a42fca8000429fbf5a7826bcb089ae8e5e67f070103428d49f29e4e"
            ),
        },
    ),
    "rocrtst": (
        {
            "path": "bin/intercept_queue_logic_test",
            "count": 9,
            "sha256": (
                "427e22cbc625d6c60ae4a6707dbbcbdaf2ad49b54bc788a03e1b52f5b505bf36"
            ),
        },
        {
            "path": "bin/rocrtst_poll_backoff_test",
            "count": 5,
            "sha256": (
                "5d850b2b916442ce4f45ea65c41ecf7a98a2b489fc9dbd254e960f0a90dd3d04"
            ),
        },
        {
            "path": "bin/rocrtst_doorbell_type_test",
            "count": 13,
            "sha256": (
                "f138745c36bd5cd39ce4b089c253c5ab677bcfeeb40fdda9f543169ef0face3d"
            ),
        },
    ),
}

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")
_SKIPPED_RE = re.compile(r"\[\s*SKIPPED\s*\]\s+\d+ tests?")


def _with_option(value: str, option: str) -> str:
    return f"{value}:{option}" if value else option


def _test_environment() -> dict[str, str]:
    env = native_host_asan_environment()
    detect_leaks = (
        "detect_leaks=0"
        if env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1"
        else "detect_leaks=1"
    )
    env["ASAN_OPTIONS"] = _with_option(env.get("ASAN_OPTIONS", ""), detect_leaks)
    env["ASAN_OPTIONS"] = _with_option(env["ASAN_OPTIONS"], "halt_on_error=1")
    env["LSAN_OPTIONS"] = _with_option(env.get("LSAN_OPTIONS", ""), "exitcode=23")
    return env


def _require_no_gpu_nodes() -> None:
    present = [path for path in (Path("/dev/kfd"), Path("/dev/dri")) if path.exists()]
    if present:
        raise RuntimeError(
            f"host-only runner unexpectedly exposes GPU nodes: {present}"
        )


def _require_direct_asan(executable: Path, env: dict[str, str]) -> None:
    require_direct_clang_asan(executable, env)


def _parse_gtest_names(output: str) -> list[str]:
    names: list[str] = []
    suite = None
    for raw_line in output.splitlines():
        core = raw_line.split("#", 1)[0].rstrip()
        if not core.strip():
            continue
        if not raw_line[0].isspace() and core.endswith("."):
            suite = core.strip()
            continue
        if suite is not None and raw_line[0].isspace():
            case = core.strip()
            if case:
                names.append(f"{suite}{case}")
    return names


def _inventory_digest(names: list[str]) -> str:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _run_component(prefix: Path, component: str, env: dict[str, str]) -> None:
    for test in COMPONENTS[component]:
        executable = prefix / test["path"]
        _require_direct_asan(executable, env)

        listed = subprocess.run(
            [str(executable), "--gtest_list_tests"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        names = _parse_gtest_names(listed.stdout)
        digest = _inventory_digest(names)
        if len(names) != test["count"] or digest != test["sha256"]:
            raise RuntimeError(
                "host-ASAN inventory changed before execution: "
                f"expected count={test['count']}, sha256={test['sha256']}; "
                f"got count={len(names)}, sha256={digest}"
            )

        command = [str(executable)]
        print(f"++ Exec {shlex.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        combined = result.stdout + result.stderr
        passed = _PASSED_RE.search(combined)
        if not passed or int(passed.group(1)) != test["count"]:
            raise RuntimeError(
                f"unexpected pass count for {executable.name}: expected {test['count']}"
            )
        if _SKIPPED_RE.search(combined):
            raise RuntimeError(f"host-ASAN suite skipped tests: {executable.name}")


def main() -> int:
    try:
        component = os.environ["TEST_COMPONENT"]
        if component not in COMPONENTS:
            raise RuntimeError(f"unsupported runtime host-ASAN component: {component}")
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        _require_no_gpu_nodes()
        _run_component(prefix, component, _test_environment())
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
