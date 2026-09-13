#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the relocatable StinkyTofu native host test payload."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from host_asan_instrumentation import (
    native_host_asan_environment,
    require_direct_clang_asan,
)


def _required_file(path: Path) -> bool:
    if path.is_file():
        return True
    print(
        f"ERROR: required StinkyTofu host-ASAN artifact is missing: {path}",
        file=sys.stderr,
    )
    return False


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    logging.info("++ Exec %s", shlex.join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> int:
    test_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve() / "stinkytofu"
    rocm_path = test_dir.parent.parent
    unit_tests = test_dir / "unit_tests"
    generator_test = test_dir / "test_gen_instructions"
    checker = test_dir / "stinkytofu-check"
    optimizer = test_dir / "stinkytofu-opt"
    architecture_file = test_dir / "architectures.txt"
    source_dir = test_dir / "source"

    required = (unit_tests, generator_test, checker, optimizer, architecture_file)
    if not all(_required_file(path) for path in required):
        return 1

    architectures = [
        line.strip()
        for line in architecture_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not architectures:
        print(
            f"ERROR: no StinkyTofu architectures listed in {architecture_file}",
            file=sys.stderr,
        )
        return 1

    filecheck_inputs = sorted((test_dir / "filecheck").rglob("*.stir"))
    filecheck_inputs.extend(sorted((test_dir / "filecheck").rglob("*.s")))
    if not filecheck_inputs:
        print("ERROR: no StinkyTofu FileCheck inputs were packaged", file=sys.stderr)
        return 1

    # The StinkyTofu executables are installed two levels below the prefix, so
    # their $ORIGIN-relative RUNPATH does not reach the flattened artifact
    # libraries. Supply the explicit, positive set of directories used by the
    # amd-llvm and sysdeps artifacts without discarding a caller-provided path.
    env = native_host_asan_environment()
    existing_ld_path = env.get("LD_LIBRARY_PATH")
    library_paths = [
        rocm_path / "lib",
        rocm_path / "lib" / "rocm_sysdeps" / "lib",
        rocm_path / "lib" / "llvm" / "lib",
        *sorted((rocm_path / "lib" / "llvm" / "lib" / "clang").glob("*/lib/linux")),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            *(str(path) for path in library_paths),
            *([existing_ld_path] if existing_ld_path else []),
        ]
    )

    # The optimizer is launched by stinkytofu-check rather than this wrapper,
    # but it is still part of the explicit native payload and must independently
    # prove direct instrumentation before any StinkyTofu process executes.
    for executable in (unit_tests, generator_test, checker, optimizer):
        require_direct_clang_asan(executable, env)

    _run([str(unit_tests)], test_dir, env)
    _run([str(generator_test), str(source_dir), *architectures], test_dir, env)
    for test_input in filecheck_inputs:
        _run(
            [str(checker), str(test_input), "--stinkytofu-opt", str(optimizer)],
            test_dir,
            env,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
