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


def _required_file(path: Path) -> bool:
    if path.is_file():
        return True
    print(f"ERROR: required StinkyTofu host-ASAN artifact is missing: {path}", file=sys.stderr)
    return False


def _run(command: list[str], cwd: Path) -> None:
    logging.info("++ Exec %s", shlex.join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    test_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve() / "stinkytofu"
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
        print(f"ERROR: no StinkyTofu architectures listed in {architecture_file}", file=sys.stderr)
        return 1

    filecheck_inputs = sorted((test_dir / "filecheck").rglob("*.stir"))
    filecheck_inputs.extend(sorted((test_dir / "filecheck").rglob("*.s")))
    if not filecheck_inputs:
        print("ERROR: no StinkyTofu FileCheck inputs were packaged", file=sys.stderr)
        return 1

    _run([str(unit_tests)], test_dir)
    _run([str(generator_test), str(source_dir), *architectures], test_dir)
    for test_input in filecheck_inputs:
        _run(
            [str(checker), str(test_input), "--stinkytofu-opt", str(optimizer)],
            test_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
