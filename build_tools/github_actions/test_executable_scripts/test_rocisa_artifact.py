#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify that the installed BLAS artifacts contain a loadable rocisa module."""

import os
import subprocess
import sys
from pathlib import Path

from pytest_runner import build_environment


def main() -> None:
    bin_dir = os.environ.get("THEROCK_BIN_DIR")
    if not bin_dir:
        raise RuntimeError("THEROCK_BIN_DIR must point into an installed ROCm tree")
    rocm_path = Path(bin_dir).resolve().parent
    module_relative_path = Path("rocisa") / "_rocisa.abi3.so"
    runtime_module = rocm_path / "lib" / "hipblaslt" / module_relative_path
    test_module = (
        rocm_path / "share" / "hipblaslt" / "tensilelite" / module_relative_path
    )
    for module in (runtime_module, test_module):
        if not module.is_file():
            raise FileNotFoundError(f"rocisa extension missing from artifact: {module}")

    script = """
import pathlib
import sys
import rocisa
from rocisa import _rocisa

expected = pathlib.Path(sys.argv[1]).resolve()
loaded = pathlib.Path(_rocisa.__file__).resolve()
if loaded != expected:
    raise RuntimeError(f"loaded {loaded}, expected {expected}")
print(f"Loaded installed rocisa extension: {loaded}")
"""
    for module in (runtime_module, test_module):
        env = build_environment(rocm_path, "tensilelite")
        if module == runtime_module:
            # The public install path must resolve its libraries from RPATH.
            env.pop("LD_LIBRARY_PATH", None)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(module.parent.parent), env["PYTHONPATH"])
        )
        subprocess.run(
            [sys.executable, "-c", script, str(module)],
            cwd=rocm_path,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
