#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run unchanged DASan tests against the downloaded full-stack installation."""

import argparse
import ctypes
from pathlib import Path
import subprocess
import sys


def device_count(rocm: Path) -> int:
    runtime = ctypes.CDLL(str(rocm / "lib/libamdhip64.so"))
    runtime.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.hipGetDeviceCount.restype = ctypes.c_int
    count = ctypes.c_int()
    status = runtime.hipGetDeviceCount(ctypes.byref(count))
    if status != 0 or count.value < 2:
        raise RuntimeError(
            f"Full DASan validation requires multiple GPUs: HIP status={status}, count={count.value}"
        )
    return count.value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-path", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rocm = args.rocm_path.resolve()
    bundle = args.bundle.resolve()
    subprocess.check_call(
        [
            sys.executable,
            str(Path(__file__).with_name("dasan_smoke.py")),
            "--rocm-path",
            str(rocm),
            "--llvm-lit",
            str(bundle / "llvm-lit.py"),
            "--lit-suite",
            str(bundle / "suite"),
            "--gpu-arch",
            "gfx950",
            "--compile-arch",
            "gfx950",
            "--gpu-device-count",
            str(device_count(rocm)),
            "--openmp",
            "--require-multi-device",
            "--shared-runtime",
            "--output-dir",
            str(args.output_dir),
        ]
    )


if __name__ == "__main__":
    main()
