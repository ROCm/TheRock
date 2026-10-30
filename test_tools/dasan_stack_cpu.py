#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check shared DASan/HIP/ROCr initialization on a CPU build runner."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

SOURCE = r"""
#include <hip/hip_runtime_api.h>
#include <cstdio>

int main() {
  int devices = -1;
  hipError_t status = hipGetDeviceCount(&devices);
  std::printf("DASAN_STACK_CPU hip_status=%d devices=%d\n", int(status), devices);
  if (status != hipSuccess && status != hipErrorNoDevice) return 1;
  if (status == hipSuccess && devices < 0) return 2;
  return 0;
}
"""


def command(argv: list[str], env: dict, output: Path, timeout: int = 120):
    output.with_suffix(".command.txt").write_text(shlex.join(argv) + "\n")
    result = subprocess.run(
        argv,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Command exited {result.returncode}: {shlex.join(argv)}; see {output}"
        )
    return result.stdout


def check_result(stdout: str) -> dict:
    matches = re.findall(
        r"^DASAN_STACK_CPU hip_status=(-?\d+) devices=(-?\d+)$", stdout, re.MULTILINE
    )
    if len(matches) != 1:
        raise RuntimeError("Missing or ambiguous HIP initialization result")
    status, devices = map(int, matches[0])
    if status not in (0, 100) or (status == 0 and devices < 0):
        raise RuntimeError(
            f"Unexpected HIP initialization result: status={status}, devices={devices}"
        )
    return {"hip_status": status, "devices": devices, "no_gpu": status == 100}


def validate(rocm: Path, output: Path) -> dict:
    rocm = rocm.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "state": "running",
        "rocm_path": str(rocm),
        "linkage": "shared",
        "gpu_tests_executed": False,
    }
    try:
        clang = rocm / "lib/llvm/bin/clang++"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(rocm / p) for p in ("lib", "lib/llvm/lib", "lib/rocm_sysdeps/lib")]
            + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
        library = command(
            [str(clang), "-print-file-name=libclang_rt.dasan.so"],
            env,
            output / "runtime-query.log",
        ).strip()
        runtime = Path(library)
        if not runtime.is_absolute() or not runtime.is_file():
            raise RuntimeError(
                f"Selected compiler cannot locate its shared DASan runtime: {library!r}"
            )
        if not runtime.resolve().is_relative_to(rocm):
            raise RuntimeError(
                f"Shared DASan runtime is outside selected installation: {runtime}"
            )
        env["LD_LIBRARY_PATH"] = (
            str(runtime.parent) + os.pathsep + env["LD_LIBRARY_PATH"]
        )
        summary["host_runtime"] = str(runtime)
        summary["library_path"] = env["LD_LIBRARY_PATH"]
        source = output / "hip-initialize.cpp"
        source.write_text(SOURCE, encoding="utf-8")
        executable = output / "hip-initialize"
        # Both DSOs must load, including ROCr on hosts without a GPU. Avoid the
        # compiler's default static DASan mixed with instrumented shared HIP.
        command(
            [
                str(clang),
                "-fsanitize=daddress",
                "-shared-libsan",
                "-D__HIP_PLATFORM_AMD__=1",
                "-I" + str(rocm / "include"),
                str(source),
                "-L" + str(rocm / "lib"),
                "-Wl,--no-as-needed",
                "-lamdhip64",
                "-lhsa-runtime64",
                "-Wl,--as-needed",
                "-o",
                str(executable),
            ],
            env,
            output / "compile.log",
        )
        stdout = command([str(executable)], env, output / "execute.log", timeout=60)
        summary.update(check_result(stdout), state="passed")
        return summary
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    validate(args.rocm_path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
