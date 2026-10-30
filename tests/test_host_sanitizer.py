#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build host sanitizer controls with an installed ROCm compiler and ccache.

This integration test needs Linux, ROCm development artifacts, CMake, Ninja,
and ccache. It compiles HIP device code but does not require a GPU. Build logs,
objects and sanitizer reports are retained in --build-dir for inspection.
"""

import argparse
import os
from pathlib import Path
import shutil
import shlex
import subprocess


SOURCE_DIR = Path(__file__).resolve().parent / "host_sanitizer"


def run(command: list[str], log: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-path", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--sanitizer", choices=("TSAN", "HOST_ASAN"), required=True)
    parser.add_argument("--gpu-arch", default="gfx942")
    args = parser.parse_args()
    prefix = args.rocm_path.resolve()
    root = args.build_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    ccache = shutil.which("ccache")
    if not ccache:
        raise RuntimeError("ccache is required to test the compiler launcher")
    llvm_bin = prefix / "lib" / "llvm" / "bin"
    compiler = llvm_bin / "clang++"
    env = os.environ.copy()
    # Test the project's launcher policy, independent of the caller's bypass.
    env.pop("CCACHE_DISABLE", None)
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    env["CCACHE_DIR"] = str(root / "cache")
    env["CCACHE_TEMPDIR"] = str(root / "cache-tmp")
    env["TSAN_OPTIONS"] = "halt_on_error=1:exitcode=86"
    env["ASAN_OPTIONS"] = "halt_on_error=1:exitcode=87"
    runtime_dir = run(
        [str(compiler), "-print-runtime-dir"], root / "runtime.log", env
    ).strip()
    rpaths = f"{runtime_dir};{prefix / 'lib'}"
    symbol = "__tsan_write" if args.sanitizer == "TSAN" else "__asan_report"
    report = (
        "ThreadSanitizer: data race"
        if args.sanitizer == "TSAN"
        else "AddressSanitizer: heap-use-after-free"
    )
    mode = "race" if args.sanitizer == "TSAN" else "use-after-free"
    expected_exit = 86 if args.sanitizer == "TSAN" else 87

    for launcher in ("direct", "ccache"):
        build = root / launcher
        env["CCACHE_LOGFILE"] = str(root / f"{launcher}-ccache.log")
        command = [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_C_COMPILER={llvm_bin / 'clang'}",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DCMAKE_HIP_COMPILER={compiler}",
            f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={prefix}",
            f"-DCMAKE_HIP_ARCHITECTURES={args.gpu_arch}",
            f"-DTHEROCK_SANITIZER={args.sanitizer}",
            f"-DCMAKE_BUILD_RPATH={rpaths}",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        ]
        for language in ("C", "CXX", "HIP"):
            command.append(
                f"-DCMAKE_{language}_COMPILER_LAUNCHER={ccache if launcher == 'ccache' else ''}"
            )
        run(command, root / f"{launcher}-configure.log", env)
        run(
            ["cmake", "--build", str(build), "--verbose"],
            root / f"{launcher}-build.log",
            env,
        )
        hip_flags = next(
            line.split("=", 1)[1]
            for line in (build / "CMakeCache.txt").read_text().splitlines()
            if line.startswith("CMAKE_HIP_FLAGS:STRING=")
        )
        device_ir = root / f"{launcher}-device.ll"
        run(
            [
                str(compiler),
                *shlex.split(hip_flags),
                "-x",
                "hip",
                "--offload-device-only",
                f"--offload-arch={args.gpu_arch}",
                "-S",
                "-emit-llvm",
                str(SOURCE_DIR / "probe.hip"),
                "-o",
                str(device_ir),
            ],
            root / f"{launcher}-device.log",
            env,
        )
        ir = device_ir.read_text()
        if "device_probe" not in ir or "@__tsan" in ir or "@__asan" in ir:
            raise AssertionError(
                f"Expected an unsanitized device kernel in {device_ir}"
            )
        for language, extension in (("c", "c"), ("cxx", "cpp"), ("hip", "hip")):
            obj = build / "CMakeFiles/sanitizer_probe.dir" / f"probe.{extension}.o"
            symbols = run(
                [str(llvm_bin / "llvm-nm"), "--undefined-only", str(obj)],
                root / f"{launcher}-{language}-symbols.log",
                env,
            )
            if symbol not in symbols:
                raise AssertionError(f"Missing memory-access instrumentation in {obj}")
            executable = str(build / "sanitizer_control")
            run(
                [executable, language, "clean"],
                root / f"{launcher}-{language}-clean.log",
                env,
            )
            result = subprocess.run(
                [executable, language, mode],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            log = root / f"{launcher}-{language}-report.log"
            log.write_text(result.stdout, encoding="utf-8")
            if result.returncode != expected_exit or report not in result.stdout:
                raise AssertionError(
                    f"Expected sanitizer finding and exit {expected_exit}; see {log}"
                )
            print(
                f"PASS {args.sanitizer} {launcher} {language}: instrumented, clean control passed, defect detected"
            )


if __name__ == "__main__":
    main()
