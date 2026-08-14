#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
profiler-hub installation consumption test.

This test verifies that the profiler-hub package built by TheRock can be
properly consumed by an external project using CMake's find_package. It tests
the CMake packaging/installation correctness, not profiler-hub functionality.

profiler-hub is Linux-only (disable_platforms = ["windows"]), so this script
has no Windows handling.
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()
TEST_PROJECT_DIR = SCRIPT_DIR / "profiler_hub_install_tests"

logging.basicConfig(level=logging.INFO)


def configure_and_build(
    build_dir: Path, artifacts_path: Path, cxx_compiler: str, c_compiler: str, env
):
    """Configure (against an installed artifact tree) and build the consumer."""
    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(TEST_PROJECT_DIR),
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={artifacts_path}",
        f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
        f"-DCMAKE_C_COMPILER={c_compiler}",
        "--log-level=WARNING",
    ]
    logging.info(f"++ Configure: {shlex.join(configure_cmd)}")
    subprocess.run(configure_cmd, check=True, cwd=THEROCK_DIR, env=env)

    build_cmd = ["cmake", "--build", str(build_dir)]
    logging.info(f"++ Build: {shlex.join(build_cmd)}")
    subprocess.run(build_cmd, check=True, cwd=THEROCK_DIR, env=env)


def run_ctest(build_dir: Path, env):
    test_cmd = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--no-tests=error",
        "--timeout",
        "120",
    ]
    logging.info(f"++ Test: {shlex.join(test_cmd)}")
    subprocess.run(test_cmd, check=True, cwd=THEROCK_DIR, env=env)


def with_ld_library_path(env: dict, rocm_lib: str) -> dict:
    env = env.copy()
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = f"{rocm_lib}:{existing}" if existing else rocm_lib
    return env


def without_ld_library_path(env: dict) -> dict:
    env = env.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return env


def system_cxx17_capability():
    """Probe whether the runner's distro C/C++ toolchain can consume profiler-hub.

    Returns (ok, cxx_compiler_or_reason, c_compiler). Checks capability directly
    (via the `__cplusplus` macro under `-std=c++17`) rather than a hardcoded
    per-compiler-family version table, since that table would need maintenance
    every time a new compiler family shows up in a runner image.
    """
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return False, "no system C++ compiler (c++/g++/clang++) found on PATH", None
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return False, "no system C compiler (cc/gcc/clang) found on PATH", None
    probe = subprocess.run(
        [cxx, "-std=c++17", "-x", "c++", "-E", "-dM", "-"],
        input="",
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or "__cplusplus 201703L" not in probe.stdout:
        return (
            False,
            f"{cxx} does not report C++17 support (__cplusplus 201703L absent)",
            None,
        )
    return True, cxx, cc


def run_tests(build_dir: Path):
    """Configure, build, and test the profiler-hub package."""
    # Locally, can set OUTPUT_ARTIFACTS_DIR=build/dist/rocm for testing
    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    rocm_lib = str(artifacts_path / "lib")
    base_env = os.environ.copy()

    # We configure and build the test project externally (not during TheRock
    # build) to emulate how a consumer would build against the installed
    # profiler-hub artifacts. This catches packaging issues that only manifest
    # during external consumption.

    # Configuration 1: ROCm's own clang -- the toolchain profiler-hub ships with.
    rocm_clang_dir = build_dir / "rocm-clang"
    configure_and_build(
        rocm_clang_dir,
        artifacts_path,
        f"{artifacts_path}/lib/llvm/bin/clang++",
        f"{artifacts_path}/lib/llvm/bin/clang",
        base_env,
    )

    # Run 1: with LD_LIBRARY_PATH set -- the baseline find_package/link/run proof.
    run_ctest(rocm_clang_dir, with_ld_library_path(base_env, rocm_lib))

    # Run 2: without LD_LIBRARY_PATH -- informational only, surfaces RPATH/RUNPATH breakage.
    try:
        run_ctest(rocm_clang_dir, without_ld_library_path(base_env))
    except subprocess.CalledProcessError as exc:
        print(f"[A3] no-LD_LIBRARY_PATH run: FAIL — {exc}")
    else:
        print("[A3] no-LD_LIBRARY_PATH run: PASS")

    # Configuration 2: the distro/system compiler. profiler-hub is host-only
    # C++17, not HIP, and its real consumers build with the system toolchain,
    # not ROCm's clang -- prove that path too, alongside the first.
    ok, cxx_or_reason, cc = system_cxx17_capability()
    if not ok:
        print(f"[D2] system-compiler configuration: SKIPPED — {cxx_or_reason}")
        return
    system_dir = build_dir / "system-compiler"
    configure_and_build(system_dir, artifacts_path, cxx_or_reason, cc, base_env)
    run_ctest(system_dir, with_ld_library_path(base_env, rocm_lib))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test profiler-hub package installation and consumption"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory path (will be created if doesn't exist). "
        "If not specified, uses temporary directory that is auto-deleted.",
    )
    args = parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {OUTPUT_ARTIFACTS_DIR}")

    if args.build_dir:
        build_dir = args.build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using persistent build directory: {build_dir}")
        run_tests(build_dir)
        logging.info(f"Build artifacts retained in: {build_dir}")
    else:
        logging.info("Using temporary build directory (auto-cleanup)")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_tests(Path(temp_dir))

    logging.info("All profiler-hub install tests passed!")
