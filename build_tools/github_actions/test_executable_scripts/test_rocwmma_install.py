#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build installed rocWMMA consumers against the artifact SDK, without a GPU."""

import argparse
import logging
import os
import platform
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def run_tests(sdk: Path, build_dir: Path, gpu_targets: str) -> None:
    """Run independent consumers; retain build directories and logs on failure."""
    sdk = sdk.resolve()
    build_dir = build_dir.resolve()
    # One compile target is enough for this interface test. Never probe hardware:
    # CI supplies concrete targets even on CPU runners.
    targets = [t for t in re.split(r"[,;\s]+", gpu_targets) if t]
    if not targets or any(not re.fullmatch(r"gfx[0-9a-f]+", t) for t in targets):
        raise ValueError(
            "Supply concrete GPU targets via --gpu-targets or AMDGPU_TARGETS"
        )
    gpu_arch = targets[0]
    source = sdk / "share/rocwmma/tests/package_consumer"
    is_windows = platform.system() == "Windows"
    compiler = sdk / "lib/llvm/bin" / ("clang++.exe" if is_windows else "clang++")
    cases = {"no_openmp": source}
    # TheRock ships LLVM OpenMP on Linux only. This case intentionally requires
    # FindOpenMP to work, then checks rocWMMA's exported target dependencies.
    if not is_windows:
        cases["openmp"] = source / "openmp"
    for required in [compiler, sdk / "lib/cmake/rocwmma/rocwmma-config.cmake"] + [
        path / "CMakeLists.txt" for path in cases.values()
    ]:
        if not required.is_file():
            raise FileNotFoundError(f"Required SDK file is missing: {required}")

    env = os.environ.copy()
    env["HIP_PLATFORM"] = "amd"
    env["ROCM_PATH"] = str(sdk)
    env["HIP_PATH"] = str(sdk)
    if is_windows:
        env["PATH"] = os.pathsep.join(
            [str(sdk / "bin"), str(sdk / "lib/llvm/bin"), env.get("PATH", "")]
        )
    else:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                str(sdk / "lib"),
                str(sdk / "lib/llvm/lib"),
                env.get("LD_LIBRARY_PATH", ""),
            ]
        )

    for name, case_source in cases.items():
        case_build = build_dir / name
        # Refuse cached builds: preloaded package targets can hide regressions.
        case_build.mkdir(parents=True, exist_ok=False)
        configure = [
            "cmake",
            "-S",
            str(case_source),
            "-B",
            str(case_build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={sdk};{sdk / 'lib/llvm'}",
            f"-Drocwmma_DIR={sdk / 'lib/cmake/rocwmma'}",
            f"-Dhip_DIR={sdk / 'lib/cmake/hip'}",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DGPU_TARGETS={gpu_arch}",
        ]
        if name == "no_openmp":
            configure.extend(
                [
                    f"-DCMAKE_HIP_COMPILER={compiler}",
                    f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={sdk}",
                    f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
                ]
            )
        if is_windows:
            configure.append("-DCMAKE_RC_COMPILER=rc.exe")
        commands = {
            "configure": configure,
            "build": ["cmake", "--build", str(case_build), "--verbose"],
            "test": [
                "ctest",
                "--test-dir",
                str(case_build),
                "--output-on-failure",
                "--no-tests=error",
            ],
        }
        for phase, command in commands.items():
            logging.info("%s/%s: %s", name, phase, shlex.join(command))
            # Keep a log for each phase and also print it into the CI job log.
            log = case_build / f"{phase}.log"
            with log.open("w") as output:
                result = subprocess.run(
                    command, env=env, stdout=output, stderr=subprocess.STDOUT
                )
            print(log.read_text(errors="replace"), flush=True)
            result.check_returncode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk",
        type=Path,
        default=os.getenv("OUTPUT_ARTIFACTS_DIR"),
        help="SDK prefix (defaults to OUTPUT_ARTIFACTS_DIR)",
    )
    parser.add_argument(
        "--gpu-targets",
        default=os.getenv("AMDGPU_TARGETS", ""),
        help="Concrete compile targets, separated by commas or semicolons",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Directory for fresh consumer builds and logs; retained after the run",
    )
    args = parser.parse_args()
    if args.sdk is None:
        parser.error("--sdk or OUTPUT_ARTIFACTS_DIR is required")
    build_dir = args.build_dir or Path(tempfile.mkdtemp(prefix="rocwmma-install-"))
    logging.basicConfig(level=logging.INFO)
    logging.info("Consumer builds and logs: %s", build_dir.resolve())
    run_tests(args.sdk, build_dir, args.gpu_targets)


if __name__ == "__main__":
    main()
