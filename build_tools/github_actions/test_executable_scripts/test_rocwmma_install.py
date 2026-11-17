#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build installed rocWMMA consumers against the artifact SDK, without a GPU."""

import argparse
from dataclasses import dataclass
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

# These settings identify the SDK/toolchain under test. Case-specific cache
# options may change consumer behavior, but must not select another SDK.
RUNNER_CMAKE_OPTIONS = {
    "CMAKE_PREFIX_PATH",
    "rocwmma_DIR",
    "hip_DIR",
    "GPU_TARGETS",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_HIP_COMPILER",
    "CMAKE_RC_COMPILER",
    "CMAKE_HIP_COMPILER_ROCM_ROOT",
    "CMAKE_HIP_ARCHITECTURES",
    "CMAKE_TOOLCHAIN_FILE",
    "ROCM_PATH",
    "HIP_PATH",
    "HIP_ROOT_DIR",
    "CMAKE_GENERATOR",
}
PHASES = ("configure", "build", "test")


@dataclass(frozen=True)
class ConsumerCase:
    name: str
    source: Path
    hip: bool
    phase: str
    cmake_options: dict[str, str]


def load_cases(source: Path, system: str) -> list[ConsumerCase]:
    """Validate installed registrations before selecting this platform's cases."""
    manifest = source / "tests.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Required test manifest is missing: {manifest}. "
            "Rebuild the rocWMMA test artifacts with manifest support."
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"version", "tests"}:
        raise ValueError(f"{manifest}: expected version and tests fields")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError(f"{manifest}: unsupported manifest version")
    if not isinstance(data["tests"], list) or not data["tests"]:
        raise ValueError(f"{manifest}: tests must be a nonempty list")

    cases = []
    names = set()
    allowed = {"name", "source", "platforms", "hip", "phase", "cmake_options"}
    for entry in data["tests"]:
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError(f"{manifest}: invalid test entry: {entry!r}")
        name = entry.get("name")
        # Lowercase names avoid collisions on Windows. No path separators or
        # trailing dots are allowed in names used as build directories.
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"{manifest}: invalid case name: {name!r}")
        if name in names or re.fullmatch(r"con|prn|aux|nul|com[0-9]|lpt[0-9]", name):
            raise ValueError(f"{manifest}: duplicate or reserved case name: {name}")
        names.add(name)
        relative = entry.get("source")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or ":" in relative
        ):
            raise ValueError(
                f"{name}: source must be a relative path using forward slashes"
            )
        case_source = (source / relative).resolve()
        if Path(relative).is_absolute() or not case_source.is_relative_to(
            source.resolve()
        ):
            raise ValueError(f"{name}: source must remain inside {source}")
        if not (case_source / "CMakeLists.txt").is_file():
            raise FileNotFoundError(f"{name}: missing {case_source / 'CMakeLists.txt'}")
        platforms = entry.get("platforms", ["linux", "windows"])
        if (
            not isinstance(platforms, list)
            or not platforms
            or any(p not in ("linux", "windows") for p in platforms)
        ):
            raise ValueError(f"{name}: platforms must list linux and/or windows")
        hip = entry.get("hip", False)
        phase = entry.get("phase", "test")
        if type(hip) is not bool or phase not in PHASES:
            raise ValueError(
                f"{name}: hip must be boolean and phase must be one of {PHASES}"
            )
        options = entry.get("cmake_options", {})
        if not isinstance(options, dict) or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(value, str)
            for key, value in options.items()
        ):
            raise ValueError(
                f"{name}: cmake_options must map bare cache names to strings"
            )
        if reserved := RUNNER_CMAKE_OPTIONS.intersection(options):
            raise ValueError(
                f"{name}: runner-controlled CMake options: {sorted(reserved)}"
            )
        if system not in platforms:
            logging.info("Skipping %s: supported platforms are %s", name, platforms)
            continue
        cases.append(ConsumerCase(name, case_source, hip, phase, options))
    if not cases:
        raise ValueError(f"{manifest}: no applicable cases for {system}")
    return cases


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
    cases = load_cases(source, platform.system().lower())
    for required in [compiler, sdk / "lib/cmake/rocwmma/rocwmma-config.cmake"]:
        if not required.is_file():
            raise FileNotFoundError(f"Required SDK file is missing: {required}")
    for case in cases:
        if (build_dir / case.name).exists():
            raise FileExistsError(
                f"Refusing cached consumer build: {build_dir / case.name}"
            )

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

    for case in cases:
        name = case.name
        case_build = build_dir / name
        # Refuse cached builds: preloaded package targets can hide regressions.
        case_build.mkdir(parents=True, exist_ok=False)
        configure = [
            "cmake",
            "-S",
            str(case.source),
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
        if case.hip:
            configure.extend(
                [
                    f"-DCMAKE_HIP_COMPILER={compiler}",
                    f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={sdk}",
                    f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
                ]
            )
        if is_windows:
            configure.append("-DCMAKE_RC_COMPILER=rc.exe")
        configure.extend(
            f"-D{key}={value}" for key, value in case.cmake_options.items()
        )
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
            if phase == case.phase:
                break


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
