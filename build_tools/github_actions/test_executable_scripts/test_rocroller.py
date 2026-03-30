#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", "")
platform = os.getenv("RUNNER_OS", "linux").lower()

BUILD_DIR = Path(os.getenv("THEROCK_BUILD_DIR", THEROCK_DIR / "build"))
ROCROLLER_STAGE_DIR = BUILD_DIR / "math-libs" / "BLAS" / "rocRoller" / "stage"

# Resolve bin dir to an absolute path for path arithmetic
bin_dir = Path(THEROCK_BIN_DIR).resolve() if THEROCK_BIN_DIR else None


def find_first(
    candidates: list[Path],
    label: str,
    predicate=Path.is_file,
) -> Path:
    found = next((p for p in candidates if predicate(p)), None)
    if not found:
        raise FileNotFoundError(
            f"{label} not found in: {', '.join(map(str, candidates))}"
        )
    return found


# Locate rrtest Python library (from rocroller)
# CI:    <bin_dir>/../scripts/lib/rrtest  (artifact flattened into build/)
# Local: <ROCROLLER_STAGE_DIR>/scripts/lib/rrtest
rrtest_candidates = []
if bin_dir:
    rrtest_candidates.append(bin_dir.parent / "scripts" / "lib")
rrtest_candidates.append(ROCROLLER_STAGE_DIR / "scripts" / "lib")

rrtest_lib = find_first(
    rrtest_candidates, "rrtest", predicate=lambda p: (p / "rrtest").is_dir()
)
sys.path.insert(0, str(rrtest_lib))

from rrtest import get_test_commands  # noqa: E402

# Locate test-profiles.yaml
profiles_candidates = []
if bin_dir:
    profiles_candidates.append(bin_dir.parent / "test-profiles.yaml")
profiles_candidates.append(ROCROLLER_STAGE_DIR / "test-profiles.yaml")

profiles_yaml = find_first(profiles_candidates, "test-profiles.yaml")


# Locate a test binary by name
def find_bin(name: str) -> Path:
    candidates = []
    if bin_dir:
        candidates.append(bin_dir / name)
    candidates.append(
        BUILD_DIR / "math-libs" / "BLAS" / "rocRoller" / "build" / "test" / name
    )
    return find_first(candidates, name)


# Sharding (gtest reads these from the environment)
env = os.environ.copy()
env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
env["GTEST_TOTAL_SHARDS"] = str(int(os.getenv("TOTAL_SHARDS", "1")))

# Runtime libs
if platform == "linux":
    THEROCK_DIST_DIR = BUILD_DIR / "core" / "clr" / "dist"
    llvm_libdir = THEROCK_DIST_DIR / "lib" / "llvm" / "lib"
    ld_parts = [
        str(THEROCK_DIST_DIR / "lib"),
        str(THEROCK_DIST_DIR / "lib64"),
        str(llvm_libdir),
    ]
    if not bin_dir:
        # Local superbuild: also include rocRoller build/stage/dist lib dirs
        ld_parts += [
            str(BUILD_DIR / "math-libs" / "BLAS" / "rocRoller" / "build"),
            str(ROCROLLER_STAGE_DIR / "lib"),
            str(BUILD_DIR / "math-libs" / "BLAS" / "rocRoller" / "dist" / "lib"),
        ]
    seen, ld_clean = set(), []
    for p in ld_parts:
        if p and p not in seen:
            seen.add(p)
            ld_clean.append(p)
    env["LD_LIBRARY_PATH"] = ":".join(ld_clean)
    env["ROCM_PATH"] = str(THEROCK_DIST_DIR)
    env["HIP_PATH"] = str(THEROCK_DIST_DIR)

# TEST_TYPE maps directly to a profile name defined in test-profiles.yaml
profile = os.getenv("TEST_TYPE", "full").lower()

# Get per-framework commands from rrtest
commands = get_test_commands(profile, config_file=profiles_yaml)


def run_cmd(cmd: list[str]) -> None:
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(THEROCK_DIR), check=True, env=env)


# Run gtest binaries
extra_gtest = os.getenv("EXTRA_GTEST_ARGS", "")
for args in commands.get("gtest", []):
    bin_name = Path(args[0]).name
    cmd = [str(find_bin(bin_name))] + args[1:]
    if extra_gtest:
        cmd += shlex.split(extra_gtest)
    run_cmd(cmd)

# Run catch2 binaries
shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
total_shards = int(os.getenv("TOTAL_SHARDS", "1"))
for args in commands.get("catch2", []):
    bin_name = Path(args[0]).name
    cmd = [str(find_bin(bin_name))] + args[1:]
    cmd += ["--shard-index", str(shard_index), "--shard-count", str(total_shards)]
    run_cmd(cmd)

# Run pytest suites
# Pytest runs from the artifact root (where pytest.ini, client/, scripts/ all live)
artifact_root = bin_dir.parent if bin_dir else ROCROLLER_STAGE_DIR
pytest_env = env.copy()
pytest_env["ROCROLLER_BUILD_DIR"] = str(artifact_root)
existing_pythonpath = pytest_env.get("PYTHONPATH", "")
scripts_lib = str(artifact_root / "scripts" / "lib")
pytest_env["PYTHONPATH"] = (
    f"{scripts_lib}:{existing_pythonpath}" if existing_pythonpath else scripts_lib
)
for args in commands.get("pytest", []):
    cmd = args[:]  # args[0] is "pytest", rest are paths/flags from rrtest
    logging.info(f"++ Exec [{artifact_root}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(artifact_root), check=True, env=pytest_env)
