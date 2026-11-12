#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Prepare the pinned host emulator for a registered MIOpen target.

Only rocjitsu is built here. MIOpen's runner and CPU-allocation utility must
already have been fetched from the artifacts under test.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miopen_rocjitsu_targets import load_targets

ROCJITSU_REPO = "https://github.com/ROCm/rocm-systems.git"
ROCJITSU_REF = "6f5d9658a4b13656e0c1a3719d86a320c9206dbe"


def run(cmd: list[str], **kwargs) -> None:
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_build_tools() -> None:
    # The no_rocm CI container runs as tester with passwordless sudo.
    if (
        all(shutil.which(tool) for tool in ("cmake", "ninja", "c++", "git"))
        and Path("/usr/include/libdrm").exists()
    ):
        return
    run(["sudo", "apt-get", "update"])
    run(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake",
            "ninja-build",
            "build-essential",
            "libdrm-dev",
            "git",
        ]
    )


def build_rocjitsu(workdir: Path, jobs: int) -> None:
    src = workdir / "rocm-systems"
    if not src.exists():
        run(["git", "init", str(src)])
        run(["git", "-C", str(src), "remote", "add", "origin", ROCJITSU_REPO])
    run(["git", "-C", str(src), "config", "core.sparseCheckout", "true"])
    (src / ".git" / "info" / "sparse-checkout").write_text(
        "emulation/rocjitsu\nruntimes/api-headers\nshared/machine-readable-isa/isa\n",
        encoding="utf-8",
    )
    # Re-establish the pin even when a previous preparation left a source tree.
    run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", ROCJITSU_REF])
    run(["git", "-C", str(src), "checkout", "--detach", ROCJITSU_REF])
    build = workdir / "build"
    run(
        [
            "cmake",
            "-S",
            str(src / "emulation" / "rocjitsu"),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            f"-DFETCHCONTENT_BASE_DIR={build / 'third_party'}",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "rocjitsu_bin",
            "rocjitsu_shared",
            "-j",
            str(max(1, min(jobs, 8))),
        ]
    )


def normalize_profile(profile: dict, arch: str) -> dict:
    """Bound emulator host work without changing the SKU's CU topology."""
    config = copy.deepcopy(profile)
    vm = config["vm"]
    gpu = vm["gpu"]
    digits = arch.removeprefix("gfx")
    expected_version = (
        int(digits[:-2]) * 10000 + int(digits[-2], 16) * 100 + int(digits[-1], 16)
    )
    if gpu["device"]["gfx_target_version"] != expected_version:
        raise ValueError(f"Pinned profile does not describe {arch}")
    if vm.get("target", arch) != arch:
        raise ValueError(f"Pinned profile target does not match {arch}")
    vm["target"] = arch
    gpu["num_gpus"] = 1
    allocation = {
        "num_threads": 1,
        "cpu_dispatch_threads": 1,
        "async_helper_threads": 0,
    }
    config.update(allocation)
    config["thread_allocations"] = [allocation.copy()]
    return config


def prepared_paths(arch: str) -> tuple[Path, Path, Path]:
    work = (Path("rocjitsu-work") / arch).resolve()
    return work, work / "build/tools/rocjitsu/rocjitsu", work / "prepared-config.json"


def main(argv: list[str] | None = None) -> int:
    targets = load_targets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=targets)
    args = parser.parse_args(argv)
    assets = (
        Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "build")).resolve()
        / "share/miopen/rocjitsu"
    )
    if not (assets / "rocjitsu_cpu.py").is_file():
        raise RuntimeError(f"PR-installed MIOpen rocjitsu assets missing: {assets}")
    sys.path.insert(0, str(assets))
    from rocjitsu_cpu import available_cpu_count

    work, launcher, prepared_config = prepared_paths(args.arch)
    ensure_build_tools()
    work.mkdir(parents=True, exist_ok=True)
    build_rocjitsu(work, jobs=available_cpu_count())
    profile = (
        work / "rocm-systems/emulation/rocjitsu/configs" / targets[args.arch]["config"]
    )
    config = normalize_profile(
        json.loads(profile.read_text(encoding="utf-8")), args.arch
    )
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RuntimeError(
            f"Prepared rocjitsu launcher missing or not executable: {launcher}"
        )
    prepared_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared rocjitsu launcher: {launcher}", flush=True)
    print(f"Prepared rocjitsu config: {prepared_config}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        sys.exit(f"::error::{error}")
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 1)
