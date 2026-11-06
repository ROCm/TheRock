#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installs the extra packages that PyTorch's ROCm CI image adds to its environment.

PyTorch's ROCm CI Docker image installs packages beyond the torch wheel's own
dependencies, at versions pinned in the PyTorch source tree under
`.ci/docker/ci_commit_pins/`. Installing the same packages makes TheRock's
PyTorch test environment match PyTorch CI, so that defects which only appear
with those packages present (for example ROCM-29472) are caught here too.

Today the only such package is rocm-origami. It is a source package that builds
against the ROCm SDK (its CMake project calls `find_package(hip)`), so this
script installs `rocm[devel]` first when it is missing.

When PyTorch adds a package to its CI image, add it to `PINNED_PACKAGES`. Never
copy a version into TheRock: versions are always read from the PyTorch checkout,
so a pin bump in PyTorch takes effect on the next run.

Usage:
    python external-builds/pytorch/install_pytorch_ci_packages.py \\
        --pytorch-dir external-builds/pytorch/pytorch \\
        --index-url https://nightly.repo.amd.com/rocm/whl-next/
"""

import argparse
import importlib.util
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

THIS_DIR = Path(__file__).resolve().parent
PIN_DIR = Path(".ci") / "docker" / "ci_commit_pins"

PIP_INSTALL_ATTEMPTS = 2
PIP_RETRY_WAIT_SECONDS = 30


@dataclass(frozen=True)
class PinnedPackage:
    # Distribution name on PyPI.
    name: str
    # File under .ci/docker/ci_commit_pins/ holding the version PyTorch installs.
    pin_file: str
    # Whether the package builds from source against the ROCm SDK.
    needs_rocm_devel: bool


PINNED_PACKAGES = [
    PinnedPackage("rocm-origami", "rocm-origami.txt", needs_rocm_devel=True),
]


def run(
    cmd: list[str], *, env: dict[str, str] | None = None, capture: bool = False
) -> str:
    print(f"++ Exec: {shlex.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd, env=env, check=True, text=True, stdout=subprocess.PIPE if capture else None
    )
    return result.stdout.strip() if capture else ""


def read_pin(pin_dir: Path, package: PinnedPackage) -> Version | None:
    pin_path = pin_dir / package.pin_file
    if not pin_path.is_file():
        return None
    text = pin_path.read_text().strip()
    try:
        return Version(text)
    except InvalidVersion as e:
        raise ValueError(
            f"Invalid {package.name} version '{text}' in {pin_path}"
        ) from e


def pip_install(
    specs: list[str],
    *,
    index_url: str = "",
    find_links: str = "",
    env: dict[str, str] | None = None,
) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if index_url:
        cmd.append(f"--index-url={index_url}")
    if find_links:
        cmd.append(f"--find-links={find_links}")
    cmd.extend(specs)
    for attempt in range(1, PIP_INSTALL_ATTEMPTS + 1):
        try:
            run(cmd, env=env)
            return
        except subprocess.CalledProcessError:
            if attempt == PIP_INSTALL_ATTEMPTS:
                raise
            print(
                f"pip install failed (attempt {attempt} of {PIP_INSTALL_ATTEMPTS}), "
                f"retrying in {PIP_RETRY_WAIT_SECONDS}s",
                flush=True,
            )
            time.sleep(PIP_RETRY_WAIT_SECONDS)


def rocm_sdk(*args: str, capture: bool = True) -> str:
    return run([sys.executable, "-m", "rocm_sdk", *args], capture=capture)


def rocm_devel_installed() -> bool:
    return importlib.util.find_spec("rocm_sdk_devel") is not None


def ensure_rocm_devel(*, index_url: str, find_links: str) -> None:
    if rocm_devel_installed():
        print("rocm[devel] is already installed", flush=True)
        return
    rocm_version = rocm_sdk("version")
    pip_install(
        [f"rocm[devel]=={rocm_version}"], index_url=index_url, find_links=find_links
    )


def rocm_build_env() -> dict[str, str]:
    """Returns an environment for building source packages against the ROCm SDK."""
    rocm_sdk("init", capture=False)
    cmake_prefix = rocm_sdk("path", "--cmake")
    rocm_root = Path(rocm_sdk("path", "--root"))
    env = dict(os.environ)
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        p for p in (cmake_prefix, env.get("CMAKE_PREFIX_PATH", "")) if p
    )
    # The test containers ship clang but not gcc.
    llvm_bin = rocm_root / "lib" / "llvm" / "bin"
    env.setdefault("CC", str(llvm_bin / "clang"))
    env.setdefault("CXX", str(llvm_bin / "clang++"))
    return env


def installed_version(name: str) -> Version:
    # Reads package metadata only: importing a package such as origami next to
    # an incompatible ROCm SDK library can crash the process, and that failure
    # belongs in the tests, not in this install step.
    return Version(
        run(
            [
                sys.executable,
                "-c",
                f"import importlib.metadata as m; print(m.version({name!r}))",
            ],
            capture=True,
        )
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Install the extra packages that PyTorch's ROCm CI image installs."
    )
    p.add_argument(
        "--pytorch-dir",
        type=Path,
        default=THIS_DIR / "pytorch",
        help="PyTorch source checkout to read the pinned versions from",
    )
    p.add_argument(
        "--index-url",
        default="",
        help="Package index for rocm[devel] (the index torch was installed from)",
    )
    p.add_argument(
        "--find-links",
        default="",
        help="Flat package page for rocm[devel], used together with --index-url",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the packages that would be installed without installing them",
    )
    args = p.parse_args(argv)

    if platform.system() != "Linux":
        print(
            "PyTorch installs these packages only in its Linux ROCm CI image; "
            f"nothing to do on {platform.system()}"
        )
        return 0

    pin_dir = args.pytorch_dir / PIN_DIR
    if not pin_dir.is_dir():
        raise FileNotFoundError(
            f"{pin_dir} does not exist; --pytorch-dir must be a PyTorch checkout"
        )

    pins: list[tuple[PinnedPackage, Version]] = []
    for package in PINNED_PACKAGES:
        version = read_pin(pin_dir, package)
        if version is None:
            print(f"{package.name}: not pinned in {pin_dir}, skipping")
        else:
            pins.append((package, version))
    if not pins:
        return 0

    specs = [f"{package.name}=={version}" for package, version in pins]
    if args.dry_run:
        print(f"Would install: {' '.join(specs)}")
        return 0

    env = None
    if any(package.needs_rocm_devel for package, _ in pins):
        ensure_rocm_devel(index_url=args.index_url, find_links=args.find_links)
        env = rocm_build_env()

    # The pinned packages come from PyPI, like in PyTorch's CI image.
    pip_install(specs, env=env)

    for package, version in pins:
        found = installed_version(package.name)
        if found != version:
            raise RuntimeError(
                f"Expected {package.name}=={version} after install, found {found}"
            )
        print(f"Installed {package.name}=={found}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
