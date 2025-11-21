#!/usr/bin/env python
"""Test suite for the promote_from_rc_to_final package promotion script.

This test suite validates the functionality of promote_from_rc_to_final.py, which
promotes release candidate (RC) packages to final releases by removing RC suffixes
from version strings (e.g., 7.9.0rc1 → 7.9.0).

The test suite downloads RC packages from https://rocm.prereleases.amd.com/ and runs
comprehensive validation tests to ensure that:
  1. Complete promotion of all ROCm SDK and PyTorch packages works correctly
  2. Partial promotions (only ROCm or only PyTorch) fail as expected
  3. Promoted packages have correct filenames without RC suffixes
  4. All promoted wheels have consistent version strings
  5. Promoted packages can be successfully installed in a virtual environment

NOTES:
  - Tests run in isolated temporary directories (including venv) to avoid polluting the workspace
  - Original RC packages are downloaded fresh or loaded from cache for each test run
   (set with --cache-dir, directory is created if it doesn't exist)
  - Tests use real package files to ensure end-to-end validation
  - Platform is auto-detected based on the system but can be overridden with the --platform argument.

TEST SCENARIOS:
  - checkPromoteEverything: Tests promotion of all packages. This test should SUCCEED.

  - checkPromoteOnlyRocm: Tests promotion of only ROCm SDK packages while leaving
    PyTorch packages as RC. This test should FAIL.

  - checkPromoteOnlyTorch: Tests promotion of only PyTorch-related packages while
    leaving ROCm SDK packages as RC. This test should FAIL.

PACKAGE TYPES TESTED:
  - ROCm SDK packages: rocm, rocm_sdk_core, rocm_sdk_devel, rocm_sdk_libraries_*
  - PyTorch packages: torch, torchaudio, torchvision, pytorch_triton_rocm
  - Distribution tarballs: therock-dist-{platform}-gfx{arch}-{version}.tar.gz

PREREQUISITES:
  - pip install -r ./build_tools/packaging/requirements.txt

USAGE:
  # Test on current platform (auto-detected):
  python ./build_tools/packaging/tests/promote_from_rc_to_final_test.py

  # Use cached packages to speed up repeated test runs:
  python ./build_tools/packaging/tests/promote_from_rc_to_final_test.py --cache-dir=/tmp/package_cache
"""

import argparse
import shutil
import sys
import os
from pathlib import Path
import tempfile
from packaging.version import Version
from pkginfo import Wheel
import subprocess
import urllib
import platform as platform_module

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import promote_from_rc_to_final

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import setup_venv


def checkPromotedFileNames(dir_path: Path, platform: str) -> tuple[bool, str]:
    if platform == "linux":
        expected_promoted_pkgs = [
            "rocm-7.9.0.tar.gz",
            "rocm_sdk_core-7.9.0-py3-none-linux_x86_64.whl",
            "rocm_sdk_devel-7.9.0-py3-none-linux_x86_64.whl",
            "rocm_sdk_libraries_gfx94x_dcgpu-7.9.0-py3-none-linux_x86_64.whl",
            "pytorch_triton_rocm-3.3.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
            "torch-2.7.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
            "torchaudio-2.7.1a0+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
            "torchvision-0.22.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
            "therock-dist-linux-gfx1151-7.9.0.tar.gz",
        ]
    else:
        expected_promoted_pkgs = [
            "rocm-7.9.0.tar.gz",
            "rocm_sdk_core-7.9.0-py3-none-win_amd64.whl",
            "rocm_sdk_devel-7.9.0-py3-none-win_amd64.whl",
            "rocm_sdk_libraries_gfx1151-7.9.0-py3-none-win_amd64.whl",
            "torch-2.9.0+rocm7.9.0-cp312-cp312-win_amd64.whl",
            "torchaudio-2.9.0+rocm7.9.0-cp312-cp312-win_amd64.whl",
            "torchvision-0.24.0+rocm7.9.0-cp312-cp312-win_amd64.whl",
            "therock-dist-windows-gfx1151-7.9.0.tar.gz",
        ]

    # get files and strip path from them
    files = dir_path.glob("*")
    files = [file.name for file in files]

    if len(files) != len(expected_promoted_pkgs):
        return (
            False,
            f"Files found and expected promoted packages are not the same amount ({len(files)} vs {len(expected_promoted_pkgs)})",
        )

    for file in files:
        if not file in expected_promoted_pkgs:
            return False, f"{file} not matching any of the expected package names"

    return True, ""


def checkAllWheelsSameVersion(
    dir_path: Path, expected_version: Version
) -> tuple[bool, str]:
    for file in dir_path.glob("*.whl"):
        wheel = Wheel(file)
        version = Version(wheel.version)

        if (
            str(version) == str(expected_version) and version.local == None
        ):  # rocm packages
            continue
        elif str(version.local) == "rocm" + str(expected_version):  # torch packages
            continue
        else:
            return (
                False,
                f"{file} has version {version}, but expected version is {expected_version}",
            )

    return True, ""


def checkInstallation(dir_path: Path) -> tuple[bool, str]:
    """
    Note: dir_path must be a TemporaryDirectory, otherwise you must clean up the .venv created here yourself.
    """
    try:
        setup_venv.create_venv(dir_path / ".venv")
        python_exe = setup_venv.find_venv_python(dir_path / ".venv")
        if python_exe is None:
            return (
                False,
                "Problem when installing temporary venv: Python executable not found",
            )

        # only install rocm wheels/sdist and pytorch wheels, not therock-dist tarball or .venv
        packages = [
            p
            for p in dir_path.glob("*")
            if p.name != ".venv" and "therock-dist" not in p.name
        ]

        proc = subprocess.run(
            [python_exe, "-m", "pip", "install"] + packages,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    return True, ""


def checkPromoteEverything(
    dir_path: Path, expected_version: Version, platform: str
) -> tuple[bool, str]:
    print("")
    print(
        "================================================================================="
    )
    print("TEST: Testing promotion of all packages")
    print(
        "================================================================================="
    )
    success = False
    with tempfile.TemporaryDirectory(
        prefix="PromoteRcToFinalTest-PromoteEverything-"
    ) as tmp:
        tmp_dir = Path(tmp)
        # make a copy in a separate dir to not pollute it
        for file in dir_path.glob("*"):
            shutil.copy2(file, tmp_dir)

        promote_from_rc_to_final.main(tmp_dir, delete=True)
        success = True

        for func_name, res in [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, platform)),
            (
                "checkAllWheelsSameVersion",
                checkAllWheelsSameVersion(tmp_dir, expected_version),
            ),
            ("checkInstallation", checkInstallation(tmp_dir)),
        ]:
            if not res[0]:
                print("")
                print(
                    f"[ERROR] Failure to promote the packages (failure captured by {func_name}):"
                )
                print(res[1])
                success = False
                break
    print("")
    print(
        "================================================================================="
    )
    print(
        "TEST DONE: Testing promotion of all packages. Result:"
        + (" SUCCESS" if success else " FAILURE")
    )
    print(
        "================================================================================="
    )
    return success


def checkPromoteOnlyRocm(
    dir_path: Path, expected_version: Version, platform: str
) -> bool:  # should fail
    print("")
    print(
        "================================================================================="
    )
    print("TEST: Testing promotion of only rocm packages")
    print(
        "================================================================================="
    )
    success = False
    with tempfile.TemporaryDirectory(
        prefix="PromoteRcToFinalTest-PromoteOnlyRocm-"
    ) as tmp:
        tmp_dir = Path(tmp)
        # make a copy in a separate dir to not pollute it
        for file in dir_path.glob("*"):
            shutil.copy2(file, tmp_dir)

        promote_from_rc_to_final.main(tmp_dir, match_files="rocm*", delete=True)

        success = True

        for func_name, res in [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, platform)),
            (
                "checkAllWheelsSameVersion",
                checkAllWheelsSameVersion(tmp_dir, expected_version),
            ),
            ("checkInstallation", checkInstallation(tmp_dir)),
        ]:
            if res[0]:
                success = False
                print("")
                print(
                    f"[ERROR] checkPromoteOnlyRocm: Promotion of packages successful, eventhough it shouldnt be"
                )
                print("Function that succeeded (and should NOT have): " + func_name)
                proc = subprocess.run(
                    ["ls", tmp_dir], capture_output=True, encoding="utf-8"
                )
                print(proc.stdout)
                break
    print("")
    print(
        "================================================================================="
    )
    print(
        "TEST DONE: Testing promotion of only rocm packages. Result:"
        + (" SUCCESS" if success else " FAILURE")
    )
    print(
        "================================================================================="
    )
    return success


def checkPromoteOnlyTorch(
    dir_path: Path, expected_version: Version, platform: str
) -> bool:  # should fail
    print("")
    print(
        "================================================================================="
    )
    print("TEST: Testing promotion of only PyTorch packages")
    print(
        "================================================================================="
    )
    success = False
    with tempfile.TemporaryDirectory(
        prefix="PromoteRcToFinalTest-PromoteOnlyTorch-"
    ) as tmp:
        tmp_dir = Path(tmp)
        # make a copy in a separate dir to not pollute it
        for file in dir_path.glob("*"):
            shutil.copy2(file, tmp_dir)

        promote_from_rc_to_final.main(tmp_dir, match_files="*torch*", delete=True)

        success = True

        for func_name, res in [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, platform)),
            (
                "checkAllWheelsSameVersion",
                checkAllWheelsSameVersion(tmp_dir, expected_version),
            ),
            ("checkInstallation", checkInstallation(tmp_dir)),
        ]:
            if res[0]:
                success = False
                print("")
                print(
                    f"[ERROR] checkPromoteOnlyTorch: Promotion of packages successful, eventhough it shouldnt be"
                )
                print("Function that succeeded (and should NOT have): " + func_name)
                proc = subprocess.run(
                    ["ls", tmp_dir], capture_output=True, encoding="utf-8"
                )
                print(proc.stdout)
                break
    print("")
    print(
        "================================================================================="
    )
    print(
        "TEST DONE: Testing promotion of only PyTorch packages. Result:"
        + (" SUCCESS" if success else " FAILURE")
    )
    print(
        "================================================================================="
    )
    return success


def fetchPackage(URL: str, package_name: str, tmp_dir: Path, cache_dir: Path) -> None:
    # Check first if the package is cached
    if cache_dir is not None:
        if (cache_dir / package_name).exists():
            print(f"  Found in cache: {package_name}")
            shutil.copy2(cache_dir / package_name, tmp_dir / package_name)
            return
    # Otherwise download the package
    print(f"  Downloading {package_name}")
    # Use safe encoding, otherwise CURL gets unhappy with the "+" in the URL
    url_safe_encoding = URL + urllib.parse.quote(package_name)
    print(url_safe_encoding)
    subprocess.run(
        ["curl", "--output", tmp_dir / package_name, url_safe_encoding],
        check=True,
    )
    # Let's cache the package
    if cache_dir is not None:
        print(f"  Caching {package_name}")
        shutil.copy2(tmp_dir / package_name, cache_dir / package_name)


def getLinuxPackagesLinks() -> tuple[list[tuple[str, str]], Version, Version]:
    # download some version
    URL = "https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu/"
    version = Version("7.9.0rc1")
    expected_version = Version("7.9.0")
    packages = [
        "rocm-7.9.0rc1.tar.gz",
        "rocm_sdk_core-7.9.0rc1-py3-none-linux_x86_64.whl",
        "rocm_sdk_devel-7.9.0rc1-py3-none-linux_x86_64.whl",
        "rocm_sdk_libraries_gfx94x_dcgpu-7.9.0rc1-py3-none-linux_x86_64.whl",
        "pytorch_triton_rocm-3.3.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
        "torch-2.7.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
        "torchaudio-2.7.1a0+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
        "torchvision-0.22.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
    ]

    url_and_packages = [(URL, package) for package in packages]
    url_and_packages.append(
        (
            "https://rocm.prereleases.amd.com/tarball/",
            "therock-dist-linux-gfx1151-7.9.0rc1.tar.gz",
        )
    )

    return url_and_packages, version, expected_version


def getWindowsPackagesLinks() -> tuple[list[tuple[str, str]], Version, Version]:
    # download some version
    URL = "https://rocm.prereleases.amd.com/whl/gfx1151/"
    version = Version("7.9.0rc1")
    expected_version = Version("7.9.0")
    packages = [
        "rocm-7.9.0rc1.tar.gz",
        "rocm_sdk_core-7.9.0rc1-py3-none-win_amd64.whl",
        "rocm_sdk_devel-7.9.0rc1-py3-none-win_amd64.whl",
        "rocm_sdk_libraries_gfx1151-7.9.0rc1-py3-none-win_amd64.whl",
        "torch-2.9.0+rocm7.9.0rc1-cp312-cp312-win_amd64.whl",
        "torchaudio-2.9.0+rocm7.9.0rc1-cp312-cp312-win_amd64.whl",
        "torchvision-0.24.0+rocm7.9.0rc1-cp312-cp312-win_amd64.whl",
    ]

    url_and_packages = [(URL, package) for package in packages]
    url_and_packages.append(
        (
            "https://rocm.prereleases.amd.com/tarball/",
            "therock-dist-windows-gfx1151-7.9.0rc1.tar.gz",
        )
    )

    return url_and_packages, version, expected_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Tests promotion of packages from release candidate to final release (e.g. 7.10.0rc1 --> 7.10.0).
"""
    )
    parser.add_argument(
        "--platform",
        help="OS platform: either 'linux' (default) or 'win'",
        default="linux" if platform_module.system() != "Windows" else "windows",
    )
    parser.add_argument(
        "--cache-dir",
        help="Path to the directory that contains the cache of the packages",
        type=Path,
        default=None,
    )
    p = parser.parse_args(sys.argv[1:])
    platform = p.platform
    cache_dir = p.cache_dir

    if cache_dir is not None:
        if not cache_dir.exists():
            print(f"  Creating cache directory: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)

    # make tmpdir
    with tempfile.TemporaryDirectory(prefix=f"PromoteRcToFinalTest-{platform}-") as tmp:
        tmp_dir = Path(tmp)
        if platform == "linux":
            url_and_packages, version, expected_version = getLinuxPackagesLinks()
        elif platform == "windows":  # win
            url_and_packages, version, expected_version = getWindowsPackagesLinks()
        else:
            raise ValueError(f"Unknown platform: {platform}")

        print(
            f"Testing promotion of {version} to {expected_version} on platform {platform}"
        )
        print(f"Fetching packages", end="")
        print(f"  Using cache directory: {cache_dir}")
        for URL, package in url_and_packages:
            fetchPackage(URL, package, tmp_dir, cache_dir)
        print(" ...done")

        res_everything = checkPromoteEverything(tmp_dir, expected_version, platform)
        res_rocm = checkPromoteOnlyRocm(tmp_dir, expected_version, platform)
        res_torch = checkPromoteOnlyTorch(tmp_dir, expected_version, platform)

        print("")
        print("")
        print(
            "================================================================================="
        )
        print("SUMMARY")
        print(
            "================================================================================="
        )
        print("checkPromoteEverything: " + ("SUCCESS" if res_everything else "FAILURE"))
        print("checkPromoteOnlyRocm: " + ("SUCCESS" if res_rocm else "FAILURE"))
        print("checkPromoteOnlyTorch: " + ("SUCCESS" if res_torch else "FAILURE"))
        print(
            "================================================================================="
        )
