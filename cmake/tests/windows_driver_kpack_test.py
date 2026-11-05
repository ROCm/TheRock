# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise driver feature selection and HIP's subproject configuration."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

THEROCK_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(THEROCK_ROOT / "build_tools/topology_to_cmake.py"),
            "--topology",
            str(THEROCK_ROOT / "BUILD_TOPOLOGY.toml"),
            "--output",
            str(tmp_path / "topology.cmake"),
            "--branch-config-output",
            str(tmp_path / "build/cmake/therock_branch_config.cmake"),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    # Evaluate the real feature graph and core rules without building subprojects.
    (tmp_path / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.25)
project(driver_kpack_test NONE)
set(THEROCK_SOURCE_DIR "@ROOT@")
set(THEROCK_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${THEROCK_SOURCE_DIR}/rocm-systems")
list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_features)
include("${THEROCK_SOURCE_DIR}/FLAGS.cmake")
set(THEROCK_ENABLE_ALL OFF)
set(THEROCK_ENABLE_CORE ON)
include("${CMAKE_CURRENT_SOURCE_DIR}/topology.cmake")
include(therock_windows_driver)
therock_finalize_features()

function(therock_check_subproject_targets out_var)
  set(${out_var} ON PARENT_SCOPE)
endfunction()
function(therock_cmake_subproject_declare name)
  set_property(GLOBAL APPEND PROPERTY declared_projects "${name}")
  if(name STREQUAL "hip-clr")
    cmake_parse_arguments(PARSE_ARGV 1 ARG
      "USE_DIST_AMDGPU_TARGETS;BACKGROUND_BUILD"
      "EXTERNAL_SOURCE_DIR;BINARY_DIR;COMPILER_TOOLCHAIN"
      "CMAKE_ARGS;BUILD_DEPS;RUNTIME_DEPS;INTERFACE_PROGRAM_DIRS;INTERFACE_LINK_DIRS;INTERFACE_INSTALL_RPATH_DIRS")
    file(WRITE "${CMAKE_BINARY_DIR}/hip_args.txt" "${ARG_CMAKE_ARGS}")
    file(WRITE "${CMAKE_BINARY_DIR}/hip_deps.txt" "${ARG_RUNTIME_DEPS}")
  endif()
endfunction()
function(therock_cmake_subproject_activate)
endfunction()
function(therock_cmake_subproject_provide_package)
endfunction()
function(therock_cmake_subproject_glob_c_sources)
endfunction()
function(therock_test_validate_shared_lib)
endfunction()
function(therock_provide_artifact name)
  set_property(GLOBAL APPEND PROPERTY provided_artifacts "${name}")
endfunction()
include("${THEROCK_SOURCE_DIR}/core/CMakeLists.txt")
get_property(projects GLOBAL PROPERTY declared_projects)
get_property(artifacts GLOBAL PROPERTY provided_artifacts)
file(WRITE "${CMAKE_BINARY_DIR}/projects.txt" "${projects}")
file(WRITE "${CMAKE_BINARY_DIR}/artifacts.txt" "${artifacts}")
file(WRITE "${CMAKE_BINARY_DIR}/kpack_enabled.txt" "${THEROCK_ENABLE_KPACK}")
""".replace(
            "@ROOT@", THEROCK_ROOT.as_posix()
        )
    )
    return tmp_path


def configure(
    source_dir: Path, system: str, *options: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(source_dir / "build"),
            "-GNinja",
            f"-DCMAKE_SYSTEM_NAME={system}",
            "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
            *options,
        ],
        capture_output=True,
        text=True,
    )


def assert_kpack(source_dir: Path, enabled: bool) -> None:
    build = source_dir / "build"
    value = "ON" if enabled else "OFF"
    assert (build / "kpack_enabled.txt").read_text() == value
    assert f"-DROCM_KPACK_ENABLED={value}" in (
        build / "hip_args.txt"
    ).read_text().split(";")
    for filename, item in (
        ("hip_deps.txt", "rocm-kpack"),
        ("projects.txt", "rocm-kpack"),
        ("artifacts.txt", "core-kpack"),
    ):
        assert (item in (build / filename).read_text().split(";")) == enabled


@pytest.mark.parametrize(
    "system,driver,split,enabled",
    [
        ("Windows", "ON", "OFF", False),
        ("Windows", "OFF", "ON", True),
        ("Linux", "OFF", "ON", True),
        ("Linux", "ON", "ON", True),
    ],
)
def test_kpack_build_variants(
    source_dir: Path, system: str, driver: str, split: str, enabled: bool
) -> None:
    result = configure(
        source_dir,
        system,
        f"-DTHEROCK_FLAG_WINDOWS_DRIVER_BUILD={driver}",
        f"-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS={split}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_kpack(source_dir, enabled)
    flags = json.loads((source_dir / "build/flag_settings.json").read_text())
    assert flags["KPACK_SPLIT_ARTIFACTS"] == (split == "ON")


def test_reconfigure_driver_variant(source_dir: Path) -> None:
    # Keep an explicit ON in the cache to catch accidental re-enablement and
    # check that returning to the SDK variant restores the original setting.
    for driver, split, enabled in (
        ("OFF", "ON", True),
        ("ON", "OFF", False),
        ("OFF", "ON", True),
    ):
        result = configure(
            source_dir,
            "Windows",
            "-DTHEROCK_ENABLE_KPACK=ON",
            f"-DTHEROCK_FLAG_WINDOWS_DRIVER_BUILD={driver}",
            f"-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS={split}",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert_kpack(source_dir, enabled)


def test_driver_rejects_split_artifacts(source_dir: Path) -> None:
    result = configure(source_dir, "Windows", "-DTHEROCK_FLAG_WINDOWS_DRIVER_BUILD=ON")
    assert result.returncode != 0
    assert "-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF" in result.stderr
