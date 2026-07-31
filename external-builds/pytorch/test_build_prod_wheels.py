import os
from pathlib import Path

from build_prod_wheels import _setup_common_build_env


def test_common_build_env_includes_rocm_sysdeps_on_linux(tmp_path: Path):
    rocm_dir = tmp_path / "rocm"
    cmake_prefix = rocm_dir / "lib" / "cmake"

    env = _setup_common_build_env(
        cmake_prefix=cmake_prefix,
        rocm_dir=rocm_dir,
        pytorch_rocm_arch="gfx942",
        triton_dir=None,
        is_windows=False,
    )

    assert env["CMAKE_PREFIX_PATH"].split(os.pathsep) == [
        str(cmake_prefix),
        str(rocm_dir / "lib" / "rocm_sysdeps"),
    ]


def test_common_build_env_does_not_add_linux_sysdeps_on_windows(tmp_path: Path):
    rocm_dir = tmp_path / "rocm"
    cmake_prefix = rocm_dir / "lib" / "cmake"

    env = _setup_common_build_env(
        cmake_prefix=cmake_prefix,
        rocm_dir=rocm_dir,
        pytorch_rocm_arch="gfx942",
        triton_dir=None,
        is_windows=True,
    )

    assert env["CMAKE_PREFIX_PATH"] == str(cmake_prefix)
