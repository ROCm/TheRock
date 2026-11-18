# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for torch_wheel_runpath.py."""

import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import torch_wheel_runpath

DEVEL_LIB = "/opt/python/cp312-cp312/lib/python3.12/site-packages/_rocm_sdk_devel/lib"


class IndexRuntimeLibrariesTest(unittest.TestCase):
    def test_paths_are_relative_to_site_packages(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td)
            core = site / "_rocm_sdk_core"
            libraries = site / "_rocm_sdk_libraries"
            for d, name in [
                (core / "lib", "libamdhip64.so.7"),
                (core / "lib" / "host-math" / "lib", "librocm-openblas.so.0"),
                (libraries / "lib", "libMIOpen.so.1"),
                (libraries / "lib", "libamdhip64.so.7"),
            ]:
                d.mkdir(parents=True, exist_ok=True)
                (d / name).write_bytes(b"")
            (core / "lib" / "README.txt").write_text("not a library")

            index = torch_wheel_runpath.index_runtime_libraries([core, libraries])

        self.assertEqual(
            index,
            {
                # The first package listed wins for a name present in both.
                "libamdhip64.so.7": "_rocm_sdk_core/lib",
                "librocm-openblas.so.0": "_rocm_sdk_core/lib/host-math/lib",
                "libMIOpen.so.1": "_rocm_sdk_libraries/lib",
            },
        )


class ComputeRunpathTest(unittest.TestCase):
    INDEX = {
        "libamdhip64.so.7": "_rocm_sdk_core/lib",
        "librocprofiler-sdk.so.1": "_rocm_sdk_core/lib",
        "librocm-openblas.so.0": "_rocm_sdk_core/lib/host-math/lib",
        "libMIOpen.so.1": "_rocm_sdk_libraries/lib",
    }

    def test_libtorch_cpu_points_at_installed_core(self):
        runpath = torch_wheel_runpath.compute_runpath(
            "torch/lib",
            f"$ORIGIN:{DEVEL_LIB}/host-math/lib:{DEVEL_LIB}",
            [
                "libc10.so",
                "librocm-openblas.so.0",
                "librocprofiler-sdk.so.1",
                "libamdhip64.so.7",
                "libc.so.6",
            ],
            {"libc10.so", "libtorch_cpu.so"},
            self.INDEX,
        )
        self.assertEqual(
            runpath,
            "$ORIGIN"
            ":$ORIGIN/../../_rocm_sdk_core/lib/host-math/lib"
            ":$ORIGIN/../../_rocm_sdk_core/lib",
        )

    def test_keeps_existing_relative_entries_and_dedupes(self):
        runpath = torch_wheel_runpath.compute_runpath(
            "torch/lib",
            f"$ORIGIN:$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib:{DEVEL_LIB}:$ORIGIN",
            ["libMIOpen.so.1", "libamdhip64.so.7"],
            set(),
            self.INDEX,
        )
        self.assertEqual(
            runpath,
            "$ORIGIN"
            ":$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib"
            ":$ORIGIN/../../_rocm_sdk_libraries/lib"
            ":$ORIGIN/../../_rocm_sdk_core/lib",
        )

    def test_relative_to_each_file_directory(self):
        runpath = torch_wheel_runpath.compute_runpath(
            "torch", f"$ORIGIN/lib:{DEVEL_LIB}", ["libamdhip64.so.7"], set(), self.INDEX
        )
        self.assertEqual(runpath, "$ORIGIN/lib:$ORIGIN/../_rocm_sdk_core/lib")

    def test_no_rocm_dependency_only_drops_absolute_entries(self):
        runpath = torch_wheel_runpath.compute_runpath(
            "torch/bin", f"$ORIGIN/../lib:{DEVEL_LIB}", ["libc10.so"], set(), {}
        )
        self.assertEqual(runpath, "$ORIGIN/../lib")


class ExtractWheelTest(unittest.TestCase):
    def test_keeps_executable_bit(self):
        with tempfile.TemporaryDirectory() as td:
            wheel = Path(td) / "torch-2.15.0+rocm10.3.0-cp312-cp312-linux_x86_64.whl"
            with zipfile.ZipFile(wheel, "w") as zf:
                info = zipfile.ZipInfo("torch/bin/torch_shm_manager")
                info.external_attr = (stat.S_IFREG | 0o755) << 16
                zf.writestr(info, b"\x7fELF")
                zf.writestr("torch/__init__.py", "")

            root = torch_wheel_runpath.extract_wheel(wheel, Path(td) / "out")

            self.assertEqual(root.name, "torch-2.15.0+rocm10.3.0")
            manager = root / "torch" / "bin" / "torch_shm_manager"
            self.assertTrue(os.access(manager, os.X_OK))
            self.assertTrue(torch_wheel_runpath.is_elf(manager))
            self.assertFalse(torch_wheel_runpath.is_elf(root / "torch" / "__init__.py"))


if __name__ == "__main__":
    unittest.main()
