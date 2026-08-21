# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(
    0,
    os.fspath(Path(__file__).resolve().parents[2] / "external-builds" / "pytorch"),
)
import rewrite_torch_runpath


class DiscoverRuntimeLibDirsTest(unittest.TestCase):
    def test_skips_devel_and_collects_family_libraries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_rocm_sdk_core" / "lib" / "rocm_sysdeps" / "lib").mkdir(
                parents=True
            )
            (root / "_rocm_sdk_core" / "lib" / "host-math" / "lib").mkdir(parents=True)
            (root / "_rocm_sdk_libraries_gfx94x_dcgpu" / "lib").mkdir(parents=True)
            (root / "_rocm_sdk_devel" / "lib").mkdir(parents=True)

            dirs = rewrite_torch_runpath.discover_rocm_runtime_lib_dirs([root])

        self.assertIn("_rocm_sdk_core/lib", dirs)
        self.assertIn("_rocm_sdk_core/lib/rocm_sysdeps/lib", dirs)
        self.assertIn("_rocm_sdk_core/lib/host-math/lib", dirs)
        self.assertIn("_rocm_sdk_libraries_gfx94x_dcgpu/lib", dirs)
        self.assertNotIn("_rocm_sdk_devel/lib", dirs)

    def test_fallback_when_sdk_missing(self):
        with tempfile.TemporaryDirectory() as td:
            dirs = rewrite_torch_runpath.discover_rocm_runtime_lib_dirs([Path(td)])
        self.assertEqual(dirs, list(rewrite_torch_runpath.FALLBACK_RUNTIME_LIB_DIRS))


class RpathForSharedObjectTest(unittest.TestCase):
    def test_lib_so_uses_two_dotdots(self):
        rpath = rewrite_torch_runpath.rpath_for_shared_object(
            Path("torch/lib/libtorch_hip.so"),
            ["_rocm_sdk_core/lib", "_rocm_sdk_libraries_gfx94x_dcgpu/lib"],
        )
        self.assertTrue(rpath.startswith("$ORIGIN/../../_rocm_sdk_core/lib:"), rpath)
        self.assertIn("$ORIGIN/../../_rocm_sdk_libraries_gfx94x_dcgpu/lib", rpath)
        self.assertTrue(rpath.endswith(":$ORIGIN"), rpath)
        self.assertNotIn("$ORIGIN/lib", rpath)

    def test_top_level_extension_uses_one_dotdot(self):
        rpath = rewrite_torch_runpath.rpath_for_shared_object(
            Path("torch/_C.cpython-312-x86_64-linux-gnu.so"),
            ["_rocm_sdk_core/lib"],
        )
        self.assertEqual(rpath, "$ORIGIN/../_rocm_sdk_core/lib:$ORIGIN:$ORIGIN/lib")

    def test_builder_paths_are_detected(self):
        self.assertTrue(
            rewrite_torch_runpath.rpath_contains_builder_path(
                "$ORIGIN:/opt/_internal/cpython-3.13.3/lib/python3.13/"
                "site-packages/_rocm_sdk_devel/lib"
            )
        )
        self.assertTrue(
            rewrite_torch_runpath.rpath_contains_builder_path(
                "$ORIGIN:/opt/python/cp313-cp313/lib/python3.13/"
                "site-packages/_rocm_sdk_devel/lib"
            )
        )
        self.assertFalse(
            rewrite_torch_runpath.rpath_contains_builder_path(
                "$ORIGIN/../../_rocm_sdk_core/lib:$ORIGIN"
            )
        )


class RecordAndIterTest(unittest.TestCase):
    def test_iter_shared_objects_skips_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real = root / "torch" / "lib" / "libfoo.so.1"
            real.parent.mkdir(parents=True)
            real.write_bytes(b"not-an-elf")
            link = root / "torch" / "lib" / "libfoo.so"
            link.symlink_to(real.name)
            (root / "torch" / "lib" / "notes.txt").write_text("skip me")

            found = [
                p.relative_to(root).as_posix()
                for p in rewrite_torch_runpath.iter_shared_objects(root)
            ]
        self.assertEqual(found, ["torch/lib/libfoo.so.1"])

    def test_write_record_hashes_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist_info = root / "torch-1.0.dist-info"
            dist_info.mkdir()
            payload = root / "torch" / "lib" / "x.so"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"abc")
            (dist_info / "METADATA").write_text("name: torch\n")
            (dist_info / "RECORD").write_text("placeholder\n")

            rewrite_torch_runpath.write_record(root)
            record = (dist_info / "RECORD").read_text()

        self.assertIn("torch/lib/x.so,", record)
        self.assertIn(rewrite_torch_runpath.record_digest(b"abc"), record)
        self.assertIn("torch-1.0.dist-info/RECORD,,", record)


class RepackWheelTest(unittest.TestCase):
    def test_repack_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "unpacked"
            (root / "torch").mkdir(parents=True)
            (root / "torch" / "hi.txt").write_text("hello")
            dest = Path(td) / "out.whl"
            rewrite_torch_runpath._repack_wheel(root, dest)
            self.assertTrue(dest.is_file())
            with zipfile.ZipFile(dest) as zf:
                self.assertEqual(zf.read("torch/hi.txt"), b"hello")


if __name__ == "__main__":
    unittest.main()
